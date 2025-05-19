import logging
import os
import dotenv
import torch
import gc
from typing import List, Optional, Union, Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoConfig
)
from peft import PeftConfig
from .base_language_model import BaseLanguageModel

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

logger = logging.getLogger(__name__)

class HfCausalModel(BaseLanguageModel):
    DTYPE_MAPPING = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("HfCausalModel Specific Arguments")
        group.add_argument("--maximum_token", type=int, default=8000, help="Maximum token length allowed by the tokenizer/model context window.")
        group.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum number of new tokens to generate.")
        group.add_argument("--dtype", choices=list(HfCausalModel.DTYPE_MAPPING.keys()), default="bf16", help="Data type for model loading (fp32, fp16, bf16).")
        group.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none", help="Quantization type (none, 4bit, 8bit). Requires bitsandbytes.")
        group.add_argument("--attn_implementation", default="flash_attention_2",
                           choices=["eager", "sdpa", "flash_attention_2", None],
                           help="Attention implementation (requires compatible transformers/hardware). Set to None to let HF decide.")
        group.add_argument("--generation_mode", type=str, default="greedy",
                           choices=["greedy", "beam", "sampling", "group-beam"], # Removed early stopping variants for simplicity, can be added back via kwargs
                           help="Default generation strategy.")
        group.add_argument("--generation_k", type=int, default=1, help="Parameter 'k' for generation (e.g., num_beams, num_return_sequences for sampling, num_groups for group-beam). Must be >= 1.")
        group.add_argument("--chat_model", default='true', type=lambda x: (str(x).lower() == 'true'), help="Apply chat template if true.")
        group.add_argument("--use_assistant_model", default='false', type=lambda x: (str(x).lower() == 'true'), help="Load an assistant model (if specified). Usage depends on external logic.")
        group.add_argument("--assistant_model_path", type=str, default=None, help="Path to the assistant model.")
        group.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face API token (optional, reads from env var HF_TOKEN by default).")
        group.add_argument("--low_cpu_mem_usage", action='store_true', help="Use low_cpu_mem_usage during model loading (useful for large models).")

    def __init__(self, args, **kwargs):
        self.args = args
        self._loaded_model_path: Optional[str] = None
        self._loaded_assistant_path: Optional[str] = None

        self.maximum_token: int = args.maximum_token
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[PreTrainedModel] = None
        self.assistant_model: Optional[PreTrainedModel] = None
        self.generation_cfg: Optional[GenerationConfig] = None

        self.device_map: Union[str, Dict, None] = kwargs.get("device_map", "auto")
        logger.info(f"HfCausalModel initialized. Target device_map: '{self.device_map}'")


    def token_len(self, text: str) -> int:
        """Calculates the number of tokens in a string using the loaded tokenizer."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call prepare_for_inference first.")
        try:
            # Tokenize without adding special tokens just to get length
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception as e:
            logger.error(f"Error calculating token length for text: '{text[:100]}...'. Error: {e}", exc_info=True)
            return -1 # Indicate error


    def prepare_for_inference(self, model_path: Optional[str] = None) -> None:
        target_model_path = model_path or self._get_model_path_from_args()
        if not target_model_path:
             raise ValueError("model_path is required either via argument or in args.")

        # --- Check if reloading is needed ---
        if self._loaded_model_path == target_model_path and self.model is not None and self.tokenizer is not None:
            logger.info(f"Model and tokenizer for '{target_model_path}' seem already loaded.")
            self._setup_generation_config(target_model_path)
            self._check_and_update_assistant_model(target_model_path)
            return

        logger.info(f"Preparing model for inference. Target path: '{target_model_path}'")
        if self._loaded_model_path and self._loaded_model_path != target_model_path:
             logger.warning(f"Switching loaded model from '{self._loaded_model_path}' to '{target_model_path}'.")
             self._unload_resources() # Unload previous resources first

        # --- Load Tokenizer ---
        try:
            self.tokenizer = self._load_tokenizer(target_model_path)
            logger.info(f"Tokenizer loaded: {self.tokenizer.__class__.__name__}")
            self._update_max_token_length() # Update based on loaded tokenizer
        except Exception as e:
            logger.error(f"Error loading tokenizer from '{target_model_path}': {e}", exc_info=True)
            self._unload_resources() # Cleanup
            raise

        # --- Load Main Model ---
        try:
            self.model = self._load_model(target_model_path)
            self._loaded_model_path = target_model_path
            self._log_model_placement() # Log where model is placed
        except Exception as e:
            logger.error(f"Error loading model from '{target_model_path}': {e}", exc_info=True)
            self._unload_resources() # Cleanup
            self._log_common_loading_errors(e) # Provide specific advice
            raise

        # --- Load/Unload Assistant Model ---
        self._check_and_update_assistant_model(target_model_path)

        # --- Setup Generation Config ---
        try:
            self._setup_generation_config(target_model_path)
            logger.info("Generation config setup complete.")
        except Exception as e:
            logger.error(f"Error setting up generation config for '{target_model_path}': {e}", exc_info=True)
            self._unload_resources() # Cleanup
            raise
        logger.info(f"Model preparation complete for '{target_model_path}'.")

    def _get_model_path_from_args(self) -> Optional[str]:
         """Tries to determine the model path from args (needs context or better arg names)."""
         path = getattr(self.args, "explore_model_path", None) or \
                getattr(self.args, "predict_model_path", None) or \
                getattr(self.args, "model_path", None)
         if path == "None": return None # Handle string "None"
         return path

    def _update_max_token_length(self) -> None:
        """Updates self.maximum_token based on tokenizer and args."""
        if self.tokenizer is None: return
        try:
            tokenizer_max_len = getattr(self.tokenizer, 'model_max_length', self.args.maximum_token)
            if not isinstance(tokenizer_max_len, int) or tokenizer_max_len > 100_000:
                tokenizer_max_len = self.args.maximum_token
            self.maximum_token = min(tokenizer_max_len, self.args.maximum_token)
            logger.info(f"Effective maximum token length set to: {self.maximum_token}")
        except Exception as e:
             logger.error(f"Could not determine tokenizer max length: {e}. Using default: {self.args.maximum_token}")
             self.maximum_token = self.args.maximum_token

    def _check_and_update_assistant_model(self, main_model_path: str) -> None:
        """Loads or unloads the assistant model based on current args."""
        target_assistant_path = self.args.assistant_model_path if self.args.use_assistant_model else None

        if target_assistant_path == self._loaded_assistant_path:
             logger.debug(f"Assistant model state consistent ('{target_assistant_path}'). No change needed.")
             return # No change needed

        if self.assistant_model is not None:
             logger.info(f"Unloading previous assistant model from '{self._loaded_assistant_path}'.")
             del self.assistant_model
             self.assistant_model = None
             self._loaded_assistant_path = None
             if torch.cuda.is_available():
                  gc.collect()
                  torch.cuda.empty_cache()

        if target_assistant_path and target_assistant_path != "None":
             try:
                  logger.info(f"Loading assistant model from '{target_assistant_path}'...")
                  self.assistant_model = self._load_model(target_assistant_path)
                  self._loaded_assistant_path = target_assistant_path
                  logger.info(f"Assistant model loaded successfully.")
                  self._log_model_placement(is_assistant=True) # Log its placement
             except Exception as e:
                  logger.error(f"Failed to load assistant model from '{target_assistant_path}': {e}", exc_info=True)
                  self.assistant_model = None
                  self._loaded_assistant_path = None

    def _unload_resources(self) -> None:
        """Helper to clean up loaded model, tokenizer, and assistant resources."""
        logger.info("Unloading model resources...")
        unloaded = False
        if hasattr(self, 'model') and self.model is not None:
            del self.model; self.model = None; unloaded = True
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
             del self.tokenizer; self.tokenizer = None; unloaded = True
        if hasattr(self, 'assistant_model') and self.assistant_model is not None:
             del self.assistant_model; self.assistant_model = None; unloaded = True

        self._loaded_model_path = None
        self._loaded_assistant_path = None

        if unloaded and torch.cuda.is_available():
            logger.info("Attempting garbage collection and CUDA cache clearing after unload.")
            gc.collect()
            torch.cuda.empty_cache()
        logger.info("Model resources unloaded complete.")

    def _log_model_placement(self, is_assistant: bool = False) -> None:
         """Logs the device placement of the main or assistant model."""
         model_ref = self.assistant_model if is_assistant else self.model
         model_name = "Assistant" if is_assistant else "Main"
         if model_ref is None: return

         if hasattr(model_ref, "hf_device_map"):
             logger.info(f"{model_name} model device map: {model_ref.hf_device_map}")
             # Optionally log detailed map if needed for debugging complex setups
             # for name, device in model_ref.hf_device_map.items():
             #      logger.debug(f"  Layer {name}: {device}")
         elif hasattr(model_ref, "device"):
             logger.info(f"{model_name} model loaded on device: {model_ref.device}")
         else:
              try: # Try getting device from first parameter
                   dev = next(model_ref.parameters()).device
                   logger.info(f"{model_name} model loaded (checked first param device): {dev}")
              except Exception:
                   logger.info(f"{model_name} model loaded. Cannot determine exact device placement.")

    def _log_common_loading_errors(self, exception: Exception) -> None:
         """Provides helpful logging messages for common model loading errors."""
         err_str = str(exception).lower()
         if "cuda error: device-side assert triggered" in err_str:
              logging.error("CUDA device-side assert triggered. Common causes: "
                           "1) Multi-GPU issues (try CUDA_VISIBLE_DEVICES=0). "
                           "2) Mismatched transformers/model/tokenizer versions. "
                           "3) Indexing errors in custom code using the model's outputs.")
         elif "out of memory" in err_str:
              logging.error("CUDA out of memory. Try: "
                           "1) Quantization (--quant 4bit/8bit). "
                           "2) --low_cpu_mem_usage. "
                           "3) Reducing max sequence length or batch size (if applicable). "
                           "4) Using a smaller model. "
                           "5) Ensuring only one model instance per GPU unless VRAM allows.")
         elif "connection error" in err_str or "repository not found" in err_str:
              logging.error(f"Connection error or model not found at path/ID. Check path, network connection, and HF token (--hf_token or HF_TOKEN env var). Path: {self._loaded_model_path or 'N/A'}")


    def _load_tokenizer(self, model_path_or_id: str) -> PreTrainedTokenizer:
        """Loads tokenizer with error handling and special token checks."""
        logger.info(f"Loading tokenizer from '{model_path_or_id}'...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path_or_id,
                token=self.args.hf_token,
                trust_remote_code=True
            )

            special_tokens_added = False
            if tokenizer.pad_token is None:
                if tokenizer.eos_token:
                     tokenizer.pad_token = tokenizer.eos_token
                     logger.warning(f"Tokenizer missing pad_token, setting pad_token to eos_token ('{tokenizer.eos_token}').")
                     special_tokens_added = True
                else:
                     logger.error("Tokenizer missing both pad_token and eos_token. Generation might fail.")
                     # Potentially add a default like "<|pad|>" but this requires resizing embeddings
                     # raise ValueError("Tokenizer requires at least an EOS token if PAD token is missing.")
            # Add other checks if needed (e.g., BOS token for some models)
            # if tokenizer.bos_token is None: logger.warning("Tokenizer missing bos_token.")

            # Resize embeddings if special tokens were added programmatically (Important!)
            # This part needs the model object, so maybe call this *after* model loading?
            # Or handle resizing during model loading itself if possible.
            # For now, just log the addition. Resizing logic might need to be integrated elsewhere.
            # if special_tokens_added:
            #      logger.info("Special tokens added to tokenizer. Ensure model embeddings are resized if necessary.")

            return tokenizer
        except Exception as e:
             logger.error(f"Failed to load tokenizer from '{model_path_or_id}': {e}", exc_info=True)
             raise
         
    def _load_model(self, model_path_or_id: str) -> PreTrainedModel:
        logger.info(f"Loading model from '{model_path_or_id}' with dtype={self.args.dtype}, quant={self.args.quant}, attn={self.args.attn_implementation or 'auto'}...")

        if self.tokenizer is None:
            logger.error("Tokenizer not loaded before model loading. This is a critical prerequisite.")
            raise ValueError("Tokenizer must be loaded before calling _load_model.")

        expected_vocab_size = len(self.tokenizer)
        
        # --- 参数准备 (不包含 config 对象，它将根据情况分别加载) ---
        common_load_kwargs = {
            "torch_dtype": self.DTYPE_MAPPING.get(self.args.dtype),
            "device_map": self.device_map,
            "token": self.args.hf_token,
            "trust_remote_code": True
        }

        if common_load_kwargs["torch_dtype"] is None:
            logger.warning(f"Invalid dtype '{self.args.dtype}' specified. Defaulting to torch.float16.")
            common_load_kwargs["torch_dtype"] = torch.float16
        if hasattr(self.args, "attn_implementation") and self.args.attn_implementation:
            logger.info(f"Requesting attn_implementation='{self.args.attn_implementation}'")
            common_load_kwargs["attn_implementation"] = self.args.attn_implementation
        else:
            logger.info("Using default attention implementation.")
            if "attn_implementation" in common_load_kwargs: del common_load_kwargs["attn_implementation"]
        
        load_in_4bit = False
        load_in_8bit = False
        if hasattr(self.args, "quant") and self.args.quant != "none":
            if self.args.quant == "4bit":
                logger.info("Applying 4-bit quantization.")
                load_in_4bit = True
                common_load_kwargs["load_in_4bit"] = True
                common_load_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
                common_load_kwargs["bnb_4bit_use_double_quant"] = True
                common_load_kwargs["bnb_4bit_quant_type"] = "nf4"
            elif self.args.quant == "8bit":
                logger.info("Applying 8-bit quantization.")
                load_in_8bit = True
                common_load_kwargs["load_in_8bit"] = True
            else:
                logger.warning(f"Unsupported quantization value: {self.args.quant}. Ignoring.")
        
        if (load_in_4bit or load_in_8bit) and isinstance(self.device_map, str) and self.device_map == "auto":
            logger.debug("Using device_map='auto' with quantization.")

        if hasattr(self.args, "low_cpu_mem_usage") and self.args.low_cpu_mem_usage:
            logger.info("Using low_cpu_mem_usage=True.")
            common_load_kwargs["low_cpu_mem_usage"] = True
        
        model: PreTrainedModel
        
        try:
            for attempt in range(2): # Flash Attention 重试循环
                # 在重试循环内部复制 common_load_kwargs，以防 attn_implementation 被修改
                current_attempt_load_kwargs = common_load_kwargs.copy()
                try:
                    is_adapter_path = os.path.exists(os.path.join(model_path_or_id, "adapter_config.json"))

                    if is_adapter_path:
                        from peft import PeftModel, PeftConfig
                        logger.info(f"Path '{model_path_or_id}' detected as a PEFT adapter. Loading base model first.")
                        
                        peft_config_adapter = PeftConfig.from_pretrained(model_path_or_id, token=self.args.hf_token)
                        base_model_name_or_path = peft_config_adapter.base_model_name_or_path
                        logger.info(f"Base model for PEFT adapter: '{base_model_name_or_path}'")

                        # 1. 使用其 *原始* 配置加载基础模型
                        # 注意：此时不传递已修改的 config 对象给 from_pretrained，让它加载原始大小
                        logger.info(f"Loading base model '{base_model_name_or_path}' with its original configuration first.")
                        base_model = AutoModelForCausalLM.from_pretrained(
                            base_model_name_or_path,
                            **current_attempt_load_kwargs # common_load_kwargs 不包含修改后的 config
                        )
                        
                        # 2. 如果需要，调整已加载基础模型的大小以匹配 tokenizer
                        if base_model.config.vocab_size != expected_vocab_size:
                            logger.info(
                                f"Base model '{base_model_name_or_path}' loaded with vocab_size {base_model.config.vocab_size}. "
                                f"Resizing to match tokenizer length {expected_vocab_size}."
                            )
                            base_model.resize_token_embeddings(expected_vocab_size)
                            # 验证 resize 是否生效
                            if base_model.config.vocab_size != expected_vocab_size or \
                               base_model.get_input_embeddings().weight.shape[0] != expected_vocab_size:
                                logger.error("CRITICAL: Base model vocab_size or embedding matrix size mismatch AFTER resize_token_embeddings!")
                                raise RuntimeError("Failed to resize base model token embeddings correctly.")
                        else:
                            logger.info(
                                f"Base model '{base_model_name_or_path}' vocab_size ({base_model.config.vocab_size}) "
                                f"already matches tokenizer length ({expected_vocab_size}). No resize needed."
                            )

                        # 3. 加载PEFT适配器到调整后的基础模型
                        logger.info(f"Loading PEFT adapter from '{model_path_or_id}' onto the (potentially resized) base model.")
                        model = PeftModel.from_pretrained(base_model, model_path_or_id, is_trainable=False)
                    
                    else: # 是一个完整模型的路径
                        logger.info(f"Path '{model_path_or_id}' detected as a full model directory.")
                        config_obj_full_model = AutoConfig.from_pretrained(
                            model_path_or_id,
                            token=self.args.hf_token,
                            trust_remote_code=True
                        )
                        if config_obj_full_model.vocab_size != expected_vocab_size:
                            logger.warning(
                                f"Full model config vocab_size ({config_obj_full_model.vocab_size}) from '{model_path_or_id}' "
                                f"does not match tokenizer length ({expected_vocab_size}). Updating config."
                            )
                            config_obj_full_model.vocab_size = expected_vocab_size
                        
                        current_attempt_load_kwargs_full_model = current_attempt_load_kwargs.copy()
                        current_attempt_load_kwargs_full_model['config'] = config_obj_full_model
                        
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path_or_id, 
                            **current_attempt_load_kwargs_full_model
                        )
                        # 对于已保存的完整微调模型，加载后其大小应该已经和 tokenizer 匹配
                        # 如果不匹配，说明保存的完整模型本身有问题
                        if model.get_input_embeddings().weight.shape[0] != expected_vocab_size:
                             logger.warning(
                                f"Full model from '{model_path_or_id}' has embedding size "
                                f"{model.get_input_embeddings().weight.shape[0]} which does not match tokenizer "
                                f"length {expected_vocab_size} even after using potentially modified config. "
                                f"This suggests the saved model weights are for a different vocab size. Attempting forceful resize."
                            )
                             model.resize_token_embeddings(expected_vocab_size) # 尝试强制修复

                    break # 成功加载，跳出重试循环
                except ImportError as ie_inner:
                    if 'flash_attn' in str(ie_inner).lower() and attempt == 0: 
                        logger.error("Flash Attention requested but not installed or failed. "
                                     "Attempting to load without flash attention...")
                        if "attn_implementation" in current_attempt_load_kwargs:
                            del current_attempt_load_kwargs["attn_implementation"]
                        # 如果 attn_implementation 是在 config 对象上设置的，也需要处理 (假设它不在 common_load_kwargs)
                        # (在当前实现中，attn_implementation 主要通过 common_load_kwargs 传递)
                    else:
                        raise ie_inner 
            
            # --- 最终检查和返回 ---
            if model.get_input_embeddings().weight.shape[0] != expected_vocab_size:
                logger.error(
                    f"CRITICAL: Final loaded model's embedding matrix size ({model.get_input_embeddings().weight.shape[0]}) "
                    f"from '{model_path_or_id}' (tokenizer length: {expected_vocab_size}) does NOT match. "
                )
                # raise ValueError("Fatal: Model embedding size and tokenizer length mismatch after all loading attempts.")
            else:
                logger.info(
                    f"Model loaded. Final embedding matrix size: {model.get_input_embeddings().weight.shape[0]}. "
                    f"Tokenizer length: {expected_vocab_size}."
                )

            model.eval()
            logger.info("Model set to evaluation mode.")
            return model

        except ImportError as ie_outer: 
            if 'bitsandbytes' in str(ie_outer).lower():
                logger.error("Quantization requested but bitsandbytes is not installed. Install with `pip install bitsandbytes`.")
            else:
                logger.error(f"Unhandled ImportError during model loading: {ie_outer}", exc_info=True)
            raise ie_outer
        except Exception as e:
            logger.error(f"General error during model loading from '{model_path_or_id}': {e}", exc_info=True)
            raise
        
    def _load_generation_config(self, model_path_or_id: str) -> GenerationConfig:
        """Loads generation config, with fallback for PEFT models."""
        try:
            logger.debug(f"Attempting to load GenerationConfig from '{model_path_or_id}'")
            # Pass token if needed for private models
            config = GenerationConfig.from_pretrained(model_path_or_id, token=self.args.hf_token)
            logger.info(f"Loaded GenerationConfig from '{model_path_or_id}'.")
            return config
        except Exception as e1:
            logger.warning(f"Failed to load GenerationConfig from '{model_path_or_id}' ({e1}). Checking PEFT...")
            try:
                # Only proceed if it looks like a local path potentially containing adapter_config.json
                if os.path.isdir(model_path_or_id):
                     peft_config = PeftConfig.from_pretrained(model_path_or_id, token=self.args.hf_token)
                     base_model_name = peft_config.base_model_name_or_path
                     logger.info(f"Detected PEFT adapter. Attempting to load GenerationConfig from base '{base_model_name}'")
                     config = GenerationConfig.from_pretrained(base_model_name, token=self.args.hf_token)
                     logger.info(f"Loaded GenerationConfig from base model '{base_model_name}'.")
                     return config
                else:
                    logger.warning("Not a local directory, cannot check for PEFT config.")
                    raise ValueError("Not a PEFT adapter path") # Trigger final fallback
            except Exception as e2:
                logger.error(f"Failed to load GenerationConfig from PEFT base model ({locals().get('base_model_name', 'N/A')}) or not a PEFT model: {e2}. Using default GenerationConfig.")
                return GenerationConfig() # Return default


    def _setup_generation_config(self, model_path: str) -> None:
        """Initializes and configures the default GenerationConfig based on args."""
        self.generation_cfg = self._load_generation_config(model_path)
        if hasattr(self.args, "max_new_tokens"):
             self.generation_cfg.max_new_tokens = self.args.max_new_tokens

        if self.tokenizer:
             if self.tokenizer.eos_token_id is not None:
                  self.generation_cfg.eos_token_id = self.tokenizer.eos_token_id
             if self.tokenizer.pad_token_id is not None:
                  self.generation_cfg.pad_token_id = self.tokenizer.pad_token_id
             elif self.generation_cfg.eos_token_id is not None: # Fallback pad = eos
                  self.generation_cfg.pad_token_id = self.generation_cfg.eos_token_id
                  logger.info(f"GenerationConfig using eos_token_id ({self.generation_cfg.eos_token_id}) as pad_token_id.")
             else:
                  logger.error("Cannot set generation config pad/eos tokens: Tokenizer missing required tokens.")

        k = max(1, getattr(self.args, 'generation_k', 1))
        self.args.generation_k = k

        mode = getattr(self.args, 'generation_mode', 'greedy')
        logger.info(f"Configuring generation for mode='{mode}' with k={k}")

        self.generation_cfg.do_sample = False
        self.generation_cfg.num_beams = 1
        self.generation_cfg.num_beam_groups = 1
        self.generation_cfg.early_stopping = False
        self.generation_cfg.num_return_sequences = 1 # Default to 1

        if mode == "greedy":
            self._configure_greedy()
        elif mode == "sampling":
            self._configure_sampling()
        elif mode == "beam":
            self._configure_beam(early_stopping=False)
        elif mode == "beam-early-stopping":
            self._configure_beam(early_stopping=True)
        elif mode == "group-beam":
            self._configure_group_beam(early_stopping=False)
        elif mode == "group-beam-early-stopping":
             self._configure_group_beam(early_stopping=True)
        else:
             logger.warning(f"Unknown generation_mode '{mode}'. Using default greedy settings.")
             self._configure_greedy()

        # Set num_return_sequences based on 'k' for relevant modes AFTER mode config
        if mode in ["sampling", "beam", "group-beam"]: # , "beam-early-stopping", "group-beam-early-stopping"
             self.generation_cfg.num_return_sequences = k

        logger.debug(f"Final default GenerationConfig: {self.generation_cfg.to_dict()}")


    # --- Configuration Helper Methods ---
    # These now only modify the self.generation_cfg set in _setup_generation_config

    def _configure_greedy(self):
        self.generation_cfg.do_sample = False
        self.generation_cfg.num_beams = 1

    def _configure_sampling(self):
        self.generation_cfg.do_sample = True
        self.generation_cfg.num_beams = 1
        # Set defaults if not loaded from config
        self.generation_cfg.temperature = getattr(self.generation_cfg, 'temperature', 0.7)
        self.generation_cfg.top_p = getattr(self.generation_cfg, 'top_p', 0.9)
        # self.generation_cfg.top_k = getattr(self.generation_cfg, 'top_k', 50) # Often better to use top_p OR top_k

    def _configure_beam(self, early_stopping: bool = False):
        self.generation_cfg.do_sample = False
        self.generation_cfg.num_beams = self.args.generation_k # k is num_beams
        self.generation_cfg.early_stopping = early_stopping

    def _configure_group_beam(self, early_stopping: bool = False):
        num_groups = self.args.generation_k
        if num_groups <= 1:
            logger.warning(f"generation_mode='group-beam' requires generation_k > 1 (received k={self.args.generation_k}). "
                        f"Falling back to standard beam search configuration with num_beams=k.")
            # Apply standard beam search config instead
            self._configure_beam(early_stopping=early_stopping) # Call beam config helper
            # Ensure num_beam_groups is reset if necessary (should be default 1)
            self.generation_cfg.num_beam_groups = 1
            return # Exit after applying fallback

        # Proceed with group beam config only if num_groups > 1
        self.generation_cfg.num_beam_groups = num_groups
        self.generation_cfg.num_beams = max(num_groups, num_groups * 2)
        # ... (rest of group beam config) ...
        self.generation_cfg.do_sample = False
        self.generation_cfg.diversity_penalty = getattr(self.args, 'diversity_penalty', 0.9)
        self.generation_cfg.early_stopping = early_stopping


    def prepare_model_prompt(self, query: str) -> str:
        """Applies chat template to the query if configured."""
        if getattr(self.args, 'chat_model', False): # Use getattr for safety
            if self.tokenizer is None:
                 # Raise error as this indicates prepare_for_inference wasn't called
                 raise RuntimeError("Tokenizer not initialized for chat template application.")
            try:
                messages = [{"role": "user", "content": query}]
                templated_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True # Crucial for inference
                )
                # Check if template application returned None or empty (can happen with some tokenizers/templates)
                if not templated_prompt:
                     logger.warning("apply_chat_template returned empty result. Falling back to raw query.")
                     return query
                return templated_prompt
            except Exception as e:
                 logger.error(f"Failed to apply chat template. Falling back to raw query. Error: {e}", exc_info=True)
                 return query
        return query


    @torch.inference_mode() # Use decorator - ensures no grads for the whole method
    def generate_sentence(self, llm_input: str, temp_generation_mode: Optional[str] = None, **kwargs) -> Optional[Union[str, List[str]]]:
        # if self.model is None or self.tokenizer is None or self.generation_cfg is None:
        #     logger.error("Model, Tokenizer, or Generation Config not initialized. Call prepare_for_inference first.")
        #     return None

        try:
            inputs = self.tokenizer(
                llm_input,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=self.maximum_token
            )
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device)
            actual_input_length = input_ids.shape[1]

            if actual_input_length == self.maximum_token:
                 logger.warning(f"Input was truncated to maximum token length ({self.maximum_token}).")
            logger.info(f"Generating with input length: {actual_input_length} tokens.")

        except Exception as e:
            logger.error(f"Error during input tokenization/truncation: {e}", exc_info=True)
            return None

        try:
            gen_config = self.generation_cfg # Start with default
            if temp_generation_mode:
                 temp_config = self._get_temp_generation_config(temp_generation_mode)
                 if temp_config: gen_config = temp_config # Use temp config if valid
                 else: logger.warning(f"Could not get temp config for '{temp_generation_mode}', using default.")
            if kwargs:
                 updated_config = self._update_generation_config(gen_config, kwargs)
                 if updated_config: gen_config = updated_config # Use updated config if valid
                 else: logger.warning(f"Could not update config with kwargs '{kwargs}', using previous config.")

            # Final check on essential tokens in the config to be used
            if gen_config.pad_token_id is None or gen_config.eos_token_id is None:
                  logger.error(f"Cannot generate: Final generation config missing pad_token_id or eos_token_id.")
                  return None
            logger.debug(f"Using final GenerationConfig: {gen_config.to_dict()}")

        except Exception as e:
             logger.error(f"Error preparing generation config: {e}", exc_info=True)
             return None

        # --- Generation ---
        try:
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
                # No need to pass these explicitly if they are in gen_config
                # return_dict_in_generate=True,
                # pad_token_id=gen_config.pad_token_id
            )
        except Exception as e:
            logger.error(f"Error during model.generate call: {e}", exc_info=True)
            self._log_common_loading_errors(e) # Log OOM etc.
            return None

        # --- Decoding ---
        try:
            generated_sequences = outputs.sequences
            num_outputs = generated_sequences.shape[0]

            if num_outputs == 1:
                 result = self._decode_generated_text(generated_sequences[0], actual_input_length)
            else:
                 # Decode all returned sequences
                 result = [self._decode_generated_text(seq, actual_input_length) for seq in generated_sequences]
                 # Filter out potential empty strings if decoding failed for some sequences?
                 # result = [r for r in result if r is not None] # Assumes _decode returns None on error

            output_count_str = f"{len(result)}" if isinstance(result, list) else "1"
            logger.info(f"Generation successful. Output count: {output_count_str}")
            return result

        except Exception as e:
            logger.error(f"Error decoding generated sequences: {e}", exc_info=True)
            return None


    def _get_temp_generation_config(self, mode: str) -> Optional[GenerationConfig]:
        """Creates a temporary GenerationConfig for a specific mode, inheriting essential tokens."""
        if self.generation_cfg is None or self.tokenizer is None:
             logger.error("Cannot create temp generation config, base config or tokenizer missing.")
             return None
        try:
            config = GenerationConfig(
                 max_new_tokens=self.args.max_new_tokens,
                 return_dict_in_generate=True,
                 pad_token_id=self.generation_cfg.pad_token_id,
                 eos_token_id=self.generation_cfg.eos_token_id
            )
            k = max(1, self.args.generation_k)

            if mode == "greedy":
                 config.do_sample = False; config.num_beams = 1; config.num_return_sequences = 1
            elif mode == "beam":
                 config.do_sample = False; config.num_beams = k; config.num_return_sequences = k
            elif mode == "sampling":
                 config.do_sample = True; config.num_beams = 1; config.num_return_sequences = k
                 config.temperature = getattr(self.args, 'temperature', 0.7)
                 config.top_p = getattr(self.args, 'top_p', 0.9)
            elif mode == "group-beam":
                num_groups = k # k is already max(1, ...)
                if num_groups <= 1:
                    logger.warning(f"Temporary mode 'group-beam' requires generation_k > 1 (using k={k}). "
                                f"Applying standard beam search config instead for this call.")
                    config.do_sample = False
                    config.num_beams = k
                    config.num_beam_groups = 1
                    config.num_return_sequences = k
                else:
                    config.num_beam_groups = num_groups
                    config.num_beams = max(num_groups, num_groups * 2)
                    config.do_sample = False
                    config.diversity_penalty = getattr(self.args, 'diversity_penalty', 0.9)
                    config.num_return_sequences = k
            # Add early stopping variants if needed
            # elif mode == "beam-early-stopping":
            #      config.do_sample = False; config.num_beams = k; config.num_return_sequences = k; config.early_stopping = True
            # elif mode == "group-beam-early-stopping":
            #      num_groups = k; config.num_beam_groups = num_groups; config.num_beams = max(num_groups, num_groups * 2)
            #      config.do_sample = False; config.diversity_penalty = getattr(self.args, 'diversity_penalty', 0.9); config.num_return_sequences = k; config.early_stopping = True
            else:
                 logger.error(f"Unsupported temporary generation mode requested: '{mode}'.")
                 return None # Return None if mode is invalid

            logger.debug(f"Created temporary GenerationConfig for mode '{mode}'.")
            return config
        except Exception as e:
             logger.error(f"Failed to create temporary generation config for mode '{mode}': {e}", exc_info=True)
             return None


    def _update_generation_config(self, config: GenerationConfig, kwargs: Dict[str, Any]) -> Optional[GenerationConfig]:
        """Creates a *new* GenerationConfig by updating an existing one with valid kwargs."""
        if not kwargs: return config
        try:
            new_config_dict = config.to_dict()
            valid_updates = {}
            for key, value in kwargs.items():
                 if key in new_config_dict:
                      valid_updates[key] = value
                 else:
                      logger.warning(f"Ignoring unsupported generation parameter override via kwargs: {key}={value}")

            if not valid_updates: return config

            new_config_dict.update(valid_updates)
            new_config = GenerationConfig(**new_config_dict)
            logger.debug(f"Updated GenerationConfig with kwargs: {valid_updates}")
            return new_config
        except Exception as e:
             logger.error(f"Failed to update generation config with kwargs {kwargs}: {e}", exc_info=True)
             return None

    def _decode_generated_text(self, sequence: torch.Tensor, input_length: int) -> Optional[str]:
        """Decodes the generated part of a sequence tensor."""
        if self.tokenizer is None:
            logger.error("Cannot decode text, tokenizer is not initialized.")
            return "[Decoding Error: Tokenizer Missing]"
        try:
            generated_ids = sequence[min(input_length, len(sequence)):]
            if len(generated_ids) == 0:
                 logger.warning("No new tokens were generated (sequence length <= input length).")
                 return ""
            decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return decoded_text.strip()
        except Exception as e:
             logger.error(f"Error decoding sequence (length {len(sequence)}, input {input_length}): {e}", exc_info=False)
             return "[Decoding Error]"