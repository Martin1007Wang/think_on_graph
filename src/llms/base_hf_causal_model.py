# -*- coding: utf-8 -*-
import torch
import logging
import os
import gc
from typing import List, Optional, Union, Dict, Any, Tuple
from unsloth import FastLanguageModel
from transformers import GenerationConfig,PreTrainedTokenizer,PreTrainedModel
from .base_language_model import BaseLanguageModel, GenerationError, ModelNotReadyError, ConfigurationError

import dotenv
dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
logger = logging.getLogger(__name__)


class HfCausalModel(BaseLanguageModel):
    """
    A Hugging Face Causal Language Model wrapper, optimized with Unsloth for efficient loading and inference.
    This class provides a structured interface for model preparation, generation, and resource management.
    """
    DTYPE_MAPPING = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }
    
    @classmethod
    def from_args(cls, args: Any) -> 'HfCausalModel':
        logger.info(f"Instantiating HfCausalModel using the 'from_args' factory method.")
        return cls(args)
    
    @staticmethod
    def add_args(parser: Any):
        # ... (add_args content remains unchanged)
        group = parser.add_argument_group("HfCausalModel Specific Arguments")
        group.add_argument("--maximum_token", type=int, default=8000, help="Maximum token length allowed by the tokenizer/model context window.")
        group.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
        group.add_argument("--dtype", choices=list(HfCausalModel.DTYPE_MAPPING.keys()), default="bf16", help="Data type for model loading (fp32, fp16, bf16).")
        group.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none",help="Quantization type (none, 4bit, 8bit). Requires bitsandbytes.")
        group.add_argument("--attn_implementation", default="flash_attention_2",choices=["eager", "sdpa", "flash_attention_2", None],help="Attention implementation (requires compatible transformers/hardware). Set to None to let HF decide.")
        group.add_argument("--generation_mode", type=str, default="greedy",choices=["greedy", "beam", "sampling", "group-beam"],help="Default generation strategy.")
        group.add_argument("--generation_k", type=int, default=1, help="Parameter 'k' for generation (e.g., num_beams, num_return_sequences for sampling, num_groups for group-beam). Must be >= 1.")
        group.add_argument("--chat_model", default='true', type=lambda x: (str(x).lower() == 'true'), help="Apply chat template if true.")
        group.add_argument("--use_assistant_model", default='false', type=lambda x: (str(x).lower() == 'true'), help="Load an assistant model (if specified). Usage depends on external logic.")
        group.add_argument("--assistant_model_path", type=str, default=None, help="Path to the assistant model.")
        group.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face API token (optional, reads from env var HF_TOKEN by default).")
        group.add_argument("--low_cpu_mem_usage", action='store_true', help="Use low_cpu_mem_usage during model loading (useful for large models).")


    def __init__(self, args: Any, **kwargs: Any):
        # --- MODIFIED: Call super().__init__() to set up counters ---
        super().__init__()
        self.args = args
        self._loaded_model_path: Optional[str] = None
        # _is_ready is now inherited from base class

        self.maximum_token: int = args.maximum_token
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[PreTrainedModel] = None
        self.generation_cfg: Optional[GenerationConfig] = None

        self.device_map: Union[str, Dict, None] = kwargs.get("device_map", "auto")
        logger.info(f"HfCausalModel initialized. Target device_map: '{self.device_map}'")

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def prepare_for_inference(self, model_path: Optional[str] = None) -> None:
        # ... (prepare_for_inference content remains unchanged)
        target_model_path = model_path or self._get_model_path_from_args()
        if not target_model_path:
            raise ConfigurationError("model_path is required either via argument or in args.")

        if self._loaded_model_path == target_model_path and self.is_ready:
            logger.info(f"Model and tokenizer for '{target_model_path}' are already loaded and prepared.")
            return
        
        if self._loaded_model_path and self._loaded_model_path != target_model_path:
            logger.warning(f"Switching model from '{self._loaded_model_path}' to '{target_model_path}'.")
            self.unload_resources()

        logger.info(f"Preparing model for inference. Target path: '{target_model_path}'")
        try:
            self.model, self.tokenizer = self._load_model_and_tokenizer_with_unsloth(target_model_path)
            self._loaded_model_path = target_model_path
            
            logger.info("Applying Unsloth's final inference optimizations. This may take a moment for JIT compilation...")
            FastLanguageModel.for_inference(self.model)
            logger.info("Inference optimization complete. Model is ready for high-speed generation.")
            
            self._setup_generation_config(target_model_path)
            self._update_max_token_length()
            
            self._is_ready = True
            logger.info(f"Model preparation complete for '{target_model_path}'.")

        except Exception as e:
            logger.error(f"Fatal error during model preparation for '{target_model_path}': {e}", exc_info=True)
            self.unload_resources()
            raise ModelNotReadyError(f"Failed to prepare model: {e}") from e

    def _load_model_and_tokenizer_with_unsloth(self, model_path_or_id: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        # ... (_load_model_and_tokenizer_with_unsloth content remains unchanged)
        logger.info(f"Loading model and tokenizer from '{model_path_or_id}' using Unsloth...")
        
        load_in_4bit = getattr(self.args, 'quant', 'none') == '4bit'
        load_in_8bit = getattr(self.args, 'quant', 'none') == '8bit'
        dtype_str = getattr(self.args, 'dtype', 'bf16')
        torch_dtype = self.DTYPE_MAPPING.get(dtype_str, torch.bfloat16)

        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path_or_id,
                max_seq_length=self.maximum_token,
                dtype=torch_dtype,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                token=getattr(self.args, 'hf_token', HF_TOKEN),
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.warning("Unsloth: Tokenizer's pad_token was not set. Setting it to eos_token.")

            logger.info("Unsloth: Model and tokenizer loaded successfully.")
            model.eval()
            return model, tokenizer

        except Exception as e:
            logger.error(f"Unsloth failed to load model from '{model_path_or_id}': {e}", exc_info=True)
            self._log_common_loading_errors(e)
            raise


    def unload_resources(self) -> None:
        if not self._is_ready and self._loaded_model_path is None:
            logger.info("No resources to unload.")
            return

        logger.info("Unloading model resources...")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._loaded_model_path = None
        self._is_ready = False

        if torch.cuda.is_available():
            logger.info("Attempting garbage collection and CUDA cache clearing after unload.")
            gc.collect()
            torch.cuda.empty_cache()
        logger.info("Model resources unloaded successfully.")

    @torch.inference_mode()
    def generate_sentence(self, prompts: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        if not self.is_ready or self.model is None or self.tokenizer is None or self.generation_cfg is None:
            raise ModelNotReadyError("Model is not ready for inference. Call prepare_for_inference() first.")

        is_single_prompt = isinstance(prompts, str)
        prompts_list = [prompts] if is_single_prompt else prompts
        
        try:
            inputs = self.tokenizer(prompts_list, padding=True, return_tensors="pt").to(self.model.device)
            
            final_generation_config = self.generation_cfg.to_dict()
            final_generation_config.update(kwargs)
            
            # --- MODIFIED: Increment call counter ---
            self.call_counter += 1
            
            outputs = self.model.generate(**inputs, **final_generation_config)

            # --- MODIFIED: Add token counting logic ---
            # Counts all non-padding tokens in the output (prompt + completion)
            if self.tokenizer:
                num_tokens = torch.sum(outputs != self.tokenizer.pad_token_id).item()
                self.token_counter += num_tokens
            
            decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # This logic to strip prompt from output seems incorrect for batch, let's correct it
            # The original prompt length in tokens might differ after padding
            input_lengths = inputs['input_ids'].shape[1]
            results = [
                self.tokenizer.decode(output[input_lengths:], skip_special_tokens=True).strip()
                for output in outputs
            ]

            return results[0] if is_single_prompt else results
            
        except Exception as e:
            logger.error(f"Error during batch generation: {e}", exc_info=True)
            raise GenerationError(f"Generation failed: {e}") from e

    # ... (Other private methods like _get_model_path_from_args, etc., remain unchanged)
    def _get_model_path_from_args(self) -> Optional[str]:
        path = getattr(self.args, "explore_model_path", None) or \
               getattr(self.args, "predict_model_path", None) or \
               getattr(self.args, "model_path", None)
        return path if path != "None" else None

    def _update_max_token_length(self) -> None:
        if self.tokenizer is None: return
        try:
            tokenizer_max_len = getattr(self.tokenizer, 'model_max_length', self.args.maximum_token)
            if not isinstance(tokenizer_max_len, int) or tokenizer_max_len > 1e5:
                tokenizer_max_len = self.args.maximum_token
            self.maximum_token = min(tokenizer_max_len, self.args.maximum_token)
            logger.info(f"Effective maximum token length set to: {self.maximum_token}")
        except Exception:
            self.maximum_token = self.args.maximum_token

    def _log_common_loading_errors(self, exception: Exception) -> None:
        err_str = str(exception).lower()
        if "out of memory" in err_str:
            logger.error("CUDA out of memory. Try using quantization (--quant 4bit/8bit) or a smaller model.")
        elif "connection error" in err_str or "repository not found" in err_str:
            logger.error(f"Connection error or model not found. Check path/ID, network, and HF token.")

    def _setup_generation_config(self, model_path: str) -> None:
        try:
            self.generation_cfg = GenerationConfig.from_pretrained(
                model_path, token=getattr(self.args, 'hf_token', HF_TOKEN)
            )
        except Exception:
            logger.warning(f"Could not load generation_config.json from '{model_path}'. Using default.")
            self.generation_cfg = GenerationConfig()

        if self.tokenizer:
            self.generation_cfg.pad_token_id = self.tokenizer.pad_token_id
            self.generation_cfg.eos_token_id = self.tokenizer.eos_token_id
        
        self.generation_cfg.max_new_tokens = getattr(self.args, 'max_new_tokens', 512)
        
        mode = getattr(self.args, 'generation_mode', 'greedy')
        k = getattr(self.args, 'generation_k', 1)
        
        logger.info(f"Setting up generation config for mode='{mode}' with k={k}.")

        if mode == 'greedy':
            self.generation_cfg.do_sample = False
            self.generation_cfg.num_beams = 1
        elif mode == 'beam':
            self.generation_cfg.do_sample = False
            self.generation_cfg.num_beams = k
        elif mode == 'sampling':
            self.generation_cfg.do_sample = True
            self.generation_cfg.top_k = 50 
            self.generation_cfg.top_p = 0.9
            self.generation_cfg.temperature = 0.7
            self.generation_cfg.num_return_sequences = k
        elif mode == 'group-beam':
            self.generation_cfg.num_beams = k
            self.generation_cfg.num_beam_groups = k 
            
        self.generation_cfg.use_cache = True
        logger.info(f"Default GenerationConfig prepared with the following settings: {self.generation_cfg.to_dict()}")