from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from .base_language_model import BaseLanguageModel
import os
import dotenv
from peft import PeftConfig
from typing import List, Optional, Union, Dict, Any
import logging
from pathlib import Path
from transformers import PreTrainedTokenizer, PreTrainedModel

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

logger = logging.getLogger(__name__)

class HfCausalModel(BaseLanguageModel):
    """
    Hugging Face Causal Language Model wrapper for text generation.
    
    This class provides functionality to load and use Hugging Face causal language
    models for text generation with various configurations.
    """
    
    DTYPE_MAPPING = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        group = parser.add_argument_group("HfCausalModel")
        group.add_argument("--maximum_token", type=int, default=8192, 
                          help="Maximum token length for input")
        group.add_argument("--max_new_tokens", type=int, default=1024,
                          help="Maximum number of new tokens to generate")
        group.add_argument("--dtype", choices=list(HfCausalModel.DTYPE_MAPPING.keys()), 
                          default="bf16", help="Model precision")
        group.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none",
                          help="Model quantization level")
        group.add_argument("--attn_implementation", 
                          default="flash_attention_2",
                          choices=["eager", "sdpa", "flash_attention_2"],
                          help="Attention implementation method")
        group.add_argument("--generation_mode", type=str, default="greedy",
                          choices=["greedy", "beam", "sampling", "group-beam", 
                                   "beam-early-stopping", "group-beam-early-stopping"],
                          help="Text generation strategy")
        group.add_argument("--generation_k", type=int, default=1, 
                          help="Number of paths/sequences to generate")
        group.add_argument("--chat_model", default='true', 
                          type=lambda x: (str(x).lower() == 'true'),
                          help="Whether the model is a chat model")
        group.add_argument("--use_assistant_model", default='false', 
                          type=lambda x: (str(x).lower() == 'true'),
                          help="Whether to use an assistant model")
        group.add_argument("--assistant_model_path", type=str, default=None,
                          help="Path to the assistant model")
        group.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"),
                          help="Hugging Face API token")

    def __init__(self, args):
        """
        Initialize the model with given arguments.
        
        Args:
            args: Command line arguments containing model configuration
        """
        super().__init__(args)
        self.args = args
        self.maximum_token = args.maximum_token
        self.hf_token = args.hf_token if hasattr(args, "hf_token") else os.getenv("HF_TOKEN")
        if self.hf_token is None:
            logger.warning("Hugging Face token is not provided.")
        self.model = None
        self.tokenizer = None
        self.assistant_model = None
        self.generation_cfg = None
        
    def token_len(self, text: str) -> int:
        """
        Get the tokenized length of the input text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens in the text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call prepare_for_inference first.")
        return len(self.tokenizer.tokenize(text))

    def prepare_for_inference(self):
        """
        Load and prepare the model, tokenizer and configuration for inference.
        """
        if self.model is not None and self.tokenizer is not None:
            logger.info("Model and tokenizer already loaded, skipping preparation")
            return

        # Load tokenizer
        logger.info(f"Loading tokenizer from {self.args.model_path}")
        self.tokenizer = self._load_tokenizer(self.args.model_path)
        
        # Load main model
        logger.info(f"Loading model from {self.args.model_path}")
        try:
            self.model = self._load_model(self.args.model_path)
        except RuntimeError as e:
            logger.error(f"Failed to load model due to memory constraints: {e}")
            raise
        
        # Load assistant model if specified
        if self.args.use_assistant_model and self.args.assistant_model_path:
            logger.info(f"Loading assistant model from {self.args.assistant_model_path}")
            try:
                self.assistant_model = self._load_model(self.args.assistant_model_path)
            except RuntimeError as e:
                logger.error(f"Failed to load assistant model due to memory constraints: {e}")
                raise
        
        # Update maximum token length based on loaded tokenizer
        self.maximum_token = self.tokenizer.model_max_length
        
        # Prepare generation configuration
        self._setup_generation_config()
        
        logger.info("Model preparation complete")
        
    def _load_tokenizer(self, model_path: str) -> PreTrainedTokenizer:
        """Load and configure the tokenizer."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                token=self.hf_token, 
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {model_path}: {e}")
            raise

        # Configure tokenizer padding and special tokens
        tokenizer.padding_side = "right"
        special_tokens_dict = {}
        
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = "<PAD>"
            
        if special_tokens_dict:
            tokenizer.add_special_tokens(special_tokens_dict)
            
        return tokenizer
    
    def _load_model(self, model_path: str) -> PreTrainedModel:
        """Load the model with proper configuration."""
        kwargs = {
            "device_map": "auto",
            "token": self.hf_token,
            "torch_dtype": self.DTYPE_MAPPING.get(self.args.dtype),
            "load_in_8bit": self.args.quant == "8bit",
            "load_in_4bit": self.args.quant == "4bit",
            "trust_remote_code": True,
            "attn_implementation": self.args.attn_implementation,
        }
        
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        
        # Resize token embeddings if the tokenizer vocabulary was modified
        if self.tokenizer is not None and model.config.vocab_size != len(self.tokenizer):
            model.resize_token_embeddings(len(self.tokenizer))
            
        # Disable model caching for faster inference
        model.config.use_cache = False
        
        return model
    
    def _setup_generation_config(self):
        """Setup the generation configuration based on arguments."""
        try:
            self.generation_cfg = GenerationConfig.from_pretrained(self.args.model_path)
        except Exception as e:
            logger.warning(f"Failed to load generation config from model path: {e}")
            try:
                peft_config = PeftConfig.from_pretrained(self.args.model_path)
                self.generation_cfg = GenerationConfig.from_pretrained(peft_config.base_model_name_or_path)
            except Exception as e2:
                logger.warning(f"Failed to load generation config from peft model: {e2}")
                self.generation_cfg = GenerationConfig()
        
        # Apply common configurations
        self.generation_cfg.max_new_tokens = self.args.max_new_tokens
        self.generation_cfg.return_dict_in_generate = True
        
        # Validate generation_k
        if self.args.generation_k < 1:
            logger.warning(f"Invalid generation_k ({self.args.generation_k}), setting to 1")
            self.args.generation_k = 1
        
        # Configure based on generation mode
        if self.args.generation_mode == "greedy":
            self._configure_greedy_generation()
        elif self.args.generation_mode == "sampling":
            self._configure_sampling_generation()
        elif self.args.generation_mode == "beam":
            self._configure_beam_search()
        elif self.args.generation_mode == "beam-early-stopping":
            self._configure_beam_search(early_stopping=True)
        elif self.args.generation_mode == "group-beam":
            self._configure_group_beam_search()
        elif self.args.generation_mode == "group-beam-early-stopping":
            self._configure_group_beam_search(early_stopping=True)
            
    def _configure_greedy_generation(self):
        """Configure for greedy generation."""
        self.generation_cfg.do_sample = False
        self.generation_cfg.num_return_sequences = 1
        
    def _configure_sampling_generation(self):
        """Configure for sampling-based generation."""
        self.generation_cfg.do_sample = True
        self.generation_cfg.num_return_sequences = self.args.generation_k
        self.generation_cfg.temperature = 0.7
        self.generation_cfg.top_p = 0.9
        
    def _configure_beam_search(self, early_stopping: bool = False):
        """Configure for beam search generation."""
        self.generation_cfg.do_sample = False
        self.generation_cfg.num_beams = self.args.generation_k
        self.generation_cfg.num_return_sequences = self.args.generation_k
        self.generation_cfg.early_stopping = early_stopping
        
    def _configure_group_beam_search(self, early_stopping: bool = False):
        """Configure for group beam search generation."""
        self.generation_cfg.do_sample = False
        self.generation_cfg.num_beams = self.args.generation_k * 2
        self.generation_cfg.num_return_sequences = self.args.generation_k
        self.generation_cfg.num_beam_groups = min(self.args.generation_k, 5)
        self.generation_cfg.diversity_penalty = 1.0
        self.generation_cfg.early_stopping = early_stopping

    def prepare_model_prompt(self, query: str) -> str:
        """
        Format the input query according to the model type.
        
        Args:
            query: User query text
            
        Returns:
            Formatted prompt ready for the model
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call prepare_for_inference first.")
            
        if self.args.chat_model:
            chat_query = [{"role": "user", "content": query}]
            try:
                return self.tokenizer.apply_chat_template(
                    chat_query, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.error(f"Failed to apply chat template: {e}")
                raise
        return query
    
    @torch.inference_mode()
    def generate_sentence(self, llm_input: str, temp_generation_mode: Optional[str] = None, **kwargs) -> Union[str, List[str]]:
        """
        Generate text based on the input.
        
        Args:
            llm_input: The input text to generate from
            temp_generation_mode: Temporary generation mode for this specific call
                Options: "greedy", "beam", "sampling", "group-beam"
            **kwargs: Additional arguments to pass to the generation function
            
        Returns:
            Generated text or list of texts if multiple sequences are requested
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not initialized. Call prepare_for_inference first.")
            
        # Tokenize the input
        inputs = self.tokenizer(
            llm_input, 
            return_tensors="pt", 
            add_special_tokens=False,
            truncation=True,
            max_length=self.maximum_token - self.args.max_new_tokens
        )
        
        # Move tensors to the same device as the model
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        
        try:
            # Create a temporary generation config if needed
            generation_config = self.generation_cfg
            if temp_generation_mode:
                generation_config = self._get_temp_generation_config(temp_generation_mode)
            
            # Override with any provided kwargs
            if kwargs:
                generation_config = self._update_generation_config(generation_config, kwargs)
            
            # Generate text
            res = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Process the results
            if len(res.sequences) == 1:
                return self._decode_generated_text(res.sequences[0], input_ids.shape[1])
            return [self._decode_generated_text(seq, input_ids.shape[1]) for seq in res.sequences]
                
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            raise

    def _get_temp_generation_config(self, mode: str) -> GenerationConfig:
        """Create a temporary generation config based on mode."""
        # 创建基础配置
        config = GenerationConfig(
            max_new_tokens=self.args.max_new_tokens,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        if mode == "greedy":
            # 贪婪搜索的基本配置
            config.do_sample = False
            config.num_beams = 1
            config.num_return_sequences = 1
            
        elif mode == "beam":
            # beam search配置
            config.do_sample = False
            config.num_beams = max(4, self.args.generation_k)
            config.num_return_sequences = 1
            config.num_beam_groups = 1
            
        elif mode == "sampling":
            # 采样配置
            config.do_sample = True
            config.num_beams = 1
            config.num_return_sequences = 1
            config.temperature = 0.7
            config.top_p = 0.9
            
        elif mode == "group-beam":
            # 分组beam search配置
            config.do_sample = False
            config.num_beams = max(4, self.args.generation_k * 2)
            config.num_return_sequences = 1
            config.num_beam_groups = min(max(2, self.args.generation_k), 5)
            config.diversity_penalty = 1.0
        
        return config

    def _update_generation_config(self, config: GenerationConfig, kwargs: Dict[str, Any]) -> GenerationConfig:
        """Update generation config with provided kwargs."""
        # 创建新的配置
        new_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            return_dict_in_generate=config.return_dict_in_generate,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # 复制所有非None的属性
        for key, value in config.__dict__.items():
            if not key.startswith('_') and value is not None:
                setattr(new_config, key, value)
            
        # 更新新的参数
        for key, value in kwargs.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
            else:
                logger.warning(f"Ignoring unsupported generation parameter: {key}")
        
        return new_config

    def _decode_generated_text(self, sequence, input_length: int) -> str:
        """Helper function to decode generated text."""
        return self.tokenizer.decode(
            sequence[input_length:],
            skip_special_tokens=True
        )