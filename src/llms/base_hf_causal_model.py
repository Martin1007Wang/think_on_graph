from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from .base_language_model import BaseLanguageModel
import os
import dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftConfig
from typing import List, Optional, Union, Dict, Any, Tuple
import logging
from pathlib import Path
from transformers import (
    PreTrainedTokenizer, 
    PreTrainedModel
)

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
        group.add_argument("--maximun_token", type=int, default=4096, 
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
        group.add_argument("--k", type=int, default=1, 
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
        self.maximun_token = args.maximun_token
        self.hf_token = args.hf_token if hasattr(args, "hf_token") else os.getenv("HF_TOKEN")
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
        # Load tokenizer
        logger.info(f"Loading tokenizer from {self.args.model_path}")
        self.tokenizer = self._load_tokenizer(self.args.model_path)
        
        # Load main model
        logger.info(f"Loading model from {self.args.model_path}")
        self.model = self._load_model(self.args.model_path)
        
        # Load assistant model if specified
        if self.args.use_assistant_model and self.args.assistant_model_path:
            logger.info(f"Loading assistant model from {self.args.assistant_model_path}")
            self.assistant_model = self._load_model(self.args.assistant_model_path)
        
        # Update maximum token length based on loaded tokenizer
        self.maximun_token = self.tokenizer.model_max_length
        
        # Prepare generation configuration
        self._setup_generation_config()
        
        logger.info("Model preparation complete")
        
    def _load_tokenizer(self, model_path: str) -> PreTrainedTokenizer:
        """Load and configure the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            token=self.hf_token, 
            trust_remote_code=True
        )
        
        # Configure tokenizer padding and special tokens
        tokenizer.padding_side = "right"
        special_tokens_dict = {}
        
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = "<PAD>"
            
        # Add any other required special tokens here
        
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
                # Try loading from PeftModel
                peft_config = PeftConfig.from_pretrained(self.args.model_path)
                self.generation_cfg = GenerationConfig.from_pretrained(peft_config.base_model_name_or_path)
            except Exception as e2:
                logger.warning(f"Failed to load generation config from peft model: {e2}")
                # Create default config
                self.generation_cfg = GenerationConfig()
        
        # Apply common configurations
        self.generation_cfg.max_new_tokens = self.args.max_new_tokens
        self.generation_cfg.return_dict_in_generate = True
        
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
        self.generation_cfg.num_return_sequences = self.args.k
        
    def _configure_beam_search(self, early_stopping=False):
        """Configure for beam search generation."""
        self.generation_cfg.do_sample = False
        self.generation_cfg.num_beams = self.args.k
        self.generation_cfg.num_return_sequences = self.args.k
        self.generation_cfg.early_stopping = early_stopping
        
    def _configure_group_beam_search(self, early_stopping=False):
        """Configure for group beam search generation."""
        self.generation_cfg.do_sample = False
        self.generation_cfg.num_beams = self.args.k
        self.generation_cfg.num_return_sequences = self.args.k
        self.generation_cfg.num_beam_groups = self.args.k
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
            return self.tokenizer.apply_chat_template(
                chat_query, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            return query
    
    @torch.inference_mode()
    def generate_sentence(self, llm_input: str, **kwargs) -> Union[str, List[str]]:
        """
        Generate text based on the input.
        
        Args:
            llm_input: The input text to generate from
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
            max_length=self.maximun_token - self.args.max_new_tokens
        )
        
        # Move tensors to the same device as the model
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        
        try:
            # Generate text
            res = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=self.generation_cfg,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Process the results
            if len(res.sequences) == 1:
                # Single result
                return self._decode_generated_text(res.sequences[0], input_ids.shape[1])
            else:
                # Multiple results
                return [self._decode_generated_text(seq, input_ids.shape[1]) for seq in res.sequences]
                
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            return None
            
    def _decode_generated_text(self, sequence, input_length):
        """Helper function to decode generated text."""
        return self.tokenizer.decode(
            sequence[input_length:],
            skip_special_tokens=True
        )
