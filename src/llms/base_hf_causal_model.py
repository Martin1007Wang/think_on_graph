from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from .base_language_model import BaseLanguageModel
import os
import dotenv
from peft import PeftConfig
from typing import List, Optional, Union, Dict, Any
import logging
from transformers import PreTrainedTokenizer, PreTrainedModel

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
        group = parser.add_argument_group("HfCausalModel")
        group.add_argument("--maximum_token", type=int, default=8000, help="Max input token length")
        group.add_argument("--max_new_tokens", type=int, default=1024, help="Max generation length")
        group.add_argument("--dtype", choices=list(HfCausalModel.DTYPE_MAPPING.keys()), default="bf16")
        group.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
        group.add_argument("--attn_implementation", default="flash_attention_2",
                          choices=["eager", "sdpa", "flash_attention_2"])
        group.add_argument("--generation_mode", type=str, default="greedy",
                          choices=["greedy", "beam", "sampling", "group-beam", 
                                   "beam-early-stopping", "group-beam-early-stopping"])
        group.add_argument("--generation_k", type=int, default=1)
        group.add_argument("--chat_model", default='true', 
                          type=lambda x: (str(x).lower() == 'true'))
        group.add_argument("--use_assistant_model", default='false', 
                          type=lambda x: (str(x).lower() == 'true'))
        group.add_argument("--assistant_model_path", type=str, default=None)
        group.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))

    def __init__(self, args):
        self.args = args
        self.maximum_token = args.maximum_token
        self.tokenizer = None
        self.model = None
        self.assistant_model = None
        self.generation_cfg = None

    def token_len(self, text: str) -> int:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call prepare_for_inference first.")
        result = self.tokenizer(text, add_special_tokens=False, return_length=True)
        if isinstance(result["length"], list):
            return result["length"][0]
        return result["length"]

    def prepare_for_inference(self):
        print(self.args.model_path)
        if not hasattr(self,"tokenizer") or self.tokenizer is None:
            self.tokenizer = self._load_tokenizer(self.args.model_path)
        if not hasattr(self,"model") or self.model is None:
            self.model = self._load_model(self.args.model_path)
        if self.args.use_assistant_model and self.args.assistant_model_path:
            if not hasattr(self,"assistant_model") or self.assistant_model is None:
                self.assistant_model = self._load_model(self.args.assistant_model_path)
        else:
            self.assistant_model = None
        self.maximum_token = self.tokenizer.model_max_length
        self._setup_generation_config()
        
    def _load_tokenizer(self, model_path: str) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=self.args.hf_token, trust_remote_code=True)
        return tokenizer
    
    def _load_model(self, model_path: str) -> PreTrainedModel:
        num_gpus = torch.cuda.device_count()
        attn_impl = self.args.attn_implementation
        logger.info(f"Using attn_implementation='{attn_impl}' on {num_gpus} GPU(s)")
        kwargs = {
            "device_map": "auto",
            "token": self.args.hf_token,
            "torch_dtype": self.DTYPE_MAPPING.get(self.args.dtype),
            "load_in_8bit": self.args.quant == "8bit",
            "load_in_4bit": self.args.quant == "4bit",
            "trust_remote_code": True,
            "attn_implementation": attn_impl,
        }
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        return model
    
    def _load_generation_config(self) -> GenerationConfig:
        try:
            return GenerationConfig.from_pretrained(self.args.model_path)
        except Exception as e:
            logger.warning(f"Failed to load generation config from model path: {e}")
            try:
                peft_config = PeftConfig.from_pretrained(self.args.model_path)
                return GenerationConfig.from_pretrained(peft_config.base_model_name_or_path)
            except Exception as e2:
                logger.warning(f"Failed to load generation config from peft model: {e2}")
                return GenerationConfig()
                
    def _setup_generation_config(self):
        self.generation_cfg = self._load_generation_config()
        self.generation_cfg.max_new_tokens = self.args.max_new_tokens
        self.generation_cfg.return_dict_in_generate = True
        self.args.generation_k = max(1, self.args.generation_k)
        mode_config = {
            "greedy": lambda: self._configure_greedy(),
            "sampling": lambda: self._configure_sampling(),
            "beam": lambda: self._configure_beam(),
            "beam-early-stopping": lambda: self._configure_beam_early_stopping(),
            "group-beam": lambda: self._configure_group_beam(),
            "group-beam-early-stopping": lambda: self._configure_group_beam_early_stopping()
        }
        mode_config.get(self.args.generation_mode, lambda: None)()
            
    def _configure_greedy(self):
        self.generation_cfg.do_sample = False
        self.generation_cfg.num_return_sequences = 1
        
    def _configure_sampling(self):
        self.generation_cfg.do_sample = True
        self.generation_cfg.num_return_sequences = self.args.generation_k
        
    def _configure_beam(self, early_stopping: bool = False):
        self.generation_cfg.do_sample = False
        self.generation_cfg.num_beams = self.args.generation_k
        self.generation_cfg.num_return_sequences = self.args.generation_k
        self.generation_cfg.early_stopping = early_stopping

    def _configure_group_beam(self, early_stopping: bool = False):
        self.generation_cfg.do_sample = False
        self.generation_cfg.num_beams = self.args.generation_k
        self.generation_cfg.num_return_sequences = self.args.generation_k
        self.generation_cfg.num_beam_groups = self.args.generation_k
        self.generation_cfg.diversity_penalty = 1.0
        self.generation_cfg.early_stopping = early_stopping

    def prepare_model_prompt(self, query: str) -> str:
        if self.args.chat_model:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": query}],
                tokenize=False,
                add_generation_prompt=True
            )
        return query
    
    @torch.inference_mode()
    def generate_sentence(self, llm_input: str, temp_generation_mode: Optional[str] = None, **kwargs) -> Union[str, List[str]]:
        if self.token_len(llm_input) > self.maximum_token:
            logger.error(f"Input too long: {self.token_len(llm_input)} tokens, truncating to {self.maximum_token} tokens")
            llm_input = self.tokenizer.decode(self.tokenizer(llm_input, add_special_tokens=False, return_tensors="pt").input_ids[0, -self.maximum_token:],skip_special_tokens=True)
        inputs = self.tokenizer(llm_input, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        actual_input_length = input_ids.shape[1]
        logger.info(f"Input length: {actual_input_length} tokens")
        
        try:
            generation_config = self.generation_cfg
            if temp_generation_mode:
                generation_config = self._get_temp_generation_config(temp_generation_mode)
            if kwargs:
                generation_config = self._update_generation_config(generation_config, kwargs)
            with torch.no_grad():
                res = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config, return_dict_in_generate=True, pad_token_id=self.tokenizer.eos_token_id)
            input_length = input_ids.shape[1]
            if len(res.sequences)==1:
                result = self._decode_generated_text(res.sequences[0],input_length)
            else:
                result = [self._decode_generated_text(seq,input_length) for seq in res.sequences]
            return result
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            raise
        finally:
            # 安全地释放资源
            try:
                # 首先释放引用
                del input_ids, attention_mask, inputs
                if 'res' in locals(): del res
                
                # 调用垃圾回收
                import gc
                gc.collect()
                
                # 尝试清理 CUDA 缓存，但捕获可能的错误
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except RuntimeError as e:
                        logger.warning(f"Failed to empty CUDA cache: {e}")
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
                # 继续执行，不让清理错误影响主要功能

    def _get_temp_generation_config(self, mode: str) -> GenerationConfig:
        config = GenerationConfig(
            max_new_tokens=self.args.max_new_tokens,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        mode_configs = {
            "greedy": {
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1
            },
            "beam": {
                "do_sample": False,
                "num_beams": max(4, self.args.generation_k),
                "num_return_sequences": 1,
                "num_beam_groups": 1
            },
            "sampling": {
                "do_sample": True,
                "num_beams": 1,
                "num_return_sequences": 1,
                "temperature": 0.7,
                "top_p": 0.9
            },
            "group-beam": {
                "do_sample": False,
                "num_beams": max(4, self.args.generation_k * 2),
                "num_return_sequences": 1,
                "num_beam_groups": min(max(2, self.args.generation_k), 5),
                "diversity_penalty": 1.0
            }
        }
        if mode in mode_configs:
            for k, v in mode_configs[mode].items():
                setattr(config, k, v)
        return config

    def _update_generation_config(self, config: GenerationConfig, kwargs: Dict[str, Any]) -> GenerationConfig:
        new_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            return_dict_in_generate=config.return_dict_in_generate,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        for key, value in config.__dict__.items():
            if not key.startswith('_') and value is not None:
                setattr(new_config, key, value)
        for key, value in kwargs.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
            else:
                logger.warning(f"Ignoring unsupported parameter: {key}")
        return new_config

    def _decode_generated_text(self, sequence, input_length: int) -> str:
        return self.tokenizer.decode(sequence[input_length:], skip_special_tokens=True)