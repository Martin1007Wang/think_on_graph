# -*- coding: utf-8 -*-
import torch
import logging
import os
import gc
from typing import List, Optional, Union, Dict, Any, Tuple
from unsloth import FastLanguageModel
from transformers import GenerationConfig, PreTrainedTokenizer, PreTrainedModel
from .base_language_model import BaseLanguageModel, GenerationError, ModelNotReadyError, ConfigurationError
import torch.nn.functional as F

import dotenv
dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
logger = logging.getLogger(__name__)


class HfCausalModel(BaseLanguageModel):
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
        group = parser.add_argument_group("HfCausalModel Specific Arguments")
        group.add_argument("--max_length", type=int, default=8000, help="Maximum token length allowed by the tokenizer/model context window.")
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
        super().__init__()
        self.args = args
        self._loaded_model_path: Optional[str] = None
        self._is_ready: bool = False  # 修正：添加缺失的 _is_ready 属性初始化
        self.max_length: int = args.max_length
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[PreTrainedModel] = None
        self.generation_cfg: Optional[GenerationConfig] = None
        self.device_map: Union[str, Dict, None] = kwargs.get("device_map", "auto")
        
        # 修正：添加缺失的计数器属性（基类可能需要这些）
        self.call_counter: int = 0
        self.token_counter: int = 0
        
        logger.info(f"HfCausalModel initialized. Target device_map: '{self.device_map}'")

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def prepare_for_inference(self, model_path: Optional[str] = None) -> None:
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
        logger.info(f"Loading model and tokenizer from '{model_path_or_id}' using Unsloth...")
        load_in_4bit = getattr(self.args, 'quant', 'none') == '4bit'
        load_in_8bit = getattr(self.args, 'quant', 'none') == '8bit'
        dtype_str = getattr(self.args, 'dtype', 'bf16')
        torch_dtype = self.DTYPE_MAPPING.get(dtype_str, torch.bfloat16)
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path_or_id,
                max_seq_length=self.max_length,
                dtype=torch_dtype,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                token=getattr(self.args, 'hf_token', HF_TOKEN),
            )
            if tokenizer.pad_token is None or tokenizer.pad_token_id != tokenizer.eos_token_id:
                logger.warning(
                    f"Tokenizer's pad_token is not set or not equal to eos_token. "
                    f"Overriding to eos_token ({tokenizer.eos_token_id}) for consistency with fine-tuning."
                )
                tokenizer.pad_token = tokenizer.eos_token
            # === CRITICAL CONSISTENCY FIX ===
            # 确保推理时也使用左填充，与MPO训练完全一致
            if tokenizer.padding_side != 'left':
                logger.warning(f"Tokenizer's padding_side is '{tokenizer.padding_side}'. Overriding to 'left' for batch scoring/generation.")
                tokenizer.padding_side = 'left'
            logger.info("Unsloth: Model and tokenizer loaded and configured successfully.")
            model.eval()
            return model, tokenizer
        except Exception as e:
            logger.error(f"Unsloth failed to load model from '{model_path_or_id}': {e}", exc_info=True)
            self._log_common_loading_errors(e)
            raise

    @torch.inference_mode()
    def score_candidate_relations_batched(self, prompt: str, candidate_relations: List[str]) -> Dict[str, float]:
        """批量评分候选关系的方法"""
        if not self.is_ready or self.model is None or self.tokenizer is None:
            raise ModelNotReadyError("Model is not ready for scoring.")
        if not candidate_relations:
            return {}
        
        # 修正：使用正确的属性名
        prompt_encoded = self.tokenizer(
            prompt, 
            truncation=True, 
            max_length=self.max_length,  # 修正：使用 self.max_length 而不是 self.max_length
            add_special_tokens=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        relation_scores = self._get_relation_logits(
            self.model,
            prompt_encoded['input_ids'],
            prompt_encoded['attention_mask'],
            [candidate_relations]  # 包装为batch格式
        )
        
        scores_dict = {}
        for i, relation in enumerate(candidate_relations):
            if i < relation_scores.size(1):
                score = torch.sigmoid(relation_scores[0, i]).item()
                scores_dict[relation] = score
            else:
                scores_dict[relation] = 0.0
        
        return scores_dict
    
    def _get_relation_logits(
        self, 
        model, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidates: List[List[str]],
        batch_size: int = 16  # 减小批次大小避免内存问题
    ) -> torch.Tensor:
        """计算关系logits的核心方法"""
        device = input_ids.device
        batch_size_samples = input_ids.size(0)
        max_candidates = max(len(cands) for cands in candidates) if candidates else 0
        
        if max_candidates == 0:
            return torch.full((batch_size_samples, 0), -1e9, device=device, dtype=torch.float)
        
        relation_scores = torch.full(
            (batch_size_samples, max_candidates), 
            -1e9, 
            device=device, 
            dtype=torch.float
        )
        
        for sample_idx, sample_candidates in enumerate(candidates):
            if not sample_candidates:
                continue
            
            context_len = attention_mask[sample_idx].sum().item()
            context_tokens = input_ids[sample_idx, :context_len]
            num_candidates = len(sample_candidates)
            
            for start_idx in range(0, num_candidates, batch_size):
                end_idx = min(start_idx + batch_size, num_candidates)
                batch_candidates = sample_candidates[start_idx:end_idx]
                batch_inputs = []
                batch_metadata = []
                
                for cand_idx, relation_str in enumerate(batch_candidates):
                    relation_tokens = self.tokenizer.encode(
                        relation_str, 
                        add_special_tokens=False
                    )
                    if not relation_tokens:
                        continue
                    
                    full_input = torch.cat([
                        context_tokens,
                        torch.tensor(relation_tokens, device=device, dtype=torch.long)
                    ], dim=0)
                    
                    batch_inputs.append(full_input)
                    batch_metadata.append((start_idx + cand_idx, len(relation_tokens), context_len))
                
                if not batch_inputs:
                    continue
                
                padded_inputs = torch.nn.utils.rnn.pad_sequence(
                    batch_inputs, 
                    batch_first=True, 
                    padding_value=self.tokenizer.pad_token_id
                )
                padded_attention = (padded_inputs != self.tokenizer.pad_token_id).long()
                
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(
                        input_ids=padded_inputs,
                        attention_mask=padded_attention,
                        use_cache=False,
                        return_dict=True
                    )
                    logits = outputs.logits
                
                log_probs = F.log_softmax(logits, dim=-1)
                
                for i, (global_cand_idx, relation_len, ctx_len) in enumerate(batch_metadata):
                    if i >= log_probs.size(0):
                        continue
                    
                    start_pos = ctx_len - 1
                    end_pos = ctx_len + relation_len - 1
                    
                    if end_pos >= padded_inputs.size(1):
                        continue
                    
                    target_logits = log_probs[i, start_pos:end_pos]
                    target_tokens = padded_inputs[i, ctx_len:end_pos + 1]
                    
                    token_scores = target_logits.gather(
                        dim=1, 
                        index=target_tokens.unsqueeze(-1)
                    ).squeeze(-1)
                    
                    avg_score = token_scores.mean()
                    relation_scores[sample_idx, global_cand_idx] = avg_score
        
        return relation_scores
    
    @torch.inference_mode()
    def generate_sentence(self, prompts: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """生成文本的方法"""
        if not self.is_ready or self.model is None or self.tokenizer is None or self.generation_cfg is None:
            raise ModelNotReadyError("Model is not ready for inference.")

        is_single_prompt = isinstance(prompts, str)
        prompts_list = [prompts] if is_single_prompt else prompts
        
        try:
            # 确保分词器是左填充，这对于批量生成至关重要
            original_padding_side = self.tokenizer.padding_side  # 修正：保存原始设置
            if self.tokenizer.padding_side != 'left':
                logger.warning("Temporarily setting padding_side to 'left' for batch generation.")
                self.tokenizer.padding_side = 'left'

            inputs = self.tokenizer(prompts_list, padding=True, return_tensors="pt").to(self.model.device)
            
            final_generation_config = self.generation_cfg.to_dict()
            final_generation_config.update(kwargs)
            
            self.call_counter += 1
            
            outputs = self.model.generate(**inputs, **final_generation_config)
            
            # Token计数逻辑
            if self.tokenizer:
                num_tokens = torch.sum(outputs != self.tokenizer.pad_token_id).item()
                self.token_counter += num_tokens
            
            # --- 正确的批量解码逻辑 ---
            results = []
            for i, prompt_text in enumerate(prompts_list):
                # 获取输入长度
                input_len = len(inputs['input_ids'][i])
                # 只解码新生成的token
                generated_tokens = outputs[i][input_len:]
                result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                results.append(result)

            # 修正：恢复原始的padding_side设置
            self.tokenizer.padding_side = original_padding_side
            
            return results[0] if is_single_prompt else results
            
        except Exception as e:
            logger.error(f"Error during batch generation: {e}", exc_info=True)
            # 修正：确保在异常情况下也恢复padding_side
            if 'original_padding_side' in locals():
                self.tokenizer.padding_side = original_padding_side
            raise GenerationError(f"Generation failed: {e}") from e

    def unload_resources(self) -> None:
        """卸载资源的方法"""
        if not self._is_ready and self._loaded_model_path is None:
            logger.info("No resources to unload.")
            return
        logger.info("Unloading model resources...")
        
        # 修正：安全地删除资源
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if hasattr(self, 'generation_cfg') and self.generation_cfg is not None:
            del self.generation_cfg
            self.generation_cfg = None
            
        self._loaded_model_path = None
        self._is_ready = False
        
        if torch.cuda.is_available():
            logger.info("Attempting garbage collection and CUDA cache clearing after unload.")
            gc.collect()
            torch.cuda.empty_cache()
        logger.info("Model resources unloaded successfully.")

    def _get_model_path_from_args(self) -> Optional[str]:
        """从参数中获取模型路径"""
        path = getattr(self.args, "explore_model_path", None) or \
               getattr(self.args, "predict_model_path", None) or \
               getattr(self.args, "model_path", None)
        return path if path != "None" else None

    def _update_max_token_length(self) -> None:
        """更新最大token长度"""
        if self.tokenizer is None: 
            return
        try:
            tokenizer_max_len = getattr(self.tokenizer, 'model_max_length', self.args.max_length)
            if not isinstance(tokenizer_max_len, int) or tokenizer_max_len > 1e5:
                tokenizer_max_len = self.args.max_length
            self.max_length = min(tokenizer_max_len, self.args.max_length)
            logger.info(f"Effective maximum token length set to: {self.max_length}")
        except Exception:
            self.max_length = self.args.max_length

    def _log_common_loading_errors(self, exception: Exception) -> None:
        """记录常见的加载错误"""
        err_str = str(exception).lower()
        if "out of memory" in err_str:
            logger.error("CUDA out of memory. Try using quantization (--quant 4bit/8bit) or a smaller model.")
        elif "connection error" in err_str or "repository not found" in err_str:
            logger.error(f"Connection error or model not found. Check path/ID, network, and HF token.")

    def _setup_generation_config(self, model_path: str) -> None:
        """设置生成配置"""
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
        self.generation_cfg.max_length = None  # Let max_new_tokens control the length
        
        mode = getattr(self.args, 'generation_mode', 'greedy')
        k = getattr(self.args, 'generation_k', 1)
        logger.info(f"Setting up generation config for mode='{mode}' with k={k}.")
        
        if mode in ['greedy', 'beam', 'group-beam']:
            self.generation_cfg.do_sample = False
            self.generation_cfg.temperature = None
            self.generation_cfg.top_p = None
            self.generation_cfg.top_k = None
            if mode == 'greedy':
                self.generation_cfg.num_beams = 1
            elif mode == 'beam':
                self.generation_cfg.num_beams = k
            elif mode == 'group-beam':
                self.generation_cfg.num_beams = k
                self.generation_cfg.num_beam_groups = k
        elif mode == 'sampling':
            self.generation_cfg.do_sample = True
            self.generation_cfg.num_beams = 1
            self.generation_cfg.top_k = 50 
            self.generation_cfg.top_p = 0.9
            self.generation_cfg.temperature = 0.7
            self.generation_cfg.num_return_sequences = k
        
        self.generation_cfg.use_cache = True
        logger.info(f"Default GenerationConfig prepared with the following settings: {self.generation_cfg.to_dict()}")