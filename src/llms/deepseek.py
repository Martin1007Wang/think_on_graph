import logging
import os
import time
import uuid
import argparse
from typing import List, Optional, Union, Any, Generator
from dataclasses import dataclass, field

import tiktoken
from openai import OpenAI

from .base_language_model import BaseLanguageModel, ModelNotReadyError, ConfigurationError, GenerationError

logger = logging.getLogger(__name__)

@dataclass
class ApiModelConfig:
    # ... (ApiModelConfig content remains unchanged)
    model_name: str = "deepseek-ai/DeepSeek-V3"
    base_url: str = "https://api.siliconflow.cn/v1"
    api_key_env_var: str = "SILICONFLOW_API_KEY"
    retry: int = 500
    timeout: int = 60
    system_prompt: str = "You are a knowledge graph reasoning expert..."
    max_context_tokens: int = 32000
    max_new_tokens: int = 4096
    temperature: float = 1.0

class SiliconFlowLLM(BaseLanguageModel):
    ENCODING_MODEL = "cl100k_base"

    def __init__(self, config: ApiModelConfig):
        # --- MODIFIED: Call super().__init__() to set up counters ---
        super().__init__()
        self.config = config
        self.client: Optional[OpenAI] = None
        # _is_ready is now inherited
        self.api_key = os.getenv(self.config.api_key_env_var)
        if not self.api_key:
            raise ConfigurationError(f"API key not found in env var: {self.config.api_key_env_var}")
        logger.info(f"SiliconFlowLLM instance created for model: '{config.model_name}'")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'SiliconFlowLLM':
        # ... (from_args content remains unchanged)
        config_fields = {f.name for f in ApiModelConfig.__dataclass_fields__.values()}
        config_kwargs = {}
        for key, value in vars(args).items():
            if key in config_fields and value is not None:
                config_kwargs[key] = value

        if hasattr(args, 'api_model_name') and args.api_model_name:
            config_kwargs['model_name'] = args.api_model_name

        config = ApiModelConfig(**config_kwargs)
        return cls(config)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        # ... (add_args content remains unchanged)
        group = parser.add_argument_group("API Model Specific Arguments")
        group.add_argument('--retry', type=int, help="Number of retries for API calls.")
        group.add_argument('--api_model_name', type=str, help="Name of the API model to use.")
        group.add_argument('--api_key_env_var', type=str, help="Environment variable for the API key.")
        group.add_argument('--base_url', type=str, help="API base URL.")

    def prepare_for_inference(self, **kwargs: Any) -> None:
        if self._is_ready: return
        logger.info("Initializing SiliconFlow client...")
        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.config.base_url, timeout=self.config.timeout)
            self._is_ready = True
            logger.info("SiliconFlow client initialized successfully.")
        except Exception as e:
            self._is_ready = False
            raise ConfigurationError(f"Failed to initialize API client: {e}") from e

    def unload_resources(self) -> None:
        if not self._is_ready: return
        self.client = None
        self._is_ready = False
        logger.info("SiliconFlow client resources cleared.")

    def token_len(self, text: str) -> int:
        try:
            encoding = tiktoken.get_encoding(self.ENCODING_MODEL)
            return len(encoding.encode(text, disallowed_special=()))
        except Exception:
            return int(len(str(text).split()) * 1.5)

    def prepare_model_prompt(self, query: str) -> str:
        return query

    def generate_sentence(
        self, 
        llm_input: Union[str, List[str]], 
        **generation_kwargs: Any
    ) -> Union[Optional[str], List[Optional[str]]]:
        if isinstance(llm_input, str):
            return self._generate_single(llm_input, **generation_kwargs)
        elif isinstance(llm_input, list):
            results = []
            for prompt in llm_input:
                results.append(self._generate_single(prompt, **generation_kwargs))
            return results
        else:
            raise TypeError(f"Unsupported input type for llm_input: {type(llm_input)}")

    def _generate_single(self, prompt: str, **generation_kwargs: Any) -> Optional[str]:
        if not self.is_ready or not self.client:
            raise ModelNotReadyError("SiliconFlow client is not initialized.")

        trace_id = uuid.uuid4().hex[:8]
        messages = [{"role": "system", "content": self.config.system_prompt}, {"role": "user", "content": prompt}]
        
        # ... (Truncation logic remains unchanged)
        total_tokens = sum(self.token_len(msg['content']) for msg in messages)
        if total_tokens > self.config.max_context_tokens:
            original_tokens = total_tokens
            buffer_tokens = self.config.max_new_tokens
            system_prompt_tokens = self.token_len(self.config.system_prompt)
            overhead_tokens = system_prompt_tokens + buffer_tokens
            max_user_content_tokens = self.config.max_context_tokens - overhead_tokens
            
            user_content = prompt 
            user_content_tokens = self.token_len(user_content)

            if user_content_tokens > max_user_content_tokens:
                lines = user_content.split('\n')
                while self.token_len('\n'.join(lines)) > max_user_content_tokens and len(lines) > 1:
                    lines.pop()
                user_content = '\n'.join(lines)
            
            messages = [{"role": "system", "content": self.config.system_prompt}, {"role": "user", "content": user_content}]
            final_tokens = sum(self.token_len(msg['content']) for msg in messages)
            logger.warning(
                f"[{trace_id}] Context length exceeded. Truncated user content from "
                f"{original_tokens} to {final_tokens} total tokens to fit within the "
                f"{self.config.max_context_tokens} limit."
            )
        
        api_params = {
            "model": self.config.model_name, "messages": messages,
            "temperature": self.config.temperature, "max_tokens": self.config.max_new_tokens
        }
        
        for attempt in range(self.config.retry):
            try:
                # --- MODIFIED: Increment call counter and add token counting ---
                self.call_counter += 1
                response = self.client.chat.completions.create(**api_params)
                
                if response.usage and response.usage.total_tokens:
                    self.token_counter += response.usage.total_tokens

                result = response.choices[0].message.content
                if result: return result.strip()

            except Exception as e:
                logger.error(f"[{trace_id}] API call failed (attempt {attempt + 1}): {e}")
                if attempt >= self.config.retry - 1:
                    raise GenerationError(f"API call failed after retries: {e}") from e
                backoff_time = 2 ** attempt
                time.sleep(backoff_time)
        return None