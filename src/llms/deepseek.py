import time
import os
from openai import OpenAI
from .base_language_model import BaseLanguageModel
import dotenv
import tiktoken
from typing import Optional
import logging
import uuid

dotenv.load_dotenv()

os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'
logger = logging.getLogger(__name__)

def get_token_limit(model='deepseek-chat'):
    num_tokens_limit = 32000
    return num_tokens_limit

class SiliconFlowLLM(BaseLanguageModel):
    # 模型名称映射
    MODELS = {
        'deepseek-chat': 'deepseek-ai/DeepSeek-V3'
    }

    @staticmethod
    def add_args(parser):
        if not any(action.dest == 'retry' for action in parser._actions):
           parser.add_argument('--retry', type=int, help="retry time", default=5)
        if not any(action.dest == 'model_path' for action in parser._actions):
           parser.add_argument('--model_path', type=str, default='None')
        if not any(action.dest == 'deepseek_api_key' for action in parser._actions):
           parser.add_argument('--deepseek_api_key', type=str, help="DeepSeek API Key", default=None)
           
    def __init__(self, args):
        super().__init__(args)
        self.retry = args.retry
        self.model_name = self.MODELS.get(args.predict_model_name, 'deepseek-chat')
        self.maximun_token = get_token_limit(self.model_name)
        self.system_prompt = "You are a knowledge graph reasoning expert. Your task is to provide clear and precise answers based on the information provided."
        # 从命令行参数获取API密钥，如果没有则从环境变量获取
        # self.api_key = args.deepseek_api_key if hasattr(args, 'deepseek_api_key') and args.deepseek_api_key else os.environ.get('DEEPSEEK_API_KEY')
        self.api_key = os.environ.get('SILICONFLOW_API_KEY')
        
        if not self.api_key:
            logger.warning("DeepSeek API Key not found in arguments or environment variables. API calls will likely fail.")
        
    def token_len(self, text):
        """Returns the number of tokens used by a list of messages."""
        try:
            # 使用 cl100k_base 作为默认 tokenizer
            encoding = tiktoken.get_encoding("cl100k_base")
            num_tokens = len(encoding.encode(text))
        except KeyError:
            # 如果获取 tokenizer 失败，使用一个简单的估算
            num_tokens = len(text.split()) * 1.3  # 粗略估算
        return num_tokens
    
    def prepare_for_inference(self, model_kwargs={}):
        try:
            # 直接使用DeepSeek官方示例中的方式创建客户端
            client = OpenAI(
                api_key=self.api_key,
                # base_url="https://api.deepseek.com"
                base_url="https://api.siliconflow.cn/v1"
            )
            self.client = client
            logger.info(f"Successfully initialized DeepSeek client with API key: {self.api_key[:4]}...{self.api_key[-4:] if self.api_key and len(self.api_key) > 8 else '****'}")
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek client: {str(e)}")
            raise
    
    def prepare_model_prompt(self, query):
        '''
        Add model-specific prompt to the input
        '''
        return query
    
    def generate_sentence(self, llm_input, temp_generation_mode: Optional[str] = None, **kwargs):
        # 记录一下Trace ID，方便追踪和排查问题
        trace_id = str(uuid.uuid4())[:8]
        logger.info(f"[{trace_id}] Starting DeepSeek API call")
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": llm_input}
        ]
        
        cur_retry = 0
        num_retry = self.retry
        
        # 检查输入是否过长
        input_length = self.token_len(llm_input)
        logger.info(f"[{trace_id}] Input length: {input_length} tokens (Max: {self.maximun_token})")
        
        # 检查是否包含"history_size_exceeded"关键词，这可能是一个困难的案例
        if "history_size_exceeded" in llm_input or "exploration_history_size" in llm_input:
            logger.warning(f"[{trace_id}] Detected potential difficult case with large history size")
        
        if input_length > self.maximun_token:
            logger.warning(f"[{trace_id}] Input length {input_length} is too long. The maximum token is {self.maximun_token}. Truncating input.")
            # 截断输入
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(llm_input)
            truncated_tokens = tokens[:self.maximun_token]
            llm_input = encoding.decode(truncated_tokens)
            messages[1]["content"] = llm_input
            logger.info(f"[{trace_id}] Input truncated to {len(truncated_tokens)} tokens")
        
        temperature = 0.0
        if temp_generation_mode == "beam":
            # 对于beam search保持temperature为0
            temperature = 0.0
        elif temp_generation_mode == "sample":
            # 对于采样，设置适当的temperature
            temperature = 0.7
        
        while cur_retry <= num_retry:
            try:
                logger.info(f"[{trace_id}] Sending request to DeepSeek API (attempt {cur_retry+1}/{num_retry+1})")
                # 参考DeepSeek官方示例格式进行API调用
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=kwargs.get("max_tokens", 2048),
                    stream=False
                )
                
                result = response.choices[0].message.content.strip()
                logger.info(f"[{trace_id}] Successfully generated response with {self.model_name} (length: {len(result)})")
                return result
            except Exception as e:
                logger.error(f"[{trace_id}] Error while generating with {self.model_name}:")
                logger.error(f"[{trace_id}] Input length: {input_length} tokens")
                logger.error(f"[{trace_id}] Exception: {e}")
                logger.error(f"[{trace_id}] Retry {cur_retry}/{num_retry}")
                time.sleep(30)
                cur_retry += 1
                continue
        
        logger.error(f"[{trace_id}] Failed to generate response after {num_retry} retries")
        return None
