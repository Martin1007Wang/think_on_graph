import time
import os
from openai import OpenAI
from .base_language_model import BaseLanguageModel
import dotenv
import tiktoken
dotenv.load_dotenv()

os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'

def get_token_limit(model='gpt-4'):
    """Returns the token limitation of provided model"""
    
    if model in ['deepseek-ai/DeepSeek-V3','deepseek-ai/DeepSeek-R1']:
        num_tokens_limit = 64000
    elif model in ['deepseek-ai/DeepSeek-R1-Distill-Llama-70B','deepseek-ai/DeepSeek-R1-Distill-Qwen-32B','deepseek-ai/DeepSeek-R1-Distill-Qwen-14B','deepseek-ai/DeepSeek-R1-Distill-Llama-8B','deepseek-ai/DeepSeek-R1-Distill-Qwen-7B','deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B']:
        num_tokens_limit = 32000
    else:
        raise NotImplementedError(f"""get_token_limit() is not implemented for model {model}.""")
    return num_tokens_limit

PROMPT = """{instruction}

{input}"""

class SiliconFlowLLM(BaseLanguageModel):
    # 模型名称映射
    MODELS = {
        'deepseek-v3': 'deepseek-ai/DeepSeek-V3',
        'deepseek-r1': 'deepseek-ai/DeepSeek-R1',
        'deepseek-r1-70b': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        'deepseek-r1-32b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        'deepseek-r1-14b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
        'deepseek-r1-8b': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        'deepseek-r1-7b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'deepseek-r1-1.5b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    }

    @staticmethod
    def add_args(parser):
        parser.add_argument('--retry', type=int, help="retry time", default=5)
        parser.add_argument('--model_path', type=str, default='None')
        parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
        parser.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
           
    def __init__(self, args):
        super().__init__(args)
        self.retry = args.retry
        self.model_name = self.MODELS[args.model_name]
        self.maximun_token = get_token_limit(self.model_name)
        
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
        client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
        base_url="https://api.siliconflow.cn/v1"
        )
        self.client = client
    
    def prepare_model_prompt(self, query):
        '''
        Add model-specific prompt to the input
        '''
        return query
    
    def generate_sentence(self, llm_input):
        query = [{"role": "user", "content": llm_input}]
        cur_retry = 0
        num_retry = self.retry
        # Chekc if the input is too long
        input_length = self.token_len(llm_input)
        if input_length > self.maximun_token:
            print(f"Input lengt {input_length} is too long. The maximum token is {self.maximun_token}.\n Right tuncate the input to {self.maximun_token} tokens.")
            llm_input = llm_input[:self.maximun_token]
        while cur_retry <= num_retry:
            try:
                response = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = query,
                    timeout=60,
                    temperature=0.0
                    )
                result = response.choices[0].message.content.strip() # type: ignore
                return result
            except Exception as e:
                print("Message: ", llm_input)
                print("Number of token: ", self.token_len(llm_input))
                print(e)
                time.sleep(30)
                cur_retry += 1
                continue
        return None
