
import json
import logging
import string
from functools import lru_cache, wraps

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


def read_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        prompt_template = f"""{f.read()}"""
    return prompt_template

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_multiple_jsonl(file_path_list):
    data = []
    for path in file_path_list:
        data.extend(load_jsonl(path))
    return data

def list_to_string(l: list) -> str:
    prompt = '"{}"'
    return ', '.join([prompt.format(i) for i in l])

def rule_to_string(rule: list, sep_token = "<SEP>", bop = "<PATH>", eop = "</PATH>") -> str:
    if len(rule) == 1:
        rule_string = rule[0]
    else:
        rule_string = sep_token.join(rule)
    return bop + rule_string + eop

class InstructFormater(object):
    def __init__(self, prompt_path):
        '''
        _summary_

        Args:
            prompt_template (_type_): 
            instruct_template (_type_): _description_
        '''
        self.prompt_template = read_prompt(prompt_path)

    def format(self, instruction, message):
        return self.prompt_template.format(instruction=instruction, input=message)
    
def error_handler(default_return=None, error_message="Error in operation"):
    """集中式错误处理装饰器，减少重复的try-except块"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{error_message}: {e}", exc_info=True)
                return default_return
        return wrapper
    return decorator