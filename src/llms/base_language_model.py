class BaseLanguageModel(object):
    @staticmethod
    def add_args(parser):
        return

    def __init__(self, args):
        self.args = args

    def load_model(self, **kwargs):
        raise NotImplementedError

    def token_len(self, text):
        raise NotImplementedError
    
    def prepare_for_inference(self, **model_kwargs):
        raise NotImplementedError
    
    def prepare_model_prompt(self, query):
        raise NotImplementedError
    
    def generate_sentence(self, llm_input):
        raise NotImplementedError