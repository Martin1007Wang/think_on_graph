import argparse
from abc import ABC, abstractmethod
from typing import Any, List, Union, Dict, Optional, Generator, Tuple

class ModelNotReadyError(Exception):
    pass

class ConfigurationError(Exception):
    pass

class GenerationError(Exception):
    pass

class BaseLanguageModel(ABC):
    def __init__(self) -> None:
        """Initializes base attributes, including counters for statistics."""
        self._is_ready: bool = False
        self.call_counter: int = 0
        self.token_counter: int = 0

    def reset_counters(self) -> None:
        """Resets the call and token counters to zero."""
        self.call_counter = 0
        self.token_counter = 0

    def get_counters(self) -> Tuple[int, int]:
        """Returns the current count of LLM calls and tokens."""
        return self.call_counter, self.token_counter

    @classmethod
    @abstractmethod
    def from_args(cls, args: argparse.Namespace) -> 'BaseLanguageModel':
        raise NotImplementedError

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        pass

    @abstractmethod
    def prepare_for_inference(self, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate_sentence(
        self,
        llm_input: Union[str, List[str]],
        **generation_kwargs: Any
    ) -> Union[str, List[str], Generator[Union[str, List[str]], None, None], None]:
        raise NotImplementedError

    def prepare_model_prompt(self, query: str) -> str:
        return query
    
    def token_len(self, text: str) -> int:
        return len(text.split())

    def unload_resources(self) -> None:
        self._is_ready = False
        pass

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def __enter__(self) -> 'BaseLanguageModel':
        if not self.is_ready:
            self.prepare_for_inference()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.unload_resources()