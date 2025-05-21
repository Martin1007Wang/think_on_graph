# base_language_model.py
import argparse
from typing import Any, List, Union, Dict, Optional, Iterable, Generator # Added Iterable, Generator

class ModelNotReadyError(Exception):
    """Custom exception for when the model is not ready for an operation."""
    pass

class TokenizationError(Exception):
    """Custom exception for errors during tokenization."""
    pass

class GenerationError(Exception):
    """Custom exception for errors during text generation."""
    pass

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class BaseLanguageModel(object):
    """
    Abstract base class for language models.
    Provides a common interface for loading models, tokenizing text,
    and generating sentences.
    """

    def __init__(self, args: Any): # args can be argparse.Namespace or a similar config object
        """
        Initializes the BaseLanguageModel.

        Args:
            args: Configuration arguments, typically from argparse.
        """
        self.args = args
        self._is_ready: bool = False # Flag to indicate if the model is prepared

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """
        Adds model-specific arguments to an ArgumentParser.

        Args:
            parser: The ArgumentParser to add arguments to.
        """
        # Example:
        # parser.add_argument("--model_specific_arg", type=str, default="default_value")
        return

    def prepare_for_inference(self, model_path: Optional[str] = None, **model_kwargs: Any) -> None:
        """
        Prepares the model for inference.
        This typically involves loading the model weights, tokenizer, and any
        other necessary configurations. Sets the _is_ready flag upon success.

        Args:
            model_path: Optional path to the model. If None, implementation should define behavior.
            **model_kwargs: Additional keyword arguments for model loading and preparation.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
            ConfigurationError: If there's an issue with configuration.
            FileNotFoundError: If model files are not found.
            Exception: For other model loading failures.
        """
        raise NotImplementedError

    def token_len(self, text: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Calculates the number of tokens in a string or list of strings.

        Args:
            text: A single string or a list of strings.

        Returns:
            The number of tokens as an integer, or a list of integers.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
            ModelNotReadyError: If the tokenizer is not initialized.
        """
        raise NotImplementedError

    def prepare_model_prompt(self, query: Union[str, List[str]], **kwargs: Any) -> Union[str, List[str]]:
        """
        Prepares the input query (or queries) into the format expected by the model,
        e.g., applying chat templates.

        Args:
            query: A single query string or a list of query strings.
            **kwargs: Additional arguments for prompt preparation.


        Returns:
            The prepared prompt string or list of prepared prompt strings.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
            ModelNotReadyError: If the tokenizer is not initialized.
        """
        raise NotImplementedError

    def generate_sentence(
        self,
        llm_input: Union[str, List[str]],
        **generation_kwargs: Any
    ) -> Union[str, List[str], Generator[Union[str, List[str]], None, None]]:
        """
        Generates a sentence (or sentences) based on the input.
        This method should handle both single and batch inputs.
        It can also be implemented to support streaming.

        Args:
            llm_input: The input string or list of strings for the language model.
            **generation_kwargs: Keyword arguments to control the generation process
                                (e.g., max_new_tokens, temperature, top_p, do_sample,
                                 num_beams, early_stopping, streamer).

        Returns:
            A generated string, a list of generated strings, or a generator for streaming.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
            ModelNotReadyError: If the model is not ready for inference.
            TokenizationError: If input tokenization fails.
            GenerationError: If the generation process itself fails.
        """
        raise NotImplementedError

    def unload_resources(self) -> None:
        """
        Unloads model and tokenizer resources to free up memory.
        Sets the _is_ready flag to False.
        """
        # Default implementation can be empty if not all models need explicit unloading beyond __del__
        self._is_ready = False
        # Subclasses should implement specific unloading logic (e.g., del self.model, self.tokenizer, torch.cuda.empty_cache())
        pass

    @property
    def is_ready(self) -> bool:
        """Returns True if the model is prepared for inference, False otherwise."""
        return self._is_ready

    def __enter__(self):
        """
        Context manager entry point.
        Ensures the model is prepared for inference.
        You might want to pass model_path or other args from self.args here if needed.
        """
        if not self.is_ready:
            # Assuming prepare_for_inference can get model_path from self.args if not provided
            self.prepare_for_inference()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Context manager exit point.
        Ensures resources are unloaded.
        """
        self.unload_resources()

