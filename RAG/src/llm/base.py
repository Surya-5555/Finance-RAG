"""
Abstract base class for LLM providers.
"""
from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """
    Abstract interface for generating text responses from an LLM.
    Provides a unified contract for different LLM backends (e.g., Ollama, OpenAI).
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a text response given a prompt.

        Args:
            prompt (str): The input text prompt to send to the LLM.

        Returns:
            str: The generated text response.
        """
        pass
