"""
Ollama implementation of the LLM provider.
"""
import yaml
import os
from pathlib import Path
from ollama import Client
from dotenv import load_dotenv

from .base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """
    LLM provider using local Ollama service.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """
        Initialize the Ollama provider.

        Args:
            model_name (str | None): Overrides the model name from config if provided.
        """
        load_dotenv()
        
        # Determine model name
        self.model_name = model_name if model_name else self._load_config_value("llm_model", "qwen3:4b")
        
        if not self.model_name or not isinstance(self.model_name, str):
            raise ValueError("A valid LLM model name must be provided in config or as an argument.")

        self.temperature = float(self._load_config_value("temperature", 0.0))
        
        # Initialize client connecting to OLLAMA_BASE_URL (defaults to http://localhost:11434)
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = Client(host=ollama_base_url)

    def _load_config_value(self, key: str, default: str | float) -> str | float:
        """
        Load a configuration value from the config.yaml file.
        
        Args:
            key (str): The configuration key to retrieve.
            default (str | float): The default value if not found or if config is missing.
            
        Returns:
            str | float: The loaded configuration value.
        """
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
        if not config_path.exists():
            return default
            
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            
        return config.get(key, default)

    def generate(self, prompt: str) -> str:
        """
        Generate a text response given a prompt using Ollama.

        Args:
            prompt (str): The input text prompt to send to the LLM.

        Returns:
            str: The generated text response.
            
        Raises:
            ValueError: If the prompt is empty or not a string.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "temperature": self.temperature
            }
        )
        
        return response.get("response", "")

    def close(self) -> None:
        """
        Close the underlying Ollama client.
        """
        try:
            self.client.close()
        except Exception:
            pass

