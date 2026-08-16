"""
Prompt Manager for loading and caching versioned prompt templates.
"""
import yaml
from pathlib import Path
from typing import Dict

class PromptManager:
    """
    Manages loading and caching of externalized prompt templates.
    """

    def __init__(self) -> None:
        """
        Initialize the PromptManager and cache.
        """
        self._cache: Dict[str, str] = {}
        self.config_path = Path(__file__).resolve().parent.parent.parent / "config"
        self.prompts_dir = self.config_path / "prompts"
        
        # Ensure directory exists
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_version = self._load_default_version()

    def _load_default_version(self) -> str:
        """
        Extract the default prompt version from config.yaml.
        """
        yaml_path = self.config_path / "config.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config.get("prompt", {}).get("version", "rag_answer_v1")
        return "rag_answer_v1"

    def get_prompt(self, version: str = None) -> str:
        """
        Fetch a prompt template string by version name.
        Uses in-memory caching to avoid repeated disk reads.
        
        Args:
            version (str, optional): The prompt version (e.g., 'rag_answer_v1').
                                     Defaults to the version in config.yaml.
                                     
        Returns:
            str: The raw prompt template string.
            
        Raises:
            FileNotFoundError: If the prompt file does not exist.
        """
        target_version = version or self.default_version
        
        if target_version in self._cache:
            return self._cache[target_version]
            
        prompt_file = self.prompts_dir / f"{target_version}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt template missing: {prompt_file}")
            
        with open(prompt_file, "r", encoding="utf-8") as f:
            template = f.read()
            
        self._cache[target_version] = template
        return template
