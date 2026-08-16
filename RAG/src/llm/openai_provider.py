import os
from langchain_openai import ChatOpenAI

class OpenAILLMProvider:
    """
    OpenAI LLM provider using GPT-4o for generation.
    """
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.1):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is missing.")
            
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key
        )
        
    def generate(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.
        """
        response = self.llm.invoke(prompt)
        return response.content
