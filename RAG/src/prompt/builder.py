"""
Prompt Builder for constructing RAG prompts from retrieved context.
"""
import time
import logging
import os
from typing import Any, Dict, List
from src.prompt.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class PromptBuilder:
    """
    Builder for RAG prompts, injecting context into externalized templates.
    """
    
    def __init__(self, prompt_manager: PromptManager = None) -> None:
        """
        Initialize the PromptBuilder.
        
        Args:
            prompt_manager (PromptManager, optional): Manager to load templates. 
                                                      Defaults to a new instance.
        """
        self.prompt_manager = prompt_manager or PromptManager()
        
    def format_context(self, contexts: List[Dict[str, Any]]) -> str:
        """
        Format the retrieved documents into a string with explicit source citations.
        
        Args:
            contexts (List[Dict[str, Any]]): Top-N reranked chunks.
                                             Expected format: {"document": Document(...), ...}
        Returns:
            str: The formatted context string block.
        """
        formatted_docs = []
        for i, item in enumerate(contexts, 1):
            doc = item.get("document")
            if not doc:
                continue
                
            meta = doc.metadata
            text = doc.page_content
            
            # Extract metadata
            source = meta.get("source", "unknown")
            filename = meta.get("filename") or os.path.basename(source)
            page = meta.get("page")
            title = meta.get("title")
            chunk_id = meta.get("chunk_id")
            
            # Format header
            header = f"[Source {i}]\nSource: {source}\nFile: {filename}"
            if page is not None:
                header += f"\nPage: {page}"
            if title:
                header += f"\nTitle: {title}"
            if chunk_id:
                header += f"\nChunk ID: {chunk_id}"
                
            formatted_docs.append(f"{header}\n\n{text}")
            
        return "\n\n".join(formatted_docs)
        
    def build(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """
        Build the final prompt string by injecting context and query into the template.
        
        Args:
            query (str): The user's question.
            contexts (List[Dict[str, Any]]): The retrieved Top-N chunks.
            
        Returns:
            str: The finalized prompt string ready for the LLM.
        """
        start_time = time.time()
        
        # 1. Load the template
        template = self.prompt_manager.get_prompt()
        
        # 2. Format the retrieved context block
        context_str = self.format_context(contexts)
        
        # 3. Inject variables safely
        final_prompt = template.replace("{context}", context_str).replace("{question}", query)
        
        latency = time.time() - start_time
        logger.info(f"Prompt construction completed in {latency:.4f}s using version '{self.prompt_manager.default_version}'. Included {len(contexts)} chunks.")
        
        return final_prompt
