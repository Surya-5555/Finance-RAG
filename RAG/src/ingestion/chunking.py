"""
Document chunking module for the RAG pipeline.
"""

import os
import yaml
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


def load_config() -> dict:
    """
    Load main configuration from config.yaml.
    """
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
    if not config_path.exists():
        return {}
        
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_chunking_config() -> dict:
    """
    Load chunking configuration from the config.yaml file.
    
    Returns:
        dict: A dictionary containing chunking configuration parameters.
    """
    config = load_config()
    return config.get("chunking", {"chunk_size": 800, "chunk_overlap": 100})


def chunk_documents(documents: list[Document]) -> list[dict]:
    """
    Split a list of LangChain Document objects into smaller chunks based on token-based configuration,
    and convert them into standard dictionaries with metadata.
    
    Args:
        documents (list[Document]): A list of LangChain Document objects.
        
    Returns:
        list[dict]: A list of chunked dictionaries containing 'text' and 'metadata'.
    """
    if not documents:
        return []

    config = load_config()
    embedding_model = config.get("embedding_model", "BAAI/bge-small-en-v1.5")
    
    # Load tokenizer corresponding to configured embedding model
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)

    chunk_config = load_chunking_config()
    chunk_size = chunk_config.get("chunk_size", 800)
    chunk_overlap = chunk_config.get("chunk_overlap", 100)

    # Initialize tokenizer-aware splitter
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    result = []
    file_chunk_counters = {}
    
    for doc in documents:
        # Split individual page/document
        chunks = text_splitter.split_documents([doc])
        
        for chunk in chunks:
            meta = chunk.metadata.copy()
            source = meta.get("source", "unknown")
            filename = os.path.basename(source) if source != "unknown" else "unknown"
            
            # Extract PDF page if present
            page = meta.get("page")
            if page is not None:
                try:
                    page = int(page)
                except ValueError:
                    page = None
            
            title = meta.get("title")
            
            # Determine count key for sequential chunk indexing
            if page is not None:
                counter_key = f"{source}_page_{page}"
            else:
                counter_key = source
                
            file_chunk_counters[counter_key] = file_chunk_counters.get(counter_key, 0) + 1
            idx = file_chunk_counters[counter_key]
            
            if page is not None:
                chunk_id = f"{filename}_page_{page}_chunk_{idx}"
            else:
                chunk_id = f"{filename}_chunk_{idx}"
                
            result.append({
                "text": chunk.page_content,
                "metadata": {
                    "source": source,
                    "filename": filename,
                    "page": page,
                    "title": title if title else None,
                    "chunk_id": chunk_id
                }
            })
            
    return result
