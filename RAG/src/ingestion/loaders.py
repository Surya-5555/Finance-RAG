import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, BSHTMLLoader

def load_documents(path: str) -> list[Document]:
    """
    Load documents from a file based on its extension.
    
    Args:
        path (str): The path to the file to load.
        
    Returns:
        list[Document]: A list of LangChain Document objects.
        
    Raises:
        ValueError: If the file extension is not supported or file is missing.
    """
    if not os.path.exists(path):
        raise ValueError(f"File not found: {path}")

    _, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext in [".md", ".txt"]:
        loader = TextLoader(path)
    elif ext in [".html", ".htm"]:
        loader = BSHTMLLoader(path)
    else:
        raise ValueError(f"Unsupported file type '{ext}' for path: {path}")

    return loader.load() #convert into a langchain document with metadata
