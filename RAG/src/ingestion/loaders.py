import os
import pdfplumber
from langchain_core.documents import Document

def load_documents(path: str) -> list[Document]:
    """
    Load documents from a PDF file using pdfplumber to extract text and tables separately.
    Tables are converted to markdown format and yielded as separate documents.
    
    Args:
        path (str): The path to the PDF file to load.
        
    Returns:
        list[Document]: A list of LangChain Document objects (multiple per page).
        
    Raises:
        ValueError: If the file is not found or unsupported.
    """
    if not os.path.exists(path):
        raise ValueError(f"File not found: {path}")

    _, ext = os.path.splitext(path)
    if ext.lower() != ".pdf":
        raise ValueError(f"Unsupported file type '{ext}'. Finance RAG only supports PDF.")

    docs = []
    
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                docs.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": path, 
                            "page": i + 1, 
                            "filename": os.path.basename(path),
                            "is_table": False
                        }
                    )
                )
            
            # Extract tables
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue
                md_table = []
                for row_idx, row in enumerate(table):
                    clean_row = [str(cell).replace('\n', ' ') if cell else "" for cell in row]
                    md_table.append("| " + " | ".join(clean_row) + " |")
                    # Add separator after header
                    if row_idx == 0:
                        md_table.append("|" + "|".join(["---"] * len(clean_row)) + "|")
                
                table_text = "\n".join(md_table)
                docs.append(
                    Document(
                        page_content=table_text,
                        metadata={
                            "source": path, 
                            "page": i + 1, 
                            "filename": os.path.basename(path),
                            "is_table": True
                        }
                    )
                )
            
    return docs
