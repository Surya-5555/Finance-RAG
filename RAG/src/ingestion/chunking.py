import os
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_metadata_from_filename(filename: str) -> dict:
    """
    Extract company, quarter, and year from filename.
    """
    meta = {}
    
    # Extract quarter
    q_match = re.search(r'(Q[1-4])', filename, re.IGNORECASE)
    if q_match:
        meta['quarter'] = q_match.group(1).upper()
        
    # Extract year
    y_match = re.search(r'(20\d{2}|FY\d{2})', filename, re.IGNORECASE)
    if y_match:
        meta['year'] = y_match.group(1).upper()
        
    # Extract company
    parts = re.split(r'[-_ ]', os.path.splitext(filename)[0])
    for part in parts:
        if part.upper() not in ['Q1', 'Q2', 'Q3', 'Q4'] and not re.match(r'(20\d{2}|FY\d{2})', part.upper()):
            if part.lower() not in ['results', 'press', 'release', 'earnings', 'qtr', 'financial']:
                meta['company'] = part
                break
                
    return meta

def chunk_documents(documents: list[Document]) -> list[dict]:
    """
    Split a list of LangChain Document objects into smaller chunks using
    RecursiveCharacterTextSplitter, adding finance-specific metadata.
    """
    if not documents:
        return []

    chunk_size = 1200
    chunk_overlap = 200

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    result = []
    
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        
        for idx, chunk in enumerate(chunks):
            meta = chunk.metadata.copy()
            source = meta.get("source", "unknown")
            filename = meta.get("filename", os.path.basename(source))
            
            # Extract additional finance metadata
            finance_meta = extract_metadata_from_filename(filename)
            meta.update(finance_meta)
            
            # Form chunk_id
            page = meta.get("page", "unknown")
            is_table = meta.get("is_table", False)
            
            chunk_type = "table" if is_table else "narrative"
            chunk_id = f"{filename}_p{page}_{chunk_type}_{idx}"
            
            meta["chunk_id"] = chunk_id
            
            result.append({
                "text": chunk.page_content,
                "metadata": meta
            })
            
    return result
