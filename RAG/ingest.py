"""
Unified Ingestion Pipeline entry point.
"""
import os
import argparse
from pathlib import Path
from src.ingestion.loaders import load_documents
from src.ingestion.chunking import chunk_documents
from src.embeddings.huggingface import HuggingFaceEmbeddingProvider
from src.vectorstore.weaviate_store import WeaviateVectorStore

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Weaviate RAG collection.")
    parser.add_argument("--dir", type=str, default="data", help="Directory to scan recursively.")
    parser.add_argument("--recreate", action="store_true", help="Force recreate collection before ingestion.")
    args = parser.parse_args()
    
    source_dir = Path(args.dir).resolve()
    if not source_dir.exists():
        print(f"Error: Directory '{source_dir}' does not exist.")
        return
        
    print(f"Initializing embedding provider and Weaviate store...")
    embed_provider = HuggingFaceEmbeddingProvider()
    store = WeaviateVectorStore()
    
    try:
        if args.recreate:
            print(f"Recreating collection '{store.collection_name}'...")
            store.delete_collection()
            
        store.create_collection()
        
        # Scan files recursively
        supported_extensions = {".pdf", ".md", ".txt", ".html", ".htm"}
        files_to_process = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() in supported_extensions:
                    files_to_process.append(os.path.join(root, file))
                    
        total_files = len(files_to_process)
        print(f"Found {total_files} supported documents in '{source_dir}'.")
        
        ingested_count = 0
        skipped_count = 0
        error_count = 0
        
        for idx, file_path in enumerate(files_to_process, 1):
            normalized_path = os.path.abspath(file_path)
            
            # Check for duplicate ingestion
            if store.is_file_ingested(normalized_path):
                print(f"[{idx}/{total_files}] Skipping (already ingested): {os.path.basename(file_path)}")
                skipped_count += 1
                continue
                
            print(f"[{idx}/{total_files}] Processing: {os.path.basename(file_path)}")
            try:
                # 1. Load documents
                documents = load_documents(normalized_path)
                
                # 2. Chunk documents (token-based)
                chunks = chunk_documents(documents)
                
                if not chunks:
                    print(f"  No chunks created for {os.path.basename(file_path)}")
                    skipped_count += 1
                    continue
                    
                # 3. Generate embeddings
                texts = [chunk["text"] for chunk in chunks]
                metadatas = [chunk["metadata"] for chunk in chunks]
                embeddings = embed_provider.embed_documents(texts)
                
                # 4. Insert into Weaviate
                store.add_documents(texts, metadatas, embeddings)
                print(f"  Successfully ingested {len(chunks)} chunks.")
                ingested_count += 1
                
            except Exception as e:
                print(f"  Error processing {os.path.basename(file_path)}: {e}")
                error_count += 1
                
        print(f"\nIngestion summary:")
        print(f"  Total processed: {ingested_count}")
        print(f"  Total skipped: {skipped_count}")
        print(f"  Total errors: {error_count}")
        
    finally:
        store.close()

if __name__ == "__main__":
    main()
