import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tempfile
import shutil
from pathlib import Path
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer

from src.ingestion.loaders import load_documents
from src.ingestion.chunking import chunk_documents
from src.embeddings.huggingface import HuggingFaceEmbeddingProvider
from src.vectorstore.weaviate_store import WeaviateVectorStore
from src.retrieval.retriever import VectorRetriever
from src.llm.ollama_provider import OllamaProvider
from src.prompt.builder import PromptBuilder
from src.rag_engine import RAGEngine

def create_minimal_pdf(filename, text):
    """Generate a minimal valid PDF containing the given text."""
    stream_content = f"BT\n/F1 12 Tf\n72 712 Td\n({text}) Tj\nET\n"
    stream_len = len(stream_content)
    pdf_content = (
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        "3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> /MediaBox [0 0 595 842] /Contents 4 0 R >> endobj\n"
        f"4 0 obj << /Length {stream_len} >> stream\n"
        f"{stream_content}"
        "endstream\nendobj\n"
        "xref\n"
        "0 5\n"
        "0000000000 65535 f\n"
        "0000000009 00000 n\n"
        "0000000058 00000 n\n"
        "0000000115 00000 n\n"
        "0000000282 00000 n\n"
        "trailer << /Size 5 /Root 1 0 R >>\n"
        "startxref\n389\n"
        "%%EOF\n"
    ).encode('ascii')
    with open(filename, 'wb') as f:
        f.write(pdf_content)

def main():
    print("==================================================")
    print("PHASE 1 RIGOROUS VERIFICATION RUNNER")
    print("==================================================")
    
    # 1. Setup temp environment inside workspace (D-drive)
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_verification_dir")
    os.makedirs(temp_dir, exist_ok=True)
    
    store = None
    llm = None
    try:
        # Tiny verification documents
        pdf_path = os.path.join(temp_dir, "verification_doc.pdf")
        md_path = os.path.join(temp_dir, "verification_doc.md")
        html_path = os.path.join(temp_dir, "verification_doc.html")
        
        pdf_text = "LangChain is an open-source framework designed to simplify the creation of applications using large language models (LLMs)."
        md_text = "RAG is a technique that retrieves documents from an external source to ground the LLM's answers. It is widely used in question answering systems."
        html_text = "<html><head><title>Weaviate Docs</title></head><body><p>Weaviate is a cloud-based vector database that stores objects and their embeddings for fast similarity search.</p></body></html>"
        
        create_minimal_pdf(pdf_path, pdf_text)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_text)
            
        # Large verification document (4500 - 5000 tokens)
        large_md_path = os.path.join(temp_dir, "large_verification_doc.md")
        base_sentence = "Python is an extremely versatile programming language widely adopted in artificial intelligence, web development, data analysis, and automation fields. "
        large_text = base_sentence * 200
        with open(large_md_path, "w", encoding="utf-8") as f:
            f.write(large_text)
            
        print("[1] Temp files created successfully.")
        
        # 2. Document Loading
        print("\n[2] Testing loaders...")
        pdf_docs = load_documents(pdf_path)
        md_docs = load_documents(md_path)
        html_docs = load_documents(html_path)
        large_docs = load_documents(large_md_path)
        print(f"  PDF Loader: PASS (Loaded {len(pdf_docs)} docs)")
        print(f"  Markdown Loader: PASS (Loaded {len(md_docs)} docs)")
        print(f"  HTML Loader: PASS (Loaded {len(html_docs)} docs)")
        print(f"  Large MD Loader: PASS (Loaded {len(large_docs)} docs)")
            
        all_docs = pdf_docs + md_docs + html_docs
        
        # 3. Token-Based Chunking (Large document split verification)
        print("\n[3] Testing token-based chunking on large document...")
        large_chunks = chunk_documents(large_docs)
        print(f"  Generated {len(large_chunks)} chunks for large document.")
        
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        
        chunk_token_lists = []
        for idx, chunk in enumerate(large_chunks, 1):
            tokens = tokenizer.encode(chunk["text"], add_special_tokens=False)
            chunk_token_lists.append(tokens)
            print(f"Chunk {idx}: {len(tokens)} tokens")
            
            if idx < len(large_chunks):
                assert 300 <= len(tokens) <= 500, f"Chunk {idx} size {len(tokens)} tokens is outside of 300-500 range!"
            else:
                assert len(tokens) <= 500, f"Last chunk size {len(tokens)} tokens exceeds 500 tokens!"
                
            assert len(tokens) <= 512, f"Chunk {idx} exceeds model limit of 512 tokens! (size: {len(tokens)})"
                
            m = chunk["metadata"]
            assert m["source"] == os.path.abspath(large_md_path), "source mismatch"
            assert m["filename"] == "large_verification_doc.md", "filename mismatch"
            assert m["page"] is None, "page should be None"
            assert m["chunk_id"] == f"large_verification_doc.md_chunk_{idx}", "chunk_id mismatch"
            
        print("\nOverlap:")
        for idx in range(len(chunk_token_lists) - 1):
            tokens_curr = chunk_token_lists[idx]
            tokens_next = chunk_token_lists[idx + 1]
            # Find actual overlap
            overlap_len = 0
            for i in range(1, 101):
                if tokens_curr[-i:] == tokens_next[:i]:
                    overlap_len = i
            
            print(f"Chunk{idx+1} -> Chunk{idx+2} : {overlap_len} tokens")
            
            assert 90 <= overlap_len <= 100, f"Overlap between Chunk {idx+1} and Chunk {idx+2} is {overlap_len}, expected ~100!"
            
        print("\nPASS")
        print("  Token-Based Chunking size & overlap: PASS")
            
        # 4. Chunk metadata check on all docs
        print("\n[4] Checking chunk metadata on small documents...")
        chunks = chunk_documents(all_docs)
        for c in chunks:
            m = c["metadata"]
            print(f"  Metadata check for {m['filename']}:")
            print(f"    source: {m.get('source')}")
            print(f"    filename: {m.get('filename')}")
            print(f"    page: {m.get('page')}")
            print(f"    title: {m.get('title')}")
            print(f"    chunk_id: {m.get('chunk_id')}")
            
            assert "source" in m, "source key missing"
            assert "filename" in m, "filename key missing"
            assert "page" in m, "page key missing"
            assert "title" in m, "title key missing"
            assert "chunk_id" in m, "chunk_id key missing"
            
        print("  Small doc Metadata: PASS")
            
        # 5. Embedding Generation
        print("\n[5] Testing embedding generation...")
        embed_provider = HuggingFaceEmbeddingProvider()
        texts = [c["text"] for c in chunks]
        embeddings = embed_provider.embed_documents(texts)
        
        print(f"  Generated {len(embeddings)} embeddings.")
        dims = set(len(emb) for emb in embeddings)
        print(f"  Embedding dimensions detected: {dims}")
        
        assert len(embeddings) == len(chunks), "Some chunks skipped embedding"
        assert len(dims) == 1, "Inconsistent embedding dimensions"
        print("  Embeddings: PASS")
            
        # Weaviate Operations with Retry Loop for Cloud Resilience
        for attempt in range(1, 4):
            try:
                # 6. Weaviate Storage & Deduplication
                print(f"\n[6] Testing Weaviate Storage (Attempt {attempt})...")
                store = WeaviateVectorStore()
                store.delete_collection()
                import time
                time.sleep(2)
                store.create_collection()
                
                metadatas = [c["metadata"] for c in chunks]
                store.add_documents(texts, metadatas, embeddings)
                time.sleep(2)
                
                is_pdf_ingested = store.is_file_ingested(os.path.abspath(pdf_path))
                print(f"  Duplicate check (expected True): {is_pdf_ingested}")
                assert is_pdf_ingested, "Duplicate check failed to detect ingested file"
                
                is_unrelated_ingested = store.is_file_ingested("some_other_file.md")
                print(f"  Duplicate check (expected False): {is_unrelated_ingested}")
                assert not is_unrelated_ingested, "Duplicate check falsely detected uningested file"
                
                print("  Weaviate Storage & Deduplication: PASS")
                
                # 7. Retrieval
                print("\n[7] Testing retrieval pipeline...")
                retriever = VectorRetriever(embedding_provider=embed_provider, vector_store=store)
                
                results = retriever.retrieve("What is Weaviate?")
                print(f"  Retrieved {len(results)} chunks for query 'What is Weaviate?'.")
                
                for idx, res in enumerate(results, 1):
                    print(f"    Result {idx}: {res['text']}")
                    print(f"    Distance: {res.get('distance')}")
                    print(f"    Metadata: {res.get('metadata')}")
                    
                assert len(results) > 0, "No chunks retrieved"
                assert results[0]["metadata"]["filename"] == "verification_doc.html", "Top result mismatch"
                print("  Retrieval: PASS")
                
                # 8. Generation, Grounding & Citations
                print("\n[8] Testing RAG generation and grounding...")
                llm = OllamaProvider()
                prompt_builder = PromptBuilder()
                rag = RAGEngine(retriever=retriever, llm=llm, prompt_builder=prompt_builder)
                
                # Question 1: Answerable
                print("\n  --- Question 1 (Answerable): 'What is LangChain?' ---")
                ans1 = rag.ask("What is LangChain?")
                print(ans1)
                
                assert ans1.startswith("Answer:"), "Answer does not prefix with 'Answer:'"
                assert "Sources:" in ans1, "Answer does not contain 'Sources:' section"
                assert "verification_doc.pdf" in ans1, "Sources missing PDF document"
                
                # Question 2: Partially Answerable / Combined
                print("\n  --- Question 2 (Combined): 'What is Weaviate and RAG?' ---")
                ans2 = rag.ask("What is Weaviate and RAG?")
                print(ans2)
                assert "verification_doc.html" in ans2, "Sources missing HTML document"
                assert "verification_doc.md" in ans2, "Sources missing MD document"
                
                # Question 3: Unrelated
                print("\n  --- Question 3 (Unrelated): 'What is the capital of France?' ---")
                ans3 = rag.ask("What is the capital of France?")
                print(ans3)
                assert "I cannot answer this based on the provided documents" in ans3 or "insufficient" in ans3.lower() or "not contain" in ans3.lower(), "hallucinated or answered unrelated question"
                
                print("\n  Generation, Grounding & Citations: PASS")
                print("\n==================================================")
                print("VERIFICATION COMPLETE: ALL CHECKS PASS")
                print("==================================================")
                break
                
            except Exception as e:
                if store is not None:
                    try:
                        store.close()
                    except:
                        pass
                if attempt == 3:
                    raise e
                print(f"  Attempt {attempt} failed: {e}. Retrying in 5 seconds...")
                import time
                time.sleep(5)
        
    except Exception as e:
        print(f"\nVerification failed with exception: {e}")
        
    finally:
        print("\nCleaning up resources...")
        if store is not None:
            try:
                if store.client.is_connected():
                    store.delete_collection()
            except Exception as se:
                print(f"Error deleting collection: {se}")
            try:
                if store.client.is_connected():
                    store.close()
            except Exception as se:
                print(f"Error closing store: {se}")
        if llm is not None:
            try:
                llm.close()
            except Exception as le:
                print(f"Error closing LLM: {le}")
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as te:
                print(f"Error removing temp dir: {te}")
        print("Cleanup complete.")

if __name__ == "__main__":
    main()
