# Production-Grade RAG Pipeline

This is the implementation of Phase 2 of our robust, production-grade Retrieval-Augmented Generation pipeline.

## Architecture

The system has been completely decoupled into distinct functional layers representing state-of-the-art enterprise RAG architecture:

```text
User Question
        │
        ▼
Query Embedding
        │
 ┌──────┴──────┐
 ▼             ▼
BM25      Vector Search
 │             │
 └──────┬──────┘
        ▼
HybridRetriever
        ▼
RRFRetriever
        ▼
CrossEncoderReranker
        ▼
Top-N Chunks
        ▼
PromptBuilder
        ▼
EvidenceValidator
    ┌──────┴──────┐
    ▼             ▼
Supported    Unsupported
    │             │
    ▼             ▼
 Call LLM     Decline
    │
    ▼
Grounded Answer + Citations
```

### Stage Breakdown

1. **Hybrid Retrieval (`HybridRetriever`)**
   Executes exact-match keyword search (BM25) and semantic vector search in parallel.

2. **Reciprocal Rank Fusion (`RRFRetriever`)**
   Merges the two disparate retrieval lists, removing duplicates and mathematically scoring chunks based on their combined rank across both methods.

3. **Cross-Encoder Re-Ranking (`CrossEncoderReranker`)**
   Evaluates each candidate chunk directly against the query using a HuggingFace Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`). This provides a highly accurate semantic relevance score (0.0 to 1.0) and truncates the list to the `Top-N`.

4. **Prompt Builder (`PromptBuilder` & `PromptManager`)**
   Dynamically loads external text templates (`config/prompts/rag_answer_v1.txt`) and strictly injects the context and metadata.

5. **Evidence Validation (`EvidenceValidator`)**
   Before touching the LLM, the system validates the chunks against configured relevance thresholds. If the Cross-Encoder indicates poor relevance, the LLM is bypassed entirely to prevent hallucination.

### Orchestration (`RAGEngine`)

The `rag_engine.py` orchestrates these components linearly and outputs a strict JSON-compatible dictionary response.

#### Example Supported Response
```json
{
    "answer": "LangChain provides an abstraction layer for LLMs [Source 1].",
    "supported": true,
    "sources": [
        {
            "source": "/data/langchain_docs.pdf",
            "filename": "langchain_docs.pdf",
            "page": 14,
            "title": "Introduction",
            "chunk_id": "chunk_492"
        }
    ]
}
```

#### Example Declined Response
If validation fails due to lack of evidence:
```json
{
    "answer": "I don't have enough information in the retrieved documents to answer this question reliably.",
    "supported": false,
    "sources": []
}
```

## Running Verification

You can instantly verify the complete end-to-end data flow using the provided test script:
```bash
python verify_phase2.py
```