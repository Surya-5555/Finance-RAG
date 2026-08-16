# Retrieval Module

This module implements the complete retrieval pipeline for the RAG system. It combines broad initial searches (Vector and BM25) with mathematical fusion (RRF) and deep semantic reranking (Cross-Encoder) to deliver the most precise context possible to the LLM.

### Architecture

All retrievers implement the abstract interface defined in `src/retrieval/base.py`. The data flows sequentially through the following pipeline:

**1. VectorRetriever & 2. BM25Retriever**
- Initial broad search using semantic similarity (Dense Vectors) and exact term matching (Sparse BM25).

**3. HybridRetriever**
- Executes the Vector and BM25 retrievers independently and returns an unmerged dictionary of both result sets.

**4. RRFRetriever (Reciprocal Rank Fusion)**
- Wraps the HybridRetriever and fuses the results based purely on their retrieved rank mathematically (`1 / (k + rank)`). It deduplicates chunks and preserves metadata.

**5. CrossEncoderReranker**
- Wraps the RRFRetriever. It takes the fused list, scores every query-document pair deeply, resorts them, and truncates the list to the `Top-N` absolute best matches.

### Cross-Encoders vs. Bi-Encoders (Vector Search)

**Bi-Encoders (Vector Search):**
In Vector Search, the query and document are embedded *independently* into vectors, and their distance is computed. This is extremely fast and scalable to millions of documents, but it is shallow—it lacks deep understanding of how specific query terms interact with specific document terms.

**Cross-Encoders:**
A Cross-Encoder feeds both the query and the document into the transformer model *simultaneously* (`[CLS] Query [SEP] Document [SEP]`). The attention mechanism calculates the exact relationships between the words in the query and the words in the document. This is highly accurate and precise, but it is extremely computationally expensive.

### Why Reranking Improves Precision
Because a Cross-Encoder evaluates the exact pair together, it can detect subtle nuances, negations, and complex reasoning that Bi-Encoders miss. However, because it is so slow, we cannot run a Cross-Encoder over our entire database.

**Why RRF is still useful before reranking:**
We use RRF to surface the top 20-100 most diverse and promising candidates from our fast but shallow search engines. The Cross-Encoder then acts as a highly accurate "magnifying glass", reranking only those top candidates to fix any mistakes the shallow searches made.

**Why only Top-N documents are kept:**
Language Models have a finite context window and suffer from "Lost in the Middle" syndrome (where they ignore context in the middle of long prompts). By aggressively filtering the reranked list down to a small `Top-N` (e.g., 5), we ensure the LLM is only fed the highest-quality, strictly relevant context, reducing hallucinations and prompt cost.

### Worked Numerical Example

**1. Hybrid Search (Broad)**
- Vector Search: `Chunk A` (Rank 1), `Chunk B` (Rank 2)
- BM25 Search: `Chunk B` (Rank 1), `Chunk C` (Rank 2)

**2. RRF Fusion (k=60)**
- `Chunk B` (Rank 1 BM25, Rank 2 Vector) = Score 0.03251
- `Chunk A` (Rank 1 Vector) = Score 0.01639
- `Chunk C` (Rank 2 BM25) = Score 0.01612

*Current Order: B, A, C*

**3. Cross-Encoder Reranking**
The Cross-Encoder evaluates the pairs:
- `Query` + `Chunk B` -> 0.12
- `Query` + `Chunk A` -> 0.98 (Perfect Match!)
- `Query` + `Chunk C` -> 0.45

*New Order: A, C, B*

**4. Final Top-N Filter (Top-N = 2)**
Output sent to the Prompt Builder:
1. `Chunk A` (Cross-Encoder: 0.98, RRF: 0.01639)
2. `Chunk C` (Cross-Encoder: 0.45, RRF: 0.01612)

*(Chunk B is discarded as it fell outside the Top-N limit).*

### Future Integration (Prompt 4)
The output is a fully enriched, strictly ordered list of dictionaries ready for citation enforcement and prompt building, containing the `document`, `rrf_score`, `cross_encoder_score`, and `retrievers` provenance.
