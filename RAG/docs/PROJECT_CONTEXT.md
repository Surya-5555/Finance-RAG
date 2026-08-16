# RAG System — Project Context

## Problem Statement
Build a production-grade RAG system that:
- Ingests content (PDFs, markdown, webpages)
- Chunks into 500–800 tokens with 100 token overlap
- Stores chunks as embeddings in Weaviate vector database
- Retrieves top-k most relevant chunks for a given query
- Generates answers that cite exactly where the information came from
- Explicitly DECLINES to answer if retrieved chunks do not support the response (no hallucination)

## Goal
Answer user queries with proper citations, grounded strictly in retrieved context.

---

## Architecture: 3 Phases

### Phase 1 — Foundation
Inject content → chunk → embed → store → retrieve → generate with citations.

Components:
- Orchestration: LangGraph (stateful graph, not linear chain)
- Document Loaders: LangChain Community Loaders (PDF, markdown, HTML)
- Chunking: RecursiveCharacterTextSplitter — 500 to 800 tokens, 100 token overlap
- Embeddings: BAAI/bge-small-en-v1.5 (development), BAAI/bge-base-en-v1.5 (production)
- Vector DB: Weaviate (local Docker)
- LLM: LLM Interface Layer with adapters — Ollama (local), OpenAI, Claude, Gemini
- Backend: FastAPI (async, streaming)
- Storage: PostgreSQL (metadata, sessions, audit logs)

### Phase 2 — Production Retrieval Quality
Upgrade retrieval from basic vector search to hybrid search with reranking.

Components:
- Keyword Retrieval: BM25
- Semantic Retrieval: BGE Embeddings
- Fusion: RRF (Reciprocal Rank Fusion) — zero tuning needed
- Re-ranker: BAAI/bge-reranker-base (default), BAAI/bge-reranker-large (if VRAM allows)
- Citation Enforcement: Custom LangGraph node — system declines if chunks don't support the answer
- Prompt Management: Versioned YAML config files (prompts/config/)
- Config Management: config.yaml (embedding_model, reranker_model, retrieval_k, rrf_k, prompt_version, faithfulness_threshold)
- Feature Flags: flags.yaml (use_reranker: true, use_hybrid_search: true, use_citation_enforcement: true)

### Phase 3 — Measurement of Faithfulness
Curate a golden dataset of 50–200 manually verified QA pairs. Write an offline evaluation script using RAGAS that measures:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Wire RAGAS evaluation into CI pipeline via GitHub Actions. Every PR automatically triggers an evaluation run. If quality drops below the faithfulness threshold defined in config.yaml, the build fails.

---

## Tech Stack (Final — v2)

| Role | Tool |
|---|---|
| Orchestration | LangGraph |
| Backend | FastAPI |
| Embeddings (dev) | BAAI/bge-small-en-v1.5 |
| Embeddings (prod) | BAAI/bge-base-en-v1.5 |
| Vector DB | Weaviate |
| Keyword Search | BM25 |
| Fusion | RRF |
| Re-ranker | BAAI/bge-reranker-base / large |
| LLM | Interface Layer (Ollama / OpenAI / Claude / Gemini) |
| Database | PostgreSQL |
| Evaluation | RAGAS |
| Tracing | Langfuse |
| Metrics | Prometheus |
| Dashboards | Grafana |
| Logging | OpenTelemetry |
| CI/CD | GitHub Actions |
| Containers (now) | Docker Compose |
| Containers (future) | Kubernetes |

---

## Constraints
- Zero spend. Fully free and open source.
- Runs locally on Windows, 16GB RAM, RTX 3050 (4GB VRAM)
- LLM served via Ollama with 4-bit quantized models
- No OpenAI API keys required for core pipeline
- LLM interface layer allows swapping to paid providers without changing retrieval architecture

---

## Key Engineering Decisions (defend these in interviews)

1. LangGraph over LangChain chains — RAG is a stateful decision loop, not a linear pipeline. Retry logic, citation enforcement, and conditional re-retrieval require a graph.
2. Hybrid BM25 + vector search — vector search understands meaning but misses exact terms. BM25 handles precise keywords. RRF merges both ranked lists with zero tuning.
3. Cross-encoder reranker — bi-encoders (used in retrieval) score query and chunk independently. Cross-encoder sees them together, giving much higher precision at the cost of ~50–100ms latency.
4. Citation enforcement node — hallucination is architecturally impossible. If chunks don't support the answer, the system explicitly declines rather than guessing.
5. Versioned prompts in YAML — prompts are part of system architecture. They must be diffable, reviewable, and rollback-able in git.
6. Config + feature flags — config.yaml controls all hyperparameters. flags.yaml enables A/B testing of pipeline components without code changes.
7. LLM interface layer — the retrieval pipeline calls llm.generate(prompt). The adapter handles the provider. Switching from Ollama to GPT-4 is a config change, not a code change.
8. RAGAS in CI — quality is a regression target, not a one-time measurement. Every PR must pass the faithfulness threshold or the build fails.
