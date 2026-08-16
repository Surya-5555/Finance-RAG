# RAG System — Agent Instructions

## Rules you must follow

1. Always refer to PROJECT_CONTEXT.md for architecture decisions. Never deviate from the defined tech stack without flagging it.

2. File structure to maintain:
   rag/
   ├── docs/
   │   ├── PROJECT_CONTEXT.md
   │   └── AGENT_INSTRUCTIONS.md
   ├── config/
   │   ├── config.yaml
   │   └── flags.yaml
   ├── prompts/
   │   └── (versioned prompt YAML files)
   ├── src/
   │   ├── ingestion/
   │   ├── retrieval/
   │   ├── reranker/
   │   ├── llm/          ← interface layer + adapters
   │   ├── graph/        ← LangGraph nodes and edges
   │   ├── evaluation/
   │   └── api/          ← FastAPI app
   ├── tests/
   ├── golden_dataset.json
   ├── docker-compose.yml
   └── .github/workflows/eval.yml

3. When building the LLM layer, always implement the interface pattern:
   - BaseLLMAdapter (abstract)
   - OllamaAdapter(BaseLLMAdapter)
   - OpenAIAdapter(BaseLLMAdapter)
   - ClaudeAdapter(BaseLLMAdapter)
   - GeminiAdapter(BaseLLMAdapter)
   The graph nodes call BaseLLMAdapter.generate() only. Never call a provider SDK directly from graph nodes.

4. When building LangGraph nodes, every node must read from and write to a typed StateGraph state object. No passing raw dicts between nodes.

5. The citation enforcement node must:
   - Check if retrieved chunks contain information that supports the query
   - If yes: route to generate node
   - If no: return a structured decline response, never hallucinate
   - If borderline: route back to retrieval with a broader query (max 2 retries, tracked in state)

6. config.yaml must be the single source of truth for:
   - embedding_model
   - reranker_model
   - retrieval_k
   - rrf_k
   - prompt_version
   - faithfulness_threshold
   Never hardcode these values anywhere in source code.

7. flags.yaml controls:
   - use_reranker
   - use_hybrid_search
   - use_citation_enforcement
   Each flag must be checked at runtime so the pipeline can run with any combination disabled for A/B testing.

8. RAGAS evaluation must run against golden_dataset.json. The GitHub Actions workflow must fail if faithfulness score < faithfulness_threshold from config.yaml.

9. Every FastAPI endpoint must be async. Streaming responses required for the generate endpoint.

10. Docker Compose must bring up: FastAPI, Weaviate, PostgreSQL, Langfuse, Prometheus, Grafana in a single `docker compose up` command.

## Documentation Discipline

Every time you write or modify code inside a service folder under src/ (ingestion, retrieval, reranker, llm, graph, evaluation, api), you must also create or update a README.md inside that exact same folder. This is mandatory, not optional — but it only applies to the folder you are actually working in. Never create documentation files anywhere else in the repo as a side effect of this rule.

Each service README should explain, in plain language: what this module does, what it takes as input and what it returns, the one or two design decisions behind how it's built, and how the rest of the system calls into it. Write in simple, clear English. No emojis, no marketing language, no filler words. Keep it short — a few well-written paragraphs is enough, not a wall of text.

If new code changes what an existing part of a README describes, rewrite that part so the doc matches the current code exactly. Do not leave the old description sitting there and just tack a new note underneath it. The README must always reflect what the code actually does right now, with no outdated sections left behind.
