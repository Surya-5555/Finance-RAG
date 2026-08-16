# LLM Provider Module

This module implements the Large Language Model (LLM) abstraction layer for the RAG pipeline. It handles sending prompts and returning raw text completions.

### Purpose
To generate textual responses based on retrieved contexts and user queries, keeping the generation logic fully decoupled from the rest of the orchestration pipeline.

### Inputs
- **Initialization**: Can take an optional `model_name` string to override the configuration.
- **Generation**: `generate(prompt: str)` takes a single, formatted text prompt string.

### Outputs
- `generate(prompt: str)` returns a single `str` representing the LLM's response.

### Architecture
The module utilizes an abstract base class (`BaseLLMProvider`) to enforce a standardized `generate()` method across any potential LLM vendor. 

Currently, the `OllamaProvider` is implemented using the official `ollama` Python package to interface with a locally running model server (defined by `OLLAMA_BASE_URL` in `.env`). It dynamically reads `llm_model` and `temperature` configurations from `config/config.yaml`.

### Why an Abstraction Layer?
By creating a strict `BaseLLMProvider` contract, the orchestration (LangGraph) layer will never depend directly on the `ollama` package or its specific API parameters (like `options={"temperature": 0.0}`). 

If you want to migrate to OpenAI, Gemini, or Anthropic later:
1. Create a `src/llm/openai_provider.py`
2. Inherit from `BaseLLMProvider`
3. Implement `generate(prompt: str) -> str` using the OpenAI client
4. Swap the initialized provider in your orchestration script.

None of the prompt templates, RAG graphs, or vector store code will require any modification.

### Design Decisions
- **No Orchestration**: This module does not contain any LangChain logic, chains, prompt templates, or retrieval steps. It strictly takes a string and outputs a string.
- **Fail-Fast Validation**: Empty strings raise an immediate `ValueError` to prevent unnecessary network calls. Missing config files gracefully fallback to sensible defaults without silently corrupting state.
