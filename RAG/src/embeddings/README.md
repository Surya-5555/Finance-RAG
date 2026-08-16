# Embeddings Module

This module provides an abstraction layer for transforming raw text strings into dense vector representations (embeddings). It is designed to be highly modular and extensible.

### Purpose
To generate numerical representations of documents and user queries that capture their semantic meaning, enabling semantic search and similarity comparisons within a vector database.

### Provider Abstraction
The system utilizes an abstract base class (`BaseEmbeddingProvider`) to define a standardized interface for embedding models. This ensures that the rest of the RAG pipeline does not depend on a specific implementation. It makes it trivial to swap out local models (like `sentence-transformers`) for external APIs (like OpenAI or Voyage) in the future without changing the core retrieval or ingestion logic.

### Inputs
- `embed_documents(texts: list[str])`: A list of text chunks (strings) to be embedded.
- `embed_query(text: str)`: A single user query string to be embedded.

### Outputs
- `embed_documents`: `list[list[float]]` representing a list of embedding vectors corresponding to the input documents.
- `embed_query`: `list[float]` representing a single embedding vector for the query.

### Design Decisions
- **Provider-Agnostic Interface**: We explicitly abstract away the embedding vendor. The core logic only knows about `BaseEmbeddingProvider`.
- **Config-Driven Instantiation**: The specific model (e.g., `BAAI/bge-small-en-v1.5`) is dynamically read from `config/config.yaml`, ensuring that changes to the model do not require code changes.
- **Fail-Fast Validation**: The implementations enforce type checking and empty input validation to fail early and provide clear error messages.
- **No Vector DB Coupling**: This module strictly returns native Python types (`list[float]`) and has absolutely no knowledge of Weaviate or any retrieval/storage logic.
