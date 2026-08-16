# Vector Store Module

This module implements the storage and retrieval primitives for the RAG pipeline using Weaviate Cloud.

### Purpose
To provide a dedicated layer for storing embedded documents and performing vector similarity searches. It abstracts the specifics of the Weaviate v4 Python client so the higher-level RAG logic remains completely agnostic to the database implementation.

### Inputs
- **Document Addition**:
  - `texts`: A list of strings representing the document chunks.
  - `metadatas`: A list of dictionaries containing metadata (`source`, `page`) for each chunk.
  - `embeddings`: A list of float vectors representing the dense embeddings of the chunks.
- **Search**:
  - `query_vector`: A single float vector representing the user query.
  - `limit`: The maximum number of results to return.

### Outputs
- **Search**: Returns a list of dictionaries where each dictionary contains:
  - `text`: The matched chunk string.
  - `metadata`: The `source` and `page` associated with the chunk.
  - `distance`: The semantic distance score returned by Weaviate.

### Collection Structure
The collection schema is defined dynamically in the `create_collection()` method:
- **`text`**: Text property storing the actual chunk content.
- **`source`**: Text property storing the origin document filename.
- **`page`**: Integer property storing the page number (if applicable).
- **Vectorizer**: Disabled (`Configure.Vectorizer.none()`) because embeddings are generated externally by the `src/embeddings` module.

### Design Decisions
- **Bring Your Own Vectors**: We purposefully disable Weaviate's internal vectorization engine to guarantee that we have full control over the embedding models locally. This ensures consistency across chunking, query embedding, and vector storage.
- **Fail-Fast Validation**: The inputs to `add_documents` are validated to ensure `texts`, `metadatas`, and `embeddings` align perfectly. Missing environments variables (`WEAVIATE_URL`, `WEAVIATE_API_KEY`) will immediately crash the initialization.
- **Strict Separation of Concerns**: This module contains absolutely no BM25 keyword search, hybrid search logic, cross-encoder reranking, or graph orchestration. It serves merely as a dumb, fast vector index.

### Future Usage
Future retrieval components (like a retriever node in LangGraph) will initialize `WeaviateVectorStore`, embed the user's query using the `src/embeddings` layer, and call `similarity_search(query_vector)`. The results will then be passed to a cross-encoder reranker for further sorting.
