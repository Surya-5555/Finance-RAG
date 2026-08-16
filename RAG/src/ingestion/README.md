# Ingestion Module

This module handles the extraction and parsing of raw documents into standard LangChain `Document` objects, as well as splitting them into smaller chunks suitable for embedding. It serves as the very first steps in the RAG pipeline.

## Data Loading

### Inputs
- `path` (`str`): The absolute or relative file path pointing to the raw document.

### Outputs
- `list[Document]`: A list of LangChain Document objects containing the extracted `page_content` and source `metadata`.

### Supported Formats
- PDF (`.pdf`) via `PyPDFLoader`
- Markdown and Text (`.md`, `.txt`) via `TextLoader`
- HTML (`.html`, `.htm`) via `BSHTMLLoader`

### Design Decisions
- **Automatic detection**: The loader dynamically selects the appropriate LangChain Community parsing class based on the file extension. 
- **Fail-fast validation**: If an unsupported file type is provided, the function immediately raises a clear `ValueError` instead of silently failing or returning empty data.

## Document Chunking

### Purpose
To split large documents into smaller, manageable chunks that fit within embedding models' context windows, while maintaining semantic context through overlapping.

### Inputs
- `documents` (`list[Document]`): A list of LangChain Document objects generated from the loading phase.

### Outputs
- `list[Document]`: A list of chunked LangChain Document objects.

### Chunking Strategy
- Uses `RecursiveCharacterTextSplitter` from `langchain_text_splitters`.
- It recursively splits text based on a hierarchy of characters (e.g., paragraphs, sentences, words) to keep semantically related pieces together.
- Chunk size and overlap are dynamically loaded from `config/config.yaml`.

### Metadata Preservation
- The original document metadata (e.g., `source`, `page`) is inherently preserved and copied into every resulting chunked `Document` object.

### Design Decisions
- **No Hardcoding**: Chunking parameters (`chunk_size`, `chunk_overlap`) are decoupled from code and defined in the central YAML configuration.
- **Modularity**: Operates strictly on `Document` objects without embedding or vector DB coupling.

### System Integration
The orchestration graph or pipeline calls `load_documents(path)` during the initial data ingestion phase. The returned documents are then forwarded to the chunking components, before proceeding to embedding and vector storage.
