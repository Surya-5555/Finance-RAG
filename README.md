# Finance RAG

A production-grade Retrieval-Augmented Generation (RAG) system specialized for financial documents (e.g., quarterly reports, 10-Ks, 10-Qs). This architecture is engineered to guarantee extremely high retrieval precision and mathematically eliminate numerical hallucinations using strict citation enforcement.

## System Architecture

The architecture is divided into three major phases:

### Phase 1: Ingestion & Storage
*   **PDF Parsing (Table-Aware):** Uses `pdfplumber` to extract text while explicitly maintaining the layout of financial tables. Tables are converted into clean Markdown format to ensure the LLM comprehends tabular data seamlessly.
*   **Metadata-Rich Chunking:** Uses `RecursiveCharacterTextSplitter` to divide the text into manageable chunks. Crucially, the system automatically tags every chunk with specific financial metadata (`company`, `year`, `quarter`) extracted from the filename, and flags table chunks with an `is_table` boolean.
*   **Embeddings & Vector Store:** Chunks are vectorized using OpenAI's `text-embedding-3-small` model and stored persistently in a local `ChromaDB` instance, enabling immediate usage across application restarts without re-indexing.

### Phase 2: Retrieval & Generation
*   **Hybrid Retrieval Pipeline:**
    *   **Semantic Search:** Queries the ChromaDB vector store to find conceptually related chunks.
    *   **Keyword Search (BM25):** Executes an exact-match keyword search, which is vital in finance (e.g., querying for specific product names or exact fiscal quarters).
    *   **Reciprocal Rank Fusion (RRF):** Fuses the results from both the Semantic and BM25 retrievers into a single, cohesive ranked list.
*   **Cross-Encoder Reranking:** Passes the fused results through a `sentence-transformers` Cross-Encoder. This computationally heavy step re-scores the relevance of the query against the retrieved chunks, surfacing the top 4 most highly relevant pieces of evidence.
*   **Generation (LLM):** Uses OpenAI's `GPT-4o` (temperature = `0.1` for maximum determinism) to formulate a concise answer based strictly on the retrieved chunks.
*   **Citation Enforcement:** To strictly prevent hallucination, the `EvidenceValidator` intercepts the final answer. It uses Regex to extract every numerical figure in the generated answer and checks if those exact digits exist in the retrieved text. If an ungrounded number is found, the system truthfully responds with: *"The information is not available in the uploaded documents."*

### Phase 3: API, UI & Observability
*   **FastAPI Backend:** A fully asynchronous REST API running the pipeline with endpoints for:
    *   `POST /ingest`: Upload and index multiple PDFs.
    *   `POST /ask`: Query the RAG engine and return the answer alongside exact source file names and page citations.
    *   `GET /stats`: Retrieve system statistics (chunks indexed, models used).
    *   `POST /eval`: Trigger the evaluation suite.
*   **Streamlit Frontend:** A sleek, decoupled user interface providing a chat-like experience and document upload hub.
*   **Observability:** The core `RAGEngine` is heavily instrumented with `langfuse` (`@observe` decorators) to provide deep production tracing of latency, prompt structures, and token usage.
*   **Evaluation Engine:** An automated test suite using `ragas` to rigorously benchmark Context Precision, Context Recall, Answer Relevancy, and Faithfulness against a golden dataset of financial questions.

## Setup & Execution

### Prerequisites
1. Python 3.10+
2. Install dependencies:
   ```bash
   pip install -r RAG/requirements.txt
   ```
3. Set your OpenAI API key in a `.env` file at the root of the project:
   ```env
   OPENAI_API_KEY=sk-your-key-here
   ```

### Running the Application

**1. Start the FastAPI Backend**
```bash
cd RAG
uvicorn api.main:app --reload
```
*The API will be available at http://localhost:8000*

**2. Start the Streamlit Frontend**
Open a new terminal and run:
```bash
cd RAG
streamlit run app.py
```
*The UI will automatically open in your browser at http://localhost:8501*

### Running Tests and Evaluations
To execute the unit test suite:
```bash
$env:PYTHONPATH="RAG"  # (Windows)
# export PYTHONPATH="RAG" (Linux/Mac)
pytest RAG/tests/
```

To run the Ragas golden dataset evaluation:
```bash
python RAG/evaluation/run_evaluation.py
```
