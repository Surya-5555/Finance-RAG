# Generation & Prompting Module

This module represents the final stage of the RAG pipeline before data hits the Large Language Model. It is responsible for strictly formatting prompts, versioning system instructions, and aggressively enforcing evidence thresholds to prevent hallucinations.

### Architecture

**1. PromptManager**
- **Location**: `src/prompt/prompt_manager.py`
- Loads external `.txt` files from the `config/prompts/` directory.

**2. PromptBuilder**
- **Location**: `src/prompt/builder.py`
- Injects the `Top-N Chunks` outputted by the Retrieval layer strictly into the configured template.

**3. EvidenceValidator**
- **Location**: `src/generation/evidence_validator.py`
- Evaluates the reranked chunks against minimum relevance thresholds.

### Architectural Decisions

#### Why Externalize and Version Prompts?
Prompts are not just text; they are executable architecture. A slight modification to a system prompt can drastically alter the LLM's output format, tone, and safety constraints. 
By moving prompts outside of Python source code (`config/prompts/rag_answer_v1.txt`), we allow prompt engineers to tweak behaviors without deploying code changes. By versioning them (`v1`, `v2`), we ensure we can track performance drift and immediately rollback if a new prompt causes regressions in the LLM's behavior.

#### Citation Enforcement Architecture (Anti-Hallucination)
Before any API call is made to the LLM, the `EvidenceValidator` intercepts the `Top-N Chunks` array. It scans the `cross_encoder_score` attached to each chunk. 

If we require `minimum_chunks: 2` with a `minimum_cross_encoder_score: 0.50`, and the retrieval pipeline only surfaced 1 highly relevant document, the validation **fails**. 
By enforcing this natively in code, we guarantee that the LLM is never even invoked if the underlying retrieval data is poor, saving compute costs and eliminating hallucinations at the source.

#### Why Decline Unsupported Questions?
If a user asks an internal knowledge base "What is the capital of France?", the LLM inherently knows the answer from its pre-training data. However, if the RAG system retrieved zero documents about France, the LLM answering correctly is technically a hallucination *within the context of the specific business domain*. By declining unsupported questions ("I don't have enough information..."), we force the system to act exclusively as a retrieval-augmented assistant, strictly answering only what it can prove.

### Complete Flow Example

**1. Top-N Chunks from Reranker**
```json
[
  {
    "document": Document(page_content="Apple is a tech company.", metadata={"filename": "doc1.txt"}),
    "cross_encoder_score": 0.95
  },
  {
    "document": Document(page_content="They make iPhones.", metadata={"filename": "doc2.txt"}),
    "cross_encoder_score": 0.88
  }
]
```

**2. Evidence Validation**
`EvidenceValidator` checks the chunks. 
- Chunk 1 score (0.95) > 0.50. (Valid: 1)
- Chunk 2 score (0.88) > 0.50. (Valid: 2)
- Total valid chunks (2) >= minimum_chunks (2).
- **Result:** `{"supported": true}`

**3. Prompt Builder**
The builder consumes the chunks and the `rag_answer_v1.txt` template.

**Final Generated Prompt:**
```text
--------------------------------
System

You are a retrieval-augmented assistant.

Answer ONLY using the provided context.

If the context is insufficient,
say you don't know.

Always cite sources using the exact [Source X] format at the end of the relevant sentence.

--------------------------------
Context

[Source 1]
File: doc1.txt

Apple is a tech company.

[Source 2]
File: doc2.txt

They make iPhones.

--------------------------------
Question

What does Apple make?
--------------------------------
```

**4. Final Execution**
Because evidence was supported, the prompt above is sent to the LLM, resulting in a grounded answer:
*"Apple makes iPhones [Source 2]."*

If evidence had failed (e.g. cross-encoder scores were 0.12), the pipeline would instantly short-circuit and return the declined response:
*"I don't have enough information in the retrieved documents to answer this question reliably."*
