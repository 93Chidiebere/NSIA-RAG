title: Nigeria State Intelligence Assistant (NSIA)
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false



# Nigeria State Intelligence Assistant (NSIA)

A local Retrieval-Augmented Generation (RAG) system that answers factual questions about Nigerian states, Local Government Areas (LGAs), and universities — powered by a hybrid retriever and a locally hosted Mistral 7B language model via Ollama.

---

## Overview

NSIA ingests live data from public Nigerian APIs, transforms the structured JSON responses into natural language documents, indexes them with both dense (FAISS) and sparse (BM25) retrievers, and routes user questions through an augmented prompt to a local Mistral 7B instance. The system is fully CPU-compatible and requires no paid API keys.

---

## Architecture

```
Public APIs
    │
    ▼
Structured-to-Unstructured Transformation
    │
    ▼
Text Corpus
    ├── Dense Embeddings (all-MiniLM-L6-v2) → FAISS Index
    └── Tokenized Corpus → BM25 Index
         │
         ▼
    Hybrid Retriever (Dense + BM25 → deduplicated top-k)
         │
         ▼
    Mistral [INST] Prompt Builder
         │
         ▼
    Ollama (local Mistral 7B) → Answer
```

---

## Data Sources

| Source | API Endpoint | Content |
|---|---|---|
| Nigeria States & LGAs | `https://nga-states-lga.onrender.com/fetch` | All 36 states + FCT and their LGAs |
| Nigerian Universities | `http://universities.hipolabs.com/search?country=Nigeria` | University names, domains, and websites |

All 37 state/FCT documents and all available university records are ingested and stored as natural language documents before indexing.

---

## Dependencies

Install all required packages with:

```bash
pip install pandas numpy requests sentence-transformers faiss-cpu==1.7.4 transformers torch accelerate rank-bm25 ollama
```

| Package | Role |
|---|---|
| `sentence-transformers` | Dense embedding model (`all-MiniLM-L6-v2`) |
| `faiss-cpu` | Vector similarity search index |
| `rank-bm25` | Sparse keyword retrieval (BM25Okapi) |
| `ollama` | Local LLM interface for Mistral 7B |
| `transformers` + `torch` | Optional: TinyLlama fallback (commented out) |
| `requests` / `pandas` | API ingestion and data handling |

---

## Setup

### 1. Install Ollama

Download and install Ollama from [https://ollama.com](https://ollama.com), then pull the Mistral model:

```bash
ollama pull mistral
```

Ensure the Ollama background service is running before executing the notebook.

### 2. Run the Notebook

Execute cells in order. The notebook is self-contained — no manual data downloads are required. API calls are made at runtime.

---

## How It Works

### Step 1 — API Ingestion

The notebook calls two public REST APIs and inspects their response schemas using a reusable `inspect_api()` utility. Each data source has a dedicated ingestion module (`ingest_states()`, `ingest_universities()`) that returns a list of document dictionaries with the following schema:

```python
{
    "source":      "states_api" | "universities_api",
    "entity_type": "state" | "university",
    "entity_name": "<name>",
    "content":     "<natural language text>"
}
```

### Step 2 — Structured-to-Unstructured Transformation

Structured JSON (e.g., a list of LGA names) is converted into descriptive text passages. This transformation is necessary because RAG retrieval performs significantly better over natural language than raw key-value data.

Example output for a state document:

```
State: Lagos

Local Government Areas:
Agege, Ajeromi-Ifelodun, Alimosho, Amuwo-Odofin, Apapa, ...
```

### Step 3 — Embedding and Indexing

A text corpus is extracted from all documents. `all-MiniLM-L6-v2` encodes each document into a 384-dimensional vector. These vectors are stored in a FAISS `IndexFlatL2` index and saved to disk as `NSIA.index`. A BM25Okapi index is built in parallel over a tokenized version of the same corpus.

### Step 4 — Hybrid Retrieval

The `hybrid_retrieve(query, top_k=5)` function:
1. Encodes the query as a dense vector and fetches the top-k nearest neighbours from FAISS.
2. Scores all documents with BM25 and fetches the top-k by score.
3. Merges both result sets, deduplicates by content, and returns up to `top_k` unique documents.

### Step 5 — Prompt Construction and Generation

`build_prompt(query, retrieved_docs)` assembles a Mistral-compatible `[INST]` prompt. The system instruction instructs the model to answer strictly from the retrieved context and return `"I could not find that information."` when the answer is absent. The prompt is sent to a locally running Mistral 7B via `ollama.generate()` with `temperature=0.0` for deterministic output.

---

## Example Queries

```python
test_queries = [
    "Which LGAs are in Lagos?",
    "List universities in Nigeria",
    "Which state contains Ikeja?",
    "Tell me about University of Lagos"
]
```

---

## Generation Parameters

| Parameter | Value | Purpose |
|---|---|---|
| `temperature` | `0.0` | Deterministic, fact-grounded output |
| `top_p` | `0.9` | Nucleus sampling ceiling |
| `num_predict` | `150` | Maximum output tokens |

---

## Saved Artifacts

| File | Description |
|---|---|
| `NSIA.index` | FAISS vector index (persisted to disk with `faiss.write_index`) |

The BM25 index and document list are held in memory and must be rebuilt on each session. For production use, serialize them with `pickle`.

---

## Notes

- **TinyLlama** (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) is included as a commented-out alternative LLM for fully offline or low-memory environments.
- The States & LGAs API does not reliably accept state names from the `/fetch` bulk endpoint; the notebook therefore iterates over a hardcoded list of all 37 entities using the `?state=` query parameter.
- The `extract_answer()` function referenced in the batch evaluation loop is treated as optional; if not defined, the raw model output is used directly as the final answer.

---

## License

This project is released for research and educational use. Data is sourced from publicly available APIs. The Mistral 7B model is subject to the [Mistral AI Terms of Use](https://mistral.ai/terms).
