# GyaanSetu — AI-Powered NCERT Tutor

> **GyaanSetu** (ज्ञानसेतु) means *Bridge of Knowledge* — an intelligent RAG-based tutor that answers NCERT curriculum questions using AI.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Table of Contents

- [What is GyaanSetu?](#what-is-gyaansetu)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [Docker Setup](#docker-setup)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [Getting Help](#getting-help)
- [Contributing](#contributing)
- [Maintainers](#maintainers)

---

## What is GyaanSetu?

GyaanSetu is a full-stack educational AI assistant built around **Retrieval-Augmented Generation (RAG)**. It ingests NCERT textbook content from HuggingFace, indexes it into a FAISS vector store, and uses a large language model to answer student questions with relevant context sourced directly from the curriculum.

The system currently supports **Class 10 Science** and is designed to be extended to other classes and subjects.

---

## Key Features

- 🔍 **RAG-powered answers** — retrieves the most relevant NCERT chunks before generating a response
- 🧠 **Semantic search** — uses `sentence-transformers/all-MiniLM-L6-v2` embeddings and FAISS for fast similarity search
- 🤖 **Pluggable LLM** — supports [OpenRouter](https://openrouter.ai/) (any hosted model) or a built-in mock LLM for offline development
- 📊 **Built-in evaluation** — measures retrieval accuracy, generation quality, latency, and embedding coherence
- ⚡ **FastAPI backend** — clean REST API with CORS support, health checks, and automatic docs at `/docs`
- 🎨 **React frontend** — landing page, interactive chat interface, and an evaluation dashboard
- 🐳 **Docker support** — single-command containerized deployment

---

## Architecture

```
HuggingFace Dataset
        │
        ▼
┌─────────────────────────────────────────┐
│           Ingestion Pipeline            │
│  HFLoader → Cleaner → Chunker → Embedder│
└──────────────────┬──────────────────────┘
                   │  FAISS index + metadata
                   ▼
┌──────────────────────────────────────────┐
│              RAG Service                 │
│  embed query → FAISS search → build      │
│  context → LLM (OpenRouter / MockLLM)    │
└──────────────────┬───────────────────────┘
                   │  JSON response
                   ▼
┌──────────────────────────────────────────┐
│            FastAPI Backend               │
│  POST /chat  GET /evaluation  GET /health│
└──────────────────┬───────────────────────┘
                   │  HTTP
                   ▼
┌──────────────────────────────────────────┐
│         React Frontend (Vite)            │
│   Home  │  Chat  │  Evaluation Dashboard │
└──────────────────────────────────────────┘
```

---

## Project Structure

```
GyaanSetu/
├── api/                    # FastAPI application (routes, schemas)
├── config/                 # Environment-based settings
├── evaluation/             # Evaluation framework & test data
├── ingestion/              # Data loading, cleaning, chunking, embedding
├── llm/                    # LLM factory, OpenRouter client, MockLLM, prompt builder
├── rag/                    # Core RAG service (retrieve → context → generate)
├── scripts/                # Utility scripts for ingestion, RAG, and retrieval testing
├── ui/react_app/           # React 19 + Vite frontend (Home, Chat, Evaluation pages)
├── vector_store/           # FAISS index builder and loader
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

| Tool | Minimum Version |
|------|----------------|
| Python | 3.11 |
| Node.js | 18 |
| npm | 9 |
| Docker *(optional)* | 20 |

### Backend Setup

**1. Clone the repository**

```bash
git clone https://github.com/anan5093/GyaanSetu.git
cd GyaanSetu
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

**3. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Copy the example below into a `.env` file in the project root and fill in your values:

```dotenv
# LLM (set USE_MOCK_LLM=false to use a real model)
USE_MOCK_LLM=true
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=nvidia/nemotron-3-super:free

# Dataset / retrieval
CLASS_ID=10
SUBJECT=science
TOP_K=3
SCORE_THRESHOLD=0.3

# API server
API_HOST=127.0.0.1
API_PORT=8000
FRONTEND_URL=http://localhost:5173
ENABLE_LOGS=true
```

**5. Run the data ingestion pipeline**

This step downloads the NCERT dataset from HuggingFace, cleans it, chunks it, and builds the FAISS index.

```bash
python scripts/test_ingestion.py
python -m vector_store.build_faiss_index
```

**6. Start the API server**

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

### Frontend Setup

```bash
cd ui/react_app
npm install
npm run dev
```

The app is served at [http://localhost:5173](http://localhost:5173) by default.

---

### Docker Setup

Build and run the full backend in a single container:

```bash
docker build -t gyaansetu .
docker run -p 10000:10000 \
  -e USE_MOCK_LLM=false \
  -e OPENROUTER_API_KEY=your_key_here \
  gyaansetu
```

The API is then available at [http://localhost:10000](http://localhost:10000).

---

## Configuration

All settings are controlled via environment variables (or a `.env` file loaded by `python-dotenv`).

| Variable | Default | Description |
|----------|---------|-------------|
| `CLASS_ID` | `10` | NCERT class number |
| `SUBJECT` | `science` | Subject name |
| `TOP_K` | `3` | Number of chunks retrieved per query |
| `SCORE_THRESHOLD` | `0.3` | Minimum cosine-similarity score to include a chunk |
| `USE_MOCK_LLM` | `true` | `false` to use a real OpenRouter model |
| `OPENROUTER_API_KEY` | — | Your [OpenRouter](https://openrouter.ai/) API key |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3-super:free` | Model identifier on OpenRouter |
| `MAX_TOKENS` | `300` | Maximum tokens for LLM responses |
| `API_HOST` | `127.0.0.1` | Host for the FastAPI server |
| `API_PORT` | `8000` | Port for the FastAPI server |
| `FRONTEND_URL` | `http://localhost:5173` | Allowed CORS origin |
| `ENABLE_LOGS` | `true` | Print verbose startup and request logs |

---

## API Reference

### `POST /chat`

Ask a question about the NCERT curriculum.

**Request body**

```json
{
  "question": "What is Newton's second law of motion?"
}
```

**Response**

```json
{
  "answer": "Newton's second law states that ...",
  "sources": [
    { "topic": "Force and Laws of Motion", "chapter": "Force and Laws of Motion", "type": "concept" }
  ],
  "metrics": {
    "latency": 0.42,
    "chunks_used": 3
  }
}
```

---

### `GET /evaluation`

Runs the full evaluation suite and returns metrics.

```json
{
  "status": "success",
  "evaluation": {
    "retrieval": { ... },
    "generation": { ... },
    "latency": { ... },
    "embedding": { ... }
  }
}
```

---

### `GET /health`

Returns the server and RAG initialization status.

```json
{ "status": "ok", "rag_loaded": true }
```

---

## Evaluation

GyaanSetu ships with a built-in evaluation framework under `evaluation/`. It measures:

| Module | What it measures |
|--------|-----------------|
| `retrieval_eval.py` | Whether relevant chunks are retrieved for test questions |
| `generation_eval.py` | Quality of generated answers against expected responses |
| `latency_eval.py` | End-to-end query latency (p50 / p95) |
| `embedding_eval.py` | Semantic coherence of keyword embeddings |

Test data lives in `evaluation/test_data.json`. You can run the evaluation via the `/evaluation` endpoint or directly:

```bash
python scripts/test_rag.py
```

---

## Getting Help

- **Bug reports & feature requests** — open an issue on the [GitHub Issues](https://github.com/anan5093/GyaanSetu/issues) page
- **Interactive API docs** — available at `http://localhost:8000/docs` when the server is running
- **OpenRouter models** — see [openrouter.ai/models](https://openrouter.ai/models) for supported LLM identifiers
- **NCERT dataset** — hosted on HuggingFace at [`KadamParth/NCERT_Science_10th`](https://huggingface.co/datasets/KadamParth/NCERT_Science_10th)

---

## Contributing

Contributions are welcome! Please open an issue first to discuss significant changes.

1. Fork the repository and create a feature branch
2. Make your changes with clear commit messages
3. Ensure existing tests pass:

```bash
python scripts/test_ingestion.py
python scripts/test_rag.py
python scripts/test_vector_retrieval.py
```
4. Open a pull request against `main`

---

## Maintainers

| Name | GitHub |
|------|--------|
| Anand | [@anan5093](https://github.com/anan5093) |

---

*GyaanSetu is an open-source project. See [LICENSE](LICENSE) for details.*
