# GraphRAG vs. Conventional RAG: Codebase Q&A Demo

This is a demonstration repository designed to evaluate and showcase the effectiveness of **GraphRAG** against conventional RAG systems. It allows for side-by-side comparison of retrieval strategies including AST-based, Code-specific, and Recursive chunking.

## Overview

This system allows you to upload or point to a Python codebase, index it using various strategies, and compare how different RAG architectures handle complex source code queries. It includes a sophisticated hallucination detection mechanism (SelfCheckGPT) to further validate the reliability of each approach.

### Key Features

- **Multi-Strategy Chunking**: Optimize retrieval by choosing between `AST`, `Code`, or `Recursive` chunking.
- **Hallucination Self-Check**: Uses a `SelfCheckGPT`-inspired approach to verify generated answers by comparing them against sampled responses and calculating embedding similarity.
- **RAG Agent**: Built on LangGraph to provide structured reasoning and document retrieval.
- **Integrated Architecture**: A FastAPI backend coupled with a modern React + Tailwind CSS frontend.
- **Vector Search**: Persistent indexing using ChromaDB and HuggingFace embeddings.

## Setup & Installation

### Prerequisites

- [Python 3.12+](https://www.python.org/downloads/)
- [UV](https://github.com/astral-sh/uv) (Python package manager)
- [Node.js & npm](https://nodejs.org/) (for Frontend)
- An AI Provider API Key (e.g., Groq)

### 1. Backend Configuration

Clone the repository and install dependencies:

```powershell
uv sync
```

Create a `.env` file by copying the example and filling in your API keys:

```powershell
cp .env.example .env
```

Your `.env` file should contain:

- `GROQ_API_KEY`: Required for LLM access.
- `LANGSMITH_API_KEY`: Optional, for tracing and debugging.
- `LANGSMITH_TRACING`: Set to `true` to enable LangSmith.

### 2. Frontend Configuration

Navigate to the frontend directory and install dependencies:

```powershell
cd src/frontend
npm install
```

## Running the Application

You can start both the Backend (FastAPI) and Frontend (Vite) simultaneously using the convenience script:

```powershell
python start.py
```

- **Frontend**: [http://localhost:5173](http://localhost:5173)
- **API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Testing the Self-Check

1. **Upload your codebase** by clicking the upload icon in the sidebar and selecting a `.zip` file containing your Python code.
2. **Select the database** (collection) that was just created from the header dropdown.
3. **Ask a question** about your code (e.g., "What does the Calculator class do?").
4. Once the answer is generated, click the **"Self Check"** (Shield icon) button.
5. The system will generate a secondary "sampled" response internally and show you a **Similarity Score**. If the score is low (<= 50%), it will flag a potential hallucination.

## Evaluation Framework

This project includes an automated evaluation framework to test the four RAG prototypes (P1-P4) against 30 categorized questions.

### RAG Prototypes

| Prototype | Strategy | Description |
|-----------|----------|-------------|
| **P1** | `function` | Baseline function-level chunking |
| **P2** | `ast` | AST-aware code structure chunking |
| **P3** | `context` | Context-enriched AST chunking |
| **P4** | `graph` | GraphRAG with knowledge graph traversal |

### Running the Evaluation

1. **Start the API server**:

   ```powershell
   uv run uvicorn src.api.main:app --reload
   ```

2. **Run the evaluation** (in a new terminal):

   ```powershell
   uv run python scripts/run_evaluation.py --zip data/Cancer_Detection.zip
   ```

   **Options:**
   - `--zip PATH` - Path to codebase zip file (default: `data/Cancer_Detection.zip`)
   - `--output PATH` - Output JSON path (default: `data/evaluation_results_raw.json`)
   - `--strategies STRATS` - Comma-separated strategies (e.g., `context,graph`)
   - `--skip-indexing` - Reuse existing indexed collections

3. **Generate visualization charts**:

   ```powershell
   uv run python scripts/generate_charts.py
   ```

   Charts are saved to `data/charts/`.

### Evaluation Output

- `data/evaluation_results_raw.json` - Raw LLM responses, latency, selfcheck scores
- `data/evaluation_final.json` - Manual correctness assessments and hallucination classifications
- `data/charts/` - Visualization charts (accuracy, latency, consistency)

### Question Categories

| Category | Questions | Description |
|----------|-----------|-------------|
| **1. Simple Lookup** | Q1-Q10 | Find class/function locations |
| **2. Local Context** | Q11-Q20 | Parameters, variables, dependencies |
| **3. Global Relational** | Q21-Q30 | Cross-file tracing, architecture |

### Expected Results

Based on evaluation, P4 (GraphRAG) outperforms other strategies on Category 3 (global/relational) questions, while P2/P3 provide faster responses for simple lookups.
