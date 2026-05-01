# TriageForge — Support Ticket Triage Agent

A terminal-based AI agent that triages support tickets across HackerRank, Claude, and Visa ecosystems using hybrid RAG (BM25 + dense retrieval) over the provided support corpus.

## Setup

```bash
# 1. Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp ../.env.example ../.env
# Edit ../.env and add your GROQ_API_KEY (free at console.groq.com)
```

## Usage

```bash
# Run on the test tickets (default paths)
python main.py

# Or specify paths explicitly
python main.py --input ../support_tickets/support_tickets.csv \
               --output ../support_tickets/output.csv

# Force rebuild the corpus index
python main.py --force-reindex
```

First run downloads the embedding model (~33MB) and builds the index (~1-2 min).
Subsequent runs use the cached index and complete in ~90 seconds for 29 tickets.

## Evaluate against sample data

```bash
python eval/run_sample.py
```

This compares predictions against `sample_support_tickets.csv` and reports per-row accuracy.

## Architecture

```
main.py          CLI entry, orchestration loop
indexer.py       Corpus chunking + BM25/FAISS index build (cached)
retriever.py     Hybrid BM25 + dense retrieval with RRF fusion
preprocess.py    Text cleaning + language detection
llm.py           Groq/HuggingFace LLM client (JSON mode)
prompts.py       System prompt + few-shot examples
decide.py        Threshold-based escalation gate
postprocess.py   Product area derivation + citation enforcement
schema.py        Pydantic models for structured output
taxonomy.py      Corpus-derived product area taxonomy
```

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes (if using Groq) | — | Groq API key |
| `HF_TOKEN` | Yes (if using HF) | — | HuggingFace token |
| `LLM_PROVIDER` | No | `groq` | `groq` or `huggingface` |
| `LLM_MODEL` | No | per-provider | Override model name |

## Cost

- Groq free tier: $0
- HuggingFace fallback: ~$0.50 per full run (29 tickets)
- Embeddings: local (BAAI/bge-small-en-v1.5), $0
