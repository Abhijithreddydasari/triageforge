# TriageForge — Support Ticket Triage Agent

Terminal-based AI agent that triages support tickets across HackerRank, Claude, and Visa using hybrid RAG over the provided corpus.

> **For judges:** See [`DESIGN.md`](DESIGN.md) for full architecture rationale, pipeline diagrams, and design trade-offs.

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp ../.env.example ../.env
# Edit ../.env — add GROQ_API_KEY (free at console.groq.com)
```

## Run

```bash
# Process all test tickets (default paths)
python main.py

# Specify paths
python main.py --input ../support_tickets/support_tickets.csv \
               --output ../support_tickets/output.csv

# Force rebuild corpus index
python main.py --force-reindex
```

First run downloads the embedding model (~33MB) and builds the index (~8 min on CPU).
Subsequent runs use cached index and complete in ~9 min for 29 tickets.

## Evaluate

```bash
python eval/run_sample.py
```

Compares predictions against `sample_support_tickets.csv`, reports per-row accuracy.

## Architecture

```
main.py          CLI orchestrator
indexer.py       Corpus chunking + BM25/FAISS index (cached to data/index/)
retriever.py     Hybrid BM25 + dense retrieval, RRF fusion
preprocess.py    Text cleaning, language detection, query translation
llm.py           Groq/HuggingFace LLM client (JSON mode, retries)
prompts.py       System prompt, product area taxonomy, few-shot examples
decide.py        Retrieval-score threshold gate for forced escalation
postprocess.py   PII redaction, product_area validation, citation enforcement
schema.py        Pydantic models for structured output
taxonomy.py      Corpus-derived product area taxonomy
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes (Groq) | — | Groq API key (free tier) |
| `HF_TOKEN` | Yes (HF) | — | HuggingFace token |
| `LLM_PROVIDER` | No | `groq` | `groq` or `huggingface` |
| `LLM_MODEL` | No | per-provider | Model override |

## Key Design Choices

- **Hybrid retrieval:** BM25 catches exact tokens (phone numbers, acronyms), dense catches semantic similarity, RRF fuses them.
- **One LLM call per ticket:** Classification and response are coupled; splitting them adds latency without quality gain.
- **Dual-signal product area:** LLM picks from closed taxonomy + chunk path consensus overrides incorrect LLM picks.
- **Direct tone:** Responses calibrated to match expected output style — warm, no filler, numbered steps, specific data.
- **Multilingual:** Non-English queries translated to English for retrieval; LLM responds in source language; justification always in English.
- **Determinism:** temperature=0, seed=42, DetectorFactory.seed=42, content-hashed index cache.

See [DESIGN.md](DESIGN.md) for full decision rationale.
