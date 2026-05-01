# TriageForge — Design Document

## Summary

Terminal-based support ticket triage agent. Classifies and responds to tickets across HackerRank, Claude, and Visa using hybrid RAG, grounded entirely in the provided corpus. One LLM call per ticket, with PII redaction and multilingual support.

## Architecture

```mermaid
flowchart TD
    Input[CSV / Single Ticket / Interactive] --> Pre[Preprocess]
    Pre --> |langdetect| LangCheck{English?}
    LangCheck --> |yes| Ret[Hybrid Retriever]
    LangCheck --> |no| Trans[Translate to English] --> Ret
    
    Index[(Cached Index<br/>BM25 + FAISS)] --> Ret
    
    Ret --> Gate{RRF Score<br/>≥ 0.015?}
    Gate --> |no| ForcedEsc[Force Escalated Path]
    Gate --> |yes| LLM[Groq LLM<br/>JSON Mode<br/>temp=0, seed=42]
    ForcedEsc --> LLM
    
    LLM --> Post[Postprocess]
    Post --> |PII redaction| Post2[Validate Enums]
    Post2 --> |citation check| Out[output.csv]
```

## Retrieval Pipeline

```mermaid
flowchart LR
    Query[Clean Query] --> BM25[BM25<br/>Top 20]
    Query --> Dense[Dense FAISS<br/>Top 20]
    BM25 --> RRF[Reciprocal Rank<br/>Fusion k=60]
    Dense --> RRF
    RRF --> Filter{Company<br/>Filter?}
    Filter --> Top4[Top 4 Chunks<br/>≤800 chars each]
```

## CLI Commands

```mermaid
flowchart LR
    CLI[python main.py] --> Batch[--input/--output<br/>Batch CSV]
    CLI --> Single[--ticket 'issue'<br/>Single Ticket]
    CLI --> REPL[--interactive<br/>REPL Mode]
    CLI --> Health[--status<br/>Health Check]
    CLI --> Reindex[--force-reindex<br/>Rebuild Index]
```

## Design Decisions

### 1. Hybrid Retrieval (BM25 + Dense + RRF)

Combines sparse and dense retrieval with Reciprocal Rank Fusion.

- **BM25:** Catches exact tokens — phone numbers, "SCIM", "3-D Secure"
- **Dense (bge-small-en-v1.5):** Catches semantic similarity and paraphrase
- **RRF k=60:** Standard constant (Cormack et al. 2009). Lower k = more weight to top results.
- **Retrieval floor (0.015):** A doc must appear in at least one top-20 list. Below this → forced escalation.

### 2. One LLM Call Per Ticket

Classification and response generation are coupled. Splitting them adds latency without quality gain — the model needs retrieval context to decide whether to escalate.

### 3. Model Strategy

| Purpose | Model | Reason |
|---|---|---|
| Development/testing | `llama-3.1-8b-instant` | High rate limit, fast iteration |
| Final submission | `llama-3.3-70b-versatile` | Best quality for the one run that counts |

Both via Groq free tier. Switch by changing `LLM_MODEL` in `.env`.

### 4. Product Area Taxonomy

Closed set of canonical labels derived from corpus folder structure:

| Company | Areas |
|---|---|
| HackerRank | screen, community, interviews, settings, skillup, library, engage, integrations |
| Claude | conversation_management, privacy, billing, api, teams, claude_code, claude_desktop, safeguards, connectors |
| Visa | travel_support, general_support, fraud_protection, dispute_resolution |
| General | general |

LLM picks from this set — prevents inconsistent free-text labels.

### 5. PII Redaction

Regex-based removal before output:
- Email addresses → `[email redacted]`
- Order IDs (cs_live_*, ord_*) → `[order_id redacted]`
- Card numbers (16 digits) → `[card redacted]`

### 6. Multilingual Support

Non-English tickets → translate query to English for retrieval → LLM responds in original language (instructed in system prompt rule 6).

### 7. Error Handling

- Startup validation: checks API key format, data directory, index status
- Per-ticket: caught exceptions → graceful escalation with error in justification
- Rate limits: exponential backoff (5 retries, 10s base delay)
- Health check command: `python main.py --status`

## Failure Modes

| Failure | Mitigation | Residual Risk |
|---|---|---|
| No relevant docs | RRF threshold → force escalate | Unnecessary escalation |
| Prompt injection | System prompt isolation | Novel attacks |
| Hallucination | Grounding rules + citations | Subtle paraphrase |
| Wrong product area | Closed taxonomy constraint | Category ambiguity |
| Non-English query | Query translation before retrieval | Translation quality |
| PII leak | Regex redaction | Novel PII formats |
| Rate limits | Backoff + model fallback | Extended outages |
| Missing API key | Startup validation with clear message | — |

## Performance

| Metric | Value |
|---|---|
| Latency/ticket | ~3-5s |
| Full run (29 tickets) | ~9 min |
| Cost | $0 (Groq free tier) |
| Index build | ~8 min (first run) |
| Index load | ~3s (cached) |
| Determinism | temperature=0, seed=42 |
| Status accuracy | 100% (sample set) |
| Request type accuracy | 100% (sample set) |
