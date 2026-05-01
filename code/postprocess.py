"""Post-process LLM output: validate enums, PII redaction, add citations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from schema import Chunk, LLMResponse, TicketResult
from taxonomy import area_from_chunk_path

VALID_STATUS = {"replied", "escalated"}
VALID_REQUEST_TYPE = {"product_issue", "feature_request", "bug", "invalid"}

VALID_PRODUCT_AREAS = {
    "screen", "community", "interviews", "settings", "skillup",
    "library", "engage", "integrations",
    "conversation_management", "privacy", "billing", "api", "teams",
    "claude_code", "claude_desktop", "safeguards", "connectors",
    "travel_support", "general_support", "fraud_protection", "dispute_resolution",
    "general",
}

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_ORDER_ID_RE = re.compile(r"\b(cs_live_[a-zA-Z0-9]+|ord_[a-zA-Z0-9]+|order[_\s]?id[:\s]*\S+)", re.IGNORECASE)
_CC_RE = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")


def _redact_pii(text: str) -> str:
    """Remove user PII (emails, order IDs, card numbers) from response text."""
    text = _EMAIL_RE.sub("[email redacted]", text)
    text = _ORDER_ID_RE.sub("[order_id redacted]", text)
    text = _CC_RE.sub("[card redacted]", text)
    return text


def _ensure_citation(response_text: str, chunks: List[Chunk]) -> str:
    """Append a source citation if the response doesn't already have one."""
    if not chunks:
        return response_text
    if "[source:" in response_text.lower():
        return response_text

    top_source = chunks[0].source_path
    return f"{response_text}\n\n[source: {top_source}]"


def postprocess(
    llm_response: LLMResponse,
    chunks: List[Chunk],
    data_dir: Path,
    forced_escalation: bool = False,
) -> TicketResult:
    """Validate and finalize the LLM response into a TicketResult."""
    status = llm_response.status.lower().strip()
    if status not in VALID_STATUS:
        status = "escalated"

    if forced_escalation:
        status = "escalated"

    request_type = llm_response.request_type.lower().strip()
    if request_type not in VALID_REQUEST_TYPE:
        request_type = "product_issue"

    product_area = llm_response.product_area.strip().lower().replace(" ", "_").replace("-", "_")
    if not product_area or product_area not in VALID_PRODUCT_AREAS:
        fallback = area_from_chunk_path(chunks[0].source_path, data_dir) if chunks else "general"
        product_area = fallback if fallback in VALID_PRODUCT_AREAS else product_area or "general"

    response_text = _redact_pii(llm_response.response)
    response_text = _ensure_citation(response_text, chunks)

    return TicketResult(
        issue="",
        subject="",
        company="",
        response=response_text,
        product_area=product_area,
        status=status,
        request_type=request_type,
        justification=llm_response.justification,
    )
