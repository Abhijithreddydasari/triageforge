"""Post-process LLM output: validate enums, add citations."""

from __future__ import annotations

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


def _area_from_chunks(chunks: List[Chunk], data_dir: Path) -> str:
    """Derive product area from the majority of chunk source paths."""
    if not chunks:
        return "general"
    areas = [area_from_chunk_path(c.source_path, data_dir) for c in chunks[:4]]
    areas = [a for a in areas if a and a in VALID_PRODUCT_AREAS]
    if not areas:
        return "general"
    from collections import Counter
    return Counter(areas).most_common(1)[0][0]


def _chunk_area_consensus(chunks: List[Chunk], data_dir: Path) -> bool:
    """True if all top chunks agree on the same product area."""
    if not chunks:
        return False
    areas = [area_from_chunk_path(c.source_path, data_dir) for c in chunks[:4]]
    areas = [a for a in areas if a and a in VALID_PRODUCT_AREAS]
    return len(set(areas)) == 1 and len(areas) >= 2


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

    chunk_area = _area_from_chunks(chunks, data_dir)
    if not product_area or product_area not in VALID_PRODUCT_AREAS:
        product_area = chunk_area if chunk_area in VALID_PRODUCT_AREAS else "general"
    elif chunk_area and chunk_area in VALID_PRODUCT_AREAS and chunk_area != product_area:
        # If chunks unanimously point to one area and LLM disagrees, trust chunks
        if _chunk_area_consensus(chunks, data_dir):
            product_area = chunk_area

    response_text = _ensure_citation(llm_response.response, chunks)

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
