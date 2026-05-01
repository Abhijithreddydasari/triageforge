"""Post-process LLM output: validate enums, add citations."""

from __future__ import annotations

from pathlib import Path
from typing import List

from schema import Chunk, LLMResponse, TicketResult
from taxonomy import area_from_chunk_path

VALID_STATUS = {"replied", "escalated"}
VALID_REQUEST_TYPE = {"product_issue", "feature_request", "bug", "invalid"}


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

    product_area = llm_response.product_area.strip()
    if not product_area:
        product_area = area_from_chunk_path(chunks[0].source_path, data_dir) if chunks else "general"

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
