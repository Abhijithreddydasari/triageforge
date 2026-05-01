from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


class TicketInput:
    """Raw ticket as loaded from the CSV."""

    def __init__(self, index: int, issue: str, subject: str, company: str):
        self.index = index
        self.issue = issue.strip() if issue else ""
        self.subject = subject.strip() if subject else ""
        raw = company.strip() if company else ""
        self.company = raw if raw.lower() not in ("", "none") else None


@dataclass
class Chunk:
    """A single chunk of corpus text with provenance metadata."""

    id: str
    text: str
    source_path: str
    company: str
    area_path: str
    rrf_score: float = 0.0


class LLMResponse(BaseModel):
    """Structured JSON the LLM must return — validated by Pydantic."""

    status: Literal["replied", "escalated"]
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"]
    product_area: str = Field(description="Support category / domain area")
    response: str = Field(description="User-facing answer grounded in corpus")
    justification: str = Field(description="Why this decision was made")


@dataclass
class TicketResult:
    """Final processed result for one ticket, ready for CSV output."""

    issue: str
    subject: str
    company: str
    response: str = ""
    product_area: str = ""
    status: str = ""
    request_type: str = ""
    justification: str = ""
