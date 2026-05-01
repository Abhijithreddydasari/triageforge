"""Pure threshold functions for escalation gating.

No ML, no LLM calls — just measurable rules.
"""

from __future__ import annotations

from typing import List, Optional

from schema import Chunk

RETRIEVAL_SCORE_FLOOR = 0.015


def should_force_escalate(
    top_chunks: List[Chunk],
    language: str,
) -> Optional[str]:
    """Return an escalation reason string, or None if no forced escalation.

    This is a pre-LLM gate. The LLM can still choose to escalate on its own;
    this function only handles cases where we *know* escalation is needed
    regardless of LLM output.
    """
    if not top_chunks:
        return "No relevant documentation found in corpus"

    top_score = top_chunks[0].rrf_score
    if top_score < RETRIEVAL_SCORE_FLOOR:
        return (
            f"Retrieval confidence too low (top score {top_score:.4f} "
            f"< threshold {RETRIEVAL_SCORE_FLOOR})"
        )

    return None
