"""Ticket preprocessing: clean text, detect language."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from langdetect import detect, LangDetectException


@dataclass
class PreprocessedTicket:
    original_issue: str
    original_subject: str
    clean_issue: str
    clean_subject: str
    language: str  # ISO 639-1
    query: str  # the text used for retrieval


def _clean(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def _detect_lang(text: str) -> str:
    if not text or len(text.strip()) < 10:
        return "en"
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def preprocess(issue: str, subject: str) -> PreprocessedTicket:
    """Clean and analyze a single ticket."""
    clean_issue = _clean(issue)
    clean_subject = _clean(subject)
    lang = _detect_lang(clean_issue)

    if clean_subject and clean_subject.lower() not in clean_issue.lower():
        query = f"{clean_subject} {clean_issue}"
    else:
        query = clean_issue

    return PreprocessedTicket(
        original_issue=issue,
        original_subject=subject,
        clean_issue=clean_issue,
        clean_subject=clean_subject,
        language=lang,
        query=query,
    )
