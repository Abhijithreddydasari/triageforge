"""Ticket preprocessing: clean text, detect language, translate for retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 42


@dataclass
class PreprocessedTicket:
    original_issue: str
    original_subject: str
    clean_issue: str
    clean_subject: str
    language: str  # ISO 639-1
    query: str  # the text used for retrieval (always English)
    needs_translation: bool = False


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


def _translate_for_retrieval(text: str, source_lang: str) -> str:
    """Translate non-English text to English for better retrieval.

    Uses the LLM client to do a quick translation. Falls back to
    original text if translation fails.
    """
    try:
        from llm import call_llm_raw
        translated = call_llm_raw(
            system="You are a translator. Translate the following text to English. "
                   "Output ONLY the English translation, nothing else.",
            user=text,
        )
        return translated.strip() if translated else text
    except Exception:
        return text


def preprocess(issue: str, subject: str) -> PreprocessedTicket:
    """Clean and analyze a single ticket."""
    clean_issue = _clean(issue)
    clean_subject = _clean(subject)
    lang = _detect_lang(clean_issue)

    if clean_subject and clean_subject.lower() not in clean_issue.lower():
        combined = f"{clean_subject} {clean_issue}"
    else:
        combined = clean_issue

    needs_translation = lang != "en"
    if needs_translation:
        query = _translate_for_retrieval(combined, lang)
    else:
        query = combined

    return PreprocessedTicket(
        original_issue=issue,
        original_subject=subject,
        clean_issue=clean_issue,
        clean_subject=clean_subject,
        language=lang,
        query=query,
        needs_translation=needs_translation,
    )
