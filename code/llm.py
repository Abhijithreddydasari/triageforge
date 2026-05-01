"""Thin LLM client wrapper: Groq primary, HuggingFace Inference fallback.

Controlled by environment variables:
  LLM_PROVIDER  = groq | huggingface   (default: groq)
  LLM_MODEL     = model name           (default: llama-3.3-70b-versatile)
  GROQ_API_KEY  = ...
  HF_TOKEN      = ...
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from schema import LLMResponse

MAX_RETRIES = 5
RETRY_BASE_DELAY = 10.0


def _get_provider() -> str:
    return os.environ.get("LLM_PROVIDER", "groq").lower()


def _get_model() -> str:
    provider = _get_provider()
    default = (
        "llama-3.3-70b-versatile"
        if provider == "groq"
        else "meta-llama/Llama-3.3-70B-Instruct"
    )
    return os.environ.get("LLM_MODEL", default)


def _call_groq(system: str, user: str, model: str) -> str:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        seed=42,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def _call_huggingface(system: str, user: str, model: str) -> str:
    from huggingface_hub import InferenceClient

    client = InferenceClient(
        model=model,
        token=os.environ["HF_TOKEN"],
    )
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=1024,
        response_format={"type": "json_object"},
        seed=42,
    )
    return response.choices[0].message.content


def call_llm_raw(system: str, user: str) -> str:
    """Call the LLM and return raw text (no JSON parsing). Used for translation."""
    provider = _get_provider()
    model = _get_model()

    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            seed=42,
            max_tokens=512,
        )
        return response.choices[0].message.content
    else:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=model, token=os.environ["HF_TOKEN"])
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=512,
            seed=42,
        )
        return response.choices[0].message.content


def call_llm(system_prompt: str, user_prompt: str) -> LLMResponse:
    """Call the configured LLM and return a validated LLMResponse.

    Retries with exponential backoff on transient failures.
    """
    provider = _get_provider()
    model = _get_model()

    call_fn = _call_groq if provider == "groq" else _call_huggingface

    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            raw = call_fn(system_prompt, user_prompt, model)
            parsed = json.loads(raw)
            return LLMResponse(**parsed)
        except json.JSONDecodeError as e:
            last_error = e
            print(f"[llm] JSON parse error on attempt {attempt + 1}: {e}")
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "rate_limit" in err_str or "429" in err_str or "too many" in err_str:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"[llm] Rate limited, retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"[llm] Error on attempt {attempt + 1}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY)

    raise RuntimeError(
        f"LLM call failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )
