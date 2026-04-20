"""OpenAI judge client with retries.

Used by HEx-PHI (harmfulness 1-5) and MT-Bench (quality 1-10).
"""
from __future__ import annotations

import os
from typing import Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


def _client():
    from openai import OpenAI  # Imported lazily so unit imports don't require openai.

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set; the GPT judge is required for "
            "HEx-PHI and MT-Bench. Disable those benchmarks or set the env var."
        )
    return OpenAI(api_key=api_key)


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def judge_chat(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = "gpt-4",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """Call the chat-completions API once with retry/backoff."""
    client = _client()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content or ""


def extract_first_int(text: str, lo: int, hi: int) -> Optional[int]:
    """Parse the first integer in ``[lo, hi]`` found in ``text``, or None."""
    import re

    for m in re.finditer(r"-?\d+", text):
        try:
            v = int(m.group(0))
        except ValueError:
            continue
        if lo <= v <= hi:
            return v
    return None
