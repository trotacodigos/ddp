import os
import random
from typing import Any


def get_keys(name: str) -> list[str]:
    """Read one or more comma-separated API keys from an environment variable."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set.")
    keys = [k.strip() for k in value.split(",") if k.strip()]
    random.shuffle(keys)
    return keys


def rough_token_count(text: str) -> int:
    return len(text.split()) if text else 0


def strip_thinking(text: str) -> str:
    """Drop a leading <think>...</think> block if present."""
    return text.split("</think>", 1)[-1] if "</think>" in text else text


def extract_token_usage(usage: Any) -> tuple[int | None, int | None]:
    if not usage:
        return None, None
    if isinstance(usage, dict):
        get = usage.get
    else:
        get = lambda k: getattr(usage, k, None)  # noqa: E731
    in_tok = get("prompt_tokens") or get("input_tokens") or get("prompt_eval_count")
    out_tok = get("completion_tokens") or get("output_tokens") or get("eval_count")
    return in_tok, out_tok
