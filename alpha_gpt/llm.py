"""Shared LLM plumbing: OpenRouter client + robust JSON calls.

One place for everything that talks to the model, so the debate agents, the
moderator, the factory generators, and the single-agent baseline all parse model
output and retry the same way (instead of each re-implementing it).
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from typing import Any, Callable

from openai import OpenAI

from alpha_gpt.config import Config

logger = logging.getLogger(__name__)


class UsageMeter:
    """Thread-safe accumulator of REAL billed token usage across ``call_json`` calls.

    Pass one meter through a unit of paid work (e.g. one debate) and read the totals
    after: every billed response — including repair retries whose JSON failed to parse —
    is recorded, so cost telemetry reflects what the API actually charged rather than a
    flat per-debate estimate."""

    def __init__(self):
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def record(self, response) -> None:
        u = getattr(response, "usage", None)
        pt = int(getattr(u, "prompt_tokens", 0) or 0)
        ct = int(getattr(u, "completion_tokens", 0) or 0)
        with self._lock:
            self.calls += 1
            self.prompt_tokens += pt
            self.completion_tokens += ct

# HTTP statuses worth retrying: rate limit + transient server errors. A 4xx like 400
# (bad request) or 402 (no credits) is NOT retryable — fail fast instead of looping.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_BACKOFF_SECONDS = 30.0


def make_client(config: Config) -> OpenAI:
    """OpenAI-compatible client. Routes by model id: a provider-prefixed id
    ('deepseek/deepseek-v3.2') goes to OpenRouter; a bare id ('gpt-5.4-mini') goes to OpenAI
    direct (using OPENAI_API_KEY). ``max_retries=0`` so the retry policy lives in exactly one
    place (``_create_with_backoff``) — the SDK's own default retries would otherwise stack on
    top of ours and multiply attempts under a rate-limit wall."""
    if "/" in config.llm_model:  # provider-prefixed -> OpenRouter
        return OpenAI(base_url=config.openrouter_base_url,
                      api_key=config.openrouter_api_key, max_retries=0)
    return OpenAI(base_url=config.openai_base_url,
                  api_key=config.openai_api_key, max_retries=0)


def _completion_params(model: str, *, temperature: float, max_tokens: int) -> dict:
    """Per-model request params. OpenAI's GPT-5 / o-series reasoning models reject the legacy
    ``max_tokens`` (they require ``max_completion_tokens``) and reject any non-default
    ``temperature`` (only 1 is allowed), 400-ing otherwise. Everything else (DeepSeek, GPT-4o,
    …) takes the classic params. Keying on the model id — provider prefix stripped — keeps the
    existing OpenRouter/DeepSeek path working unchanged."""
    name = (model or "").split("/")[-1]
    if name.startswith(("gpt-5", "o1", "o3", "o4")):
        return {"max_completion_tokens": max_tokens}  # temperature omitted -> defaults to 1
    return {"max_tokens": max_tokens, "temperature": temperature}


def _is_retryable(exc: Exception) -> bool:
    """True for rate-limit / transient API errors (worth a backoff), False otherwise."""
    # OpenAI reports an exhausted balance as a 429 with code "insufficient_quota" — that is
    # a PERMANENT condition (no amount of backoff refills the account), so fail fast instead
    # of burning the full retry budget on every call of a doomed run.
    if "insufficient_quota" in str(exc):
        return False
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in _RETRYABLE_STATUS:
        return True
    # Timeouts / connection drops carry no status code.
    return type(exc).__name__ in (
        "RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError",
    )


def _create_with_backoff(client: OpenAI, *, api_retries: int, label: str = "", **kwargs):
    """Call chat.completions.create, retrying transient errors with exponential backoff
    + jitter. Non-retryable errors (and the final attempt) re-raise so callers see them."""
    for attempt in range(api_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - we re-raise unless it's retryable
            if attempt >= api_retries or not _is_retryable(exc):
                raise
            sleep = min(_MAX_BACKOFF_SECONDS, (2 ** attempt) + random.uniform(0, 0.5))
            logger.warning("%sretryable API error (attempt %d/%d): %s; backing off %.1fs",
                           f"[{label}] " if label else "", attempt + 1, api_retries, exc, sleep)
            time.sleep(sleep)
    raise RuntimeError("unreachable: retry loop exhausted")  # loop always returns or re-raises


def extract_json_payload(content: str | None) -> Any | None:
    """Parse a JSON object/array from a model response, tolerating code fences and prose."""
    if not content:
        return None
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0]

    content = content.strip()
    if not content:
        return None

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost {...} or [...] span.
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = content.find(opener), content.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _default_repair_prompt(original_prompt: str, invalid_response: str) -> str:
    return (
        "The previous response was not valid JSON.\n\n"
        f"Original prompt:\n{original_prompt}\n\n"
        f"Invalid response:\n{invalid_response}\n\n"
        "Rewrite the answer as valid JSON only. No markdown, commentary, or explanation."
    )


def call_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    empty_factory: Callable[[], Any] = dict,
    retries: int = 2,
    temperature: float = 0.5,
    retry_temperature: float = 0.0,
    max_tokens: int = 4000,
    api_retries: int = 5,
    repair_prompt: Callable[[str, str], str] = _default_repair_prompt,
    label: str = "",
    usage: "UsageMeter | None" = None,
) -> Any:
    """Call the model and parse JSON, retrying once with a repair prompt on bad output.

    Returns the parsed payload, or ``empty_factory()`` if every attempt fails. The
    first attempt uses ``temperature``; repair attempts drop to ``retry_temperature``
    (0.0) for determinism. ``api_retries`` is the SEPARATE budget for transient API
    errors (rate limits / 5xx) — each call backs off and retries up to that many times
    before giving up, which is what lets us push request concurrency without 429 storms.
    ``usage``: optional :class:`UsageMeter`; every billed response (including repair
    attempts whose JSON fails to parse) is recorded on it for honest cost accounting.
    """
    retries = max(1, retries)
    last_response = ""
    for attempt in range(retries):
        if attempt == 0:
            prompt, temp = user_prompt, temperature
        else:
            prompt, temp = repair_prompt(user_prompt, last_response), retry_temperature
        try:
            response = _create_with_backoff(
                client, api_retries=api_retries, label=label,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                **_completion_params(model, temperature=temp, max_tokens=max_tokens),
            )
            if usage is not None:
                usage.record(response)  # billed regardless of whether the JSON parses
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                logger.warning("%sresponse hit the token cap (max_tokens=%d) and was truncated "
                               "mid-output; raise the cap if this recurs",
                               f"[{label}] " if label else "", max_tokens)
            last_response = (choice.message.content or "").strip()
            payload = extract_json_payload(last_response)
            if payload is not None:
                return payload
        except Exception as exc:
            logger.error("%scall_json failed on attempt %s: %s", f"[{label}] " if label else "", attempt + 1, exc)
            break
    return empty_factory()
