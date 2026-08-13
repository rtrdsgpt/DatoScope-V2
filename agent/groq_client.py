"""
Multi-key rotation for the Groq client.

A single free-tier Groq key's rate limit gets burned through well before an
agentic run (multiple tool-calling turns, RAG-grounded explanations) is
done, so a comma-separated key list lets a run keep going on the next key
instead of stalling on retry/backoff. Same pattern as the sibling
Financial-Anomaly-Detection-Using-RAG project's `groq_key_rotation.py`,
reused here since it's a well-tested solution to the same problem.
"""

from __future__ import annotations

_RATE_LIMIT_HINTS = (
    "rate limit", "rate_limit_exceeded", "429",
    "per day", "tpd", "rpd", "daily quota", "quota exceeded",
)


def parse_api_keys(raw: str) -> list[str]:
    """Splits a comma-separated key list, e.g. GROQ_API_KEY="gsk_abc,gsk_def"
    in .env for round-robin rotation when one key's rate limit is hit. A
    single key with no comma still works exactly as before."""
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def is_rate_limit_error(message: str) -> bool:
    """Pattern-matched against the error message text since Groq doesn't
    expose this as a structured field uniformly on the Python SDK's
    exception types."""
    lowered = str(message).lower()
    return any(hint in lowered for hint in _RATE_LIMIT_HINTS)


class RotatingGroqClient:
    """Presents a single `create_chat_completion(**kwargs)` method backed by
    a real `groq.Groq` client, transparently rotating to the next configured
    API key and retrying the same call when the current key hits a rate
    limit. With a single key, behaves exactly like calling
    `groq.Groq(api_key=key).chat.completions.create(...)`."""

    def __init__(self, api_keys: list[str]):
        if not api_keys:
            raise ValueError("RotatingGroqClient needs at least one API key")
        from groq import Groq

        self._Groq = Groq
        self._api_keys = api_keys
        self._index = 0
        self._client = self._Groq(api_key=self._api_keys[0])

    def create_chat_completion(self, **kwargs):
        while True:
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as e:
                if is_rate_limit_error(str(e)) and self._index + 1 < len(self._api_keys):
                    self._index += 1
                    print(
                        f"  Groq key #{self._index + 1}/{len(self._api_keys)} hit a rate limit; "
                        f"switching to the next configured key",
                        flush=True,
                    )
                    self._client = self._Groq(api_key=self._api_keys[self._index])
                    continue
                raise


def get_client() -> RotatingGroqClient:
    import os

    from dotenv import load_dotenv

    load_dotenv()
    keys = parse_api_keys(os.environ.get("GROQ_API_KEY", ""))
    if not keys:
        raise RuntimeError("GROQ_API_KEY is not configured (see .env.example)")
    return RotatingGroqClient(keys)
