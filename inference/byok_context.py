"""Per-request BYOK credentials carried through the pipeline via a ContextVar.

The graph nodes (router → … → synthesizer) do not thread credentials through
their signatures — they call ``call_llm_*`` which builds an ``InferenceRouter``
with no per-request key. To make a visitor's *own* LLM key actually power their
request (the whole point of "Bring Your Own Key"), we stash the credentials in a
``contextvars.ContextVar`` at the top of ``run_rag_pipeline[_stream]``.

``ContextVar`` propagates across ``asyncio`` task boundaries (``gather``,
``astream``), so every node — and every parallel LLM call inside a node — sees
the same per-request creds without any signature plumbing. The token is reset in
a ``finally`` so the value never leaks between requests on a reused worker.

When no BYOK key is present the ContextVar holds ``None`` and the router falls
back to the owner's cached clients exactly as before.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass

# Providers whose BYOK client is built from a bearer/API key.
_KEY_PROVIDERS = frozenset({"groq", "openai", "anthropic"})


@dataclass(frozen=True)
class ByokRuntime:
    """Per-request BYOK credentials resolved from the visitor's request headers.

    Attributes:
        provider: Visitor's chosen provider ("groq" / "openai" / "anthropic" /
            "ollama"), already allow-list validated. None = no BYOK.
        user_key: Visitor's API key for a key-based provider. None for Ollama.
        ollama_url: Visitor's Ollama instance URL (only for provider="ollama").
    """

    provider: str | None = None
    user_key: str | None = None
    ollama_url: str | None = None

    def is_active(self) -> bool:
        """True when these creds can actually drive a per-request LLM client.

        A key-based provider needs a non-empty key; Ollama needs a URL. Anything
        else (missing provider, key without a provider, ollama without a URL)
        is *not* active — the router falls back to the owner's clients.
        """
        prov = (self.provider or "").lower()
        if prov in _KEY_PROVIDERS:
            return bool(self.user_key and self.user_key.strip())
        if prov == "ollama":
            return bool(self.ollama_url and self.ollama_url.strip())
        return False


_byok_ctx: contextvars.ContextVar[ByokRuntime | None] = contextvars.ContextVar(
    "byok_runtime", default=None
)


def set_byok_runtime(runtime: ByokRuntime | None) -> contextvars.Token:
    """Bind ``runtime`` for the current async context. Returns a reset token."""
    return _byok_ctx.set(runtime)


def get_byok_runtime() -> ByokRuntime | None:
    """Return the BYOK creds bound to the current async context, or None."""
    return _byok_ctx.get()


def reset_byok_runtime(token: contextvars.Token) -> None:
    """Restore the previous ContextVar value (call in a ``finally``)."""
    _byok_ctx.reset(token)
