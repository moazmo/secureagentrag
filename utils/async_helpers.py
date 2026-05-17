"""Async helper utilities for bridging sync and async contexts."""

from __future__ import annotations

import asyncio
from typing import Any


def run_async(coro) -> Any:
    """Run an async coroutine from a synchronous context.

    Handles the case where an event loop may already be running
    (e.g., inside Jupyter, Streamlit, or certain async frameworks).
    Uses nest_asyncio when a loop is already running.

    Args:
        coro: Awaitable coroutine to execute.

    Returns:
        The result of the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        return asyncio.run(coro)
