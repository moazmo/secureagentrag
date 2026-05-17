"""Custom chat message rendering with citation support."""

from __future__ import annotations

import streamlit as st


def render_assistant_message(
    response_text: str,
    citations: list[dict],
    confidence: float,
    routing_info: dict | None = None,
) -> None:
    """Render an assistant response with citations, confidence badge, and routing info.

    Args:
        response_text: The main generated response text.
        citations: List of citation dicts (source_file, page_number, relevance_score, chunk_text).
        confidence: Confidence score between 0 and 1.
        routing_info: Optional dict with provider/model used for generation.
    """
    st.markdown(response_text)

    # Confidence badge
    if confidence >= 0.8:
        badge = f"🟢 Confidence: {confidence:.0%}"
    elif confidence >= 0.6:
        badge = f"🟡 Confidence: {confidence:.0%}"
    else:
        badge = f"🔴 Confidence: {confidence:.0%}"
    st.caption(badge)

    # Citations expander
    if citations:
        with st.expander(f"📚 Sources ({len(citations)} citations)"):
            for i, citation in enumerate(citations, 1):
                source = citation.get("source_file", "Unknown")
                page = citation.get("page_number", 0)
                score = citation.get("relevance_score", 0.0)
                snippet = citation.get("chunk_text", "")

                st.markdown(f"**{i}. {source}** — Page {page}")
                st.progress(score, text=f"Relevance: {score:.0%}")
                if snippet:
                    st.caption(snippet[:200] + ("..." if len(snippet) > 200 else ""))
                if i < len(citations):
                    st.divider()

    # Routing info
    if routing_info:
        provider = routing_info.get("provider", "unknown")
        model = routing_info.get("model", "unknown")
        forced = routing_info.get("forced_local", False)
        info_text = f"⚙️ {provider}/{model}"
        if forced:
            info_text += " (forced local — sensitive data)"
        st.caption(info_text)


def render_security_blocked(message: str) -> None:
    """Render a warning/error message when security check fails.

    Args:
        message: The security block message to display.
    """
    st.warning(f"🚫 **Access Denied**: {message}", icon="⚠️")


def render_thinking_indicator() -> None:
    """Render a custom thinking/processing indicator."""
    st.status("🤔 Processing your query through the RAG pipeline...", state="running")
