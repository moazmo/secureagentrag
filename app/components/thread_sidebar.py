"""Conversation thread selector — extracted from ``app/views/chat.py``.

Renders the inline expander above the chat input that lets the operator
switch between persisted conversation threads, start a fresh one, or
delete the current one. All state lives in ``st.session_state``
(``active_thread_id`` + ``chat_history``); the persistent backing store is
``utils.conversation_store``.
"""

from __future__ import annotations

import uuid

import streamlit as st

from utils.conversation_store import conversation_store


def render_thread_sidebar() -> None:
    """Render the thread selector + create / load / delete controls.

    Side effects:
        - Mutates ``st.session_state.active_thread_id`` on selection /
          creation / deletion.
        - Mutates ``st.session_state.chat_history`` when a different thread
          is loaded (replaces the in-memory transcript with the persisted
          one) or when a thread is deleted (clears).
        - Calls ``st.rerun()`` after any state change so the chat view
          re-renders against the new thread.
    """
    user = st.session_state.current_user

    with st.expander("🧵 Conversation Threads", expanded=False):
        threads = conversation_store.list_threads(
            user_id=user["user_id"],
            org_id=user["org_id"],
            limit=20,
        )

        if not threads:
            st.info("No saved conversations yet.")
            if st.button("Start New Thread", width="stretch"):
                st.session_state.active_thread_id = str(uuid.uuid4())
                st.session_state.chat_history = []
                st.rerun()
            return

        thread_options = {
            t["thread_id"]: f"Thread {t['thread_id'][:8]}... ({t['message_count']} msgs)"
            for t in threads
        }
        thread_options["new"] = "+ Start New Thread"

        current = st.session_state.get("active_thread_id", "new")
        selected = st.selectbox(
            "Active Thread",
            options=list(thread_options.keys()),
            format_func=lambda x: thread_options[x],
            index=(list(thread_options.keys()).index(current) if current in thread_options else 0),
            key="thread_selector",
        )

        if selected == "new":
            if st.button("Create New Thread", width="stretch"):
                st.session_state.active_thread_id = str(uuid.uuid4())
                st.session_state.chat_history = []
                st.rerun()
            return

        if selected != st.session_state.get("active_thread_id"):
            thread = conversation_store.load_thread(selected)
            if thread:
                st.session_state.active_thread_id = selected
                st.session_state.chat_history = [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "citations": msg.metadata.get("citations", []),
                        "confidence": msg.metadata.get("confidence", 0.0),
                        "routing_info": msg.metadata.get("routing_info"),
                    }
                    for msg in thread.messages
                ]
                st.rerun()

        if st.button("🗑️ Delete Thread", type="secondary", width="stretch") and (
            conversation_store.delete_thread(selected)
        ):
            st.session_state.active_thread_id = None
            st.session_state.chat_history = []
            st.rerun()
