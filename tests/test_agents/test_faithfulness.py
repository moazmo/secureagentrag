"""Tests for the NLI-based citation faithfulness gate."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

import core.agents.faithfulness as faith_mod
from config.settings import settings
from core.agents.faithfulness import (
    _cited_indices,
    _parse_yes_no,
    _split_sentences,
    check_faithfulness,
)


def _doc(idx: int, text: str) -> dict:
    return {
        "doc_id": f"d{idx}",
        "text": text,
        "score": 0.9,
        "relevant": True,
        "metadata": {"source_file": f"src{idx}.txt", "page_number": 1, "sensitivity_level": "low"},
    }


def _state(generation: str, docs: list[dict]) -> dict:
    return {
        "generation": generation,
        "citations": [],
        "relevant_documents": docs,
        "documents": docs,
        "query_sensitivity": "low",
        "prefer_cloud": False,
    }


class TestUnits:
    def test_split_sentences_basic(self):
        out = _split_sentences("First claim [1]. Second claim [2]. Third with no cite.")
        assert len(out) == 3

    def test_split_sentences_strips_think_blocks(self):
        out = _split_sentences("<think>noise</think>Real claim [1].")
        assert out == ["Real claim [1]."]

    def test_cited_indices_handles_both_formats(self):
        assert _cited_indices("Mixed [1] and [[3]] sources [2].") == [3, 1, 2] or sorted(
            _cited_indices("Mixed [1] and [[3]] sources [2].")
        ) == [1, 2, 3]

    def test_cited_indices_skips_markdown_links(self):
        # `[text](url)` must not be picked up
        assert _cited_indices("See [docs](https://x) and [1].") == [1]

    def test_parse_yes_no_strict(self):
        assert _parse_yes_no("yes") is True
        assert _parse_yes_no("Yes.") is True
        assert _parse_yes_no("yes — clearly supported") is True
        assert _parse_yes_no("no") is False
        assert _parse_yes_no("maybe") is False
        assert _parse_yes_no("") is False
        # Reasoning block must be ignored, classification on the cleaned tail
        assert _parse_yes_no("<think>weighing</think>no") is False


class TestGateDisabled:
    def test_passthrough_when_disabled(self):
        state = _state("Some claim [1].", [_doc(1, "support text")])
        with patch.object(settings, "faithfulness_gate_enabled", False):
            result = asyncio.run(check_faithfulness(state))
        assert result["faithfulness_ratio"] == 1.0
        assert result["faithfulness_unsupported"] == []
        # Disabled gate must not rewrite the generation
        assert "generation" not in result
        assert result["audit_trail"][0]["action"] == "skip"
        assert result["audit_trail"][0]["reason"] == "disabled"


class TestGateEnabled:
    def test_all_supported_keeps_text_and_ratio_1(self):
        docs = [_doc(1, "Cats are mammals."), _doc(2, "Dogs bark.")]
        state = _state("Cats are mammals [1]. Dogs bark [2].", docs)

        async def _fake_llm(prompt, **_kwargs):
            return "yes"

        with (
            patch.object(settings, "faithfulness_gate_enabled", True),
            patch.object(settings, "faithfulness_gate_mode", "flag"),
            patch.object(faith_mod, "call_llm_async", _fake_llm),
        ):
            result = asyncio.run(check_faithfulness(state))

        assert result["faithfulness_ratio"] == 1.0
        assert result["faithfulness_unsupported"] == []
        # No annotation should appear on a clean answer
        assert "*[unsupported]*" not in result["generation"]

    def test_flag_mode_marks_unsupported_sentences(self):
        docs = [_doc(1, "Cats are mammals."), _doc(2, "Dogs bark.")]
        state = _state("Cats are mammals [1]. Cats can fly [2].", docs)

        async def _fake_llm(prompt, **_kwargs):
            # The second claim is the fabricated one — model says no.
            if "can fly" in prompt:
                return "no"
            return "yes"

        with (
            patch.object(settings, "faithfulness_gate_enabled", True),
            patch.object(settings, "faithfulness_gate_mode", "flag"),
            patch.object(faith_mod, "call_llm_async", _fake_llm),
        ):
            result = asyncio.run(check_faithfulness(state))

        assert result["faithfulness_ratio"] == 0.5
        assert len(result["faithfulness_unsupported"]) == 1
        assert "Cats can fly" in result["faithfulness_unsupported"][0]["sentence"]
        assert "*[unsupported]*" in result["generation"]
        # The supported sentence must still be present and unmarked.
        assert "Cats are mammals [1]." in result["generation"]
        audit = result["audit_trail"][0]
        assert audit["unsupported"] == 1
        assert audit["below_threshold"] is True  # 0.5 < default 0.7

    def test_drop_mode_removes_unsupported_sentences(self):
        docs = [_doc(1, "Cats are mammals."), _doc(2, "Dogs bark.")]
        state = _state("Cats are mammals [1]. Cats can fly [2].", docs)

        async def _fake_llm(prompt, **_kwargs):
            return "no" if "can fly" in prompt else "yes"

        with (
            patch.object(settings, "faithfulness_gate_enabled", True),
            patch.object(settings, "faithfulness_gate_mode", "drop"),
            patch.object(faith_mod, "call_llm_async", _fake_llm),
        ):
            result = asyncio.run(check_faithfulness(state))

        assert result["faithfulness_ratio"] == 0.5
        # Dropped sentence must be gone.
        assert "Cats can fly" not in result["generation"]
        assert "Cats are mammals [1]." in result["generation"]
        # No flag marker in drop mode
        assert "*[unsupported]*" not in result["generation"]

    def test_drop_mode_all_unsupported_refuses(self):
        docs = [_doc(1, "Irrelevant text.")]
        state = _state("Made-up claim [1].", docs)

        async def _fake_llm(prompt, **_kwargs):
            return "no"

        with (
            patch.object(settings, "faithfulness_gate_enabled", True),
            patch.object(settings, "faithfulness_gate_mode", "drop"),
            patch.object(faith_mod, "call_llm_async", _fake_llm),
        ):
            result = asyncio.run(check_faithfulness(state))

        assert result["faithfulness_ratio"] == 0.0
        assert "could not find sentence-level support" in result["generation"].lower()

    def test_no_cited_sentences_returns_full_credit(self):
        # Refusal-style answers carry no `[N]` markers and must not be
        # penalised by the gate.
        docs = [_doc(1, "anything")]
        state = _state("I cannot answer that question.", docs)

        async def _fake_llm(prompt, **_kwargs):
            raise AssertionError("LLM must not be called when no cited sentences")

        with (
            patch.object(settings, "faithfulness_gate_enabled", True),
            patch.object(faith_mod, "call_llm_async", _fake_llm),
        ):
            result = asyncio.run(check_faithfulness(state))

        assert result["faithfulness_ratio"] == 1.0
        assert result["faithfulness_unsupported"] == []
        assert result["audit_trail"][0]["reason"] == "no_cited_sentences"

    def test_out_of_range_cite_marked_unsupported(self):
        # `[5]` is fabricated — only one document exists.
        docs = [_doc(1, "Cats are mammals.")]
        state = _state("Cats are mammals [5].", docs)

        async def _fake_llm(prompt, **_kwargs):
            raise AssertionError("LLM must not be called when cite is out of range")

        with (
            patch.object(settings, "faithfulness_gate_enabled", True),
            patch.object(settings, "faithfulness_gate_mode", "flag"),
            patch.object(faith_mod, "call_llm_async", _fake_llm),
        ):
            result = asyncio.run(check_faithfulness(state))

        assert result["faithfulness_ratio"] == 0.0
        assert result["faithfulness_unsupported"][0]["verdict"] == "no_cited_index"
        assert "*[unsupported]*" in result["generation"]

    def test_llm_error_fails_open(self):
        # Transient errors must not silently drop the answer.
        docs = [_doc(1, "Cats are mammals.")]
        state = _state("Cats are mammals [1].", docs)

        async def _fake_llm(prompt, **_kwargs):
            raise RuntimeError("ollama down")

        with (
            patch.object(settings, "faithfulness_gate_enabled", True),
            patch.object(faith_mod, "call_llm_async", _fake_llm),
        ):
            result = asyncio.run(check_faithfulness(state))

        assert result["faithfulness_ratio"] == 1.0
        assert result["faithfulness_unsupported"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
