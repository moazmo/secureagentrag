"""Unit tests for the structured-extraction core (Tier X / ADR-041)."""

from __future__ import annotations

import pytest

from core.extraction import (
    ExtractionField,
    build_extraction_prompt,
    normalise_fields,
    parse_extraction_response,
)


def _fields() -> list[ExtractionField]:
    return [
        ExtractionField("seller", "string", "the seller / vendor name"),
        ExtractionField("total", "number", "grand total incl. VAT"),
        ExtractionField("paid", "boolean", "whether the invoice is paid"),
    ]


def test_normalise_drops_nameless_and_caps():
    raw = [{"name": "a"}, {"type": "string"}, {"name": ""}, {"name": "b", "type": "number"}]
    out = normalise_fields(raw)
    assert [f.name for f in out] == ["a", "b"]
    assert out[1].safe_type() == "number"


def test_normalise_empty_raises():
    with pytest.raises(ValueError):
        normalise_fields([{"type": "string"}])  # no usable name


def test_safe_type_falls_back():
    assert ExtractionField("x", "weird").safe_type() == "string"
    assert ExtractionField("x", "integer").safe_type() == "integer"


def test_build_prompt_lists_keys_and_caps_text():
    long = "Z" * 50_000
    p = build_extraction_prompt(long, _fields())
    assert '"seller"' in p and '"total"' in p and '"paid"' in p
    # Text is capped well under the raw length.
    assert len(p) < 20_000


def test_parse_clean_json_coerces_types():
    raw = '{"seller": "ACME Co", "total": "1,234.50", "paid": "yes"}'
    out = parse_extraction_response(raw, _fields())
    assert out["seller"] == "ACME Co"
    assert out["total"] == 1234.5
    assert out["paid"] is True


def test_parse_strips_fences_and_think_blocks():
    raw = '<think>reasoning</think>\n```json\n{"seller": "X", "total": 5, "paid": false}\n```'
    out = parse_extraction_response(raw, _fields())
    assert out["seller"] == "X"
    assert out["total"] == 5.0
    assert out["paid"] is False


def test_parse_missing_fields_become_null_and_extra_keys_dropped():
    raw = '{"seller": "Y", "garbage": 1}'
    out = parse_extraction_response(raw, _fields())
    assert set(out.keys()) == {"seller", "total", "paid"}
    assert out["total"] is None and out["paid"] is None


def test_parse_garbage_returns_all_null_shape():
    out = parse_extraction_response("not json at all", _fields())
    assert out == {"seller": None, "total": None, "paid": None}


def test_parse_finds_json_inside_prose():
    raw = 'Here is the result: {"seller":"Z","total":9,"paid":true} hope it helps'
    out = parse_extraction_response(raw, _fields())
    assert out["seller"] == "Z" and out["total"] == 9.0 and out["paid"] is True
