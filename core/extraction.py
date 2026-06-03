"""Structured-data extraction: document text -> JSON against a field schema.

This is the *extraction mode* (Tier X) — the second face of the platform next to
RAG Q&A. Instead of "ask a question, get a cited answer," the caller supplies a
small **field schema** (name + type + description per field) and gets a single
validated JSON object back. No retrieval, no vector DB — just parse → one
``json_mode`` LLM call → validate.

It reuses the same inference router as the RAG pipeline, so the visitor's BYOK
key powers the call and the same sensitivity-routing applies (HIGH-sensitivity
docs stay local on a self-hosted deploy). Kept framework-free so it is unit
testable without FastAPI.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from utils.logging import get_logger

logger = get_logger(__name__)

# Bound the document text fed to the model so a long PDF cannot blow the
# token budget / rate limit. Extraction targets a handful of fields, so the
# salient content is almost always near the top of the document.
MAX_EXTRACTION_CHARS = 12_000

# Field types we coerce to. Anything else is treated as a string.
_ALLOWED_TYPES = frozenset({"string", "number", "integer", "boolean", "date"})


@dataclass(frozen=True)
class ExtractionField:
    """One field to pull out of a document.

    Attributes:
        name: JSON key to emit (e.g. ``"total_amount"``).
        type: One of ``string`` / ``number`` / ``integer`` / ``boolean`` /
            ``date``. Unknown types fall back to ``string``.
        description: Plain-language hint that tells the model what to look for
            (e.g. "the grand total including VAT, as a number").
    """

    name: str
    type: str = "string"
    description: str = ""

    def safe_type(self) -> str:
        t = (self.type or "string").lower().strip()
        return t if t in _ALLOWED_TYPES else "string"


def normalise_fields(raw_fields: list[dict]) -> list[ExtractionField]:
    """Coerce a list of raw field dicts into validated ``ExtractionField`` objects.

    Drops entries without a usable ``name``; caps the count so a caller cannot
    request hundreds of fields in one prompt. Raises ``ValueError`` when nothing
    usable remains.
    """
    out: list[ExtractionField] = []
    for f in raw_fields or []:
        if not isinstance(f, dict):
            continue
        name = str(f.get("name", "")).strip()
        if not name:
            continue
        out.append(
            ExtractionField(
                name=name,
                type=str(f.get("type", "string")),
                description=str(f.get("description", "")).strip(),
            )
        )
        if len(out) >= 25:  # hard cap — keep the prompt + output bounded
            break
    if not out:
        raise ValueError("no usable fields in the extraction schema")
    return out


def build_extraction_prompt(text: str, fields: list[ExtractionField]) -> str:
    """Build a strict JSON-only extraction prompt."""
    field_lines = "\n".join(
        f'- "{f.name}" ({f.safe_type()}): {f.description or "extract this field"}' for f in fields
    )
    keys = ", ".join(f'"{f.name}"' for f in fields)
    return (
        "You are a precise document data-extraction engine. Extract the fields "
        "below from the DOCUMENT and return a SINGLE valid JSON object — nothing "
        "else, no markdown fences, no commentary.\n\n"
        "RULES:\n"
        "1. Output exactly these keys and no others: " + keys + ".\n"
        "2. Use the field type as a hint. Numbers as JSON numbers, booleans as "
        "true/false, dates as ISO-8601 strings (YYYY-MM-DD) when possible.\n"
        "3. If a field is not present in the document, set its value to null. "
        "Do NOT invent values.\n"
        "4. Answer in the document's own language for free-text values "
        "(Arabic documents -> Arabic values).\n\n"
        f"FIELDS:\n{field_lines}\n\n"
        f"DOCUMENT:\n{text[:MAX_EXTRACTION_CHARS]}\n\n"
        "Return ONLY the JSON object:"
    )


def parse_extraction_response(raw: str, fields: list[ExtractionField]) -> dict:
    """Parse the model's JSON, keep only the requested keys, coerce types.

    Robust to a model that wraps the JSON in ``` fences or adds a ``<think>``
    preamble. Always returns a dict with **every** requested key present
    (missing -> ``None``), so the caller gets a stable shape.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()
    # Strip a leading ```json / ``` fence if present.
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    # Fall back to the first {...} block if there is still surrounding prose.
    if not cleaned.startswith("{"):
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        cleaned = m.group(0) if m else "{}"

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("extraction_json_parse_failed", preview=cleaned[:120])
        data = {}
    if not isinstance(data, dict):
        data = {}

    result: dict = {}
    for f in fields:
        result[f.name] = _coerce(data.get(f.name), f.safe_type())
    return result


def _coerce(value: object, typ: str):
    """Best-effort coerce a raw JSON value to the requested field type."""
    if value is None:
        return None
    try:
        if typ == "integer":
            return int(float(str(value).replace(",", "").strip()))
        if typ == "number":
            return float(str(value).replace(",", "").strip())
        if typ == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("true", "yes", "1", "نعم")
        # string / date — return as-is string
        return value if isinstance(value, (str, int, float, bool)) else str(value)
    except (ValueError, TypeError):
        return value  # keep the raw value rather than dropping data


async def extract_fields(
    text: str,
    fields: list[ExtractionField],
    *,
    prefer_cloud: bool = True,
    sensitivity_level: str = "low",
) -> dict:
    """Run one ``json_mode`` extraction call and return the validated result.

    Returns a dict: ``{"fields": {...}, "model": str, "provider": str,
    "latency_ms": float, "raw": str}``. Never raises on a bad LLM response —
    returns all-null fields so the caller always gets a stable shape.
    """
    from core.agents.router import call_llm_with_decision

    prompt = build_extraction_prompt(text, fields)
    raw, decision, response = await call_llm_with_decision(
        prompt,
        system_prompt="You output only valid JSON. No prose, no markdown fences.",
        sensitivity_level=sensitivity_level,
        prefer_cloud=prefer_cloud,
        json_mode=True,
    )
    parsed = parse_extraction_response(raw or "", fields)
    return {
        "fields": parsed,
        "model": decision.model if decision else "unknown",
        "provider": decision.provider if decision else "unknown",
        "latency_ms": response.latency_ms if response else 0.0,
        "raw": raw or "",
    }
