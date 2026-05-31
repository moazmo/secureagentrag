"""Self-query retrieval — extract structured metadata filters from natural language.

When a user asks "What did the engineering team say about risk in Q1 2024?",
self-query extracts:
- roles contains "engineer"
- source_file matches a date-pattern (if available)
- sensitivity_level (if implied)

These filters are merged with the RBAC filter and passed to Qdrant so retrieval
is scoped before embedding search runs, reducing noise and improving precision.

The extraction is done by a small local LLM prompt (cheap, fast) and falls back
to no filtering if parsing fails.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from core.agents.router import call_llm_async
from utils.logging import get_logger

logger = get_logger(__name__)

# SECURITY: the self-query extractor must NEVER surface access-control fields
# (org_id / roles / sensitivity_level). Those come exclusively from the
# authenticated UserContext and are applied by the RBAC filter. Letting the LLM
# emit them from untrusted query text would allow a crafted question to shift
# the tenant/clearance scope. Only content-descriptive fields (filename, date)
# are safe to infer here.
_SELF_QUERY_PROMPT = (
    "You are a metadata filter extractor. Given a user question, identify any "
    "constraints that could be expressed as document metadata filters.\n\n"
    "Available filter fields:\n"
    "- source_file: exact filename if mentioned (e.g., 'report.pdf')\n"
    "- date_after: ISO date if the query asks for documents after a date\n"
    "- date_before: ISO date if the query asks for documents before a date\n\n"
    "Rules:\n"
    "1. Only include filters that are EXPLICITLY or STRONGLY implied by the query.\n"
    "2. If no filters can be extracted, return an empty object {{}}.\n"
    "3. NEVER guess filenames or dates that are not in the query.\n"
    "4. Respond with VALID JSON only — no markdown, no explanation.\n\n"
    "Question: {query}\n\n"
    "Filters (JSON):"
)


async def extract_self_query_filters(
    query: str,
    *,
    sensitivity_level: str = "low",
    prefer_cloud: bool = False,
) -> dict[str, Any]:
    """Extract structured metadata filters from a natural language query.

    Falls back to an empty dict on any parsing failure so retrieval still runs.

    Args:
        query: User's natural language query.
        sensitivity_level: Passed to the inference router.
        prefer_cloud: User routing preference.

    Returns:
        Dict of filter field → value. Empty dict if nothing extractable.
    """
    try:
        raw = await call_llm_async(
            _SELF_QUERY_PROMPT.format(query=query),
            system_prompt="You extract structured metadata filters from questions. Output valid JSON only.",
            sensitivity_level=sensitivity_level,
            prefer_cloud=prefer_cloud,
        )
        # Strip markdown code fences if the model wrapped JSON in ```json ... ```
        cleaned = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        filters = json.loads(cleaned)
        if not isinstance(filters, dict):
            logger.warning("self_query_parse_not_dict", raw=raw[:200])
            return {}

        # Validate and coerce types
        validated: dict[str, Any] = {}
        for key, value in filters.items():
            if value is None or value == "":
                continue
            if key in ("date_after", "date_before"):
                # Try to parse as ISO date; skip if invalid
                try:
                    datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                    validated[key] = str(value)
                except ValueError:
                    logger.debug("self_query_invalid_date", key=key, value=value)
                    continue
            elif key == "roles" and isinstance(value, list):
                validated[key] = [str(r) for r in value if r]
            else:
                validated[key] = str(value)

        if validated:
            logger.info("self_query_filters_extracted", filters=list(validated.keys()))
        return validated

    except json.JSONDecodeError as exc:
        logger.warning(
            "self_query_json_parse_failed", error=str(exc), raw=raw[:200] if "raw" in dir() else ""
        )
        return {}
    except Exception as exc:
        logger.warning("self_query_extraction_failed", error=str(exc))
        return {}


def build_qdrant_filter_conditions(filters: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert self-query filter dict to Qdrant condition descriptors.

    These descriptors are consumed by ``QdrantManager.build_combined_filter``
    to produce actual ``qdrant_client.models.Filter`` objects.

    Args:
        filters: Output from ``extract_self_query_filters``.

    Returns:
        List of condition dicts with ``key`` and ``match``/``range`` info.
    """
    from qdrant_client import models

    # Access-control fields are never honoured from self-query output, even if a
    # jailbroken model emits them — they belong to the authenticated UserContext
    # and the RBAC filter. Defence in depth on top of the prompt restriction.
    _FORBIDDEN = {"org_id", "roles", "sensitivity_level", "sensitivity_level_int", "clearance"}

    conditions: list[dict[str, Any]] = []
    for key, value in filters.items():
        if key in _FORBIDDEN:
            logger.warning("self_query_dropped_forbidden_field", field=key)
            continue
        if key == "source_file":
            conditions.append({"key": "source_file", "match": models.MatchValue(value=value)})
        elif key == "date_after":
            from datetime import datetime

            ts = datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            conditions.append(
                {
                    "key": "ingested_at",
                    "range": models.Range(gte=ts),
                }
            )
        elif key == "date_before":
            from datetime import datetime

            ts = datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            conditions.append(
                {
                    "key": "ingested_at",
                    "range": models.Range(lte=ts),
                }
            )
    return conditions
