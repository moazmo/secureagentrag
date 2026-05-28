"""Tests for the /metrics exposition endpoint on the FastAPI surface."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("prometheus_client")


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from interfaces.api import app

    with TestClient(app) as c:
        yield c


def test_metrics_endpoint_serves_prometheus_exposition(client):
    """/metrics returns 200 in Prometheus text format with no auth."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    # HTTP-level metrics from prometheus-fastapi-instrumentator.
    assert "# HELP" in body
    assert "# TYPE" in body


def test_custom_metric_appears_after_pipeline_run(client):
    """A recorded pipeline run surfaces the custom RAG metric in /metrics."""
    from utils.metrics import record_pipeline_run

    record_pipeline_run(
        {
            "guardrails_passed": True,
            "security_passed": True,
            "synth_provider": "groq",
            "faithfulness_unsupported": [],
        },
        latency_ms=1000.0,
    )
    body = client.get("/metrics").text
    assert "rag_pipeline_requests_total" in body
    assert "inference_routed_by_provider_total" in body
