"""Integration tests for the FastAPI REST surface.

Uses FastAPI TestClient to exercise all endpoints without starting a server.
All external dependencies (RAG pipeline, health checks, ingestion, audit) are
mocked so the suite runs without Ollama or Qdrant.
"""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from interfaces.api import mint_dev_token


def _make_token(user_id: str, roles: list[str] | None = None, **kwargs):
    """Helper to build a bearer token for a given user context."""
    payload = {
        "user_id": user_id,
        "org_id": kwargs.get("org_id", "acme_corp"),
        "roles": roles or ["viewer"],
        "clearance_level": kwargs.get("clearance_level", 1),
    }
    return mint_dev_token(payload)


@pytest.fixture()
def client():
    """Yield a FastAPI TestClient with fresh state."""
    # Import here so we can conditionally skip if fastapi is missing
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from interfaces.api import app, rate_limiter

    # Reset rate limiter buckets so tests don't interfere
    rate_limiter._buckets.clear()

    with TestClient(app) as c:
        yield c


class TestHealthz:
    """Tests for the liveness probe."""

    def test_returns_ok(self, client):
        """/healthz requires no auth and returns a simple status."""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestReadyz:
    """Tests for the readiness probe."""

    @patch("interfaces.api.run_health_checks", new_callable=AsyncMock)
    def test_returns_200_when_healthy(self, mock_health, client):
        """All services healthy → 200 with overall_healthy=true."""
        report = MagicMock()
        report.overall_healthy = True
        report.to_dict.return_value = {"overall_healthy": True, "services": []}
        mock_health.return_value = report

        response = client.get("/readyz")

        assert response.status_code == 200
        assert response.json()["overall_healthy"] is True

    @patch("interfaces.api.run_health_checks", new_callable=AsyncMock)
    def test_returns_503_when_unhealthy(self, mock_health, client):
        """Any required service down → 503 with overall_healthy=false."""
        report = MagicMock()
        report.overall_healthy = False
        report.to_dict.return_value = {"overall_healthy": False, "services": []}
        mock_health.return_value = report

        response = client.get("/readyz")

        assert response.status_code == 503
        assert response.json()["overall_healthy"] is False


class TestQuery:
    """Tests for POST /query."""

    @patch("interfaces.api.run_rag_pipeline", new_callable=AsyncMock)
    def test_success(self, mock_pipeline, client):
        """A valid token + matching user_id yields a QueryResponse."""
        mock_pipeline.return_value = {
            "generation": "The answer is 42.",
            "citations": [],
            "confidence_score": 0.95,
            "needs_human_review": False,
            "query_type": "simple",
            "retry_count": 0,
            "security_passed": True,
            "guardrails_passed": True,
            "synth_provider": "ollama",
            "synth_model": "qwen3:8b",
            "synth_latency_ms": 1000.0,
            "synth_usage": {},
        }
        token = _make_token("alice", ["viewer"])

        response = client.post(
            "/query",
            json={"query": "What is the answer?", "user_id": "alice"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "The answer is 42."
        assert body["confidence_score"] == 0.95
        assert body["blocked"] is False
        mock_pipeline.assert_awaited_once()

    def test_missing_auth(self, client):
        """No Authorization header → 401."""
        response = client.post("/query", json={"query": "What?", "user_id": "alice"})
        assert response.status_code == 401

    def test_invalid_token(self, client):
        """A malformed token → 401."""
        response = client.post(
            "/query",
            json={"query": "What?", "user_id": "alice"},
            headers={"Authorization": "Bearer not-base64!!!"},
        )
        assert response.status_code == 401

    def test_user_id_mismatch(self, client):
        """Token user_id must match body user_id → 403 on mismatch."""
        token = _make_token("alice")
        response = client.post(
            "/query",
            json={"query": "What?", "user_id": "bob"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403
        assert "mismatch" in response.json()["detail"]

    @patch("interfaces.api.run_rag_pipeline", new_callable=AsyncMock)
    def test_rate_limit(self, mock_pipeline, client):
        """Excessive requests from the same user trigger 429."""
        mock_pipeline.return_value = {
            "generation": "ans",
            "citations": [],
            "confidence_score": 0.0,
            "needs_human_review": False,
            "query_type": "simple",
            "retry_count": 0,
            "security_passed": True,
            "guardrails_passed": True,
            "synth_provider": "",
            "synth_model": "",
            "synth_latency_ms": 0.0,
            "synth_usage": {},
        }
        token = _make_token("rate_test", ["viewer"])
        # Exhaust the burst bucket
        for _ in range(20):
            client.post(
                "/query",
                json={"query": "What?", "user_id": "rate_test"},
                headers={"Authorization": f"Bearer {token}"},
            )

        response = client.post(
            "/query",
            json={"query": "What?", "user_id": "rate_test"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 429
        assert "rate limit" in response.json()["detail"].lower()


class TestIngest:
    """Tests for POST /ingest."""

    @patch("ingestion.pipeline.IngestionPipeline")
    @patch("core.agents.retriever._get_hybrid_searcher")
    def test_success(self, mock_get_searcher, mock_pipeline_cls, client):
        """A user with the 'user' role can ingest documents."""
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.file_path = "/docs/report.pdf"
        mock_result.status = "success"
        mock_result.num_chunks = 5
        mock_result.point_ids = ["p1", "p2"]
        mock_result.errors = []
        mock_result.processing_time_seconds = 1.23
        mock_pipeline.ingest_document = AsyncMock(return_value=mock_result)
        mock_pipeline_cls.return_value = mock_pipeline

        searcher = MagicMock()
        searcher._qdrant = MagicMock()
        searcher._embeddings = MagicMock()
        searcher._bm25_index = MagicMock()
        mock_get_searcher.return_value = searcher

        token = _make_token("ingest_user", ["user"])
        response = client.post(
            "/ingest",
            json={
                "file_path": "/docs/report.pdf",
                "user_id": "ingest_user",
                "sensitivity_level": "medium",
            },
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["num_chunks"] == 5
        mock_pipeline.ingest_document.assert_awaited_once()

    def test_missing_role(self, client):
        """A viewer-only user cannot access /ingest → 403."""
        token = _make_token("viewer_only", ["viewer"])
        response = client.post(
            "/ingest",
            json={"file_path": "/docs/x.pdf", "user_id": "viewer_only"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403

    def test_user_id_mismatch(self, client):
        """Token user_id must match body user_id → 403."""
        token = _make_token("alice", ["user"])
        response = client.post(
            "/ingest",
            json={"file_path": "/docs/x.pdf", "user_id": "bob"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403


class TestAudit:
    """Tests for /audit and /audit/verify."""

    @patch("interfaces.api.audit_logger")
    def test_list_requires_admin(self, mock_audit, client):
        """Only admin role can list audit entries."""
        mock_entry = MagicMock()
        mock_entry.model_dump.return_value = {"action": "upload", "user_id": "u1"}
        mock_audit.get_entries.return_value = [mock_entry]

        token = _make_token("admin_user", ["admin"])
        response = client.get(
            "/audit",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert len(body["items"]) == 1
        mock_audit.get_entries.assert_called_once()

    def test_list_denied_for_non_admin(self, client):
        """A non-admin user receives 403 on /audit."""
        token = _make_token("viewer", ["viewer"])
        response = client.get(
            "/audit",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403

    @patch("interfaces.api.audit_logger")
    def test_verify_requires_admin(self, mock_audit, client):
        """Only admin can verify the hash-chain."""
        mock_audit.verify_chain.return_value = {"valid": True, "entries": 10}

        token = _make_token("admin_user", ["admin"])
        response = client.post(
            "/audit/verify",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json()["valid"] is True
        mock_audit.verify_chain.assert_called_once()

    def test_verify_denied_for_non_admin(self, client):
        """A non-admin user receives 403 on /audit/verify."""
        token = _make_token("viewer", ["viewer"])
        response = client.post(
            "/audit/verify",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403


class TestMintDevToken:
    """Tests for the token helper."""

    def test_roundtrip(self):
        """A minted token decodes back to the original payload."""
        from config.settings import settings

        # Force legacy unsigned mode for this specific test so the base64
        # roundtrip is meaningful.
        with patch.object(settings, "jwt_secret", None):
            payload = {"user_id": "test", "roles": ["admin"], "clearance_level": 3}
            token = mint_dev_token(payload)
            decoded = json.loads(base64.b64decode(token).decode("utf-8"))
            assert decoded == payload


class TestJWTAuth:
    """Signed-JWT path through the FastAPI surface.

    When SAR_JWT_SECRET is set, /token mints a real JWT and /query verifies
    signature, expiry, and claims.
    """

    @patch("interfaces.api.run_rag_pipeline", new_callable=AsyncMock)
    def test_signed_jwt_accepted_on_query(self, mock_pipeline, client):
        from config.settings import settings

        mock_pipeline.return_value = {
            "generation": "ok",
            "citations": [],
            "confidence_score": 0.9,
            "needs_human_review": False,
            "query_type": "simple",
            "retry_count": 0,
            "security_passed": True,
            "guardrails_passed": True,
            "synth_provider": "ollama",
            "synth_model": "qwen3:8b",
            "synth_latency_ms": 1.0,
            "synth_usage": {},
        }

        with patch.object(settings, "jwt_secret", "TEST-SECRET-XYZ"):
            tok = client.post(
                "/token",
                json={"user_id": "alice", "org_id": "acme", "roles": ["viewer"]},
            )
            assert tok.status_code == 200
            access = tok.json()["access_token"]
            assert access.count(".") == 2  # header.payload.signature

            response = client.post(
                "/query",
                json={"query": "hi", "user_id": "alice"},
                headers={"Authorization": f"Bearer {access}"},
            )
        assert response.status_code == 200
        # The pipeline must have been called with a thread_id that carries
        # the jti so audit can correlate the call back to a token.
        call_kwargs = mock_pipeline.await_args.kwargs
        assert call_kwargs["thread_id"].startswith("api-alice-")
        assert call_kwargs["thread_id"] != "api-alice-unsigned"

    def test_token_endpoint_503_without_secret(self, client):
        from config.settings import settings

        with patch.object(settings, "jwt_secret", None):
            response = client.post(
                "/token",
                json={"user_id": "alice", "org_id": "acme", "roles": ["viewer"]},
            )
        assert response.status_code == 503

    def test_legacy_base64_still_works_when_secret_unset(self, client):
        """Backwards-compat path keeps existing smoke scripts running.

        When ``SAR_JWT_SECRET`` is unset, the base64(json) token shape is
        accepted (with a runtime warning) so the e2e harness and Streamlit
        demo do not break in dev. This test pins that contract.
        """
        from config.settings import settings

        with (
            patch.object(settings, "jwt_secret", None),
            patch("interfaces.api.run_rag_pipeline", new_callable=AsyncMock) as mock_pipeline,
        ):
            mock_pipeline.return_value = {
                "generation": "ok",
                "citations": [],
                "confidence_score": 0.0,
                "needs_human_review": False,
                "query_type": "simple",
                "retry_count": 0,
                "security_passed": True,
                "guardrails_passed": True,
                "synth_provider": "",
                "synth_model": "",
                "synth_latency_ms": 0.0,
                "synth_usage": {},
            }
            token = _make_token("legacy_user", ["viewer"])
            r = client.post(
                "/query",
                json={"query": "hi", "user_id": "legacy_user"},
                headers={"Authorization": f"Bearer {token}"},
            )
        assert r.status_code == 200
