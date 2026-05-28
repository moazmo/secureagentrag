"""Deploy the SecureAgentRAG backend to its Hugging Face Space.

Usage::

    # one-time per session (token from .env)
    uv run python scripts/deploy_hf_space.py

    # dry-run (lists what would upload, sets no secrets)
    uv run python scripts/deploy_hf_space.py --dry-run

The script:

1. Pushes secrets (Qdrant URL + key, Groq key) to the Space settings
   panel via the HF Hub API. Secrets are NOT included in the build context.
2. Builds an HF-flavored README.md with the required YAML frontmatter
   (sdk=docker, app_port=7860).
3. Uses ``HfApi.upload_folder`` to push the source tree to the Space repo,
   filtering out everything in ``.dockerignore`` plus a few extras.
4. Renames ``Dockerfile.hf`` to ``Dockerfile`` on the Space side so HF picks
   it up as the build entrypoint.
5. Polls the Space runtime until it reaches ``RUNNING`` (or fails). On
   success, prints the public URL and runs a single ``/healthz`` smoke
   probe.

See ``launch-plan/04-hf-space-deploy.md`` for the broader plan.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
SPACE_REPO_ID = "LeomordKaly/secureagentrag-api"
SPACE_HOST = "LeomordKaly-secureagentrag-api.hf.space"
HF_README_BODY = """\
---
title: SecureAgentRAG API
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Privacy-first multi-agent RAG (BYOK demo)
---

# SecureAgentRAG API

Production backend for the [SecureAgentRAG](https://github.com/moazmo/secureagentrag) public demo.

- **Frontend:** https://secureagentrag-web.vercel.app
- **Source:** https://github.com/moazmo/secureagentrag (branch `deploy/prod-launch`)
- **License:** MIT

This Space hosts the FastAPI surface only. The Streamlit UI on `main`
remains for local development; recruiters interact with the platform via
the Next.js frontend deployed on Vercel.

## Mode

Runs in BYOK (Bring Your Own Key) mode:

- `POST /byok/chat` accepts visitor-supplied LLM credentials via headers
- `POST /byok/chat/stream` is the SSE variant that surfaces phase / token /
  blocked / final events for the live trace UI
- `GET  /byok/audit` returns the visitor's last 50 PII-redacted audit
  entries so the frontend can display the SHA-256 chain
- Owner-key fallback is throttled to 3 requests per IP per hour (Groq free
  tier protection) and consults `X-Forwarded-For` first so the throttle is
  not bypassed by HF's reverse proxy
- Each visitor gets a session-scoped Qdrant collection that auto-purges
  every 24 hours
- Phoenix instrumentation is hard-disabled (no third-party telemetry sees
  prompts or keys)
- Every audit-log persist runs through `utils.pii.redact` with regression
  tests for the Groq / OpenAI / Anthropic / HF / Vercel / Qdrant JWT shapes
- `SAR_ALLOW_CLOUD_FOR_HIGH=true` -- HIGH-sensitivity content is allowed to
  synthesize on the cloud LLM since this deploy has no local Ollama. The
  frontend renders a "sensitive: routed to cloud" badge on those answers.

## Demo personas

| Persona     | Clearance | Roles                              | Sees                                                                                  |
|-------------|-----------|------------------------------------|---------------------------------------------------------------------------------------|
| engineer    | 2 (med)   | engineering                        | public handbook, eng runbook, incident runbook, infra ADR, ML model card, NIST RMF    |
| compliance  | 3 (high)  | compliance, legal                  | public handbook, security policy, finance Q3, vendor MSA, ML model card, NIST, HR     |
| executive   | 3 (high)  | executive, compliance, engineering | union of the above                                                                    |

The RBAC filter is enforced at the Qdrant payload layer (`org_id` keyword +
`sensitivity_level_int` range + `roles` match-any). Chunks the persona is
not authorised to see are physically not returned, regardless of
cosine-similarity score.

## Endpoints

| Path                    | Purpose                                                       |
|-------------------------|---------------------------------------------------------------|
| `GET  /healthz`         | Liveness probe (used by GitHub Actions keepalive cron)        |
| `GET  /readyz`          | Readiness -- pings Qdrant Cloud + Groq (Ollama skipped here)  |
| `POST /byok/chat`       | Public-demo chat (BYOK or throttled owner-key)                |
| `POST /byok/chat/stream`| SSE variant -- emits phase / token / blocked / final events  |
| `GET  /byok/audit`      | Session-scoped audit export (PII redacted, SHA-256 chained)   |
| `POST /query`           | Authenticated JWT endpoint (dev / staging compat)             |

## Operator notes

- 600+ tests passing on the source repo at the commit pinned in
  `private/roadmap.md`.
- Built from `Dockerfile.hf` in the source tree -- this Space copy is
  renamed to `Dockerfile` so HF picks it up automatically.
- CPU Basic hardware (2 vCPU, 16 GB RAM). Cold cross-encoder load adds
  ~5 s to the first request after wake; subsequent queries answer in
  <1 s end-to-end against the Vercel frontend.
"""

# Files / dirs to ship to the Space. HfApi.upload_folder uses .gitignore-style
# globs; we mirror the .dockerignore intent so the Space repo stays minimal.
SPACE_ALLOW_PATTERNS = [
    "config/**",
    "core/**",
    "inference/**",
    "interfaces/**",
    "retrieval/**",
    "ingestion/**",
    "utils/**",
    "evaluation/calibration.json",
    "evaluation/__init__.py",
    "pyproject.toml",
    "uv.lock",
    "Dockerfile.hf",
    # README on the Space is generated below from HF_README_BODY; the source
    # README is intentionally excluded to avoid wiping the Space's YAML
    # frontmatter on next push.
]

SPACE_IGNORE_PATTERNS = [
    "*.pyc",
    "*__pycache__*",
    "tests/**",
    "app/**",
    "launch-plan/**",
    "private/**",
    "audit_logs/**",
    "conversations/**",
    "checkpoints/**",
    "data/**",
    ".venv/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    ".coverage",
    ".env",
    ".env.example",
    "scripts/deploy_hf_space.py",  # the deploy script itself doesn't ship
    "evaluation/nightly.py",
    "evaluation/golden_set.jsonl",
    "evaluation/baseline.json",
    "evaluation/benchmarks/**",
    "evaluation/nist_rerank_gold.jsonl",
]


def _push_secrets(api: HfApi, dry_run: bool = False) -> None:
    """Mirror local SAR_* env vars into the Space secrets panel."""
    needed = {
        "SAR_QDRANT_URL": os.environ.get("SAR_QDRANT_CLOUD_URL"),
        "SAR_QDRANT_API_KEY": os.environ.get("SAR_QDRANT_CLOUD_API_KEY"),
        "SAR_GROQ_API_KEY": os.environ.get("SAR_GROQ_API_KEY"),
    }
    missing = [k for k, v in needed.items() if not v]
    if missing:
        raise SystemExit(
            f"Missing env vars in .env: {missing}. "
            "Phase 1 smokes must complete before phase 3."
        )
    for name, value in needed.items():
        # Truncate so logs never echo a full key.
        masked = (value[:6] + "..." + value[-4:]) if value else ""
        print(f"  secret {name} = {masked}")
        if not dry_run:
            api.add_space_secret(repo_id=SPACE_REPO_ID, key=name, value=value)


def _stage_readme(staging_dir: Path) -> Path:
    """Write the HF-flavored README.md into a staging dir."""
    target = staging_dir / "README.md"
    target.write_text(HF_README_BODY, encoding="utf-8")
    return target


def _stage_dockerfile(staging_dir: Path) -> Path:
    """Copy Dockerfile.hf to ``staging_dir/Dockerfile`` so HF picks it up."""
    source = ROOT / "Dockerfile.hf"
    if not source.exists():
        raise SystemExit("Dockerfile.hf not found at repo root")
    target = staging_dir / "Dockerfile"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _poll_until_running(api: HfApi, timeout_s: int = 900) -> str:
    """Block until the Space reports RUNNING or a terminal error."""
    deadline = time.monotonic() + timeout_s
    last_stage: str | None = None
    while time.monotonic() < deadline:
        rt = api.get_space_runtime(SPACE_REPO_ID)
        if rt.stage != last_stage:
            print(f"  [{int(time.monotonic())}s] stage = {rt.stage}")
            last_stage = rt.stage
        if rt.stage == "RUNNING":
            return rt.stage
        if rt.stage in {
            "BUILD_ERROR",
            "RUNTIME_ERROR",
            "APP_STARTING_ERROR",
            "CONFIG_ERROR",
            "PAUSED",
        }:
            raise SystemExit(f"Space build failed: stage={rt.stage}")
        time.sleep(10)
    raise SystemExit(f"Space did not reach RUNNING within {timeout_s}s")


def _smoke_health(url_base: str) -> None:
    """Run a single /healthz probe from this machine."""
    import httpx

    url = f"https://{url_base}/healthz"
    print(f"  curl {url}")
    r = httpx.get(url, timeout=60)
    print(f"  -> HTTP {r.status_code} {r.text[:200]}")
    if r.status_code != 200:
        raise SystemExit(f"healthz returned {r.status_code}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be uploaded; skip secret writes and the upload itself.",
    )
    parser.add_argument(
        "--skip-secrets",
        action="store_true",
        help="Skip writing secrets to the Space panel (use if they already exist).",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set in .env (phase 1a smoke is the source)")

    api = HfApi(token=token)
    who = api.whoami()
    print(f"deploy_hf_space.py -> {SPACE_REPO_ID} as {who.get('name')}")

    if not args.skip_secrets:
        print("== secrets ==")
        _push_secrets(api, dry_run=args.dry_run)

    print("== staging README + Dockerfile ==")
    # The HF Space upload accepts a single folder root. We stage the two
    # generated files into a temp dir alongside the repo root so we can
    # selectively allow them without disturbing the canonical source.
    staging_root = ROOT / ".hf-deploy-staging"
    staging_root.mkdir(exist_ok=True)
    readme_path = _stage_readme(staging_root)
    dockerfile_path = _stage_dockerfile(staging_root)
    print(f"  README:     {readme_path.relative_to(ROOT)}")
    print(f"  Dockerfile: {dockerfile_path.relative_to(ROOT)}")

    if args.dry_run:
        print(
            "== dry-run summary ==\n"
            + textwrap.indent(
                "allow patterns:\n  - " + "\n  - ".join(SPACE_ALLOW_PATTERNS),
                "  ",
            )
        )
        return 0

    print(f"== upload source -> https://huggingface.co/spaces/{SPACE_REPO_ID} ==")
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        commit_message="deploy: phase 3 BYOK backend (Dockerfile.hf, FastAPI on 7860)",
        allow_patterns=SPACE_ALLOW_PATTERNS,
        ignore_patterns=SPACE_IGNORE_PATTERNS,
    )

    print("== upload staging README + Dockerfile ==")
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        commit_message="deploy: HF-flavored README with YAML frontmatter",
    )
    api.upload_file(
        path_or_fileobj=str(dockerfile_path),
        path_in_repo="Dockerfile",
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        commit_message="deploy: rename Dockerfile.hf -> Dockerfile on the Space side",
    )

    print("== poll runtime ==")
    _poll_until_running(api)

    print("== smoke /healthz from this machine ==")
    _smoke_health(SPACE_HOST)

    print()
    print(f"  Live URL: https://{SPACE_HOST}")
    print("  done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
