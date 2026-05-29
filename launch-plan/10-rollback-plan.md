# 10 — Rollback Plan

If anything in production goes sideways, here is how to undo it without losing work.

## Rollback target

- **`main` is frozen at commit `56c8c98`.** This is the last known good Streamlit-only state. Tests green, lint green, format green. Always recoverable.

## Scenarios

### Scenario 1: HF Space build fails repeatedly

**Symptoms:** HF Space shows red "Build error" status. `huggingface.co/spaces/LeomordKaly/secureagentrag-api/logs` shows the failure reason.

**Action:**

1. Force-push the last good Space commit:
   ```bash
   cd hf-space-clone
   git reset --hard <last-good-sha>
   git push --force origin main
   ```
2. If no good commit exists yet, push the smoke-test "hello world" Dockerfile from phase 1
3. Investigate the build log, fix locally, push again
4. **Do not delete the Space.** Recreating loses the secrets settings.

### Scenario 2: Qdrant Cloud cluster down or capacity exceeded

**Symptoms:** Backend logs show `ConnectionError` or `qdrant_client.http.exceptions.ResponseHandlingException`. Vercel frontend shows persistent 500s.

**Action:**

1. Check Qdrant Cloud status page
2. If their side, wait
3. If our side (1 GB exceeded):
   - Trigger immediate session purge: `curl -X POST https://LeomordKaly-secureagentrag-api.hf.space/admin/purge-sessions`
   - Reduce `SAR_SESSION_COLLECTION_TTL_HOURS` from 24 to 6 temporarily
   - If still over capacity, delete all `documents_sess_*` collections and add a warning to the frontend
4. **Last resort:** switch the backend to self-hosted Qdrant inside the HF Space (data lost on restart but service stays up)

### Scenario 3: Vercel frontend broken

**Symptoms:** `secureagentrag.vercel.app` returns 500 or shows broken layout.

**Action:**

1. From Vercel dashboard → Deployments → roll back to previous deployment (one-click)
2. Investigate locally on `secureagentrag-web` repo
3. Push fix → Vercel auto-deploys
4. **Total time to recover:** under 60 seconds (rollback is instant)

### Scenario 4: BYOK key leaked into audit log

**This is a security incident, not a normal rollback.**

**Action:**

1. Take the HF Space offline immediately:
   - `huggingface.co/spaces/LeomordKaly/secureagentrag-api/settings` → Pause
2. Notify any user whose key was logged
3. Wipe the audit logs on the Space:
   - `huggingface-cli space-files LeomordKaly/secureagentrag-api delete /tmp/audit_logs --recursive`
4. Patch the redaction regex
5. Add a regression test
6. Re-enable

See `11-security-checklist.md` for prevention.

### Scenario 5: Groq owner key exceeded (30 RPM hit)

**Symptoms:** Frontend shows "owner key throttled" for many visitors.

**Action:**

1. This is by design. The owner-key throttle is intentional.
2. Drop the per-IP quota from 10/hour to 1/hour in `SAR_BYOK_OWNER_KEY_QUOTA_PER_HOUR`
3. Add a more aggressive "BYOK recommended" banner in the frontend
4. If sustained traffic warrants it, owner can add a CC to Groq for 10× rate limits (Groq does not gate the free tier with a CC, but adding one raises limits)

### Scenario 6: Hostinger landing page goes 500

**Action:** Re-upload the static HTML files via hPanel File Manager. Hostinger shared hosting rarely has runtime failures since the page is pure static.

### Scenario 7: Demo video has an embarrassing error

**Action:** Replace the YouTube video with a new upload (unlisted, same URL pattern). Update the iframe src in `landing/index.html`. Re-upload via hPanel.

### Scenario 8: Need to fully abandon the launch

If for any reason the launch needs to be scrapped:

1. Delete the HF Space: `huggingface-cli delete-space LeomordKaly/secureagentrag-api` (or via UI)
2. Delete the Qdrant Cloud cluster from `cloud.qdrant.io`
3. Delete the Vercel project from `vercel.com/dashboard`
4. Remove the landing page files from Hostinger `public_html/`
5. On the secureagentrag git repo:
   ```bash
   git checkout main
   git branch -D deploy/prod-launch              # destroys local
   git push origin --delete deploy/prod-launch   # destroys remote
   ```
6. **`main` is still at `56c8c98`. Project state restored.**

## Pre-launch sanity checks

Before any merge to `main`:

- [ ] All 487 existing tests pass on `deploy/prod-launch`
- [ ] New tests added for BYOK pass
- [ ] `ruff check .` clean
- [ ] `ruff format --check .` clean
- [ ] HF Space `/health` returns 200 from Egypt
- [ ] Vercel deploy succeeds and is reachable from Egypt
- [ ] BYOK key redaction regression test passes
- [ ] Per-IP throttle test passes
- [ ] Session purge test passes

If any check fails, fix on `deploy/prod-launch` before merging. Do not merge until green.
