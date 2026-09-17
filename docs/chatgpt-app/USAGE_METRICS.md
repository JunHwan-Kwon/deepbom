# ChatGPT usage evidence

Administrators use `https://deepbom.org/web/admin.html` → **ChatGPT usage
evidence**. The existing administrator login protects the dashboard and
`GET /api/admin/usage?days=30&cohort=public`. No login is added for plugin users.

Choose 7, 30, or 90 days (UTC), public use or declared tests, then export a daily
CSV or a JSON evidence snapshot. JSON includes the measurement definitions and
limitations. The dashboard contains no browser keys, session IDs, model names,
model hashes, email addresses, or IP addresses. Historical use before deployment
cannot be reconstructed. This collector covers the ChatGPT widget and `/mcp`,
not CLI, local MCP, `/mcp/claude`, or the separate website analyzer.

## What the numbers mean

| Metric | Definition |
| --- | --- |
| MCP requests, errors, reports | Daily service counts for all callers, including bots, tests and retries; not unique analyses or people. |
| Analysis started/completed/failed | Opt-in widget runs; the first terminal outcome wins. A completion means the widget received a published result. |
| Terminal success rate | Completed / (completed + failed). Runs with no terminal event are excluded; it is not an overall completion rate. |
| Format share | Completed opt-in reports by ONNX, TFLite, GGUF, SafeTensors, Core ML, or ExecuTorch. |
| Visualization | Runs where a view was rendered, including the initial automatic view. Not proof a person viewed it. |
| Prepared exports | Once per export type per widget run, for SVG, PNG, Word ZIP, CycloneDX or SPDX. Not confirmed file downloads. |
| File handoffs | Runs where ChatGPT accepted a generated-file upload of the given type. Not confirmed downloads or rendered replies. |
| Estimated active browsers | Distinct consenting persistent browser keys with an analysis start in the selected period. Not verified people/accounts. |
| Returning browsers | Active in the selected period and observed on at least two UTC dates in the retained 90-day history. Same-day reloads alone do not qualify. |

The event/detail key is unique per widget run, so retries do not inflate it.
Reloads are new runs. Storage partitioning, multiple devices, cleared storage,
opt-outs, and forged clients affect estimates. With unavailable browser storage,
runs can contribute consented event counts but not unique/revisit counts.
These estimates cannot be extrapolated into total user counts without additional
evidence. Public posts should say, for example, “N consenting browser instances
and M completed opt-in analysis reports during [UTC dates].”

Mark your browser **This browser is for testing** before demo/reviewer tests.
It moves this key's retained usage to the test cohort. Unmarked test/reviewer
runs cannot be recognized automatically. Service counters always include tests.

## Consent and retention

Sharing is off by default. Opt-in covers the current widget run and subsequent
visits while the 90-day preference remains valid. No key is created or usage
request sent before opt-in. A random 32-byte key is stored locally; the database
stores a domain-separated HMAC using `SESSION_SECRET`. No fingerprinting or
host account ID is used. Tokens expire after 24 hours. Request bodies are bounded
to 8 KiB and only documented dimensions are accepted.

Unchecking stops future sharing. **Delete my usage** stops sharing and removes
all sessions/events for the retained key; it clears browser storage only after
the deletion succeeds. Clearing storage first loses the deletion handle. Public
service counters and already exported/published aggregate snapshots cannot be
attributed to that key and remain. A changed `SESSION_SECRET` also changes the
hash; keep it stable for the retention window when possible. After a required
secret rotation, old rows remain until automatic expiry and cannot be matched
by the new hash for deletion.

Events expire from queries after 90 days. A daily 00:17 UTC cron deletes expired
sessions with cascading event deletion and old service counters; physical
retention can extend by up to one day. Monitor scheduled invocation failures.
Cloudflare Time Travel/backups, if enabled, follow the provider's retention and
must not be restored into active collection without reapplying deletion controls.

## Rollout and rollback

The maintainer requested coordinated deployment of engine 1.102.0 and the
consented usage feature. Agent contract 1.0.1 is the deliberate privacy/transfer
change candidate prepared under
[`AGENT_CONTRACT_VERSIONING.md`](../AGENT_CONTRACT_VERSIONING.md). Its baseline
records `candidate_unsubmitted`, not platform approval. The previous contract
remains in Git history. Server deployment and registry publication do not update
the account-owned OpenAI submission or establish its approval.
The exact detected change and the completed verification are recorded in
[`usage-review-change.json`](usage-review-change.json). The portal import draft
is the repository-root `chatgpt-app-submission.json`; no automatic portal upload
or compliance attestation has been performed.

1. Run `npm run check:usage-metrics`, `npm run check:chatgpt-widget`,
   `npm run check:chatgpt-app-metadata`, and `npm run check:worker-config` using Node 24.
2. Before deploying, reconcile the updated privacy/data-flow description with
   any pending plugin review. The repository files do not update the portal.
3. For manual database maintenance, inspect pending migrations with
   `npx wrangler@4.131.1 d1 migrations list deepbom_auth --remote`.
   Apply the intended migration before enabling collection:
   `npx wrangler@4.131.1 d1 migrations apply deepbom_auth --remote`.
   This must include `0012_usage_metrics.sql`; review any earlier pending work.
4. Deploy the Worker and rebuilt web assets together through the existing
   Cloudflare deployment workflow. It executes the idempotent `0012` SQL file
   before activation, without applying unrelated pending migrations.
   `USAGE_METRICS_ENABLED=true`, `DB`,
   `SESSION_SECRET`, the rate-limit binding, and the cleanup cron are required.
5. In a new ChatGPT widget, verify that sharing starts off, opt into declared
   test use, analyze once, export once, and inspect the test-cohort dashboard.
   Verify that **Delete my usage** removes those events. Check anonymous admin
   requests return 401 and ordinary members receive 403.

Set `USAGE_METRICS_ENABLED=false` to stop collection while retaining deletion
access, historical administrator reads, and scheduled cleanup. Analysis and
exports continue when collection is disabled or the database is unavailable.
Do not delete tables to roll back. Existing artifacts and account tables are
unchanged. No paid analytics vendor or OpenAI inference API is added; requests,
D1 reads/writes, and retained data consume the existing Cloudflare quotas.
