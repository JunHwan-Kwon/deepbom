# Usage evidence by channel

Administrators use `https://deepbom.org/web/admin.html` → **Usage by channel**.
The existing administrator login protects the dashboard,
`GET /api/admin/usage?days=30&cohort=public&channel=all`, and
`GET /api/admin/usage/downloads?days=30`. No login is added for analyzer users.

Choose 7, 30, or 90 UTC days, a channel, and public use or declared tests.
Daily completed-analysis and weekly active/returning-browser charts use the
same filters. CSV preserves channel and coverage; JSON also preserves metric
definitions and the separately sourced npm snapshot. No browser keys, session
IDs, model names/hashes, email addresses, or IP addresses appear in exports.

## Channel coverage

| Channel | Available measurement |
| --- | --- |
| ChatGPT | Consented widget analysis runs, feature use and browser estimates. |
| Web | Consented website analysis runs, feature use and separate browser estimates. |
| CLI | Execution is **not measured**. npm downloads are shown separately as distribution counts. |
| Local MCP | Execution is **not measured**. The npm package is shared with the CLI. |
| Claude | **Not measured** until instrumentation with its own stated consent scope is added. |

Channel attribution is declared by the client; it is not authenticated proof of
which host sent a request. Legacy records belong to the ChatGPT widget. Each
measured channel has a recorded collection start. Earlier dates and unsupported
channels show **Not measured**, never an invented zero. The first UTC date and
today can be partial. Zero means no reports received, not no actual use.
Combined-channel views sum analysis events but never display a combined unique
browser or people count. A person may use more than one channel or device.

## Metric definitions

| Metric | Definition |
| --- | --- |
| Analysis started/completed/failed | One analysis run, deduplicated by event/detail. First terminal outcome wins. ChatGPT completion means its bounded result was published; Web completion means the workbench finished rendering its analysis. |
| Unfinished | Started in the selected period with no terminal report received; may be running, abandoned or interrupted. It is not an automatic failure. |
| Terminal success rate | Completed / (completed + failed). Excludes unfinished runs; not an overall completion rate. |
| Format share | Completed opt-in analyses by supported artifact format, within the selected channel/cohort. |
| Visualization | Runs that rendered the measured view. Initial automatic views count; not proof a person viewed it. |
| Prepared exports | Once per export type per run after preparation succeeds; not confirmed downloads. ChatGPT supports SVG, PNG, Word ZIP, CycloneDX and SPDX. Web measures SVG, PNG ZIP, CycloneDX, Model Views ZIP and Evidence Package ZIP. Other web reports are outside this counter. |
| File handoffs | Runs where ChatGPT accepted a generated-file upload. Not confirmed downloads or rendered replies. |
| Estimated active browsers | Distinct consenting persistent browser keys with an analysis start in the period, within one channel. Not verified people/accounts. |
| Returning browsers | Active in the period and observed on at least two UTC dates in retained 90-day history. Same-day reruns alone do not qualify. |
| Weekly browsers | Distinct keys active in the Monday-based UTC week, within the selected dates. Returning means a visit has an earlier observed date; edge weeks can be partial. |
| MCP service counters | `/mcp` requests/errors/reports from all callers, including tests, bots and retries. Independent of browser-channel/cohort filters. Not unique analyses or people. |

Explicit reruns create new runs; delivery retries do not. A stale export that
finishes after a new web analysis starts is excluded instead of being attributed
to the new run. Storage partitioning, multiple devices, cleared storage, opt-outs
and forged clients affect estimates. Runs without persistent storage contribute
consented events but not unique/revisit counts. Do not extrapolate to total users.

Mark **This browser is for testing** before demos and tests. This reclassifies the
key's retained use. Unmarked tests cannot be detected automatically. Service and
npm counters always include tests. An appropriate public claim is “N consenting
browser instances and M completed opt-in analyses on Web during [UTC dates]”,
with ChatGPT reported separately and the collection scope disclosed.

## npm distribution counts

The administrator dashboard reads the fixed public npm downloads API for
`deepbom`. It requests the last 7/30/90 complete UTC days, excluding today, and
caches the response for one hour. The source URL, retrieval time, missing-day
count and daily values are included in JSON. Missing days stay unknown; provider
failure shows unavailable while the usage dashboard remains usable.

Downloads are neither unique installations, CLI executions nor people. They
include automated/test traffic and local MCP package downloads. No test-cohort
filter applies. PyPI, Cargo and GitHub are not included. Only the package name
and date interval go to npm; no model, browser or account identifiers do. CLI and
local MCP contain no new telemetry.

## Consent and retention

Sharing is off by default. Opt-in covers the current analysis run and subsequent
visits while the 90-day preference remains valid. No key is created or usage
request sent before opt-in. Consent and keys are separate for ChatGPT and Web. A random 32-byte key is stored locally; the database
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

Engine **1.103.0** includes this channel extension. Agent contract **1.0.2** is
the explicit updated privacy candidate under
[`AGENT_CONTRACT_VERSIONING.md`](../AGENT_CONTRACT_VERSIONING.md). The baseline
records `candidate_unsubmitted`, not platform approval. Deployment and registry
publication do not update or approve the account-owned OpenAI submission.
Historical and current changes are recorded in
[`usage-review-change.json`](usage-review-change.json). The root
`chatgpt-app-submission.json` remains a portal import draft.

1. Run `npm run check:usage-metrics`, `npm run check:usage-dashboard`,
   `npm run check:web-usage`, `npm run check:chatgpt-widget`, and deployment checks.
2. Reconcile the privacy candidate with the pending plugin review in the portal.
   Repository files do not update that account-owned review automatically.
3. Deploy through the existing Cloudflare workflow. It executes idempotent
   `0012_usage_metrics.sql` and `0013_usage_channels.sql` before activating the
   Worker and assets, without applying unrelated pending migrations.
   `USAGE_METRICS_ENABLED=true`, `DB`, `SESSION_SECRET`, the rate-limit binding,
   and the cleanup cron are required. Existing sessions remain readable.
4. In ChatGPT and Web separately, verify default-off sharing, mark test use,
   opt in, analyze/export, and verify the test dashboard and **Delete my usage**.
   Anonymous administrator requests must return 401; ordinary members get 403.

Set `USAGE_METRICS_ENABLED=false` to stop collection while retaining deletion,
historical administrator reads, and cleanup. Analysis/exports continue through
collector outages. Do not delete tables to roll back. Existing account tables
are unchanged. No paid analytics vendor or OpenAI inference API is added;
requests, D1 work and storage consume existing Cloudflare quotas.
