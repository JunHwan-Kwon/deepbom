import assert from 'node:assert/strict';
import { DatabaseSync } from 'node:sqlite';
import { readFileSync, readdirSync } from 'node:fs';
import { createHash, randomBytes } from 'node:crypto';
import { routeUsage, readUsageSummary, recordUsageService, pruneUsage, USAGE_CONSENT_VERSION } from '../worker/usage-metrics.js';
import { createChatGptUsage } from '../web/lib/chatgpt-usage.js';
import worker from '../worker/index.js';
import { readPackageDownloads } from '../worker/usage-downloads.js';

// D1-compatible adapter over real SQLite: transactions, uniqueness and cascading
// deletion are exercised, rather than imitating the query results.
const db = new DatabaseSync(':memory:');
db.exec('PRAGMA foreign_keys = ON');
for (const file of readdirSync('migrations').filter((f) => f.endsWith('.sql')).sort()) db.exec(readFileSync(`migrations/${file}`, 'utf8'));
class Statement {
  constructor(sql, values = []) { this.sql = sql; this.values = values; }
  bind(...values) { return new Statement(this.sql, values); }
  execute() {
    let values = this.values;
    const numbered = [...this.sql.matchAll(/\?(\d+)/g)];
    const sql = numbered.length ? this.sql.replace(/\?\d+/g, '?') : this.sql;
    if (numbered.length) values = numbered.map((m) => this.values[Number(m[1]) - 1]);
    const statement = db.prepare(sql);
    return { success: true, results: statement.all(...values).map((row) => ({ ...row })) };
  }
  async run() { return this.execute(); }
  async all() { return this.execute(); }
  async first() { return this.execute().results[0] || null; }
}
const env = {
  USAGE_METRICS_ENABLED: 'true', SESSION_SECRET: 'test-only-not-a-production-secret',
  DB: { prepare: (sql) => new Statement(sql), async batch(statements) {
    db.exec('BEGIN');
    try { const rows = statements.map((s) => s.execute()); db.exec('COMMIT'); return rows; }
    catch (error) { db.exec('ROLLBACK'); throw error; }
  } },
};
const DAY = 86400000;
const today = Date.parse('2026-09-17T12:00:00Z');
const key = () => randomBytes(32).toString('base64url');
const event = (name, detail = '', format = 'onnx') => ({ event: name, detail, format });
const request = (path, body, origin = 'https://deepbom-org.web-sandbox.oaiusercontent.com') => new Request(`https://deepbom.org/api/usage/${path}`, { method: 'POST', headers: { origin, 'content-type': 'application/json' }, body: JSON.stringify(body) });
const call = (path, body, time = today) => routeUsage(request(path, body), env, time);
async function session(visitor, time = today, cohort = 'public', identity_scope = 'browser') {
  const response = await call('session', { visitor_key: visitor, consent_version: USAGE_CONSENT_VERSION, cohort, identity_scope }, time);
  assert.equal(response.status, 200); return (await response.json()).token;
}
async function send(token, events, time = today) {
  const response = await call('events', { token, events }, time); assert.equal(response.status, 200); return response;
}
const a = key(), b = key(), tester = key();
assert.equal((await routeUsage(request('session', {}, 'https://evil.example'), env)).status, 403);
assert.equal((await routeUsage(request('session', {}), { ...env, USAGE_METRICS_ENABLED: 'false' })).status, 503);
assert.equal((await call('session', { visitor_key: a, consent_version: 'old', cohort: 'public', identity_scope: 'browser' })).status, 400);
assert.equal((await call('session', { visitor_key: a, consent_version: USAGE_CONSENT_VERSION, cohort: 'public', identity_scope: 'browser', filename: 'private.onnx' })).status, 400);
assert.equal((await call('session', { invalid: 'x'.repeat(9000) })).status, 400);
assert.equal((await routeUsage(request('session', {}), { ...env, USAGE_RATE_LIMITER: { limit: async () => ({ success: false }) } })).status, 429);
const old = await session(a, today - DAY);
await send(old, [event('analysis_started'), event('analysis_completed'), event('export_prepared', 'cyclonedx')], today - DAY);
await send(old, [event('analysis_completed'), event('analysis_failed'), event('export_prepared', 'cyclonedx')], today - DAY);
assert.equal((await call('events', { token: old, events: [event('analysis_started')] }, today)).status, 400, 'Expired tokens must fail');
const current = await session(a);
await send(current, [event('export_prepared', 'png')]); // Out of sequence: ignored.
await send(current, [event('analysis_started'), event('analysis_completed'), event('export_prepared', 'png'), event('file_shared', 'png'), event('visualization_rendered', 'architecture-overview')]);
const sameDay = await session(a);
await send(sameDay, [event('analysis_started'), event('analysis_completed')]);
const failed = await session(b);
await send(failed, [event('analysis_started'), event('analysis_failed'), event('analysis_completed')]);
const test = await session(tester, today, 'test');
await send(test, [event('analysis_started'), event('analysis_completed')]);
const noStorage = await session(key(), today, 'public', 'session');
await send(noStorage, [event('analysis_started'), event('analysis_completed')]);
assert.equal((await call('events', { token: current + 'x', events: [event('analysis_started')] })).status, 400);
assert.equal((await call('events', { token: current, events: [event('export_prepared', 'private-file-name')] })).status, 400);
assert.equal((await call('events', { token: current, events: [{ ...event('analysis_completed'), sha256: 'private-hash' }] })).status, 400);
let summary = await readUsageSummary(env, { days: 7 }, today);
assert.equal(summary.totals.started, 5);
assert.equal(summary.totals.completed, 4);
assert.equal(summary.totals.failed, 1);
assert.equal(summary.totals.unique_browsers, 2, 'Session-only storage cannot inflate browser counts');
assert.equal(summary.totals.returning_browsers, 1, 'Same-day reload is not a returning browser');
assert.equal(summary.totals.export_preparations, 2, 'Retries do not increase export counts');
assert.equal(summary.totals.terminal_success_rate, 0.8);
assert.equal((await readUsageSummary(env, { days: 7, cohort: 'test' }, today)).totals.completed, 1);
assert.doesNotMatch(JSON.stringify(summary), /visitor_hash|session_id|visitor_key/);
assert(!JSON.stringify(db.prepare('SELECT * FROM usage_sessions').all()).includes(a), 'Raw browser key must not be stored');
assert.equal((await call('cohort', { visitor_key: a, cohort: 'test' })).status, 200);
assert.equal((await readUsageSummary(env, { days: 7 }, today)).totals.completed, 1);
assert.equal((await readUsageSummary(env, { days: 7, cohort: 'test' }, today)).totals.completed, 4);
await recordUsageService(env, [{ metric: 'http_request' }, { metric: 'http_request' }, { metric: 'analysis_report_received', format: 'onnx' }, { metric: 'private-name' }], today);
assert.equal((await call('forget', { visitor_key: a })).status, 200);
assert.equal((await call('events', { token: current, events: [event('analysis_started')] })).status, 410);
assert.equal((await readUsageSummary(env, { days: 7, cohort: 'test' }, today)).totals.completed, 1);
assert.equal((await readUsageSummary(env, {}, today)).service.find((row) => row.metric === 'http_request').count, 2);
assert.equal((await routeUsage(request('forget', { visitor_key: tester }), { ...env, USAGE_METRICS_ENABLED: 'false' }, today)).status, 200, 'Deletion stays available with collection disabled');
const ancient = await session(key(), today - 92 * DAY);
await send(ancient, [event('analysis_started'), event('analysis_completed')], today - 92 * DAY);
await recordUsageService(env, [{ metric: 'http_request' }], today - 92 * DAY);
await pruneUsage(env, today);
assert.equal(db.prepare("SELECT count(*) AS n FROM usage_sessions WHERE created_at < '2026-06-19'").get().n, 0);
assert.deepEqual(db.prepare('PRAGMA foreign_key_check').all(), []);
assert.equal(db.prepare('SELECT count(*) AS n FROM usage_service_daily').get().n, 2);

// The actual admin route must enforce the existing account role.
const adminUrl = 'https://deepbom.org/api/admin/usage?days=7';
assert.equal((await worker.fetch(new Request(adminUrl), env)).status, 401);
for (const role of ['member', 'admin']) {
  const token = `test-${role}`;
  db.prepare('INSERT INTO users(id, email, created_at, updated_at, role) VALUES (?, ?, ?, ?, ?)').run(role, `${role}@example.test`, new Date().toISOString(), new Date().toISOString(), role);
  db.prepare('INSERT INTO sessions(id_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)').run(createHash('sha256').update(token).digest('hex'), role, new Date().toISOString(), new Date(Date.now() + DAY).toISOString());
  const response = await worker.fetch(new Request(adminUrl, { headers: { cookie: `audit_session=${token}` } }), env);
  assert.equal(response.status, role === 'admin' ? 200 : 403);
}
const savedWarnings = console.warn; console.warn = () => {};
try {
  const rpc = new Request('https://deepbom.org/mcp', { method: 'POST', body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: 'deepbom_capabilities', arguments: {} } }) });
  const response = await worker.fetch(rpc, { ...env, DB: { prepare() { throw new Error('database offline'); } } });
  assert.equal(response.status, 200); assert((await response.json()).result, 'Telemetry outage must not break MCP');
} finally { console.warn = savedWarnings; }

// Client consent, deletion, and storage behavior against the real collector.
const values = new Map(); const storage = { getItem: (k) => values.get(k) || null, setItem: (k, v) => values.set(k, v), removeItem: (k) => values.delete(k) };
const calls = []; let failDelete = false;
const fetcher = async (url, options) => {
  const body = JSON.parse(options.body); calls.push({ path: url.pathname, body });
  if (failDelete && url.pathname.endsWith('/forget')) return new Response('{}', { status: 503 });
  assert.equal(options.credentials, 'omit');
  return routeUsage(new Request(url, { ...options, headers: { ...options.headers, origin: 'https://deepbom.org' } }), env, today);
};
const client = createChatGptUsage({ endpoint: 'https://deepbom.org', fetcher, storage, now: () => today });
client.track('analysis_started'); client.track('analysis_completed', { format: 'onnx' }); await client.flush();
assert.equal(calls.length, 0, 'No statistics requests before consent');
assert.equal(values.size, 0, 'No identifier before consent');
await client.setEnabled(true);
assert.equal(calls[0].body.identity_scope, 'browser');
assert.equal(calls.filter((c) => c.path.endsWith('/events')).length, 1);
client.track('export_prepared', { format: 'onnx', detail: 'spdx' }); await client.flush();
const afterExport = calls.length;
client.track('export_prepared', { format: 'onnx', detail: 'spdx' }); await client.flush();
assert.equal(calls.length, afterExport);
await client.setEnabled(false);
client.track('export_prepared', { format: 'onnx', detail: 'png' }); await client.flush();
assert.equal(calls.length, afterExport, 'Withdrawal stops transmission');
await client.setEnabled(true); await client.setCohort('test');
assert.equal(client.state().cohort, 'test');
assert.equal(client.state().enabled, true);
failDelete = true; assert.equal(await client.forget(), false);
assert.equal(client.state().enabled, false); assert.equal(client.state().hasHistory, true);
failDelete = false; assert.equal(await client.forget(), true);
assert.equal(client.state().hasHistory, false); assert.equal(values.size, 0);
const volatile = createChatGptUsage({ endpoint: 'https://deepbom.org', fetcher, storage: null, now: () => today });
volatile.track('analysis_started'); await volatile.setEnabled(true);
assert.equal(calls.filter((c) => c.path.endsWith('/session')).at(-1).body.identity_scope, 'session');
const beforeWithdrawal = calls.length;
volatile.storageChanged(null); volatile.track('analysis_failed'); await volatile.flush();
assert.equal(calls.length, beforeWithdrawal, 'Another panel deleting preferences stops this panel');
const expired = createChatGptUsage({ endpoint: 'https://deepbom.org', fetcher, storage: { getItem: () => JSON.stringify({ enabled: true, key: key(), version: USAGE_CONSENT_VERSION, cohort: 'public', expires: today - 1 }) }, now: () => today });
assert.equal(expired.state().enabled, false);

// Channel attribution, legacy migration, browser separation and run identity.
db.exec(readFileSync('migrations/0013_usage_channels.sql', 'utf8')); // Safe to rerun at every deployment.
const sharedKey = key();
const legacyToken = await session(sharedKey, today - DAY);
const legacyId = JSON.parse(Buffer.from(legacyToken.split('.')[0], 'base64url')).id;
db.prepare('DELETE FROM usage_session_channels WHERE session_id = ?').run(legacyId);
await send(legacyToken, [event('analysis_started'), event('analysis_completed')], today - DAY);
const modernToken = await session(sharedKey);
await send(modernToken, [event('analysis_started'), event('analysis_completed')]);
const webValues = new Map(), webCalls = [];
const webStorage = { getItem: (k) => webValues.get(k) || null, setItem: (k, v) => webValues.set(k, v), removeItem: (k) => webValues.delete(k) };
const webClient = createChatGptUsage({ channel: 'web', endpoint: 'https://deepbom.org', storage: webStorage, now: () => today,
  fetcher: async (url, options) => { webCalls.push(JSON.parse(options.body)); return fetcher(url, options); },
});
assert.equal(webClient.storageKey, 'deepbom.web.usage.v1');
webClient.beginRun(); webClient.track('analysis_started'); webClient.track('analysis_completed', { format: 'onnx' });
assert.equal(webCalls.length, 0);
await webClient.setEnabled(true);
assert.equal(webCalls[0].channel, 'web');
webClient.beginRun(); webClient.track('analysis_started'); webClient.track('analysis_failed'); await webClient.flush();
webClient.beginRun(); webClient.track('analysis_started'); await webClient.flush();
let byWeb = await readUsageSummary(env, { channel: 'web', days: 7 }, today);
assert.equal(byWeb.totals.started, 3); assert.equal(byWeb.totals.completed, 1); assert.equal(byWeb.totals.failed, 1); assert.equal(byWeb.totals.unfinished, 1);
assert.equal(byWeb.totals.unique_browsers, 1); assert.equal(byWeb.totals.returning_browsers, 0);
assert.equal(byWeb.weekly_browsers[0].browsers, 1, 'Weekly unique browsers must not sum daily/run counts');
assert(byWeb.events.every((r) => r.channel === 'web'));
const allChannels = await readUsageSummary(env, { channel: 'all', days: 7 }, today);
assert.equal(allChannels.totals.unique_browsers, null, 'No cross-channel people estimate');
assert.equal(allChannels.totals.returning_browser_rate, null);
assert.equal(allChannels.channels.find((c) => c.id === 'cli').totals, null);
assert.equal((await readUsageSummary(env, { channel: 'claude' }, today)).totals, null);
assert.equal((await readUsageSummary(env, { channel: 'chatgpt' }, today)).totals.returning_browsers, 1, 'Legacy and attributed ChatGPT sessions share one browser history');
assert.equal((await call('session', { visitor_key: key(), consent_version: USAGE_CONSENT_VERSION, cohort: 'public', identity_scope: 'browser', channel: 'cli' })).status, 400);
assert.equal((await call('session', null)).status, 400);
for (const channel of [null, 1, {}, 'unknown']) assert.equal((await call('session', { visitor_key: key(), consent_version: USAGE_CONSENT_VERSION, cohort: 'public', identity_scope: 'browser', channel })).status, 400);
await webClient.setCohort('test');
assert.equal((await readUsageSummary(env, { channel: 'web' }, today)).totals.started, 0);
assert.equal((await readUsageSummary(env, { channel: 'web', cohort: 'test' }, today)).totals.started, 3);
await webClient.forget();
assert.equal((await readUsageSummary(env, { channel: 'web', cohort: 'test' }, today)).totals.started, 0);
assert.deepEqual(db.prepare('PRAGMA foreign_key_check').all(), []);
assert((await readUsageSummary(env, { channel: 'chatgpt' }, today)).totals.completed > 0, 'Deleting web usage preserves unrelated ChatGPT history');

// Slow delivery from a previous run cannot attach its outcome to the next run.
let releaseRequest; const hold = new Promise((resolve) => { releaseRequest = resolve; }); let held = false;
const slow = createChatGptUsage({ channel: 'web', endpoint: 'https://deepbom.org', storage: null, now: () => today, fetcher: async (url, options) => {
  if (!held) { held = true; await hold; } return fetcher(url, options);
} });
await slow.setEnabled(true); slow.track('analysis_started'); slow.track('analysis_completed', { format: 'onnx' });
await Promise.resolve(); slow.beginRun(); slow.track('analysis_started'); slow.track('analysis_failed');
releaseRequest(); await slow.flush(); await slow.setEnabled(false);
const slowStats = await readUsageSummary(env, { channel: 'web' }, today);
assert.equal(slowStats.totals.completed, 1); assert.equal(slowStats.totals.failed, 1);
await slow.forget();

// Downloads stay separate, bounded, cached, and unavailable instead of zero.
let fetched = 0, cached;
const cache = { async match() { return cached?.clone(); }, async put(_key, value) { cached = value; } };
const npmFetcher = async (url) => { fetched++; assert.match(url, /\/downloads\/range\/2026-09-10:2026-09-16\/deepbom$/); return Response.json({ package: 'deepbom', start: '2026-09-10', end: '2026-09-16', downloads: [{ day: '2026-09-10', downloads: 12 }, { day: '2026-09-16', downloads: 0 }] }); };
const downloads = await readPackageDownloads(7, { now: today, cache, fetcher: npmFetcher });
assert.equal(downloads.reported_total, 12); assert.equal(downloads.missing_days, 5); assert.equal(downloads.daily[1].downloads, null);
await readPackageDownloads(7, { now: today, cache, fetcher: npmFetcher }); assert.equal(fetched, 1);
await assert.rejects(readPackageDownloads(7, { now: today, cache: null, fetcher: async () => new Response('offline', { status: 503 }) }));
await assert.rejects(readPackageDownloads(7, { now: today, cache: null, fetcher: async () => new Response('x'.repeat(32769)) }));
for (const downloads of [[{ day: '2026-09-10', downloads: -1 }], [{ day: '2026-09-10', downloads: 1 }, { day: '2026-09-10', downloads: 2 }], [{ day: '2026-09-17', downloads: 2 }]]) {
  await assert.rejects(readPackageDownloads(7, { now: today, cache: null, fetcher: async () => Response.json({ package: 'deepbom', start: '2026-09-10', end: '2026-09-16', downloads }) }));
}
await assert.rejects(readPackageDownloads(8, { now: today, cache: null, fetcher: npmFetcher }));
assert.equal((await worker.fetch(new Request('https://deepbom.org/api/admin/usage/downloads?days=7'), env)).status, 401);
const memberCookie = createHash('sha256').update('test-member').digest('hex'); assert(db.prepare('SELECT id_hash FROM sessions WHERE id_hash = ?').get(memberCookie));
assert.equal((await worker.fetch(new Request('https://deepbom.org/api/admin/usage/downloads?days=7', { headers: { cookie: 'audit_session=test-member' } }), env)).status, 403);
db.close();
console.log('Usage metrics passed: consent, privacy allowlists, signed tokens, state ordering, deduplication, UTC revisits, test exclusion, deletion, retention, admin authorization, and telemetry outage isolation.');
