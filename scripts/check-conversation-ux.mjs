import assert from "node:assert/strict";
import { initializeNumericalWasm } from "../web/lib/numerical-wasm.js";
import { readWidgetState, updateWidgetState, widgetLanguage } from "../web/lib/chatgpt-widget-state.js";
import { queryFollowUps } from "../web/lib/evidence-query.js";
import { evidenceRowKey, evidenceBinding, captureEvidenceView, restoreEvidenceRow, shareEvidenceTask, evidenceNote, reusableEvidenceQuestion } from "../web/lib/evidence-exploration.js";
await initializeNumericalWasm();

const host = { locale: "ko-KR", setWidgetState(value) { this.saved = value; } }; // delayed widgetState echo
const a = "a".repeat(64), b = "b".repeat(64);
updateWidgetState(host, a, { modelContent: { schema: "visual", view: "architecture-overview" }, imageIds: ["image-a"] });
updateWidgetState(host, a, { modelContent: { selected_evidence: { subject_ref: "layer-b" } }, privateContent: { evidence_query_report: "report-a" } });
updateWidgetState(host, a, { modelContent: { view: "block-detail" } });
assert.equal(host.saved.modelContent.selected_evidence.subject_ref, "layer-b");
assert.equal(host.saved.modelContent.schema, "visual");
assert.deepEqual(host.saved.imageIds, ["image-a"]);
assert.equal(readWidgetState(host).privateContent.evidence_query_report, "report-a");
updateWidgetState(host, b, { modelContent: { selected_evidence: null } });
assert.equal(host.saved.modelContent.artifact_sha256, b);
assert.equal(host.saved.modelContent.view, undefined);
assert.equal(host.saved.imageIds, undefined);
assert.equal(host.saved.privateContent.evidence_query_report, undefined);
assert.equal(updateWidgetState({}, a, {}), false);
assert.equal(updateWidgetState({ setWidgetState() { throw new Error("unsupported"); } }, a, {}), false);
assert.equal(widgetLanguage(host), "ko");
assert.equal(widgetLanguage({ locale: "en-US" }), "en");

const rows = ["first", "second"].map(ref => ({ subject_ref: ref, kind: "weight_evidence", title: ref, details: {} }));
const result = { artifact: { sha256: a }, query: { section: "weights", weight_view: "distribution", search: "encoder", offset: 0, limit: 2, profile_ids: [] }, rows, coverage: { next_offset: 2 } };
const actions = queryFollowUps(result, rows[1]).actions;
for (const action of actions.filter(row => row.id.startsWith("weight_"))) {
  assert.equal(action.query.subject_ref, "second");
  assert.equal(action.query.offset, 0);
  assert.equal(action.query.search, undefined);
}
assert.equal(actions.find(row => row.id === "next_page").query.search, "encoder");
assert.equal(actions.find(row => row.id === "next_page").query.offset, 2);
assert.throws(() => queryFollowUps(result, { ...rows[1] }), /not part/);
assert.deepEqual(queryFollowUps({ ...result, rows: [], coverage: { next_offset: null }, query: { ...result.query, search: "" } }).actions, []);
console.log("Conversation UX: shared follow-up selection, pagination, host state merging, artifact reset, unsupported hosts and locale checks passed.");

// Presentation state must remain tied to exact evidence and preserve unknowns.
const bound = { ...result, artifact: { sha256: a, format: "onnx" }, analyzer_version: "2.5.0", model_ir_sha256: b,
  result_sha256: "c".repeat(64), context: { scan_policy: { resolved_mode: "integrity" }, evidence_envelope_sha256: "d".repeat(64) },
  evidence_boundary: "Static evidence; runtime not measured." };
const snapshot = captureEvidenceView(bound, rows[1], { scrollLeft: 135, scrollTop: 260 });
assert.equal(restoreEvidenceRow(bound, snapshot), rows[1]);
for (const changed of [{ ...bound, artifact: { ...bound.artifact, sha256: b } }, { ...bound, model_ir_sha256: a }, { ...bound, result_sha256: a }, { ...bound, context: { ...bound.context, scan_policy: { resolved_mode: "structure" } } }]) assert.equal(restoreEvidenceRow(changed, snapshot), null);
assert.notEqual(evidenceRowKey({ ...rows[1], details: { profile_id: "one" } }), evidenceRowKey({ ...rows[1], details: { profile_id: "two" } }));
assert.equal(typeof evidenceBinding(bound), "string");
let pages = [];
for (let i = 0; i < 30; i++) pages.push(captureEvidenceView(bound, rows[1], { scrollLeft: 0, scrollTop: i }, pages));
assert(JSON.stringify(pages).length < 600_000, "History strips recursive page histories instead of exponential growth");
const metric = { ...rows[1], details: { unknown: null, exact_count: "9007199254740993" }, detail_truncations: [] };
const note = evidenceNote({ ...bound, rows: [metric] }, metric, "ko");
assert(note.includes('"unknown":null') && note.includes('9007199254740993') && note.includes(a));
assert.throws(() => evidenceNote(bound, { ...rows[1] }), /does not belong/);
for (const language of ["en", "ko"]) for (const section of ["operators", "operator", "findings", "improvements", "profiles", "placement", "fusion", "weights"]) {
  const text = reusableEvidenceQuestion({ ...bound, query: { section, subject_ref: "private-subject", profile_ids: ["private-profile"] } }, language);
  assert(text.length > 20);
  for (const privateValue of [a, b, "private-subject", "private-profile"]) assert(!text.includes(privateValue));
  assert.match(text, /payload/);
}
let attempts = 0, finish;
const shared = shareEvidenceTask(() => { attempts++; return new Promise(resolve => { finish = resolve; }); });
const pending = [shared(), shared(), shared()]; await Promise.resolve();
assert.equal(attempts, 1); finish(bound);
assert((await Promise.all(pending)).every(value => value === bound));
assert.equal(await shared(), bound); assert.equal(attempts, 1);
let tries = 0;
const retry = shareEvidenceTask(async () => { if (++tries === 1) throw new Error("retry me"); return bound; });
await assert.rejects(retry(), /retry me/); assert.equal(await retry(), bound); assert.equal(tries, 2);
console.log("Evidence exploration: exact binding restoration, bounded history, copy scope, bilingual reusable prompts and coalesced/retryable weight work passed.");
