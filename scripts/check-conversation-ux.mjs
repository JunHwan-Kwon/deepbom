import assert from "node:assert/strict";
import { initializeNumericalWasm } from "../web/lib/numerical-wasm.js";
import { readWidgetState, updateWidgetState, widgetLanguage } from "../web/lib/chatgpt-widget-state.js";
import { queryFollowUps } from "../web/lib/evidence-query.js";
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
