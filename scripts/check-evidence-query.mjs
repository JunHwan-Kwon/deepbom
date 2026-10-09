import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import Ajv from "ajv";
import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { buildArtifactEvidenceEnvelope } from "../web/lib/artifact-evidence-envelope.js";
import { buildReviewSummary } from "../web/lib/review-summary.js";
import { buildExecutionPlacementEvidence } from "../web/lib/execution-placement-evidence.js";
import { queryEvidence } from "../web/lib/evidence-query.js";
import { buildEvidenceQueryVisual } from "../web/lib/evidence-query-view.js";
import { EVIDENCE_QUERY_RESULT_JSON_SCHEMA, normalizeEvidenceQuery, sealEvidenceQueryResult, validateEvidenceQueryResult } from "../web/lib/evidence-query-contract.js";
import { analyzeExecuTorchModel } from "../web/executorch.js";
import { sha256BytesHex } from "../web/lib/sha256-sync.js";
import { decodeFixtureBase64, EXECUTORCH_ADD_PTE_BASE64 } from "./fixtures/executorch-fixtures.mjs";
import { buildWeightEvidence } from "../web/lib/weight-analysis.js";
import { buildWeightQueryVisual } from "../web/lib/weight-query-visual.js";
import { seal } from "../web/lib/numerical-ir/common.js";

const ajv = new Ajv({ strict: false, allowUnionTypes: true });
const schema = ajv.compile(EVIDENCE_QUERY_RESULT_JSON_SCHEMA);
const contexts = new Map();
for (const [format, file] of [
  ["onnx", "gpu_partition_probe.onnx"], ["tflite", "mobilenet_v2_1.0_224_quant.tflite"],
  ["gguf", "tinymqa1m.Q4_0.gguf"], ["safetensors", "nanofable-1m-fp16.safetensors"], ["coreml", "MNISTClassifier.mlmodel"],
]) {
  const path = `web/samples/${file}`;
  const processResult = spawnSync(process.execPath, ["bin/deepbom.mjs", "audit", path, "--compact"], { encoding: "utf8", maxBuffer: 64 * 1024 * 1024, timeout: 120_000 });
  assert.equal(processResult.status, 0, `${format}: ${processResult.stderr}`);
  const raw = JSON.parse(processResult.stdout);
  const context = contextFor(raw, path.split("/").pop(), (await readFile(path)).length);
  contexts.set(format, context);
}
const bytes = decodeFixtureBase64(EXECUTORCH_ADD_PTE_BASE64);
const execuTorch = analyzeExecuTorchModel(bytes, "add.pte"); execuTorch.model_sha256 = sha256BytesHex(bytes);
contexts.set("executorch", contextFor(execuTorch, "add.pte", bytes.length));

for (const [format, context] of contexts) {
  const expected = context.artifactIrContext.model_summary.rows;
  let offset = 0; const actual = [];
  do {
    const result = queryEvidence(context, { section: "operators", offset, limit: 2 });
    assert(schema(result), `${format}: ${JSON.stringify(schema.errors)}`);
    assert.equal(result.artifact.sha256, context.analysis.model_sha256);
    assert.equal(result.coverage.total_rows, expected.length);
    actual.push(...result.rows);
    offset = result.coverage.next_offset;
  } while (offset !== null);
  assert.deepEqual(actual.map(row => row.subject_ref), expected.map(row => row.subject_ref), `${format}: pagination conserves all identities`);
  for (const row of actual) {
    const original = expected.find(item => item.subject_ref === row.subject_ref);
    assert.deepEqual(row.details.metrics, original.metrics, `${format}: preserve native metric contracts`);
    assert.deepEqual(row.details.aggregates, original.aggregates, `${format}: preserve exact storage values`);
  }
  for (const section of ["findings", "fusion", "improvements", "profiles", "placement"]) {
    const result = queryEvidence(context, { section, limit: 3 });
    assert(schema(result), `${format}/${section}: ${JSON.stringify(schema.errors)}`);
    assert.deepEqual(queryEvidence(context, { section, limit: 3 }), result, `${format}/${section}: deterministic query`);
    const visual = buildEvidenceQueryVisual(result);
    assert(visual.svg.includes(result.artifact.sha256));
    assert(visual.svg.includes(result.result_sha256));
    assert.equal(visual.render_model.rows.length, result.rows.length);
  }
  if (expected.length) {
    const selected = queryEvidence(context, { section: "operator", subject_ref: expected[0].subject_ref });
    assert.equal(selected.rows.length, 1);
    assert.equal(selected.rows[0].subject_ref, expected[0].subject_ref);
  }
}

const tflite = contexts.get("tflite");
// Decode a real model, then ensure the transport preserves the shared numerical
// results rather than recomputing approximations or using truncated plot data.
const weightContext = contexts.get("onnx");
const weightBytes = await readFile("web/samples/gpu_partition_probe.onnx");
weightContext.weightEvidence = await buildWeightEvidence(weightContext.artifactIrContext.model_ir, weightContext.analysis, weightBytes);
for (const view of ["distribution", "channels", "similarity", "spectrum", "sparsity", "quantization"]) {
  const result = queryEvidence(weightContext, { section: "weights", weight_view: view, limit: 40 });
  assert(schema(result), JSON.stringify(schema.errors));
  assert.equal(result.coverage.total_rows, weightContext.weightEvidence.weight_analysis.tensors.length);
  for (const row of result.rows) {
    const source = weightContext.weightEvidence.weight_analysis.tensors.find(item => item.weight_ref === row.subject_ref);
    assert.deepEqual(row.details.statistics, source.statistics);
    assert.equal(row.details.value_count, source.value_count);
    if (!row.detail_truncations.length && view !== "distribution") assert.deepEqual(row.details.feature, source[view]);
    const visual = buildWeightQueryVisual(weightContext.weightEvidence, row.subject_ref, view, weightContext.artifactIrContext.model_ir);
    if (visual) { assert(visual.svg.includes(result.artifact.sha256)); assert(visual.render_model.page.width_mm > 0); }
  }
}
assert.throws(() => queryEvidence({ ...weightContext, weightEvidence: undefined }, { section: "weights" }), /must complete/);
const wrongWeight = structuredClone(weightContext.weightEvidence);
wrongWeight.weight_analysis.source.artifact_sha256 = "0".repeat(64);
assert.throws(() => queryEvidence({ ...weightContext, weightEvidence: wrongWeight }, { section: "weights" }), /source.*match|sha256 mismatch/);
// A recomputed digest must not bypass the common numerical and binding rules.
for (const mutate of [
  evidence => { evidence.weight_analysis.tensors[0].storage_ref = "storage:foreign"; },
  evidence => { evidence.weight_analysis.coverage.assessed_count += 1; },
  evidence => { evidence.weight_analysis.tensors[0].value_count = "999999"; },
]) {
  const altered = structuredClone(weightContext.weightEvidence);
  mutate(altered);
  delete altered.weight_analysis.weight_analysis_sha256;
  altered.weight_analysis = seal(altered.weight_analysis, "weight_analysis_sha256");
  assert.throws(() => queryEvidence({ ...weightContext, weightEvidence: altered }, { section: "weights" }), /binding|coverage|count/);
  assert.throws(() => buildWeightQueryVisual(altered, altered.weight_analysis.tensors[0].weight_ref, "distribution", weightContext.artifactIrContext.model_ir), /binding|coverage|count/);
}
assert.throws(() => normalizeEvidenceQuery({ section: "operators", weight_view: "distribution" }), /Invalid weight/);
assert.throws(() => normalizeEvidenceQuery({ section: "weights", source_index: 0 }), /weight subject_ref/);
const placement = buildExecutionPlacementEvidence(tflite.analysis);
for (const profile of placement.static_profiles) {
  const result = queryEvidence(tflite, { section: "placement", profile_ids: [profile.profile_id], limit: 40 });
  assert.deepEqual(result.context.profiles[0].state_counts, profile.state_counts);
  assert.equal(result.coverage.total_rows, profile.op_count);
  assert(result.rows.every(row => row.details.model_subject_ref && row.details.runtime_assignment === "not_observed"));
}
assert.throws(() => queryEvidence(tflite, { section: "placement", profile_ids: ["invented_delegate"] }), /Unknown or unavailable/);
assert.throws(() => queryEvidence(tflite, { section: "operator", subject_ref: "missing" }), /exactly one/);
assert.throws(() => queryEvidence(tflite, { section: "operators", target: "wrong" }), /not analyzed/);
assert.throws(() => normalizeEvidenceQuery({ section: "operators", limit: 0 }), /limit/);
assert.throws(() => normalizeEvidenceQuery({ section: "operators", source_index: 0, subject_ref: "op" }), /not both/);
assert.throws(() => normalizeEvidenceQuery({ section: "operators", arbitrary: "ignored?" }), /Unexpected/);

const first = queryEvidence(tflite, { section: "operators", limit: 1 });
for (const edit of [
  result => { result.artifact.sha256 = "0".repeat(64); },
  result => { result.coverage.returned_rows += 1; },
  result => { result.coverage.next_offset = 0; },
  result => { result.query.offset = 7; },
]) { const altered = structuredClone(first); edit(altered); assert.throws(() => validateEvidenceQueryResult(altered)); }
const resealed = structuredClone(first); resealed.coverage.next_offset = 0; delete resealed.result_sha256;
assert.throws(() => validateEvidenceQueryResult(sealEvidenceQueryResult(resealed)), /cursor/);

const safe = contexts.get("safetensors");
assert.equal(queryEvidence(safe, { section: "placement" }).status, "not_assessable");
const fusion = queryEvidence(tflite, { section: "fusion", limit: 40 });
assert(fusion.rows.every(row => row.details.serialized_fused_activation === tflite.analysis.ops.find(op => op.artifact_ir_subject_ref === row.subject_ref).fused_activation));
assert(fusion.rows.every(row => row.details.runtime_fusion === "not_observed"));
const empty = queryEvidence(tflite, { section: "operators", search: "nonexistent_qwerty" });
assert.equal(empty.status, "no_matches"); assert.equal(empty.rows.length, 0);
const offsetPastEnd = queryEvidence(tflite, { section: "operators", offset: 999999 });
assert.equal(offsetPastEnd.coverage.next_offset, null);

const scoped = structuredClone(tflite);
const duplicate = structuredClone(scoped.artifactIrContext.model_summary.rows[0]);
duplicate.subject_ref += ":nested";
scoped.artifactIrContext.model_summary.rows.push(duplicate);
assert.throws(() => queryEvidence(scoped, { section: "operator", source_index: duplicate.order.source }), /multiple scopes/);
assert.equal(queryEvidence(scoped, { section: "operator", subject_ref: duplicate.subject_ref }).rows[0].subject_ref, duplicate.subject_ref);
const large = structuredClone(tflite);
large.envelope.findings = [{ id: "fixture", title: "<script>alert(1)</script>", severity: "high", notes: Array.from({ length: 80 }, () => "detail".repeat(500)) }];
const bounded = queryEvidence(large, { section: "findings" });
assert.equal(bounded.coverage.detail_truncated, true);
assert.equal(bounded.status, "partial");
assert(bounded.rows[0].detail_truncations.includes("/notes"));
assert(!buildEvidenceQueryVisual(bounded).svg.includes("<script>"));
assert(buildEvidenceQueryVisual(bounded).svg.includes("&lt;script&gt;"));

const workload = structuredClone(tflite);
workload.envelope.findings = [];
workload.artifactIrContext.model_summary.rows = workload.artifactIrContext.model_summary.rows.slice(0, 3);
const decimals = ["9007199254740992", "9007199254740993", null];
workload.artifactIrContext.model_summary.rows.forEach((row, i) => { row.metrics.macs.decimal = decimals[i]; });
assert.deepEqual(queryEvidence(workload, { section: "improvements" }).rows.map(row => row.details.metrics.macs.decimal), [decimals[1], decimals[0], null], "Workload ordering must preserve integer precision and unknowns");

const workerUrl = new URL("../worker/chatgpt-mcp.js", import.meta.url);
if (existsSync(workerUrl)) {
  const { routeChatGptMcp, CHATGPT_MCP_CONTRACT } = await import(workerUrl.href);
  async function rpc(name, args) {
    return (await routeChatGptMcp(new Request("https://deepbom.org/mcp", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/call", params: { name, arguments: args } }) }))).json();
  }
  const file = { file_id: "test-file", download_url: "https://files.oaiusercontent.com/artifact", file_name: "model.tflite" };
  const args = { file, expected_sha256: first.artifact.sha256, query: { section: "placement", profile_ids: ["tflite_gpu"] } };
  const tool = CHATGPT_MCP_CONTRACT.tools.find(row => row.name === "deepbom_query_file");
  assert(ajv.compile(tool.inputSchema)(args));
  const start = await rpc("deepbom_query_file", args);
  assert.equal(start.result.structuredContent.status, "browser_query_started");
  assert(ajv.compile(tool.outputSchema)(start.result.structuredContent));
  assert.equal(start.result.structuredContent.analysis_depth, "structure");
  assert.equal((await rpc("deepbom_query_file", { ...args, analysis_depth: "payload_integrity" })).result.structuredContent.analysis_depth, "payload_integrity");
  assert(!JSON.stringify(start.result.structuredContent).includes(file.download_url));
  const published = await rpc("deepbom_publish_query", { result: first });
  assert.deepEqual(published.result.structuredContent, first);
  for (const invalid of [
    { ...args, expected_sha256: "wrong" }, { ...args, query: { section: "placement", target: "unregistered" } },
    { ...args, query: { section: "operator" } }, { ...args, extra: true },
    { ...args, analysis_depth: "invented_scan" },
  ]) assert((await rpc("deepbom_query_file", invalid)).error);
  console.log("Private MCP query transport checks passed.");
} else console.log("Private MCP route checks omitted from this public source export; all shared query projections were exercised.");
console.log("Evidence query checks passed: six formats, conserved pagination, exact metric projections, source-bound placement, fusion boundary, digests and unsupported requests.");

function contextFor(raw, filename, size) {
  raw.file_size_bytes = size;
  const artifactIrContext = getArtifactIrContext(raw, { filename, sha256: raw.model_sha256, format: raw.format, size });
  const analysis = artifactIrContext.primary_view;
  const envelope = buildArtifactEvidenceEnvelope(analysis, { hash: raw.model_sha256, fileSizeBytes: size, filename });
  const summary = buildReviewSummary({ analysis, envelope, artifactIrContext });
  return { analysis, artifactIrContext, summary, envelope };
}
