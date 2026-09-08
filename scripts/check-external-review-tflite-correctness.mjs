import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";

import {
  compute_delegation_repair,
  compute_deployment_frontier,
  initSync,
} from "../pkg/tflite_wasm_audit.js";
import { normalizeTfliteAnalysisContract } from "../web/lib/tflite-analysis-contract.js";

const fixtureRoot = path.resolve("corpus/external-review/fixtures");
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const manifest = JSON.parse(await readFile(path.join(fixtureRoot, "manifest.json"), "utf8"));
const quantRiskManifest = JSON.parse(await readFile(path.join(fixtureRoot, "quant-scale-risk.manifest.json"), "utf8"));
assert.equal(manifest.schema, "deepbom.external_review_tflite_fixtures.v1");
assert.deepEqual(manifest.generator, { numpy: "1.26.4", tensorflow: "2.16.1" });
assert.equal(quantRiskManifest.schema, "deepbom.external_review_quant_risk_fixture.v1");
assert.deepEqual(quantRiskManifest.generator, { numpy: "1.26.4", tensorflow: "2.11.1" });

const fixtures = new Map();
for (const row of manifest.fixtures) {
  const fixturePath = path.join(fixtureRoot, row.path);
  const bytes = await readFile(fixturePath);
  assert.equal(bytes.length, row.bytes, `${row.path} byte length changed`);
  assert.equal(createHash("sha256").update(bytes).digest("hex"), row.sha256, `${row.path} digest changed`);
  fixtures.set(row.path, { ...row, path: fixturePath });
}
{
  const row = quantRiskManifest.fixture;
  const fixturePath = path.join(fixtureRoot, row.path);
  const bytes = await readFile(fixturePath);
  assert.equal(bytes.length, row.bytes, `${row.path} byte length changed`);
  assert.equal(createHash("sha256").update(bytes).digest("hex"), row.sha256, `${row.path} digest changed`);
  fixtures.set(row.path, { ...row, path: fixturePath });
}

const stale = audit("space-to-batch-stale-shapes.tflite");
assert.equal(stale.total_macs, 589_824);
assert.equal(stale.total_macs_decimal, "589824");
assert.equal(stale.mac_confidence, "exact");
assert.equal(stale.mac_assessment.complete, true);
assert.equal(stale.mac_assessment.assessed_compute_ops, 1);
assert.equal(stale.tflite_shape_reconciliation.serialized_conflict_count, 3);
assert.deepEqual(reconciled(stale, "SpaceToBatchND"), {
  serialized_shape: [1, 10, 10, 16],
  effective_shape: [4, 10, 10, 16],
});
assert.deepEqual(reconciled(stale, "PartitionedCall:0"), {
  serialized_shape: [1, 8, 8, 16],
  effective_shape: [4, 8, 8, 16],
});
assertSummary("space-to-batch-stale-shapes.tflite", /589,824 MACs/);
assertEnvelope("space-to-batch-stale-shapes.tflite", {
  total_macs: 589_824,
  status: "assessed",
  assessed: 1,
  unassessed: 0,
});

const sixteenByEight = audit("conv-16x8.tflite");
assert.equal(sixteenByEight.total_macs, 112_896);
assert.equal(sixteenByEight.mac_confidence, "exact");
assert.equal(sixteenByEight.quantization_status.classification, "full_integer_16x8");
assert.equal(sixteenByEight.quantization_status.quantized_compute_mac_percent, 1);
assert.equal(sixteenByEight.quantization_status.quantized_compute_ops, 1);
assert.equal(sixteenByEight.quantization_status.int16_tensors, 2);
assert.equal(sixteenByEight.ops[0].quantization_state, "quantized_compute_16x8");
assert.equal(sixteenByEight.ops[0].quantized_compute_path, true);
const sixteenByEightBias = sixteenByEight.tensors.find((tensor) => tensor.name === "Conv2D");
const sixteenByEightWeight = sixteenByEight.tensors.find((tensor) => tensor.name === "Conv2D1");
assert.equal(sixteenByEightBias.dtype, "INT64");
assert.equal(sixteenByEightBias.constant_buffer, true);
assert.equal(sixteenByEightBias.quant_scales, sixteenByEightWeight.quant_scales);
assert.equal(sixteenByEightBias.zero_point_min, 0);
assert.equal(sixteenByEightBias.zero_point_max, 0);
assert.equal(sixteenByEight.estimated_int8_speedup, 1);
assert.match(sixteenByEight.estimated_int8_speedup_detail, /no 16x8 speedup is modeled/i);
assertSummary("conv-16x8.tflite", /112,896 MACs/);
assertEnvelope("conv-16x8.tflite", {
  total_macs: 112_896,
  status: "assessed",
  assessed: 1,
  unassessed: 0,
});

const dynamic = audit("dynamic-reshape.tflite");
assert.equal(Object.hasOwn(dynamic, "total_macs"), true);
assert.equal(dynamic.total_macs, null);
assert.equal(dynamic.total_macs_decimal, null);
assert.equal(dynamic.mac_confidence, "symbolic");
assert.equal(dynamic.mac_assessment.complete, false);
assert.equal(dynamic.mac_assessment.assessed_compute_ops, 0);
assert.equal(dynamic.mac_assessment.not_assessed_compute_ops, 1);
assert.equal(dynamic.mac_assessment.total_assessed_macs_decimal, "0");
assert.equal(dynamic.dynamic_shape_cost_contract.total_macs_formula_status, "exact_symbolic_integer_polynomial");
assert.equal(dynamic.dynamic_shape_cost_contract.total_macs_formula.expression, "4096*D2");
const dynamicComputeOp = dynamic.ops.find((op) => op.name === "FULLY_CONNECTED");
assert(dynamicComputeOp);
assert.equal(dynamicComputeOp.macs, null);
assert.equal(dynamicComputeOp.macs_status, "not_assessed");
assert.match(dynamicComputeOp.macs_reason, /dynamic|unknown|positive dimensions/i);
assert.equal(dynamic.ops.some((op) => Number.isFinite(op.macs) && op.macs < 0), false);
assertSummary("dynamic-reshape.tflite", /MACs not assessable/);
assertEnvelope("dynamic-reshape.tflite", {
  total_macs: null,
  status: "not_assessed_numeric_symbolic_total_available",
  assessed: 0,
  unassessed: 1,
});

const browserBoundary = normalizeTfliteAnalysisContract({
  format: "tflite",
  total_macs: undefined,
  total_macs_decimal: undefined,
  mac_assessment: { confidence: "symbolic" },
});
assert.equal(browserBoundary.total_macs, null);
assert.equal(browserBoundary.total_macs_decimal, null);
assert.equal(browserBoundary.mac_confidence, "symbolic");

const dynamicDiff = cli(["diff", fixture("dynamic-reshape.tflite").path, fixture("dynamic-reshape.tflite").path]);
assert.equal(dynamicDiff.status, 1);
assert.match(dynamicDiff.stderr, /complete numeric MAC ledgers/i);

const dynamicExplore = cli(["explore", fixture("dynamic-reshape.tflite").path]);
assert.equal(dynamicExplore.status, 1);
assert.match(dynamicExplore.stderr, /complete numeric source MAC ledger/i);

assert.throws(
  () => compute_delegation_repair(
    new Uint8Array(readFileSync(fixture("dynamic-reshape.tflite").path)),
    "dynamic-reshape.tflite",
    "android_mid_a55",
  ),
  /complete numeric primary-subgraph MAC ledger/i,
);
assert.throws(
  () => compute_deployment_frontier(
    new Uint8Array(readFileSync(fixture("dynamic-reshape.tflite").path)),
    "dynamic-reshape.tflite",
    JSON.stringify(["android_mid_a55", "rpi4_a72"]),
  ),
  /complete numeric MAC ledgers/i,
);

const quantRisk = audit("quant-scale-risk.tflite");
const maximumRiskOp = quantRisk.ops.find((op) => op.index === quantRisk.quantization_status.max_quantization_risk_op_index);
assert.equal(quantRisk.quantization_status.max_quantization_risk, "risk");
assert.equal(quantRisk.quantization_status.max_quantization_risk_op_name, "CONV_2D");
assert(maximumRiskOp);
assert.equal(maximumRiskOp.quant_risk, "risk");
assert(maximumRiskOp.quant_scale_ratio >= quantRiskManifest.fixture.expected.minimum_scale_ratio);
assert.equal(quantRisk.quantization_status.max_quantization_risk_detail, maximumRiskOp.quant_risk_detail);

console.log("External-review TFLite correctness verified: source-reconciled shapes, full 16x8 tensor contracts, symbolic MAC withholding, and maximum quantization risk are stable across CLI, envelope, diff, and redesign surfaces.");

function audit(name) {
  const run = cli(["audit", fixture(name).path, "--output-format", "json-compact"]);
  assert.equal(run.status, 0, run.stderr);
  return JSON.parse(run.stdout);
}

function assertSummary(name, expected) {
  const run = cli(["audit", fixture(name).path]);
  assert.equal(run.status, 0, run.stderr);
  assert.match(run.stdout, expected);
}

function assertEnvelope(name, expected) {
  const run = cli(["audit", fixture(name).path, "--output-format", "envelope", "--compact"]);
  assert.equal(run.status, 0, run.stderr);
  const envelope = JSON.parse(run.stdout);
  assert.equal(envelope.identity.sha256, fixture(name).sha256);
  assert.equal(envelope.graph.total_macs, expected.total_macs);
  assert.equal(envelope.graph.mac_assessment_status, expected.status);
  assert.equal(envelope.graph.mac_coverage.assessed_compute_ops, expected.assessed);
  assert.equal(envelope.graph.mac_coverage.unassessed_compute_ops, expected.unassessed);
}

function reconciled(analysis, tensorName) {
  const row = analysis.tflite_shape_reconciliation.rows.find((item) => item.tensor_name === tensorName);
  assert(row, `Missing shape-reconciliation row for ${tensorName}`);
  return { serialized_shape: row.serialized_shape, effective_shape: row.effective_shape };
}

function fixture(name) {
  const row = fixtures.get(name);
  assert(row, `Fixture ${name} is absent from the manifest`);
  return row;
}

function cli(args) {
  return spawnSync(process.execPath, ["bin/deepbom.mjs", ...args], {
    cwd: process.cwd(),
    encoding: "utf8",
    maxBuffer: 256 * 1024 * 1024,
  });
}
