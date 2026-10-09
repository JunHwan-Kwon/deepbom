import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import Ajv2020 from "ajv/dist/2020.js";
import { numericalEvidence } from "../sdk/index.mjs";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { statisticsOf, HISTOGRAM_EDGES } from "../web/lib/numerical-ir/statistics.js";
import { describeDistribution, compareDistributions } from "../web/lib/numerical-ir/distribution.js";
import { buildNumericalDetails, validateNumericalDetails, activationInventory } from "../web/lib/numerical-details.js";
import { buildActivationIr } from "../web/lib/activation-ir.js";
import { buildWeightIr } from "../web/lib/weight-ir.js";
import { analyzeOnnxModel } from "../web/onnx.js";
import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256BytesHex, sha256TextHex } from "../web/lib/sha256-sync.js";
import { seal } from "../web/lib/numerical-ir/common.js";
import { model, node, tensor, valueInfo, valueInfoWithoutShape, float32 } from "./onnx-proto-fixture.mjs";

// Oracle: sort raw values, independently compute nearest-rank percentiles and
// signs. Test signed zero, histogram boundaries and extreme finite values.
const cases = [[], [NaN, Infinity, -Infinity], [-0, 0, NaN], [3, 3, 3], [-2, -1, 0, 0, 1, 3], HISTOGRAM_EDGES];
let seed = 48731;
const random = () => ((seed = Math.imul(seed, 1664525) + 1013904223 >>> 0) / 2 ** 32);
for (let i = 0; i < 180; i++) cases.push(Array.from({ length: 1 + Math.floor(random() * 200) }, () => random() < .3 ? 0 : (random() - .5) * 2 ** Math.floor(random() * 2098 - 1074)));
for (const values of cases) {
  const s = statisticsOf(values), d = describeDistribution(s), sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  assert.equal(d.sign_counts.negative, String(sorted.filter(v => v < 0).length));
  assert.equal(d.sign_counts.positive, String(sorted.filter(v => v > 0).length));
  assert.equal(d.sign_counts.zero, String(sorted.filter(v => v === 0).length));
  for (const q of d.quantiles.intervals) {
    const actual = sorted[Math.ceil(q.percent * sorted.length / 100) - 1];
    assert(actual >= q.lower && (q.upper_closed ? actual <= q.upper : actual < q.upper), JSON.stringify({ actual, q }));
    if (q.exact) assert(actual === q.lower);
  }
  assert.equal(d.quantiles.intervals.length, sorted.length ? 7 : 0);
  assert.equal(d.all_values_zero, values.length ? values.every(v => v === 0) : null);
}
const unsafe = describeDistribution(statisticsOf([9007199254740993n, 0n]));
assert.equal(unsafe.sign_counts.negative, null); assert.equal(unsafe.sign_counts.zero, "1");
assert.equal(unsafe.quantiles.reason, "unsafe_integer_histogram_unavailable");
assert.equal(unsafe.zero_fraction_of_finite.value, .5);
assert.equal(describeDistribution(statisticsOf([])).finite_fraction.value, null);
const compare = (a, b) => compareDistributions(statisticsOf(a), statisticsOf(b));
assert.equal(compare([0, 1], [0, 0, 1, 1]).histogram_total_variation.value, 0);
assert.equal(compare([0, 0], [1, 1]).histogram_total_variation.value, 1);
assert.equal(compare([0, 1, 1], [0, 0, 1]).histogram_total_variation.value, 1 / 3);
assert.equal(compare([1.25], [1.5]).histogram_total_variation.value, 0, "same bin is not equal raw values");
assert.equal(compare([NaN], [1]).reason, "no_finite_values");
assert.equal(compare([9007199254740993n], [1]).reason, "histogram_unavailable");
assert.equal(compare([-Number.MAX_VALUE], [Number.MAX_VALUE]).mean_delta, null, "delta overflow stays unknown");

const bytes = new Uint8Array(model([node("Add", "first", ["x", "w"], ["middle"]), node("Add", "second", ["middle", "w"], ["y"])], [tensor("w", 1, [4], float32([-2, 0, 1, 3]))], [valueInfo("x", 1, [4])], [valueInfo("y", 1, [4])], 13));
const analysis = analyzeOnnxModel(bytes, "details.onnx"); analysis.model_sha256 = sha256BytesHex(bytes);
const ir = getArtifactIrContext(analysis, { filename: "details.onnx", format: "onnx", sha256: analysis.model_sha256, size: bytes.length }).model_ir;
const weight = await buildWeightIr(ir, analysis, bytes), original = canonicalJson(ir);
const value = name => ir.program.values.find(v => v.name === name);
const row = (name, values) => ({ value_ref: value(name).id, native_locator: name, dtype: "FLOAT32", shape: [4], values });
const config = { optimization: "disabled" };
const capture = { schema: "deepbom.activation_capture.v1", source: weight.source, run: { id: "a", entry_region_ref: value("x").region_ref, started_at: "2026-10-07T00:00:00Z", runtime: { name: "fixture", version: "1", configured_providers: ["CPU"], device: null }, collector: { name: "fixture", version: "1", sha256: "a".repeat(64) }, execution: { artifact_sha256: analysis.model_sha256, instrumented_artifact_sha256: null, configuration: config, configuration_sha256: sha256TextHex(canonicalJson(config)) }, probe: { kind: "synthetic_ones", description: "Synthetic fixture; not an actual execution" }, runtime_evidence: null }, inputs: [row("x", [1, 1, 1, 1])], requested_value_refs: [value("middle").id, value("y").id], captures: [row("middle", [0, 1, 2, 4])], missing: [{ value_ref: value("y").id, reason: "not_preserved" }] };
const a = buildActivationIr(ir, capture), currentCapture = structuredClone(capture);
currentCapture.run.id = "b"; currentCapture.captures[0].values = [1, 1, 1, 1];
const b = buildActivationIr(ir, currentCapture);
const evidence = { weightIr: weight, activationIr: b, baselineActivationIr: a }, details = buildNumericalDetails(ir, evidence);
validateNumericalDetails(details, ir, evidence);
const shape = new Ajv2020({ strict: true, allErrors: true });
shape.addSchema(JSON.parse(await readFile("docs/schemas/deepbom-numerical-ir-v1.schema.json", "utf8")));
const validateShape = shape.compile(JSON.parse(await readFile("docs/schemas/deepbom-numerical-details-v1.schema.json", "utf8")));
for (const document of [details, buildNumericalDetails(ir, { weightIr: weight }), buildNumericalDetails(ir, { activationIr: a })]) assert(validateShape(document), JSON.stringify(validateShape.errors));
assert.equal(validateShape({ ...details, unexpected: true }), false);
assert.equal(details.weights[0].distribution.sign_counts.negative, "1");
assert.equal(details.activation_comparison.same_inputs, true);
assert.equal(details.activation_comparison.coverage.compared_count, 1);
assert.equal(details.activation_comparison.coverage.not_assessed_count, 1);
assert.equal(details.activation_comparison.tensors[0].histogram_total_variation.value, .75);
const inventory = activationInventory(ir, b);
assert.equal(inventory.length, 3); assert.equal(inventory.find(r => r.capture_status === "missing").statistics, null);
assert.equal(details.activation.flow.length, 2);
assert.equal(details.activation.flow[0].ports.find(p => p.value_ref === value("x").id).capture_status, "input");
assert.equal(details.activation.flow[0].ports.find(p => p.value_ref === value("w").id).capture_status, "stored_value");
assert.equal(details.activation.flow[1].ports.find(p => p.value_ref === value("y").id).capture_status, "missing");
assert.equal(canonicalJson(ir), original);
const unrequested = structuredClone(capture); unrequested.missing = []; unrequested.requested_value_refs.pop();
const u = buildActivationIr(ir, unrequested);
assert.equal(activationInventory(ir, u).find(r => r.value_ref === value("y").id).capture_status, "not_requested");
const union = buildNumericalDetails(ir, { activationIr: a, baselineActivationIr: u }).activation_comparison;
assert.equal(union.coverage.union_requested_count, 2); assert.equal(union.coverage.not_assessed_count, 1);
const modifiedInputs = structuredClone(capture); modifiedInputs.inputs[0].values = [0, 0, 0, 0];
assert.equal(buildNumericalDetails(ir, { activationIr: buildActivationIr(ir, modifiedInputs), baselineActivationIr: a }).activation_comparison.same_inputs, false);
const modifiedRuntime = structuredClone(capture); modifiedRuntime.run.runtime.version = "2"; modifiedRuntime.run.execution.instrumented_artifact_sha256 = "b".repeat(64);
const executionDiff = buildNumericalDetails(ir, { activationIr: buildActivationIr(ir, modifiedRuntime), baselineActivationIr: a }).activation_comparison;
assert.equal(executionDiff.same_runtime, false); assert.equal(executionDiff.same_instrumentation, false);
const unmapped = structuredClone(capture); unmapped.captures.push({ ...row("middle", [0, 0, 0, 0]), value_ref: null, native_locator: "unbound" });
const unmappedDetail = buildNumericalDetails(ir, { activationIr: buildActivationIr(ir, unmapped), baselineActivationIr: a });
assert.equal(unmappedDetail.activation_comparison.coverage.candidate_unmapped_count, 1);
assert.equal(unmappedDetail.activation_comparison.coverage.compared_count, 1, "unmapped values never match by name");
assert.throws(() => buildNumericalDetails(ir, {}));
assert.throws(() => buildNumericalDetails(ir, { weightIr: weight, baselineActivationIr: a }));
const tamper = structuredClone(details); tamper.weights[0].distribution.sign_counts.zero = "42"; delete tamper.numerical_details_sha256;
assert.throws(() => validateNumericalDetails(seal(tamper, "numerical_details_sha256"), ir, evidence));
assert.throws(() => buildNumericalDetails(ir, { activationIr: { ...a, source: { ...a.source, artifact_sha256: "0".repeat(64) } } }));

// Valid unknown-rank graph, different observed shapes: never align by length
// or silently pool two differently shaped activation tensors.
const dynamicBytes = new Uint8Array(model([node("Identity", "copy", ["x"], ["y"])], [], [valueInfoWithoutShape("x", 1)], [valueInfoWithoutShape("y", 1)], 13));
const dynamicAnalysis = analyzeOnnxModel(dynamicBytes, "dynamic.onnx"); dynamicAnalysis.model_sha256 = sha256BytesHex(dynamicBytes);
const dynamicModel = getArtifactIrContext(dynamicAnalysis, { filename: "dynamic.onnx", format: "onnx", sha256: dynamicAnalysis.model_sha256, size: dynamicBytes.length }).model_ir;
const dynamicWeight = await buildWeightIr(dynamicModel, dynamicAnalysis, dynamicBytes);
const dynamicCapture = n => {
  const d = structuredClone(capture); d.source = dynamicWeight.source; d.run.execution.artifact_sha256 = dynamicAnalysis.model_sha256;
  const makeRow = name => ({ value_ref: dynamicModel.program.values.find(v => v.name === name).id, native_locator: name, dtype: "FLOAT32", shape: [n], values: Array(n).fill(1) });
  d.inputs = [makeRow("x")]; d.captures = [makeRow("y")]; d.requested_value_refs = [d.captures[0].value_ref]; d.missing = [];
  return buildActivationIr(dynamicModel, d);
};
assert.equal(buildNumericalDetails(dynamicModel, { activationIr: dynamicCapture(2), baselineActivationIr: dynamicCapture(4) }).activation_comparison.tensors[0].reason, "shape_mismatch");

// Exercise the real CLI and reject incompatible flags before analysis.
const directory = await mkdtemp(path.join(tmpdir(), "deepbom-numerical-details-"));
try {
  const artifact = path.join(directory, "details.onnx"); await writeFile(artifact, bytes);
  const run = args => spawnSync(process.execPath, ["bin/deepbom.mjs", ...args], { encoding: "utf8", maxBuffer: 16 * 1024 * 1024 });
  const contextResult = run(["audit", artifact, "--output-format", "json", "--section", "model_ir"]);
  assert.equal(contextResult.status, 0, contextResult.stderr);
  const cliModel = JSON.parse(contextResult.stdout).sections.model_ir;
  const cliWeight = await buildWeightIr(cliModel, analysis, bytes);
  const cliPrior = { ...capture, source: cliWeight.source }, cliCurrent = { ...currentCapture, source: cliWeight.source };
  for (const [name, document] of [["prior.json", cliPrior], ["current.json", cliCurrent]]) await writeFile(path.join(directory, name), JSON.stringify(document));
  const cliA = buildActivationIr(cliModel, cliPrior), cliB = buildActivationIr(cliModel, cliCurrent);
  const cliExpected = buildNumericalDetails(cliModel, { weightIr: cliWeight, activationIr: cliB, baselineActivationIr: cliA });
  const result = run(["audit", artifact, "--output-format", "json", "--weight-analysis", "--activation-evidence", path.join(directory, "current.json"), "--activation-baseline", path.join(directory, "prior.json"), "--section", "numerical_details,activation_ir,activation_baseline_ir,weight_ir"]);
  assert.equal(result.status, 0, result.stderr);
  const output = JSON.parse(result.stdout).sections;
  assert.deepEqual(output.numerical_details, cliExpected, "CLI / common engine parity");
  assert.deepEqual(output.activation_baseline_ir, cliA);
  const fromSdk = await numericalEvidence(artifact, { weights: true, activationCapture: path.join(directory, "current.json"), activationBaseline: path.join(directory, "prior.json"), expectedSha256: analysis.model_sha256 });
  assert.deepEqual(fromSdk.sections.numerical_details, cliExpected, "Node SDK / common engine parity");
  for (const options of [{}, { weights: "yes" }, { weights: true, activationBaseline: "missing" }, { weightMapping: "mapping.json" }, { weights: true, expectedSha256: "bad" }, { weights: true, unknown: true }]) await assert.rejects(numericalEvidence(artifact, options), TypeError);
  // Run the real Python facade against this checkout's engine, bypassing only
  // wheel extraction in the harness. Arguments, execution and JSON parsing stay real.
  const py = spawnSync("python3", ["-c", `import json,sys,subprocess
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path('channels/python/src').resolve()))
import deepbom
import deepbom.api as api
real_run = subprocess.run
def run_engine(argv, **kwargs):
    return real_run([sys.argv[1], 'bin/deepbom.mjs', *argv[1:]], **kwargs)
with patch.object(api, '_verified_engine', return_value=(Path(sys.argv[1]), None)), patch.object(api.subprocess, 'run', side_effect=run_engine):
    print(json.dumps(deepbom.numerical_evidence(sys.argv[2], weights=True, activation_capture=sys.argv[3], activation_baseline=sys.argv[4])))
`, process.execPath, artifact, path.join(directory, "current.json"), path.join(directory, "prior.json")], { encoding: "utf8", maxBuffer: 16 * 1024 * 1024 });
  assert.equal(py.status, 0, py.stderr);
  assert.deepEqual(JSON.parse(py.stdout).sections.numerical_details, cliExpected, "Python SDK / common engine parity");
  for (const args of [["audit", artifact, "--activation-baseline", "absent.json"], ["diff", artifact, artifact, "--activation-evidence", path.join(directory, "current.json"), "--activation-baseline", path.join(directory, "prior.json")], ["audit", artifact, "--activation-evidence", path.join(directory, "current.json"), "--summary"]]) assert.equal(run(args).status, 1);
  assert.match(run(["--help"]).stdout, /--activation-baseline/);
} finally { await rm(directory, { recursive: true, force: true }); }
console.log("Numerical detail: 186 sorted-value percentile oracles, exact signs and TV, run identity, capture gaps, bound port flow, schema and tamper rejection; CLI / Node / Python parity passed.");
