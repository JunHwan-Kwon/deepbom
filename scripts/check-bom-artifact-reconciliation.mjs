import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { reconcileCycloneDx17Artifact } from "../web/lib/bom-artifact-reconciliation.js";
import { componentPropertyReconciliationKind } from "../web/lib/deepbom-property-taxonomy.js";
import { analyzeOnnxModel } from "../web/onnx.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { buildPublicCycloneDx17ArtifactContract } from "../web/lib/public-cyclonedx-export.js";
import { model, node, tensor, valueInfo, float32, int32 } from "./onnx-proto-fixture.mjs";
import { spawnSync } from "node:child_process";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";

import { assertCycloneDx17 } from "./cyclonedx-17-schema.mjs";

const root = process.cwd();
const scratch = path.join(root, ".local-validation", "bom-artifact-reconciliation");
const artifact = path.join(root, "web", "samples", "tiny_decoder_llm.onnx");
const bomPath = path.join(scratch, "artifact.cdx.json");
const baselinePath = path.join(scratch, "artifact.interface-contract.json");

await rm(scratch, { recursive: true, force: true });
await mkdir(scratch, { recursive: true });

run([artifact, "--output-format", "cyclonedx", "--timestamp", "2026-09-13T00:00:00Z", "-o", bomPath]);
const bom = JSON.parse(await readFile(bomPath, "utf8"));
assertCycloneDx17(bom, "standalone reconciliation fixture");

const exact = jsonRun(["verify", artifact, "--bom", bomPath, "--compact"]);
assert.equal(exact.status, 0, exact.stderr);
assert.equal(exact.document.schema, "deepbom.bom_artifact_reconciliation.v1");
assert.equal(exact.document.binding.status, "selected_by_sha256");
assert.equal(exact.document.gate_result, "pass");
assert.equal(exact.document.counts.mismatch, 0);
assert.ok(exact.document.counts.match > 0);

const declared = structuredClone(bom);
declared.metadata.component.licenses = [{ license: { name: "Test-only declaration" } }];
const declaredPath = path.join(scratch, "declared.cdx.json");
await writeFile(declaredPath, `${JSON.stringify(declared, null, 2)}\n`, "utf8");
const declarationResult = jsonRun(["verify", artifact, "--bom", declaredPath, "--compact"]);
assert.equal(declarationResult.status, 0, declarationResult.stderr);
assert.equal(declarationResult.document.comparisons.find((row) => row.field === "component.licenses")?.status,
  "not_assessable_from_artifact");

const contradicted = structuredClone(bom);
contradicted.metadata.component.type = "library";
const contradictedPath = path.join(scratch, "contradicted.cdx.json");
await writeFile(contradictedPath, `${JSON.stringify(contradicted, null, 2)}\n`, "utf8");
const contradiction = jsonRun(["verify", artifact, "--bom", contradictedPath, "--compact"]);
assert.equal(contradiction.status, 2, contradiction.stderr);
assert.equal(contradiction.document.status, "artifact_contradiction_observed");
assert.equal(contradiction.document.comparisons.find((row) => row.field === "component.type")?.status, "mismatch");

const unbound = structuredClone(bom);
unbound.metadata.component.hashes[0].content = "0".repeat(64);
const unboundPath = path.join(scratch, "unbound.cdx.json");
await writeFile(unboundPath, `${JSON.stringify(unbound, null, 2)}\n`, "utf8");
const ambiguous = jsonRun(["verify", artifact, "--bom", unboundPath, "--compact"]);
assert.equal(ambiguous.status, 3, ambiguous.stderr);
assert.equal(ambiguous.document.status, "ambiguous_subject");

const explicit = jsonRun(["verify", artifact, "--bom", unboundPath, "--component-ref", unbound.metadata.component["bom-ref"], "--compact"]);
assert.equal(explicit.status, 2, explicit.stderr);
assert.equal(explicit.document.binding.status, "selected_by_component_ref");
assert.equal(explicit.document.comparisons.find((row) => row.field === "component.hashes[SHA-256]")?.status, "mismatch");

const duplicate = structuredClone(bom);
duplicate.components = [structuredClone(duplicate.metadata.component)];
const duplicatePath = path.join(scratch, "duplicate.cdx.json");
await writeFile(duplicatePath, `${JSON.stringify(duplicate, null, 2)}\n`, "utf8");
const duplicateResult = jsonRun(["verify", artifact, "--bom", duplicatePath, "--compact"]);
assert.equal(duplicateResult.status, 3, duplicateResult.stderr);
assert.equal(duplicateResult.document.status, "ambiguous_subject");

const unrelatedDuplicate = structuredClone(bom);
unrelatedDuplicate.components = [
  { type: "library", name: "duplicate-a", "bom-ref": "duplicate-ref" },
  { type: "library", name: "duplicate-b", "bom-ref": "duplicate-ref" },
];
const unrelatedDuplicatePath = path.join(scratch, "unrelated-duplicate.cdx.json");
await writeFile(unrelatedDuplicatePath, `${JSON.stringify(unrelatedDuplicate, null, 2)}\n`, "utf8");
const unrelatedDuplicateResult = jsonRun(["verify", artifact, "--bom", unrelatedDuplicatePath, "--compact"]);
assert.equal(unrelatedDuplicateResult.status, 3, unrelatedDuplicateResult.stderr);
assert.equal(unrelatedDuplicateResult.document.binding.status, "ambiguous_subject");
assert.match(unrelatedDuplicateResult.document.binding.diagnostics[0], /Duplicate bom-ref/);

const markdown = run(["verify", artifact, "--bom", bomPath, "--render", "markdown"]);
assert.match(markdown.stdout, /\| Field \| Status \| Artifact observation \| BOM declaration \|/);
assert.match(markdown.stdout, /not a complete BOM validation/i);

run(["contract", "capture", artifact, "--timestamp", "2026-09-13T00:00:00Z", "--compact", "-o", baselinePath]);
const baseline = JSON.parse(await readFile(baselinePath, "utf8"));
assert.equal(baseline.schema, "deepbom.artifact_derived_interface_baseline.v1");
assert.equal(baseline.source_kind, "artifact_derived_baseline");
assert.match(baseline.artifact_sha256, /^[a-f0-9]{64}$/);
assert.match(baseline.analyzer.build_content_sha256, /^[a-f0-9]{64}$/);
assert.match(baseline.baseline_contract_sha256, /^[a-f0-9]{64}$/);
assert.match(baseline.interpretation_boundary, /not an approved production declaration/i);
const verifiedBaseline = jsonRun(["verify", artifact, "--contract", baselinePath, "--compact"]);
assert.equal(verifiedBaseline.status, 0, verifiedBaseline.stderr);
assert.equal(verifiedBaseline.document.comparison.status, "bound_exact_artifact_baseline");


// Independent small INT8 graph: 1*2*2*2 outputs, one multiply per output.
// Six INT8 tensors: x/y/w zero-points, w, qx, qy. Five float tensors:
// x/y and the three scales. Bias is INT32. This does not derive expectations
// from the exporter under test.
const qdqBytes = model([
  node("QuantizeLinear", "q", ["x", "xs", "xz"], ["qx"]),
  node("QLinearConv", "conv", ["qx", "xs", "xz", "w", "ws", "wz", "ys", "yz", "b"], ["qy"]),
  node("DequantizeLinear", "dq", ["qy", "ys", "yz"], ["y"]),
], [
  tensor("xs", 1, [], float32([0.5])), tensor("xz", 3, [], new Uint8Array([0])),
  tensor("w", 3, [2, 1, 1, 1], new Uint8Array([1, 2])),
  tensor("ws", 1, [2], float32([0.25, 0.5])), tensor("wz", 3, [2], new Uint8Array([0, 0])),
  tensor("ys", 1, [], float32([0.75])), tensor("yz", 3, [], new Uint8Array([0])),
  tensor("b", 6, [2], int32([4, -4])),
], [valueInfo("x", 1, [1, 1, 2, 2])], [valueInfo("y", 1, [1, 2, 2, 2])], 13);
const qdqPath = path.join(scratch, "int8-qdq.onnx");
await writeFile(qdqPath, qdqBytes);
const qdqBomPath = path.join(scratch, "int8-qdq.cdx.json");
run([qdqPath, "--output-format", "cyclonedx", "--timestamp", "2026-01-01T00:00:00Z", "-o", qdqBomPath]);
const qdqBom = JSON.parse(await readFile(qdqBomPath, "utf8"));
const qdqProperties = new Map(qdqBom.metadata.component.properties.map((entry) => [entry.name, entry.value]));
for (const [name, value] of Object.entries({
  "deepbom:model:int8TensorCount": "6",
  "deepbom:model:uint8TensorCount": "0",
  "deepbom:model:floatTensorCount": "5",
  "deepbom:model:perAxisTensorCount": "1",
  "deepbom:model:perAxisQuantizationPresent": "true",
  "deepbom:model:serializedQuantizeOperatorCount": "1",
  "deepbom:model:serializedDequantizeOperatorCount": "1",
  "deepbom:model:fullIntegerQuantized": "false",
  "mlbom:model:quantizationClassification": "integer_internal_float_io",
  "mlbom:model:quantizedComputeMacRatio": "1",
  "mlbom:model:totalMacs": "8",
})) assert.equal(qdqProperties.get(name), value, `independent QDQ observation ${name}`);

const lie = structuredClone(qdqBom);
for (const property of lie.metadata.component.properties) {
  if (property.name === "mlbom:model:quantizationClassification") property.value = "full_integer_quantized";
  if (property.name === "mlbom:model:fullIntegerQuantized") property.value = "true";
  if (property.name === "mlbom:model:quantizedComputeMacRatio") property.value = "0";
}
const liePath = path.join(scratch, "mlbom-lie.cdx.json");
await writeFile(liePath, JSON.stringify(lie));
const detected = jsonRun(["verify", qdqPath, "--bom", liePath, "--compact"]);
assert.equal(detected.status, 2, "all three falsified compatibility claims must block");
assert.equal(detected.document.counts.mismatch, 3);
assert.equal(detected.document.property_coverage.by_namespace.mlbom.mismatch, 3);

let mutationCount = 0;
for (const expected of [bom, qdqBom]) {
  const sha256 = expected.metadata.component.hashes[0].content;
  const reconcile = (document) => reconcileCycloneDx17Artifact({ bom: document, expectedBom: expected, artifact: { sha256 } });
  const same = reconcile(expected);
  assert.equal(same.result_label, "NO_CONTRADICTION");
  assert.equal(same.coverage_complete, false, "publisher/context declarations cannot be authenticated from model bytes");
  assert.equal(same.property_coverage.accounted, expected.metadata.component.properties.length);
  assert.equal(Object.values(same.property_coverage.counts).reduce((sum, n) => sum + n, 0), same.property_coverage.supplied);
  for (const [index, property] of expected.metadata.component.properties.entries()) {
    if (componentPropertyReconciliationKind(property.name) !== "artifact_observation") continue;
    const mutated = structuredClone(expected);
    mutated.metadata.component.properties[index].value = `${property.value}!falsified`;
    const result = reconcile(mutated);
    assert.equal(result.gate_result, "block", property.name);
    assert.equal(result.counts.mismatch, 1, property.name);
    mutationCount++;
  }
  for (const badFirst of [true, false]) {
    const duplicate = structuredClone(expected);
    const property = { name: "deepbom:model:int8TensorCount", value: "4242" };
    if (badFirst) duplicate.metadata.component.properties.unshift(property);
    else duplicate.metadata.component.properties.push(property);
    assert.equal(reconcile(duplicate).gate_result, "block", "no first/last-value-wins bypass");
    assert.equal(reconcile(duplicate).property_coverage.duplicate_occurrences, 1);
  }
  const unknown = structuredClone(expected);
  unknown.metadata.component.properties.push({ name: "another-vendor:claim", value: "4242" }, { name: "mlbom:future:claim", value: "false" });
  assert.equal(reconcile(unknown).property_coverage.counts.unsupported_mapping, 2);
  assert.equal(reconcile(unknown).property_coverage.accounted, unknown.metadata.component.properties.length);
  const minimal = structuredClone(expected);
  minimal.metadata.component.properties = [];
  const minimalResult = reconcile(minimal);
  assert.equal(minimalResult.gate_result, "pass", "absence is not an observed contradiction");
  assert.equal(minimalResult.result_label, "NO_CONTRADICTION");
  assert.equal(minimalResult.coverage_complete, false);
  assert.equal(minimalResult.property_coverage.compared, 0);
  assert.ok(minimalResult.counts.absent_in_bom > 0);
  const extraHash = structuredClone(expected);
  extraHash.metadata.component.hashes.push({ alg: "SHA-256", content: "0".repeat(64) });
  assert.equal(reconcile(extraHash).gate_result, "block", "one matching hash must not conceal another incorrect digest");
  extraHash.metadata.component.hashes[1].content = "invalid";
  assert.equal(reconcile(extraHash).gate_result, "block", "malformed second SHA-256 must not disappear");
  const unavailable = structuredClone(expected);
  unavailable.metadata.component.properties.push({ name: "mlbom:model:computeMacs", value: "123" });
  if (!expected.metadata.component.properties.some((property) => property.name === "mlbom:model:computeMacs")) {
    assert.equal(reconcile(unavailable).comparisons.find((row) => row.property_name === "mlbom:model:computeMacs").status, "not_assessable_from_artifact");
  }
  const wrong = structuredClone(expected);
  wrong.metadata.component.hashes[0].content = "0".repeat(64);
  assert.equal(reconcile(wrong).binding.reason, "sha256_mismatch");
  delete wrong.metadata.component.hashes;
  assert.equal(reconcile(wrong).binding.reason, "no_sha256_declaration");
  wrong.metadata.component.properties = {};
  assert.equal(reconcile(wrong).gate_result, "pending", "invalid containers must not crash");
}

// Exact decimal declarations must not compare equal after Number rounding.
for (const [name, truth, lie] of [
  ["deepbom:model:llmSerializedParameterCount", "9007199254740993", "9007199254740992"],
  ["mlbom:model:quantizedComputeMacRatio", "1", "0.99999999999999999"],
]) {
  const expected = structuredClone(qdqBom);
  expected.metadata.component.properties.find((property) => property.name === name).value = truth;
  const changed = structuredClone(expected);
  changed.metadata.component.properties.find((property) => property.name === name).value = lie;
  const result = reconcileCycloneDx17Artifact({ bom: changed, expectedBom: expected,
    artifact: { sha256: expected.metadata.component.hashes[0].content } });
  assert.equal(result.counts.mismatch, 1, `${name}: decimal precision must not be discarded`);
}

// Both producer profiles must reconcile, and changing profiles must not hide
// an observable alias that exists in the other producer.
const qdqAnalysis = analyzeOnnxModel(qdqBytes, "int8-qdq.onnx");
const qdqHash = createHash("sha256").update(qdqBytes).digest("hex");
const exportOptions = { hash: qdqHash, fileSizeBytes: qdqBytes.length, generatedAt: "2026-01-01T00:00:00Z" };
const cliProfile = buildMlBomDocument(qdqAnalysis, exportOptions);
const publicProfile = buildPublicCycloneDx17ArtifactContract(qdqAnalysis, exportOptions);
assertCycloneDx17(publicProfile, "browser reconciliation fixture");
const reconcileProfile = (document) => reconcileCycloneDx17Artifact({ bom: document, expectedBom: cliProfile,
  additionalExpectedBoms: [publicProfile], artifact: { sha256: qdqHash } });
assert.equal(reconcileProfile(publicProfile).counts.mismatch, 0);
const publicLie = structuredClone(publicProfile);
publicLie.metadata.component.properties.push({ name: "deepbom:model:int8TensorCount", value: "4242" });
assert.equal(reconcileProfile(publicLie).gate_result, "block");
for (const property of publicProfile.metadata.component.properties) {
  if (componentPropertyReconciliationKind(property.name) !== "artifact_observation") continue;
  const changed = structuredClone(publicProfile);
  changed.metadata.component.properties.find((row) => row.name === property.name).value += "!falsified";
  assert.equal(reconcileProfile(changed).gate_result, "block", property.name);
  mutationCount++;
}
// Unknown values must remain unknown in BOTH producers, not false or zero.
const unknownAnalysis = { ...qdqAnalysis, quantization_status: { label: "Not assessed", classification: "not_assessed", full_integer: null,
  quantized_compute_ops: null, compute_ops: null, quantized_compute_mac_percent: null }, per_channel_tensors: null };
for (const producer of [buildMlBomDocument, buildPublicCycloneDx17ArtifactContract]) {
  const properties = producer(unknownAnalysis, exportOptions).metadata.component.properties;
  for (const name of ["deepbom:model:fullIntegerQuantized", "deepbom:model:int8TensorCount", "deepbom:model:perAxisTensorCount",
    "deepbom:model:quantizedComputeOperatorCount", "deepbom:model:computeOperatorCount", "deepbom:model:quantizedComputeMacRatio", "mlbom:model:quantizedComputeMacRatio"]) {
    assert.equal(properties.some((property) => property.name === name), false, `unknown is not zero: ${name}`);
  }
}
// Exercise real adapters beyond ONNX, with exact per-occurrence conservation.
for (const sample of ["mobilenet_v2_1.0_224_quant.tflite", "tinymqa1m.Q4_0.gguf", "nanofable-1m-fp16.safetensors", "MNISTClassifier.mlmodel"]) {
  const samplePath = path.join(root, "web/samples", sample);
  const sampleBomPath = path.join(scratch, `${sample}.cdx.json`);
  run([samplePath, "--output-format", "cyclonedx", "--timestamp", "2026-01-01T00:00:00Z", "-o", sampleBomPath]);
  const sampleBom = JSON.parse(await readFile(sampleBomPath, "utf8"));
  assertCycloneDx17(sampleBom, sample);
  const roundtrip = jsonRun(["verify", samplePath, "--bom", sampleBomPath, "--compact"]);
  assert.equal(roundtrip.status, 0, `${sample}: ${roundtrip.stderr}`);
  assert.equal(roundtrip.document.counts.mismatch, 0, sample);
  assert.equal(roundtrip.document.property_coverage.accounted, sampleBom.metadata.component.properties.length, sample);
  for (const [index, property] of sampleBom.metadata.component.properties.entries()) {
    if (componentPropertyReconciliationKind(property.name) !== "artifact_observation") continue;
    const changed = structuredClone(sampleBom);
    changed.metadata.component.properties[index].value += "!falsified";
    const result = reconcileCycloneDx17Artifact({ bom: changed, expectedBom: sampleBom, artifact: roundtrip.document.artifact });
    assert.equal(result.gate_result, "block", `${sample}: ${property.name}`);
    mutationCount++;
  }
}
const summary = run(["verify", artifact, "--bom", bomPath]).stdout;
assert.match(summary, /Result: NO_CONTRADICTION/);
assert.match(summary, /BOM properties: \d+ supplied \| \d+ accounted \| \d+ compared/);
const help = run(["--help"]).stdout;
for (const spelling of ["--bom", "--expected-sha256", "--summary", "deepbom batch"]) {
  assert.ok(help.indexOf(spelling) < 2100, `${spelling} must be discoverable near the beginning of help`);
}
// New compatibility facts must opt into reconciliation before they ship.
const compatibilitySource = await readFile(path.join(root, "web/lib/report-mlbom-compat.js"), "utf8");
for (const match of compatibilitySource.matchAll(/property\("(mlbom:model:[^"]+)"/g)) {
  assert.equal(componentPropertyReconciliationKind(match[1]), "artifact_observation", match[1]);
}
console.log(`BOM reconciliation checks passed (${mutationCount} individually falsified properties, INT8 QDQ numeric oracle, duplicate/unknown declarations, both export profiles).`);


function run(args) {
  const result = spawnSync(process.execPath, ["bin/deepbom.mjs", ...args], {
    cwd: root,
    encoding: "utf8",
    env: { ...process.env, DEEPBOM_PROGRESS: "0" },
    maxBuffer: 32 * 1024 * 1024,
  });
  if (result.status !== 0) throw new Error(`deepbom ${args.join(" ")} failed (${result.status}): ${result.stderr || result.stdout}`);
  return result;
}

function jsonRun(args) {
  const result = spawnSync(process.execPath, ["bin/deepbom.mjs", ...args], {
    cwd: root,
    encoding: "utf8",
    env: { ...process.env, DEEPBOM_PROGRESS: "0" },
    maxBuffer: 32 * 1024 * 1024,
  });
  return { status: result.status, stderr: result.stderr, document: JSON.parse(result.stdout) };
}
