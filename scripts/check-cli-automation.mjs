import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import AjvDraft04 from "ajv-draft-04";
import addFormats from "ajv-formats";

import {
  buildCliCapabilities,
  buildSarifDocument,
  evaluateDefectGate,
  evaluateFindingPolicy,
  evaluateGatePolicyProfile,
  renderCliError,
  resolveGenerationTimestamp,
  writeOutputAtomically,
} from "../bin/deepbom-automation.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const scratch = path.join(root, ".local-validation", "cli-automation");
const schemaPath = path.join(root, "reference", "sarif", "2.1.0", "sarif-schema-2.1.0.json");
const expectedSchemaSha256 = "c3b4bb2d6093897483348925aaa73af03b3e3f4bd4ca38cef26dcb4212a2682e";
const version = JSON.parse(await readFile(path.join(root, "package.json"), "utf8")).version;

await rm(scratch, { recursive: true, force: true });

const capabilities = buildCliCapabilities(version, {
  defaultTarget: "android_mid_a55",
  deltaTargets: ["android_mid_a55", "rpi4_a72"],
});
assert.equal(capabilities.schema, "deepbom.cli_capabilities.v1");
assert.deepEqual(capabilities.commands.find((row) => row.name === "audit").outputs,
  ["summary", "envelope", "json", "json-compact", "cyclonedx", "sarif"]);
assert.equal(capabilities.default_audit_output, "summary");
assert.deepEqual(capabilities.output_contracts.summary, {
  media_type: "text/plain; charset=utf-8",
  derived_from: "deepbom.review_summary.v1",
  stability: "bounded_human_projection",
});
assert.deepEqual(capabilities.output_contracts.analysis.compatibility_alias_for, ["json", "json-compact"]);
assert.equal(capabilities.inputs.symbolic_stdin, false);
assert.equal(capabilities.automation.atomic_file_output, true);
assert.deepEqual(capabilities.automation.builtin_policy_profiles.map((row) => row.id), ["engineering", "regulatory"]);
assert.equal(capabilities.exit_codes[2].includes("policy"), true);
assert.deepEqual(capabilities.commands.find((row) => row.name === "accelerator collect nvidia").outputs,
  ["deepbom.accelerator_profile.v1"]);
assert.equal(capabilities.inputs.remote_sources.huggingface, "full_commit_required");
assert.equal(capabilities.inputs.remote_sources.remote_code_execution, false);
assert.equal(capabilities.commands.find((row) => row.name === "graph").outputs.includes("deepbom.graph_ir.v1"), true);
assert.equal(capabilities.commands.find((row) => row.name === "graph").outputs.includes("deepbom.artifact_ir.v2"), true);
assert.deepEqual(capabilities.commands.find((row) => row.name === "placement").outputs,
  ["deepbom.placement_comparison.v1"]);
assert.equal(capabilities.commands.find((row) => row.name === "diff").outputs[0], "summary");
assert.deepEqual(capabilities.commands.find((row) => row.name === "integrate").targets, ["codex", "claude-code", "generic"]);
assert.equal(capabilities.accelerator_profiles.imports.litert_qualcomm_compiler_dispatch_evidence,
  "deepbom.litert_qualcomm_compiler_dispatch_evidence.v1");

const syntheticEnvelope = {
  schema: "deepbom.artifact_evidence_envelope.v1",
  identity: {
    filename: "fixture model.onnx",
    format: "onnx",
    sha256: "a".repeat(64),
    byte_length: 1024,
  },
  findings: [
    {
      id: "EA-TEST-0001",
      title: "High test finding",
      severity: "high",
      status: "open",
      evidence_class: "DERIVED",
      finding_kind: "artifact_defect",
      summary: "A deterministic high-severity fixture.",
      interpretation: "Synthetic test only.",
      recommendation: "Review the fixture.",
      source_pointers: ["/graph/operator_count"],
      rule_id: "deepbom.test.high.v1",
    },
    {
      id: "EA-TEST-0002",
      title: "Low test finding",
      severity: "low",
      status: "open",
      evidence_class: "OBSERVED",
      finding_kind: "evidence_gap",
      summary: "A deterministic low-severity fixture.",
      interpretation: null,
      recommendation: null,
      source_pointers: [],
      rule_id: "deepbom.test.low.v1",
    },
  ],
  envelope_sha256: "b".repeat(64),
  evidence_boundary: "Synthetic automation fixture.",
};

const pass = evaluateFindingPolicy(syntheticEnvelope, "none");
assert.equal(pass.status, "pass");
assert.equal(pass.finding_count, 2);
const blockHigh = evaluateFindingPolicy(syntheticEnvelope, "high");
assert.equal(blockHigh.status, "block");
assert.deepEqual(blockHigh.blocking_finding_ids, ["EA-TEST-0001"]);
const blockLow = evaluateFindingPolicy(syntheticEnvelope, "low");
assert.equal(blockLow.blocking_finding_count, 2);
const defectGate = evaluateDefectGate(syntheticEnvelope);
assert.equal(defectGate.status, "block");
assert.deepEqual(defectGate.blocking_finding_ids, ["EA-TEST-0001"]);
const engineeringGate = evaluateGatePolicyProfile(syntheticEnvelope, "engineering");
assert.deepEqual(engineeringGate.blocking_finding_ids, ["EA-TEST-0001"]);
const regulatoryGate = evaluateGatePolicyProfile(syntheticEnvelope, "regulatory");
assert.deepEqual(regulatoryGate.blocking_finding_ids, ["EA-TEST-0001", "EA-TEST-0002"]);
assert.match(regulatoryGate.interpretation_boundary, /does not determine legal compliance/);

const sarif = buildSarifDocument(syntheticEnvelope, { version, policyResult: blockHigh });
assert.equal(sarif.version, "2.1.0");
assert.equal(sarif.runs[0].tool.driver.rules.length, syntheticEnvelope.findings.length);
assert.equal(sarif.runs[0].results.length, syntheticEnvelope.findings.length);
assert.equal(sarif.runs[0].results[0].level, "error");
assert.equal(sarif.runs[0].results[1].level, "note");
assert.equal(sarif.runs[0].artifacts[0].location.uri, "fixture%20model.onnx");
assert.match(sarif.runs[0].results[0].partialFingerprints["deepbomFinding/v1"], /^[a-f0-9]{64}$/);
assert.equal(sarif.runs[0].results[1].properties.deepbomFindingKind, "evidence_gap");

const schemaBytes = await readFile(schemaPath);
assert.equal(createHash("sha256").update(schemaBytes).digest("hex"), expectedSchemaSha256,
  "The offline OASIS SARIF schema must remain source-pinned.");
const sarifSchema = JSON.parse(schemaBytes.toString("utf8"));
const ajv = new AjvDraft04({ allErrors: true, strict: false });
addFormats(ajv);
const validateSarif = ajv.compile(sarifSchema);
assert.equal(validateSarif(sarif), true, JSON.stringify(validateSarif.errors));

assert.equal(resolveGenerationTimestamp("2026-08-30T00:00:00Z"), "2026-08-30T00:00:00.000Z");
assert.equal(resolveGenerationTimestamp("", { SOURCE_DATE_EPOCH: "0" }), "1970-01-01T00:00:00.000Z");
assert.throws(() => resolveGenerationTimestamp("", { SOURCE_DATE_EPOCH: "1.5" }), /non-negative integer/);

const atomicPath = path.join(scratch, "nested", "result.json");
await writeOutputAtomically(atomicPath, "first\n", { noClobber: true });
assert.equal(await readFile(atomicPath, "utf8"), "first\n");
await assert.rejects(writeOutputAtomically(atomicPath, "second\n", { noClobber: true }), /already exists/);
assert.equal(await readFile(atomicPath, "utf8"), "first\n", "no-clobber must preserve the existing bytes");
await writeOutputAtomically(atomicPath, "second\n");
assert.equal(await readFile(atomicPath, "utf8"), "second\n");

const structuredError = JSON.parse(renderCliError(new Error("No such file or directory"), ["--error-format", "json"]));
assert.equal(structuredError.schema, "deepbom.cli_error.v1");
assert.equal(structuredError.code, "input_unavailable");

const capabilityRun = run(["capabilities", "--compact"]);
assert.equal(JSON.parse(capabilityRun.stdout).schema, "deepbom.cli_capabilities.v1");
const helpRun = run(["--help"]);
assert.doesNotMatch(helpRun.stdout, /perspective|CycloneDX 2\.0|working group|pull request/i);

const selfTest = JSON.parse(run(["self-test", "--compact"]).stdout);
assert.equal(selfTest.schema, "deepbom.cli_self_test.v1");
assert.equal(selfTest.status, "pass");
assert.equal(selfTest.checks.every((row) => row.status === "pass"), true);
const ruleIndex = JSON.parse(run(["explain-rule", "--list", "--compact"]).stdout);
assert.equal(ruleIndex.schema, "deepbom.rule_explanation_index.v1");
assert.equal(ruleIndex.rules.some((row) => row.rule_id === "onnx.conv.output-shape"), true);

const onnxPath = "web/samples/mnist-8.onnx";
const envelopeRun = run(["audit", onnxPath, "--format", "envelope", "--compact"], {
  env: { ...process.env, SOURCE_DATE_EPOCH: "0" },
});
const envelope = JSON.parse(envelopeRun.stdout);
assert.equal(envelope.schema, "deepbom.artifact_evidence_envelope.v1");
assert.equal(envelope.generated_at, "1970-01-01T00:00:00.000Z");
assert.match(envelope.envelope_sha256, /^[a-f0-9]{64}$/);
assert.equal(envelope.findings.every((finding) => ["artifact_defect", "caution", "evidence_gap"].includes(finding.finding_kind)), true);

const sectionIndex = JSON.parse(run(["audit", onnxPath, "--list-sections", "--compact"]).stdout);
assert.equal(sectionIndex.schema, "deepbom.analysis_sections.v1");
assert.equal(sectionIndex.sections.includes("summary"), true);
assert.equal(sectionIndex.sections.includes("findings"), true);
const summarySelection = JSON.parse(run(["audit", onnxPath, "--section", "summary", "--compact"], {
  env: { ...process.env, SOURCE_DATE_EPOCH: "0" },
}).stdout);
assert.equal(summarySelection.schema, "deepbom.analysis_selection.v1");
assert.equal(summarySelection.sections.summary.schema, "deepbom.review_summary.v1");
assert.equal(summarySelection.sections.summary.evidence_envelope_sha256, envelope.envelope_sha256);
assert.equal(summarySelection.sections.summary.reproduction.schema, "deepbom.reproduction_command.v1");
assert.equal(summarySelection.sections.summary.reproduction.expected_sha256, summarySelection.artifact.sha256);
assert.match(summarySelection.sections.summary.reproduction.shell_command, /npx -y deepbom@\d+\.\d+\.\d+ audit/);
assert.match(summarySelection.sections.summary.reproduction.shell_command, /--expected-sha256 [a-f0-9]{64} --summary$/);
const pointerSelection = JSON.parse(run(["audit", onnxPath, "--pointer", "/format", "--compact"]).stdout);
assert.equal(pointerSelection.schema, "deepbom.analysis_pointer_result.v1");
assert.equal(pointerSelection.value, "onnx");
const stableCycloneDx = JSON.parse(run(["audit", onnxPath, "--output-format", "cyclonedx", "--compact"]).stdout);
assert.equal(stableCycloneDx.bomFormat, "CycloneDX");
assert.equal(stableCycloneDx.specVersion, "1.7");

const safeTensorsPath = path.join(scratch, "bounded-summary.safetensors");
await writeFile(safeTensorsPath, safeTensorsFixture());
const safeTensorsSummaryRun = run(["audit", safeTensorsPath, "--output-format", "summary"]);
assert.match(safeTensorsSummaryRun.stdout, /Storage: 2 tensors \| 16 B \| F32 1, I32 1 \| integrity assessed/);
assert.match(safeTensorsSummaryRun.stdout, /Graph: executable graph not serialized by this artifact format/);
const safeTensorsSummary = JSON.parse(run(["audit", safeTensorsPath, "--section", "summary", "--compact"]).stdout).sections.summary;
assert.deepEqual(safeTensorsSummary.storage.encodings.map((row) => [row.dtype, row.tensor_count]), [["F32", 1], ["I32", 1]]);
assert.equal(safeTensorsSummary.storage.declared_tensor_bytes, 16);
assert.equal(safeTensorsSummary.storage.numerical_integrity_status, "assessed");
assert.equal(safeTensorsSummary.storage.quantization_contract_status, "not_applicable_no_quantization_declaration");

const sarifPath = path.join(scratch, "mnist.sarif");
const sarifRun = run(["audit", onnxPath, "--format", "sarif", "--output", sarifPath]);
assert.equal(sarifRun.stdout, "", "file output must keep stdout clean");
const cliSarif = JSON.parse(await readFile(sarifPath, "utf8"));
assert.equal(validateSarif(cliSarif), true, JSON.stringify(validateSarif.errors));
assert.equal(cliSarif.runs[0].results.length, envelope.findings.length);

const noClobberRun = run(["capabilities", "--output", sarifPath, "--no-clobber"], { expectSuccess: false });
assert.equal(noClobberRun.status, 1);
assert.match(noClobberRun.stderr, /Output already exists/);

for (const args of [
  ["capabilities", "--timestamp", "2026-08-30T00:00:00Z"],
  ["capabilities", "--output", "-", "--no-clobber"],
  ["audit", onnxPath, "--timestamp", "2026-08-30T00:00:00Z"],
  ["audit", onnxPath, "--json", "--compact"],
  ["audit", onnxPath, "--fail-on", "high", "--policy-output", "-"],
]) {
  const rejected = run(args, { expectSuccess: false });
  assert.equal(rejected.status, 1, `${args.join(" ")} must reject an ineffective or ambiguous option combination`);
  assert.match(rejected.stderr, /does not accept|cannot protect stdout|applies only|mutually exclusive|requires a file path/);
}

const policyPath = path.join(scratch, "policy.json");
const policyRun = run(["audit", onnxPath, "--fail-on", "high", "--policy-output", policyPath, "--compact"], {
  expectSuccess: false,
});
assert.equal(policyRun.status, 2);
assert.doesNotThrow(() => JSON.parse(policyRun.stdout), "policy failure must preserve machine-readable stdout");
const policy = JSON.parse(await readFile(policyPath, "utf8"));
assert.equal(policy.schema, "deepbom.cli_finding_policy_result.v1");
assert.equal(policy.status, "block");
assert.equal(policy.blocking_finding_count > 0, true);

const defectPolicyPath = path.join(scratch, "defect-policy.json");
const defectRun = run(["audit", onnxPath, "--gate", "defects", "--policy-output", defectPolicyPath, "--compact"]);
assert.doesNotThrow(() => JSON.parse(defectRun.stdout));
const defectPolicy = JSON.parse(await readFile(defectPolicyPath, "utf8"));
assert.equal(defectPolicy.schema, "deepbom.cli_defect_gate_result.v1");
assert.equal(defectPolicy.finding_kind_counts.evidence_gap > 0, true);

const regulatoryPolicyPath = path.join(scratch, "regulatory-policy.json");
const regulatoryRun = run(["audit", onnxPath, "--policy", "regulatory", "--policy-output", regulatoryPolicyPath, "--compact"], { expectSuccess: false });
assert.equal(regulatoryRun.status, 2);
assert.doesNotThrow(() => JSON.parse(regulatoryRun.stdout));
const regulatoryPolicy = JSON.parse(await readFile(regulatoryPolicyPath, "utf8"));
assert.equal(regulatoryPolicy.profile, "regulatory");
assert.equal(regulatoryPolicy.finding_kind_counts.evidence_gap > 0, true);
assert.equal(regulatoryPolicy.blocking_finding_ids.includes("EA-LIM-0001"), true);
const engineeringRun = run(["audit", onnxPath, "--policy", "engineering", "--compact"]);
assert.equal(engineeringRun.status, 0);

const unchangedDelta = JSON.parse(run([
  "diff",
  "web/samples/mobilenet_v1_025_224_float.tflite",
  "web/samples/mobilenet_v1_025_224_float.tflite",
  "--compact",
]).stdout);
const unchangedDeltaSummary = run([
  "diff",
  "web/samples/gpu_partition_probe.onnx",
  "web/samples/gpu_partition_probe.onnx",
  "--summary",
]).stdout;
assert.match(unchangedDeltaSummary, /deterministic semantic artifact diff/);
assert.match(unchangedDeltaSummary, /Graph: 5 matched \| 0 changed/);
assert.equal(unchangedDelta.change_impact.schema, "deepbom.change_impact.v1");
assert.equal(unchangedDelta.change_impact.highest_action, "no_change_observed");

const missingRun = run(["audit", "missing.onnx", "--error-format", "json"], { expectSuccess: false });
assert.equal(missingRun.status, 1);
assert.equal(JSON.parse(missingRun.stderr).code, "input_unavailable");

console.log("CLI automation checks passed (capability discovery, canonical envelope, OASIS SARIF 2.1.0, finding gate, structured stderr, reproducible timestamp, and atomic output).");

function run(args, { expectSuccess = true, env = process.env } = {}) {
  const result = spawnSync(process.execPath, ["bin/deepbom.mjs", ...args], {
    cwd: root,
    encoding: "utf8",
    env,
    maxBuffer: 128 * 1024 * 1024,
  });
  if (expectSuccess && result.status !== 0) {
    throw new Error(`CLI failed: ${args.join(" ")}\n${result.stdout}\n${result.stderr}`);
  }
  return result;
}

function safeTensorsFixture() {
  const header = Buffer.from(JSON.stringify({
    weight: { dtype: "F32", shape: [2], data_offsets: [0, 8] },
    index: { dtype: "I32", shape: [2], data_offsets: [8, 16] },
  }), "utf8");
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(header.length));
  const payload = Buffer.alloc(16);
  payload.writeFloatLE(0.25, 0);
  payload.writeFloatLE(-0.5, 4);
  payload.writeInt32LE(3, 8);
  payload.writeInt32LE(7, 12);
  return Buffer.concat([prefix, header, payload]);
}
