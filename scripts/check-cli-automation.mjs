import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, rm } from "node:fs/promises";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import AjvDraft04 from "ajv-draft-04";
import addFormats from "ajv-formats";

import {
  buildCliCapabilities,
  buildSarifDocument,
  evaluateFindingPolicy,
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
  ["analysis", "envelope", "cyclonedx", "sarif"]);
assert.equal(capabilities.inputs.symbolic_stdin, false);
assert.equal(capabilities.automation.atomic_file_output, true);
assert.equal(capabilities.exit_codes[2].includes("policy"), true);

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

const sarif = buildSarifDocument(syntheticEnvelope, { version, policyResult: blockHigh });
assert.equal(sarif.version, "2.1.0");
assert.equal(sarif.runs[0].tool.driver.rules.length, syntheticEnvelope.findings.length);
assert.equal(sarif.runs[0].results.length, syntheticEnvelope.findings.length);
assert.equal(sarif.runs[0].results[0].level, "error");
assert.equal(sarif.runs[0].results[1].level, "note");
assert.equal(sarif.runs[0].artifacts[0].location.uri, "fixture%20model.onnx");
assert.match(sarif.runs[0].results[0].partialFingerprints["deepbomFinding/v1"], /^[a-f0-9]{64}$/);

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

const onnxPath = "web/samples/mnist-8.onnx";
const envelopeRun = run(["audit", onnxPath, "--format", "envelope", "--compact"], {
  env: { ...process.env, SOURCE_DATE_EPOCH: "0" },
});
const envelope = JSON.parse(envelopeRun.stdout);
assert.equal(envelope.schema, "deepbom.artifact_evidence_envelope.v1");
assert.equal(envelope.generated_at, "1970-01-01T00:00:00.000Z");
assert.match(envelope.envelope_sha256, /^[a-f0-9]{64}$/);

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
