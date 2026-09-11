import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";

const scratch = path.resolve(".local-validation/external-review-defect-gate");
await rm(scratch, { recursive: true, force: true });
await mkdir(scratch, { recursive: true });

const nanPath = path.join(scratch, "nan-weight.safetensors");
const zeroPath = path.join(scratch, "zero-weight.safetensors");
const nanBytes = safetensorsF32("weight", [Number.NaN, 1]);
const zeroBytes = safetensorsF32("weight", [0, 0]);
await writeFile(nanPath, nanBytes);
await writeFile(zeroPath, zeroBytes);

assert.equal(
  createHash("sha256").update(nanBytes).digest("hex"),
  "e65661c71d739ca23dfe9e6f1f83af86323c14d072a62895a5762e2d8efeaaef",
  "The external-review NaN fixture identity changed.",
);

const envelopeRun = run(["audit", nanPath, "--output-format", "envelope", "--compact"]);
assert.equal(envelopeRun.status, 0, envelopeRun.stderr);
const envelope = JSON.parse(envelopeRun.stdout);
const defect = envelope.findings.find((finding) => finding.id === "EA-SER-0001");
assert(defect, "The full-payload NaN finding is missing from the canonical envelope.");
assert.equal(defect.finding_kind, "artifact_defect");
assert.equal(defect.severity, "high");
assert.equal(defect.evidence_class, "DERIVED");
assert.equal(defect.source_evidence_class, "OBSERVED/DERIVED");

const summaryRun = run(["audit", nanPath]);
assert.equal(summaryRun.status, 0, summaryRun.stderr);
assert.match(summaryRun.stdout, /\b1 Artifact defects\b/, "The human summary must surface the serialized defect.");

const sarifRun = run(["audit", nanPath, "--output-format", "sarif", "--compact"]);
assert.equal(sarifRun.status, 0, sarifRun.stderr);
const sarif = JSON.parse(sarifRun.stdout);
const sarifResult = sarif.runs[0].results.find((result) => result.ruleId === "EA-SER-0001");
assert(sarifResult, "SARIF must retain EA-SER-0001.");
assert.equal(sarifResult.properties.deepbomFindingKind, "artifact_defect");

const gateRun = run(["audit", nanPath, "--output-format", "envelope", "--gate", "defects", "--compact"]);
assert.equal(gateRun.status, 2, "A serialized NaN must block the artifact-defect gate.");
assert.doesNotThrow(() => JSON.parse(gateRun.stdout), "A blocked gate must preserve valid structured output.");
assert.match(gateRun.stderr, /review policy blocked/);

const zeroEnvelopeRun = run(["audit", zeroPath, "--output-format", "envelope", "--compact"]);
assert.equal(zeroEnvelopeRun.status, 0, zeroEnvelopeRun.stderr);
const zeroEnvelope = JSON.parse(zeroEnvelopeRun.stdout);
assert.equal(zeroEnvelope.findings.some((finding) => finding.id === "EA-SER-0001"), false);
const zeroStructureReview = zeroEnvelope.findings.find((finding) => finding.id === "EA-SER-0002");
assert(zeroStructureReview, "The all-zero review signal should remain visible.");
assert.equal(zeroStructureReview.title, "Exact all-zero serialized tensor payloads detected");
assert.match(zeroStructureReview.summary, /1\/1 fully decoded tensor payload/);
assert.equal(zeroStructureReview.finding_kind, "caution", "All-zero storage is not an unconditional artifact defect.");
const zeroGateRun = run(["audit", zeroPath, "--gate", "defects", "--compact"]);
assert.equal(zeroGateRun.status, 0, "An all-zero tensor without non-finite values must not be blocked as a defect.");

console.log("External-review defect gate verified: NaN blocks, SARIF preserves the finding, and all-zero storage remains a caution.");

function safetensorsF32(name, values) {
  const payload = Buffer.alloc(values.length * 4);
  values.forEach((value, index) => payload.writeFloatLE(value, index * 4));
  const metadata = {
    [name]: {
      dtype: "F32",
      shape: [values.length],
      data_offsets: [0, payload.length],
    },
  };
  const rawHeader = Buffer.from(JSON.stringify(metadata), "utf8");
  const padding = (8 - (rawHeader.length % 8)) % 8;
  const header = Buffer.concat([rawHeader, Buffer.alloc(padding, 0x20)]);
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(header.length));
  return Buffer.concat([length, header, payload]);
}

function run(args) {
  return spawnSync(process.execPath, ["bin/deepbom.mjs", ...args], {
    cwd: process.cwd(),
    encoding: "utf8",
    maxBuffer: 256 * 1024 * 1024,
  });
}
