/** Example application policy; DEEPBOM's engine supplies every model fact. */
import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import { inspect, diff, captureContract, verifyContract } from "deepbom";

const directory = path.resolve(process.argv[2] || ".local-validation/sdk-examples");
const inputs = JSON.parse(await readFile(path.join(directory, "inputs.json"), "utf8"));
// Keep each completed decision with its own self-contained ONNX inputs and
// evidence. A failed rerun cannot leave an old decision looking like a new one.
await mkdir(path.join(directory, "release-reviews"), { recursive: true });
const bundle = await mkdtemp(path.join(directory, "release-reviews/run-"));
for (const name of ["baseline", "candidate", "incompatible"]) {
  const bytes = await readFile(path.join(directory, inputs[name].file));
  assert.equal(digest(bytes), inputs[name].sha256, `${name} input changed`);
  await writeFile(path.join(bundle, `${name}.onnx`), bytes, { flag: "wx" });
}
const baselinePath = path.join(bundle, "baseline.onnx");
const baseline = await inspect(baselinePath, { scan: "full", expectedSha256: inputs.baseline.sha256 });
await writeJson(path.join(bundle, "baseline.summary.json"), baseline);
const contractPath = path.join(bundle, "baseline.contract.json");
await writeJson(contractPath, await captureContract(baselinePath));
// Captured contracts bind the baseline hash. Verify that baseline itself;
// candidate comparison uses the engine's semantic diff, not a weakened hash check.
const baselineVerification = await verifyContract(baselinePath, contractPath);
assert.equal(baselineVerification.artifact.sha256, baseline.artifact.sha256);
await writeJson(path.join(bundle, "baseline.verification.json"), baselineVerification);
for (const name of ["candidate", "incompatible"]) {
  const candidatePath = path.join(bundle, `${name}.onnx`);
  const summary = await inspect(candidatePath, { scan: "full", expectedSha256: inputs[name].sha256 });
  const comparison = await diff(baselinePath, candidatePath);
  // A file replacement between SDK calls must not silently change the subjects.
  assert.equal(comparison.baseline.sha256, baseline.artifact.sha256);
  assert.equal(comparison.candidate.sha256, summary.artifact.sha256);
  assert.equal(comparison.schema, "deepbom.semantic_artifact_diff.v1");
  assert.equal(typeof comparison.graph_delta.input_contract_changed, "boolean");
  assert.equal(typeof comparison.graph_delta.output_contract_changed, "boolean");
  const interfaceBlocked = comparison.graph_delta.input_contract_changed || comparison.graph_delta.output_contract_changed;
  // This deliberately conservative policy belongs to this application example.
  // Static success alone cannot release even the compatible candidate.
  const hasDefects = summary.verdict.artifact_defect_count > 0;
  const disposition = interfaceBlocked || hasDefects ? "reject" : "hold_for_external_evaluation";
  const reasons = [];
  if (interfaceBlocked) reasons.push("Candidate contradicts the captured baseline interface.");
  if (hasDefects) reasons.push("A static artifact defect was observed.");
  if (comparison.graph_delta.interface_comparison_status !== "assessed_serialized_contracts") reasons.push("Serialized interface comparison is not assessable.");
  if (comparison.storage_delta.payload_comparison.unassessed_object_count) reasons.push("Some matched storage payloads have no paired digest; equal weights are not established.");
  reasons.push("Task quality, output equivalence and runtime requirements require separate evidence.");
  await writeJson(path.join(bundle, `${name}.comparison.json`), comparison);
  await writeJson(path.join(bundle, `${name}.summary.json`), summary);
  const evidenceFiles = [];
  for (const file of ["baseline.onnx", `${name}.onnx`, "baseline.summary.json", "baseline.contract.json", "baseline.verification.json", `${name}.summary.json`, `${name}.comparison.json`]) {
    evidenceFiles.push({ file, sha256: digest(await readFile(path.join(bundle, file))) });
  }
  await writeJson(path.join(bundle, `${name}.decision.json`), {
    example_policy: "static-interface-review-v1", disposition,
    baseline_sha256: baseline.artifact.sha256, candidate_sha256: summary.artifact.sha256,
    external_evaluation: "not_provided", actual_deployment: "not_performed",
    evidence_files: evidenceFiles, reasons,
    interface_comparison_status: comparison.graph_delta.interface_comparison_status,
    payload_comparison: comparison.storage_delta.payload_comparison,
  });
  assert.equal(disposition, name === "incompatible" ? "reject" : "hold_for_external_evaluation");
  console.log(`${name}: ${disposition}`);
}
await writeJson(path.join(bundle, "completed.json"), { status: "complete", example_policy: "static-interface-review-v1" });
console.log(`Completed review bundle: ${bundle}`);
async function writeJson(file, value) { await writeFile(file, `${JSON.stringify(value, null, 2)}\n`); }
function digest(bytes) { return createHash("sha256").update(bytes).digest("hex"); }
