import assert from "node:assert/strict";
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

console.log("BOM reconciliation and artifact-derived interface baseline checks passed.");

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
