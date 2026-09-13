import assert from "node:assert/strict";
import { readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { spawnSync } from "node:child_process";

import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import { validateArtifactEvidenceEnvelope } from "../web/lib/artifact-evidence-envelope.js";

const root = process.cwd();
const scratch = path.join(root, ".local-validation", "integration-kit");
await rm(scratch, { recursive: true, force: true });
const artifact = "corpus/external-review/fixtures/gguf-q4-0.gguf";
const evidencePath = path.join(scratch, "artifact.envelope.json");
runNode(["bin/deepbom.mjs", "audit", artifact, "--format", "envelope", "--compact", "--output", evidencePath]);
const evidence = JSON.parse(await readFile(evidencePath, "utf8"));
assert.equal(validateArtifactEvidenceEnvelope(evidence).valid, true);

const authoringPath = path.join(scratch, "authoring-input.json");
runNode(["integrations/bom-authoring/deepbom-evidence-adapter.mjs", "--artifact", artifact, "--output", authoringPath]);
const authoring = JSON.parse(await readFile(authoringPath, "utf8"));
assert.equal(authoring.schema, "deepbom.bom_authoring_input.v1");
assert.equal(authoring.artifact.sha256, evidence.identity.sha256);
assert.match(authoring.adapter_output_sha256, /^[a-f0-9]{64}$/);
assert.match(authoring.interpretation_boundary, /does not infer publisher declarations/);

const predicatePath = path.join(scratch, "predicate.intoto.jsonl");
runNode(["integrations/oci/build-attestation-predicate.mjs", "--evidence", evidencePath, "--output", predicatePath]);
const predicate = JSON.parse(await readFile(predicatePath, "utf8"));
assert.equal(predicate._type, "https://in-toto.io/Statement/v1");
assert.equal(predicate.subject[0].digest.sha256, evidence.identity.sha256);
assert.equal(predicate.predicate.evidence_envelope_sha256, evidence.envelope_sha256);
assert.equal(predicate.predicate.claims_excluded.includes("regulatory_conformity"), true);

const manifestPath = path.join(scratch, "model-store-manifest.json");
const manifestOutput = path.join(scratch, "model-store-result.json");
await writeFile(manifestPath, `${JSON.stringify({
  schema: "deepbom.model_store_audit_manifest.v1",
  artifacts: [{ path: artifact, expected_sha256: evidence.identity.sha256 }],
}, null, 2)}\n`);
runNode(["integrations/model-store/audit-manifest.mjs", manifestPath, manifestOutput]);
const modelStore = JSON.parse(await readFile(manifestOutput, "utf8"));
assert.equal(modelStore.artifacts[0].status, "audited");
assert.equal(modelStore.artifacts[0].artifact_sha256, evidence.identity.sha256);
assert.match(modelStore.boundary, /does not discover, upload, register, approve, or deploy/);

const qmsPath = path.join(scratch, "engineering-review.json");
runNode([
  "validation/qms/create-record.mjs",
  "--evidence", evidencePath,
  "--output", qmsPath,
  "--reviewer", "fixture-reviewer",
  "--reviewed-at", "2026-09-13T00:00:00Z",
  "--disposition", "follow_up_required",
]);
const qmsRecord = JSON.parse(await readFile(qmsPath, "utf8"));
const qmsSchema = JSON.parse(await readFile(path.join(root, "validation", "qms", "engineering-evidence-record.schema.json"), "utf8"));
const ajv = new Ajv2020({ allErrors: true, strict: false });
addFormats(ajv);
const validateQms = ajv.compile(qmsSchema);
assert.equal(validateQms(qmsRecord), true, JSON.stringify(validateQms.errors));
assert.equal(qmsRecord.artifact.sha256, evidence.identity.sha256);
assert.match(qmsRecord.limitations.join(" "), /not a determination of safety/);

const preCommit = runNode(["integrations/pre-commit/audit-staged.mjs", artifact]);
assert.match(preCommit.stdout, /DEEPBOM .* deployment-artifact audit/);
const pythonSource = await readFile(path.join(root, "integrations", "pipeline", "evidence_hook.py"), "utf8");
const pythonSyntax = spawnSync("python", ["-c", "import ast,sys; ast.parse(sys.stdin.read())"], { input: pythonSource, encoding: "utf8" });
assert.equal(pythonSyntax.status, 0, pythonSyntax.stderr);

const action = await readFile(path.join(root, ".github", "actions", "deepbom-audit", "action.yml"), "utf8");
for (const value of ["expected-sha256", "evidence_sha256", "--format envelope", "--gate", "--policy"]) assert(action.includes(value));
assert.doesNotMatch(action, /curl|wget|upload-artifact|pull-request|comment/i);
const containerDockerfile = await readFile(path.join(root, "channels", "container", "Dockerfile"), "utf8");
assert.match(containerDockerfile, /^FROM node:24\.12\.0-bookworm-slim@sha256:[a-f0-9]{64}$/m);
assert.match(containerDockerfile, /^USER node$/m);
assert.doesNotMatch(containerDockerfile, /:latest|curl|wget/i);
const integrationReadme = await readFile(path.join(root, "integrations", "README.md"), "utf8");
assert.doesNotMatch(integrationReadme, /adopted by|official integration|approved by/i);
console.log("DeepBOM integration kit passed: local action contract, authoring adapter, model-store manifest, pipeline hook, OCI predicate, pre-commit gate, and bounded QMS record.");

function runNode(args) {
  const run = spawnSync(process.execPath, args, { cwd: root, encoding: "utf8", maxBuffer: 128 * 1024 * 1024 });
  assert.equal(run.status, 0, `${args.join(" ")}\n${run.stdout}\n${run.stderr}`);
  return run;
}
