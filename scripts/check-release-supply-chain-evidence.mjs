import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { spawnSync } from "node:child_process";

const root = process.cwd();
const packageDocument = JSON.parse(await readFile(path.join(root, "package.json"), "utf8"));
const scratch = path.join(root, ".local-validation", "release-supply-chain");
const release = path.join(scratch, "release");
const sbomPath = path.join(release, "deepbom-self-sbom.cdx.json");
const provenancePath = path.join(release, "deepbom-build-provenance.intoto.jsonl");
const subjectPath = path.join(release, "deepbom-test-asset.bin");
const attestationStatusPath = path.join(release, "deepbom-github-attestation-status.json");
await rm(scratch, { recursive: true, force: true });
await mkdir(release, { recursive: true });
await writeFile(subjectPath, Buffer.from("deepbom-release-subject\n"));
const commit = "a".repeat(40);
const run = spawnSync(process.execPath, [
  "scripts/build-release-supply-chain-evidence.mjs",
  "--sbom", sbomPath,
  "--provenance", provenancePath,
  "--subject-dir", release,
  "--commit", commit,
], { cwd: root, encoding: "utf8", env: { ...process.env, SOURCE_DATE_EPOCH: "0" } });
assert.equal(run.status, 0, `${run.stdout}\n${run.stderr}`);

const sbom = JSON.parse(await readFile(sbomPath, "utf8"));
assert.equal(sbom.bomFormat, "CycloneDX");
assert.equal(sbom.specVersion, "1.5");
assert.equal(sbom.metadata.tools.some((tool) => tool.name === "cli" && tool.vendor === "npm"), true);
assert.equal(sbom.metadata.component.name, "deepbom");
assert.equal(sbom.metadata.component.licenses[0].license.id, "Apache-2.0");
assert.equal(sbom.metadata.timestamp, "1970-01-01T00:00:00.000Z");
assert.equal(sbom.metadata.properties.some((item) => item.name === "deepbom:sbom:source-commit" && item.value === commit), true);
assert.equal((sbom.components || []).length > 0, true);

const statement = JSON.parse(await readFile(provenancePath, "utf8"));
assert.equal(statement._type, "https://in-toto.io/Statement/v1");
assert.equal(statement.predicateType, "https://slsa.dev/provenance/v1");
assert.deepEqual(statement.subject, [{
  name: "deepbom-test-asset.bin",
  digest: { sha256: createHash("sha256").update(await readFile(subjectPath)).digest("hex") },
}]);
assert.equal(statement.predicate.buildDefinition.internalParameters.source_commit, commit);
assert.equal(statement.predicate.runDetails.metadata.invocationId, "local-untrusted-build");
await writeFile(attestationStatusPath, `${JSON.stringify({
  schema: "deepbom.github_attestation_status.v1",
  status: "not_produced",
  reason: "github_attestations_unavailable_for_user_owned_private_repository",
  release_version: packageDocument.version,
  source_commit: commit,
  replacement_evidence: ["deepbom-build-provenance.intoto.jsonl", "deepbom-self-sbom.cdx.json", "SHA256SUMS"],
  interpretation_boundary: "The bundled provenance record is unsigned and is not an identity-backed GitHub artifact attestation.",
}, null, 2)}\n`);
const fallback = spawnSync(process.execPath, [
  "scripts/verify-published-local-provenance.mjs",
  "--release-dir", release,
  "--artifact", path.basename(subjectPath),
  "--version", packageDocument.version,
  "--commit", commit,
], { cwd: root, encoding: "utf8" });
assert.equal(fallback.status, 0, `${fallback.stdout}\n${fallback.stderr}`);

await writeFile(subjectPath, Buffer.from("tampered-release-subject\n"));
const tamperedFallback = spawnSync(process.execPath, [
  "scripts/verify-published-local-provenance.mjs",
  "--release-dir", release,
  "--artifact", path.basename(subjectPath),
  "--version", packageDocument.version,
  "--commit", commit,
], { cwd: root, encoding: "utf8" });
assert.notEqual(tamperedFallback.status, 0, "Local provenance fallback must reject a changed subject.");

console.log("Release self-SBOM and unsigned local provenance fallback contracts passed; identity-backed release attestation remains conditional on repository support.");
