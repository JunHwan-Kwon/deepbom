import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";

const options = parseOptions(process.argv.slice(2));
const releaseDir = path.resolve(required(options, "release-dir"));
const artifactName = path.basename(required(options, "artifact"));
const expectedVersion = required(options, "version");
const expectedCommit = required(options, "commit");
assert.match(expectedCommit, /^[a-f0-9]{40}$/i, "--commit must be a full Git commit hash");

const artifactBytes = await readFile(path.join(releaseDir, artifactName));
const provenance = JSON.parse(await readFile(path.join(releaseDir, "deepbom-build-provenance.intoto.jsonl"), "utf8"));
const status = JSON.parse(await readFile(path.join(releaseDir, "deepbom-github-attestation-status.json"), "utf8"));

assert.equal(provenance._type, "https://in-toto.io/Statement/v1");
assert.equal(provenance.predicateType, "https://slsa.dev/provenance/v1");
assert.equal(provenance.predicate?.buildDefinition?.externalParameters?.version, expectedVersion);
assert.equal(provenance.predicate?.buildDefinition?.internalParameters?.source_commit, expectedCommit);
const subject = provenance.subject?.find((row) => row?.name === artifactName);
assert(subject, `Local provenance does not contain subject ${artifactName}`);
assert.equal(subject.digest?.sha256, createHash("sha256").update(artifactBytes).digest("hex"));

assert.equal(status.schema, "deepbom.github_attestation_status.v1");
assert.equal(status.status, "not_produced");
assert.equal(status.reason, "github_attestations_unavailable_for_user_owned_private_repository");
assert.equal(status.release_version, expectedVersion);
assert.equal(status.source_commit, expectedCommit);
assert.match(status.interpretation_boundary, /unsigned/);
assert.match(status.interpretation_boundary, /not an identity-backed GitHub artifact attestation/);

console.log(`Published local provenance fallback passed for ${artifactName}; GitHub identity-backed attestation is explicitly recorded as unavailable.`);

function parseOptions(args) {
  const parsed = {};
  for (let index = 0; index < args.length; index += 1) {
    const token = args[index];
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const name = token.slice(2);
    const value = args[++index];
    if (!value || value.startsWith("--")) throw new Error(`--${name} requires a value.`);
    parsed[name] = value;
  }
  return parsed;
}

function required(options, name) {
  if (!options[name]) throw new Error(`--${name} is required.`);
  return options[name];
}
