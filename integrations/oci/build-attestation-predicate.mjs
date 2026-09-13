#!/usr/bin/env node
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { validateArtifactEvidenceEnvelope } from "../../web/lib/artifact-evidence-envelope.js";

const options = parseOptions(process.argv.slice(2));
if (!options.evidence || !options.output) throw new Error("--evidence and --output are required.");
const evidenceBytes = await readFile(path.resolve(options.evidence));
const evidence = JSON.parse(evidenceBytes);
const validation = validateArtifactEvidenceEnvelope(evidence);
if (!validation.valid) throw new Error(`Invalid evidence envelope: ${validation.errors.join(", ")}`);
const statement = {
  _type: "https://in-toto.io/Statement/v1",
  subject: [{ name: evidence.identity.filename, digest: { sha256: evidence.identity.sha256 } }],
  predicateType: "https://deepbom.org/attestations/artifact-evidence/v1",
  predicate: {
    evidence_schema: evidence.schema,
    evidence_envelope_sha256: evidence.envelope_sha256,
    evidence_file_sha256: createHash("sha256").update(evidenceBytes).digest("hex"),
    analyzer: evidence.provenance?.analyzer || null,
    interpretation_boundary: evidence.evidence_boundary,
    claims_excluded: ["runtime_behavior", "task_quality", "safety", "regulatory_conformity"],
  },
};
const output = path.resolve(options.output);
await mkdir(path.dirname(output), { recursive: true });
await writeFile(output, `${JSON.stringify(statement)}\n`, "utf8");
console.log(`Wrote unsigned DeepBOM attestation predicate to ${output}. Sign and bind it in the caller's trusted release environment.`);

function parseOptions(args) {
  const parsed = {};
  for (let index = 0; index < args.length; index += 2) {
    const name = args[index];
    const value = args[index + 1];
    if (!name?.startsWith("--") || !value) throw new Error("Options must be --name value pairs.");
    parsed[name.slice(2)] = value;
  }
  return parsed;
}
