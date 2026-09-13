#!/usr/bin/env node
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { validateArtifactEvidenceEnvelope } from "../../web/lib/artifact-evidence-envelope.js";

const options = parseOptions(process.argv.slice(2));
for (const required of ["evidence", "output", "reviewer", "disposition", "reviewed-at"]) if (!options[required]) throw new Error(`--${required} is required.`);
if (!/^(accepted_for_engineering_use|rejected|follow_up_required)$/.test(options.disposition)) throw new Error("Unsupported disposition.");
if (!Number.isFinite(Date.parse(options["reviewed-at"]))) throw new Error("--reviewed-at must be an ISO date-time.");
const evidence = JSON.parse(await readFile(path.resolve(options.evidence), "utf8"));
const validation = validateArtifactEvidenceEnvelope(evidence);
if (!validation.valid) throw new Error(`Invalid evidence envelope: ${validation.errors.join(", ")}`);
const record = {
  schema: "deepbom.engineering_evidence_review_record.v1",
  artifact: { filename: evidence.identity.filename, sha256: evidence.identity.sha256 },
  evidence: { schema: evidence.schema, sha256: evidence.envelope_sha256, location: options.evidence },
  review: {
    reviewer: options.reviewer,
    reviewed_at: new Date(options["reviewed-at"]).toISOString(),
    disposition: options.disposition,
    ...(options.notes ? { notes: options.notes } : {}),
  },
  limitations: [
    "The record covers static artifact evidence and the stated reviewer disposition only.",
    "It is not a determination of safety, effectiveness, clinical validity, legal compliance, or regulatory conformity.",
  ],
};
const output = path.resolve(options.output);
await mkdir(path.dirname(output), { recursive: true });
await writeFile(output, `${JSON.stringify(record, null, 2)}\n`, "utf8");

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
