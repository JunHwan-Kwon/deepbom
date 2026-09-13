#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const options = parseOptions(process.argv.slice(2));
if (!options.artifact) throw new Error("--artifact is required.");
const args = [path.join(root, "bin", "deepbom.mjs"), "audit", options.artifact, "--format", "envelope", "--compact"];
if (options["expected-sha256"]) args.push("--expected-sha256", options["expected-sha256"]);
const run = spawnSync(process.execPath, args, { cwd: process.cwd(), encoding: "utf8", maxBuffer: 128 * 1024 * 1024 });
if (run.status !== 0) {
  process.stderr.write(run.stderr || "DeepBOM evidence generation failed.\n");
  process.exit(run.status ?? 1);
}
const envelope = JSON.parse(run.stdout);
const document = {
  schema: "deepbom.bom_authoring_input.v1",
  artifact: envelope.identity,
  observed: {
    interfaces: envelope.interfaces,
    graph: envelope.graph,
    metadata: envelope.metadata,
    format_extensions: envelope.format_extensions,
  },
  findings: summarizeFindings(envelope.findings || []),
  evidence: {
    schema: envelope.schema,
    sha256: envelope.envelope_sha256,
    canonical_document: options.output ? path.basename(options.output).replace(/\.json$/i, ".envelope.json") : null,
  },
  interpretation_boundary: "This adapter transports artifact-observable DeepBOM evidence into an authoring pipeline. It does not infer publisher declarations or make the resulting BOM complete or conformant.",
};
document.adapter_output_sha256 = createHash("sha256").update(JSON.stringify(document)).digest("hex");
if (options.output) {
  const output = path.resolve(options.output);
  await mkdir(path.dirname(output), { recursive: true });
  await writeFile(output, `${JSON.stringify(document, null, 2)}\n`, "utf8");
  const envelopePath = output.replace(/\.json$/i, ".envelope.json");
  await writeFile(envelopePath, `${JSON.stringify(envelope, null, 2)}\n`, "utf8");
} else {
  process.stdout.write(`${JSON.stringify(document, null, 2)}\n`);
}

function summarizeFindings(findings) {
  const counts = { artifact_defect: 0, caution: 0, evidence_gap: 0 };
  for (const finding of findings) if (Object.hasOwn(counts, finding.finding_kind)) counts[finding.finding_kind] += 1;
  return { counts, ids: findings.map((item) => item.id).sort() };
}

function parseOptions(args) {
  const parsed = {};
  for (let index = 0; index < args.length; index += 1) {
    const token = args[index];
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const value = args[++index];
    if (!value || value.startsWith("--")) throw new Error(`${token} requires a value.`);
    parsed[token.slice(2)] = value;
  }
  return parsed;
}
