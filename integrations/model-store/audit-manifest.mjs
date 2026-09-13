#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const [manifestArg, outputArg] = process.argv.slice(2);
if (!manifestArg || !outputArg) throw new Error("Usage: audit-manifest.mjs <input-manifest.json> <result.json>");
const manifest = JSON.parse(await readFile(path.resolve(manifestArg), "utf8"));
if (manifest.schema !== "deepbom.model_store_audit_manifest.v1" || !Array.isArray(manifest.artifacts)) throw new Error("Unsupported manifest schema.");
const rows = [];
for (const item of manifest.artifacts) {
  if (!item || typeof item.path !== "string") throw new Error("Every artifact entry requires an explicit path.");
  const args = [path.join(root, "bin", "deepbom.mjs"), "audit", item.path, "--format", "envelope", "--compact"];
  if (item.expected_sha256) args.push("--expected-sha256", item.expected_sha256);
  const run = spawnSync(process.execPath, args, { cwd: process.cwd(), encoding: "utf8", maxBuffer: 128 * 1024 * 1024 });
  if (run.status !== 0) {
    rows.push({ path: item.path, status: "failed", exit_code: run.status, diagnostic: (run.stderr || "").trim() });
  } else {
    const evidence = JSON.parse(run.stdout);
    rows.push({ path: item.path, status: "audited", artifact_sha256: evidence.identity.sha256, evidence_sha256: evidence.envelope_sha256 });
  }
}
const result = {
  schema: "deepbom.model_store_audit_result.v1",
  artifacts: rows,
  boundary: "Only explicitly listed local artifacts were read. This adapter does not discover, upload, register, approve, or deploy models.",
};
const output = path.resolve(outputArg);
await mkdir(path.dirname(output), { recursive: true });
await writeFile(output, `${JSON.stringify(result, null, 2)}\n`, "utf8");
if (rows.some((row) => row.status === "failed")) process.exitCode = 1;
