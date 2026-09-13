import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";

const root = path.resolve(".");
const manifestPath = path.join(root, "config/model-ir-upstream-sources.v1.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
const sources = manifest.sources.filter((row) => Array.isArray(row.files) && row.files.length);
const args = process.argv.slice(2);
const rootIndex = args.indexOf("--clone-root");
const cloneRoot = rootIndex >= 0 ? path.resolve(args[rootIndex + 1] || "") : null;
const networkClone = args.includes("--clone");
const outputIndex = args.indexOf("--output");
const output = outputIndex >= 0 ? path.resolve(args[outputIndex + 1] || "") : null;
assert(!(cloneRoot && networkClone), "Use either --clone-root or --clone.");

let temporary = null;
let verificationRoot = cloneRoot;
if (networkClone) {
  temporary = await mkdtemp(path.join(os.tmpdir(), "deepbom-model-ir-upstream-"));
  verificationRoot = temporary;
}

const results = [];
try {
  for (const source of sources) {
    const directoryName = source.id === "tensorflow_graphdef_savedmodel" ? "tensorflow" : source.id;
    const checkout = verificationRoot ? path.join(verificationRoot, directoryName) : null;
    if (networkClone) clonePinnedSource(source, checkout);
    const row = {
      id: source.id,
      repository: source.repository,
      commit: source.commit,
      mode: verificationRoot ? "pinned_git_checkout_and_file_hashes" : "manifest_contract_only",
      files: [],
    };
    if (verificationRoot) {
      assert.equal(git(checkout, ["rev-parse", "HEAD"]), source.commit, `${source.id}: checkout commit mismatch`);
    }
    for (const file of source.files) {
      assert.match(file.path, /^(?!\/)(?!.*\.\.\/)[A-Za-z0-9_./-]+$/);
      assert.match(file.sha256, /^[a-f0-9]{64}$/);
      assert(Array.isArray(file.markers) && file.markers.length > 0);
      if (!verificationRoot) {
        row.files.push({ path: file.path, expected_sha256: file.sha256, status: "recorded_not_retrieved" });
        continue;
      }
      const bytes = await readFile(path.join(checkout, ...file.path.split("/")));
      const digest = createHash("sha256").update(bytes).digest("hex");
      assert.equal(digest, file.sha256, `${source.id}:${file.path}: SHA-256 mismatch`);
      const text = bytes.toString("utf8");
      for (const marker of file.markers) assert(text.includes(marker), `${source.id}:${file.path}: missing marker ${marker}`);
      row.files.push({ path: file.path, expected_sha256: file.sha256, observed_sha256: digest, marker_count: file.markers.length, status: "verified" });
    }
    results.push(row);
  }
  const document = {
    schema: "deepbom.model_ir_preview_source_verification.v1",
    manifest: "config/model-ir-upstream-sources.v1.json",
    verification_mode: verificationRoot ? "pinned_git_checkout_and_file_hashes" : "manifest_contract_only",
    sources: results,
    interpretation_boundary: "This record verifies immutable upstream source identity, selected file bytes, and bounded semantic markers. It does not establish upstream endorsement, runtime behavior, regulatory approval, or standards conformance.",
  };
  if (output) await writeFile(output, `${JSON.stringify(document, null, 2)}\n`, "utf8");
  console.log(`Model IR preview source verification passed (${results.length} source families; ${results.reduce((sum, row) => sum + row.files.length, 0)} pinned files; ${document.verification_mode}).`);
} finally {
  if (temporary) await rm(temporary, { recursive: true, force: true });
}

function clonePinnedSource(source, target) {
  run(null, ["clone", "--filter=blob:none", "--no-checkout", `https://github.com/${source.repository}.git`, target]);
  run(target, ["sparse-checkout", "init", "--no-cone"]);
  run(target, ["sparse-checkout", "set", ...source.files.map((row) => row.path)]);
  run(target, ["fetch", "--depth", "1", "origin", source.commit]);
  run(target, ["checkout", "--detach", "FETCH_HEAD"]);
}

function git(cwd, command) {
  const safe = cwd ? ["-c", `safe.directory=${path.resolve(cwd).replaceAll("\\", "/")}`] : [];
  return run(cwd, [...safe, ...command]).trim();
}
function run(cwd, command) {
  const result = spawnSync("git", command, { cwd: cwd || undefined, encoding: "utf8", windowsHide: true });
  if (result.status !== 0) throw new Error(`git ${command.join(" ")} failed: ${result.stderr || result.stdout}`);
  return result.stdout;
}
