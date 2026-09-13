import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";

const root = process.cwd();
const roots = [
  ".github/actions/deepbom-audit",
  "channels/container",
  "channels/huggingface",
  "integrations",
  "validation/qms",
  "docs/PUBLIC_README.md",
  "skills/deepbom",
];
const files = [];
for (const relative of roots) files.push(...await collect(path.join(root, relative)));

const forbidden = [
  [/\b(?:IBM|Matt(?:hew)?\s+Rutkowski|Bartowski|Dependency-Track|cdxgen)\b/i, "unapproved named integration"],
  [/\b(?:adopted|approved|endorsed|certified)\s+by\b/i, "unverified adoption claim"],
  [/\b(?:official|normative)\s+(?:CycloneDX\s+)?(?:tool|implementation|integration|validator)\b/i, "unapproved official-status claim"],
  [/(?:PR|pull request|issue)\s*#(?:862|990|1067|1075)\b/i, "standards-participation work item"],
  [/cyclonedx[\\/]evidence|perspective-semantics-study|tensor-quantization-metadata-study/i, "private standards-evidence path"],
];
for (const file of files) {
  const source = await readFile(file, "utf8");
  for (const [pattern, label] of forbidden) {
    assert(!pattern.test(source), `${path.relative(root, file)} contains ${label}: ${pattern}`);
  }
}

console.log(`Public adoption boundary passed (${files.length} product-facing files; no named partner, official-status, or private PR-evidence claim).`);

async function collect(target) {
  const info = await statSafe(target);
  if (!info) return [];
  if (info.isFile()) return [target];
  const result = [];
  for (const entry of await readdir(target, { withFileTypes: true })) {
    const nested = path.join(target, entry.name);
    if (entry.isDirectory()) result.push(...await collect(nested));
    else if (entry.isFile() && !/\.(?:png|jpg|gif|wasm|onnx|gguf)$/i.test(entry.name)) result.push(nested);
  }
  return result;
}

async function statSafe(target) {
  try {
    return await import("node:fs/promises").then(({ stat }) => stat(target));
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}
