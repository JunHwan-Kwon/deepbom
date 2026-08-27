import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { XNNPACK_DELEGATE_RULEPACK_METADATA } from "../web/lib/xnnpack-rulepack-metadata.js";
import { fetchPinnedBytes } from "./fetch-pinned-source.mjs";

const selectorPath = "protected/deepbom_wasm/src/xnnpack_selector.rs";
const selectorSource = readFileSync(selectorPath, "utf8");
const readmeSnapshotPath = "reference/xnnpack-readme/README-v2.21.0.md";
const readmeSnapshot = readFileSync(readmeSnapshotPath);
verifySha256(readmeSnapshot, XNNPACK_DELEGATE_RULEPACK_METADATA.sourceSha256, readmeSnapshotPath);

const readmeRefs = [
  XNNPACK_DELEGATE_RULEPACK_METADATA.sourceCommit,
  XNNPACK_DELEGATE_RULEPACK_METADATA.sourceTagAlias,
  XNNPACK_DELEGATE_RULEPACK_METADATA.sourceTagCommit,
];
for (const ref of readmeRefs) {
  const url = `https://raw.githubusercontent.com/tensorflow/tensorflow/${ref}/${XNNPACK_DELEGATE_RULEPACK_METADATA.sourcePath}`;
  const bytes = await fetchBytes(url);
  verifySha256(bytes, XNNPACK_DELEGATE_RULEPACK_METADATA.sourceSha256, url);
  if (!Buffer.from(bytes).equals(readmeSnapshot)) {
    throw new Error(`${url} is not byte-identical to ${readmeSnapshotPath}.`);
  }
}

const sourceCommit = rustConstant("XNNPACK_SOURCE_COMMIT");
const sourceSpecs = new Map([
  ["gemm", {
    path: "src/configs/gemm-config.c",
    sha256: rustConstant("XNNPACK_GEMM_CONFIG_SHA256"),
  }],
  ["dwconv", {
    path: "src/configs/dwconv-config.c",
    sha256: rustConstant("XNNPACK_DWCONV_CONFIG_SHA256"),
  }],
]);

const references = extractCandidateSourceReferences(selectorSource);
if (!references.length) throw new Error(`${selectorPath} contains no candidate source references.`);

const verified = [];
for (const [sourceId, spec] of sourceSpecs) {
  const url = `https://raw.githubusercontent.com/google/XNNPACK/${sourceCommit}/${spec.path}`;
  const bytes = await fetchBytes(url);
  const actualSha256 = verifySha256(bytes, spec.sha256, spec.path);
  const lines = new TextDecoder().decode(bytes).replace(/\r\n/g, "\n").split("\n");
  if (lines.at(-1) === "") lines.pop();
  const sourceReferences = references.filter((reference) => reference.sourceId === sourceId);
  if (!sourceReferences.length) throw new Error(`${spec.path} has no protected selector references.`);
  for (const reference of sourceReferences) {
    if (reference.start < 1 || reference.end < reference.start || reference.end > lines.length) {
      throw new Error(`${spec.path} has an invalid source range ${reference.label} for ${lines.length} lines.`);
    }
    const sourceSlice = lines.slice(reference.start - 1, reference.end).join("\n").trim();
    if (!sourceSlice || !/xnn_|XNN_/.test(sourceSlice)) {
      throw new Error(`${spec.path} ${reference.label} does not resolve to an XNNPACK configuration slice.`);
    }
  }
  verified.push(`${spec.path} ${bytes.byteLength} B ${actualSha256} (${sourceReferences.length} refs)`);
}

console.log(
  `Pinned TensorFlow README and XNNPACK source verification passed:\n`
  + `  - ${readmeSnapshotPath} ${readmeSnapshot.byteLength} B ${XNNPACK_DELEGATE_RULEPACK_METADATA.sourceSha256} `
  + `(main ${XNNPACK_DELEGATE_RULEPACK_METADATA.sourceCommit}, tag ${XNNPACK_DELEGATE_RULEPACK_METADATA.sourceTagAlias}, tag commit ${XNNPACK_DELEGATE_RULEPACK_METADATA.sourceTagCommit})\n`
  + `${verified.map((item) => `  - ${item}`).join("\n")}`,
);

async function fetchBytes(url) {
  return fetchPinnedBytes(url, { label: "XNNPACK pin" });
}

function verifySha256(bytes, expected, label) {
  const actual = createHash("sha256").update(bytes).digest("hex");
  if (actual !== expected) throw new Error(`${label} SHA-256 mismatch: ${actual} !== ${expected}`);
  return actual;
}

function rustConstant(name) {
  const match = selectorSource.match(new RegExp(`const\\s+${name}[^=]*=\\s*\\n?\\s*"([0-9a-f]+)"`));
  if (!match) throw new Error(`${selectorPath} is missing ${name}.`);
  return match[1];
}

function extractCandidateSourceReferences(source) {
  const calls = balancedCalls(source, "candidate");
  const references = [];
  for (const call of calls) {
    const ranges = [...call.matchAll(/L(\d+)-L(\d+)/g)];
    if (!ranges.length) continue;
    const hasGemm = /\bgemm\s*,/.test(call);
    const hasDwconv = /\bdwconv\s*,/.test(call);
    if (hasGemm === hasDwconv) {
      throw new Error(`A protected selector candidate has an ambiguous source binding: ${call.slice(0, 100)}...`);
    }
    const sourceId = hasGemm ? "gemm" : "dwconv";
    for (const match of ranges) {
      references.push({
        sourceId,
        start: Number(match[1]),
        end: Number(match[2]),
        label: match[0],
      });
    }
  }
  return references;
}

function balancedCalls(source, functionName) {
  const calls = [];
  const pattern = new RegExp(`\\b${functionName}\\s*\\(`, "g");
  let match;
  while ((match = pattern.exec(source))) {
    const open = source.indexOf("(", match.index);
    let depth = 0;
    let quote = "";
    let escaped = false;
    for (let index = open; index < source.length; index += 1) {
      const char = source[index];
      if (quote) {
        if (escaped) escaped = false;
        else if (char === "\\") escaped = true;
        else if (char === quote) quote = "";
        continue;
      }
      if (char === '"' || char === "'") {
        quote = char;
      } else if (char === "(") {
        depth += 1;
      } else if (char === ")") {
        depth -= 1;
        if (depth === 0) {
          calls.push(source.slice(open + 1, index));
          pattern.lastIndex = index + 1;
          break;
        }
      }
    }
    if (depth !== 0) throw new Error(`Unbalanced ${functionName}(...) call in ${selectorPath}.`);
  }
  return calls;
}
