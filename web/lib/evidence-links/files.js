import { LINK_LIMITS, requireLink } from "./contracts.js";
import { normalizeMetadataInput } from "./omop.js";
import { sha256BytesHex } from "../sha256-sync.js";
import { parseStrictJson } from "../strict-json.js";

export function parseMetadataText(text) {
  requireLink(typeof text === "string" && new TextEncoder().encode(text).length <= LINK_LIMITS.input_bytes, "metadata JSON exceeds 2 MiB.");
  const document = parseStrictJson(text, "evidence metadata");
  return normalizeMetadataInput(document).input;
}

export function observeEvidenceBytes(node, bytes) {
  requireLink(bytes instanceof Uint8Array && bytes.byteLength <= LINK_LIMITS.file_bytes, "evidence file exceeds 32 MiB.");
  const observation = { node_ref: node.id, sha256: sha256BytesHex(bytes), byte_length: bytes.byteLength };
  if (node.kind === "bom") {
    requireLink(bytes.byteLength <= LINK_LIMITS.input_bytes, "BOM JSON exceeds 2 MiB.");
    observation.document = parseStrictJson(new TextDecoder("utf-8", { fatal: true }).decode(bytes), "evidence BOM");
  }
  return observation;
}

export async function observeBrowserEvidence(input, files) {
  const { nodes } = normalizeMetadataInput(input), byPath = new Map();
  for (const file of files) {
    const name = file.webkitRelativePath || file.name;
    requireLink(!byPath.has(name), "selected evidence files have ambiguous duplicate paths."); byPath.set(name, file);
  }
  let total = 0;
  const observations = [], used = new Set();
  for (const node of nodes) {
    if (!node.file || !byPath.has(node.file)) continue;
    const file = byPath.get(node.file); total += file.size;
    requireLink(file.size <= LINK_LIMITS.file_bytes && total <= LINK_LIMITS.total_file_bytes, "evidence files exceed the per-file or total byte bound.");
    observations.push(observeEvidenceBytes(node, new Uint8Array(await file.arrayBuffer()))); used.add(node.file);
  }
  requireLink([...byPath.keys()].every(name => used.has(name)), "a selected file is not referenced by metadata; remove it or add its explicit file mapping.");
  return observations;
}
