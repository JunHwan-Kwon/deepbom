import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, stat, writeFile } from "node:fs/promises";
import { basename, resolve } from "node:path";

import { parseGgufRuntimeEnvironmentDocument } from "../web/lib/gguf-runtime-environment.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";

const args = parseArgs(process.argv.slice(2));
const paths = Object.fromEntries(["model", "manifest", "binary", "cmake-cache", "build-attestation"].map((key) => [key, resolve(required(args, key))]));
const [model, manifest, binary, cmakeCache, buildAttestation] = await Promise.all([
  fileEvidence(paths.model), fileEvidence(paths.manifest, true), fileEvidence(paths.binary), fileEvidence(paths["cmake-cache"]), fileEvidence(paths["build-attestation"], true),
]);
const manifestDocument = parseJson(manifest.bytes, "GGUF runtime manifest");
const attestationDocument = parseJson(buildAttestation.bytes, "llama.cpp build attestation");
const normalized = parseGgufRuntimeEnvironmentDocument(manifestDocument, {
  format: "gguf",
  filename: basename(paths.model),
  model_sha256: model.sha256,
}, { fileSha256: manifest.sha256 });

requireEqual(normalized.runtime.source_alignment, "exact_pinned_source_match", "runtime source alignment");
requireEqual(normalized.runtime.binary_sha256, binary.sha256, "runtime binary SHA-256");
requireEqual(normalized.build.cmake_cache_sha256, cmakeCache.sha256, "CMake cache SHA-256");
requireEqual(normalized.build.attestation.file_sha256, buildAttestation.sha256, "build-attestation file SHA-256");
requireEqual(normalized.build.attestation.canonical_sha256, sha256TextHex(canonicalJson(attestationDocument)), "build-attestation canonical SHA-256");
requireEqual(normalized.build.attestation.document.binary.filename, basename(paths.binary), "attested binary filename");
if (args["require-inference"] === true) {
  requireEqual(normalized.observations.inference_status, "observed_success", "inference observation");
  if (!normalized.compute_graph || normalized.compute_graph.successful_dispatch_count < 1 || normalized.compute_graph.failed_dispatch_count !== 0) {
    fail("Required inference capture lacks an exclusively successful scheduler-dispatch ledger.");
  }
}

const body = {
  schema: "deepbom.gguf_runtime_capture_verification.v1",
  status: "pass",
  evidence_class: "INDEPENDENT_LOCAL_RECOMPUTATION",
  files: {
    model: compact(model),
    manifest: compact(manifest),
    binary: compact(binary),
    cmake_cache: compact(cmakeCache),
    build_attestation: compact(buildAttestation),
  },
  source_alignment: normalized.runtime.source_alignment,
  requested_backend_profile_id: normalized.selection.requested_backend_profile_id,
  observations: normalized.observations,
  compute_graph: normalized.compute_graph ? {
    trace_sha256: normalized.compute_graph.trace_sha256,
    graph_count: normalized.compute_graph.graph_count,
    dispatched_graph_count: normalized.compute_graph.dispatched_graph_count,
    dispatch_count: normalized.compute_graph.dispatch_count,
    successful_dispatch_count: normalized.compute_graph.successful_dispatch_count,
    failed_dispatch_count: normalized.compute_graph.failed_dispatch_count,
    original_node_count: normalized.compute_graph.original_node_count,
    scheduled_node_count: normalized.compute_graph.scheduled_node_count,
    split_count: normalized.compute_graph.split_count,
    original_backend_transition_edge_count: normalized.compute_graph.original_backend_transition_edge_count,
    scheduled_backend_transition_edge_count: normalized.compute_graph.scheduled_backend_transition_edge_count,
  } : null,
  boundary: "This receipt independently recomputes selected local file hashes and validates the complete runtime manifest, embedded build attestation, source alignment, scheduler graph conservation, backend assignment, and dispatch status. It proves only the captured process calls on this host; it does not establish uncaptured paths, selected microkernels, latency generalization, or task quality.",
};
const receipt = { ...body, receipt_sha256: sha256TextHex(canonicalJson(body)) };
if (args.output) await writeFile(resolve(args.output), `${JSON.stringify(receipt, null, 2)}\n`, "utf8");
console.log(`GGUF runtime capture verified: ${receipt.receipt_sha256}`);
console.log(`artifact ${model.sha256}; binary ${binary.sha256}; manifest ${manifest.sha256}`);
console.log(`graphs ${receipt.compute_graph?.graph_count || 0}; dispatched ${receipt.compute_graph?.dispatched_graph_count || 0}; successful ${receipt.compute_graph?.successful_dispatch_count || 0}`);

function compact(value) {
  return { filename: value.filename, byte_length: value.byte_length, sha256: value.sha256 };
}

async function fileEvidence(path, retain = false) {
  const info = await stat(path);
  if (!info.isFile()) fail(`${path} is not a regular file.`);
  const result = { filename: basename(path), byte_length: info.size, sha256: await hashFile(path) };
  if (retain) result.bytes = await readFile(path, "utf8");
  return result;
}

function hashFile(path) {
  return new Promise((resolveHash, reject) => {
    const hash = createHash("sha256");
    const stream = createReadStream(path);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", () => resolveHash(hash.digest("hex")));
  });
}

function parseJson(source, label) {
  try {
    const value = JSON.parse(source);
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("root must be an object");
    return value;
  } catch (error) {
    fail(`${label} is invalid JSON: ${error.message}`);
  }
}

function parseArgs(tokens) {
  const parsed = {};
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (!token.startsWith("--")) fail(`Unexpected argument: ${token}.`);
    const [key, inline] = token.slice(2).split("=", 2);
    if (!key) fail("Empty argument name.");
    if (inline != null) parsed[key] = inline;
    else if (tokens[index + 1] && !tokens[index + 1].startsWith("--")) parsed[key] = tokens[++index];
    else parsed[key] = true;
  }
  return parsed;
}

function required(values, key) {
  if (typeof values[key] !== "string" || !values[key]) fail(`Missing required --${key}.`);
  return values[key];
}

function requireEqual(actual, expected, label) {
  if (actual !== expected) fail(`${label} mismatch: ${actual ?? "missing"} != ${expected ?? "missing"}.`);
}

function fail(message) {
  throw new Error(message);
}
