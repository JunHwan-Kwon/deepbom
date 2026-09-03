import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { gunzipSync, gzipSync } from "node:zlib";

import { analyzeOnnxModel } from "../web/onnx.js";
import { attachOnnxContractConflictCapsule } from "../web/lib/onnx-contract-conflict.js";
import { canonicalJson } from "../web/lib/report-utils.js";

const SOURCE_SWEEP = "corpus/public-onnx-residual-sweep.v1.json.gz";
const OUTPUT = "corpus/onnx-contract-conflict-corpus.v1.json.gz";
const CACHE = path.resolve(".local-validation", "onnx-contract-conflict-models");
const download = process.argv.includes("--download");
const sweepBytes = await readFile(SOURCE_SWEEP);
const sweep = JSON.parse(gunzipSync(sweepBytes));
const sourceRows = (sweep.rows || []).filter((row) => row.format === "onnx"
  && (Number(row.coverage_residuals?.invalid_node_output_count || 0) > 0
    || Number(row.coverage_residuals?.conditionally_invalid_node_output_count || 0) > 0));
if (!sourceRows.length) throw new Error("The pinned ONNX residual sweep contains no invalid-contract artifacts.");
await mkdir(CACHE, { recursive: true });

const analyses = [];
for (const [index, row] of sourceRows.entries()) {
  const localPath = path.join(CACHE, `${row.artifact_sha256}.onnx`);
  let bytes;
  try {
    bytes = await readFile(localPath);
  } catch {
    if (!download) throw new Error(`Missing ${localPath}; rerun with --download to fetch the pinned artifact bytes.`);
    const url = huggingFaceUrl(row);
    process.stdout.write(`[${index + 1}/${sourceRows.length}] downloading ${row.repository_id}/${row.path}\n`);
    const response = await fetch(url, { redirect: "follow" });
    if (!response.ok) throw new Error(`Download failed (${response.status}) for ${url}`);
    bytes = Buffer.from(await response.arrayBuffer());
    await writeFile(localPath, bytes);
  }
  if (sha256(bytes) !== row.artifact_sha256) throw new Error(`${row.repository_id}/${row.path}: artifact SHA-256 mismatch.`);
  const analysis = analyzeOnnxModel(new Uint8Array(bytes), path.posix.basename(row.path));
  analysis.model_sha256 = row.artifact_sha256;
  analysis.artifact_set = {
    artifact_set_sha256: sha256(Buffer.from(canonicalJson({ repository: row.repository_id, revision: row.revision, path: row.path, artifact_sha256: row.artifact_sha256 }))),
    source: {
      canonical_locator: `hf://${row.repository_id}@${row.revision}/${row.path}`,
      revision: row.revision,
      immutability: { revision: row.revision },
    },
  };
  attachOnnxContractConflictCapsule(analysis);
  compareSummary(row, analysis.onnx_contract_conflict.summary);
  analyses.push({ row, localPath, capsule: analysis.onnx_contract_conflict });
}

const reference = runOfficialReferenceProbe(analyses.map((row) => row.localPath));
const referenceBySha = new Map(reference.rows.map((row) => [row.artifact_sha256, row]));
const artifacts = analyses.map(({ row, capsule }) => ({
  repository_id: row.repository_id,
  revision: row.revision,
  path: row.path,
  artifact_size_bytes: row.artifact_size_bytes,
  artifact_sha256: row.artifact_sha256,
  source_analysis_sha256: row.analysis_sha256,
  capsule,
  official_onnx_reference: referenceBySha.get(row.artifact_sha256) || null,
})).sort((left, right) => left.artifact_sha256.localeCompare(right.artifact_sha256));
const aggregate = aggregateSummary(artifacts);
const body = {
  schema: "deepbom.onnx_contract_conflict_corpus.v1",
  method: {
    population: "Every unique-byte ONNX artifact in the pinned public residual sweep with at least one unconditional or condition-bound invalid serialized shape contract.",
    derivation: "Each conflict capsule is rebuilt from the artifact bytes with the current source-pinned ONNX shape and dynamic-cost ledgers, then cross-checked against the source sweep denominators.",
    independent_reference: "The same bytes are separately checked with the official Python ONNX checker and strict shape inference. Their result is comparative evidence and is not used to alter DEEPBOM findings.",
    non_claim: "Nine selected artifacts establish reproducible failure modes, not ecosystem prevalence. External-data payload absence may independently affect the official checker result and is retained rather than conflated with shape-contract validity.",
  },
  source_sweep: {
    path: SOURCE_SWEEP,
    sha256: sha256(sweepBytes),
    schema: sweep.schema,
    onnx_artifact_count: (sweep.rows || []).filter((row) => row.format === "onnx").length,
  },
  official_reference: {
    schema: reference.schema,
    onnx_version: unique(reference.rows.map((row) => row.onnx_version).filter(Boolean)).join(",") || null,
    requirements_path: "requirements-onnx-corpus.txt",
    probe_path: "scripts/probe-onnx-contract-reference.py",
  },
  artifact_count: artifacts.length,
  aggregate,
  artifacts,
  hash_contract: {
    algorithm: "SHA-256",
    canonicalization: "UTF-8 RFC8785-style canonical JSON implemented by web/lib/report-utils.js canonicalJson",
    excluded_pointers: ["/corpus_sha256"],
  },
};
const corpus = { ...body, corpus_sha256: sha256(Buffer.from(canonicalJson(body))) };
await writeFile(OUTPUT, gzipSync(Buffer.from(`${JSON.stringify(corpus)}\n`), { level: 9, mtime: 0 }));
console.log(`Wrote ${OUTPUT}: ${artifacts.length} capsules, ${aggregate.unconditional_root_conflict_count} unconditional roots, ${aggregate.condition_bound_invalid_variant_count} condition-bound variants, ${aggregate.blocked_mac_row_count} blocked MAC rows.`);

function compareSummary(row, summary) {
  const source = row.coverage_residuals;
  const expected = {
    unconditional_root_conflict_count: Number(source.declaration_conflict_count || 0) + Number(source.semantic_contract_conflict_count || 0),
    condition_bound_invalid_variant_count: Number(source.conditional_invalid_variant_count || 0),
    invalid_node_output_count: Number(source.invalid_node_output_count || 0),
    conditionally_invalid_node_output_count: Number(source.conditionally_invalid_node_output_count || 0),
    downstream_blocked_node_count: Number(source.blocked_by_upstream_contract_conflict_node_count || 0),
    blocked_mac_row_count: Number(source.total_macs_artifact_contract_conflict_op_count || 0),
  };
  for (const [key, value] of Object.entries(expected)) if (summary[key] !== value) {
    throw new Error(`${row.repository_id}/${row.path}: capsule ${key} ${summary[key]} != pinned residual ${value}.`);
  }
  const expectedHistogram = source.total_macs_unresolved_op_histogram || [];
  if (canonicalJson(summary.blocked_mac_op_histogram) !== canonicalJson(expectedHistogram)) {
    throw new Error(`${row.repository_id}/${row.path}: blocked-MAC histogram differs from the pinned residual.`);
  }
}

function aggregateSummary(artifacts) {
  const sum = (key) => artifacts.reduce((total, row) => total + Number(row.capsule.summary[key] || 0), 0);
  const histogram = new Map();
  for (const row of artifacts) for (const item of row.capsule.summary.blocked_mac_op_histogram || []) {
    histogram.set(item.name, (histogram.get(item.name) || 0) + item.count);
  }
  return {
    unconditional_root_conflict_count: sum("unconditional_root_conflict_count"),
    declaration_root_conflict_count: sum("declaration_root_conflict_count"),
    semantic_root_conflict_count: sum("semantic_root_conflict_count"),
    condition_bound_invalid_variant_count: sum("condition_bound_invalid_variant_count"),
    invalid_node_output_count: sum("invalid_node_output_count"),
    conditionally_invalid_node_output_count: sum("conditionally_invalid_node_output_count"),
    downstream_blocked_node_count: sum("downstream_blocked_node_count"),
    blocked_mac_row_count: sum("blocked_mac_row_count"),
    unresolved_root_reference_count: sum("unresolved_root_reference_count"),
    blocked_mac_op_histogram: [...histogram.entries()].sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0])).map(([name, count]) => ({ name, count })),
    official_checker_status_histogram: statusHistogram(artifacts.map((row) => row.official_onnx_reference?.checker?.status)),
    official_strict_shape_inference_status_histogram: statusHistogram(artifacts.map((row) => row.official_onnx_reference?.strict_shape_inference?.status)),
  };
}

function runOfficialReferenceProbe(paths) {
  return {
    schema: "deepbom.onnx_official_reference_probe.v1",
    rows: paths.map((file) => {
      const checker = runOfficialReferenceOperation(file, "checker");
      const strictShapeInference = runOfficialReferenceOperation(file, "strict_shape_inference");
      if (checker.artifact_sha256 !== strictShapeInference.artifact_sha256) {
        throw new Error(`${file}: isolated ONNX reference probes disagree on artifact identity.`);
      }
      return {
        artifact_sha256: checker.artifact_sha256,
        onnx_version: checker.onnx_version,
        checker: checker.checker,
        strict_shape_inference: strictShapeInference.strict_shape_inference,
      };
    }),
  };
}

function runOfficialReferenceOperation(file, operation) {
  const result = spawnSync("python", ["scripts/probe-onnx-contract-reference.py", "--operation", operation, file], {
    encoding: "utf8",
    maxBuffer: 4 * 1024 * 1024,
    timeout: 120_000,
  });
  if (result.status === 0) {
    const parsed = JSON.parse(result.stdout);
    if (parsed.schema !== "deepbom.onnx_official_reference_probe.v1" || !parsed.row) {
      throw new Error(`${file}: official ONNX reference probe returned an unexpected document.`);
    }
    return parsed.row;
  }
  const artifactSha256 = sha256(requireBytes(file));
  const timedOut = result.error?.code === "ETIMEDOUT";
  return {
    artifact_sha256: artifactSha256,
    onnx_version: null,
    [operation]: {
      status: timedOut ? "timeout" : "crash",
      exception_type: timedOut ? "ProcessTimeout" : "ProcessExit",
      message: String(result.stderr || result.stdout || result.error?.message || `exit=${result.status}; signal=${result.signal || "none"}`).slice(0, 2000),
    },
  };
}

function requireBytes(file) {
  return readFileSync(file);
}

function statusHistogram(values) {
  const counts = new Map();
  for (const value of values) counts.set(value || "missing", (counts.get(value || "missing") || 0) + 1);
  return [...counts.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([name, count]) => ({ name, count }));
}

function huggingFaceUrl(row) {
  return `https://huggingface.co/${row.repository_id}/resolve/${row.revision}/${row.path.split("/").map(encodeURIComponent).join("/")}`;
}

function unique(values) { return [...new Set(values)]; }
function sha256(bytes) { return createHash("sha256").update(bytes).digest("hex"); }
