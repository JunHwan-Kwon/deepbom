import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import { analyzeOnnxModel } from "../web/onnx.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { float32, model, node, tensor, valueInfo } from "./onnx-proto-fixture.mjs";

const OUTPUT_DIR = "corpus/onnx-extension-contract-corpus";
const MANIFEST_PATH = "corpus/onnx-extension-contract-corpus.v1.json";
const ONNX_COMMIT = "2bb50465112feca9003e1ed654d77f01ff1415ca";
const ORT_EXT_COMMIT = "bd0e21c11187e0b8a2385d1c61122a4d259a53a0";
const GENERATOR_SHA256 = sha256(await readFile(new URL(import.meta.url)));

await mkdir(OUTPUT_DIR, { recursive: true });
const artifacts = [
  generated("onnx-quantize-linear-runtime-per-axis", runtimePerAxis("QuantizeLinear"), {
    origin: "source_derived_conformance_fixture",
    source_repository: "onnx/onnx",
    source_commit: ONNX_COMMIT,
    source_path: "onnx/backend/test/case/node/quantizelinear.py",
    source_sha256: "89c33d77c460c39f16f706a0e182fcada5efd29d6b478907fb8cdd9237f181e8",
    license: "Apache-2.0",
  }),
  generated("onnx-dequantize-linear-runtime-per-axis", runtimePerAxis("DequantizeLinear"), {
    origin: "source_derived_conformance_fixture",
    source_repository: "onnx/onnx",
    source_commit: ONNX_COMMIT,
    source_path: "onnx/backend/test/case/node/dequantizelinear.py",
    source_sha256: "66787951091ebb1972188d00e4df539fc89394182e6480c2b1be9b2b2fd5c751",
    license: "Apache-2.0",
  }),
  generated("deepbom-static-per-axis-qdq", staticPerAxisQdq(), {
    origin: "deepbom_conformance_fixture",
    source_repository: null,
    source_commit: null,
    source_path: "scripts/build-onnx-extension-contract-corpus.mjs",
    source_sha256: GENERATOR_SHA256,
    license: "All-rights-reserved test fixture; no model weights or learned parameters",
  }),
  await pinnedUpstream("onnxruntime-extensions-custom-op-test", {
    url: `https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/${ORT_EXT_COMMIT}/test/data/custom_op_test.onnx`,
    size_bytes: 223,
    sha256: "3c35f6d0ba40415670605a533e3231ab3dfb9f0dd6761d56ad93ecf950527282",
    origin: "upstream_binary_test_artifact",
    source_repository: "microsoft/onnxruntime-extensions",
    source_commit: ORT_EXT_COMMIT,
    source_path: "test/data/custom_op_test.onnx",
    source_sha256: "3c35f6d0ba40415670605a533e3231ab3dfb9f0dd6761d56ad93ecf950527282",
    license: "MIT",
  }),
];

for (const artifact of artifacts) {
  if (!artifact._bytes) continue;
  await writeFile(artifact.path, artifact._bytes);
  delete artifact._bytes;
}

await pinnedFile("onnx.LICENSE", `https://raw.githubusercontent.com/onnx/onnx/${ONNX_COMMIT}/LICENSE`, 11358, "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30");
await pinnedFile("onnxruntime-extensions.LICENSE", `https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/${ORT_EXT_COMMIT}/LICENSE`, 1141, "c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383");

const body = {
  schema: "deepbom.onnx_extension_contract_corpus.v1",
  purpose: "Positive contract coverage for ONNX per-axis Q/DQ and non-ORT external custom-domain handling.",
  population_boundary: "Purposeful conformance corpus; counts do not estimate ecosystem prevalence or runtime support.",
  source_pins: [
    { repository: "onnx/onnx", commit: ONNX_COMMIT, license: "Apache-2.0", license_path: `${OUTPUT_DIR}/onnx.LICENSE`, license_sha256: "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30" },
    { repository: "microsoft/onnxruntime-extensions", commit: ORT_EXT_COMMIT, license: "MIT", license_path: `${OUTPUT_DIR}/onnxruntime-extensions.LICENSE`, license_sha256: "c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383" },
  ],
  artifacts,
  summary: {
    artifact_count: artifacts.length,
    per_axis_qdq_artifact_count: artifacts.filter((row) => row.measurement.per_axis_tensor_count > 0).length,
    complete_static_affine_artifact_count: artifacts.filter((row) => row.measurement.binding_status === "pass").length,
    runtime_value_unresolved_artifact_count: artifacts.filter((row) => row.measurement.binding_status === "partial").length,
    external_custom_domain_artifact_count: artifacts.filter((row) => row.measurement.external_custom_node_count > 0).length,
    total_operator_count: artifacts.reduce((sum, row) => sum + row.measurement.operator_count, 0),
  },
  limitations: [
    "The two ONNX source-derived fixtures establish schema-default per-axis structure while affine values remain runtime inputs.",
    "The complete static fixture establishes deterministic decoding and cardinality checks, not ecosystem frequency.",
    "The custom-domain artifact establishes unresolved external-registry handling, not availability of its runtime library or device kernel.",
    "Observed execution-provider assignment, task accuracy, and latency remain outside this static corpus.",
  ],
};
const manifest = { ...body, ledger_sha256: sha256(canonicalJson(body)) };
await writeFile(MANIFEST_PATH, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
console.log(`Wrote ${MANIFEST_PATH}: ${artifacts.length} contract artifacts.`);

function generated(id, bytes, provenance) {
  const filename = path.join(OUTPUT_DIR, `${id}.onnx`);
  return writeGenerated(id, filename, bytes, provenance);
}

function writeGenerated(id, filename, bytes, provenance) {
  return {
    id,
    path: filename.replaceAll("\\", "/"),
    size_bytes: bytes.length,
    sha256: sha256(bytes),
    provenance,
    measurement: measurement(analyzeOnnxModel(bytes, path.basename(filename))),
    _bytes: bytes,
  };
}

async function pinnedUpstream(id, provenance) {
  const filename = path.join(OUTPUT_DIR, `${id}.onnx`);
  await pinnedFile(`${id}.onnx`, provenance.url, provenance.size_bytes, provenance.sha256);
  const bytes = new Uint8Array(await readFile(filename));
  const { url: _url, size_bytes: _size, sha256: _sha, ...source } = provenance;
  return { id, path: filename.replaceAll("\\", "/"), size_bytes: bytes.length, sha256: sha256(bytes), provenance: source, measurement: measurement(analyzeOnnxModel(bytes, path.basename(filename))) };
}

async function pinnedFile(name, url, size, digest) {
  const filename = path.join(OUTPUT_DIR, name);
  let bytes = null;
  try { bytes = new Uint8Array(await readFile(filename)); } catch (error) { if (error?.code !== "ENOENT") throw error; }
  if (!bytes || bytes.length !== size || sha256(bytes) !== digest) {
    const response = await fetch(url, { redirect: "follow", headers: { "User-Agent": "DeepBOM-contract-corpus/1.0" } });
    if (!response.ok) throw new Error(`${name}: download failed with HTTP ${response.status}.`);
    bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.length !== size || sha256(bytes) !== digest) throw new Error(`${name}: upstream bytes differ from the pinned identity.`);
    await writeFile(filename, bytes);
  }
  return bytes;
}

function runtimePerAxis(opType) {
  const quantize = opType === "QuantizeLinear";
  const dataInput = quantize ? valueInfo("x", 1, [1, 3, 3, 2]) : valueInfo("x", 2, [1, 3, 3, 2]);
  const dataOutput = quantize ? valueInfo("y", 2, [1, 3, 3, 2]) : valueInfo("y", 1, [1, 3, 3, 2]);
  return model(
    [node(opType, opType.toLowerCase(), ["x", "scale", "zero_point"], ["y"])],
    [],
    [dataInput, valueInfo("scale", 1, [3]), valueInfo("zero_point", 2, [3])],
    [dataOutput],
    25, [], 13, `deepbom_${opType.toLowerCase()}_runtime_per_axis`,
  );
}

function staticPerAxisQdq() {
  return model(
    [node("QuantizeLinear", "quantize", ["x", "scale", "zero_point"], ["qx"]), node("DequantizeLinear", "dequantize", ["qx", "scale", "zero_point"], ["y"])],
    [tensor("scale", 1, [3], float32([0.125, 0.25, 0.5])), tensor("zero_point", 2, [3], new Uint8Array([128, 127, 126]))],
    [valueInfo("x", 1, [1, 3, 3, 2])], [valueInfo("y", 1, [1, 3, 3, 2])],
    25, [], 8, "deepbom_static_per_axis_qdq",
  );
}

function measurement(analysis) {
  const binding = analysis.onnx_quantization_binding || {};
  return {
    operator_count: Number(analysis.operator_count || 0),
    tensor_count: Number(analysis.tensor_count || 0),
    explicit_qdq_boundary_count: Number(binding.explicit_qdq_boundary_count || 0),
    binding_status: binding.status || "not_applicable",
    valid_binding_count: Number(binding.valid_binding_count || 0),
    unresolved_binding_count: Number(binding.unresolved_binding_count || 0),
    per_axis_tensor_count: Number(analysis.per_channel_tensors || 0),
    external_custom_node_count: Number(analysis.onnx_domain_analysis?.external_custom_node_count || 0),
    external_custom_domains: analysis.onnx_domain_analysis?.external_custom_domains || [],
  };
}

function sha256(value) { return createHash("sha256").update(value).digest("hex"); }
