import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

import { analyzeOnnxModel } from "../web/onnx.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectDeepEqual, expectEqual } = createCheck("ONNX extension contract corpus check");
const manifest = JSON.parse(await readFile("corpus/onnx-extension-contract-corpus.v1.json", "utf8"));
const { ledger_sha256: ledger, ...body } = manifest;
expectEqual(manifest.schema, "deepbom.onnx_extension_contract_corpus.v1", "Corpus schema must remain explicit.");
expectEqual(ledger, sha256(canonicalJson(body)), "Corpus ledger must bind every manifest field.");
expectEqual(manifest.artifacts.length, 4, "The contract corpus must retain all reviewed strata.");
expect(manifest.population_boundary.includes("do not estimate ecosystem prevalence"), "The conformance corpus must not be presented as prevalence evidence.");

const analyses = new Map();
for (const artifact of manifest.artifacts) {
  const bytes = new Uint8Array(await readFile(artifact.path));
  expectEqual(bytes.length, artifact.size_bytes, `${artifact.id}: byte length changed.`);
  expectEqual(sha256(bytes), artifact.sha256, `${artifact.id}: SHA-256 changed.`);
  const analysis = analyzeOnnxModel(bytes, artifact.id);
  analyses.set(artifact.id, analysis);
  expectEqual(Number(analysis.operator_count || 0), artifact.measurement.operator_count, `${artifact.id}: operator count changed.`);
  expectEqual(Number(analysis.tensor_count || 0), artifact.measurement.tensor_count, `${artifact.id}: tensor count changed.`);
  expectEqual(analysis.onnx_quantization_binding?.status || "not_applicable", artifact.measurement.binding_status, `${artifact.id}: binding status changed.`);
  expectEqual(Number(analysis.onnx_domain_analysis?.external_custom_node_count || 0), artifact.measurement.external_custom_node_count, `${artifact.id}: external-domain count changed.`);
}

for (const id of ["onnx-quantize-linear-runtime-per-axis", "onnx-dequantize-linear-runtime-per-axis"]) {
  const analysis = analyses.get(id);
  const binding = analysis.onnx_quantization_binding.bindings[0];
  expectEqual(analysis.onnx_quantization_binding.status, "partial", `${id}: runtime values must remain partial.`);
  expectEqual(binding.status, "not_assessed_runtime_parameter_values", `${id}: present runtime parameters must not be labeled missing.`);
  expectEqual(binding.parameterization, "per_axis", `${id}: declared vector scale must remain per-axis.`);
  expectEqual(binding.axis, 1, `${id}: schema-default axis must be one.`);
  expectEqual(binding.axis_source, "schema_default_axis_1", `${id}: default-axis provenance changed.`);
  expectEqual(binding.scale_count, 3, `${id}: declared scale cardinality changed.`);
  expectEqual(binding.scale_value_count, 0, `${id}: runtime values must not be invented.`);
  expectEqual(binding.scale_cardinality_source, "declared_tensor_shape", `${id}: cardinality provenance changed.`);
  expectEqual(binding.cardinality_status, "pass", `${id}: structural cardinality must be assessed.`);
  expectEqual(binding.value_evidence_class, "RUNTIME_REQUIRED", `${id}: value evidence boundary changed.`);
  expectEqual(analysis.per_channel_tensors, 1, `${id}: per-axis inventory count changed.`);
}

const complete = analyses.get("deepbom-static-per-axis-qdq");
expectEqual(complete.onnx_quantization_binding.status, "pass", "Static per-axis Q/DQ must remain fully assessable.");
expectEqual(complete.onnx_quantization_binding.valid_binding_count, 2, "Static Q and DQ contracts must both validate.");
expectEqual(complete.onnx_quantization_binding.explicit_qdq_boundary_count, 2, "Static Q/DQ boundaries must be counted exactly.");
expectDeepEqual(complete.onnx_quantization_binding.bindings.map((row) => row.scale_values), [[0.125, 0.25, 0.5], [0.125, 0.25, 0.5]], "Static per-axis scale values changed.");
expect(complete.onnx_quantization_binding.bindings.every((row) => row.axis === 1 && row.cardinality_status === "pass" && row.value_evidence_class === "OBSERVED"), "Static per-axis contracts must preserve axis, cardinality, and observed values.");
expectDeepEqual(complete.onnx_quantization_binding.boundary_edges.map((row) => [row.input_payload_bytes, row.output_payload_bytes]), [[72, 18], [18, 72]], "Q/DQ boundary payload conservation changed.");

const custom = analyses.get("onnxruntime-extensions-custom-op-test");
expectDeepEqual(custom.onnx_domain_analysis.external_custom_domains, ["ai.onnx.contrib"], "External custom-domain identity changed.");
expectEqual(custom.onnx_domain_analysis.external_custom_node_count, 2, "Custom node count changed.");
expectEqual(custom.onnx_domain_analysis.ort_contrib_node_count, 0, "External custom nodes must not be relabeled as com.microsoft contrib nodes.");
expect(custom.ops.every((row) => row.domain === "ai.onnx.contrib" && row.standard_domain === false && row.macs === null && row.macs_status === "not_assessed"), "Custom ops must retain domain identity and fail closed for unsupported semantics.");

expectEqual(manifest.summary.per_axis_qdq_artifact_count, 3, "Per-axis artifact summary changed.");
expectEqual(manifest.summary.complete_static_affine_artifact_count, 1, "Complete static affine summary changed.");
expectEqual(manifest.summary.runtime_value_unresolved_artifact_count, 2, "Runtime-value boundary summary changed.");
expectEqual(manifest.summary.external_custom_domain_artifact_count, 1, "External custom-domain summary changed.");
done("ONNX extension corpus passed (per-axis runtime/static Q/DQ and upstream external custom domain). ");

function sha256(value) { return createHash("sha256").update(value).digest("hex"); }
