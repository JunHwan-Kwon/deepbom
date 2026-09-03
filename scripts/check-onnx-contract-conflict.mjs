import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import Ajv2020 from "ajv/dist/2020.js";

import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { deriveCurrentArtifactCapabilityRow } from "../web/lib/format-capability-view.js";
import {
  attachOnnxContractConflictCapsule,
  validateOnnxContractConflictCapsule,
} from "../web/lib/onnx-contract-conflict.js";
import { onnxShapeInferenceMarkdown } from "../web/lib/report-engineering-onnx.js";
import { buildArtifactEvidenceEnvelope } from "../web/lib/artifact-evidence-envelope.js";
import { buildMlBomCompatibilityProjection } from "../web/lib/report-mlbom-compat.js";

const schema = JSON.parse(await readFile("docs/schemas/deepbom-onnx-contract-conflict-capsule-v1.schema.json", "utf8"));
const validateSchema = new Ajv2020({ strict: true, allErrors: true }).compile(schema);
const sha256 = "a".repeat(64);
const declaration = { node_index: 0, op_name: "Conv", tensor_name: "bad", field: "shape", declared: [1, 2], inferred: [1, 3] };
const semantic = { node_index: 1, op_name: "MatMul", output_names: ["semantic"], reason: "matmul_inner_dimension_mismatch", details: { left: 2, right: 3 } };
const analysis = {
  format: "onnx", filename: "invalid-contract.onnx", model_sha256: sha256, file_size: 100,
  operator_count: 3, tensor_count: 5, input_tensor_indices: [0], output_tensor_indices: [4],
  ops: [
    { index: 0, name: "Conv", domain: "ai.onnx", inputs: [0], outputs: [2], macs_status: "not_assessed" },
    { index: 1, name: "MatMul", domain: "ai.onnx", inputs: [2], outputs: [4], macs_status: "not_assessed" },
    { index: 2, name: "Add", domain: "ai.onnx", inputs: [4], outputs: [3], macs_status: "not_assessed" },
  ],
  tensors: [
    { index: 0, name: "input", dtype: "FLOAT32", shape: [1, 2], role: "input" },
    { index: 1, name: "weights", dtype: "FLOAT32", shape: [2, 2] },
    { index: 2, name: "bad", dtype: "FLOAT32", shape: [1, 2], contract_status: "invalid", contract_conflict: { ...declaration, reason: "inferred_contract_conflicts_with_declared_output", root_conflict: declaration } },
    { index: 3, name: "conditional", dtype: "FLOAT32", shape: [-1, 2], conditional_shape_contract: { status: "assessed_partial", variant_failures: [{ status: "invalid", reason: "squeeze_axis_dimension_not_one", conditions: [{ key: "D0", value: "false" }] }] } },
    { index: 4, name: "semantic", dtype: "FLOAT32", shape: [], role: "output", contract_status: "invalid", contract_conflict: { ...semantic, root_conflict: semantic } },
  ],
  onnx_shape_inference: {
    schema: "deepbom.onnx_shape_inference.test", status: "fail", evidence_class: "SOURCE_PINNED_AND_DERIVED",
    declaration_conflicts: [declaration], semantic_contract_conflicts: [semantic],
    invalid_node_output_count: 2, conditionally_invalid_node_output_count: 1,
    rule_unresolved_nodes: [{ node_index: 2, op_name: "Add", reason: "blocked_by_upstream_contract_conflict:semantic", blocked_by: { tensor_name: "semantic", root_conflict: semantic } }],
  },
  dynamic_shape_cost_contract: {
    schema: "deepbom.dynamic_shape_cost_contract.test",
    total_macs_unresolved_ops: [{
      op_index: 0, op_name: "Conv", resolution_class: "artifact_contract_conflict", reason: "invalid output contract",
      blocking_tensors: [{ index: 2, name: "bad", role: "output" }], root_conflicts: [declaration],
    }],
  },
};

attachOnnxContractConflictCapsule(analysis);
const capsule = analysis.onnx_contract_conflict;
assert.equal(validateSchema(capsule), true, JSON.stringify(validateSchema.errors));
assert.equal(capsule.status, "INVALID_CONTRACT");
assert.deepEqual(capsule.summary, {
  unconditional_root_conflict_count: 2,
  declaration_root_conflict_count: 1,
  semantic_root_conflict_count: 1,
  condition_bound_invalid_variant_count: 1,
  invalid_node_output_count: 2,
  conditionally_invalid_node_output_count: 1,
  downstream_blocked_node_count: 1,
  blocked_mac_row_count: 1,
  blocked_mac_op_histogram: [{ name: "Conv", count: 1 }],
  unresolved_root_reference_count: 0,
});
const context = getArtifactIrContext(analysis, { filename: analysis.filename, format: "onnx", sha256, size: 100 });
assert(validateOnnxContractConflictCapsule(capsule, context.artifact_ir));
assert.equal(deriveCurrentArtifactCapabilityRow("onnx", analysis).cells[1].label, "Invalid contract");
assert.match(onnxShapeInferenceMarkdown(analysis), /Serialized Contract Validity .* INVALID_CONTRACT/);
assert.equal(buildArtifactEvidenceEnvelope(analysis, { hash: sha256 }).format_extensions.onnx.contract_conflict_capsule.capsule_sha256, capsule.capsule_sha256);
const properties = buildMlBomCompatibilityProjection(analysis, { hash: sha256 }).componentProperties;
assert.equal(properties.find((row) => row.name === "deepbom:model:serializedContractStatus")?.value, "INVALID_CONTRACT");
assert.equal(properties.find((row) => row.name === "deepbom:model:contractConflictBlockedMacRows")?.value, "1");

const tampered = structuredClone(capsule);
tampered.summary.blocked_mac_row_count = 2;
assert.throws(() => validateOnnxContractConflictCapsule(tampered), /digest does not match/);

const clean = structuredClone(analysis);
clean.onnx_shape_inference = { ...clean.onnx_shape_inference, status: "assessed", declaration_conflicts: [], semantic_contract_conflicts: [], invalid_node_output_count: 0, conditionally_invalid_node_output_count: 0, rule_unresolved_nodes: [] };
clean.dynamic_shape_cost_contract = { ...clean.dynamic_shape_cost_contract, total_macs_unresolved_ops: [] };
clean.tensors = clean.tensors.map((row) => ({ ...row, contract_status: "assessed", contract_conflict: null, conditional_shape_contract: null }));
attachOnnxContractConflictCapsule(clean);
assert.equal(clean.onnx_contract_conflict.status, "ASSESSED_NO_CONFLICT");
assert.equal(clean.onnx_contract_conflict.summary.blocked_mac_row_count, 0);

console.log("ONNX contract-conflict capsule passed (schema, digest, conservation, Artifact IR subjects, UI, report, envelope, and ML-BOM projection).");
