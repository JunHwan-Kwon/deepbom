import assert from "node:assert/strict";

import { analyzeExecuTorchModel } from "../web/executorch.js";
import { buildQuantizationEvidence, buildStaticAnalysisExport } from "../web/lib/report-evidence.js";
import { buildQuantizationContractChecks } from "../web/lib/report-quantization-contracts.js";
import { EXECUTORCH_ADD_PTE_BASE64, decodeFixtureBase64 } from "./fixtures/executorch-fixtures.mjs";

const SHA = "a".repeat(64);

const executorch = analyzeExecuTorchModel(decodeFixtureBase64(EXECUTORCH_ADD_PTE_BASE64), "add.pte");
const executorchStatic = buildStaticAnalysisExport(executorch);
const executorchQuant = buildQuantizationEvidence(executorch);
assert.equal(executorchStatic.quant_research_coverage, null);
assert.equal(executorchStatic.quantization_scope, null);
assert.match(executorchStatic.export_notes.target_dependent, /No TFLite target-dependent/);
assert.equal(executorchQuant.quant_research_coverage, null);
assert.equal(executorchQuant.status.classification, "not_assessed_format_contract_unbound");
assert.equal(executorchQuant.scope.xnnpack_fallback_or_break_ops, null);
assert.equal(executorchQuant.quantization_contract_checks.status, "not_assessed");
assert.equal(Object.hasOwn(executorchQuant.quantization_contract_checks, "bias_scale"), false);

const ggufChecks = buildQuantizationContractChecks({
  format: "gguf",
  quantization_status: { classification: "block_or_tensor_encoded_weights", unsupported_encoding_tensor_count: 0 },
  gguf: {
    payload_coverage_status: "complete_without_gaps_or_overlaps",
    type_traits_source: { type_traits_source_sha256: SHA, block_layout_source_sha256: SHA },
  },
}, "gguf");
assert.equal(ggufChecks.status, "pass");
assert.equal(ggufChecks.serialized_contract.status, "assessed_source_pinned_storage_contract");
assert.equal(Object.hasOwn(ggufChecks, "bias_scale"), false);

const safetensorsChecks = buildQuantizationContractChecks({
  format: "safetensors",
  quantization_status: { classification: "weight_container" },
  safetensors: { quantization_contract: { status: "not_applicable_no_quantization_declaration" } },
});
assert.equal(safetensorsChecks.status, "not_assessed");
assert.equal(Object.hasOwn(safetensorsChecks, "qdq_boundaries"), false);

const coreMlChecks = buildQuantizationContractChecks({
  format: "coreml",
  quantization_status: { label: "Core ML quantization", assessment_status: "assessed" },
  coreml: { quantization_contract: { status: "assessed" } },
});
assert.equal(coreMlChecks.status, "pass");
assert.equal(Object.hasOwn(coreMlChecks, "residual_add"), false);

console.log("Cross-format quantization evidence isolation passed (TFLite/ONNX rules cannot leak into serialized contracts).");
