import { createCheck } from "./check-assert.mjs";
import { buildCycloneDx20ParameterContractPreview, validateCycloneDx20ParameterContractPreview } from "../web/lib/cyclonedx-20-preview.js";
import { buildInterfaceQuantizationContractLedger } from "../web/lib/quantization-contract-summary.js";
import { canonicalContractSha256, compareInterfaceContracts, parseProductionInterfaceContract } from "../web/lib/interface-contract.js";

const { done, expect, expectDeepEqual, expectEqual } = createCheck("interface contract comparison check");
const artifactSha256 = "a".repeat(64);
const implementationSha256 = "b".repeat(64);
const analysis = {
  filename: "multi_io.onnx",
  format: "onnx",
  model_sha256: artifactSha256,
  inputs: [
    { index: 0, name: "image", dtype: "UINT8", shape: [1, 224, 224, 3], shape_signature: [], quant_scales: 1, quant_zero_points: 1, scale_sample: [0.0078125], zero_point_sample: [128], quantized_dimension: 0 },
    { index: 1, name: "token_ids", dtype: "INT64", shape: [1, 16], shape_signature: [], quant_scales: 0, quant_zero_points: 0, scale_sample: [], zero_point_sample: [] },
  ],
  outputs: [
    { index: 2, name: "scores", dtype: "FLOAT32", shape: [1, 8], shape_signature: [], quant_scales: 0, quant_zero_points: 0, scale_sample: [], zero_point_sample: [] },
  ],
};

const ledger = buildInterfaceQuantizationContractLedger(analysis);
expectEqual(ledger.schema, "deepbom.interface_quantization_contracts.v1.3", "ledger schema");
expectEqual(ledger.parameter_count, 3, "every external parameter enters the contract");
expectEqual(ledger.quantized_parameter_count, 1, "affine parameter count");
expectEqual(ledger.unquantized_parameter_count, 2, "no-affine mapping count");
expectEqual(ledger.parameters[1].quantization.affine_mapping_status, "no_affine_mapping_declared", "integer token IDs are not mislabeled as quantized");
expectEqual(canonicalContractSha256({ b: 2, a: 1 }), canonicalContractSha256({ a: 1, b: 2 }), "canonical hash ignores object insertion order");

const declaration = {
  schema: "deepbom.production_interface_contract.v1",
  artifact_sha256: artifactSha256,
  implementation_sha256: implementationSha256,
  parameters: ledger.parameters.map((row) => ({
    direction: row.direction,
    ordinal: row.ordinal,
    interface_contract_sha256: row.interface_contract_sha256,
  })),
};
const exact = compareInterfaceContracts(ledger, declaration, artifactSha256);
expectEqual(exact.status, "bound_exact_contract", "hash-only exact declaration");
expectEqual(exact.gate_result, "pass", "exact declaration gate");

const structuredDeclaration = {
  ...declaration,
  parameters: structuredClone(ledger.parameters),
};
const structuredExact = compareInterfaceContracts(ledger, structuredDeclaration, artifactSha256);
expectEqual(structuredExact.status, "bound_exact_contract", "field-complete exact declaration");
const mismatched = structuredClone(structuredDeclaration);
mismatched.parameters[0].quantization.zero_points[0] = 127;
const mismatch = compareInterfaceContracts(ledger, mismatched, artifactSha256);
expectEqual(mismatch.status, "contradiction_interface_contract_mismatch", "field mismatch status");
expect(mismatch.mismatches.some((row) => row.parameter_id === "input:0:T0" && row.field === "quantization.zero_points"), "mismatch identifies the exact parameter and field");
expect(mismatch.mismatches.every((row) => row.evidence_pointer.startsWith("/parameters/")), "parameter mismatches include deterministic JSON pointers");

const duplicate = structuredClone(declaration);
duplicate.parameters.push(structuredClone(duplicate.parameters[0]));
const invalid = compareInterfaceContracts(ledger, duplicate, artifactSha256);
expectEqual(invalid.status, "invalid_declaration", "duplicate parameter identities fail closed");
expect(invalid.declaration_validation.errors.some((message) => message.includes("Duplicate parameter identity")), "duplicate identity reason");

const malformedVector = structuredClone(structuredDeclaration);
malformedVector.parameters[0].quantization.scales.push("discard-me");
const malformedVectorResult = compareInterfaceContracts(ledger, malformedVector, artifactSha256);
expectEqual(malformedVectorResult.status, "invalid_declaration", "invalid numeric entries must not be silently filtered");
expect(malformedVectorResult.declaration_validation.errors.some((message) => message.includes("quantization.scales[1]")), "invalid vector entry location");

const malformedHash = structuredClone(structuredDeclaration);
malformedHash.artifact_sha256 = "not-a-sha256";
expectEqual(compareInterfaceContracts(ledger, malformedHash, artifactSha256).status, "invalid_declaration", "malformed supplied hashes fail closed");

const partial = { parameters: structuredClone(ledger.parameters) };
expectEqual(compareInterfaceContracts(ledger, partial, artifactSha256).status, "partial_artifact_and_implementation_hash_missing", "both missing release hashes remain explicit");

const preview = buildCycloneDx20ParameterContractPreview(analysis, {
  hash: artifactSha256,
  generatedAt: "2026-08-06T00:00:00.000Z",
  interfaceLedger: ledger,
  profileId: "legacy-parameter-contract-2026-08-06",
  allowHistoricalFixture: true,
});
preview.metadata.component.properties.push({ name: "deepbom:production:interfaceImplementationSha256", value: implementationSha256 });
const previewValidation = validateCycloneDx20ParameterContractPreview(preview);
expect(previewValidation.valid, `preview validation: ${previewValidation.errors.join("; ")}`);
const parsedPreview = parseProductionInterfaceContract(preview);
expect(parsedPreview.valid, `preview parser: ${parsedPreview.errors.join("; ")}`);
expectDeepEqual(parsedPreview.contract.parameters.map((row) => [row.direction, row.ordinal]), [["input", 0], ["input", 1], ["output", 0]], "preview parameter identities");
expectEqual(compareInterfaceContracts(ledger, preview, artifactSha256).status, "bound_exact_contract", "CycloneDX 2.0 draft preview round trip");
const duplicatePropertyPreview = structuredClone(preview);
duplicatePropertyPreview.metadata.component.modelCard.modelParameters.inputs[0].properties.push(
  structuredClone(duplicatePropertyPreview.metadata.component.modelCard.modelParameters.inputs[0].properties[0]),
);
expectEqual(compareInterfaceContracts(ledger, duplicatePropertyPreview, artifactSha256).status, "invalid_declaration", "duplicate CycloneDX properties fail closed");

const forgedBlockedPass = buildInterfaceQuantizationContractLedger({
  outputs: [{
    index: 0,
    name: "blocked",
    dtype: "UINT8",
    shape: [1, 300],
    shape_signature: [],
    quant_scales: 29,
    quant_zero_points: 29,
    interface_scale_values: Array(29).fill(0.1),
    interface_zero_point_values: Array(29).fill(0),
    quantized_dimension: 1,
    quantization_parameterization: "blocked",
    quantization_block_size: 10,
    quantization_scale_tensor_shape: [1, 30],
    quantization_zero_point_tensor_shape: [1, 30],
    quantization_cardinality_status: "pass",
  }],
});
expectEqual(forgedBlockedPass.parameters[0].quantization.status, "invalid_or_incomplete", "blocked cardinality is independently reconstructed instead of trusting parser pass");

const unbound = compareInterfaceContracts(ledger, null, artifactSha256);
expectEqual(unbound.status, "unbound", "missing declaration stays unbound");
expectEqual(unbound.gate_result, "pending", "missing declaration is pending, not a vulnerability");

done("Interface contract comparison passed (canonical hashing, all-I/O coverage, strict field validation, independent blocked cardinality, and CycloneDX 2.0 draft round trip).");
