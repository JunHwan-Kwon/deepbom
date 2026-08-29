import { File } from "node:buffer";

import { readArtifactBundle } from "../web/lib/artifact-bundle.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { buildEngineeringEvidenceDocument } from "../web/lib/report-evidence.js";
import { buildPublicCycloneDx17ArtifactContract } from "../web/lib/public-cyclonedx-export.js";
import { deriveCurrentArtifactCapabilityRow } from "../web/lib/format-capability-view.js";
import { buildSafeTensorsQuantizationContract } from "../web/lib/safetensors-quantization-contract.js";
import { createCheck } from "./check-assert.mjs";
import { assertCycloneDx17 } from "./cyclonedx-17-schema.mjs";

const { done, expect, expectEqual } = createCheck("SafeTensors quantization contract check");

const awqTensors = packedTensors("layer.proj", {
  qweight: ["I32", [128, 2], new Array(256).fill("0")],
  qzeros: ["I32", [1, 2], ["0", "0"]],
  scales: ["F16", [1, 16], new Array(16).fill(1)],
});
const awq = buildSafeTensorsQuantizationContract({
  quantization_config: { quant_method: "awq", bits: 4, group_size: 128, zero_point: true, version: "gemm" },
}, awqTensors, { sidecars: { quant_config: { path: "quant_config.json", document: { w_bit: 4, q_group_size: 128, zero_point: true, version: "GEMM" } } } });
expectEqual(awq.status, "assessed", "AWQ structural contract should pass.");
expectEqual(awq.module_count, 1, "AWQ module count should be exact.");
expectEqual(awq.logical_weight_element_count, "2048", "AWQ logical weight cardinality should be reconstructed.");
expectEqual(awq.packed_weight_code_capacity, "2048", "AWQ packed code capacity should conserve logical weights.");
expectEqual(awq.scale_element_count, "16", "AWQ scale cardinality should be exact.");
expectEqual(awq.zero_point_code_capacity, "16", "AWQ zero-point code capacity should be exact.");
expectEqual(awq.modules[0].stored_group_axis, 0, "AWQ stored group axis should be explicit.");

const gptqTensors = packedTensors("layer.proj", {
  qweight: ["I32", [16, 16]], qzeros: ["I32", [1, 2]], scales: ["F16", [1, 16]], g_idx: ["I32", [128]],
});
const gptq = buildSafeTensorsQuantizationContract({
  quantization_config: { quant_method: "gptq", bits: 4, group_size: 128, sym: true, desc_act: true },
}, gptqTensors);
expectEqual(gptq.status, "assessed", "GPTQ structural contract should pass.");
expectEqual(gptq.modules[0].zero_point_storage_transform, "packed_zero_code_minus_one", "GPTQ packed zero transform should remain source-specific.");
expectEqual(gptq.logical_weight_element_count, "2048", "GPTQ logical weight cardinality should be reconstructed.");
expectEqual(gptq.packed_weight_code_capacity, "2048", "GPTQ packed code capacity should conserve logical weights.");

const hqqTensors = packedTensors("model.layers.0.mlp.down_proj", {
  W_q: ["U8", [16, 64]], scale: ["F16", [32, 1]], zero: ["F16", [32, 1]],
  shape: ["I64", [2], ["16", "128"]], nbits: ["I32", [], ["4"]],
  group_size: ["I32", [], ["64"]], axis: ["I32", [], ["1"]],
  packing: ["U8", [7], [..."4bit_u8"].map((value) => String(value.charCodeAt(0)))],
});
const hqq = buildSafeTensorsQuantizationContract({
  quantization_config: {
    quant_method: "hqq",
    quant_config: { weight_quant_params: { nbits: 4, group_size: 64, axis: 1, channel_wise: true, view_as_float: false } },
  },
}, hqqTensors);
expectEqual(hqq.status, "assessed", "HQQ encoded SafeTensors contract should pass.");
expectEqual(hqq.logical_weight_element_count, "2048", "HQQ original shape should restore exact logical cardinality.");
expectEqual(hqq.packed_weight_storage_bits, "8192", "HQQ U8 packed payload should restore exact storage bits.");
expectEqual(hqq.packing_padding_bits, "0", "HQQ 4-bit packing should conserve without padding for this shape.");
expectEqual(hqq.modules[0].packing_layout, "4bit_u8", "HQQ source-defined packing identity should be explicit.");

const hqqDynamicTensors = [
  ...packedTensors("model.layers.0.self_attn.q_proj", {
    W_q: ["U8", [16, 64]], scale: ["F16", [32, 1]], zero: ["F16", [32, 1]],
    shape: ["I64", [2], ["16", "128"]], nbits: ["I32", [], ["4"]], group_size: ["I32", [], ["64"]], axis: ["I32", [], ["1"]],
    packing: ["U8", [7], [..."4bit_u8"].map((value) => String(value.charCodeAt(0)))],
  }),
  ...packedTensors("model.layers.1.mlp.down_proj", {
    W_q: ["U8", [4, 128]], scale: ["F16", [16, 1]], zero: ["F16", [16, 1]],
    shape: ["I64", [2], ["16", "128"]], nbits: ["I32", [], ["2"]], group_size: ["I32", [], ["128"]], axis: ["I32", [], ["1"]],
    packing: ["U8", [7], [..."2bit_u8"].map((value) => String(value.charCodeAt(0)))],
  }),
].map((tensor, index) => ({ ...tensor, index, shard_path: index < 8 ? "model-00001-of-00002.safetensors" : "model-00002-of-00002.safetensors" }));
const hqqDynamic = buildSafeTensorsQuantizationContract({
  quantization_config: {
    quant_method: "hqq",
    quant_config: {
      "self_attn.q_proj": { weight_quant_params: { nbits: 4, group_size: 64, axis: 1, channel_wise: true, view_as_float: false } },
      "mlp.down_proj": { weight_quant_params: { nbits: 2, group_size: 128, axis: 1, channel_wise: true, view_as_float: false } },
    },
  },
}, hqqDynamicTensors);
expectEqual(hqqDynamic.status, "assessed", "HQQ module-tagged dynamic config should be assessed per encoded module.");
expectEqual(hqqDynamic.configuration_scope, "source_bound_module_tag", "HQQ dynamic ownership rule should be explicit.");
expectEqual(hqqDynamic.bits, null, "Heterogeneous HQQ module bit widths must not be collapsed to one model-level value.");
expectEqual(hqqDynamic.modules.find((row) => row.name.endsWith("q_proj")).bits, 4, "HQQ q_proj tag should select its 4-bit contract.");
expectEqual(hqqDynamic.modules.find((row) => row.name.endsWith("down_proj")).bits, 2, "HQQ down_proj tag should select its 2-bit contract.");
expectEqual(hqqDynamic.shard_ownership.tensor_count, hqqDynamicTensors.length, "HQQ encoded metadata tensors should participate in one shard-ownership ledger.");
expectEqual(hqqDynamic.shard_ownership.status, "assessed_all_quantization_tensors_shard_bound", "HQQ shard ownership should close only when every encoded tensor is manifest-bound.");

const compressedTensors = packedTensors("model.layers.0.self_attn.q_proj", {
  weight_packed: ["I32", [16, 16]], weight_scale: ["F16", [16, 2]],
  weight_shape: ["I64", [2], ["16", "128"]],
});
const compressed = buildSafeTensorsQuantizationContract({
  quantization_config: {
    quant_method: "compressed-tensors", format: "pack-quantized",
    config_groups: { group_0: { targets: ["Linear"], weights: { num_bits: 4, group_size: 64, strategy: "group", type: "int", symmetric: true, dynamic: false } } },
  },
}, compressedTensors);
expectEqual(compressed.status, "assessed", "compressed-tensors pack-quantized contract should pass.");
expectEqual(compressed.logical_weight_element_count, "2048", "compressed-tensors weight_shape should restore exact logical cardinality.");
expectEqual(compressed.packed_weight_storage_bits, "8192", "compressed-tensors INT32 packing should restore exact storage bits.");
expectEqual(compressed.packing_padding_bits, "0", "compressed-tensors bit packing should conserve this shape without padding.");
expectEqual(compressed.modules[0].group_count, 2, "compressed-tensors scale shape should bind the exact group count.");

const compressedScopedTensors = [
  ...packedTensors("model.layers.0.self_attn.q_proj", {
    weight_packed: ["I32", [16, 16]], weight_scale: ["F16", [16, 2]], weight_shape: ["I64", [2], ["16", "128"]], input_scale: ["F16", [1]],
  }),
  ...packedTensors("model.layers.0.mlp.down_proj", {
    weight_packed: ["I32", [16, 32]], weight_scale: ["F16", [16, 1]], weight_shape: ["I64", [2], ["16", "128"]],
  }),
].map((tensor, index) => ({ ...tensor, index, shard_path: index < 4 ? "model-00001-of-00002.safetensors" : "model-00002-of-00002.safetensors" }));
const compressedScoped = buildSafeTensorsQuantizationContract({
  quantization_config: {
    quant_method: "compressed-tensors", format: "pack-quantized",
    config_groups: {
      fallback: { targets: ["Linear"], weights: { num_bits: 8, group_size: 128, strategy: "group", type: "int", symmetric: true, dynamic: false } },
      q_proj: {
        targets: ["model.layers.0.self_attn.q_proj"],
        weights: { num_bits: 4, group_size: 64, strategy: "group", type: "int", symmetric: true, dynamic: false },
        input_activations: { num_bits: 8, strategy: "tensor", type: "int", symmetric: true, dynamic: false },
      },
    },
  },
}, compressedScopedTensors);
expectEqual(compressedScoped.status, "assessed", "compressed-tensors heterogeneous target groups should be assessed per module.");
expectEqual(compressedScoped.bits, null, "Heterogeneous compressed-tensors bit widths must not be collapsed to one model-level value.");
expectEqual(compressedScoped.modules.find((row) => row.name.endsWith("q_proj")).bits, 4, "Exact module-name target must outrank the Linear class fallback.");
expectEqual(compressedScoped.modules.find((row) => row.name.endsWith("down_proj")).bits, 8, "Linear class fallback should bind an otherwise unmatched packed module.");
expectEqual(compressedScoped.activation_quantization_contract_count, 1, "Static input activation companion should be retained as a separate contract.");
expectEqual(compressedScoped.activation_quantization_contracts[0].status, "assessed_static_serialized_companions", "Static activation scale ownership should be assessed.");
expectEqual(compressedScoped.shard_ownership.status, "assessed_all_quantization_tensors_shard_bound", "Every packed companion tensor should retain its manifest-bound shard owner.");
expectEqual(compressedScoped.shard_ownership.tensor_count, compressedScopedTensors.length, "Weight and activation companions should share the same shard-ownership ledger.");

const unsupportedPythonRegex = buildSafeTensorsQuantizationContract({
  quantization_config: {
    quant_method: "compressed-tensors", format: "pack-quantized",
    config_groups: { group_0: { targets: ["re:(?P<layer>.*q_proj)$"], weights: { num_bits: 4, group_size: 64, strategy: "group", type: "int", symmetric: true, dynamic: false } } },
  },
}, compressedTensors);
expectEqual(unsupportedPythonRegex.status, "not_assessed_compressed_tensors_config_invalid", "Python-only regex syntax must fail closed instead of being reinterpreted by JavaScript.");

const orphanKvScale = packedTensors("model.layers.0.self_attn", { k_scale: ["F16", [2]], v_scale: ["F16", [1]] });
const rejectedKvScale = buildSafeTensorsQuantizationContract({
  quantization_config: {
    quant_method: "compressed-tensors", format: "pack-quantized",
    config_groups: { group_0: { targets: ["Linear"], weights: { num_bits: 4, group_size: 64, strategy: "group", type: "int", symmetric: true, dynamic: false } } },
    kv_cache_scheme: { num_bits: 8, strategy: "tensor", type: "int", symmetric: true, dynamic: false },
  },
}, [...compressedTensors, ...orphanKvScale]);
expectEqual(rejectedKvScale.status, "fail", "Malformed standalone KV activation companions must propagate to the top-level contract.");
expect(rejectedKvScale.config_issues.some((issue) => issue.includes("k_scale_shape_must_be_scalar_cardinality_one")), "KV companion cardinality failure should remain machine-readable.");

const compressedIgnored = buildSafeTensorsQuantizationContract({
  quantization_config: {
    quant_method: "compressed-tensors", format: "pack-quantized", ignore: ["model.layers.0.self_attn.q_proj"],
    config_groups: { group_0: { targets: ["Linear"], weights: { num_bits: 4, group_size: 64, strategy: "group", type: "int", symmetric: true, dynamic: false } } },
  },
}, compressedTensors);
expectEqual(compressedIgnored.status, "fail", "An encoded packed module selected by the global ignore rule must fail closed.");
expect(compressedIgnored.modules[0].issues.includes("packed_module_matches_global_ignore"), "compressed-tensors ignore ownership mismatch should be explicit.");

const malformedHqq = structuredClone(hqqTensors);
malformedHqq.find((tensor) => tensor.name.endsWith(".shape")).numerical_integrity.decoded_values = ["16", "129"];
const rejectedHqq = buildSafeTensorsQuantizationContract({
  quantization_config: { quant_method: "hqq", quant_config: { weight_quant_params: { nbits: 4, group_size: 64, axis: 1, channel_wise: true } } },
}, malformedHqq);
expectEqual(rejectedHqq.status, "fail", "HQQ metadata/payload shape disagreement must fail closed.");
expect(rejectedHqq.modules[0].issues.includes("logical_weight_count_not_divisible_by_group_size"), "HQQ non-divisible logical grouping should be explicit.");

const conflict = buildSafeTensorsQuantizationContract({
  quantization_config: { quant_method: "awq", bits: 4, group_size: 128, zero_point: true, version: "gemm" },
}, awqTensors, { sidecars: { quant_config: { path: "quant_config.json", document: { w_bit: 8, q_group_size: 128, zero_point: true, version: "GEMM" } } } });
expectEqual(conflict.status, "fail", "Conflicting quantization declarations must fail closed.");
expect(conflict.declaration_conflicts.some((row) => row.field === "bits"), "Conflicting bit widths should be identified.");

const malformed = buildSafeTensorsQuantizationContract({
  quantization_config: { quant_method: "gptq", bits: 4, group_size: 128, sym: true },
}, packedTensors("layer.proj", {
  qweight: ["I32", [16, 16]], qzeros: ["I32", [2, 2]], scales: ["F16", [1, 16]], g_idx: ["I32", [128]],
}));
expectEqual(malformed.status, "fail", "A malformed GPTQ zero-point shape must fail.");
expect(malformed.modules[0].issues.includes("qzeros_shape_mismatch"), "GPTQ qzeros shape defect should be explicit.");

const bundleFiles = [
  browserFile("fixture/model.safetensors", safeTensorBytes(awqTensors)),
  browserFile("fixture/config.json", JSON.stringify({ quantization_config: { quant_method: "awq", bits: 4, group_size: 128, zero_point: true, version: "gemm" } })),
  browserFile("fixture/quant_config.json", JSON.stringify({ w_bit: 4, q_group_size: 128, zero_point: true, version: "GEMM" })),
];
const { analysis } = await readArtifactBundle(bundleFiles);
expectEqual(analysis.safetensors.quantization_contract.status, "assessed", "Bundle analysis should expose the AWQ contract.");
expectEqual(analysis.safetensors.quantization_contract.modules[0].quantization_payload_integrity.status, "assessed", "Bundle analysis should bind full-payload scale and packed-zero integrity without a second payload scan.");
expectEqual(analysis.artifact_bundle.model_source_file_count, 3, "Quantization config must be bound into model-source identity.");
expect(analysis.artifact_bundle.model_source_roles.includes("quantization_config"), "Model-source identity should name the quantization config role.");
expect(analysis.artifact_bundle.files.some((row) => row.role === "quantization_config" && row.required), "Quantization config should be required and hash-bound.");

const gptqPayloadTensors = packedTensors("layer.proj", {
  qweight: ["I32", [32, 64], new Array(2048).fill("0")],
  qzeros: ["I32", [2, 8], new Array(16).fill("0")],
  scales: ["F16", [2, 64], new Array(128).fill(1)],
  g_idx: ["I32", [256], [...new Array(128).fill("0"), ...new Array(128).fill("1")]],
});
const { analysis: gptqPayloadAnalysis } = await readArtifactBundle([
  browserFile("gptq/model.safetensors", safeTensorBytes(gptqPayloadTensors)),
  browserFile("gptq/config.json", JSON.stringify({ quantization_config: { quant_method: "gptq", bits: 4, group_size: 128, sym: true, desc_act: true } })),
]);
const gptqPayloadModule = gptqPayloadAnalysis.safetensors.quantization_contract.modules[0];
const gptqGroupIndex = gptqPayloadModule.quantization_payload_integrity.tensors.g_idx;
expectEqual(gptqPayloadAnalysis.safetensors.quantization_contract.status, "assessed", "A valid full GPTQ payload should remain assessed.");
expectEqual(gptqGroupIndex.status, "assessed_full_payload", "GPTQ g_idx should be scanned across its complete payload.");
expectEqual(gptqGroupIndex.semantic.role, "group_index", "GPTQ g_idx semantic ownership should be explicit.");
expectEqual(gptqGroupIndex.semantic.integer_value_contract, true, "GPTQ g_idx full payload should retain its integer contract.");
expectEqual(gptqPayloadModule.quantization_payload_integrity.tensors.scales.value_count, 128, "Scale validation must cover values beyond the retained-value preview limit.");
expectEqual(gptqPayloadModule.quantization_payload_integrity.tensors.scales.semantic.nonpositive_scale_count, 0, "Every GPTQ scale should be finite and positive.");
expectEqual(gptqGroupIndex.distinct_finite_values, 2, "GPTQ group-index cardinality should be exact over the full payload.");
expectEqual(gptqPayloadModule.quantization_payload_integrity.tensors.qzeros.selected_packed_lane_profile.lane_count, 128, "Packed zero-point lane cardinality should conserve the declared group/output contract.");
expectEqual(gptqPayloadAnalysis.tensors.find((tensor) => tensor.name.endsWith(".g_idx")).numerical_integrity.decoded_values_status,
  "not_retained_above_small_tensor_limit", "Large g_idx validation must not depend on retaining a decoded preview.");

const invalidGptqPayloadTensors = packedTensors("layer.proj", {
  qweight: ["I32", [32, 64], new Array(2048).fill("0")],
  qzeros: ["I32", [2, 8], new Array(16).fill("0")],
  scales: ["F16", [2, 64], new Array(128).fill(1)],
  g_idx: ["I32", [256], [...new Array(255).fill("0"), "2"]],
});
const { analysis: invalidGptqPayloadAnalysis } = await readArtifactBundle([
  browserFile("gptq-invalid/model.safetensors", safeTensorBytes(invalidGptqPayloadTensors)),
  browserFile("gptq-invalid/config.json", JSON.stringify({ quantization_config: { quant_method: "gptq", bits: 4, group_size: 128, sym: true, desc_act: true } })),
]);
expectEqual(invalidGptqPayloadAnalysis.safetensors.quantization_contract.status, "fail", "Out-of-domain g_idx in a payload larger than the preview limit must fail closed.");
expect(invalidGptqPayloadAnalysis.safetensors.quantization_contract.modules[0].issues.includes("g_idx_payload_outside_group_domain"),
  "Large g_idx domain failure should remain machine-readable.");

const invalidScalePayloadTensors = structuredClone(gptqPayloadTensors);
invalidScalePayloadTensors.find((tensor) => tensor.name.endsWith(".scales")).numerical_integrity.decoded_values[127] = "0";
const { analysis: invalidScalePayloadAnalysis } = await readArtifactBundle([
  browserFile("gptq-invalid-scale/model.safetensors", safeTensorBytes(invalidScalePayloadTensors)),
  browserFile("gptq-invalid-scale/config.json", JSON.stringify({ quantization_config: { quant_method: "gptq", bits: 4, group_size: 128, sym: true, desc_act: true } })),
]);
expectEqual(invalidScalePayloadAnalysis.safetensors.quantization_contract.status, "fail", "A non-positive scale beyond the preview limit must fail closed.");
expect(invalidScalePayloadAnalysis.safetensors.quantization_contract.modules[0].issues.includes("scales_payload_must_be_finite_and_positive"),
  "Large scale-domain failure should remain machine-readable.");

const { analysis: hqqBundleAnalysis } = await readArtifactBundle([
  browserFile("hqq/model.safetensors", safeTensorBytes(hqqTensors)),
  browserFile("hqq/config.json", JSON.stringify({ quantization_config: { quant_method: "hqq", quant_config: { weight_quant_params: { nbits: 4, group_size: 64, axis: 1, channel_wise: true, view_as_float: false } } } })),
]);
expectEqual(hqqBundleAnalysis.safetensors.quantization_contract.status, "assessed", "Bundle numerical scan should bind HQQ encoded metadata to the structural contract.");
expectEqual(hqqBundleAnalysis.safetensors.quantization_contract.modules[0].packing_layout, "4bit_u8", "Bundle HQQ packing string should be decoded from the serialized U8 metadata tensor.");
const capability = deriveCurrentArtifactCapabilityRow("safetensors", analysis);
expect(capability.cells[2].id === "source"
  && capability.cells[2].title.includes("AWQ 4-bit")
  && capability.cells[2].title.includes("2048 logical weight codes"), "Evidence Capability should surface the source-pinned packed-weight contract without promoting it to runtime evidence.");

const report = buildEngineeringReport(analysis);
expect(report.includes("SafeTensors Packed-weight Quantization Contract")
  && report.includes("2048 / 2048")
  && report.includes(awq.source.sha256)
  && report.includes("Complete Packed-module Ledger"), "Engineering report should expose exact packed-layout conservation and its source pin.");
const evidence = buildEngineeringEvidenceDocument(analysis, {
  reportContext: { identity: { filename: analysis.filename, format: "safetensors", sha256: analysis.model_sha256 } },
  rawEvidenceContext: { identity: { filename: analysis.filename, format: "safetensors", sha256: analysis.model_sha256 } },
});
expect(evidence.evidence.conformance_report.checks.some((row) => row.id === "CF-SAFETENSORS-QUANT-001" && row.status === "pass"), "Release conformance should independently reconstruct the packed-weight contract.");
expectEqual(evidence.evidence.static_analysis.safetensors.quantization_contract.logical_weight_element_count, "2048", "Machine evidence should retain the complete quantization contract.");

const publicBom = buildPublicCycloneDx17ArtifactContract(analysis, { generatedAt: "2026-08-24T00:00:00.000Z" });
assertCycloneDx17(publicBom, "SafeTensors packed-weight public evidence BOM");
const publicProperties = new Map(publicBom.metadata.component.properties.map((row) => [row.name, row.value]));
expectEqual(publicProperties.get("deepbom:model:packedWeightQuantizationMethod"), "awq", "CycloneDX should expose the packed-weight method.");
expectEqual(publicProperties.get("deepbom:model:packedWeightQuantizationGroupSize"), "128", "CycloneDX should expose the exact group size.");
expectEqual(publicProperties.get("deepbom:model:packedWeightQuantizationEvidencePointer"), "/format_extensions/safetensors/quantization_contract", "CycloneDX should point to the full machine evidence instead of duplicating its module ledger.");

const tampered = structuredClone(analysis);
tampered.safetensors.quantization_contract.logical_weight_element_count = "2049";
let rejected = false;
try {
  buildEngineeringEvidenceDocument(tampered, {
    reportContext: { identity: { filename: tampered.filename, format: "safetensors", sha256: tampered.model_sha256 } },
    rawEvidenceContext: { identity: { filename: tampered.filename, format: "safetensors", sha256: tampered.model_sha256 } },
  });
} catch (error) {
  rejected = String(error?.message || error).includes("CF-SAFETENSORS-QUANT-001");
}
expect(rejected, "Tampered packed-weight conservation must fail release conformance.");

done("SafeTensors AWQ/GPTQ/HQQ/compressed-tensors ownership, packing, bit conservation, UI/report data, CycloneDX projection, and fail-closed conformance passed.");

function packedTensors(base, fields) {
  let offset = 0;
  return Object.entries(fields).map(([suffix, [dtype, shape, decodedValues]], index) => {
    const width = { U8: 1, I32: 4, I64: 8, F16: 2 }[dtype];
    const bytes = shape.reduce((product, value) => product * value, 1) * width;
    const row = { index, name: `${base}.${suffix}`, dtype, shape, byte_length: bytes, data_offset: offset, data_end: offset + bytes };
    if (decodedValues) row.numerical_integrity = { decoded_values_status: "complete_exact_decimal_integer_decoding", decoded_values: decodedValues };
    offset += bytes;
    return row;
  });
}

function safeTensorBytes(tensors) {
  let offset = 0;
  const header = { __metadata__: { format: "pt" } };
  for (const tensor of tensors) {
    header[tensor.name] = { dtype: tensor.dtype, shape: tensor.shape, data_offsets: [offset, offset + tensor.byte_length] };
    offset += tensor.byte_length;
  }
  const encoded = new TextEncoder().encode(JSON.stringify(header));
  const output = new Uint8Array(8 + encoded.length + offset);
  new DataView(output.buffer).setBigUint64(0, BigInt(encoded.length), true);
  output.set(encoded, 8);
  const view = new DataView(output.buffer);
  const payloadStart = 8 + encoded.length;
  for (const tensor of tensors) {
    const values = tensor.numerical_integrity?.decoded_values || [];
    const valueCount = tensor.shape.reduce((product, dimension) => product * dimension, 1);
    for (let index = 0; index < valueCount; index += 1) {
      const value = values[index] ?? (tensor.dtype === "F16" && /(?:^|[._])(?:scales?|k_scale|v_scale)$/.test(tensor.name) ? 1 : 0);
      const position = payloadStart + tensor.data_offset;
      if (tensor.dtype === "U8") view.setUint8(position + index, Number(value));
      else if (tensor.dtype === "I32") view.setInt32(position + index * 4, Number(value), true);
      else if (tensor.dtype === "I64") view.setBigInt64(position + index * 8, BigInt(value), true);
      else if (tensor.dtype === "F16") view.setUint16(position + index * 2, float16Bits(Number(value)), true);
    }
  }
  return output;
}

function float16Bits(value) {
  const buffer = new ArrayBuffer(4);
  const view = new DataView(buffer);
  view.setFloat32(0, value, false);
  const bits = view.getUint32(0, false);
  const sign = bits >>> 16 & 0x8000;
  let exponent = (bits >>> 23 & 0xff) - 127 + 15;
  let mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | (mantissa + 0x1000 >>> 13);
  }
  if (exponent >= 31) return sign | 0x7c00;
  mantissa += 0x1000;
  if (mantissa & 0x800000) { mantissa = 0; exponent += 1; }
  return exponent >= 31 ? sign | 0x7c00 : sign | exponent << 10 | mantissa >>> 13;
}

function browserFile(path, value) {
  const file = new File([value], path.split("/").at(-1), { type: path.endsWith(".json") ? "application/json" : "application/octet-stream" });
  Object.defineProperty(file, "webkitRelativePath", { value: path });
  return file;
}
