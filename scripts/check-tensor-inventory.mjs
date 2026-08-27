import { readFileSync } from "node:fs";
import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import { analyzeOnnxModel } from "../web/onnx.js";
import {
  buildTensorInventory,
  classifyTensorRoles,
  tensorInventoryConserves,
} from "../web/lib/tensor-inventory.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { buildQuantizationContractChecks } from "../web/lib/report-quantization-contracts.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual } = createCheck("Tensor inventory check");

const synthetic = buildTensorInventory({
  format: "tflite",
  tensors: [
    { index: 10, dtype: "UINT8", constant_buffer: false, quant_scales: 1, scale_mode: "per-tensor" },
    { index: 20, dtype: "INT8", constant_buffer: true, quant_scales: 4, scale_mode: "per-channel" },
    { index: 30, dtype: "INT32", constant_buffer: true, quant_scales: 4, scale_mode: "per-channel" },
    { index: 40, dtype: "INT32", constant_buffer: true, quant_scales: 0, scale_mode: "none" },
    { index: 50, dtype: "UINT8", constant_buffer: false, quant_scales: 1, scale_mode: "per-tensor" },
  ],
  ops: [{ name: "CONV_2D", inputs: [10, 20, 30], outputs: [50] }],
});

expect(tensorInventoryConserves(synthetic), "Synthetic role and quantization ledgers should conserve all tensors.");
expectEqual(synthetic.tensor_count, 5, "Synthetic tensor count should be exact.");
expectEqual(synthetic.quantized_tensors, 4, "Synthetic quantized count should be exact.");
expectEqual(synthetic.per_channel_tensors, 2, "Kernel and bias should be per-channel.");
expectEqual(synthetic.per_tensor_tensors, 2, "Input and output activations should be per-tensor.");
expectEqual(synthetic.rows.find((row) => row.role === "kernel")?.per_channel, 1, "Kernel role should bind from the Conv input signature.");
expectEqual(synthetic.rows.find((row) => row.role === "bias")?.per_channel, 1, "Bias role should bind from the Conv input signature.");
expectEqual(synthetic.rows.find((row) => row.role === "activation")?.per_tensor, 2, "Non-constant tensors should be activations.");
expectEqual(synthetic.rows.find((row) => row.role === "metadata")?.total, 1, "Unmatched constants should remain conservative metadata/parameters.");
expectEqual(
  classifyTensorRoles({
    format: "tflite",
    tensors: [
      { index: 0, constant_buffer: false },
      { index: 1, constant_buffer: true },
      { index: 2, constant_buffer: true },
    ],
    ops: [{ name: "CONV_2D", inputs: [0, 1, 2] }],
  }).map((row) => row.role).join(","),
  "activation,kernel,bias",
  "The reusable tensor-role classifier should match the inventory ledger.",
);

const scaleCollapseAnalysis = {
  format: "tflite",
  inputs: [{ index: 0, name: "input", dtype: "INT8", shape: [1, 4, 4, 2], quant_scales: 1, scale_sample: [0.0316], zero_point_sample: [0] }],
  outputs: [{ index: 3, name: "output", dtype: "INT8", shape: [1, 4, 4, 2], quant_scales: 1, scale_sample: [0.05], zero_point_sample: [0] }],
  tensors: [
    { index: 0, name: "input", dtype: "INT8", shape: [1, 4, 4, 2], quant_scales: 1, scale_sample: [0.0316], zero_point_sample: [0] },
    { index: 1, name: "kernel", dtype: "INT8", shape: [2, 1, 1, 2], constant_buffer: true, quant_scales: 2, quantized_dimension: 0, scale_sample: [5.35e-10, 7.12e-3], zero_point_sample: [0, 0], scale_ratio_meaningful: true, scale_ratio: 7.12e-3 / 5.35e-10, scale_min: 5.35e-10, scale_max: 7.12e-3 },
    { index: 2, name: "bias", dtype: "INT32", shape: [2], constant_buffer: true, quant_scales: 2, quantized_dimension: 0, scale_sample: [0.0316 * 5.35e-10, 0.0316 * 7.12e-3], zero_point_sample: [0, 0], scale_ratio_meaningful: true, scale_ratio: 7.12e-3 / 5.35e-10, scale_min: 0.0316 * 5.35e-10, scale_max: 0.0316 * 7.12e-3 },
    { index: 3, name: "output", dtype: "INT8", shape: [1, 4, 4, 2], quant_scales: 1, scale_sample: [0.05], zero_point_sample: [0] },
  ],
  ops: [{ index: 0, name: "CONV_2D", inputs: [0, 1, 2], outputs: [3] }],
  metadata_presence: { documented_preprocessing: true },
};
const scaleCollapseChecks = buildQuantizationContractChecks(scaleCollapseAnalysis);
expectEqual(scaleCollapseChecks.bias_scale.status, "pass", "Bias scales should exactly reproduce input_scale * weight_scale.");
expect(Math.abs(scaleCollapseChecks.bias_scale.details[0].declared_input_scale - 0.0316) < 1e-12, "Bias-scale detail should expose the declared input activation scale.");
expect(Math.abs(scaleCollapseChecks.bias_scale.details[0].derived_input_scale_min - 0.0316) < 1e-9, "bias_scale / weight_scale should recover the input activation scale.");
expectEqual(scaleCollapseChecks.representable_kernel_channels.flagged_channels, 1, "The 5.35e-10 kernel scale should flag one near-zero representable channel.");
expect(Math.abs(scaleCollapseChecks.representable_kernel_channels.details[0].flagged_channels[0].maximum_representable_abs - 127 * 5.35e-10) < 1e-15, "Maximum representable INT8 weight should use the TFLite symmetric [-127,127] kernel domain.");
expect(buildFindingsRegister(scaleCollapseAnalysis).some((finding) => finding.finding_id === "EA-QNT-0118"), "Near-zero representable kernel channels should enter the action queue.");

const conflicting = buildTensorInventory({
  format: "tflite",
  tensors: [
    { index: 0, dtype: "FLOAT32", constant_buffer: false },
    { index: 1, dtype: "FLOAT32", constant_buffer: true },
    { index: 2, dtype: "FLOAT32", constant_buffer: true },
  ],
  ops: [
    { name: "CONV_2D", inputs: [0, 1, 2] },
    { name: "FULLY_CONNECTED", inputs: [0, 2] },
  ],
});
expectEqual(
  conflicting.rows.find((row) => row.role === "metadata")?.total,
  1,
  "A constant bound to conflicting kernel/bias signatures should fail conservatively to metadata/parameters.",
);

initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const quantBytes = new Uint8Array(readFileSync("web/samples/mobilenet_v2_1.0_224_quant.tflite"));
const quantAnalysis = analyze_tflite_for_target(
  quantBytes,
  "mobilenet_v2_1.0_224_quant.tflite",
  "android_mid_a55",
);
const quant = buildTensorInventory(quantAnalysis);

expect(tensorInventoryConserves(quant), "Quant MobileNet inventory should conserve every tensor and quantization mode.");
expectEqual(quant.tensor_count, 173, "Quant MobileNet tensor count should remain exact.");
expectEqual(quant.quantized_tensors, 172, "Quant MobileNet should retain 172 parameterized tensors.");
expectEqual(quant.per_tensor_tensors, 172, "Legacy quant MobileNet should retain per-tensor parameterization.");
expectEqual(quant.per_channel_tensors, 0, "Legacy quant MobileNet should contain no per-channel tensors.");
for (const [role, dtype, expected] of [
  ["kernel", "UINT8", 53],
  ["bias", "INT32", 53],
  ["activation", "UINT8", 66],
  ["metadata", "INT32", 1],
]) {
  expectEqual(
    quant.rows.find((row) => row.role === role && row.dtype === dtype)?.total,
    expected,
    `Quant MobileNet ${role}/${dtype} count should be exact.`,
  );
}

const onnxBytes = new Uint8Array(readFileSync("web/samples/sample_cnn_float.onnx"));
const onnx = buildTensorInventory(analyzeOnnxModel(onnxBytes, "sample_cnn_float.onnx"));
expect(tensorInventoryConserves(onnx), "ONNX sample inventory should conserve every tensor.");
for (const [role, expected] of [["kernel", 3], ["bias", 3], ["activation", 10]]) {
  expectEqual(
    onnx.rows.find((row) => row.role === role && row.dtype === "FLOAT32")?.total,
    expected,
    `ONNX sample ${role}/FLOAT32 count should be exact.`,
  );
}

done("Tensor role, dtype, and quantization-mode inventories are conserved.");
