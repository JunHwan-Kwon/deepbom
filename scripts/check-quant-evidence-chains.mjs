import { existsSync, readFileSync } from "node:fs";

import {
  analyze_tflite_for_target,
  initSync,
} from "../pkg/tflite_wasm_audit.js";
import {
  deriveQuantInterventionPosture,
  deriveExactChannelConvergence,
  QUANT_EVIDENCE_CHAIN_COUNT,
} from "../web/lib/quant-evidence-chains.js";
import { perAxisScaleContractEvidence } from "../web/lib/performance-visuals.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual } = createCheck("Quant evidence chains");
const modelPath = "C:/Users/junhw/Downloads/main_0604_v119_4_ckpt902087_int8.tflite";

expectEqual(QUANT_EVIDENCE_CHAIN_COUNT, 4, "Quant evidence should be organized into four proof chains.");
const scaleContract = perAxisScaleContractEvidence([{
  shape: [4, 3, 3, 3], quantized_dimension: 0, quant_scales: 4,
  scale_sample: [0.25, 0.5, 1, 2], scale_ratio: 8,
}, {
  shape: [3, 4], quantized_dimension: 1, quant_scales: 3,
  scale_sample: [0, 1, 2], scale_ratio: 2,
}]);
expectEqual(scaleContract.tensorCount, 2, "Per-axis contract summary should preserve the tensor denominator.");
expectEqual(scaleContract.resolvableCount, 2, "Both synthetic quantized axes should resolve against static shapes.");
expectEqual(scaleContract.matchedCount, 1, "Scale cardinality must be checked against shape[quantized_dimension].");
expectEqual(scaleContract.nonPositiveCount, 1, "Every decoded scale value must participate in positivity validation.");
expectEqual(scaleContract.maximumSpread, 8, "Maximum per-axis scale spread should preserve the exact decoded ratio.");
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const publicBytes = new Uint8Array(readFileSync("web/samples/mobilenet_v2_1.0_224_quant.tflite"));
const publicAnalysis = analyze_tflite_for_target(publicBytes, "mobilenet_v2_1.0_224_quant.tflite", "android_mid_a55");
const posture = deriveQuantInterventionPosture(publicAnalysis);
expectEqual(posture.schema, "deepbom.quant_intervention_posture.v1", "Intervention posture schema should be pinned.");
expectEqual(posture.classification, "full_integer", "The public INT8 sample should remain full-integer.");
expectEqual(posture.q_ops, 0, "The public full-integer sample should have no explicit QUANTIZE op.");
expectEqual(posture.dq_ops, 0, "The public full-integer sample should have no explicit DEQUANTIZE op.");
expectEqual(posture.holes, 0, "The public full-integer sample should have no mid-graph Q/DQ hole.");
expectEqual(posture.lineage.evidence, "NOT EMBEDDED", "Missing QAT/PTQ lineage should remain explicit.");
expectEqual(posture.actions.find((row) => row.id === "ptq")?.state, "not-indicated", "Generic PTQ should not be recommended for an already full-integer artifact without holes.");
expectEqual(posture.actions.find((row) => row.id === "qat")?.state, "review", "Exact-zero stored channels should route to source/QAT review.");
expectEqual(posture.actions.find((row) => row.id === "reexport")?.state, "review", "Low grid utilization should route to range-generation review.");
const mixedPosture = deriveQuantInterventionPosture({
  format: "onnx",
  quantization_status: {
    classification: "mixed_quantization",
    quantized_compute_mac_percent: 0.42,
    quantize_ops: 1,
    dequantize_ops: 1,
  },
  quant_holes: [{ op_index: 2, adjacent_mac_percent: 0.25 }],
  ops: [],
  tensors: [],
  metadata_presence: { converter_optimization_modes: [] },
});
expectEqual(mixedPosture.actions.find((row) => row.id === "ptq")?.state, "candidate", "A mixed Q/DQ path should route to a controlled PTQ candidate.");
expectEqual(mixedPosture.actions.find((row) => row.id === "qat")?.state, "conditional", "QAT should remain conditional until PTQ output evidence misses requirements.");
expectEqual(mixedPosture.lineage.evidence, "NOT EMBEDDED", "An ONNX Q/DQ graph must not invent training lineage.");
expectEqual(mixedPosture.internal_conversion_ops, 1, "Internal Q/DQ conversion count should preserve its explicit operator denominator.");
const declaredQatPosture = deriveQuantInterventionPosture({
  quantization_status: { classification: "full_integer" },
  ops: [],
  tensors: [],
  metadata_presence: { converter_optimization_modes: ["QUANTIZATION_AWARE_TRAINING"] },
});
expectEqual(declaredQatPosture.lineage.label, "QAT", "QAT should be represented as training-lineage metadata, not artifact state.");
expectEqual(declaredQatPosture.lineage.evidence, "DECLARED", "QAT metadata must be labeled DECLARED rather than OBSERVED.");
if (existsSync(modelPath)) {
  const bytes = new Uint8Array(readFileSync(modelPath));
  const analysis = analyze_tflite_for_target(bytes, "main_0604_v119_4_ckpt902087_int8.tflite", "android_mid_a55");
  const proof = deriveExactChannelConvergence(analysis);
  expect(proof, "The regression artifact should expose an exact-channel cross-ledger proof.");
  expectEqual(proof.topOpIndex, 0, "The cross-ledger proof should bind operator #000.");
  expectEqual(proof.channelIndex, 19, "The cross-ledger proof should bind channel 19.");
  expectEqual(proof.termCount, 27, "The proof should bind all 27 kernel terms.");
  expectEqual(proof.zeroWeights, 27, "All centered kernel weights should be zero for the proof channel.");
  expectEqual(proof.shift, -30, "The same channel should bind requantization shift -30.");
  expectEqual(proof.outputRange, "-128", "Both pinned rounding paths should prove output code -128.");
  expect(Math.abs(proof.ratio - 0.9755693361980697) < 1e-15, "The exact INT32 ratio should remain stable.");
}

done(existsSync(modelPath)
  ? "four proof chains; exact regression cross-ledger join #000/channel 19 passed"
  : "four proof chains; external exact-channel regression artifact unavailable");
