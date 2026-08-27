import { readFileSync } from "node:fs";

import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { buildRawDataArtifactFiles } from "../web/lib/report-evidence.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("Core isolation roofline check");
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });

const quantizedBytes = new Uint8Array(readFileSync("web/samples/mobilenet_v2_1.0_224_quant.tflite"));
const analyze = (target) => analyze_tflite_for_target(quantizedBytes, "mobilenet_v2_1.0_224_quant.tflite", target);

const rpi4 = analyze("rpi4_a72");
const rpi = rpi4.core_isolation_analysis;
expectEqual(rpi.schema, "deepbom.core_isolation_roofline.v1", "Core isolation analysis schema should be explicit.");
expectEqual(rpi.status, "assessed", "An exact four-core product profile should be assessed.");
expectEqual(rpi.performance_reference_core_count, 4, "RPi4 peak throughput should bind its four-core denominator.");
expectEqual(rpi.system_core_count_options.join(","), "4", "RPi4 should expose one exact four-core system variant.");
expectEqual(rpi.scenarios.length, 4, "RPi4 should expose one through four AI-core scenarios.");
expectEqual(rpi4.core_isolation_csv.trim().split(/\r?\n/).length, 5, "Core-isolation CSV should contain one header plus every scenario.");
const report = buildEngineeringReport(rpi4, {
  identity: {
    filename: rpi4.filename,
    format: "tflite",
    sha256: rpi4.model_sha256,
    target_label: rpi4.target_profile.label,
    operator_count: rpi4.operator_count,
    tensor_count: rpi4.tensor_count,
    total_macs: rpi4.total_macs,
  },
});
expect(report.includes("## Core Allocation Roofline (ESTIMATED_STATIC_RESOURCE_PARTITION)"), "Engineering report should expose the core-allocation evidence class.");
expect(report.includes("Shared target interface ceiling is unchanged across core counts"), "Engineering report should state that memory bandwidth is not core-scaled.");
const coreCsvFile = buildRawDataArtifactFiles(rpi4).find((file) => file.name === "static/core_isolation_roofline.csv");
expectEqual(coreCsvFile?.data, rpi4.core_isolation_csv, "Raw Data should preserve the analyzer-emitted core-allocation CSV exactly.");

for (const [index, row] of rpi.scenarios.entries()) {
  const assigned = index + 1;
  expectEqual(row.ai_assigned_core_count, assigned, "Scenario rows should be sorted by assigned core count.");
  expectEqual(row.housekeeping_core_count, 4 - assigned, "AI plus housekeeping cores should conserve the system count.");
  expect(Math.abs(row.int8_issue_ceiling_gops - 48 * assigned) <= 1e-12, "INT8 issue ceiling should scale from the explicit four-core denominator.");
  expect(Math.abs(row.fp32_issue_ceiling_gops - 12 * assigned) <= 1e-12, "FP32 issue ceiling should apply the profile's 4:1 precision factor after core scaling.");
  expectEqual(row.shared_memory_bandwidth_ceiling_gbps, 9.6, "Shared interface bandwidth must not be multiplied by assigned cores.");
  expectEqual(row.memory_bandwidth_scaling, "shared_interface_ceiling_not_core_scaled", "Bandwidth scaling semantics should be machine-readable.");
  expect(Math.abs(row.compute_dominant_floor_us + row.memory_dominant_floor_us - row.utilization_adjusted_roofline_estimate_us) <= 1e-8, "Compute- and memory-dominant ledgers should conserve the utilization-adjusted roofline estimate.");
  expect(Math.abs(row.utilization_adjusted_roofline_estimate_us + row.predicted_runtime_overhead_us - row.steady_state_estimate_us) <= 1e-8, "Steady estimate should conserve roofline plus predicted runtime overhead.");
  expect(Math.abs(row.steady_state_estimate_us + row.one_time_packing_estimate_us - row.cold_start_estimate_us) <= 1e-8, "Cold estimate should conserve steady estimate plus one-time packing.");
  expect(row.theoretical_roofline_floor_us <= row.utilization_adjusted_roofline_estimate_us + 1e-9, "A utilization-adjusted estimate must not undercut its issue-ceiling floor.");
  expectEqual(row.compute_dominant_op_count + row.memory_dominant_op_count, row.assessed_op_count, "Every assessed op should enter exactly one dominant ledger.");
  expectEqual(row.allocation_class, assigned < 4 ? "exclusive_isolation_candidate" : "full_set_non_isolated_baseline", "Only scenarios retaining housekeeping cores may be called isolation candidates.");
}

const zynq = analyze("zynq_ultrascale_plus_a53").core_isolation_analysis;
expectEqual(zynq.system_core_count_options.join(","), "2,4", "Zynq should expose only documented dual- and quad-core product variants.");
expectEqual(zynq.scenarios.length, 6, "Zynq should emit two dual-core and four quad-core allocation scenarios.");
expect(!zynq.scenarios.some((row) => row.system_core_count === 3), "A product-family core-count range must not invent an undocumented intermediate variant.");
expect(zynq.scenarios.filter((row) => row.ai_assigned_core_count === 1).every((row) => Math.abs(row.int8_issue_ceiling_gops - 12) <= 1e-12), "Zynq one-core issue ceiling should use the four-core 48 GOPS reference denominator for both product variants.");

const a55 = analyze("android_mid_a55").core_isolation_analysis;
expectEqual(a55.status, "not_assessed", "An ISA-class profile with unbound topology should not fabricate core-allocation performance.");
expectEqual(a55.scenarios.length, 0, "Unbound core topology should emit no scenario rows.");
expect(a55.unavailable_reason.includes("core count"), "Suppressed scenarios should state the missing denominator.");

const customSpec = JSON.stringify({
  base: "android_mid_a55",
  id: "custom:a55-dual-bound",
  label: "Bound dual-core A55",
  evidence_class: "USER_DECLARED",
  evidence_note: "Exact topology and full-system planning peak supplied for scenario analysis.",
  overrides: {
    core_count_min: 2,
    core_count_max: 2,
    performance_reference_core_count: 2,
    effective_peak_gops: 100,
    effective_memory_bandwidth_gbps: 8,
  },
});
const custom = analyze(customSpec).core_isolation_analysis;
expectEqual(custom.status, "assessed", "A custom profile with every core denominator bound should be assessable.");
expectEqual(custom.scenarios.length, 2, "A bound dual-core custom profile should emit one- and two-core scenarios.");
expectEqual(custom.scenarios[0].int8_issue_ceiling_gops, 50, "Custom per-core scaling should use the declared two-core 100 GOPS reference.");
const invalidReferenceSpec = JSON.stringify({
  ...JSON.parse(customSpec),
  id: "custom:a55-invalid-reference",
  overrides: { ...JSON.parse(customSpec).overrides, performance_reference_core_count: 3 },
});
expectThrows(() => analyze(invalidReferenceSpec), "must match one declared system core-count variant", "A peak denominator outside the declared product variants must fail closed.");

const floatBytes = new Uint8Array(readFileSync("web/samples/mobilenet_v1_025_224_float.tflite"));
const floatAnalysis = analyze_tflite_for_target(floatBytes, "mobilenet_v1_025_224_float.tflite", "rpi4_a72");
const floatCompute = floatAnalysis.ops.find((op) => Number(op.ops) > 0 && !op.quantized_compute_path);
const expectedFloatComputeUs = Math.max(Number(floatCompute.ops), Number(floatCompute.macs) * 2) / ((192 / 4) * 1e9) * 1e6;
expect(Math.abs(floatCompute.bottleneck_compute_us - expectedFloatComputeUs) <= 1e-9, "Existing per-op bottleneck estimates should use the FP32 ceiling rather than the INT8 peak for float compute.");

done();
