import { existsSync, readFileSync } from "node:fs";

import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import {
  validateResidualStepResponse,
  validateResidualStepResponseDigests,
} from "../web/lib/residual-step-response.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("Residual step response check");
const filename = "mobilenet_v2_1.0_224_quant.tflite";
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const bytes = new Uint8Array(readFileSync(`web/samples/${filename}`));
const analysis = analyze_tflite_for_target(bytes, filename, "android_mid_a55");
const result = analysis.residual_step_response;

expect(validateResidualStepResponse(result, analysis), "Browser arithmetic should reconstruct every step-response contract.");
await validateResidualStepResponseDigests(result, analysis);
expectEqual(result.schema, "deepbom.residual_step_response.v1", "Step-response schema should remain stable.");
expectEqual(result.status, "assessed", "Every sample residual should be assessed.");
expectEqual(result.assessed_add_count, 10, "All sample residual ADDs should be assessed.");
expectEqual(result.contract_response_count, 30, "Current and both containment contracts should be assessed per residual.");
expectEqual(result.total_transition_count, 3_916_800, "Every legal adjacent branch transition should be counted.");
expectEqual(result.total_joint_interior_cell_count, 1_950_750, "Every common interior pair should be classified.");
expectEqual(result.current_silent_transition_count, 404_848, "Current-contract silent transitions should remain exact.");
expectEqual(result.containment_silent_transition_count, 1_321_598, "Containment-contract silent transitions should remain exact.");
expectEqual(result.containment_additional_silent_transition_count, 511_902, "Containment visibility cost should remain exact.");
expectEqual(result.current_rounded_projection_clamp_pair_count, 99_705, "Current rounded clamp pairs should remain exact.");
expectEqual(result.containment_removed_rounded_clamp_pair_count, 199_410, "Both containment candidates should remove every current clamp pair.");
expectEqual(result.retention_cost_ranking_op_indices[0], 27, "ADD #27 should remain the largest containment visibility trade-off.");
expectEqual(result.maximum_containment_silent_ratio_increase, 0.24734987745098042, "Maximum visibility loss should remain exact.");

const top = result.residual_adds.find((row) => row.op_index === 27);
const current = top.contracts.find((contract) => contract.design === "current_artifact_contract");
const global = top.contracts.find((contract) => contract.design === "globally_finest_minimum_containment");
expectEqual(current.rounded_projection_clamp_pair_count, 14_792, "ADD #27 current clamp pairs should remain exact.");
expectEqual(current.silent_transition_count, 34_984, "ADD #27 current silent transitions should remain exact.");
expectEqual(current.both_branches_visible_cell_count, 45_974, "ADD #27 current dual-visibility cells should remain exact.");
expectEqual(current.transition_ledger_sha256, "2d21535507ee4949232e871b16e4d719b95b9eb29315f59a0c867da0e8ba110c", "ADD #27 current transition ledger should remain stable.");
expectEqual(global.rounded_projection_clamp_pair_count, 0, "Global containment should remove rounded clamps.");
expectEqual(global.silent_transition_count, 65_375, "Global containment silent transitions should remain exact.");
expectEqual(global.additional_silent_transitions_vs_current, 30_391, "Global containment visibility cost should remain exact.");
expectEqual(global.both_branches_visible_cell_count, 31_306, "Global containment dual-visibility cells should remain exact.");
expectEqual(global.neither_branch_visible_cell_count, 31_401, "Global containment flat cells should remain exact.");
expectEqual(global.transition_ledger_sha256, "8adb04e1cedbf0dff06dd0c7dad814cd4d8f3c4430633f6518f205ae8c5ac11c", "ADD #27 global transition ledger should remain stable.");

const tamperedTile = structuredClone(result);
tamperedTile.residual_adds[0].contracts[0].tile_neither_branch_visible_counts[0] += 1;
expectThrows(() => validateResidualStepResponse(tamperedTile, analysis), "tile_neither", "Validator should reject a tampered influence tile.");
const tamperedBranch = structuredClone(result);
tamperedBranch.residual_adds[0].contracts[0].branch_responses[0].silent_transition_count += 1;
expectThrows(() => validateResidualStepResponse(tamperedBranch, analysis), "silent_transition_count", "Validator should reject a tampered branch count.");
const tamperedDigest = structuredClone(result);
tamperedDigest.residual_adds[0].contracts[0].transition_ledger_sha256 = "0".repeat(64);
await expectRejects(() => validateResidualStepResponseDigests(tamperedDigest, analysis), "digest", "Validator should reject a tampered transition digest.");
expect(result.interpretation_boundary.includes("not an observed activation distribution")
  && result.interpretation_boundary.includes("does not prove a runtime branch is inactive"), "Interpretation boundary should reject distribution and branch-activity overclaims.");

const segmentationPath = "C:/Users/junhw/Downloads/main_0604_v119_4_ckpt902087_int8.tflite";
if (existsSync(segmentationPath)) {
  const segmentationBytes = new Uint8Array(readFileSync(segmentationPath));
  const segmentation = analyze_tflite_for_target(segmentationBytes, "main_0604_v119_4_ckpt902087_int8.tflite", "android_mid_a55");
  expect(validateResidualStepResponse(segmentation.residual_step_response, segmentation), "The 18-ADD regression artifact should pass structural conservation.");
  await validateResidualStepResponseDigests(segmentation.residual_step_response, segmentation);
  expectEqual(segmentation.residual_step_response.candidate_add_count, 18, "The regression artifact should expose all 18 ADD candidates.");
  expectEqual(segmentation.residual_step_response.assessed_add_count, 12, "Twelve artifact-qualified residual ADDs should be assessed.");
  expectEqual(segmentation.residual_step_response.unassessed_add_count, 6, "Six non-residual or unsupported ADDs should remain explicitly unassessed.");
  expectEqual(
    segmentation.residual_step_response.residual_adds.find((row) => row.op_index === 38)?.assessment_status,
    "assessed",
    "ADD #38 should remain a valid branch witness.",
  );
}

done(`10 sample residuals plus ${existsSync(segmentationPath) ? "the 18-ADD segmentation regression" : "an optional external segmentation regression"}, exact branch transitions, conservation, and digest tamper rejection passed.`);

async function expectRejects(action, fragment, message) {
  try {
    await action();
    expect(false, message);
  } catch (error) {
    expect(String(error?.message || error).toLowerCase().includes(fragment.toLowerCase()), message);
  }
}
