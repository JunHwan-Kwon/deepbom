import { readFileSync } from "node:fs";

import {
  analyze_tflite_for_target,
  compute_deployment_delta,
  initSync,
} from "../pkg/tflite_wasm_audit.js";
import {
  DEPLOYMENT_DELTA_SCHEMA,
  validateDeploymentDelta,
} from "../web/lib/deployment-delta.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { buildStaticAnalysisExport } from "../web/lib/report-evidence.js";
import { buildPublicShareAnalysis } from "../web/lib/public-export.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("Deployment delta check");
const targetIds = ["android_mid_a55", "rpi4_a72", "x86_avx2", "wasm_simd"];
const baselineFilename = "mobilenet_v1_025_224_float.tflite";
const candidateFilename = "mobilenet_v2_1.0_224_quant.tflite";

initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const baselineBytes = new Uint8Array(readFileSync(`web/samples/${baselineFilename}`));
const candidateBytes = new Uint8Array(readFileSync(`web/samples/${candidateFilename}`));
const candidate = analyze_tflite_for_target(candidateBytes, candidateFilename, targetIds[0]);
const delta = compute_deployment_delta(
  baselineBytes,
  baselineFilename,
  candidateBytes,
  candidateFilename,
  JSON.stringify(targetIds),
);
candidate.model_sha256 = delta.candidate.sha256;
candidate.deployment_delta = delta;

expect(validateDeploymentDelta(delta, candidate), "Actual two-artifact delta should satisfy every browser invariant.");
expectEqual(delta.schema, DEPLOYMENT_DELTA_SCHEMA, "Deployment delta schema should be stable.");
expectEqual(delta.target_count, 4, "Deployment delta should cover all public planning targets.");
expectEqual(delta.alignment.matched_op_count + delta.alignment.removed_op_count, delta.baseline.operator_count, "Every baseline op should be accounted for exactly once.");
expectEqual(delta.alignment.matched_op_count + delta.alignment.added_op_count, delta.candidate.operator_count, "Every candidate op should be accounted for exactly once.");
expectEqual(delta.alignment.artifact_relation, "different_artifacts_lineage_unproven", "Different bytes must not imply model lineage.");
expect(delta.target_deltas.every((target) => Math.abs(target.conservation_error_us) <= 1e-8), "Every target should conserve the signed driver ledger.");
expect(delta.cross_target_drivers.some((driver) => driver.consistent_regression), "Actual fixtures should expose at least one cross-target-consistent modeled regression.");
expectEqual(
  JSON.stringify(compute_deployment_delta(baselineBytes, baselineFilename, candidateBytes, candidateFilename, JSON.stringify(targetIds))),
  JSON.stringify(delta),
  "Deployment delta JSON should be byte-for-byte deterministic.",
);

const identical = compute_deployment_delta(
  candidateBytes,
  candidateFilename,
  candidateBytes,
  candidateFilename,
  JSON.stringify(targetIds),
);
expectEqual(identical.alignment.artifact_relation, "identical_bytes", "Identical bytes should be identified explicitly.");
expectEqual(identical.alignment.exact_structural_match_count, candidate.ops.length, "Identical artifacts should align every op by exact structural signature.");
expectEqual(identical.alignment.added_op_count, 0, "Identical artifacts should have no added ops.");
expectEqual(identical.alignment.removed_op_count, 0, "Identical artifacts should have no removed ops.");
expect(identical.target_deltas.every((target) => target.signed_delta_us === 0 && target.drivers.every((driver) => driver.signed_delta_us === 0)), "Identical artifacts should produce exact zero deltas.");
expect(validateDeploymentDelta(identical, candidate), "Identical-artifact zero delta should satisfy every invariant.");

const tampered = structuredClone(delta);
tampered.target_deltas[0].drivers[0].signed_delta_us += 1;
expectThrows(() => validateDeploymentDelta(tampered, candidate), "signed delta invariant failed", "Browser validation should reject a tampered driver delta.");
const missingEntity = structuredClone(delta);
missingEntity.alignment_rows.pop();
expectThrows(() => validateDeploymentDelta(missingEntity, candidate), "op coverage is invalid", "Browser validation should reject an incomplete alignment ledger.");
const tamperedCandidateBinding = structuredClone(delta);
tamperedCandidateBinding.alignment_rows.find((row) => row.candidate_op_index != null).candidate_op_name = "RESHAPE";
expectThrows(() => validateDeploymentDelta(tamperedCandidateBinding, candidate), "candidate op binding is invalid", "Browser validation should reject a candidate-op identity substitution.");
const tamperedRank = structuredClone(delta);
tamperedRank.target_deltas[0].drivers.find((driver) => driver.baseline_rank != null).baseline_rank += 1;
expectThrows(() => validateDeploymentDelta(tamperedRank, candidate), "baseline rank is invalid", "Browser validation should reject a tampered target rank.");
const tamperedComponent = structuredClone(delta);
tamperedComponent.target_deltas[0].drivers[0].candidate_components.compute_us += 1;
expectThrows(() => validateDeploymentDelta(tamperedComponent, candidate), "candidate component binding invariant failed", "Browser validation should reject a tampered per-driver component ledger.");

const identity = {
  filename: candidate.filename,
  format: candidate.format,
  sha256: candidate.model_sha256,
  target_label: candidate.target_profile.label,
  operator_count: candidate.operator_count,
  tensor_count: candidate.tensor_count,
  total_macs: candidate.total_macs,
};
const report = buildEngineeringReport(candidate, { identity });
expect(report.includes("## Deployment Delta (DERIVED/ESTIMATED)"), "Engineering report should render the deployment delta section.");
expect(report.includes("Maximum independent target-ledger conservation error"), "Engineering report should disclose the conservation result.");
const staticExport = buildStaticAnalysisExport(candidate);
expectEqual(staticExport.deployment_delta.schema, DEPLOYMENT_DELTA_SCHEMA, "Static evidence should preserve the complete delta ledger.");

const publicAnalysis = buildPublicShareAnalysis(candidate);
expectEqual(publicAnalysis.deployment_delta.baseline.filename, "ARTIFACT-BASELINE", "Public export should redact the baseline filename.");
expectEqual(publicAnalysis.deployment_delta.candidate.filename, "ARTIFACT-001", "Public export should redact the candidate filename.");
expectEqual(publicAnalysis.deployment_delta.baseline.sha256, "", "Public export should redact the baseline hash.");
expectEqual(publicAnalysis.deployment_delta.candidate.sha256, "", "Public export should redact the candidate hash.");
expect(publicAnalysis.deployment_delta.target_deltas.every((target) => target.target_id.startsWith("PUBLIC-PLANNING-TARGET-")), "Public export should redact every delta target ID.");

const worst = delta.target_deltas.find((target) => target.target_id === delta.worst_relative_delta_target_id);
done(`Deployment delta passed (${delta.alignment.matched_op_count} matched, ${delta.alignment.added_op_count} added, ${delta.alignment.removed_op_count} removed; worst ${worst.target_id} ${((worst.relative_delta || 0) * 100).toFixed(1)}%; exact target-ledger conservation).`);
