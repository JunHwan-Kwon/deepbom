import {
  applyProtectedXnnpackSelectorEvidence,
  validateProtectedXnnpackSelectorEvidence,
  XNNPACK_SELECTOR_EVIDENCE_SCHEMA,
} from "../web/lib/xnnpack-selector-evidence.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("Protected XNNPACK selector evidence check");
const profileSha = "a".repeat(64);
const sourceCommit = "23a67314f7afdbb76191589ae090d82bf55afbfa";
const gemmSha = "545f0fd6bab43186819199cf846a1f1cbb5eaa2b1df9b70e5e43431185f2b7f6";
const dwconvSha = "6ee0ceb2a953ab5cd19651ad652eadf84cd198947acfdd456e1322c787bae9fa";

const analysisFixture = () => ({
  target_profile: { id: "android_mid_a55", profile_sha256: profileSha },
  format: "tflite",
  xnnpack_selector_assessment_status: "not_loaded",
  xnnpack_selector_evidence_schema: "",
  xnnpack_selector_evidence_access: "research_authorization_required",
  insights: { misaligned_ops: 0 },
  recommendations: [],
  tensors: [
    { index: 0, dtype: "FLOAT32", shape: [1, 8, 8, 8], quant_scales: 0 },
    { index: 1, dtype: "FLOAT32", shape: [10, 1, 1, 8], quant_scales: 1 },
  ],
  ops: [{
    index: 0,
    name: "CONV_2D",
    inputs: [0, 1],
    output_channels: 10,
    xnnpack_kernel_candidate: "HEURISTIC_PROFILE",
    xnnpack_kernel_evidence_class: "HEURISTIC",
    xnnpack_kernel_candidates: [],
  }],
  roofline_csv: [
    "op_index,op_name,notes,xnnpack_kernel_candidate,xnnpack_kernel_candidate_count,xnnpack_kernel_candidate_families,xnnpack_kernel_architecture_conditions,xnnpack_kernel_compile_conditions,xnnpack_kernel_runtime_conditions,xnnpack_kernel_source_refs,xnnpack_kernel_source_file_sha256s,xnnpack_kernel_alignment_multiples,xnnpack_kernel_tile_mr,xnnpack_kernel_tile_nr,xnnpack_kernel_channel_tile,xnnpack_kernel_primary_tile,xnnpack_kernel_source,xnnpack_kernel_evidence_class,xnnpack_kernel_selector_status,xnnpack_selector_artifact_facts,xnnpack_unresolved_selector_dimensions,xnnpack_no_match_reason_code,xnnpack_candidate_tail_projections,channel_alignment_multiple,channel_alignment_status,channel_tail_overhead_percent,channel_tail_overhead_percent_min,channel_tail_overhead_percent_max",
    "0,CONV_2D,\"preserve, exactly\",HEURISTIC_PROFILE,0,,,,,,,,0,0,0,0,,HEURISTIC,not loaded,,,,,0,not-assessed,0.000000,0.000000,0.000000",
  ].join("\n"),
});

const candidate = (tileNr, suffix, outputChannels = 10) => {
  const padded = Math.ceil(outputChannels / tileNr) * tileNr;
  const inactive = padded - outputChannels;
  return ({
  family: `F32 GEMM AArch64 NEON NR${tileNr}`,
  tile_mr: 4,
  tile_nr: tileNr,
  channel_tile: tileNr,
  primary_tile: 0,
  architecture_condition: "XNN_ARCH_ARM64; Cortex-A55",
  compile_condition: "AArch64 XNNPACK build",
  runtime_condition: `runtime dispatch candidate ${suffix}`,
  source_ref: `google/XNNPACK@${sourceCommit}/src/configs/gemm-config.c#L${suffix}`,
  source_file_sha256: gemmSha,
  alignment_multiple: tileNr,
  tail_projection_status: "assessed",
  padded_output_channels: padded,
  inactive_output_channels: inactive,
  inactive_lane_ratio: inactive / padded,
  });
};

const evidenceFixture = (outputChannels = 10) => ({
  schema: XNNPACK_SELECTOR_EVIDENCE_SCHEMA,
  method_version: "2026-07-16.2",
  assessment_status: "complete",
  access_scope: "research",
  target_profile_id: "android_mid_a55",
  target_profile_sha256: profileSha,
  xnnpack_source_commit: sourceCommit,
  gemm_config_sha256: gemmSha,
  dwconv_config_sha256: dwconvSha,
  assessed_op_count: 1,
  candidate_op_count: 1,
  candidate_configuration_count: 2,
  unique_candidate_op_count: 0,
  ambiguous_candidate_op_count: 1,
  no_match_op_count: 0,
  tail_assessed_op_count: 1,
  worst_case_tail_ratio: outputChannels === 8 ? 0 : 0.375,
  worst_case_tail_op_indices: [0],
  unresolved_selector_op_count: 1,
  unresolved_selector_dimension_count: 4,
  evidence_boundary: "Pinned source candidates are enumerated; the executed runtime microkernel remains unobserved.",
  ops: [{
    op_index: 0,
    op_name: "CONV_2D",
    xnnpack_kernel_candidate: "2 protected source-enumerated XNNPACK configurations",
    xnnpack_kernel_tile_mr: 0,
    xnnpack_kernel_tile_nr: 0,
    xnnpack_kernel_channel_tile: 0,
    xnnpack_kernel_primary_tile: 0,
    xnnpack_kernel_source: `google/XNNPACK@${sourceCommit}/src/configs/gemm-config.c#L100; google/XNNPACK@${sourceCommit}/src/configs/gemm-config.c#L200`,
    xnnpack_kernel_evidence_class: "SOURCE_ENUMERATED_CANDIDATE_SET",
    xnnpack_kernel_selector_status: "Multiple protected source configurations remain; no member is claimed as executed.",
    selector_artifact_facts: {
      activation_dtype: "FLOAT32",
      kernel_area: 1,
      kernel_area_status: "assessed",
      per_channel_weights: false,
      output_channels: outputChannels,
      output_channels_status: "assessed",
    },
    unresolved_selector_dimensions: ["runtime_architecture_identity", "compile_configuration", "lowering_shape", "runtime_dispatch"],
    no_match_reason_code: "NOT_APPLICABLE_CANDIDATES_REMAIN",
    xnnpack_kernel_candidates: [candidate(4, 100, outputChannels), candidate(8, 200, outputChannels)],
    xnnpack_kernel_alignment_multiples: [4, 8],
    channel_alignment_multiple: 8,
    channel_alignment_status: "misaligned",
    channel_alignment_detail: "output channels 10: candidate tail range 16.7% to 37.5%",
    channel_tail_overhead_percent: 0.375,
    channel_tail_overhead_percent_min: 1 / 6,
    channel_tail_overhead_percent_max: 0.375,
  }],
});

const analysis = analysisFixture();
const evidence = evidenceFixture();
expectEqual(validateProtectedXnnpackSelectorEvidence(analysis, evidence), true, "A complete bound selector result should validate.");
const result = applyProtectedXnnpackSelectorEvidence(analysis, evidence);
expectEqual(result, analysis, "Selector merge should update and return the active analysis object.");
expectEqual(analysis.xnnpack_selector_assessment_status, "complete", "Protected selector merge should mark the assessment complete.");
expectEqual(analysis.xnnpack_selector_evidence_schema, XNNPACK_SELECTOR_EVIDENCE_SCHEMA, "Protected selector merge should expose its evidence schema.");
expectEqual(analysis.xnnpack_selector_evidence_access, "research", "Protected selector merge should expose the executed access scope.");
expectEqual(analysis.ops[0].xnnpack_kernel_candidates.length, 2, "Protected selector merge should retain every source-enumerated candidate.");
expectEqual(analysis.ops[0].channel_tail_overhead_percent, 0.375, "Protected selector merge should retain the verified worst-case tail projection.");
expectEqual(analysis.insights.misaligned_ops, 1, "Protected selector merge should reconcile the root misalignment count.");
expectEqual(analysis.recommendations.filter((item) => String(item.title || "").includes("channel tail not aligned")).length, 1, "Protected selector merge should reconcile the channel-tail action queue.");
expect(analysis.roofline_csv.includes("SOURCE_ENUMERATED_CANDIDATE_SET"), "Roofline CSV should receive the verified selector evidence class.");
expect(analysis.roofline_csv.includes("F32 GEMM AArch64 NEON NR4 || F32 GEMM AArch64 NEON NR8"), "Roofline CSV should enumerate candidate families.");
expect(analysis.roofline_csv.includes("activation_dtype=FLOAT32; kernel_area=1"), "Roofline CSV should expose independently verified selector facts.");
expect(analysis.roofline_csv.includes("C10->16; inactive=6; ratio=0.375000; align=8; assessed"), "Roofline CSV should expose reproducible candidate tail arithmetic.");
expect(analysis.roofline_csv.includes('"preserve, exactly"'), "Roofline CSV should preserve unrelated raw quoted cells.");
expectEqual(analysis.xnnpack_selector_evidence_provenance.candidate_configuration_count, 2, "Merged provenance should preserve verified summary counts.");

const semanticOutputAnalysis = analysisFixture();
semanticOutputAnalysis.ops[0].channel_alignment_status = "graph-output-contract";
semanticOutputAnalysis.ops[0].channel_alignment_detail = "The class axis reaches the serialized graph output; generic channel padding is suppressed.";
applyProtectedXnnpackSelectorEvidence(semanticOutputAnalysis, evidenceFixture());
expectEqual(semanticOutputAnalysis.ops[0].channel_alignment_status, "graph-output-contract", "Protected selector evidence must not replace a graph-output semantic-axis exception with generic padding advice.");
expectEqual(semanticOutputAnalysis.ops[0].channel_tail_overhead_percent, 0.375, "A semantic-axis exception should retain the source-backed physical tail projection.");
expectEqual(semanticOutputAnalysis.insights.misaligned_ops, 0, "A graph-output semantic axis must not count as an actionable internal-channel misalignment.");
expectEqual(semanticOutputAnalysis.recommendations.length, 0, "A graph-output semantic axis must not receive a generic channel-padding action.");
expect(semanticOutputAnalysis.roofline_csv.includes("graph-output-contract"), "Roofline CSV should preserve the graph-output semantic-axis status after protected evidence merge.");

const tiedAnalysis = analysisFixture();
tiedAnalysis.ops[0].output_channels = 8;
const tiedEvidence = evidenceFixture(8);
Object.assign(tiedEvidence.ops[0], {
  channel_alignment_multiple: 8,
  channel_alignment_status: "aligned",
  channel_tail_overhead_percent: 0,
  channel_tail_overhead_percent_min: 0,
  channel_tail_overhead_percent_max: 0,
});
expectEqual(validateProtectedXnnpackSelectorEvidence(tiedAnalysis, tiedEvidence), true, "Equal tail projections should deterministically select the largest candidate multiple.");

const tamper = (mutate) => {
  const currentAnalysis = analysisFixture();
  const currentEvidence = evidenceFixture();
  mutate(currentEvidence, currentAnalysis);
  return { currentAnalysis, currentEvidence };
};

for (const [label, message, mutate] of [
  ["target-profile binding", "target-profile SHA-256", (value) => { value.target_profile_sha256 = "e".repeat(64); }],
  ["method binding", "method version", (value) => { value.method_version = "2026-07-16.1"; }],
  ["source commit binding", "source commit", (value) => { value.xnnpack_source_commit = "e".repeat(40); }],
  ["source-file binding", "source-file hash", (value) => { value.ops[0].xnnpack_kernel_candidates[0].source_file_sha256 = dwconvSha; }],
  ["summary count", "summary counts", (value) => { value.candidate_configuration_count = 3; }],
  ["worst-tail ledger", "worst-tail ledger", (value) => { value.worst_case_tail_ratio = 0.5; }],
  ["op identity", "op identity mismatch", (value) => { value.ops[0].op_name = "FULLY_CONNECTED"; }],
  ["artifact facts", "artifact facts", (value) => { value.ops[0].selector_artifact_facts.kernel_area = 9; }],
  ["unresolved dimensions", "unresolved dimensions", (value) => { value.ops[0].unresolved_selector_dimensions.pop(); }],
  ["alignment set", "alignment set", (value) => { value.ops[0].xnnpack_kernel_alignment_multiples = [8]; }],
  ["candidate tail arithmetic", "candidate tail arithmetic", (value) => { value.ops[0].xnnpack_kernel_candidates[1].inactive_output_channels = 5; }],
  ["tail projection", "tail projection", (value) => { value.ops[0].channel_tail_overhead_percent_max = 0.5; }],
]) {
  const { currentAnalysis, currentEvidence } = tamper(mutate);
  expectThrows(() => applyProtectedXnnpackSelectorEvidence(currentAnalysis, currentEvidence), message, `${label} tampering should be rejected.`);
  expectEqual(currentAnalysis.xnnpack_selector_assessment_status, "not_loaded", `${label} rejection must not partially mutate the analysis.`);
  expectEqual(currentAnalysis.ops[0].xnnpack_kernel_evidence_class, "HEURISTIC", `${label} rejection must preserve the public op evidence.`);
}

const invalidCsvAnalysis = analysisFixture();
invalidCsvAnalysis.roofline_csv = "op_index,op_name\n0,CONV_2D";
expectThrows(() => applyProtectedXnnpackSelectorEvidence(invalidCsvAnalysis, evidenceFixture()), "does not contain selector evidence columns", "Malformed Roofline CSV should reject the entire merge.");
expectEqual(invalidCsvAnalysis.xnnpack_selector_assessment_status, "not_loaded", "Roofline CSV rejection must not change root selector status.");
expectEqual(invalidCsvAnalysis.ops[0].xnnpack_kernel_evidence_class, "HEURISTIC", "Roofline CSV rejection must not partially replace op evidence.");

done("Protected XNNPACK selector evidence check passed (binding, arithmetic, atomic merge, and CSV preservation). ");
