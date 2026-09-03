import { artifactIrOperators, artifactIrValues } from "./artifact-ir-selectors.js";
import {
  contractDynamicDimSummary,
  modelQuantizationStatus,
  quantizationScopeExplanation,
  xnnpackLabel,
} from "./analysis.js";
import { deriveTfliteBatchOneProjection } from "./dynamic-shape-cost.js";
import {
  ANALYZER_METADATA,
  RUNTIME_COMPATIBILITY_EVIDENCE_LABEL,
  buildAnalyzerContentManifest,
  buildAnalyzerMetadata,
} from "./report-metadata.js";
import { buildChangeAnalysis, buildEngineeringReport, buildEngineeringReportArtifacts } from "./report-engineering.js";
import { assertConformance, buildConformanceReport } from "./report-conformance.js";
import { buildFindingsRegister, normalizeNativeAnalyzerFindings } from "./report-findings.js";
import { buildQuantizationContractChecks } from "./report-quantization-contracts.js";
import { csvCell, jsonForDownload, zipBinaryFile, zipTextFile } from "./report-utils.js";
import { buildStaticAuditMarkdown } from "./markdown-report.js";
import { buildInputWitnessTensor } from "./input-counterexample.js";
import { buildCandidateRgbFixture } from "./preprocessing-realizability.js";
import { buildMetricCoverageEntries, buildMetricCoverageManifest } from "./metric-coverage.js";
import { buildQuantResearchCoverage } from "./quant-research-applicability.js";
import { BROWSER_BENCHMARK_TIMING_METHOD } from "./runtime.js";
import { buildExecutionPlacementEvidence } from "./execution-placement-evidence.js";
import { buildRuntimeBackendEvidenceLedger } from "./runtime-backend-evidence-ledger.js";
import { buildRuntimeDataMovementEvidence } from "./runtime-data-movement-evidence.js";
import { buildRuntimeEvidenceSidecar } from "./runtime-evidence-sidecar.js";
import { buildOnnxRuntimeShapeBinding } from "./onnx-runtime-shape-binding.js";
import { buildSecurityPostureEvidence, collectRuntimeWarnings } from "./report-security-posture.js";
import { resolveArtifactIrContext } from "./artifact-ir-context.js";

export function buildStaticAnalysisExport(analysis) {
  const { _markdown, roofline_csv, core_isolation_csv, stage_mermaid, findings: _nativeFindings, recommendations: _nativeRecommendations, ...rest } = analysis || {};
  const format = String(analysis?.format || "tflite").toLowerCase();
  const tflite = format === "tflite";
  const onnx = format === "onnx";
  const quantResearchCoverage = tflite ? buildQuantResearchCoverage(analysis) : null;
  const researchEvidence = (key, normalize) => quantResearchCoverage?.labs.find((row) => row.evidence_key === key)?.class_supported
    ? normalize(rest[key])
    : null;
  const targetProfile = onnx && rest.target_profile
    ? {
        id: rest.target_profile.id || "",
        label: rest.target_profile.label || "",
        profile_sha256: rest.target_profile.profile_sha256 || "",
        l1_data_bytes: Number(rest.target_profile.l1_data_bytes || 0),
        applicability: "Static L1 working-set ratio reference only; TFLite delegation, packing, channel-tail, throughput, bandwidth, and ridge fields are not applicable.",
      }
    : rest.target_profile;
  const predictedPartitionBoundaries = tflite ? normalizePredictedPartitionBoundaries(rest.predicted_partition_boundaries) : rest.predicted_partition_boundaries ?? null;
  const tensorArenaPlan = tflite ? normalizeTensorArenaPlan(rest.tensor_arena_plan) : null;
  return {
    schema: ANALYZER_METADATA.schemas.staticAnalysis,
    ...rest,
    quant_research_coverage: quantResearchCoverage,
    target_profile: targetProfile,
    predicted_partition_boundaries: predictedPartitionBoundaries,
    tensor_liveness: normalizeTensorLiveness(rest.tensor_liveness),
    tensor_arena_plan: tensorArenaPlan,
    accumulator_atlas: tflite ? researchEvidence("accumulator_atlas", normalizeAccumulatorAtlas) : null,
    requantization_fidelity: tflite ? researchEvidence("requantization_fidelity", normalizeRequantizationFidelity) : null,
    kernel_extremum_witness: tflite ? researchEvidence("kernel_extremum_witness", normalizeKernelExtremumWitness) : null,
    channel_vitality: tflite ? researchEvidence("channel_vitality", normalizeChannelVitality) : null,
    rounding_equivalence: tflite ? researchEvidence("rounding_equivalence", normalizeRoundingEquivalence) : null,
    accumulator_reachability: tflite ? researchEvidence("accumulator_reachability", normalizeAccumulatorReachability) : null,
    numerical_abi_propagation: tflite ? researchEvidence("numerical_abi_propagation", normalizeNumericalAbiPropagation) : null,
    input_counterexample: tflite ? researchEvidence("input_counterexample", normalizeInputCounterexample) : null,
    preprocessing_realizability: tflite ? researchEvidence("preprocessing_realizability", normalizePreprocessingRealizability) : null,
    quantization_lattice: tflite ? researchEvidence("quantization_lattice", normalizeQuantizationLattice) : null,
    contract_migration: tflite ? researchEvidence("contract_migration", normalizeContractMigration) : null,
    residual_step_response: tflite ? researchEvidence("residual_step_response", normalizeResidualStepResponse) : null,
    residual_contract_distortion: tflite ? researchEvidence("residual_contract_distortion", normalizeResidualContractDistortion) : null,
    analyzer_metadata: buildAnalyzerMetadata(analysis),
    export_notes: {
      target_dependent: tflite
        ? "Roofline, L1, packing, predicted partition-break severity, and INT8 speedup are computed for the selected target profile."
        : "No TFLite target-dependent roofline, packing, partition-break severity, or INT8 speedup claim is applied to this format.",
      byte_units: "Numeric byte fields are raw bytes. UI compact display uses binary KiB/MiB labels.",
      static_cost_model: tflite
        ? "Fields named bottleneck_*_us and weight_packing_overhead_us are HEURISTIC formula outputs under the illustrative target planning profile, not measured target-device latency. Per-op base=max(arithmetic_ops/full_configured_effective_peak, logical_bytes/effective_bandwidth); steady_state=base+fallback_traffic; cold_start=steady_state+one_time_packing+partition_planning_setup. The partition-planning setup term is an unmeasured profile constant independent of logical boundary payload, not per-inference copy or launch latency. No separate kernel-utilization discount is applied. Report-facing summaries preserve both operating points and include a sum invariant."
        : "The TFLite CPU target cost model is not applied to this format. Format-specific symbolic costs and memory contracts remain separately labeled where available.",
      deployment_frontier: onnx
        ? "The ORT EP frontier preserves the DERIVED pinned source schema/kernel-version intersection and a separately narrowed candidate set after machine-evaluated artifact-visible type, rank, constant-input, output-count, and explicit-attribute definite exclusions. A remaining candidate is not support, GetCapability assignment, or runtime placement evidence."
        : tflite
          ? "The TFLite frontier assigns max(compute, memory) once to the active roofline-bound component, keeps one-time packing and partition-planning setup on the cold-start axis, and derives deterministic 80% steady-state hotspot prefixes and normalized Jensen-Shannon distance. Pair distance is exactly conserved across non-negative per-op attribution terms; packing/setup counterfactuals use cold totals while fallback counterfactuals use steady totals."
          : "No TFLite or ORT deployment frontier is inferred for this serialized format.",
      deployment_delta: tflite
        ? "When a distinct in-memory baseline is bound, both TFLite artifacts are independently parsed across pinned planning profiles. Each op assigns max(compute, memory) once to the active roofline-bound component. Deterministic sequence alignment is a comparison coordinate, not semantic identity or lineage; signed static component deltas are not device measurements."
        : "Not applicable in deployment-delta v1.",
      delegation_repair: tflite
        ? "Every graph op is toggled exactly once between the predicted delegate and CPU domains while all other assignments remain fixed. Each maximal contiguous predicted-CPU run is also toggled as one complete, explicitly defined island intervention while all outside assignments remain fixed. Segments and producer-to-consumer boundary edges are rebuilt deterministically; group-only gain is the exact full-island reduction beyond the best beneficial single member. These are static repair hypotheses, not proof of delegate support, minimal implementation effort, numerical equivalence, runtime copy materialization, or latency benefit."
        : "Not applicable in delegation-repair v1.",
      partition_boundary_edges: tflite
        ? "Internal producer-to-consumer edges are DERIVED from graph connectivity under the PREDICTED static execution-domain assignment. Logical payload bytes do not establish runtime copy materialization or latency."
        : "No TFLite predicted partition-boundary inventory is applied to this format.",
      tensor_arena_plan: "TFLite only. DERIVED from declared tensor shapes with the pinned TensorFlow ArenaPlanner/SimpleMemoryArena algorithm and builtin in-place registration flags. This is not observed runtime memory and excludes Prepare-time resize, node temporaries, scratch, delegates, and custom allocators.",
      accumulator_atlas: tflite
        ? "Every assessable 8-bit Conv/Depthwise Conv/rank-2 Fully Connected output channel is exactly bounded from stored centered weights, legal input codes, and stored INT32 bias under the pinned TFLite reference accumulation algebra. Decimal strings preserve integer evidence beyond JavaScript safe-integer range; this is not executed delegate evidence."
        : "Not applicable in accumulator-atlas v1.",
      requantization_fidelity: tflite
        ? "Every accumulator-bounded channel derives the artifact effective scale, pinned Q0.31 multiplier/shift encodings, encoding drift over the exact post-bias accumulator domain consumed by requantization, and separate default-double/single-rounding conservative bounds. The full envelope remains the intermediate INT32-safety domain; the runtime compile flag and executed backend remain unresolved without runtime evidence."
        : "Not applicable in requantization-fidelity v1.",
      kernel_extremum_witness: tflite
        ? "Every assessable channel receives canonical minimum and maximum full-valid receptive-field input-code witnesses derived from stored centered-weight signs. The witnesses are proven against exact accumulator endpoints and projected through both pinned default-double and TFLITE_SINGLE_ROUNDING paths. They are per-channel synthetic legal-code constructions, not full-model inputs or observed runtime activations."
        : "Not applicable in kernel-extremum-witness v1.",
      channel_vitality: tflite
        ? "Every exact stored-weight post-bias interval is projected through both pinned TFLite fixed-point build paths. Equal monotone endpoint output codes prove a constant channel over the full-valid receptive-field legal-code domain; larger spans are interval-hull upper bounds, not exact reachable-code counts or observed activation coverage."
        : "Not applicable in channel-vitality v1.",
      rounding_equivalence: tflite
        ? "Every integer in each exact post-bias int32 interval hull is partitioned into maximal runs with one ordered default/single-rounding output-code pair. State counts and counterexamples are exact for the interval hull, but interior accumulator integers can be unreachable and the result is not an observed activation frequency or task-accuracy estimate."
        : "Not applicable in rounding-equivalence v1.",
      accumulator_reachability: tflite
        ? "Stored centered weights are compressed into bounded absolute-weight denominations. GCD residue exclusion, sorted bounded-coverage extension, symmetric endpoint bands, and their intersection with exact rounding-pair segments partition every interval and divergent state into certified kernel-local reachable, provably incompatible, or unresolved classes. This does not establish full-model-input reachability or runtime frequency."
        : "Not applicable in accumulator-reachability v1.",
      numerical_abi_propagation: tflite
        ? "Every build-mode-divergent source op is joined to its accumulator-reachability partition and the complete producer/tensor/consumer graph. Exact kernel-local counterexamples, residue exclusions, unresolved states, downstream corridors, merge classes, route multiplicity, and predicted execution-domain crossings are separately preserved. A local counterexample does not prove full-model-input realization or a declared-output difference."
        : "Not applicable in numerical-abi-propagation v1.1.",
      input_counterexample: tflite
        ? "Every exact-local divergent source is classified by input origin. A direct quantized model-input Conv/Depthwise source is promoted only after a complete zero-point-filled input tensor, sparse override ledger, full-valid receptive field, exact dot-plus-bias reproduction, divergent pinned output codes, and whole-tensor SHA-256 are constructed. This proves tensor-ABI existence, not preprocessing-domain realizability or declared-output divergence."
        : "Not applicable in input-counterexample v1.",
      preprocessing_realizability: tflite
        ? "Each eligible constructive NHWC RGB input witness is evaluated under eight explicit preprocessing counterfactuals. Every source pixel code is exhaustively mapped; direct-storage rows bypass source quantization and normalized rows use exact rational half-away quantization. Code coverage, holes, collisions, inverse pixels, complete-tensor minimum error, and RGB fixture SHA-256 are derived. Candidate rows do not identify the production preprocessing pipeline."
        : "Not applicable in preprocessing-realizability v1.",
      contract_migration: tflite
        ? "For both exhaustive residual containment candidates, every direct consumer is traced. Pinned TensorFlow Conv/DW/FC and ADD parameters are regenerated; stored INT32 bias is independently rebased channel by channel, while complete downstream reachability is kept separate as a structural behavior radius rather than a metadata-change claim."
        : "Not applicable in contract-migration v1.",
      residual_step_response: tflite
        ? "For the current and both containment output contracts, every legal adjacent +1 input-code transition is projected exactly while holding the other branch fixed. Silent/visible transitions, joint branch influence classes, clamp association, reproduction error, tiles, and SHA-256 ledgers describe uniform legal-code geometry rather than observed activations or task accuracy."
        : "Not applicable in residual-step-response v1.",
      residual_contract_distortion: tflite
        ? "Both minimum-containment candidates are compared with the artifact output contract over every legal residual input-code pair. Clamp-state transitions, dequantized represented-value displacement, ideal-projection error direction, sign-class changes, quantiles, spatial tiles, worst witnesses, and SHA-256 pair ledgers describe a uniform legal-code counterfactual rather than an activation-weighted quality result."
        : "Not applicable in residual-contract-distortion v1.",
    },
    quantization_scope: tflite || onnx ? quantizationScopeExplanation(analysis) : null,
  };
}

function normalizeTensorLiveness(liveness) {
  if (!liveness) return null;
  return {
    ...liveness,
    peak_bytes: liveness.peak_bytes ?? null,
    peak_at_op: liveness.peak_at_op ?? null,
    peak_at_op_name: liveness.peak_at_op_name ?? null,
  };
}

function normalizeTensorArenaPlan(plan) {
  if (!plan) return null;
  return {
    ...plan,
    non_persistent_arena_bytes: plan.non_persistent_arena_bytes ?? null,
    persistent_arena_bytes: plan.persistent_arena_bytes ?? null,
    combined_arena_bytes: plan.combined_arena_bytes ?? null,
    allocations: (plan.allocations || []).map((allocation) => ({
      ...allocation,
      size_bytes: allocation.size_bytes ?? null,
      offset_bytes: allocation.offset_bytes ?? null,
      last_node: allocation.last_node ?? null,
      shared_with_tensor_index: allocation.shared_with_tensor_index ?? null,
    })),
  };
}

function normalizePredictedPartitionBoundaries(inventory) {
  if (!inventory) return null;
  return {
    ...inventory,
    summed_edge_payload_bytes: inventory.summed_edge_payload_bytes ?? null,
    unique_tensor_payload_bytes: inventory.unique_tensor_payload_bytes ?? null,
    edges: (inventory.edges || []).map((edge) => ({ ...edge, payload_bytes: edge.payload_bytes ?? null })),
  };
}

const LATTICE_OPTIONAL_ROW_FIELDS = [
  "output_tensor_index", "output_scale", "output_zero_point", "output_code_range",
  "legal_sum_real_range", "output_real_range", "continuous_sum_interval_coverage_ratio",
  "input_scale_ratio", "output_to_finest_input_step_ratio", "output_to_coarsest_input_step_ratio",
  "enumerated_code_pair_count", "range_escape_low_pair_count", "range_escape_high_pair_count",
  "range_escape_pair_count", "range_escape_pair_ratio", "rounded_projection_clamp_pair_count",
  "rounded_projection_clamp_pair_ratio", "complete_legal_domain_contained",
  "distinct_projected_output_code_count", "projected_output_code_utilization_ratio",
  "mean_in_range_rounding_error", "mean_in_range_rounding_error_steps",
  "maximum_in_range_rounding_error", "maximum_in_range_rounding_error_steps",
  "mean_clamped_projection_error", "mean_clamped_projection_error_steps",
  "maximum_clamped_projection_error", "maximum_clamped_projection_error_steps",
  "worst_projection_pair", "containment_candidate_count", "fixed_zero_point_containment",
  "globally_finest_containment", "domain_escape_rank",
];

function normalizeQuantizationLattice(lattice) {
  if (!lattice) return null;
  const normalized = {
    ...lattice,
    maximum_range_escape_pair_ratio: lattice.maximum_range_escape_pair_ratio ?? null,
    maximum_mean_clamped_projection_error_steps: lattice.maximum_mean_clamped_projection_error_steps ?? null,
    maximum_fixed_zero_point_scale_ratio: lattice.maximum_fixed_zero_point_scale_ratio ?? null,
    maximum_global_finest_scale_ratio: lattice.maximum_global_finest_scale_ratio ?? null,
    residual_adds: (lattice.residual_adds || []).map((row) => ({ ...row })),
  };
  for (const row of normalized.residual_adds) {
    for (const field of LATTICE_OPTIONAL_ROW_FIELDS) row[field] = row[field] ?? null;
    row.containment_frontier = row.containment_frontier || [];
  }
  return normalized;
}

function normalizeAccumulatorAtlas(atlas) {
  if (!atlas) return null;
  const optionalTopLevel = [
    "maximum_absolute_accumulator_decimal", "maximum_int32_ratio",
    "maximum_required_signed_bits", "minimum_int32_headroom_bits",
  ];
  const optionalRowFields = [
    "input_tensor_index", "weight_tensor_index", "bias_tensor_index", "input_code_range",
    "input_zero_point", "output_channel_axis", "output_channel_count",
    "accumulation_terms_per_channel", "maximum_absolute_accumulator_decimal",
    "maximum_int32_ratio", "maximum_required_signed_bits", "minimum_int32_headroom_bits",
    "metadata_only_magnitude_bound_decimal", "metadata_only_int32_ratio",
    "exact_tightening_factor", "worst_channel",
  ];
  const normalized = { ...atlas, ops: (atlas.ops || []).map((row) => ({ ...row })) };
  for (const field of optionalTopLevel) normalized[field] = normalized[field] ?? null;
  for (const row of normalized.ops) {
    for (const field of optionalRowFields) row[field] = row[field] ?? null;
  }
  return normalized;
}

function normalizeRequantizationFidelity(fidelity) {
  if (!fidelity) return null;
  const optionalTopLevel = [
    "minimum_shift", "maximum_shift", "maximum_relative_multiplier_error",
    "maximum_multiplier_error_ppm", "maximum_encoding_drift_bound_codes",
    "maximum_default_double_rounding_bound_codes", "maximum_single_rounding_bound_codes",
  ];
  const optionalRowFields = [
    "input_tensor_index", "weight_tensor_index", "output_tensor_index", "input_scale", "output_scale",
    "output_zero_point", "output_code_range", "minimum_shift", "maximum_shift",
    "maximum_relative_multiplier_error", "maximum_multiplier_error_ppm",
    "maximum_encoding_drift_bound_codes", "maximum_default_double_rounding_bound_codes",
    "maximum_single_rounding_bound_codes", "worst_channel",
  ];
  const normalized = { ...fidelity, ops: (fidelity.ops || []).map((row) => ({ ...row })) };
  for (const field of optionalTopLevel) normalized[field] = normalized[field] ?? null;
  for (const row of normalized.ops) {
    for (const field of optionalRowFields) row[field] = row[field] ?? null;
    row.channel_default_double_rounding_bound_codes = (row.channel_default_double_rounding_bound_codes || []).map((value) => value ?? null);
    row.channel_single_rounding_bound_codes = (row.channel_single_rounding_bound_codes || []).map((value) => value ?? null);
  }
  return normalized;
}

function normalizeKernelExtremumWitness(witness) {
  if (!witness) return null;
  const normalizeEndpoint = (endpoint) => endpoint ? {
    ...endpoint,
    ideal_preclamp_code: endpoint.ideal_preclamp_code ?? null,
    ideal_output_code: endpoint.ideal_output_code ?? null,
    default_scaled_accumulator: endpoint.default_scaled_accumulator ?? null,
    default_preclamp_code: endpoint.default_preclamp_code ?? null,
    default_output_code: endpoint.default_output_code ?? null,
    default_activation_clamped: endpoint.default_activation_clamped ?? null,
    default_ideal_delta_codes: endpoint.default_ideal_delta_codes ?? null,
    single_scaled_accumulator: endpoint.single_scaled_accumulator ?? null,
    single_preclamp_code: endpoint.single_preclamp_code ?? null,
    single_output_code: endpoint.single_output_code ?? null,
    single_activation_clamped: endpoint.single_activation_clamped ?? null,
    single_ideal_delta_codes: endpoint.single_ideal_delta_codes ?? null,
    build_mode_output_delta_codes: endpoint.build_mode_output_delta_codes ?? null,
  } : null;
  const normalizeChannel = (channel) => channel ? {
    ...channel,
    maximum_default_ideal_delta_codes: channel.maximum_default_ideal_delta_codes ?? null,
    maximum_single_ideal_delta_codes: channel.maximum_single_ideal_delta_codes ?? null,
    minimum: normalizeEndpoint(channel.minimum),
    maximum: normalizeEndpoint(channel.maximum),
  } : null;
  return {
    ...witness,
    maximum_default_ideal_delta_codes: witness.maximum_default_ideal_delta_codes ?? null,
    maximum_single_ideal_delta_codes: witness.maximum_single_ideal_delta_codes ?? null,
    ops: (witness.ops || []).map((row) => ({
      ...row,
      input_tensor_index: row.input_tensor_index ?? null,
      weight_tensor_index: row.weight_tensor_index ?? null,
      output_tensor_index: row.output_tensor_index ?? null,
      input_code_range: row.input_code_range ?? null,
      input_zero_point: row.input_zero_point ?? null,
      output_channel_axis: row.output_channel_axis ?? null,
      output_scale: row.output_scale ?? null,
      output_zero_point: row.output_zero_point ?? null,
      output_code_range: row.output_code_range ?? null,
      activation_code_range: row.activation_code_range ?? null,
      accumulation_terms_per_channel: row.accumulation_terms_per_channel ?? null,
      maximum_default_ideal_delta_codes: row.maximum_default_ideal_delta_codes ?? null,
      maximum_single_ideal_delta_codes: row.maximum_single_ideal_delta_codes ?? null,
      top_channels: (row.top_channels || []).map(normalizeChannel),
      worst_channel: normalizeChannel(row.worst_channel),
    })),
  };
}

function normalizeChannelVitality(vitality) {
  if (!vitality) return null;
  const normalizeTopChannel = (channel) => ({
    ...channel,
    default_minimum_preclamp_code: channel.default_minimum_preclamp_code ?? null,
    default_maximum_preclamp_code: channel.default_maximum_preclamp_code ?? null,
    default_minimum_output_code: channel.default_minimum_output_code ?? null,
    default_maximum_output_code: channel.default_maximum_output_code ?? null,
    default_inclusive_code_span: channel.default_inclusive_code_span ?? null,
    single_minimum_preclamp_code: channel.single_minimum_preclamp_code ?? null,
    single_maximum_preclamp_code: channel.single_maximum_preclamp_code ?? null,
    single_minimum_output_code: channel.single_minimum_output_code ?? null,
    single_maximum_output_code: channel.single_maximum_output_code ?? null,
    single_inclusive_code_span: channel.single_inclusive_code_span ?? null,
  });
  return {
    ...vitality,
    minimum_default_inclusive_code_span: vitality.minimum_default_inclusive_code_span ?? null,
    minimum_single_inclusive_code_span: vitality.minimum_single_inclusive_code_span ?? null,
    ops: (vitality.ops || []).map((row) => ({
      ...row,
      output_code_range: row.output_code_range ?? null,
      activation_code_range: row.activation_code_range ?? null,
      minimum_default_inclusive_code_span: row.minimum_default_inclusive_code_span ?? null,
      minimum_single_inclusive_code_span: row.minimum_single_inclusive_code_span ?? null,
      default_minimum_output_codes: (row.default_minimum_output_codes || []).map((value) => value ?? null),
      default_maximum_output_codes: (row.default_maximum_output_codes || []).map((value) => value ?? null),
      single_minimum_output_codes: (row.single_minimum_output_codes || []).map((value) => value ?? null),
      single_maximum_output_codes: (row.single_maximum_output_codes || []).map((value) => value ?? null),
      top_channels: (row.top_channels || []).map(normalizeTopChannel),
    })),
  };
}

function normalizeRoundingEquivalence(equivalence) {
  if (!equivalence) return null;
  const normalizeTopChannel = (channel) => ({
    ...channel,
    first_divergent_accumulator_decimal: channel.first_divergent_accumulator_decimal ?? null,
    first_default_output_code: channel.first_default_output_code ?? null,
    first_single_output_code: channel.first_single_output_code ?? null,
    last_divergent_accumulator_decimal: channel.last_divergent_accumulator_decimal ?? null,
    last_default_output_code: channel.last_default_output_code ?? null,
    last_single_output_code: channel.last_single_output_code ?? null,
  });
  return {
    ...equivalence,
    maximum_absolute_output_delta: equivalence.maximum_absolute_output_delta ?? null,
    ops: (equivalence.ops || []).map((row) => ({
      ...row,
      maximum_absolute_output_delta: row.maximum_absolute_output_delta ?? null,
      maximum_pair_segment_count: row.maximum_pair_segment_count ?? null,
      maximum_divergent_region_count: row.maximum_divergent_region_count ?? null,
      activation_code_range: row.activation_code_range ?? null,
      output_zero_point: row.output_zero_point ?? null,
      channel_first_divergent_accumulators_decimal: (row.channel_first_divergent_accumulators_decimal || []).map((value) => value ?? null),
      channel_first_default_output_codes: (row.channel_first_default_output_codes || []).map((value) => value ?? null),
      channel_first_single_output_codes: (row.channel_first_single_output_codes || []).map((value) => value ?? null),
      top_channels: (row.top_channels || []).map(normalizeTopChannel),
    })),
  };
}

function normalizeAccumulatorReachability(reachability) {
  if (!reachability) return null;
  const normalizeTopChannel = (channel) => ({
    ...channel,
    coverage_failure_step_index: channel.coverage_failure_step_index ?? null,
    first_exact_reachable_divergent_accumulator_decimal: channel.first_exact_reachable_divergent_accumulator_decimal ?? null,
    first_default_output_code: channel.first_default_output_code ?? null,
    first_single_output_code: channel.first_single_output_code ?? null,
    last_exact_reachable_divergent_accumulator_decimal: channel.last_exact_reachable_divergent_accumulator_decimal ?? null,
  });
  return {
    ...reachability,
    maximum_lattice_gcd: reachability.maximum_lattice_gcd ?? null,
    ops: (reachability.ops || []).map((row) => ({
      ...row,
      maximum_lattice_gcd: row.maximum_lattice_gcd ?? null,
      channel_first_exact_reachable_divergent_accumulators_decimal: (row.channel_first_exact_reachable_divergent_accumulators_decimal || []).map((value) => value ?? null),
      top_channels: (row.top_channels || []).map(normalizeTopChannel),
    })),
  };
}

function normalizeNumericalAbiPropagation(propagation) {
  if (!propagation) return null;
  return {
    ...propagation,
    unique_predicted_boundary_logical_payload_bytes: propagation.unique_predicted_boundary_logical_payload_bytes ?? null,
    maximum_model_output_op_hops: propagation.maximum_model_output_op_hops ?? null,
    maximum_model_output_graph_route_count_decimal: propagation.maximum_model_output_graph_route_count_decimal ?? null,
    graph_edges: (propagation.graph_edges || []).map((edge) => ({
      ...edge,
      logical_payload_bytes: edge.logical_payload_bytes ?? null,
    })),
    sources: (propagation.sources || []).map((source) => ({
      ...source,
      maximum_absolute_output_delta: source.maximum_absolute_output_delta ?? null,
      minimum_model_output_op_hops: source.minimum_model_output_op_hops ?? null,
      maximum_reachable_op_hops: source.maximum_reachable_op_hops ?? null,
      exact_model_output_graph_route_count_decimal: source.exact_model_output_graph_route_count_decimal ?? null,
      model_output_paths: (source.model_output_paths || []).map((path) => ({
        ...path,
        shortest_path_boundary_logical_payload_bytes: path.shortest_path_boundary_logical_payload_bytes ?? null,
        exact_graph_route_count_decimal: path.exact_graph_route_count_decimal ?? null,
      })),
    })),
  };
}

function normalizeInputCounterexample(evidence) {
  if (!evidence) return null;
  return {
    ...evidence,
    sources: (evidence.sources || []).map((source) => ({
      ...source,
      input_tensor_index: source.input_tensor_index ?? null,
      exact_model_output_graph_route_count_decimal: source.exact_model_output_graph_route_count_decimal ?? null,
      representative_witness_index: source.representative_witness_index ?? null,
    })),
  };
}

function normalizePreprocessingRealizability(evidence) {
  if (!evidence) return null;
  return {
    ...evidence,
    best_non_exact_unrealizable_element_count: evidence.best_non_exact_unrealizable_element_count ?? null,
    candidates: (evidence.candidates || []).map((candidate) => ({
      ...candidate,
      exact_rgb_fixture_sha256: candidate.exact_rgb_fixture_sha256 ?? null,
      first_unrealizable_element: candidate.first_unrealizable_element ?? null,
    })),
  };
}

function normalizeContractMigration(migration) {
  if (!migration) return null;
  return {
    ...migration,
    migrations: (migration.migrations || []).map((row) => ({
      ...row,
      scenarios: (row.scenarios || []).map((scenario) => ({
        ...scenario,
        kernel_consumers: (scenario.kernel_consumers || []).map((consumer) => ({
          ...consumer,
          weight_tensor_index: consumer.weight_tensor_index ?? null,
          bias_tensor_index: consumer.bias_tensor_index ?? null,
          output_tensor_index: consumer.output_tensor_index ?? null,
          output_scale: consumer.output_scale ?? null,
          maximum_absolute_bias_rebase_error: consumer.maximum_absolute_bias_rebase_error ?? null,
          maximum_absolute_bias_rebase_error_current_steps: consumer.maximum_absolute_bias_rebase_error_current_steps ?? null,
          maximum_absolute_bias_rebase_error_candidate_steps: consumer.maximum_absolute_bias_rebase_error_candidate_steps ?? null,
          channel_bias_rebase_error_current_steps: (consumer.channel_bias_rebase_error_current_steps || []).map((value) => value ?? null),
        })),
        add_consumers: (scenario.add_consumers || []).map((consumer) => ({
          ...consumer,
          output_tensor_index: consumer.output_tensor_index ?? null,
          current_parameters: consumer.current_parameters ?? null,
          candidate_parameters: consumer.candidate_parameters ?? null,
        })),
      })),
    })),
  };
}

function normalizeResidualStepResponse(response) {
  if (!response) return null;
  return {
    ...response,
    maximum_containment_silent_ratio_increase: response.maximum_containment_silent_ratio_increase ?? null,
    residual_adds: (response.residual_adds || []).map((row) => ({
      ...row,
      maximum_containment_silent_ratio_increase: row.maximum_containment_silent_ratio_increase ?? null,
      maximum_containment_additional_silent_transitions: row.maximum_containment_additional_silent_transitions ?? null,
      maximum_containment_removed_clamp_pairs: row.maximum_containment_removed_clamp_pairs ?? null,
      retention_cost_rank: row.retention_cost_rank ?? null,
      contracts: (row.contracts || []).map((contract) => ({
        ...contract,
        branch_responses: (contract.branch_responses || []).map((branch) => ({
          ...branch,
          worst_jump: branch.worst_jump ?? null,
          first_unclipped_silent: branch.first_unclipped_silent ?? null,
        })),
      })),
    })),
  };
}

function normalizeResidualContractDistortion(distortion) {
  if (!distortion) return null;
  return {
    ...distortion,
    maximum_rms_contract_delta_current_steps: distortion.maximum_rms_contract_delta_current_steps ?? null,
    maximum_p99_contract_delta_current_steps: distortion.maximum_p99_contract_delta_current_steps ?? null,
    residual_adds: (distortion.residual_adds || []).map((row) => ({
      ...row,
      current_output_scale: row.current_output_scale ?? null,
      current_output_zero_point: row.current_output_zero_point ?? null,
      maximum_rms_contract_delta_current_steps: row.maximum_rms_contract_delta_current_steps ?? null,
      maximum_p99_contract_delta_current_steps: row.maximum_p99_contract_delta_current_steps ?? null,
      maximum_rescued_current_clamp_pair_count: row.maximum_rescued_current_clamp_pair_count ?? null,
      distortion_rank: row.distortion_rank ?? null,
      scenarios: (row.scenarios || []).map((scenario) => ({ ...scenario })),
    })),
  };
}

export function buildFindingsEvidence(analysis, findingsContext = {}) {
  return {
    schema: ANALYZER_METADATA.schemas.findingsRegister,
    analyzer_metadata: buildAnalyzerMetadata(analysis),
    synthesis_method: {
      origin: "report_synthesis",
      method_version: ANALYZER_METADATA.schemas.findingsRegister,
      evidence_pointer_base: ANALYZER_METADATA.schemas.engineeringEvidence,
    },
    findings: buildFindingsRegister(analysis, findingsContext),
    authoritative_action_source: "findings",
    raw_analyzer_signals: normalizeNativeAnalyzerFindings(analysis, buildFindingsRegister(analysis, findingsContext)),
  };
}

export function buildMemoryCacheCsv(analysis) {
  const rows = [["op_index", "op_name", "macs", "macs_decimal", "macs_status", "macs_reason", "mac_percent", "estimated_bytes", "estimated_bytes_status", "row_working_set_bytes", "row_working_set_status", "row_working_set_ratio", "row_working_set_ratio_status", "static_bound", "xnnpack", "packing_us", "channel_tail_overhead_percent"]];
  for (const op of artifactIrOperators(analysis) || []) {
    rows.push([
      op.index,
      op.name,
      Number.isSafeInteger(Number(op.macs)) ? Number(op.macs) : "",
      op.macs_decimal ?? (Number.isSafeInteger(Number(op.macs)) ? String(op.macs) : ""),
      op.macs_status || "assessed",
      op.macs_reason || "",
      op.mac_percent == null ? "" : Number(op.mac_percent),
      op.estimated_bytes == null ? "" : Number(op.estimated_bytes),
      op.estimated_bytes_status || (op.estimated_bytes == null ? "not_assessed" : "assessed"),
      op.row_working_set_bytes == null ? "" : Number(op.row_working_set_bytes),
      op.row_working_set_status || (op.row_working_set_bytes == null ? "not_assessed" : "assessed"),
      op.row_working_set_ratio == null ? "" : Number(op.row_working_set_ratio),
      op.row_working_set_status || (op.row_working_set_ratio == null ? "not_assessed" : "assessed"),
      op.static_bound_guess || "",
      xnnpackLabel(op, analysis),
      Number(op.weight_packing_overhead_us || 0),
      Number(op.channel_tail_overhead_percent || 0),
    ]);
  }
  return rows.map((row) => row.map(csvCell).join(",")).join("\n");
}

export function buildArenaPlanCsv(analysis) {
  const rows = [["tensor_index", "tensor_name", "allocation_status", "arena", "dtype", "shape", "size_bytes", "offset_bytes", "first_node", "last_node", "shared_with_tensor_index"]];
  for (const allocation of analysis?.tensor_arena_plan?.allocations || []) {
    rows.push([
      allocation.tensor_index,
      allocation.tensor_name || "",
      allocation.allocation_status || "",
      allocation.arena || "",
      allocation.tensor_dtype || "",
      (allocation.tensor_shape || []).join("x"),
      allocation.size_bytes ?? "",
      allocation.offset_bytes ?? "",
      allocation.first_node ?? "",
      allocation.last_node ?? "",
      allocation.shared_with_tensor_index ?? "",
    ]);
  }
  return rows.map((row) => row.map(csvCell).join(",")).join("\n");
}

export function buildRuntimeAssignmentComparisonCsv(runtimeAssignment) {
  const provenance = runtimeAssignmentCsvProvenance(runtimeAssignment);
  const rows = [["artifact_sha256", "target_profile_id", "target_profile_sha256", "runtime_name", "runtime_version", "runtime_backend", "runtime_build", "duration_semantics", "evidence_class", "op_index", "op_name", "classification", "matches_prediction", "predicted_delegated", "predicted_domain", "observed_delegated", "observed_provider", "observed_partition_id", "observed_lowering_id", "observed_kernel_id", "observed_kernel", "selector_evidence_class", "resolved_selector_dimensions", "macs", "estimated_logical_bytes", "duration_us"]];
  for (const item of runtimeAssignment?.comparison?.op_comparisons || []) rows.push([
    ...provenance,
    item.op_index,
    item.op_name || "",
    item.classification || "",
    item.matches_prediction ?? "",
    item.predicted_delegated,
    item.predicted_domain || "",
    item.observed_delegated ?? "",
    item.observed_provider || "",
    item.observed_partition_id ?? "",
    item.observed_lowering_id || "",
    item.observed_kernel_id || "",
    item.observed_kernel || "",
    item.selector_evidence_class || "",
    (item.resolved_selector_dimensions || []).join("|"),
    item.macs ?? "",
    item.estimated_logical_bytes ?? "",
    item.duration_us ?? "",
  ]);
  return rows.map((row) => row.map(csvCell).join(",")).join("\n");
}

export function buildRuntimeBoundaryComparisonCsv(runtimeAssignment) {
  const provenance = runtimeAssignmentCsvProvenance(runtimeAssignment);
  const rows = [["artifact_sha256", "target_profile_id", "target_profile_sha256", "runtime_name", "runtime_version", "runtime_backend", "runtime_build", "duration_semantics", "evidence_class", "tensor_index", "tensor_name", "tensor_dtype", "tensor_shape", "producer_op_index", "producer_op_name", "consumer_op_index", "consumer_op_name", "predicted_boundary", "observed_boundary", "classification", "predicted_producer_domain", "predicted_consumer_domain", "observed_producer_domain", "observed_consumer_domain", "predicted_direction", "observed_direction", "observed_relation_reason", "payload_bytes", "materialization_status"]];
  for (const edge of runtimeAssignment?.comparison?.boundary_comparisons || []) rows.push([
    ...provenance,
    edge.tensor_index,
    edge.tensor_name || "",
    edge.tensor_dtype || "",
    (edge.tensor_shape || []).join("x") || "scalar",
    edge.producer_op_index,
    edge.producer_op_name || "",
    edge.consumer_op_index,
    edge.consumer_op_name || "",
    edge.predicted_boundary,
    edge.observed_boundary,
    edge.classification || "",
    edge.predicted_producer_domain || "",
    edge.predicted_consumer_domain || "",
    edge.observed_producer_domain || "",
    edge.observed_consumer_domain || "",
    edge.predicted_direction || "",
    edge.observed_direction || "",
    edge.observed_relation_reason || "",
    edge.payload_bytes ?? "",
    edge.materialization_status || "",
  ]);
  return rows.map((row) => row.map(csvCell).join(",")).join("\n");
}

export function buildRuntimeArenaReconciliationCsv(runtimeAssignment) {
  const reconciliation = runtimeAssignment?.arena_reconciliation || null;
  const runtimeMemory = runtimeAssignment?.runtime_memory || null;
  const provenance = runtimeAssignmentCsvProvenance(runtimeAssignment);
  const header = [
    "artifact_sha256", "target_profile_id", "target_profile_sha256", "runtime_name", "runtime_version", "runtime_backend", "runtime_build", "duration_semantics", "evidence_class",
    "memory_snapshot_id", "tensorflow_source_commit", "allocation_ledger_sha256", "projected_combined_arena_bytes", "observed_peak_combined_arena_bytes", "observed_final_combined_arena_bytes", "peak_delta_bytes", "peak_to_projection_ratio",
    "row_type", "tensor_index", "tensor_name", "artifact_tensor", "projected_present", "observed_present", "projected_arena", "observed_arena", "projected_size_bytes", "observed_size_bytes", "size_delta_bytes", "projected_offset_bytes", "observed_offset_bytes", "offset_delta_bytes", "size_match", "arena_match", "offset_match", "projected_alias_root_tensor_index", "observed_alias_root_tensor_index", "alias_root_match",
  ];
  const rows = [header];
  if (!reconciliation || !runtimeMemory) return header.map(csvCell).join(",");
  const common = [
    ...provenance,
    reconciliation.runtime_snapshot_id,
    runtimeMemory.tensorflow_source_commit || "",
    runtimeMemory.allocation_ledger_sha256 || "",
    reconciliation.projected_combined_arena_bytes ?? "",
    reconciliation.observed_peak_combined_arena_bytes ?? "",
    reconciliation.observed_final_combined_arena_bytes ?? "",
    reconciliation.peak_delta_bytes ?? "",
    reconciliation.peak_to_projection_ratio ?? "",
  ];
  for (const item of reconciliation.allocation_rows || []) rows.push([
    ...common,
    "allocation",
    item.tensor_index,
    item.tensor_name || "",
    item.artifact_tensor,
    item.projected_present,
    item.observed_present,
    item.projected_arena || "",
    item.observed_arena || "",
    item.projected_size_bytes ?? "",
    item.observed_size_bytes ?? "",
    item.size_delta_bytes ?? "",
    item.projected_offset_bytes ?? "",
    item.observed_offset_bytes ?? "",
    item.offset_delta_bytes ?? "",
    item.size_match ?? "",
    item.arena_match ?? "",
    item.offset_match ?? "",
    "", "", "",
  ]);
  for (const item of reconciliation.alias_rows || []) rows.push([
    ...common,
    "alias",
    item.tensor_index,
    ...Array(15).fill(""),
    item.projected_root_tensor_index ?? "",
    item.observed_root_tensor_index ?? "",
    item.root_match ?? "",
  ]);
  if (rows.length === 1) rows.push([...common, "summary", ...Array(19).fill("")]);
  return rows.map((row) => row.map(csvCell).join(",")).join("\n");
}

export function buildTfliteRuntimeTimingCsv(runtimeAssignment) {
  const timing = runtimeAssignment?.source?.adapter?.timing_profile || null;
  const rows = [["artifact_sha256", "target_profile_id", "target_profile_sha256", "runtime_name", "runtime_version", "runtime_backend", "runtime_build", "duration_semantics", "evidence_class", "timing_profile_sha256", "capture_id", "evidence_group", "runtime_node_index", "op_index", "partition_id", "provider", "node_type", "profile_name", "run_order", "formatter_times_called_integer_average", "run_count", "run_count_derivation_status", "event_sample_count", "first_us", "last_us", "min_us", "max_us", "sum_us", "mean_per_event_us", "mean_per_run_us", "stddev_us", "variance_us2"]];
  if (!timing) return rows[0].map(csvCell).join(",");
  const provenance = runtimeAssignmentCsvProvenance(runtimeAssignment);
  const append = (item, group) => rows.push([
    ...provenance,
    timing.profile_sha256,
    timing.capture_id,
    group,
    item.runtime_node_index ?? "",
    item.op_index ?? "",
    item.partition_id ?? "",
    item.provider || item.delegate_name || "",
    item.node_type || "",
    item.name || "",
    item.run_order,
    item.formatter_times_called_integer_average,
    item.run_count ?? "",
    item.run_count_derivation_status || "",
    item.event_sample_count,
    item.first_us,
    item.last_us,
    item.min_us,
    item.max_us,
    item.sum_us,
    item.mean_per_event_us,
    item.mean_per_run_us,
    item.stddev_us,
    item.variance_us2,
  ]);
  for (const item of timing.execution_nodes || []) append(item, item.node_kind);
  for (const item of timing.delegate_internal_events || []) append(item, "delegate_internal");
  for (const item of timing.other_primary_events || []) append(item, "other_primary_event");
  return rows.map((row) => row.map(csvCell).join(",")).join("\n");
}

function runtimeAssignmentCsvProvenance(runtimeAssignment) {
  return [
    runtimeAssignment?.artifact_sha256 || "",
    runtimeAssignment?.target_profile_id || "",
    runtimeAssignment?.target_profile_sha256 || "",
    runtimeAssignment?.runtime?.name || "",
    runtimeAssignment?.runtime?.version || "",
    runtimeAssignment?.runtime?.backend || "",
    runtimeAssignment?.runtime?.build || "",
    runtimeAssignment?.source?.duration_semantics || "unspecified",
    runtimeAssignment?.comparison?.evidence_class || "DERIVED_FROM_OBSERVED_RUNTIME",
  ];
}

export function buildModelStructureEvidence(analysis, model = {}) {
  return {
    schema: ANALYZER_METADATA.schemas.modelStructure,
    analyzer_metadata: buildAnalyzerMetadata(analysis),
    model,
    inputs: analysis.inputs || [],
    outputs: analysis.outputs || [],
    histogram: analysis.histogram || [],
    stages: analysis.stages || [],
    patterns: analysis.patterns || [],
    xnnpack_chains: analysis.xnnpack_chains || [],
    predicted_partition_boundaries: normalizePredictedPartitionBoundaries(analysis.predicted_partition_boundaries),
    tensor_arena_plan: String(analysis?.format || "tflite").toLowerCase() === "onnx"
      ? null
      : normalizeTensorArenaPlan(analysis.tensor_arena_plan),
    dynamic_inputs: contractDynamicDimSummary(analysis.inputs),
    serialized_shape_projection: deriveTfliteBatchOneProjection(analysis),
    graph_counts: {
      operators: analysis.operator_count || (artifactIrOperators(analysis) || []).length,
      tensors: analysis.tensor_count || (artifactIrValues(analysis) || []).length,
      subgraphs: analysis.subgraph_count || 1,
    },
  };
}

export function buildQuantizationEvidence(analysis, model = {}) {
  const format = String(analysis?.format || "tflite").toLowerCase();
  const tflite = format === "tflite";
  const quantResearchCoverage = tflite ? buildQuantResearchCoverage(analysis) : null;
  const researchEvidence = (key, normalize) => quantResearchCoverage?.labs.find((row) => row.evidence_key === key)?.class_supported
    ? normalize(analysis[key])
    : null;
  return {
    schema: ANALYZER_METADATA.schemas.quantizationEvidence,
    analyzer_metadata: buildAnalyzerMetadata(analysis),
    model,
    status: modelQuantizationStatus(analysis),
    scope: quantizationScopeExplanation(analysis),
    tensor_summary: {
      quantized_tensors: analysis.quantized_tensors || 0,
      per_channel_tensors: analysis.per_channel_tensors || 0,
      tensor_count: analysis.tensor_count || 0,
    },
    quant_research_coverage: quantResearchCoverage,
    quantization_contract_checks: buildQuantizationContractChecks(analysis),
    accumulator_headroom_atlas: tflite ? researchEvidence("accumulator_atlas", normalizeAccumulatorAtlas) : null,
    requantization_fidelity: tflite ? researchEvidence("requantization_fidelity", normalizeRequantizationFidelity) : null,
    kernel_extremum_witness: tflite ? researchEvidence("kernel_extremum_witness", normalizeKernelExtremumWitness) : null,
    channel_vitality: tflite ? researchEvidence("channel_vitality", normalizeChannelVitality) : null,
    rounding_equivalence: tflite ? researchEvidence("rounding_equivalence", normalizeRoundingEquivalence) : null,
    accumulator_reachability: tflite ? researchEvidence("accumulator_reachability", normalizeAccumulatorReachability) : null,
    numerical_abi_propagation: tflite ? researchEvidence("numerical_abi_propagation", normalizeNumericalAbiPropagation) : null,
    input_counterexample: tflite ? researchEvidence("input_counterexample", normalizeInputCounterexample) : null,
    preprocessing_realizability: tflite ? researchEvidence("preprocessing_realizability", normalizePreprocessingRealizability) : null,
    residual_quantization_lattice: tflite ? researchEvidence("quantization_lattice", normalizeQuantizationLattice) : null,
    residual_contract_migration: tflite ? researchEvidence("contract_migration", normalizeContractMigration) : null,
    residual_step_response: tflite ? researchEvidence("residual_step_response", normalizeResidualStepResponse) : null,
    residual_contract_distortion: tflite ? researchEvidence("residual_contract_distortion", normalizeResidualContractDistortion) : null,
    op_watchlist: (artifactIrOperators(analysis) || [])
      .filter((op) => op.quant_risk === "risk" || op.quant_risk === "warn")
      .slice(0, 80)
      .map((op) => ({
        index: op.index,
        name: op.name,
        risk: op.quant_risk,
        state: op.quantization_state,
        scale_mode: op.quant_scale_mode,
        scale_ratio: op.quant_scale_ratio,
        scale_cv: op.quant_scale_cv,
        zero_point_status: op.quant_zero_point_status,
        detail: op.quant_risk_detail,
      })),
  };
}

export function buildWeightIndicatorEvidence({
  analysis = null,
  deepBomResult = null,
  perturbationResult = null,
  deployCurvatureResult = null,
} = {}) {
  const assessment = (value, reason) => value == null
    ? { status: "not_assessed", value: null, reason }
    : { status: "assessed", value, reason: "Module completed in the browser session." };
  return {
    schema: ANALYZER_METADATA.schemas.weightIndicatorEvidence,
    analyzer_metadata: buildAnalyzerMetadata(analysis),
    deepbom: deepBomResult || null,
    perturbation_analysis: perturbationResult || null,
    deploy_curvature_basin: deployCurvatureResult || null,
    module_assessments: {
      deepbom: assessment(deepBomResult, "DEEPBOM experimental artifact-composite module was not run."),
      perturbation_analysis: assessment(perturbationResult, "Perturbation module was not run."),
      deploy_curvature_basin: assessment(deployCurvatureResult, "Deployment sensitivity module was not run."),
    },
    terminology: {
      preferred: [
        "Stored-Weight Descriptors",
        "Experimental Artifact Composite",
        "Local Synthetic Perturbation Observations",
        "Deployment-Function Sensitivity Descriptor",
      ],
      avoided_claims: [
        "poor generalization",
        "clinical robustness proven",
        "true Hessian/PAC-Bayes certificate from deployment artifact alone",
      ],
    },
  };
}

export function buildRuntimeEvidence({
  analysis = null,
  browserBucket = "unknown",
  benchmarkResults = [],
  runtimeBasinResult = null,
  preprocessingConsequenceResult = null,
  calibrationValidationResult = null,
  runtimeAssignmentEvidence = null,
  sharedArrayBufferAvailable = false,
  webgpuAvailable = false,
  webnnAvailable = false,
} = {}) {
  const benchmarks = benchmarkResults || [];
  const successfulBenchmarks = benchmarks.filter((item) => item?.ok === true);
  const failedBenchmarks = benchmarks.filter((item) => item?.ok !== true);
  const runtimeExecuted = successfulBenchmarks.length > 0 || runtimeBasinResult != null || preprocessingConsequenceResult != null || calibrationValidationResult != null;
  const browserExecuted = successfulBenchmarks.length > 0 || runtimeBasinResult != null || preprocessingConsequenceResult != null;
  const benchmarkAssessmentStatus = successfulBenchmarks.length === benchmarks.length && benchmarks.length
    ? "assessed" : successfulBenchmarks.length ? "partial" : "not_assessed";
  const runtimeEnvironmentImported = runtimeAssignmentEvidence?.schema === "deepbom.gguf_runtime_environment.v2";
  const coreMlComputePlanImported = runtimeAssignmentEvidence?.schema === "deepbom.coreml_compute_plan.v1";
  const importedAssignment = runtimeEnvironmentImported || coreMlComputePlanImported ? null : runtimeAssignmentEvidence;
  const runtimeAssignmentImported = importedAssignment != null;
  const runtimeBackendEvidenceLedger = buildRuntimeBackendEvidenceLedger(importedAssignment);
  const runtimeDataMovementEvidence = buildRuntimeDataMovementEvidence(importedAssignment);
  const onnxRuntimeShapeBinding = String(analysis?.format || "").toLowerCase() === "onnx" && importedAssignment
    ? buildOnnxRuntimeShapeBinding(analysis, importedAssignment)
    : null;
  const anyRuntimeEvidenceImported = runtimeAssignmentImported || runtimeEnvironmentImported || coreMlComputePlanImported;
  const runtimeEvidenceSidecar = anyRuntimeEvidenceImported
    ? buildRuntimeEvidenceSidecar(analysis, runtimeAssignmentEvidence)
    : null;
  return {
    schema: ANALYZER_METADATA.schemas.runtimeEvidence,
    analyzer_metadata: buildAnalyzerMetadata(analysis),
    browser_bucket: browserBucket,
    wasm_simd_expected: true,
    shared_array_buffer_available: sharedArrayBufferAvailable,
    webgpu_available: webgpuAvailable,
    webnn_available: webnnAvailable,
    runtime_execution_status: runtimeExecuted ? "executed" : failedBenchmarks.length ? "attempted_failed" : anyRuntimeEvidenceImported ? "evidence_imported" : "not_run",
    browser_execution_status: browserExecuted ? "executed" : failedBenchmarks.length ? "attempted_failed" : "not_run",
    benchmark_results: benchmarks.length ? benchmarks : null,
    runtime_basin: runtimeBasinResult || null,
    preprocessing_consequence_atlas: preprocessingConsequenceResult || null,
    representative_dataset_validation: calibrationValidationResult || null,
    runtime_assignment: importedAssignment || null,
    runtime_assignment_comparison: importedAssignment?.comparison || null,
    runtime_backend_evidence_ledger: runtimeBackendEvidenceLedger,
    runtime_data_movement_evidence: runtimeDataMovementEvidence,
    onnx_runtime_shape_binding: onnxRuntimeShapeBinding,
    runtime_evidence_sidecar: runtimeEvidenceSidecar,
    runtime_environment: runtimeEnvironmentImported ? runtimeAssignmentEvidence : null,
    coreml_compute_plan: coreMlComputePlanImported ? runtimeAssignmentEvidence : null,
    assessments: {
      benchmarks: benchmarks.length
        ? { status: benchmarkAssessmentStatus, value: benchmarks, reason: successfulBenchmarks.length
          ? `${successfulBenchmarks.length}/${benchmarks.length} browser-local benchmark backend attempt(s) completed with bound input and timing contracts.`
          : "Every browser-local benchmark attempt failed before measured execution; no latency evidence was produced." }
        : { status: "not_assessed", value: null, reason: "Browser-local benchmark was not run." },
      runtime_basin: runtimeBasinResult != null
        ? { status: "assessed", value: runtimeBasinResult, reason: "Backend consistency module completed." }
        : { status: "not_assessed", value: null, reason: "Backend consistency module was not run." },
      preprocessing_consequence_atlas: preprocessingConsequenceResult != null
        ? { status: "assessed", value: preprocessingConsequenceResult, reason: "Every explicit preprocessing counterfactual was replayed twice through the browser-local LiteRT.js WASM runtime and independently verified against the source tensor witness, input/output digests, difference counts, equivalence classes, and SHA-256 ledgers." }
        : { status: "not_assessed", value: null, reason: "Preprocessing consequence replay was not run in this browser session." },
      representative_dataset_validation: calibrationValidationResult != null
        ? { status: "assessed", value: calibrationValidationResult, reason: "A local representative-dataset capture was artifact-, dataset-, preprocessing-, and runtime-bound; interface endpoint counts, same-contract reference drift, repeated-run differences, and the JCS/SHA-256 ledger were independently reconstructed before export." }
        : { status: "not_assessed", value: null, reason: "No hash-bound representative dataset capture was imported in this browser session." },
      runtime_assignment: runtimeAssignmentImported
        ? { status: "assessed", value: importedAssignment, reason: importedAssignment?.source?.adapter
          ? importedAssignment.source.adapter.schema?.startsWith("deepbom.tflite_runtime_info_adapter.v")
            ? "Locally imported TFLite ModelRuntimeDetails evidence was bound to the active artifact by exact original-op topology and exposes the observed execution plan and delegate replacement map; imported BenchmarkProfilingData timing, when present, remains separated by execution node, delegate partition, and delegate-internal event. Runtime and capture identity are declared, and neither source proto embeds artifact SHA-256."
            : importedAssignment.source.adapter.native_capture
              ? "Locally captured pinned native ONNX Runtime profile evidence was bound to the active artifact, target profile, canonical envelope, embedded profile, package identity, collector-recorded host binary inventory, and deterministically mapped original graph ops. The browser reverified envelope/artifact/profile hashes but did not re-read native binaries."
              : "Locally imported ONNX Runtime profile evidence was bound to artifact, target profile, raw profile SHA-256, runtime identity, and deterministically mapped original graph ops."
          : "Locally imported runtime assignment evidence was bound to artifact SHA-256 and target-profile SHA-256." }
        : { status: "not_assessed", value: null, reason: "No bound runtime assignment evidence was imported." },
      runtime_assignment_comparison: runtimeAssignmentImported
        ? { status: "assessed", value: importedAssignment.comparison || null, reason: `Comparison calculation completed with internal coverage status ${importedAssignment.comparison?.status || "partial"}; incomplete rows or partition IDs remain explicitly unassessed.` }
        : { status: "not_assessed", value: null, reason: "No bound runtime assignment evidence was imported." },
      runtime_backend_evidence_ledger: runtimeBackendEvidenceLedger
        ? { status: "assessed", value: runtimeBackendEvidenceLedger, reason: "QNN, NNAPI, Core ML, WebGPU, and WebNN selected-build inclusion, capability acceptance, assignment, and execution were normalized without promoting evidence across layers." }
        : { status: "not_assessed", value: null, reason: "No imported ORT evidence can populate the selected runtime backend ledger." },
      runtime_data_movement_evidence: runtimeDataMovementEvidence
        ? { status: runtimeDataMovementEvidence.status.startsWith("observed_") ? "assessed" : "partial", value: runtimeDataMovementEvidence, reason: runtimeDataMovementEvidence.status === "observed_copy_node_payload"
          ? `${runtimeDataMovementEvidence.observed_copy_node_count} executed ORT copy node(s) expose a conserved logical output payload; physical transfer bytes remain unavailable.`
          : runtimeDataMovementEvidence.status === "observed_no_profiled_copy_nodes_for_captured_configuration"
            ? "The captured production ORT profile contains no copy-node event. This is specific to the bound configuration and is not a static zero-transfer claim."
            : "The imported ORT capture predates copy-node output-size preservation or has incomplete copy payload fields; no transfer total is promoted." }
        : { status: "not_assessed", value: null, reason: "No bound native ORT paired profile can provide copy-node runtime evidence." },
      onnx_runtime_shape_binding: onnxRuntimeShapeBinding
        ? {
          status: onnxRuntimeShapeBinding.status === "fail" ? "failed_closed"
            : onnxRuntimeShapeBinding.status.startsWith("partial") ? "partial" : "assessed",
          value: onnxRuntimeShapeBinding,
          reason: onnxRuntimeShapeBinding.status === "fail"
            ? "Imported runtime type-shape observations conflict with the serialized graph or with repeated profile events; no runtime-bound total is promoted."
            : `${onnxRuntimeShapeBinding.observed_internal_tensor_count} internal tensor contract(s) were bound; ${onnxRuntimeShapeBinding.runtime_closed_mac_op_count} previously unresolved MAC op(s) were closed and ${onnxRuntimeShapeBinding.remaining_unassessed_mac_op_count} remain unresolved.`,
        }
        : { status: "not_assessed", value: null, reason: "No bound ONNX Runtime assignment with type-shape evidence was imported." },
      runtime_evidence_sidecar: runtimeEvidenceSidecar
        ? { status: "assessed", value: runtimeEvidenceSidecar, reason: "The imported format-specific runtime document was projected into one artifact-, build-, capture-, and claim-state-bound cross-format evidence index without changing the source evidence class." }
        : { status: "not_assessed", value: null, reason: "No imported runtime evidence can populate the cross-format sidecar." },
      runtime_environment: runtimeEnvironmentImported
        ? { status: "assessed", value: runtimeAssignmentEvidence, reason: "An imported GGUF runtime manifest is bound to the active artifact, pinned source and scheduler patch, selected-build attestation, exact runtime binary, CMake cache, complete backend-option inventory, requested backend, device, and collector observations. Captured scheduler graphs retain original and inserted nodes, split ranges, backend assignment, transitions, and dispatch status." }
        : { status: "not_assessed", value: null, reason: "No bound instrumented GGUF runtime manifest was imported." },
      coreml_compute_plan: coreMlComputePlanImported
        ? { status: "assessed", value: runtimeAssignmentEvidence, reason: "An imported Core ML MLComputePlan estimate is bound to the active artifact, compiled-model digest, compute-unit configuration, runtime/source identity, and exact decoded operation order. Preferred/supported devices and relative costs remain anticipated plan evidence, not executed placement or timing." }
        : { status: "not_assessed", value: null, reason: "No bound Core ML MLComputePlan estimate was imported." },
    },
    notes: `Runtime Benchmark values are browser-local synthetic/prepared-input measurements. Successful rows bind exact executed input contracts and ${BROWSER_BENCHMARK_TIMING_METHOD}: one cold invocation before warmup, then the declared warmup count, then the declared measured sample count. Preprocessing Consequence Atlas rows are browser-local counterfactual tensor replays, not observations of production preprocessing or task accuracy. ${RUNTIME_COMPATIBILITY_EVIDENCE_LABEL} compares backend outputs when local LiteRT paths complete. Adapted ORT profile durations are arithmetic means over emitted node events and are additive only for declared sequential execution with equal per-op sample counts. TFLite ModelRuntimeDetails contributes execution-plan and delegate assignment evidence; separately imported BenchmarkProfilingData contributes timing only under strict execution-node coverage and capture-binding rules. Neither source exposes executed-microkernel evidence.`,
  };
}

export { buildSecurityPostureEvidence, collectRuntimeWarnings };

export function buildEvidenceLevelIndex(analysis, {
  runtimeEvidence = {},
  findingsContext = {},
} = {}) {
  const findingsRegister = buildFindingsEvidence(analysis, findingsContext);
  const runtimeResults = buildRuntimeEvidence({ analysis, ...runtimeEvidence });
  return {
    metric_coverage: {
      schema: ANALYZER_METADATA.schemas.metricCoverageManifest,
      format: String(analysis?.format || "tflite").toLowerCase(),
      entries: buildMetricCoverageEntries(analysis, {
        findings: findingsRegister.findings,
        runtimeResults,
      }),
    },
    findings: findingsRegister.findings,
  };
}

export function buildRawEvidenceFiles(analysis, {
  identity = {},
  runtimeEvidence = {},
  weightIndicatorEvidence = {},
  securityPosture = null,
  findingsContext = {},
} = {}) {
  const security = securityPosture || buildSecurityPostureEvidence({
    analysis,
    model: identity,
    fileSizeBytes: analysis?.file_size || 0,
    runtimeBenchmarkResults: runtimeEvidence.benchmarkResults || [],
    runtimeBasinResult: runtimeEvidence.runtimeBasinResult || null,
    preprocessingConsequenceResult: runtimeEvidence.preprocessingConsequenceResult || null,
  });
  const findings = buildFindingsEvidence(analysis, findingsContext);
  const runtimeResults = buildRuntimeEvidence({ analysis, ...runtimeEvidence });
  const executionPlacement = buildExecutionPlacementEvidence(analysis, runtimeEvidence);
  const evidenceRoot = {
    evidence: {
      static_analysis: buildStaticAnalysisExport(analysis),
      quantization: buildQuantizationEvidence(analysis, identity),
      runtime_results: runtimeResults,
      execution_placement: executionPlacement,
      findings_register: findings,
    },
    supplemental_sources: {
      roofline_csv: analysis?.roofline_csv || "",
      core_isolation_roofline_csv: analysis?.core_isolation_csv || "",
      stage_graph_mermaid: analysis?.stage_mermaid || "",
    },
  };
  const metricCoverage = buildMetricCoverageManifest(analysis, {
    findings: findings.findings,
    runtimeResults,
    evidenceRoot,
    reportAccessedFieldPaths: buildEngineeringReportArtifacts(analysis, {
      identity,
      runtimeEvidence,
      executionPlacementEvidence: executionPlacement,
    }).reportAccessedFieldPaths,
  });
  return [
    zipTextFile("raw-evidence/model_structure.json", jsonForDownload(buildModelStructureEvidence(analysis, identity))),
    zipTextFile("raw-evidence/quantization.json", jsonForDownload(buildQuantizationEvidence(analysis, identity))),
    zipTextFile("raw-evidence/memory_cache.csv", buildMemoryCacheCsv(analysis)),
    zipTextFile("raw-evidence/arena_plan.csv", buildArenaPlanCsv(analysis)),
    zipTextFile("raw-evidence/runtime_results.json", jsonForDownload(runtimeResults)),
    zipTextFile("raw-evidence/execution_placement.json", jsonForDownload(executionPlacement)),
    zipTextFile("raw-evidence/weight_indicators.json", jsonForDownload(buildWeightIndicatorEvidence({ analysis, ...weightIndicatorEvidence }))),
    zipTextFile("raw-evidence/security_posture.json", jsonForDownload(security)),
    zipTextFile("raw-evidence/analyzer_metadata.json", jsonForDownload(buildAnalyzerMetadata(analysis))),
    zipTextFile("raw-evidence/findings_register.json", jsonForDownload(findings)),
    zipTextFile("raw-evidence/metric_coverage.json", jsonForDownload(metricCoverage)),
  ];
}

export function buildEngineeringEvidenceDocument(analysis, {
  reportContext = {},
  rawEvidenceContext = {},
  mlBomDocument = null,
  graphSvgText = "",
  executionPlacementEvidence = null,
} = {}) {
  const identity = rawEvidenceContext.identity || reportContext.identity || {};
  const runtimeEvidence = rawEvidenceContext.runtimeEvidence || {};
  const weightIndicatorEvidence = rawEvidenceContext.weightIndicatorEvidence || {};
  const security = rawEvidenceContext.securityPosture || buildSecurityPostureEvidence({
    analysis,
    model: identity,
    fileSizeBytes: analysis?.file_size || 0,
    runtimeBenchmarkResults: runtimeEvidence.benchmarkResults || [],
    runtimeBasinResult: runtimeEvidence.runtimeBasinResult || null,
    preprocessingConsequenceResult: runtimeEvidence.preprocessingConsequenceResult || null,
  });
  const findingsContext = {
    ...(rawEvidenceContext.findingsContext || reportContext.findingsContext || {}),
    runtimeEvidence: rawEvidenceContext.runtimeEvidence || reportContext.runtimeEvidence || null,
  };
  const staticAnalysis = buildStaticAnalysisExport(analysis);
  const artifactIr = resolveReportArtifactIr(analysis, identity, {
    artifactIrContext: rawEvidenceContext.artifactIrContext || reportContext.artifactIrContext || null,
    artifactIr: rawEvidenceContext.artifactIr || reportContext.artifactIr || null,
    runtimeEvidence: rawEvidenceContext.runtimeEvidence || reportContext.runtimeEvidence || null,
  });
  const modelStructure = buildModelStructureEvidence(analysis, identity);
  const quantization = buildQuantizationEvidence(analysis, identity);
  const runtimeResults = buildRuntimeEvidence({ analysis, ...runtimeEvidence });
  const executionPlacement = executionPlacementEvidence || buildExecutionPlacementEvidence(analysis, runtimeEvidence);
  const weightIndicators = buildWeightIndicatorEvidence({ analysis, ...weightIndicatorEvidence });
  const findingsRegister = buildFindingsEvidence(analysis, findingsContext);
  const engineeringReportArtifacts = buildEngineeringReportArtifacts(analysis, {
    ...reportContext,
    runtimeEvidence: reportContext.runtimeEvidence || runtimeEvidence,
    executionPlacementEvidence: executionPlacement,
  });
  const engineeringReport = engineeringReportArtifacts.report;
  const changeAnalysis = buildChangeAnalysis(analysis, {
    priorSnapshot: reportContext.priorSnapshot || null,
    identity,
  });
  const evidenceRoot = {
    evidence: {
      static_analysis: staticAnalysis,
      ...(artifactIr ? { artifact_ir: artifactIr } : {}),
      quantization,
      runtime_results: runtimeResults,
      execution_placement: executionPlacement,
      findings_register: findingsRegister,
    },
    supplemental_sources: {
      roofline_csv: analysis?.roofline_csv || "",
      core_isolation_roofline_csv: analysis?.core_isolation_csv || "",
      stage_graph_mermaid: analysis?.stage_mermaid || "",
    },
  };
  const metricCoverage = buildMetricCoverageManifest(analysis, {
    findings: findingsRegister.findings,
    runtimeResults,
    evidenceRoot,
    reportAccessedFieldPaths: engineeringReportArtifacts.reportAccessedFieldPaths,
  });
  const analyzerMetadata = buildAnalyzerMetadata(analysis);
  const analyzerBuildContentManifest = buildAnalyzerContentManifest();
  const conformanceEvidenceRoot = {
    schema: ANALYZER_METADATA.schemas.engineeringEvidence,
    generated_at: reportContext.generatedAt || rawEvidenceContext.generatedAt || null,
    analyzer_metadata: analyzerMetadata,
    model: identity,
    evidence: {
      static_analysis: staticAnalysis,
      ...(artifactIr ? { artifact_ir: artifactIr } : {}),
      model_structure: modelStructure,
      quantization,
      runtime_results: runtimeResults,
      execution_placement: executionPlacement,
      weight_indicators: weightIndicators,
      security_posture: security,
      findings_register: findingsRegister,
      metric_coverage_manifest: metricCoverage,
      change_analysis: changeAnalysis,
      analyzer_build_content_manifest: analyzerBuildContentManifest,
      mlbom_cyclonedx: mlBomDocument || {},
    },
  };
  const conformanceReport = assertConformance(buildConformanceReport({
    analysis,
    staticAnalysis,
    quantization,
    findingsRegister,
    runtimeResults,
    executionPlacement,
    securityPosture: security,
    mlBomDocument: mlBomDocument || {},
    engineeringReport,
    metricCoverage,
    evidenceRoot: conformanceEvidenceRoot,
  }));

  return {
    schema: ANALYZER_METADATA.schemas.engineeringEvidence,
    generated_at: reportContext.generatedAt || rawEvidenceContext.generatedAt || null,
    analyzer_metadata: analyzerMetadata,
    model: identity,
    evidence: {
      static_analysis: staticAnalysis,
      ...(artifactIr ? { artifact_ir: artifactIr } : {}),
      model_structure: modelStructure,
      quantization,
      runtime_results: runtimeResults,
      execution_placement: executionPlacement,
      weight_indicators: weightIndicators,
      security_posture: security,
      findings_register: findingsRegister,
      metric_coverage_manifest: metricCoverage,
      conformance_report: conformanceReport,
      change_analysis: changeAnalysis,
      analyzer_build_content_manifest: analyzerBuildContentManifest,
      mlbom_cyclonedx: mlBomDocument || {},
    },
    supplemental_sources: {
      raw_static_audit_markdown: analysis?._markdown || buildStaticAuditMarkdown(analysis, analysis?.model_sha256 || "") || "",
      roofline_csv: analysis?.roofline_csv || "",
      core_isolation_roofline_csv: analysis?.core_isolation_csv || "",
      memory_cache_csv: buildMemoryCacheCsv(analysis),
      arena_plan_csv: buildArenaPlanCsv(analysis),
      runtime_assignment_comparison_csv: buildRuntimeAssignmentComparisonCsv(runtimeResults.runtime_assignment),
      runtime_boundary_comparison_csv: buildRuntimeBoundaryComparisonCsv(runtimeResults.runtime_assignment),
      runtime_arena_reconciliation_csv: buildRuntimeArenaReconciliationCsv(runtimeResults.runtime_assignment),
      tflite_runtime_timing_csv: buildTfliteRuntimeTimingCsv(runtimeResults.runtime_assignment),
      stage_graph_mermaid: analysis?.stage_mermaid || "",
      graph_neighborhood_svg: graphSvgText || "",
    },
    package_notes: {
      organization: "Machine-readable engineering evidence is consolidated in this document to keep the ZIP compact.",
      visual_exports: "Rendered PNG charts are available through the separate Visuals ZIP; their deterministic source values remain in static_analysis.",
      raw_model_bytes_included: false,
      constructive_input_tensor: analysis?.input_counterexample?.representative_witness_count
        ? "The compact engineering ZIP includes one independently reconstructable raw model-input witness tensor as input_counterexample_input.bin; its dtype, shape, sparse construction, and SHA-256 remain in static_analysis.input_counterexample."
        : "No constructive model-input tensor witness was emitted.",
      preprocessing_fixture: analysis?.preprocessing_realizability?.exact_tensor_realization_candidate_count
        ? "Exact and minimum-error RGB fixtures remain available in the viewer and Raw Data package; the compact engineering ZIP retains their formulas, LUTs, counts, and SHA-256 ledgers inside engineering_evidence.json without adding per-candidate files."
        : "No exact preprocessing fixture was emitted.",
    },
  };
}

export function buildEngineeringBundleArtifactFiles(analysis, {
  reportContext = {},
  rawEvidenceContext = {},
  mlBomDocument = null,
  graphSvgText = "",
} = {}) {
  const runtimeEvidence = rawEvidenceContext.runtimeEvidence || reportContext.runtimeEvidence || {};
  const executionPlacement = buildExecutionPlacementEvidence(analysis, runtimeEvidence);
  const files = [
    zipTextFile("engineering_report.md", buildEngineeringReport(analysis, {
      ...reportContext,
      executionPlacementEvidence: executionPlacement,
    })),
    zipTextFile("engineering_evidence.json", jsonForDownload(buildEngineeringEvidenceDocument(analysis, {
      reportContext,
      rawEvidenceContext,
      mlBomDocument,
      graphSvgText,
      executionPlacementEvidence: executionPlacement,
    }))),
  ];
  const inputWitness = analysis?.input_counterexample?.witnesses?.[0];
  if (inputWitness) files.push(zipBinaryFile("input_counterexample_input.bin", buildInputWitnessTensor(inputWitness).bytes));
  return files;
}

export function buildRawDataArtifactFiles(analysis, {
  rawEvidenceContext = {},
  mlBomDocument = null,
  graphSvgText = "",
  visualPngFiles = [],
} = {}) {
  const artifactIr = resolveReportArtifactIr(analysis, rawEvidenceContext.identity || {}, rawEvidenceContext);
  const files = [
    zipTextFile("static/raw_static_audit.md", analysis?._markdown || buildStaticAuditMarkdown(analysis, analysis?.model_sha256 || "") || ""),
    zipTextFile("static/roofline.csv", analysis?.roofline_csv || ""),
    zipTextFile("static/core_isolation_roofline.csv", analysis?.core_isolation_csv || ""),
    zipTextFile("static/stage_graph.mmd", analysis?.stage_mermaid || ""),
    zipTextFile("static/static_analysis.json", jsonForDownload(buildStaticAnalysisExport(analysis))),
    ...(artifactIr ? [zipTextFile("static/artifact_ir.json", jsonForDownload(artifactIr))] : []),
    zipTextFile("static/arena_plan.csv", buildArenaPlanCsv(analysis)),
    zipTextFile("static/mlbom_cdx.json", jsonForDownload(mlBomDocument || {})),
    ...buildRawEvidenceFiles(analysis, rawEvidenceContext),
  ];
  const runtimeAssignment = rawEvidenceContext?.runtimeEvidence?.runtimeAssignmentEvidence || null;
  if (runtimeAssignment?.comparison) {
    files.push(zipTextFile("runtime/assignment_comparison.csv", buildRuntimeAssignmentComparisonCsv(runtimeAssignment)));
    files.push(zipTextFile("runtime/boundary_comparison.csv", buildRuntimeBoundaryComparisonCsv(runtimeAssignment)));
    if (runtimeAssignment?.source?.adapter?.timing_profile) files.push(zipTextFile("runtime/tflite_timing.csv", buildTfliteRuntimeTimingCsv(runtimeAssignment)));
  }
  if (runtimeAssignment?.runtime_memory && runtimeAssignment?.arena_reconciliation) {
    files.push(zipTextFile("runtime/arena_reconciliation.csv", buildRuntimeArenaReconciliationCsv(runtimeAssignment)));
  }
  if (graphSvgText) files.push(zipTextFile("static/graph_neighborhood.svg", graphSvgText));
  const inputWitness = analysis?.input_counterexample?.witnesses?.[0];
  if (inputWitness) files.push(zipBinaryFile("static/input_counterexample_input.bin", buildInputWitnessTensor(inputWitness).bytes));
  const preprocessing = analysis?.preprocessing_realizability;
  const exactCandidate = (preprocessing?.candidates || []).find((candidate) => candidate.exact_tensor_realization);
  if (inputWitness && exactCandidate) {
    const fixture = buildCandidateRgbFixture(exactCandidate, inputWitness);
    files.push(zipBinaryFile("static/preprocessing_exact_rgb_fixture.png", fixture.png));
  }
  files.push(...(visualPngFiles || []));
  return files;
}

function resolveReportArtifactIr(analysis, identity = {}, evidenceContext = {}) {
  const size = Number(identity.byte_length?.number ?? identity.byte_length ?? identity.file_size_bytes
    ?? analysis?.file_size_bytes ?? analysis?.file_size ?? 0);
  const sha256 = String(identity.sha256 || identity.hash || analysis?.model_sha256 || analysis?.artifact_sha256 || "").trim().toLowerCase();
  if (!/^[a-f0-9]{64}$/.test(sha256)) return null;
  const runtimeEvidence = evidenceContext?.runtimeEvidence?.runtimeAssignmentEvidence
    || evidenceContext?.runtimeEvidence?.runtime_assignment
    || evidenceContext?.runtimeEvidence
    || null;
  return resolveArtifactIrContext(analysis, {
    filename: identity.filename || identity.name || analysis?.filename || "model",
    format: analysis?.format,
    sha256,
    size: Number.isSafeInteger(size) && size >= 0 ? size : 0,
    artifact_set_sha256: analysis?.artifact_set?.artifact_set_sha256 || null,
  }, {
    artifactIrContext: evidenceContext.artifactIrContext || null,
    artifactIr: evidenceContext.artifactIr || null,
    runtimeEvidence,
  })?.artifact_ir || null;
}
