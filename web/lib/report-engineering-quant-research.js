import { formatBytes, formatNumber, formatPercent, padOp } from "./format.js";
import { code, markdownTable } from "./report-utils.js";
import { buildQuantResearchCoverage } from "./quant-research-applicability.js";
import { reachabilityDivergenceChannelPartition } from "./report-engineering-derivations.js";

function isOnnxAnalysis(analysis) {
  return String(analysis?.format || "").toLowerCase() === "onnx";
}

function signedScientific(value) {
  const number = Number(value || 0);
  return `${number >= 0 ? "+" : ""}${number.toExponential(6)}`;
}

function signedCount(value) {
  const number = Number(value || 0);
  return `${number >= 0 ? "+" : ""}${formatNumber(number)}`;
}

function accumulatorAtlasMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) return "";
  const atlas = analysis?.accumulator_atlas;
  if (!atlas) return [
    "## Accumulator Headroom Lab (NOT_ASSESSED)",
    "The exact stored-weight channel accumulator analysis was not emitted.",
  ].join("\n");
  const assessed = (atlas.ops || []).filter((row) => row.assessment_status === "assessed");
  const byIndex = new Map(assessed.map((row) => [row.op_index, row]));
  const ranked = (atlas.headroom_ranking_op_indices || []).map((index) => byIndex.get(index)).filter(Boolean);
  const displayed = ranked.slice(0, 32);
  const materialBiasRows = assessed.filter(
    (row) => Number(row.bias_half_range_material_exceedance_channel_count || 0) > 0,
  );
  const unassessed = (atlas.ops || []).filter((row) => row.assessment_status !== "assessed");
  const exactInteger = (value) => {
    try { return BigInt(value).toLocaleString("en-US"); } catch { return String(value ?? "N/A"); }
  };
  return [
    "## Accumulator Headroom Lab (DERIVED EXACT INTEGER DOMAIN)",
    markdownTable(["Metric", "Value"], [
      ["Schema / method", `${atlas.schema} / ${atlas.method_version}`],
      ["Candidate / assessed / unassessed ops", `${formatNumber(atlas.candidate_op_count)} / ${formatNumber(atlas.assessed_op_count)} / ${formatNumber(atlas.unassessed_op_count)}`],
      ["Exactly bounded output channels", formatNumber(atlas.assessed_channel_count)],
      ["INT32-safe / full-domain exceedance channels", `${formatNumber(atlas.int32_safe_channel_count)} / ${formatNumber(atlas.int32_overflow_channel_count)}`],
      ["Stored INT32 bias channels / maximum use", `${formatNumber(atlas.stored_bias_channel_count)} / ${formatPercent(atlas.maximum_bias_int32_ratio)}`],
      ["Bias half-range strict / float32-adjacent / material", `${formatNumber(atlas.bias_half_range_exceedance_channel_count)} / ${formatNumber(atlas.bias_half_range_guard_adjacent_channel_count)} / ${formatNumber(atlas.bias_half_range_material_exceedance_channel_count)}`],
      ["Exact-zero kernel channels / material-bias intersection", `${formatNumber(atlas.exact_zero_kernel_channel_count)} / ${formatNumber(atlas.exact_zero_bias_half_range_material_exceedance_channel_count)}`],
      ["Half-range source reference / numerical tolerance", `${exactInteger(atlas.bias_half_range_reference_decimal)} / ${exactInteger(atlas.bias_half_range_float32_tolerance_codes)} code(s); [source](${atlas.bias_half_range_reference_source_url})`],
      ["Maximum accumulator envelope", exactInteger(atlas.maximum_absolute_accumulator_decimal)],
      ["Maximum INT32 utilization", formatPercent(atlas.maximum_int32_ratio)],
      ["Maximum required width / minimum headroom", `${formatNumber(atlas.maximum_required_signed_bits)} signed bits / ${formatNumber(atlas.minimum_int32_headroom_bits)} bits`],
      ["Pinned TensorFlow source", `${atlas.source_commit}; ${(atlas.source_urls || []).map((url) => `[source](${url})`).join(" / ")}`],
    ]),
    displayed.length ? markdownTable([
      "Rank", "Operator", "Channels × terms", "Exact max envelope", "INT32 use", "Required / headroom", "Metadata-only bound", "Tightening", "Worst channel", "Ledger SHA-256",
    ], displayed.map((row, rank) => [
      `#${rank + 1}`,
      `#${padOp(row.op_index)} ${row.op_name}`,
      `${formatNumber(row.assessed_channel_count)} × ${formatNumber(row.accumulation_terms_per_channel)}`,
      exactInteger(row.maximum_absolute_accumulator_decimal),
      formatPercent(row.maximum_int32_ratio),
      `${formatNumber(row.maximum_required_signed_bits)} / ${formatNumber(row.minimum_int32_headroom_bits)} bits`,
      `${exactInteger(row.metadata_only_magnitude_bound_decimal)} (${formatPercent(row.metadata_only_int32_ratio)})`,
      `${Number(row.exact_tightening_factor).toFixed(2)}x`,
      row.worst_channel ? `${row.worst_channel.channel_index}: [${exactInteger(row.worst_channel.accumulator_envelope_min_decimal)}, ${exactInteger(row.worst_channel.accumulator_envelope_max_decimal)}]` : "N/A",
      code(row.channel_ledger_sha256),
    ])) : "No constant 8-bit convolution-family accumulator was assessed.",
    displayed.length ? `### Interval Definition Cross-Check\n\n${markdownTable(["Operator / channel", "INT32 accumulation envelope", "Post-bias requantization interval"], displayed.slice(0, 12).map((row) => [
      row.worst_channel ? `#${padOp(row.op_index)} ${row.op_name} / ch ${row.worst_channel.channel_index}` : `#${padOp(row.op_index)} ${row.op_name}`,
      row.worst_channel ? `[${exactInteger(row.worst_channel.accumulator_envelope_min_decimal)}, ${exactInteger(row.worst_channel.accumulator_envelope_max_decimal)}]` : "N/A",
      row.worst_channel ? `[${exactInteger(row.worst_channel.post_bias_min_decimal)}, ${exactInteger(row.worst_channel.post_bias_max_decimal)}]` : "N/A",
    ]))}\n\nThe accumulation envelope is \`hull(0, dot interval, post-bias interval)\` and is used only for intermediate INT32 headroom. Requantization, Kernel Witness, and Channel Vitality consume the signed post-bias interval. An envelope maximum of 0 and a post-bias maximum of -19 for the same channel are therefore consistent.` : "",
    materialBiasRows.length
      ? `### Material Bias Half-Range Exceedance\n\n${markdownTable(["Operator", "Maximum bias / INT32 use", "Strict", "Float32-adjacent", "Material", "Exact-zero material"], materialBiasRows.map((row) => [
        `#${padOp(row.op_index)} ${row.op_name}`,
        `${exactInteger(row.maximum_absolute_bias_decimal)} / ${formatPercent(row.maximum_bias_int32_ratio)}`,
        formatNumber(row.bias_half_range_exceedance_channel_count),
        formatNumber(row.bias_half_range_guard_adjacent_channel_count),
        formatNumber(row.bias_half_range_material_exceedance_channel_count),
        formatNumber(row.exact_zero_bias_half_range_material_exceedance_channel_count),
      ]))}`
      : "No stored bias exceeded the source half-range reference beyond the float32 guard-adjacent tolerance.",
    ranked.length > displayed.length
      ? `Displayed ${formatNumber(displayed.length)} of ${formatNumber(ranked.length)} assessed op rows; structured evidence retains every op and all ${formatNumber(atlas.assessed_channel_count)} channel envelopes.`
      : `Displayed all ${formatNumber(ranked.length)} assessed op rows; structured evidence retains every channel envelope.`,
    unassessed.length ? markdownTable(["Operator", "Status", "Reason"], unassessed.map((row) => [
      `#${padOp(row.op_index)} ${row.op_name}`,
      row.assessment_status,
      row.not_assessed_reason,
    ])) : "No candidate accumulator row was left unassessed.",
    `Method: ${atlas.method}`,
    `> ${atlas.interpretation_boundary}`,
  ].join("\n\n");
}

export function quantResearchEvidenceMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) return "";
  const coverage = analysis?.quant_research_coverage || buildQuantResearchCoverage(analysis);
  const coverageRows = coverage.labs.map((row) => [
    row.label,
    row.status.toUpperCase(),
    row.reason_code,
  ]);
  const coverageSection = [
    "## Quantization Research Coverage (DERIVED)",
    markdownTable(["Field", "Value"], [
      ["Artifact class", `${coverage.artifact_class_label} (${coverage.artifact_class})`],
      ["Class-supported lab envelope", `${formatNumber(coverage.class_supported_lab_count)} / ${formatNumber(coverage.lab_count)}`],
      ["Artifact-applicable labs", formatNumber(coverage.artifact_applicable_lab_count)],
      ["Assessed / partial / not assessed", `${formatNumber(coverage.assessed_lab_count)} / ${formatNumber(coverage.partial_lab_count)} / ${formatNumber(coverage.not_assessed_lab_count)}`],
      ["Excluded as not applicable", formatNumber(coverage.not_applicable_lab_count)],
      ["Classification reason", `${coverage.artifact_class_reason_code}: ${coverage.artifact_class_detail}`],
    ]),
    `> Scan denominator policy: ${coverage.scan_denominator_policy}`,
  ].join("\n\n");
  if (!["full_integer", "mixed_integer"].includes(coverage.artifact_class)) {
    return [
      coverageSection,
      "### Lab Applicability Ledger",
      markdownTable(["Lab", "Status", "Reason code"], coverageRows),
      "Advanced integer activation-path evidence is intentionally suppressed for this artifact class. Empty or excluded labs are not evidence that the corresponding defect is absent.",
    ].join("\n\n");
  }
  return [
    coverageSection,
    accumulatorAtlasMarkdown(analysis),
    requantizationFidelityMarkdown(analysis),
    kernelExtremumWitnessMarkdown(analysis),
    channelVitalityMarkdown(analysis),
    roundingEquivalenceMarkdown(analysis),
    accumulatorReachabilityMarkdown(analysis),
    numericalAbiPropagationMarkdown(analysis),
    inputCounterexampleMarkdown(analysis),
    preprocessingRealizabilityMarkdown(analysis),
    quantizationLatticeMarkdown(analysis),
    contractMigrationMarkdown(analysis),
    residualStepResponseMarkdown(analysis),
    residualContractDistortionMarkdown(analysis),
  ].filter(Boolean).join("\n\n");
}

function requantizationFidelityMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) return "";
  const fidelity = analysis?.requantization_fidelity;
  if (!fidelity) return [
    "## Requantization Fidelity Lab (NOT_ASSESSED)",
    "The pinned Q0.31 multiplier/shift analysis was not emitted.",
  ].join("\n");
  const assessed = (fidelity.ops || []).filter((row) => row.assessment_status === "assessed");
  const byIndex = new Map(assessed.map((row) => [row.op_index, row]));
  const ranked = (fidelity.fidelity_ranking_op_indices || []).map((index) => byIndex.get(index)).filter(Boolean);
  const displayed = ranked.slice(0, 32);
  const unassessed = (fidelity.ops || []).filter((row) => row.assessment_status !== "assessed");
  const sci = (value) => value == null ? "N/A" : Number(value).toExponential(6);
  const bound = (value) => value == null ? "N/A" : Number(value).toFixed(9);
  const signedValue = (value) => value == null ? "N/A" : `${Number(value) >= 0 ? "+" : ""}${value}`;
  const shiftCoordinates = assessed.flatMap((row) => (row.channel_shifts || []).map((shift, channelIndex) => ({
    opIndex: row.op_index,
    opName: row.op_name,
    channelIndex,
    shift: Number(shift),
  })));
  const shiftCoordinateText = (shift) => {
    const matches = shiftCoordinates.filter((row) => row.shift === Number(shift));
    if (!matches.length) return "coordinate not emitted";
    const shown = matches.slice(0, 8).map((row) => `#${padOp(row.opIndex)} ${row.opName}/ch${row.channelIndex}`);
    return `${shown.join(", ")}${matches.length > shown.length ? `; +${formatNumber(matches.length - shown.length)} more` : ""}`;
  };
  const sourceRows = (fidelity.source_references || []).map((source) => [
    source.role,
    `[${source.file}](${source.url})`,
    code(source.sha256),
  ]);
  return [
    "## Requantization Fidelity Lab (DERIVED PINNED Q0.31 DOMAIN)",
    markdownTable(["Metric", "Value"], [
      ["Schema / method", `${fidelity.schema} / ${fidelity.method_version}`],
      ["Candidate / assessed / unassessed ops", `${formatNumber(fidelity.candidate_op_count)} / ${formatNumber(fidelity.assessed_op_count)} / ${formatNumber(fidelity.unassessed_op_count)}`],
      ["Assessed / bounded channels", `${formatNumber(fidelity.assessed_channel_count)} / ${formatNumber(fidelity.fixed_point_bound_channel_count)}`],
      ["Per-tensor / per-output-channel weight ops", `${formatNumber(fidelity.per_tensor_weight_op_count)} / ${formatNumber(fidelity.per_axis_weight_op_count)}`],
      ["Q0.31 shift range", `${signedValue(fidelity.minimum_shift)} .. ${signedValue(fidelity.maximum_shift)}`],
      ["Shift extrema coordinates", `min ${signedValue(fidelity.minimum_shift)} at ${shiftCoordinateText(fidelity.minimum_shift)}; max ${signedValue(fidelity.maximum_shift)} at ${shiftCoordinateText(fidelity.maximum_shift)}`],
      ["Maximum multiplier error", `${sci(fidelity.maximum_relative_multiplier_error)} relative / ${sci(fidelity.maximum_multiplier_error_ppm)} ppm`],
      ["Maximum encoding-only drift", `${sci(fidelity.maximum_encoding_drift_bound_codes)} output codes`],
      ["Maximum default / single pre-clamp bound", `${bound(fidelity.maximum_default_double_rounding_bound_codes)} / ${bound(fidelity.maximum_single_rounding_bound_codes)} output codes`],
      ["Default pre-shift overflow / build-mode encoding divergence channels", `${formatNumber(fidelity.default_pre_shift_overflow_channel_count)} / ${formatNumber(fidelity.single_rounding_encoding_divergence_channel_count)}`],
      ["Encoding drift >=0.5 / >=1 code channels", `${formatNumber(fidelity.half_code_encoding_drift_channel_count)} / ${formatNumber(fidelity.one_code_encoding_drift_channel_count)}`],
      ["Pinned TensorFlow source", fidelity.source_commit],
    ]),
    sourceRows.length ? `### Source Digest Basis\n\n${markdownTable(["Role", "Pinned file", "SHA-256"], sourceRows)}` : "",
    displayed.length ? markdownTable([
      "Rank", "Operator", "Channels", "Shift range", "Max error ppm", "Encoding drift", "Default / single bound", "Build divergence", "Ledger SHA-256",
    ], displayed.map((row, rank) => [
      `#${rank + 1}`,
      `#${padOp(row.op_index)} ${row.op_name}`,
      formatNumber(row.assessed_channel_count),
      `${signedValue(row.minimum_shift)} .. ${signedValue(row.maximum_shift)}`,
      sci(row.maximum_multiplier_error_ppm),
      sci(row.maximum_encoding_drift_bound_codes),
      `${bound(row.maximum_default_double_rounding_bound_codes)} / ${bound(row.maximum_single_rounding_bound_codes)}`,
      `${formatNumber(row.default_pre_shift_overflow_channel_count)} overflow / ${formatNumber(row.single_rounding_encoding_divergence_channel_count)} encoding`,
      code(row.channel_ledger_sha256),
    ])) : "No quantized convolution-family requantization row was assessed.",
    ranked.length > displayed.length
      ? `Displayed ${formatNumber(displayed.length)} of ${formatNumber(ranked.length)} assessed op rows; structured evidence retains all ${formatNumber(fidelity.assessed_channel_count)} channels.`
      : `Displayed all ${formatNumber(ranked.length)} assessed op rows; structured evidence retains every channel.`,
    unassessed.length ? markdownTable(["Operator", "Status", "Reason"], unassessed.map((row) => [
      `#${padOp(row.op_index)} ${row.op_name}`,
      row.assessment_status,
      row.not_assessed_reason,
    ])) : "No candidate requantization row was left unassessed.",
    `QuantizeMultiplier: ${fidelity.quantize_multiplier_formula}`,
    `Encoding drift: ${fidelity.encoding_drift_formula}`,
    `Rounding bound: ${fidelity.rounding_bound_formula}`,
    `Method: ${fidelity.method}`,
    `> ${fidelity.interpretation_boundary}`,
  ].filter(Boolean).join("\n\n");
}

function kernelExtremumWitnessMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) return "";
  const witness = analysis?.kernel_extremum_witness;
  if (!witness) return [
    "## Quantized Kernel Witness Lab (NOT_ASSESSED)",
    "The canonical per-channel extremum-witness analysis was not emitted.",
  ].join("\n");
  const assessed = (witness.ops || []).filter((row) => row.assessment_status === "assessed");
  const byIndex = new Map(assessed.map((row) => [row.op_index, row]));
  const ranked = (witness.witness_ranking_op_indices || []).map((index) => byIndex.get(index)).filter(Boolean);
  const displayed = ranked.slice(0, 32);
  const top = ranked[0] || null;
  const topChannel = top?.worst_channel || null;
  const endpointText = (endpoint) => endpoint
    ? `${endpoint.witness_code_histogram.map((item) => `${item.code} x ${formatNumber(item.count)}`).join(", ")}; dot ${endpoint.dot_product_decimal}; post-bias ${endpoint.post_bias_accumulator_decimal}; ideal/default/single ${endpoint.ideal_output_code ?? "N/A"}/${endpoint.default_output_code ?? "N/A"}/${endpoint.single_output_code ?? "N/A"}; default clamp ${endpoint.default_activation_clamped == null ? "N/A" : endpoint.default_activation_clamped ? "yes" : "no"}`
    : "N/A";
  const sourceRows = (witness.source_references || []).map((source) => [
    source.role,
    `[${source.file}](${source.url})`,
    code(source.sha256),
  ]);
  return [
    "## Quantized Kernel Witness Lab (DERIVED EXACT SYNTHETIC RECEPTIVE-FIELD DOMAIN)",
    markdownTable(["Metric", "Value"], [
      ["Schema / method", `${witness.schema} / ${witness.method_version}`],
      ["Candidate / assessed / unassessed ops", `${formatNumber(witness.candidate_op_count)} / ${formatNumber(witness.assessed_op_count)} / ${formatNumber(witness.unassessed_op_count)}`],
      ["Assessed / fixed-point projected channels", `${formatNumber(witness.assessed_channel_count)} / ${formatNumber(witness.fixed_point_assessed_channel_count)}`],
      ["Canonical min/max term assignments", formatNumber(witness.witness_assignment_count)],
      ["Default + single fixed-point endpoint executions", formatNumber(witness.fixed_point_endpoint_evaluation_count)],
      ["Default / single mismatch versus direct ideal projection", `${formatNumber(witness.default_ideal_mismatch_endpoint_count)} / ${formatNumber(witness.single_ideal_mismatch_endpoint_count)} endpoint(s)`],
      ["Default versus single output-code differences", `${formatNumber(witness.build_mode_divergent_endpoint_count)} endpoint(s); maximum default ideal delta ${witness.maximum_default_ideal_delta_codes ?? "N/A"} code(s)`],
      ["Default / single activation-clamped endpoints", `${formatNumber(witness.default_activation_clamped_endpoint_count)} / ${formatNumber(witness.single_activation_clamped_endpoint_count)}`],
      ["Default / single collapsed extremum spans", `${formatNumber(witness.default_collapsed_extrema_channel_count)} / ${formatNumber(witness.single_collapsed_extrema_channel_count)} channel(s)`],
      ["Runtime build flag", "TFLITE_SINGLE_ROUNDING not embedded in artifact"],
      ["Pinned TensorFlow source", witness.source_commit],
    ]),
    sourceRows.length ? `### Source Digest Basis\n\n${markdownTable(["Role", "Pinned file", "SHA-256"], sourceRows)}` : "",
    displayed.length ? markdownTable([
      "Rank", "Operator", "Channels x terms", "Assignments", "Default/ideal", "Build-mode delta", "Default clamps", "Collapsed spans", "Witness ledger SHA-256",
    ], displayed.map((row, rank) => [
      `#${rank + 1}`,
      `#${padOp(row.op_index)} ${row.op_name}`,
      `${formatNumber(row.assessed_channel_count)} x ${formatNumber(row.accumulation_terms_per_channel)}`,
      formatNumber(row.witness_assignment_count),
      formatNumber(row.default_ideal_mismatch_endpoint_count),
      formatNumber(row.build_mode_divergent_endpoint_count),
      formatNumber(row.default_activation_clamped_endpoint_count),
      formatNumber(row.default_collapsed_extrema_channel_count),
      code(row.witness_ledger_sha256),
    ])) : "No kernel witness row was assessed.",
    topChannel ? `### Highest-ranked Exact Channel Witness\n\n${markdownTable(["Field", "Value"], [
      ["Coordinate", `#${padOp(top.op_index)} ${top.op_name} / output channel ${topChannel.channel_index}`],
      ["Positive / negative / zero centered-weight terms", `${formatNumber(topChannel.positive_centered_weight_count)} / ${formatNumber(topChannel.negative_centered_weight_count)} / ${formatNumber(topChannel.zero_centered_weight_count)}`],
      ["Canonical minimum", endpointText(topChannel.minimum)],
      ["Canonical maximum", endpointText(topChannel.maximum)],
      ["Pattern SHA-256", code(topChannel.witness_pattern_sha256)],
    ])}` : "",
    `Ledger: ${top?.ledger_hash_method || "not emitted"}`,
    `Method: ${witness.method}`,
    `> ${witness.interpretation_boundary}`,
  ].filter(Boolean).join("\n\n");
}

function channelVitalityMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) return "";
  const vitality = analysis?.channel_vitality;
  if (!vitality) return [
    "## Quantized Channel Vitality Atlas (NOT_ASSESSED)",
    "The per-channel fixed-point vitality proof was not emitted.",
  ].join("\n");
  const assessed = (vitality.ops || []).filter((row) => row.assessment_status === "assessed");
  const byIndex = new Map(assessed.map((row) => [row.op_index, row]));
  const ranked = (vitality.vitality_ranking_op_indices || []).map((index) => byIndex.get(index)).filter(Boolean);
  const displayed = ranked.slice(0, 32);
  const topChannels = ranked.flatMap((row) => (row.top_channels || []).map((channel) => ({ row, channel })))
    .filter(({ channel }) => channel.dual_mode_constant || channel.mode_dependent_constant || Number(channel.default_inclusive_code_span) <= 15)
    .slice(0, 32);
  const sourceRows = (vitality.source_references || []).map((source) => [
    source.role,
    `[${source.file}](${source.url})`,
    code(source.sha256),
  ]);
  const range = (minimumValue, maximumValue, span) => `${minimumValue ?? "N/A"} .. ${maximumValue ?? "N/A"} (span ${span ?? "N/A"})`;
  return [
    "## Quantized Channel Vitality Atlas (DERIVED EXACT MONOTONE ENDPOINT PROOF)",
    markdownTable(["Metric", "Value"], [
      ["Schema / method", `${vitality.schema} / ${vitality.method_version}`],
      ["Candidate / assessed / unassessed ops", `${formatNumber(vitality.candidate_op_count)} / ${formatNumber(vitality.assessed_op_count)} / ${formatNumber(vitality.unassessed_op_count)}`],
      ["Assessed / dual-mode fixed-point channels", `${formatNumber(vitality.assessed_channel_count)} / ${formatNumber(vitality.fixed_point_assessed_channel_count)}`],
      ["Constant post-bias accumulator channels", formatNumber(vitality.constant_accumulator_channel_count)],
      ["Default / single constant-output channels", `${formatNumber(vitality.default_constant_output_channel_count)} / ${formatNumber(vitality.single_constant_output_channel_count)}`],
      ["Constant under both pinned build paths", formatNumber(vitality.dual_mode_constant_output_channel_count)],
      ["Variable accumulator but constant under both paths", formatNumber(vitality.nonconstant_accumulator_dual_mode_constant_channel_count)],
      ["Build-mode-dependent constant classification", formatNumber(vitality.mode_dependent_constant_output_channel_count)],
      ["Post-bias sign lock", `${formatNumber(vitality.post_bias_negative_locked_channel_count)} negative / ${formatNumber(vitality.post_bias_positive_locked_channel_count)} positive / ${formatNumber(vitality.post_bias_zero_containing_channel_count)} zero-containing`],
      ["Default / single span <=15", `${formatNumber(vitality.default_severely_constrained_channel_count)} / ${formatNumber(vitality.single_severely_constrained_channel_count)}`],
      ["Default / single full activation span", `${formatNumber(vitality.default_full_activation_span_channel_count)} / ${formatNumber(vitality.single_full_activation_span_channel_count)}`],
      ["Portfolio minimum inclusive span default / single", `${formatNumber(vitality.minimum_default_inclusive_code_span)} / ${formatNumber(vitality.minimum_single_inclusive_code_span)}`],
      ["Runtime build flag", "TFLITE_SINGLE_ROUNDING not embedded in artifact"],
      ["Pinned TensorFlow source", vitality.source_commit],
    ]),
    markdownTable(["Inclusive span", "Default channels", "Single-rounding channels"], (vitality.span_histogram || []).map((bin) => [
      bin.label,
      formatNumber(bin.default_channel_count),
      formatNumber(bin.single_rounding_channel_count),
    ])),
    sourceRows.length ? `### Source Digest Basis\n\n${markdownTable(["Role", "Pinned file", "SHA-256"], sourceRows)}` : "",
    displayed.length ? markdownTable([
      "Rank", "Operator", "Channels", "Constant accumulator", "Dual constant", "Variable constant", "Mode-dependent", "Minimum span D / S", "Vitality ledger SHA-256",
    ], displayed.map((row, rank) => [
      `#${rank + 1}`,
      `#${padOp(row.op_index)} ${row.op_name}`,
      formatNumber(row.assessed_channel_count),
      formatNumber(row.constant_accumulator_channel_count),
      formatNumber(row.dual_mode_constant_output_channel_count),
      formatNumber(row.nonconstant_accumulator_dual_mode_constant_channel_count),
      formatNumber(row.mode_dependent_constant_output_channel_count),
      `${row.minimum_default_inclusive_code_span ?? "N/A"} / ${row.minimum_single_inclusive_code_span ?? "N/A"}`,
      code(row.vitality_ledger_sha256),
    ])) : "No channel-vitality row was assessed.",
    topChannels.length ? `### Exact Channel Proof Coordinates\n\n${markdownTable([
      "Coordinate", "Post-bias interval", "Sign", "Default preclamp", "Default output hull", "Default cause", "Single preclamp", "Single output hull", "Single cause", "Mode status",
    ], topChannels.map(({ row, channel }) => [
      `#${padOp(row.op_index)} ${row.op_name} / ch ${channel.channel_index}`,
      `${channel.post_bias_minimum_decimal} .. ${channel.post_bias_maximum_decimal}`,
      channel.post_bias_sign_class,
      `${channel.default_minimum_preclamp_code ?? "N/A"} .. ${channel.default_maximum_preclamp_code ?? "N/A"}`,
      range(channel.default_minimum_output_code, channel.default_maximum_output_code, channel.default_inclusive_code_span),
      channel.default_constant_reason,
      `${channel.single_minimum_preclamp_code ?? "N/A"} .. ${channel.single_maximum_preclamp_code ?? "N/A"}`,
      range(channel.single_minimum_output_code, channel.single_maximum_output_code, channel.single_inclusive_code_span),
      channel.single_constant_reason,
      channel.mode_dependent_constant ? "build-mode-dependent" : channel.dual_mode_constant ? "dual-mode constant" : "nonconstant",
    ]))}` : "",
    `Constant proof: ${vitality.constant_proof}`,
    `Span definition: ${vitality.span_definition}`,
    `Method: ${vitality.method}`,
    `> ${vitality.interpretation_boundary}`,
  ].filter(Boolean).join("\n\n");
}

function roundingEquivalenceMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) return "";
  const equivalence = analysis?.rounding_equivalence;
  if (!equivalence) return [
    "## Fixed-Point Rounding Equivalence Lab (NOT_ASSESSED)",
    "The closed-interval default-versus-TFLITE_SINGLE_ROUNDING equivalence certificate was not emitted.",
  ].join("\n");
  const assessed = (equivalence.ops || []).filter((row) => row.assessment_status === "assessed");
  const byIndex = new Map(assessed.map((row) => [row.op_index, row]));
  const ranked = (equivalence.equivalence_ranking_op_indices || []).map((index) => byIndex.get(index)).filter(Boolean);
  const displayed = ranked.slice(0, 32);
  const topChannels = displayed.flatMap((row) => (row.top_channels || []).slice(0, 2).map((channel) => ({ row, channel }))).slice(0, 24);
  const sourceRows = (equivalence.source_references || []).map((source) => [source.role, `[${source.file}](${source.url})`, code(source.sha256)]);
  const divergentStates = Number(equivalence.divergent_state_count_decimal || 0);
  const defaultHigherStates = Number(equivalence.default_higher_state_count_decimal || 0);
  const defaultHigherAmongDivergent = divergentStates > 0 ? defaultHigherStates / divergentStates : null;
  return [
    "## Fixed-Point Rounding Equivalence Lab (DERIVED EXACT CLOSED-INTERVAL CERTIFICATE)",
    markdownTable(["Metric", "Value"], [
      ["Schema / method", `${equivalence.schema} / ${equivalence.method_version}`],
      ["Candidate / assessed / unassessed ops", `${formatNumber(equivalence.candidate_op_count)} / ${formatNumber(equivalence.assessed_op_count)} / ${formatNumber(equivalence.unassessed_op_count)}`],
      ["Assessed channels", formatNumber(equivalence.assessed_channel_count)],
      ["Equivalent / divergent channels", `${formatNumber(equivalence.equivalent_channel_count)} / ${formatNumber(equivalence.divergent_channel_count)}`],
      ["Ops containing divergence", formatNumber(equivalence.divergent_op_count)],
      ["Complete interval states", formatNumber(equivalence.interval_state_count_decimal)],
      ["Divergent interval states", `${formatNumber(equivalence.divergent_state_count_decimal)} (${formatPercent(equivalence.divergent_state_ratio)})`],
      ["Default lower / higher states", `${formatNumber(equivalence.default_lower_state_count_decimal)} / ${formatNumber(equivalence.default_higher_state_count_decimal)}`],
      ["Directional skew among divergent states", defaultHigherAmongDivergent == null ? "N/A; no divergent state" : `${formatPercent(defaultHigherAmongDivergent)} default-higher / ${formatPercent(1 - defaultHigherAmongDivergent)} default-lower`],
      ["Maximum absolute output delta", `${formatNumber(equivalence.maximum_absolute_output_delta || 0)} code`],
      ["Exact pair segments / divergent regions", `${formatNumber(equivalence.pair_segment_count)} / ${formatNumber(equivalence.divergent_region_count)}`],
      ["Pinned TensorFlow commit", code(equivalence.source_commit)],
    ]),
    markdownTable(["Divergence ratio", "Channels", "Interval states", "Divergent states"], (equivalence.divergence_histogram || []).map((bin) => [
      bin.label,
      formatNumber(bin.channel_count),
      formatNumber(bin.interval_state_count_decimal),
      formatNumber(bin.divergent_state_count_decimal),
    ])),
    displayed.length ? markdownTable([
      "Rank", "Operator", "Channels D / total", "Divergent states", "Ratio", "Default lower / higher", "Max delta", "Max segments / regions", "Certificate SHA-256",
    ], displayed.map((row, rank) => [
      `#${rank + 1}`,
      `#${padOp(row.op_index)} ${row.op_name}`,
      `${formatNumber(row.divergent_channel_count)} / ${formatNumber(row.assessed_channel_count)}`,
      formatNumber(row.divergent_state_count_decimal),
      formatPercent(row.divergent_state_ratio),
      `${formatNumber(row.default_lower_state_count_decimal)} / ${formatNumber(row.default_higher_state_count_decimal)}`,
      formatNumber(row.maximum_absolute_output_delta || 0),
      `${formatNumber(row.maximum_pair_segment_count || 0)} / ${formatNumber(row.maximum_divergent_region_count || 0)}`,
      code(row.equivalence_ledger_sha256),
    ])) : "No rounding-equivalence row was assessed.",
    ranked.length > displayed.length
      ? `Displayed ${formatNumber(displayed.length)} of ${formatNumber(ranked.length)} assessed op rows; the structured ledger retains all rows. The 80% attribution prefix is computed over the complete ranking, not only the displayed prefix.`
      : `Displayed all ${formatNumber(ranked.length)} assessed op rows.`,
    topChannels.length ? `### Exact Build-Mode Counterexamples\n\n${markdownTable([
      "Coordinate", "Accumulator interval", "States D / total", "Ratio", "First accumulator", "First default / single", "Last accumulator", "Max delta", "Encoding q,shift D / S",
    ], topChannels.map(({ row, channel }) => [
      `#${padOp(row.op_index)} ${row.op_name} / ch ${channel.channel_index}`,
      `${channel.post_bias_minimum_decimal} .. ${channel.post_bias_maximum_decimal}`,
      `${formatNumber(channel.divergent_state_count_decimal)} / ${formatNumber(channel.interval_state_count_decimal)}`,
      formatPercent(channel.divergent_state_ratio),
      channel.first_divergent_accumulator_decimal ?? "none (equivalent)",
      channel.first_divergent_accumulator_decimal == null ? "equal for complete interval" : `${channel.first_default_output_code} / ${channel.first_single_output_code}`,
      channel.last_divergent_accumulator_decimal ?? "none",
      formatNumber(channel.maximum_absolute_output_delta),
      `${channel.default_quantized_multiplier},${channel.default_shift} / ${channel.single_quantized_multiplier},${channel.single_shift}`,
    ]))}` : "",
    sourceRows.length ? `### Source Digest Basis\n\n${markdownTable(["Role", "Pinned file", "SHA-256"], sourceRows)}` : "",
    `Equivalence proof: ${equivalence.equivalence_proof}`,
    `Segmentation bound: ${equivalence.segmentation_bound}`,
    "Direction interpretation: `default higher` means the pinned default double-rounding equation emits an output code one greater than the pinned single-rounding equation for that exact post-bias integer state. With positive Q0.31 multipliers and right shifts, the double-versus-single difference is sign-dependent, while artifact post-bias intervals need not be sign-symmetric; a strongly one-sided exact state count is therefore possible without an implementation error. The directional ratio is exact for the uniformly counted interval-hull state portfolio; it is not an activation-weighted runtime bias, output-distribution shift, or accuracy estimate.",
    `Method: ${equivalence.method}`,
    `> ${equivalence.interpretation_boundary}`,
  ].filter(Boolean).join("\n\n");
}

function accumulatorReachabilityMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) return "";
  const reachability = analysis?.accumulator_reachability;
  if (!reachability) return [
    "## Accumulator Reachability Lattice (NOT_ASSESSED)",
    "The bounded-sum kernel-local accumulator reachability certificate was not emitted.",
  ].join("\n");
  const assessed = (reachability.ops || []).filter((row) => row.assessment_status === "assessed");
  const byIndex = new Map(assessed.map((row) => [row.op_index, row]));
  const ranked = (reachability.reachability_ranking_op_indices || []).map((index) => byIndex.get(index)).filter(Boolean);
  const displayed = ranked.slice(0, 32);
  const topChannels = displayed.flatMap((row) => (row.top_channels || []).slice(0, 2).map((channel) => ({ row, channel }))).slice(0, 24);
  const sourceRows = (reachability.source_references || []).map((source) => [source.role, `[${source.file}](${source.url})`, code(source.sha256)]);
  const channelPartition = reachabilityDivergenceChannelPartition(reachability);
  return [
    "## Accumulator Reachability Lattice (DERIVED EXACT KERNEL-LOCAL CERTIFICATE)",
    markdownTable(["Metric", "Value"], [
      ["Schema / method", `${reachability.schema} / ${reachability.method_version}`],
      ["Candidate / assessed / unassessed ops", `${formatNumber(reachability.candidate_op_count)} / ${formatNumber(reachability.assessed_op_count)} / ${formatNumber(reachability.unassessed_op_count)}`],
      ["Assessed channels", formatNumber(reachability.assessed_channel_count)],
      ["Complete integer / modular lattice channels", `${formatNumber(reachability.complete_integer_interval_channel_count)} / ${formatNumber(reachability.complete_modular_lattice_channel_count)}`],
      ["Partial endpoint-band / singleton channels", `${formatNumber(reachability.partial_band_channel_count)} / ${formatNumber(reachability.singleton_channel_count)}`],
      ["Interval states", formatNumber(reachability.interval_state_count_decimal)],
      ["Certified reachable / residue-excluded / unresolved states", `${formatNumber(reachability.certified_reachable_state_count_decimal)} / ${formatNumber(reachability.provably_unreachable_state_count_decimal)} / ${formatNumber(reachability.unresolved_state_count_decimal)}`],
      ["Lattice-compatible interval states", formatNumber(reachability.lattice_compatible_state_count_decimal)],
      ["Interval divergent states", formatNumber(reachability.interval_divergent_state_count_decimal)],
      ["Exact reachable / residue-excluded / unresolved divergent states", `${formatNumber(reachability.exact_reachable_divergent_state_count_decimal)} / ${formatNumber(reachability.provably_unreachable_divergent_state_count_decimal)} / ${formatNumber(reachability.unresolved_divergent_state_count_decimal)}`],
      ["Exact reachable divergent ratio", formatPercent(reachability.exact_reachable_divergent_ratio)],
      ["Divergent-channel sets (non-exclusive)", `${formatNumber(reachability.exact_reachable_divergent_channel_count)} exact / ${formatNumber(reachability.unresolved_divergent_channel_count)} unresolved; overlap ${formatNumber(channelPartition.both)}`],
      ["Exclusive assessed-channel partition", `${formatNumber(channelPartition.exactOnly)} exact-only + ${formatNumber(channelPartition.unresolvedOnly)} unresolved-only + ${formatNumber(channelPartition.both)} both + ${formatNumber(channelPartition.neither)} neither = ${formatNumber(channelPartition.total)}`],
      ["Interval-divergent but no certified reachable difference", formatNumber(reachability.interval_only_divergent_channel_count)],
      ["Maximum lattice GCD", formatNumber(reachability.maximum_lattice_gcd || 0)],
      ["Pinned TensorFlow commit", code(reachability.source_commit)],
    ]),
    displayed.length ? markdownTable([
      "Rank", "Operator", "Proof channels I / M / P / S", "Exact divergent channels", "Divergent exact / excluded / unresolved", "Max GCD", "Reachability SHA-256",
    ], displayed.map((row, rank) => [
      `#${rank + 1}`,
      `#${padOp(row.op_index)} ${row.op_name}`,
      `${formatNumber(row.complete_integer_interval_channel_count)} / ${formatNumber(row.complete_modular_lattice_channel_count)} / ${formatNumber(row.partial_band_channel_count)} / ${formatNumber(row.singleton_channel_count)}`,
      `${formatNumber(row.exact_reachable_divergent_channel_count)} / ${formatNumber(row.assessed_channel_count)}`,
      `${formatNumber(row.exact_reachable_divergent_state_count_decimal)} / ${formatNumber(row.provably_unreachable_divergent_state_count_decimal)} / ${formatNumber(row.unresolved_divergent_state_count_decimal)}`,
      formatNumber(row.maximum_lattice_gcd || 0),
      code(row.reachability_ledger_sha256),
    ])) : "No accumulator-reachability row was assessed.",
    topChannels.length ? `### Constructive Kernel-Local Counterexamples\n\n${markdownTable([
      "Coordinate", "Proof", "GCD", "States certified / excluded / unresolved", "Divergent exact / excluded / unresolved", "First exact accumulator", "Default / single", "Denomination / witness groups",
    ], topChannels.map(({ row, channel }) => [
      `#${padOp(row.op_index)} ${row.op_name} / ch ${channel.channel_index}`,
      channel.proof_status,
      formatNumber(channel.lattice_gcd),
      `${formatNumber(channel.certified_reachable_state_count_decimal)} / ${formatNumber(channel.provably_unreachable_state_count_decimal)} / ${formatNumber(channel.unresolved_state_count_decimal)}`,
      `${formatNumber(channel.exact_reachable_divergent_state_count_decimal)} / ${formatNumber(channel.provably_unreachable_divergent_state_count_decimal)} / ${formatNumber(channel.unresolved_divergent_state_count_decimal)}`,
      channel.first_exact_reachable_divergent_accumulator_decimal ?? "none",
      channel.first_exact_reachable_divergent_accumulator_decimal == null ? "none" : `${channel.first_default_output_code} / ${channel.first_single_output_code}`,
      `${formatNumber(channel.denomination_group_count)} / ${formatNumber(channel.first_exact_reachable_witness_group_count)}`,
    ]))}` : "",
    sourceRows.length ? `### Source Digest Basis\n\n${markdownTable(["Role", "Pinned file", "SHA-256"], sourceRows)}` : "",
    `Bounded-sum proof: ${reachability.bounded_sum_proof}`,
    `State conservation: ${reachability.state_conservation}`,
    `Method: ${reachability.method}`,
    `> ${reachability.interpretation_boundary}`,
  ].filter(Boolean).join("\n\n");
}

function numericalAbiPropagationMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) return "";
  const propagation = analysis?.numerical_abi_propagation;
  if (!propagation) return [
    "## Numerical ABI Propagation Atlas (NOT_ASSESSED)",
    "The tensor-level structural propagation certificate was not emitted.",
  ].join("\n");
  const sourceByIndex = new Map((propagation.sources || []).map((source) => [source.op_index, source]));
  const ranked = (propagation.propagation_ranking_op_indices || []).map((index) => sourceByIndex.get(index)).filter(Boolean);
  const displayed = ranked.slice(0, 32);
  const topSource = displayed.find((source) => source.local_reachability_status === "exact_local_counterexample")
    || displayed.find((source) => source.assessment_status === "propagates_structurally") || null;
  const topPaths = (topSource?.model_output_paths || []).slice(0, 16);
  const topMerges = (topSource?.merge_points || []).slice(0, 24);
  const edgeByIndex = new Map((propagation.graph_edges || []).map((edge) => [edge.edge_index, edge]));
  const boundaryEdges = (topSource?.corridor_edge_indices || [])
    .map((index) => edgeByIndex.get(index))
    .filter((edge) => edge?.predicted_boundary);
  const bytesOrStatus = (value, status = "") => value == null ? (status || "not assessed") : `${formatBytes(value)} (${formatNumber(value)} B)`;
  return [
    "## Numerical ABI Propagation Atlas (DERIVED EXACT LOCAL SOURCE + STRUCTURAL CORRIDOR)",
    markdownTable(["Metric", "Value"], [
      ["Schema / method", `${propagation.schema} / ${propagation.method_version}`],
      ["Candidate / divergent / equivalent / unassessed sources", `${formatNumber(propagation.candidate_source_op_count)} / ${formatNumber(propagation.divergent_source_op_count)} / ${formatNumber(propagation.equivalent_source_op_count)} / ${formatNumber(propagation.unassessed_source_op_count)}`],
      ["Output-reachable / isolated divergent sources", `${formatNumber(propagation.output_reachable_source_op_count)} / ${formatNumber(propagation.output_isolated_source_op_count)}`],
      ["Exact-local / residue-facet / unresolved-facet sources", `${formatNumber(propagation.exact_local_counterexample_source_op_count)} / ${formatNumber(propagation.residue_excluded_divergence_source_op_count)} / ${formatNumber(propagation.unresolved_divergence_source_op_count)}`],
      ["Interval divergent states = exact local + residue-excluded + unresolved", `${formatNumber(propagation.interval_divergent_state_count_decimal)} = ${formatNumber(propagation.exact_local_divergent_state_count_decimal)} + ${formatNumber(propagation.residue_excluded_divergent_state_count_decimal)} + ${formatNumber(propagation.unresolved_divergent_state_count_decimal)}`],
      ["Exact-local output-reachable / local-unassessed sources", `${formatNumber(propagation.exact_output_reachable_source_op_count)} / ${formatNumber(propagation.local_reachability_unassessed_source_op_count)}`],
      ["Graph edges / ledger SHA-256", `${formatNumber(propagation.graph_edge_count)} / ${code(propagation.graph_ledger_sha256)}`],
      ["Source-corridor edge instances", formatNumber(propagation.source_corridor_edge_instance_count)],
      ["Union reachable ops / tensors / declared outputs", `${formatNumber(propagation.unique_reachable_op_count)} / ${formatNumber(propagation.unique_reachable_tensor_count)} / ${formatNumber(propagation.unique_model_output_tensor_count)}`],
      ["Unique predicted boundary edges / logical payload", `${formatNumber(propagation.unique_predicted_boundary_edge_count)} / ${bytesOrStatus(propagation.unique_predicted_boundary_logical_payload_bytes)}`],
      ["Exact-qualified corridor instances / union ops / tensors", `${formatNumber(propagation.exact_source_corridor_edge_instance_count)} / ${formatNumber(propagation.exact_unique_reachable_op_count)} / ${formatNumber(propagation.exact_unique_reachable_tensor_count)}`],
      ["Exact-source boundary instances / unassessed payload instances", `${formatNumber(propagation.exact_source_boundary_edge_instance_count)} / ${formatNumber(propagation.unassessed_source_boundary_edge_instance_payload_count)}`],
      ["Exact-qualified unique boundary edges / logical payload", `${formatNumber(propagation.exact_unique_predicted_boundary_edge_count)} / ${bytesOrStatus(propagation.exact_unique_predicted_boundary_logical_payload_bytes)}`],
      ["Per-source boundary edge instances", formatNumber(propagation.source_boundary_edge_instance_count)],
      ["Summed per-source boundary-instance payload inventory", `${formatNumber(propagation.assessed_source_boundary_edge_instance_payload_bytes_decimal)} B; repeated source exposure inventory, not physical runtime traffic`],
      ["Reconvergence / single-branch merge source-op instances", `${formatNumber(propagation.reconvergence_source_op_instance_count)} / ${formatNumber(propagation.single_branch_merge_source_op_instance_count)}`],
      ["Maximum shortest output hops / exact graph routes", `${formatNumber(propagation.maximum_model_output_op_hops || 0)} / ${formatNumber(propagation.maximum_model_output_graph_route_count_decimal || 0)}`],
      ["Graph cycle status", propagation.graph_cycle_status],
    ]),
    displayed.length ? markdownTable([
      "Rank", "Source operator", "Local proof", "Divergent states exact / excluded / unresolved", "Reachable ops / tensors", "Output routes", "Merges R / S", "Boundaries / logical payload", "Reachability / propagation SHA-256",
    ], displayed.map((source, rank) => [
      `#${rank + 1}`,
      `#${padOp(source.op_index)} ${source.op_name}`,
      source.local_reachability_status,
      `${formatNumber(source.exact_reachable_divergent_state_count_decimal)} / ${formatNumber(source.provably_unreachable_divergent_state_count_decimal)} / ${formatNumber(source.unresolved_divergent_state_count_decimal)}`,
      `${formatNumber(source.reachable_op_count)} / ${formatNumber(source.reachable_tensor_count)}`,
      source.exact_model_output_graph_route_count_decimal ?? source.route_count_status,
      `${formatNumber(source.reconvergence_op_count)} / ${formatNumber(source.single_branch_merge_op_count)}`,
      `${formatNumber(source.predicted_boundary_edge_count)} / ${bytesOrStatus(source.assessed_boundary_logical_payload_bytes, source.unassessed_boundary_payload_edge_count ? "partially assessed" : "0 B")}`,
      `${code(source.source_reachability_ledger_sha256)} / ${code(source.propagation_ledger_sha256)}`,
    ])) : "No divergent source was structurally assessed.",
    topPaths.length ? `### Top Source Declared-Output Paths\n\n${markdownTable([
      "Output tensor", "Shortest hops", "Exact graph routes", "Shortest path ops", "Boundary edges / logical payload",
    ], topPaths.map((path) => [
      `T${path.output_tensor_index} ${path.output_tensor_name}`,
      formatNumber(path.shortest_op_hops),
      path.exact_graph_route_count_decimal ?? path.route_count_status,
      (path.shortest_path_op_indices || []).map((index) => `#${padOp(index)}`).join(" -> "),
      `${formatNumber(path.shortest_path_predicted_boundary_count)} / ${bytesOrStatus(path.shortest_path_boundary_logical_payload_bytes)}`,
    ]))}` : "",
    topMerges.length ? `### Top Source Merge Ledger\n\n${markdownTable([
      "Operator", "Class", "Minimum hops", "Influenced inputs", "Uninfluenced inputs", "Execution domain",
    ], topMerges.map((merge) => [
      `#${padOp(merge.op_index)} ${merge.op_name}`,
      merge.merge_class,
      formatNumber(merge.minimum_op_hops),
      (merge.influenced_input_tensor_indices || []).map((index) => `T${index}`).join(", ") || "none",
      (merge.uninfluenced_input_tensor_indices || []).map((index) => `T${index}`).join(", ") || "none",
      merge.predicted_execution_domain,
    ]))}` : "",
    boundaryEdges.length ? `### Top Source Predicted Boundary Ledger\n\n${markdownTable([
      "Edge", "Producer", "Tensor", "Consumer", "Direction", "Logical payload",
    ], boundaryEdges.map((edge) => [
      `E${edge.edge_index}`,
      `#${padOp(edge.producer_op_index)} ${edge.producer_op_name}`,
      `T${edge.tensor_index} ${edge.tensor_name}`,
      `#${padOp(edge.consumer_op_index)} ${edge.consumer_op_name}`,
      edge.predicted_boundary_direction,
      bytesOrStatus(edge.logical_payload_bytes, edge.payload_status),
    ]))}` : "",
    `Route definition: ${propagation.route_definition}`,
    `Ranking definition: ${propagation.ranking_definition}`,
    `Method: ${propagation.method}`,
    `> ${propagation.interpretation_boundary}`,
  ].filter(Boolean).join("\n\n");
}

function inputCounterexampleMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) return "";
  const evidence = analysis?.input_counterexample;
  if (!evidence) return [
    "## Model Input Tensor ABI Witness (NOT_ASSESSED)",
    "The complete model-input counterexample constructor did not emit evidence.",
  ].join("\n");
  const witness = evidence.witnesses?.[0] || null;
  const heading = evidence.status === "not_applicable"
    ? "## Model Input Tensor ABI Witness (NOT_APPLICABLE)"
    : witness ? "## Model Input Tensor ABI Witness (DERIVED CONSTRUCTIVE EXISTENCE CERTIFICATE)"
      : "## Model Input Tensor ABI Witness (DERIVED SOURCE CLASSIFICATION; NO CONSTRUCTIVE WITNESS)";
  const source = witness ? (evidence.sources || []).find((row) => row.op_index === witness.source_op_index) : null;
  const sourceRows = (evidence.sources || []).slice(0, 52).map((row) => [
    `#${padOp(row.op_index)} ${row.op_name}`,
    row.input_tensor_index == null ? "not assessed" : `T${row.input_tensor_index} ${row.input_tensor_name}`,
    row.input_origin,
    row.classification,
    `${formatNumber(row.exact_reachable_divergent_channel_count)} / ${formatNumber(row.exact_reachable_divergent_state_count_decimal)}`,
    row.exact_model_output_graph_route_count_decimal ?? "not assessed",
    row.representative_witness_ledger_sha256 ? code(row.representative_witness_ledger_sha256) : "none",
  ]);
  const termRows = (witness?.terms || []).map((term) => [
    term.term_index,
    term.kernel_coordinate.join(","),
    term.input_coordinate.join(","),
    formatNumber(term.input_linear_index),
    term.input_code,
    term.centered_input_code,
    term.centered_weight,
    formatNumber(term.term_product_decimal),
  ]);
  return [
    heading,
    markdownTable(["Metric", "Value"], [
      ["Schema / method", `${evidence.schema} / ${evidence.method_version}`],
      ["Exact-local source classification", evidence.source_classification_conservation],
      ["Direct model-input / constructive sources", `${formatNumber(evidence.direct_model_input_source_op_count)} / ${formatNumber(evidence.tensor_abi_constructive_source_op_count)}`],
      ["Constructive channels / exact divergent states", `${formatNumber(evidence.tensor_abi_constructive_channel_count)} / ${formatNumber(evidence.tensor_abi_constructive_divergent_state_count_decimal)}`],
      ["Output-reachable constructive sources", formatNumber(evidence.output_reachable_constructive_source_op_count)],
      ["Representative witnesses", formatNumber(evidence.representative_witness_count)],
      ["Portfolio ledger SHA-256", code(evidence.portfolio_ledger_sha256)],
    ]),
    witness ? markdownTable(["Constructive witness", "Value"], [
      ["Source / channel / output coordinate", `#${padOp(witness.source_op_index)} ${witness.source_op_name} / channel ${witness.source_channel_index} / [${witness.source_output_coordinate.join(", ")}]`],
      ["Complete model-input tensor", `T${witness.model_input_tensor_index} ${witness.model_input_tensor_name}; ${witness.model_input_shape.join(" x ")} ${witness.model_input_dtype}; scale ${Number(witness.model_input_scale).toPrecision(9)}, zero point ${witness.model_input_zero_point}`],
      ["Construction", `${formatNumber(witness.model_input_element_count)} elements filled with code ${witness.full_tensor_fill_code}, then ${formatNumber(witness.sparse_override_count)} unique sparse overrides`],
      ["Full tensor SHA-256", code(witness.full_model_input_tensor_sha256)],
      ["Kernel / patch / geometry", `${witness.kernel_shape.join(" x ")} kernel; ${witness.effective_patch_shape.join(" x ")} patch at [${witness.patch_origin_yx.join(", ")}]; stride ${witness.stride_hw.join(" x ")}; dilation ${witness.dilation_hw.join(" x ")}; ${witness.padding}`],
      ["Patch codes HWC", witness.patch_codes_hwc.join(", ")],
      ["Exact dot + bias = accumulator", `${formatNumber(witness.dot_product_decimal)} + ${formatNumber(witness.bias_decimal)} = ${formatNumber(witness.post_bias_accumulator_decimal)}`],
      ["Pinned output codes / delta", `default ${witness.default_output_code} / single-rounding ${witness.single_rounding_output_code} / ${witness.output_code_delta > 0 ? "+" : ""}${witness.output_code_delta}`],
      ["Structural declared-output routes", source?.exact_model_output_graph_route_count_decimal ?? "not assessed"],
      ["Reachability / propagation / witness SHA-256", `${code(witness.source_reachability_ledger_sha256)} / ${code(witness.source_propagation_ledger_sha256)} / ${code(witness.witness_ledger_sha256)}`],
    ]) : "No complete model-input tensor witness was constructed.",
    termRows.length ? `### Exact Kernel Term Ledger\n\n${markdownTable(["Term", "Kernel HWC", "Input NHWC", "Linear", "Code", "Centered input", "Centered weight", "Product"], termRows)}` : "",
    sourceRows.length ? `### Exact-Local Source Realizability Portfolio\n\n${markdownTable(["Source", "Input tensor", "Origin", "Classification", "Exact channels / states", "Output routes", "Witness SHA-256"], sourceRows)}` : "",
    `Proof scope: ${evidence.proof_scope}`,
    `Method: ${evidence.method}`,
    `> ${evidence.interpretation_boundary}`,
  ].filter(Boolean).join("\n\n");
}

function preprocessingRealizabilityMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) return "";
  const evidence = analysis?.preprocessing_realizability;
  if (!evidence) return [
    "## Pixel-to-Tensor Contract Lab (NOT_ASSESSED)",
    "The preprocessing realizability matrix was not emitted.",
  ].join("\n");
  const candidates = evidence.candidates || [];
  const candidateRows = candidates.map((candidate) => [
    candidate.contract_id,
    candidate.tensor_channel_order,
    candidate.exact_tensor_realization ? "EXACT" : "NON-EXACT",
    `${formatNumber(candidate.exact_tensor_element_count)} / ${formatNumber(candidate.witness_tensor_element_count)}`,
    formatNumber(candidate.unrealizable_tensor_element_count),
    formatNumber(candidate.minimum_total_absolute_tensor_code_error_decimal),
    formatNumber(candidate.maximum_absolute_tensor_code_error),
    (candidate.channel_maps || []).map((row) => `${row.reachable_tensor_code_count}/256 (${row.tensor_code_hole_count} holes)`).join(" / "),
    code(candidate.nearest_rgb_fixture_sha256),
  ]);
  const best = candidates.find((candidate) => candidate.contract_id === evidence.best_non_exact_contract_id) || null;
  const firstExact = candidates.find((candidate) => candidate.exact_tensor_realization) || null;
  const mismatchRows = best?.witness_code_realizations?.map((row) => [
    `tensor ${["R", "G", "B"][row.tensor_channel] || row.tensor_channel} -> source ${["R", "G", "B"][row.source_pixel_channel] || row.source_pixel_channel}`,
    row.target_tensor_code,
    formatNumber(row.tensor_element_count),
    row.exact_source_pixel_codes.length ? row.exact_source_pixel_codes.join(", ") : "none",
    row.selected_source_pixel_code,
    row.roundtrip_tensor_code,
    row.absolute_tensor_code_error,
  ]) || [];
  return [
    "## Pixel-to-Tensor Contract Lab (DERIVED EXHAUSTIVE COUNTERFACTUAL MATRIX)",
    markdownTable(["Metric", "Value"], [
      ["Schema / method", `${evidence.schema} / ${evidence.method_version}`],
      ["Evidence / assessment kind", `${evidence.evidence_class} / ${evidence.assessment_kind}`],
      ["Source / eligible image witnesses", `${formatNumber(evidence.source_witness_count)} / ${formatNumber(evidence.eligible_image_witness_count)}`],
      ["Ineligible image witnesses", formatNumber(evidence.ineligible_witness_count)],
      ["Candidate conservation", evidence.candidate_conservation],
      ["Exact complete-tensor contracts", `${formatNumber(evidence.exact_tensor_realization_candidate_count)}: ${evidence.exact_contract_ids.join(", ") || "none"}`],
      ["Non-exact contracts", formatNumber(evidence.non_exact_candidate_count)],
      ["Best non-exact candidate", best ? `${best.contract_id}; ${formatNumber(best.unrealizable_tensor_element_count)} unrealizable elements; minimum total |code error| ${formatNumber(best.minimum_total_absolute_tensor_code_error_decimal)}` : "none"],
      ["Canonical exact RGB fixture SHA-256", firstExact ? code(firstExact.nearest_rgb_fixture_sha256) : "none"],
      ["Portfolio ledger SHA-256", code(evidence.portfolio_ledger_sha256)],
    ]),
    `### Candidate Contract Matrix\n\n${markdownTable([
      "Contract", "Tensor order", "Result", "Exact / total elements", "Unrealizable", "Total |error|", "Max |error|", "Reachable codes by tensor channel", "RGB fixture SHA-256",
    ], candidateRows)}`,
    best && mismatchRows.length ? `### Best Non-Exact Inverse Ledger: ${best.contract_id}\n\n${markdownTable([
      "Channel map", "Target code", "Elements", "Exact source pixels", "Selected pixel", "Roundtrip code", "|error|",
    ], mismatchRows)}` : "",
    best?.first_unrealizable_element ? markdownTable(["First finite-domain mismatch", "Value"], [
      ["Tensor coordinate / linear index", `[${best.first_unrealizable_element.tensor_coordinate_nhwc.join(", ")}] / ${formatNumber(best.first_unrealizable_element.tensor_linear_index)}`],
      ["Target / nearest roundtrip", `${best.first_unrealizable_element.target_tensor_code} / ${best.first_unrealizable_element.roundtrip_tensor_code}`],
      ["Selected RGB source pixel channel / code", `${best.first_unrealizable_element.source_pixel_channel} / ${best.first_unrealizable_element.selected_source_pixel_code}`],
      ["Absolute tensor-code error", formatNumber(best.first_unrealizable_element.absolute_tensor_code_error)],
    ]) : "",
    `Proof scope: ${evidence.proof_scope}`,
    `Method: ${evidence.method}`,
    `> ${evidence.interpretation_boundary}`,
  ].filter(Boolean).join("\n\n");
}

function quantizationLatticeMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) return "";
  const lattice = analysis?.quantization_lattice;
  if (!lattice) return [
    "## Quantization Lattice Lab (NOT_ASSESSED)",
    "The exhaustive binary and concatenation code-domain analysis was not emitted.",
  ].join("\n");
  const assessed = (lattice.residual_adds || []).filter((row) => row.assessment_status === "assessed");
  const byIndex = new Map(assessed.map((row) => [row.op_index, row]));
  const ranked = (lattice.domain_escape_ranking_op_indices || []).map((index) => byIndex.get(index)).filter(Boolean);
  const displayed = ranked.slice(0, 32);
  const unassessed = (lattice.residual_adds || []).filter((row) => row.assessment_status !== "assessed");
  const binaryRows = lattice.binary_contracts || [];
  const displayedBinary = binaryRows.slice(0, 32);
  const concatenationRows = lattice.concatenation_contracts || [];
  const rangeText = (range) => Array.isArray(range) ? `[${Number(range[0]).toPrecision(6)}, ${Number(range[1]).toPrecision(6)}]` : "-";
  const stepText = (value) => value == null ? "N/A" : `${Number(value).toFixed(4)} output steps`;
  const designText = (candidate) => candidate
    ? `${Number(candidate.scale_ratio_to_current).toFixed(4)}x scale / zp ${Number(candidate.signed_zero_point_delta) >= 0 ? "+" : ""}${candidate.signed_zero_point_delta} / ${formatNumber(candidate.rounded_projection_clamp_pair_count)} clamps`
    : "unavailable";
  return [
    "## Quantization Lattice Lab (DERIVED EXHAUSTIVE DOMAIN)",
    markdownTable(["Metric", "Value"], [
      ["Schema / method", `${lattice.schema} / ${lattice.method_version}`],
      ["All candidate / assessed / unassessed operators", `${formatNumber(lattice.candidate_operator_count)} / ${formatNumber(lattice.assessed_operator_count)} / ${formatNumber(lattice.unassessed_operator_count)}`],
      ["Family status", `ADD ${lattice.residual_add_status}; other binary ${lattice.binary_status}; CONCATENATION ${lattice.concatenation_status}`],
      ["Candidate / assessed / unassessed ADDs", `${formatNumber(lattice.candidate_add_count)} / ${formatNumber(lattice.assessed_add_count)} / ${formatNumber(lattice.unassessed_add_count)}`],
      ["Other binary candidate / assessed / unassessed", `${formatNumber(lattice.candidate_binary_count)} / ${formatNumber(lattice.assessed_binary_count)} / ${formatNumber(lattice.unassessed_binary_count)}`],
      ["CONCATENATION candidate / assessed / unassessed", `${formatNumber(lattice.candidate_concatenation_count)} / ${formatNumber(lattice.assessed_concatenation_count)} / ${formatNumber(lattice.unassessed_concatenation_count)}`],
      ["Exhaustively enumerated binary code pairs", `${formatNumber(lattice.total_enumerated_code_pairs)} (${formatNumber(lattice.residual_add_enumerated_code_pairs)} residual ADD)`],
      ["Exhaustively enumerated CONCATENATION input codes", formatNumber(lattice.total_enumerated_concatenation_codes)],
      ["ADDs with complete legal-domain containment", `${formatNumber(lattice.complete_domain_containment_add_count)} / ${formatNumber(lattice.assessed_add_count)}`],
      ["ADDs with endpoint-range escape pairs", formatNumber(lattice.range_escape_add_count)],
      ["Other binary ops contained / with endpoint escape", `${formatNumber(lattice.complete_domain_containment_binary_count)} / ${formatNumber(lattice.range_escape_binary_count)}`],
      ["CONCATENATION ops with endpoint escape", formatNumber(lattice.concatenation_range_escape_count)],
      ["Maximum uniform-domain endpoint escape ratio", formatPercent(lattice.maximum_range_escape_pair_ratio)],
      ["Maximum all-pair mean clamped projection error", stepText(lattice.maximum_mean_clamped_projection_error_steps)],
      ["Containment contracts derived", `${formatNumber(lattice.containment_design_add_count)} / ${formatNumber(lattice.assessed_add_count)}`],
      ["Fixed-zero-point containment available / scale expansion", `${formatNumber(lattice.fixed_zero_point_containment_add_count)} / ${formatNumber(lattice.fixed_zero_point_scale_expansion_add_count)}`],
      ["Global finest contracts requiring zero-point shift", formatNumber(lattice.global_zero_point_shift_add_count)],
      ["Maximum fixed-zp / global-finest scale ratio", `${lattice.maximum_fixed_zero_point_scale_ratio == null ? "N/A" : `${Number(lattice.maximum_fixed_zero_point_scale_ratio).toFixed(6)}x`} / ${lattice.maximum_global_finest_scale_ratio == null ? "N/A" : `${Number(lattice.maximum_global_finest_scale_ratio).toFixed(6)}x`}`],
      ["Rounding rule", lattice.rounding_rule],
    ]),
    displayed.length ? markdownTable([
      "Rank", "ADD", "Input scales", "Output scale", "Legal sum range", "Output range",
      "Endpoint escape", "Rounded clamp", "In-range rounding mean / max", "All-pair projection mean / max", "Projected codes", "Fixed-zp containment", "Global finest containment",
    ], displayed.map((row) => [
      `#${row.domain_escape_rank}`,
      `#${padOp(row.op_index)} ${row.op_name}`,
      row.input_scales.map((value) => Number(value).toPrecision(6)).join(" / "),
      Number(row.output_scale).toPrecision(6),
      rangeText(row.legal_sum_real_range),
      rangeText(row.output_real_range),
      `${formatNumber(row.range_escape_pair_count)} / ${formatNumber(row.enumerated_code_pair_count)} (${formatPercent(row.range_escape_pair_ratio)})`,
      `${formatNumber(row.rounded_projection_clamp_pair_count)} (${formatPercent(row.rounded_projection_clamp_pair_ratio)})`,
      `${stepText(row.mean_in_range_rounding_error_steps)} / ${stepText(row.maximum_in_range_rounding_error_steps)}`,
      `${stepText(row.mean_clamped_projection_error_steps)} / ${stepText(row.maximum_clamped_projection_error_steps)}`,
      `${formatNumber(row.distinct_projected_output_code_count)} / 256`,
      designText(row.fixed_zero_point_containment),
      designText(row.globally_finest_containment),
    ])) : "No 8-bit per-tensor residual ADD was assessed.",
    ranked.length > displayed.length
      ? `Displayed ${formatNumber(displayed.length)} of ${formatNumber(ranked.length)} assessed ADD rows; the structured evidence retains the complete portfolio.`
      : `Displayed all ${formatNumber(ranked.length)} assessed ADD rows.`,
    unassessed.length ? markdownTable(["ADD", "Status", "Reason"], unassessed.map((row) => [
      `#${padOp(row.op_index)} ${row.op_name}`,
      row.assessment_status,
      row.not_assessed_reason,
    ])) : "No candidate ADD row was left unassessed.",
    displayedBinary.length ? markdownTable([
      "Operator", "Status", "Input scales", "Output scale", "Legal result range", "Output range",
      "Endpoint escape", "Rounded clamp", "Mean / max projection", "Projected codes",
    ], displayedBinary.map((row) => [
      `#${padOp(row.op_index)} ${row.op_name}`,
      row.assessment_status,
      row.assessment_status === "assessed" ? row.input_scales.map((value) => Number(value).toPrecision(6)).join(" / ") : row.not_assessed_reason,
      row.output_scale == null ? "N/A" : Number(row.output_scale).toPrecision(6),
      rangeText(row.legal_sum_real_range),
      rangeText(row.output_real_range),
      row.range_escape_pair_count == null ? "N/A" : `${formatNumber(row.range_escape_pair_count)} / ${formatNumber(row.enumerated_code_pair_count)} (${formatPercent(row.range_escape_pair_ratio)})`,
      row.rounded_projection_clamp_pair_count == null ? "N/A" : `${formatNumber(row.rounded_projection_clamp_pair_count)} (${formatPercent(row.rounded_projection_clamp_pair_ratio)})`,
      row.mean_clamped_projection_error_steps == null ? "N/A" : `${stepText(row.mean_clamped_projection_error_steps)} / ${stepText(row.maximum_clamped_projection_error_steps)}`,
      row.distinct_projected_output_code_count == null ? "N/A" : `${formatNumber(row.distinct_projected_output_code_count)} / 256`,
    ])) : "No SUB, MUL, MAXIMUM, or MINIMUM candidate was present.",
    binaryRows.length > displayedBinary.length
      ? `Displayed ${formatNumber(displayedBinary.length)} of ${formatNumber(binaryRows.length)} other binary rows; structured evidence retains all rows.`
      : "Displayed all other binary rows.",
    concatenationRows.length ? markdownTable([
      "CONCATENATION", "Status", "Inputs", "Enumerated codes", "Endpoint escape", "Rounded clamp", "Mean / max projection",
    ], concatenationRows.map((row) => [
      `#${padOp(row.op_index)}`,
      row.assessment_status,
      formatNumber(row.input_count),
      row.enumerated_code_count == null ? "N/A" : formatNumber(row.enumerated_code_count),
      row.range_escape_code_count == null ? "N/A" : formatNumber(row.range_escape_code_count),
      row.rounded_projection_clamp_code_count == null ? "N/A" : formatNumber(row.rounded_projection_clamp_code_count),
      row.mean_absolute_projection_error_output_steps == null ? "N/A" : `${stepText(row.mean_absolute_projection_error_output_steps)} / ${stepText(row.maximum_absolute_projection_error_output_steps)}`,
    ])) : "No CONCATENATION candidate was present.",
    lattice.containment_formula ? `Containment design: ${lattice.containment_formula}` : "",
    `Method: ${lattice.method}`,
    `> ${lattice.interpretation_boundary}`,
  ].join("\n\n");
}

function contractMigrationMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) return "";
  const migration = analysis?.contract_migration;
  if (!migration) return [
    "## Contract Migration Impact Lab (NOT_ASSESSED)",
    "The residual containment re-export impact analysis was not emitted.",
  ].join("\n");
  const scenarioRows = (migration.migrations || []).flatMap((row) =>
    (row.scenarios || []).map((scenario) => {
      const kernelHashes = (scenario.kernel_consumers || [])
        .filter((consumer) => consumer.assessment_status === "assessed")
        .map((consumer) => `#${padOp(consumer.op_index)} ${consumer.channel_ledger_sha256}`)
        .join("; ") || "N/A";
      return [
        `#${padOp(row.source_add_op_index)} ADD`,
        scenario.design.startsWith("fixed") ? "fixed zero-point" : "globally finest",
        `${Number(scenario.candidate_output_scale).toPrecision(8)} / ${scenario.candidate_output_zero_point}`,
        `${Number(scenario.scale_ratio_to_current).toFixed(6)}x / ${Number(scenario.signed_zero_point_delta) >= 0 ? "+" : ""}${scenario.signed_zero_point_delta}`,
        `${formatNumber(scenario.assessed_consumer_count)} / ${formatNumber(scenario.unassessed_consumer_count)}`,
        formatNumber(scenario.assessed_kernel_channel_count),
        `${formatNumber(scenario.multiplier_encoding_changed_channel_count)} / ${formatNumber(scenario.multiplier_shift_changed_channel_count)}`,
        `${formatNumber(scenario.bias_code_changed_channel_count)} / ${formatNumber(scenario.bias_int32_overflow_channel_count)}`,
        formatNumber(scenario.add_parameter_encoding_changed_count),
        `${formatNumber(row.reachable_downstream_op_count)} / ${formatNumber(row.maximum_downstream_edge_depth)}`,
        code(kernelHashes),
      ];
    }));
  const topBias = (migration.migrations || []).flatMap((row) =>
    (row.scenarios || []).flatMap((scenario) =>
      (scenario.kernel_consumers || []).flatMap((consumer) =>
        (consumer.top_channels || []).map((channel) => ({ row, scenario, consumer, channel })))))
    .sort((left, right) => Number(right.channel.absolute_bias_rebase_error_current_steps ?? Infinity)
      - Number(left.channel.absolute_bias_rebase_error_current_steps ?? Infinity)
      || left.consumer.op_index - right.consumer.op_index
      || left.channel.channel_index - right.channel.channel_index)
    .slice(0, 16);
  const sourceRows = (migration.source_references || []).map((source) => [
    source.role,
    `[${source.file}](${source.url})`,
    code(source.sha256),
  ]);
  return [
    "## Contract Migration Impact Lab (DERIVED COUNTERFACTUAL RE-EXPORT)",
    markdownTable(["Metric", "Value"], [
      ["Schema / method", `${migration.schema} / ${migration.method_version}`],
      ["Residual contracts / candidate scenarios", `${formatNumber(migration.residual_contract_count)} / ${formatNumber(migration.candidate_scenario_count)}`],
      ["Direct parameter consumers / edges", `${formatNumber(migration.direct_consumer_count)} / ${formatNumber(migration.direct_consumer_edge_count)}`],
      ["Kernel / ADD / other direct consumers", `${formatNumber(migration.kernel_consumer_count)} / ${formatNumber(migration.add_consumer_count)} / ${formatNumber(migration.other_consumer_count)}`],
      ["Assessed / unassessed consumer scenarios", `${formatNumber(migration.assessed_consumer_scenario_count)} / ${formatNumber(migration.unassessed_consumer_scenario_count)}`],
      ["Assessed kernel channel scenarios", formatNumber(migration.assessed_kernel_channel_scenario_count)],
      ["Changed multiplier encodings / shifts", `${formatNumber(migration.multiplier_encoding_changed_channel_scenario_count)} / ${formatNumber(migration.multiplier_shift_changed_channel_scenario_count)}`],
      ["Changed bias codes / candidate INT32 overflow", `${formatNumber(migration.bias_code_changed_channel_scenario_count)} / ${formatNumber(migration.bias_int32_overflow_channel_scenario_count)}`],
      ["Changed ADD parameter encodings", formatNumber(migration.add_parameter_encoding_changed_scenario_count)],
      ["Reachable downstream op union / maximum depth", `${formatNumber(migration.reachable_downstream_op_union_count)} / ${formatNumber(migration.maximum_downstream_edge_depth)}`],
      ["Pinned TensorFlow source", migration.source_commit],
    ]),
    sourceRows.length ? `### Source Digest Basis\n\n${markdownTable(["Role", "Pinned file", "SHA-256"], sourceRows)}` : "",
    scenarioRows.length ? markdownTable([
      "Source", "Candidate", "Scale / zp", "Scale ratio / zp delta", "Consumers A/U", "Kernel channels",
      "Q0.31 encoding / shift changes", "Bias code changes / overflow", "ADD encoding changes",
      "Reachable ops / max depth", "Kernel ledger SHA-256",
    ], scenarioRows) : "No residual containment candidate produced a migration scenario.",
    topBias.length ? `### Highest Bias Rebase Error\n\n${markdownTable([
      "Source / candidate", "Consumer", "Channel", "Weight scale", "Multiplier q / shift", "Bias code", "Absolute error", "Current / candidate steps", "INT32",
    ], topBias.map(({ row, scenario, consumer, channel }) => [
      `#${padOp(row.source_add_op_index)} / ${scenario.design.startsWith("fixed") ? "fixed-zp" : "global"}`,
      `#${padOp(consumer.op_index)} ${consumer.op_name}`,
      formatNumber(channel.channel_index),
      Number(channel.weight_scale).toExponential(6),
      `${channel.current_quantized_multiplier} / ${channel.current_shift} -> ${channel.candidate_quantized_multiplier} / ${channel.candidate_shift}`,
      `${channel.current_bias_code} -> ${channel.candidate_bias_code_decimal}`,
      channel.absolute_bias_rebase_error == null
        ? "N/A"
        : Number(channel.absolute_bias_rebase_error).toExponential(6),
      `${channel.absolute_bias_rebase_error_current_steps == null
        ? "N/A"
        : Number(channel.absolute_bias_rebase_error_current_steps).toFixed(6)} / ${channel.absolute_bias_rebase_error_candidate_steps == null
        ? "N/A"
        : Number(channel.absolute_bias_rebase_error_candidate_steps).toFixed(6)}`,
      channel.bias_int32_overflow ? "overflow" : "safe",
    ]))}` : "",
    `Kernel multiplier: ${migration.kernel_multiplier_formula}`,
    `Bias rebase: ${migration.bias_rebase_formula}`,
    `ADD parameters: ${migration.add_parameter_formula}`,
    `Method: ${migration.method}`,
    `> ${migration.interpretation_boundary}`,
  ].filter(Boolean).join("\n\n");
}

function residualStepResponseMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) return "";
  const response = analysis?.residual_step_response;
  if (!response) return [
    "## Residual Step Response Lab (NOT_ASSESSED)",
    "The exhaustive adjacent-code response analysis was not emitted.",
  ].join("\n");
  const rows = (response.residual_adds || []).flatMap((row) => (row.contracts || []).map((contract) => [
    `#${padOp(row.op_index)} ADD`,
    contract.design === "current_artifact_contract" ? "current"
      : contract.design === "fixed_zero_point_minimum_containment" ? "fixed-zp containment" : "globally finest containment",
    `${Number(contract.output_scale).toPrecision(8)} / ${contract.output_zero_point}`,
    formatNumber(contract.rounded_projection_clamp_pair_count),
    `${formatNumber(contract.visible_transition_count)} / ${formatNumber(contract.silent_transition_count)}`,
    `${formatPercent(contract.both_branches_visible_ratio)} / ${formatPercent(contract.neither_branch_visible_ratio)}`,
    `${signedCount(contract.removed_rounded_clamp_pairs_vs_current)} / ${signedCount(contract.additional_silent_transitions_vs_current)}`,
    code(contract.transition_ledger_sha256),
  ]));
  const topIndex = response.retention_cost_ranking_op_indices?.[0];
  const top = (response.residual_adds || []).find((row) => row.op_index === topIndex);
  const branchRows = (top?.contracts || []).flatMap((contract) => (contract.branch_responses || []).map((branch) => [
    contract.design === "current_artifact_contract" ? "current"
      : contract.design === "fixed_zero_point_minimum_containment" ? "fixed-zp" : "global",
    `input ${branch.branch_index} / T${branch.input_tensor_index}`,
    Number(branch.input_scale).toPrecision(8),
    `${formatNumber(branch.visible_transition_count)} / ${formatNumber(branch.silent_transition_count)}`,
    `${formatNumber(branch.unclipped_silent_transition_count)} / ${formatNumber(branch.clamp_associated_silent_transition_count)}`,
    formatNumber(branch.multi_code_jump_transition_count),
    Number(branch.mean_absolute_step_reproduction_error_output_steps).toFixed(6),
  ]));
  return [
    "## Residual Step Response Lab (DERIVED EXHAUSTIVE ADJACENT-CODE DOMAIN)",
    markdownTable(["Metric", "Value"], [
      ["Schema / method", `${response.schema} / ${response.method_version}`],
      ["Candidate residual ADDs", formatNumber(response.candidate_add_count)],
      ["Assessed residual ADDs / contracts", `${formatNumber(response.assessed_add_count)} / ${formatNumber(response.contract_response_count)}`],
      ["Exact branch transitions / joint interior cells", `${formatNumber(response.total_transition_count)} / ${formatNumber(response.total_joint_interior_cell_count)}`],
      ["Current silent transitions / rounded clamp pairs", `${formatNumber(response.current_silent_transition_count)} / ${formatNumber(response.current_rounded_projection_clamp_pair_count)}`],
      ["Containment additional silent transitions", signedCount(response.containment_additional_silent_transition_count)],
      ["Containment silent transitions", formatNumber(response.containment_silent_transition_count)],
      ["Containment removed rounded clamp pairs", formatNumber(response.containment_removed_rounded_clamp_pair_count)],
      ["Maximum containment visibility loss", formatPercent(response.maximum_containment_silent_ratio_increase)],
      ["Highest retention-cost ADD", topIndex == null ? "N/A" : `#${padOp(topIndex)} ADD`],
    ]),
    markdownTable(["ADD", "Contract", "Scale / zp", "Clamp pairs", "Visible / silent transitions", "Both / neither cells", "Removed clamps / added silent", "Transition ledger SHA-256"], rows),
    branchRows.length ? `### Highest Retention-cost Branch Ledger\n\n${markdownTable(["Contract", "Branch", "Input scale", "Visible / silent", "Unclipped / clamp silent", ">1 jumps", "Mean error (output steps)"], branchRows)}` : "",
    `Transition definition: ${response.transition_definition}`,
    `Joint-cell definition: ${response.joint_cell_definition}`,
    `Transition ledger: ${response.transition_ledger_hash_method}`,
    `Method: ${response.method}`,
    `> ${response.interpretation_boundary}`,
  ].filter(Boolean).join("\n\n");
}

function residualContractDistortionMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) return "";
  const distortion = analysis?.residual_contract_distortion;
  if (!distortion) return [
    "## Residual Contract Distortion Atlas (NOT_ASSESSED)",
    "The exhaustive current-versus-containment pair analysis was not emitted.",
  ].join("\n");
  const topIndex = distortion.distortion_ranking_op_indices?.[0];
  const top = (distortion.residual_adds || []).find((row) => row.op_index === topIndex);
  const rows = (distortion.residual_adds || []).flatMap((row) => (row.scenarios || []).map((scenario) => [
    `#${padOp(row.op_index)} ADD`,
    scenario.design === "fixed_zero_point_minimum_containment" ? "fixed-zp containment" : "globally finest containment",
    `${Number(scenario.candidate_output_scale).toPrecision(8)} / ${scenario.candidate_output_zero_point}`,
    formatNumber(scenario.rescued_current_clamp_pair_count),
    `${formatNumber(scenario.changed_represented_value_pair_count)} / ${formatNumber(scenario.sign_class_changed_pair_count)}`,
    `${formatNumber(scenario.ideal_error_improved_pair_count)} / ${formatNumber(scenario.ideal_error_worsened_pair_count)} / ${formatNumber(scenario.ideal_error_equal_within_tolerance_pair_count)}`,
    `${Number(scenario.root_mean_square_contract_delta_current_steps).toFixed(6)} / ${Number(scenario.p50_absolute_contract_delta_current_steps).toFixed(6)} / ${Number(scenario.p90_absolute_contract_delta_current_steps).toFixed(6)} / ${Number(scenario.p99_absolute_contract_delta_current_steps).toFixed(6)}`,
    signedScientific(scenario.signed_mean_absolute_ideal_error_delta),
    code(scenario.pair_ledger_sha256),
  ]));
  const witnesses = (top?.scenarios || []).map((scenario) => {
    const witness = scenario.worst_absolute_contract_delta_pair;
    return [
      scenario.design === "fixed_zero_point_minimum_containment" ? "fixed-zp" : "global",
      `${witness.input_0_code}, ${witness.input_1_code}`,
      Number(witness.ideal_real_sum).toPrecision(9),
      `${witness.current_raw_code} -> ${witness.current_projected_code} / ${Number(witness.current_represented_real).toPrecision(9)}`,
      `${witness.candidate_raw_code} -> ${witness.candidate_projected_code} / ${Number(witness.candidate_represented_real).toPrecision(9)}`,
      `${signedScientific(witness.signed_contract_delta_current_steps)} current steps`,
      `${Number(witness.current_absolute_ideal_error).toPrecision(7)} -> ${Number(witness.candidate_absolute_ideal_error).toPrecision(7)}`,
    ];
  });
  return [
    "## Residual Contract Distortion Atlas (DERIVED EXHAUSTIVE COUNTERFACTUAL DOMAIN)",
    markdownTable(["Metric", "Value"], [
      ["Schema / method", `${distortion.schema} / ${distortion.method_version}`],
      ["Candidate residual ADDs", formatNumber(distortion.candidate_add_count)],
      ["Assessed residual ADDs / candidate scenarios", `${formatNumber(distortion.assessed_add_count)} / ${formatNumber(distortion.scenario_count)}`],
      ["Exact current-versus-candidate pair comparisons", formatNumber(distortion.total_enumerated_pair_count)],
      ["Unique current clamp pairs", `${formatNumber(distortion.current_clamped_pair_instance_count)} across ${formatNumber(distortion.assessed_add_count)} current artifact contract(s); counted once per ADD`],
      ["Scenario-current / candidate clamps / rescued", `${formatNumber(distortion.scenario_current_clamped_pair_instance_count)} / ${formatNumber(distortion.candidate_clamped_pair_count)} / ${formatNumber(distortion.rescued_current_clamp_pair_instance_count)} across ${formatNumber(distortion.scenario_count)} candidate comparisons`],
      ["Changed represented values / sign classes", `${formatNumber(distortion.changed_represented_value_pair_count)} / ${formatNumber(distortion.sign_class_changed_pair_count)}`],
      ["Ideal-error improve / worsen / equal within tolerance", `${formatNumber(distortion.ideal_error_improved_pair_count)} / ${formatNumber(distortion.ideal_error_worsened_pair_count)} / ${formatNumber(distortion.ideal_error_equal_within_tolerance_pair_count)}`],
      ["Maximum RMS / p99 displacement", `${Number(distortion.maximum_rms_contract_delta_current_steps).toFixed(6)} / ${Number(distortion.maximum_p99_contract_delta_current_steps).toFixed(6)} current steps`],
      ["Highest distortion ADD", topIndex == null ? "N/A" : `#${padOp(topIndex)} ADD`],
    ]),
    markdownTable(["ADD", "Candidate", "Scale / zp", "Clamp rescue", "Changed value / sign", "Error improve / worsen / equal", "RMS / p50 / p90 / p99 current steps", "Mean ideal-error delta", "Pair ledger SHA-256"], rows),
    witnesses.length ? `### Highest-distortion Exact Pair Witnesses\n\n${markdownTable(["Candidate", "Input pair", "Ideal sum", "Current raw -> code / real", "Candidate raw -> code / real", "Displacement", "Ideal error current -> candidate"], witnesses)}` : "",
    `Projection definition: ${distortion.projection_definition}`,
    `Error comparison: ${distortion.error_comparison_definition}`,
    `Pair ledger: ${distortion.pair_ledger_hash_method}`,
    `Method: ${distortion.method}`,
    `> ${distortion.interpretation_boundary}`,
  ].filter(Boolean).join("\n\n");
}
