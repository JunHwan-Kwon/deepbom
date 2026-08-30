import {
  bottleneckDistributionData,
  boundTone,
  macDistributionData,
  quantTileTone,
} from "./analysis.js";
import {
  colorForTone,
  createExportCanvasShell,
  drawBreakPill,
  drawChainBlock,
  drawEmptyState,
  drawFlame,
  drawLegend,
  drawRankList,
  drawStageMixRow,
  drawTargetCard,
  roundRect,
} from "./dom.js";
import {
  formatNumber,
  formatPercent,
  formatUs,
  padOp,
} from "./format.js";
import { quantSummaryEvidence } from "./performance-visuals.js";
import { roundTiesAway } from "./quantization-math.js";
import { renderNumericalAbiPropagationCanvas } from "./numerical-abi-propagation-viewer.js";
import { renderAccumulatorReachabilityCanvas } from "./accumulator-reachability-viewer.js";
import { renderInputCounterexampleCanvas } from "./input-counterexample-viewer.js";
import { renderPreprocessingRealizabilityCanvas } from "./preprocessing-realizability-viewer.js";
import { renderPreprocessingConsequenceCanvas } from "./preprocessing-consequence-viewer.js";
import {
  buildQuantizationExposurePresentation,
  buildResourceMapPresentation,
  layoutGroupedTreemap,
} from "./evidence-treemap.js";
import { serializedTensorPresentation } from "./serialized-tensor-view.js";

export function visualPngSpecs({
  analysis,
  filename = "",
  targetProfile = {},
  targetComparisonRows = [],
  preprocessingConsequenceResult = null,
  preprocessingConsequenceCapture = null,
}) {
  return [
    ["visuals/mac_distribution.png", () => renderMacDistributionCanvas(analysis, filename)],
    ["visuals/estimated_bottleneck.png", () => renderBottleneckCanvas(analysis, filename, targetProfile)],
    ["visuals/target_comparison.png", () => renderTargetComparisonCanvas(filename, targetComparisonRows)],
    ["visuals/xnnpack_chain_flow.png", () => renderChainFlowCanvas(analysis, filename)],
    ["visuals/quantization_heatmap.png", () => renderQuantHeatmapCanvas(analysis, filename)],
    ["visuals/explorer_resource_map.png", () => renderEvidenceTreemapCanvas(analysis, filename, "resource")],
    ["visuals/quantization_exposure_map.png", () => renderEvidenceTreemapCanvas(analysis, filename, "quantization")],
    ...(analysis?.quantization_lattice?.assessed_add_count
      ? [["visuals/residual_quantization_lattice.png", () => renderResidualLatticeCanvas(analysis, filename)]]
      : []),
    ...(analysis?.accumulator_atlas?.assessed_op_count
      ? [["visuals/accumulator_headroom_atlas.png", () => renderAccumulatorAtlasCanvas(analysis, filename)]]
      : []),
    ...(analysis?.requantization_fidelity?.assessed_op_count
      ? [["visuals/requantization_fidelity.png", () => renderRequantizationFidelityCanvas(analysis, filename)]]
      : []),
    ...(analysis?.kernel_extremum_witness?.assessed_op_count
      ? [["visuals/kernel_extremum_witness.png", () => renderKernelExtremumWitnessCanvas(analysis, filename)]]
      : []),
    ...(analysis?.channel_vitality?.assessed_op_count
      ? [["visuals/channel_vitality.png", () => renderChannelVitalityCanvas(analysis, filename)]]
      : []),
    ...(analysis?.rounding_equivalence?.assessed_op_count
      ? [["visuals/rounding_equivalence.png", () => renderRoundingEquivalenceCanvas(analysis, filename)]]
      : []),
    ...(analysis?.accumulator_reachability?.assessed_op_count
      ? [["visuals/accumulator_reachability.png", () => renderAccumulatorReachabilityCanvas(analysis, filename)]]
      : []),
    ...(analysis?.numerical_abi_propagation?.output_reachable_source_op_count
      ? [["visuals/numerical_abi_propagation.png", () => renderNumericalAbiPropagationCanvas(analysis, filename)]]
      : []),
    ...(analysis?.input_counterexample?.representative_witness_count
      ? [["visuals/input_counterexample.png", () => renderInputCounterexampleCanvas(analysis, filename)]]
      : []),
    ...(analysis?.preprocessing_realizability?.candidate_evaluation_count
      ? [["visuals/preprocessing_realizability.png", () => renderPreprocessingRealizabilityCanvas(analysis, filename)]]
      : []),
    ...(preprocessingConsequenceResult
      ? [["visuals/preprocessing_consequence_atlas.png", () => renderPreprocessingConsequenceCanvas(preprocessingConsequenceResult, preprocessingConsequenceCapture, filename)]]
      : []),
    ...(analysis?.contract_migration?.residual_contract_count
      ? [["visuals/contract_migration_impact.png", () => renderContractMigrationCanvas(analysis, filename)]]
      : []),
    ...(analysis?.residual_step_response?.assessed_add_count
      ? [["visuals/residual_step_response.png", () => renderResidualStepResponseCanvas(analysis, filename)]]
      : []),
    ...(analysis?.residual_contract_distortion?.assessed_add_count
      ? [["visuals/residual_contract_distortion.png", () => renderResidualContractDistortionCanvas(analysis, filename)]]
      : []),
    ["visuals/stage_memory_mix.png", () => renderStageMemoryMixCanvas(analysis, filename)],
  ];
}

function renderRoundingEquivalenceCanvas(analysis, filename) {
  const equivalence = analysis.rounding_equivalence || {};
  const rowsByIndex = new Map((equivalence.ops || []).map((row) => [row.op_index, row]));
  const rows = (equivalence.equivalence_ranking_op_indices || []).map((index) => rowsByIndex.get(index)).filter(Boolean);
  const topRow = rows[0] || (equivalence.ops || []).find((row) => row.assessment_status === "assessed");
  const topChannel = topRow?.top_channels?.find((channel) => Number(channel.divergent_state_count_decimal || 0) > 0) || topRow?.top_channels?.[0];
  const { canvas, ctx, width, y } = createExportCanvas(
    filename,
    "Fixed-Point Rounding Equivalence Lab",
    "Exact default-versus-TFLITE_SINGLE_ROUNDING certificate over every integer in each post-bias interval hull.",
    1180,
    800,
  );
  if (!topRow || !topChannel) {
    drawEmptyState(ctx, "No fixed-point rounding-equivalence interval was assessed.", 48, y);
    return canvas;
  }
  const metrics = [
    ["Certified channels", formatNumber(equivalence.assessed_channel_count)],
    ["Equivalent", formatNumber(equivalence.equivalent_channel_count)],
    ["Divergent", formatNumber(equivalence.divergent_channel_count)],
    ["Divergent states", formatNumber(equivalence.divergent_state_count_decimal)],
    ["Maximum delta", `${formatNumber(equivalence.maximum_absolute_output_delta || 0)} code`],
  ];
  metrics.forEach(([label, value], index) => {
    const x = 48 + index * 216;
    ctx.fillStyle = index === 2 || index === 3 ? "#f8eeee" : index === 4 ? "#fbf4e8" : "#eef5f3";
    ctx.fillRect(x, y + 12, 202, 66);
    ctx.fillStyle = "#617083";
    ctx.font = "10px Inter, Arial, sans-serif";
    ctx.fillText(label, x + 11, y + 34, 180);
    ctx.fillStyle = "#17202c";
    ctx.font = "700 18px Inter, Arial, sans-serif";
    ctx.fillText(String(value), x + 11, y + 60, 180);
  });

  const histX = 48;
  const histY = y + 125;
  const histWidth = 500;
  const histHeight = 270;
  ctx.fillStyle = "#17202c";
  ctx.font = "700 15px Inter, Arial, sans-serif";
  ctx.fillText("Channel divergence-ratio distribution", histX, histY - 15);
  ctx.fillStyle = "#f3f6f8";
  ctx.fillRect(histX, histY, histWidth, histHeight);
  const histogram = equivalence.divergence_histogram || [];
  const maxChannels = Math.max(1, ...histogram.map((bin) => Number(bin.channel_count || 0)));
  const slot = histWidth / Math.max(1, histogram.length);
  histogram.forEach((bin, index) => {
    const barHeight = Math.sqrt(Number(bin.channel_count || 0) / maxChannels) * (histHeight - 67);
    const x = histX + index * slot;
    ctx.fillStyle = index === 0 ? "#2f7b82" : index >= 5 ? "#bd4448" : "#d69a34";
    ctx.fillRect(x + slot * 0.2, histY + histHeight - 38 - barHeight, slot * 0.6, barHeight);
    ctx.fillStyle = "#536273";
    ctx.font = "9px Inter, Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(bin.label, x + slot / 2, histY + histHeight - 19, slot - 4);
    ctx.fillText(formatNumber(bin.channel_count), x + slot / 2, histY + histHeight - 45 - barHeight, slot - 4);
  });
  ctx.textAlign = "left";
  ctx.strokeStyle = "#536273";
  ctx.strokeRect(histX, histY, histWidth, histHeight);

  const rankX = 590;
  const rankY = histY;
  const rankWidth = width - rankX - 48;
  ctx.fillStyle = "#17202c";
  ctx.font = "700 15px Inter, Arial, sans-serif";
  ctx.fillText("Build-mode exposure ranking", rankX, rankY - 15);
  const rankRows = rows.slice(0, 12);
  const maxRatio = Math.max(1e-12, ...rankRows.map((row) => Number(row.divergent_state_ratio || 0)));
  rankRows.forEach((row, index) => {
    const rowY = rankY + index * 22;
    ctx.fillStyle = index % 2 ? "#f8fafb" : "#f1f5f7";
    ctx.fillRect(rankX, rowY, rankWidth, 19);
    ctx.fillStyle = "#334255";
    ctx.font = index === 0 ? "700 10px Inter, Arial, sans-serif" : "10px Inter, Arial, sans-serif";
    ctx.fillText(`#${padOp(row.op_index)} ${row.op_name}`, rankX + 6, rowY + 13, 152);
    ctx.fillStyle = "#d69a34";
    ctx.fillRect(rankX + 162, rowY + 5, Number(row.divergent_state_ratio || 0) / maxRatio * (rankWidth - 260), 9);
    ctx.fillStyle = "#17202c";
    ctx.textAlign = "right";
    ctx.fillText(`${formatPercent(row.divergent_state_ratio)} / ${formatNumber(row.divergent_channel_count)} ch`, rankX + rankWidth - 7, rowY + 13, 92);
    ctx.textAlign = "left";
  });

  const certificateY = y + 438;
  ctx.fillStyle = "#17202c";
  ctx.font = "700 15px Inter, Arial, sans-serif";
  ctx.fillText("Exact counterexample certificate", 48, certificateY);
  ctx.fillStyle = "#eef3f5";
  ctx.fillRect(48, certificateY + 15, width - 96, 177);
  ctx.fillStyle = "#bd4448";
  ctx.fillRect(48, certificateY + 15, 4, 177);
  const certificateRows = [
    ["Coordinate", `#${padOp(topRow.op_index)} ${topRow.op_name} / channel ${topChannel.channel_index}`],
    ["Post-bias interval", `${topChannel.post_bias_minimum_decimal} .. ${topChannel.post_bias_maximum_decimal} (${formatNumber(topChannel.interval_state_count_decimal)} states)`],
    ["Divergent exposure", `${formatNumber(topChannel.divergent_state_count_decimal)} states / ${formatPercent(topChannel.divergent_state_ratio)} / ${formatNumber(topChannel.divergent_region_count)} regions`],
    ["First counterexample", `${topChannel.first_divergent_accumulator_decimal ?? "none"}: default ${topChannel.first_default_output_code ?? "equal"} / single ${topChannel.first_single_output_code ?? "equal"}`],
    ["Last / maximum delta", `${topChannel.last_divergent_accumulator_decimal ?? "none"} / ${formatNumber(topChannel.maximum_absolute_output_delta)} code`],
    ["Default / single encoding", `${topChannel.default_quantized_multiplier}, shift ${topChannel.default_shift} / ${topChannel.single_quantized_multiplier}, shift ${topChannel.single_shift}`],
  ];
  certificateRows.forEach(([label, value], index) => {
    const rowY = certificateY + 40 + index * 22;
    ctx.fillStyle = "#617083";
    ctx.font = "10px Inter, Arial, sans-serif";
    ctx.fillText(label, 65, rowY, 150);
    ctx.fillStyle = "#17202c";
    ctx.font = "700 11px Inter, Arial, sans-serif";
    ctx.fillText(String(value), 220, rowY, width - 280);
  });
  ctx.fillStyle = "#617083";
  ctx.font = "10px ui-monospace, SFMono-Regular, Consolas, monospace";
  ctx.fillText(`certificate ${topRow.equivalence_ledger_sha256}`, 65, certificateY + 174, width - 130);
  ctx.font = "10px Inter, Arial, sans-serif";
  ctx.fillText(`${formatNumber(equivalence.divergent_state_count_decimal)} / ${formatNumber(equivalence.interval_state_count_decimal)} interval states (${formatPercent(equivalence.divergent_state_ratio)}); interior states can be unreachable and are not observed frequencies.`, 48, certificateY + 218, width - 96);
  ctx.fillText(`Schema ${equivalence.schema} / method ${equivalence.method_version} / source ${equivalence.source_commit}`, 48, certificateY + 238, width - 96);
  return canvas;
}

function renderKernelExtremumWitnessCanvas(analysis, filename) {
  const witness = analysis.kernel_extremum_witness || {};
  const rowsByIndex = new Map((witness.ops || []).map((row) => [row.op_index, row]));
  const rows = (witness.witness_ranking_op_indices || []).map((index) => rowsByIndex.get(index)).filter(Boolean);
  const row = rows[0] || (witness.ops || []).find((item) => item.assessment_status === "assessed");
  const channel = row?.worst_channel || row?.top_channels?.[0];
  const { canvas, ctx, width, y } = createExportCanvas(
    filename,
    "Quantized Kernel Witness Lab",
    "Exact synthetic legal-code receptive-field endpoints under pinned default and TFLITE_SINGLE_ROUNDING algebra.",
    1180,
    800,
  );
  if (!row || !channel) {
    drawEmptyState(ctx, "No quantized kernel witness was assessed.", 48, y);
    return canvas;
  }

  const rankX = 48;
  const rankY = y + 24;
  const rankWidth = 625;
  const rankRows = rows.slice(0, 12);
  const rankMax = Math.max(1, ...rankRows.map((item) => Math.max(
    Number(item.default_ideal_mismatch_endpoint_count || 0),
    Number(item.build_mode_divergent_endpoint_count || 0),
  )));
  ctx.fillStyle = "#17202c";
  ctx.font = "700 15px Inter, Arial, sans-serif";
  ctx.fillText("Ranked exact endpoint differences", rankX, rankY);
  ctx.font = "11px Inter, Arial, sans-serif";
  ctx.fillStyle = "#617083";
  ctx.fillText("teal: default vs direct ideal   red: default vs single-rounding", rankX, rankY + 20);
  rankRows.forEach((item, index) => {
    const rowY = rankY + 45 + index * 31;
    const labelWidth = 185;
    const chartX = rankX + labelWidth;
    const chartWidth = rankWidth - labelWidth;
    ctx.fillStyle = index % 2 ? "#f8fafb" : "#f1f5f7";
    ctx.fillRect(rankX, rowY - 16, rankWidth, 25);
    ctx.fillStyle = "#334255";
    ctx.font = index === 0 ? "700 11px Inter, Arial, sans-serif" : "11px Inter, Arial, sans-serif";
    ctx.fillText(`#${padOp(item.op_index)} ${item.op_name}`, rankX + 7, rowY, labelWidth - 12);
    const idealWidth = Number(item.default_ideal_mismatch_endpoint_count || 0) / rankMax * chartWidth;
    const modeWidth = Number(item.build_mode_divergent_endpoint_count || 0) / rankMax * chartWidth;
    ctx.fillStyle = "#2f7b82";
    ctx.fillRect(chartX, rowY - 14, idealWidth, 7);
    ctx.fillStyle = "#bd4448";
    ctx.fillRect(chartX, rowY - 4, modeWidth, 7);
    ctx.fillStyle = "#17202c";
    ctx.font = "700 10px Inter, Arial, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(`${formatNumber(item.default_ideal_mismatch_endpoint_count)} / ${formatNumber(item.build_mode_divergent_endpoint_count)}`, rankX + rankWidth - 6, rowY);
    ctx.textAlign = "left";
  });

  const endpointX = 715;
  const endpointWidth = width - endpointX - 48;
  ctx.fillStyle = "#17202c";
  ctx.font = "700 15px Inter, Arial, sans-serif";
  ctx.fillText(`#${padOp(row.op_index)} ${row.op_name} / channel ${channel.channel_index}`, endpointX, rankY);
  ctx.fillStyle = "#617083";
  ctx.font = "11px Inter, Arial, sans-serif";
  ctx.fillText(`${formatNumber(channel.term_count)} terms: ${formatNumber(channel.positive_centered_weight_count)} positive / ${formatNumber(channel.negative_centered_weight_count)} negative / ${formatNumber(channel.zero_centered_weight_count)} zero`, endpointX, rankY + 20, endpointWidth);
  [channel.minimum, channel.maximum].forEach((endpoint, index) => {
    const cardY = rankY + 45 + index * 196;
    ctx.fillStyle = index ? "#f4f8f7" : "#f8f4f4";
    ctx.fillRect(endpointX, cardY, endpointWidth, 178);
    ctx.strokeStyle = index ? "#2f7b82" : "#bd4448";
    ctx.strokeRect(endpointX, cardY, endpointWidth, 178);
    ctx.fillStyle = "#17202c";
    ctx.font = "700 14px Inter, Arial, sans-serif";
    ctx.fillText(`${String(endpoint.endpoint || "endpoint").toUpperCase()} witness`, endpointX + 14, cardY + 24);
    const metrics = [
      ["Input codes", (endpoint.witness_code_histogram || []).map((item) => `${item.code} x ${formatNumber(item.count)}`).join(" / ")],
      ["Dot / bias / post-bias", `${endpoint.dot_product_decimal} / ${endpoint.bias_decimal} / ${endpoint.post_bias_accumulator_decimal}`],
      ["Ideal / default / single", `${endpoint.ideal_output_code ?? "N/A"} / ${endpoint.default_output_code ?? "N/A"} / ${endpoint.single_output_code ?? "N/A"}`],
      ["Default / single clamp", `${endpoint.default_activation_clamped == null ? "N/A" : endpoint.default_activation_clamped ? "yes" : "no"} / ${endpoint.single_activation_clamped == null ? "N/A" : endpoint.single_activation_clamped ? "yes" : "no"}`],
      ["Build-mode code delta", endpoint.build_mode_output_delta_codes ?? "N/A"],
    ];
    metrics.forEach(([label, value], metricIndex) => {
      const metricY = cardY + 50 + metricIndex * 24;
      ctx.fillStyle = "#617083";
      ctx.font = "10px Inter, Arial, sans-serif";
      ctx.fillText(label, endpointX + 14, metricY, 132);
      ctx.fillStyle = "#17202c";
      ctx.font = "700 11px Inter, Arial, sans-serif";
      ctx.fillText(String(value), endpointX + 150, metricY, endpointWidth - 164);
    });
  });

  const summaryY = rankY + 438;
  ctx.fillStyle = "#17202c";
  ctx.font = "700 15px Inter, Arial, sans-serif";
  ctx.fillText("Exact portfolio", rankX, summaryY);
  const summary = [
    ["Channels", formatNumber(witness.assessed_channel_count)],
    ["Term assignments", formatNumber(witness.witness_assignment_count)],
    ["Fixed-point executions", formatNumber(witness.fixed_point_endpoint_evaluation_count)],
    ["Build-mode differences", formatNumber(witness.build_mode_divergent_endpoint_count)],
    ["Default / single clamps", `${formatNumber(witness.default_activation_clamped_endpoint_count)} / ${formatNumber(witness.single_activation_clamped_endpoint_count)}`],
  ];
  summary.forEach(([label, value], index) => {
    const boxX = rankX + index * 132;
    ctx.fillStyle = "#eef3f5";
    ctx.fillRect(boxX, summaryY + 16, 120, 61);
    ctx.fillStyle = "#617083";
    ctx.font = "9px Inter, Arial, sans-serif";
    ctx.fillText(label, boxX + 8, summaryY + 35, 104);
    ctx.fillStyle = "#17202c";
    ctx.font = "700 13px Inter, Arial, sans-serif";
    ctx.fillText(String(value), boxX + 8, summaryY + 59, 104);
  });
  ctx.fillStyle = "#617083";
  ctx.font = "10px ui-monospace, SFMono-Regular, Consolas, monospace";
  ctx.fillText(`pattern ${channel.witness_pattern_sha256}`, rankX, summaryY + 103, rankWidth);
  ctx.fillText(`ledger  ${row.witness_ledger_sha256}`, rankX, summaryY + 121, rankWidth);
  ctx.font = "10px Inter, Arial, sans-serif";
  ctx.fillText("Per-channel full-valid synthetic witness; not a simultaneous full-model input or observed runtime execution.", rankX, summaryY + 148, width - rankX * 2);
  ctx.fillText(`Schema ${witness.schema} / source ${witness.source_commit}`, rankX, summaryY + 166, width - rankX * 2);
  return canvas;
}

function renderChannelVitalityCanvas(analysis, filename) {
  const vitality = analysis.channel_vitality || {};
  const rowsByIndex = new Map((vitality.ops || []).map((row) => [row.op_index, row]));
  const rows = (vitality.vitality_ranking_op_indices || []).map((index) => rowsByIndex.get(index)).filter(Boolean);
  const topRow = rows[0] || (vitality.ops || []).find((row) => row.assessment_status === "assessed");
  const { canvas, ctx, width, y } = createExportCanvas(
    filename,
    "Quantized Channel Vitality Atlas",
    "Exact monotone endpoint output hulls under pinned default and TFLITE_SINGLE_ROUNDING algebra.",
    1180,
    800,
  );
  if (!topRow) {
    drawEmptyState(ctx, "No quantized channel-vitality proof was assessed.", 48, y);
    return canvas;
  }
  const metrics = [
    ["Assessed channels", formatNumber(vitality.assessed_channel_count)],
    ["Dual-mode constant", formatNumber(vitality.dual_mode_constant_output_channel_count)],
    ["Variable but constant", formatNumber(vitality.nonconstant_accumulator_dual_mode_constant_channel_count)],
    ["Mode-dependent", formatNumber(vitality.mode_dependent_constant_output_channel_count)],
    ["Sign-locked", formatNumber(Number(vitality.post_bias_negative_locked_channel_count || 0) + Number(vitality.post_bias_positive_locked_channel_count || 0))],
  ];
  metrics.forEach(([label, value], index) => {
    const x = 48 + index * 216;
    ctx.fillStyle = index === 2 ? "#f8eeee" : index === 3 ? "#fbf4e8" : "#eef5f3";
    ctx.fillRect(x, y + 12, 202, 66);
    ctx.fillStyle = "#617083";
    ctx.font = "10px Inter, Arial, sans-serif";
    ctx.fillText(label, x + 11, y + 34, 180);
    ctx.fillStyle = "#17202c";
    ctx.font = "700 18px Inter, Arial, sans-serif";
    ctx.fillText(value, x + 11, y + 60, 180);
  });

  const histX = 48;
  const histY = y + 125;
  const histWidth = 630;
  const histHeight = 300;
  ctx.fillStyle = "#17202c";
  ctx.font = "700 15px Inter, Arial, sans-serif";
  ctx.fillText("Inclusive output-code span distribution", histX, histY - 15);
  ctx.fillStyle = "#f3f6f8";
  ctx.fillRect(histX, histY, histWidth, histHeight);
  const histogram = vitality.span_histogram || [];
  const maxCount = Math.max(1, ...histogram.flatMap((bin) => [Number(bin.default_channel_count || 0), Number(bin.single_rounding_channel_count || 0)]));
  const slot = histWidth / Math.max(1, histogram.length);
  histogram.forEach((bin, index) => {
    const defaultHeight = Math.sqrt(Number(bin.default_channel_count || 0) / maxCount) * (histHeight - 58);
    const singleHeight = Math.sqrt(Number(bin.single_rounding_channel_count || 0) / maxCount) * (histHeight - 58);
    const x = histX + index * slot;
    ctx.fillStyle = "#2f7b82";
    ctx.fillRect(x + slot * 0.18, histY + histHeight - 35 - defaultHeight, slot * 0.27, defaultHeight);
    ctx.fillStyle = "#d69a34";
    ctx.fillRect(x + slot * 0.51, histY + histHeight - 35 - singleHeight, slot * 0.27, singleHeight);
    ctx.fillStyle = "#536273";
    ctx.font = "10px Inter, Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(bin.label, x + slot / 2, histY + histHeight - 17, slot - 4);
    ctx.font = "9px Inter, Arial, sans-serif";
    ctx.fillText(`${formatNumber(bin.default_channel_count)} / ${formatNumber(bin.single_rounding_channel_count)}`, x + slot / 2, histY + histHeight - 41 - Math.max(defaultHeight, singleHeight), slot - 5);
  });
  ctx.textAlign = "left";
  ctx.strokeStyle = "#536273";
  ctx.strokeRect(histX, histY, histWidth, histHeight);
  ctx.fillStyle = "#2f7b82";
  ctx.fillRect(histX + 10, histY + 10, 10, 10);
  ctx.fillStyle = "#617083";
  ctx.font = "10px Inter, Arial, sans-serif";
  ctx.fillText("default", histX + 25, histY + 19);
  ctx.fillStyle = "#d69a34";
  ctx.fillRect(histX + 85, histY + 10, 10, 10);
  ctx.fillStyle = "#617083";
  ctx.fillText("single rounding (sqrt count scale)", histX + 100, histY + 19);

  const rankX = 720;
  const rankY = histY;
  const rankWidth = width - rankX - 48;
  ctx.fillStyle = "#17202c";
  ctx.font = "700 15px Inter, Arial, sans-serif";
  ctx.fillText("Ranked constrained operators", rankX, rankY - 15);
  rows.slice(0, 10).forEach((row, index) => {
    const rowY = rankY + index * 30;
    ctx.fillStyle = index % 2 ? "#f8fafb" : "#f1f5f7";
    ctx.fillRect(rankX, rowY, rankWidth, 25);
    ctx.fillStyle = "#334255";
    ctx.font = index === 0 ? "700 10px Inter, Arial, sans-serif" : "10px Inter, Arial, sans-serif";
    ctx.fillText(`#${padOp(row.op_index)} ${row.op_name}`, rankX + 7, rowY + 16, 142);
    ctx.fillStyle = Number(row.nonconstant_accumulator_dual_mode_constant_channel_count || 0) ? "#bd4448" : Number(row.mode_dependent_constant_output_channel_count || 0) ? "#d69a34" : "#2f7b82";
    ctx.textAlign = "right";
    ctx.fillText(`${formatNumber(row.dual_mode_constant_output_channel_count)} / ${formatNumber(row.mode_dependent_constant_output_channel_count)} / span ${row.minimum_default_inclusive_code_span ?? "N/A"}`, rankX + rankWidth - 7, rowY + 16);
    ctx.textAlign = "left";
  });

  const coordinateY = histY + histHeight + 50;
  ctx.fillStyle = "#17202c";
  ctx.font = "700 15px Inter, Arial, sans-serif";
  ctx.fillText("Exact proof coordinates", 48, coordinateY);
  const coordinates = rows.flatMap((row) => (row.top_channels || [])
    .filter((channel) => channel.dual_mode_constant || channel.mode_dependent_constant)
    .map((channel) => ({ row, channel }))).slice(0, 8);
  coordinates.forEach(({ row, channel }, index) => {
    const column = index % 4;
    const line = Math.floor(index / 4);
    const x = 48 + column * 270;
    const cardY = coordinateY + 17 + line * 82;
    ctx.fillStyle = channel.mode_dependent_constant ? "#fbf4e8" : channel.accumulator_span_decimal !== "0" ? "#f8eeee" : "#eef5f3";
    ctx.fillRect(x, cardY, 255, 70);
    ctx.fillStyle = "#17202c";
    ctx.font = "700 10px Inter, Arial, sans-serif";
    ctx.fillText(`#${padOp(row.op_index)} ${row.op_name} / ch ${channel.channel_index}`, x + 9, cardY + 17, 237);
    ctx.font = "10px ui-monospace, SFMono-Regular, Consolas, monospace";
    ctx.fillText(`D ${channel.default_minimum_output_code}..${channel.default_maximum_output_code} span ${channel.default_inclusive_code_span}`, x + 9, cardY + 37, 237);
    ctx.fillText(`S ${channel.single_minimum_output_code}..${channel.single_maximum_output_code} span ${channel.single_inclusive_code_span}`, x + 9, cardY + 54, 237);
  });
  ctx.fillStyle = "#617083";
  ctx.font = "10px Inter, Arial, sans-serif";
  ctx.fillText("Span=1 is exact over one full-valid receptive-field legal-code domain; larger spans are interval-hull upper bounds, not observed reachability.", 48, 766, width - 96);
  ctx.fillText(`Schema ${vitality.schema} / source ${vitality.source_commit}`, 48, 784, width - 96);
  return canvas;
}

function renderContractMigrationCanvas(analysis, filename) {
  const migration = analysis.contract_migration || {};
  const ranked = analysis.quantization_lattice?.domain_escape_ranking_op_indices || [];
  const row = ranked.map((index) => (migration.migrations || []).find((item) => item.source_add_op_index === index)).find(Boolean)
    || migration.migrations?.[0];
  const scenario = row?.scenarios?.find((item) => item.design === "globally_finest_minimum_containment")
    || row?.scenarios?.[0];
  const { canvas, ctx, width, y } = createExportCanvas(
    filename,
    "Contract Migration Impact",
    "Counterfactual direct parameter regeneration; downstream reachability is a structural behavior radius only.",
    1180,
    760,
  );
  if (!row || !scenario) {
    drawEmptyState(ctx, "No residual contract-migration scenario was assessed.", 48, y);
    return canvas;
  }
  const plotX = 54;
  const plotY = y + 22;
  const plotWidth = 650;
  const channelHeight = 250;
  const channels = scenario.kernel_consumers.flatMap((consumer) =>
    consumer.channel_current_shifts.map((currentShift, index) => ({
      shiftDelta: Math.abs(Number(consumer.channel_candidate_shifts[index]) - Number(currentShift)),
      biasError: Number(consumer.channel_bias_rebase_error_current_steps[index] || 0),
    })));
  ctx.fillStyle = "#f3f6f8";
  ctx.fillRect(plotX, plotY, plotWidth, channelHeight);
  const bins = Math.max(1, Math.min(plotWidth, channels.length));
  const binSize = channels.length / bins;
  const maxShift = Math.max(1, ...channels.map((item) => item.shiftDelta));
  const maxError = Math.max(1, ...channels.map((item) => item.biasError));
  for (let bin = 0; bin < bins; bin += 1) {
    const start = Math.floor(bin * binSize);
    const end = Math.max(start + 1, Math.floor((bin + 1) * binSize));
    const slice = channels.slice(start, end);
    const shift = Math.max(0, ...slice.map((item) => item.shiftDelta));
    const error = Math.max(0, ...slice.map((item) => item.biasError));
    const x = plotX + bin / bins * plotWidth;
    const barWidth = Math.max(1, plotWidth / bins);
    ctx.fillStyle = mixHex("#f4e4e4", "#bd4448", error / maxError);
    ctx.fillRect(x, plotY + channelHeight / 2, barWidth, channelHeight / 2);
    ctx.fillStyle = "#28666e";
    ctx.fillRect(x, plotY + channelHeight / 2 - shift / maxShift * (channelHeight / 2 - 4), barWidth, shift / maxShift * (channelHeight / 2 - 4));
  }
  ctx.strokeStyle = "#536273";
  ctx.strokeRect(plotX, plotY, plotWidth, channelHeight);
  ctx.fillStyle = "#536273";
  ctx.font = "12px Inter, Arial, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(`${formatNumber(channels.length)} direct kernel channel scenarios`, plotX + plotWidth / 2, plotY + channelHeight + 25);
  ctx.textAlign = "left";
  ctx.fillText(`max bias error ${maxError.toFixed(6)} current steps`, plotX + 8, plotY + 16);

  const depthCounts = new Map();
  (row.affected_ops || []).forEach((op) => depthCounts.set(op.minimum_edge_depth, (depthCounts.get(op.minimum_edge_depth) || 0) + 1));
  const depthY = plotY + 330;
  const depthHeight = 150;
  const maxDepth = Math.max(1, ...depthCounts.keys());
  const maxCount = Math.max(1, ...depthCounts.values());
  const slot = plotWidth / maxDepth;
  ctx.fillStyle = "#f3f6f8";
  ctx.fillRect(plotX, depthY, plotWidth, depthHeight);
  for (let depth = 1; depth <= maxDepth; depth += 1) {
    const height = Number(depthCounts.get(depth) || 0) / maxCount * (depthHeight - 8);
    ctx.fillStyle = depth === 1 ? "#bd4448" : "#28666e";
    ctx.fillRect(plotX + (depth - 1) * slot + 2, depthY + depthHeight - height, Math.max(2, slot - 4), height);
  }
  ctx.strokeStyle = "#536273";
  ctx.strokeRect(plotX, depthY, plotWidth, depthHeight);
  ctx.fillStyle = "#536273";
  ctx.textAlign = "center";
  ctx.fillText(`${row.reachable_downstream_op_count} reachable ops / ${maxDepth} edge depths`, plotX + plotWidth / 2, depthY + depthHeight + 25);

  const detailX = 755;
  const detailWidth = width - detailX - 48;
  ctx.textAlign = "left";
  ctx.fillStyle = "#17202c";
  ctx.font = "700 20px Inter, Arial, sans-serif";
  ctx.fillText(`#${padOp(row.source_add_op_index)} ADD`, detailX, plotY + 8);
  const metrics = [
    ["Candidate contract", `${Number(scenario.candidate_output_scale).toPrecision(7)} / zp ${scenario.candidate_output_zero_point}`],
    ["Scale / zero-point delta", `${Number(scenario.scale_ratio_to_current).toFixed(6)}x / ${Number(scenario.signed_zero_point_delta) >= 0 ? "+" : ""}${scenario.signed_zero_point_delta}`],
    ["Direct consumers", `${row.direct_consumer_count} (${scenario.assessed_consumer_count} assessed)`],
    ["Q0.31 encodings / shifts", `${formatNumber(scenario.multiplier_encoding_changed_channel_count)} / ${formatNumber(scenario.multiplier_shift_changed_channel_count)}`],
    ["Bias codes / INT32 overflow", `${formatNumber(scenario.bias_code_changed_channel_count)} / ${formatNumber(scenario.bias_int32_overflow_channel_count)}`],
    ["ADD parameter encodings", formatNumber(scenario.add_parameter_encoding_changed_count)],
    ["Structural radius", `${row.reachable_downstream_op_count} ops / depth ${row.maximum_downstream_edge_depth}`],
    ["Schema", `${migration.schema} / ${migration.method_version}`],
    ["Pinned source", migration.source_commit],
  ];
  metrics.forEach(([label, value], index) => {
    const metricY = plotY + 56 + index * 50;
    ctx.fillStyle = "#617083";
    ctx.font = "11px Inter, Arial, sans-serif";
    ctx.fillText(label, detailX, metricY);
    ctx.fillStyle = "#17202c";
    ctx.font = "700 13px Inter, Arial, sans-serif";
    ctx.fillText(String(value), detailX, metricY + 19, detailWidth);
    ctx.strokeStyle = "#d9e0ea";
    ctx.beginPath();
    ctx.moveTo(detailX, metricY + 29);
    ctx.lineTo(detailX + detailWidth, metricY + 29);
    ctx.stroke();
  });
  return canvas;
}

function renderResidualStepResponseCanvas(analysis, filename) {
  const response = analysis.residual_step_response || {};
  const topIndex = response.retention_cost_ranking_op_indices?.[0];
  const row = (response.residual_adds || []).find((item) => item.op_index === topIndex)
    || (response.residual_adds || []).find((item) => item.assessment_status === "assessed");
  const contract = row?.contracts?.find((item) => item.design === "globally_finest_minimum_containment")
    || row?.contracts?.[0];
  const { canvas, ctx, width, y } = createExportCanvas(
    filename,
    "Residual Step Response",
    "Exact legal-code branch visibility; containment is compared with current rounded clamping without activation-distribution claims.",
    1180,
    760,
  );
  if (!row || !contract) {
    drawEmptyState(ctx, "No residual step-response contract was assessed.", 48, y);
    return canvas;
  }
  const fieldX = 54;
  const fieldY = y + 20;
  const fieldSize = 470;
  const classes = exportInfluenceClasses(analysis, row, contract);
  const field = document.createElement("canvas");
  field.width = 255;
  field.height = 255;
  const fieldContext = field.getContext("2d");
  const image = fieldContext.createImageData(255, 255);
  const colors = [[94, 224, 183], [93, 169, 255], [246, 184, 95], [240, 109, 118]];
  classes.forEach((classIndex, index) => {
    const color = colors[classIndex];
    const offset = index * 4;
    image.data[offset] = color[0];
    image.data[offset + 1] = color[1];
    image.data[offset + 2] = color[2];
    image.data[offset + 3] = 255;
  });
  fieldContext.putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(field, fieldX, fieldY, fieldSize, fieldSize);
  ctx.strokeStyle = "#536273";
  ctx.strokeRect(fieldX, fieldY, fieldSize, fieldSize);
  ctx.fillStyle = "#536273";
  ctx.font = "12px Inter, Arial, sans-serif";
  ctx.fillText("input 1 code ->", fieldX, fieldY + fieldSize + 24);
  ctx.save();
  ctx.translate(fieldX - 22, fieldY + fieldSize);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("input 0 code ->", 0, 0);
  ctx.restore();
  const legend = [["both", "#5ee0b7"], ["input 0", "#5da9ff"], ["input 1", "#f6b85f"], ["neither", "#f06d76"]];
  legend.forEach(([label, color], index) => {
    const x = fieldX + index * 112;
    ctx.fillStyle = color;
    ctx.fillRect(x, fieldY + fieldSize + 44, 14, 10);
    ctx.fillStyle = "#536273";
    ctx.fillText(label, x + 20, fieldY + fieldSize + 53);
  });

  const detailX = 590;
  const detailWidth = width - detailX - 48;
  ctx.fillStyle = "#17202c";
  ctx.font = "700 20px Inter, Arial, sans-serif";
  ctx.fillText(`#${padOp(row.op_index)} ADD`, detailX, fieldY + 8);
  ctx.fillStyle = "#617083";
  ctx.font = "12px Inter, Arial, sans-serif";
  ctx.fillText("globally finest containment", detailX, fieldY + 30);
  const current = row.contracts.find((item) => item.design === "current_artifact_contract");
  const metrics = [
    ["Output contract", `${Number(contract.output_scale).toPrecision(8)} / zp ${contract.output_zero_point}`],
    ["Rounded clamp pairs", `${formatNumber(current?.rounded_projection_clamp_pair_count || 0)} -> ${formatNumber(contract.rounded_projection_clamp_pair_count)}`],
    ["Silent transitions", `${formatNumber(current?.silent_transition_count || 0)} -> ${formatNumber(contract.silent_transition_count)}`],
    ["Containment trade", `${contract.additional_silent_transitions_vs_current >= 0 ? "+" : ""}${formatNumber(contract.additional_silent_transitions_vs_current)} silent`],
    ["Both branches visible", `${formatPercent(contract.both_branches_visible_ratio)} of joint cells`],
    ["Neither branch visible", `${formatPercent(contract.neither_branch_visible_ratio)} of joint cells`],
    ["Exact portfolio", `${formatNumber(response.total_transition_count)} transitions / ${formatNumber(response.total_joint_interior_cell_count)} joint cells`],
    ["Transition ledger", contract.transition_ledger_sha256],
    ["Schema", `${response.schema} / ${response.method_version}`],
  ];
  metrics.forEach(([label, value], index) => {
    const metricY = fieldY + 68 + index * 51;
    ctx.fillStyle = "#617083";
    ctx.font = "11px Inter, Arial, sans-serif";
    ctx.fillText(label, detailX, metricY);
    ctx.fillStyle = "#17202c";
    ctx.font = "700 13px Inter, Arial, sans-serif";
    ctx.fillText(String(value), detailX, metricY + 19, detailWidth);
    ctx.strokeStyle = "#d9e0ea";
    ctx.beginPath();
    ctx.moveTo(detailX, metricY + 29);
    ctx.lineTo(detailX + detailWidth, metricY + 29);
    ctx.stroke();
  });
  return canvas;
}

function renderResidualContractDistortionCanvas(analysis, filename) {
  const distortion = analysis.residual_contract_distortion || {};
  const topIndex = distortion.distortion_ranking_op_indices?.[0];
  const row = (distortion.residual_adds || []).find((item) => item.op_index === topIndex)
    || (distortion.residual_adds || []).find((item) => item.assessment_status === "assessed");
  const scenario = row?.scenarios?.find((item) => item.design === "globally_finest_minimum_containment")
    || row?.scenarios?.[0];
  const { canvas, ctx, width, y } = createExportCanvas(
    filename,
    "Residual Contract Distortion Atlas",
    "Exact candidate-minus-current represented-value displacement over the uniform legal input-code domain.",
    1180,
    760,
  );
  if (!row || !scenario) {
    drawEmptyState(ctx, "No residual contract-distortion scenario was assessed.", 48, y);
    return canvas;
  }
  const contract = exportDistortionContract(analysis, row, scenario);
  const fieldX = 54;
  const fieldY = y + 20;
  const fieldSize = 470;
  const field = document.createElement("canvas");
  field.width = 256;
  field.height = 256;
  const fieldContext = field.getContext("2d");
  const image = fieldContext.createImageData(256, 256);
  const clip = Math.max(1e-12, Number(scenario.p99_absolute_contract_delta_current_steps));
  let pixel = 0;
  for (let q0 = contract.input0.qmin; q0 <= contract.input0.qmax; q0 += 1) {
    for (let q1 = contract.input1.qmin; q1 <= contract.input1.qmax; q1 += 1) {
      const pair = exportDistortionPair(q0, q1, contract);
      const magnitude = Math.min(1, Math.abs(pair.deltaSteps) / clip) ** 0.55;
      const color = exportMixRgb(
        [232, 238, 242],
        pair.deltaSteps < 0 ? [56, 137, 210] : [224, 151, 45],
        magnitude,
      );
      const offset = pixel * 4;
      image.data[offset] = color[0];
      image.data[offset + 1] = color[1];
      image.data[offset + 2] = color[2];
      image.data[offset + 3] = 255;
      pixel += 1;
    }
  }
  fieldContext.putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(field, fieldX, fieldY, fieldSize, fieldSize);
  ctx.strokeStyle = "#536273";
  ctx.strokeRect(fieldX, fieldY, fieldSize, fieldSize);
  ctx.fillStyle = "#536273";
  ctx.font = "12px Inter, Arial, sans-serif";
  ctx.fillText("input 1 code ->", fieldX, fieldY + fieldSize + 24);
  ctx.save();
  ctx.translate(fieldX - 22, fieldY + fieldSize);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("input 0 code ->", 0, 0);
  ctx.restore();
  [["negative", "#3889d2"], ["near zero", "#e8eef2"], ["positive", "#e0972d"]].forEach(([label, color], index) => {
    const x = fieldX + index * 145;
    ctx.fillStyle = color;
    ctx.fillRect(x, fieldY + fieldSize + 44, 14, 10);
    ctx.fillStyle = "#536273";
    ctx.fillText(label, x + 20, fieldY + fieldSize + 53);
  });
  const detailX = 590;
  const detailWidth = width - detailX - 48;
  ctx.fillStyle = "#17202c";
  ctx.font = "700 20px Inter, Arial, sans-serif";
  ctx.fillText(`#${padOp(row.op_index)} ADD`, detailX, fieldY + 8);
  ctx.fillStyle = "#617083";
  ctx.font = "12px Inter, Arial, sans-serif";
  ctx.fillText("globally finest containment versus artifact contract", detailX, fieldY + 30);
  const metrics = [
    ["Candidate contract", `${Number(scenario.candidate_output_scale).toPrecision(8)} / zp ${scenario.candidate_output_zero_point}`],
    ["Exact pair comparisons", formatNumber(scenario.enumerated_pair_count)],
    ["Clamp rescue", `${formatNumber(scenario.rescued_current_clamp_pair_count)} / ${formatNumber(scenario.candidate_clamped_pair_count)} candidate clamps`],
    ["RMS / p99 / maximum", `${Number(scenario.root_mean_square_contract_delta_current_steps).toFixed(6)} / ${Number(scenario.p99_absolute_contract_delta_current_steps).toFixed(6)} / ${Number(scenario.maximum_absolute_contract_delta_current_steps).toFixed(6)} current steps`],
    ["Ideal-error direction", `${formatNumber(scenario.ideal_error_improved_pair_count)} improve / ${formatNumber(scenario.ideal_error_worsened_pair_count)} worsen / ${formatNumber(scenario.ideal_error_equal_within_tolerance_pair_count)} equal`],
    ["Sign-class changes", formatNumber(scenario.sign_class_changed_pair_count)],
    ["Mean ideal-error delta", Number(scenario.signed_mean_absolute_ideal_error_delta).toExponential(7)],
    ["Pair ledger SHA-256", scenario.pair_ledger_sha256],
    ["Schema", `${distortion.schema} / ${distortion.method_version}`],
  ];
  metrics.forEach(([label, value], index) => {
    const metricY = fieldY + 68 + index * 51;
    ctx.fillStyle = "#617083";
    ctx.font = "11px Inter, Arial, sans-serif";
    ctx.fillText(label, detailX, metricY);
    ctx.fillStyle = "#17202c";
    ctx.font = "700 13px Inter, Arial, sans-serif";
    ctx.fillText(String(value), detailX, metricY + 19, detailWidth);
    ctx.strokeStyle = "#d9e0ea";
    ctx.beginPath();
    ctx.moveTo(detailX, metricY + 29);
    ctx.lineTo(detailX + detailWidth, metricY + 29);
    ctx.stroke();
  });
  return canvas;
}

function exportDistortionContract(analysis, row, scenario) {
  const op = (analysis.ops || []).find((item) => item.index === row.op_index && item.name === "ADD");
  if (!op) throw new Error(`ADD #${row.op_index} is unavailable for distortion export.`);
  const tensorContract = (index) => {
    const tensor = (analysis.tensors || []).find((item) => item.index === index);
    if (!tensor) throw new Error(`Tensor T${index} is unavailable for distortion export.`);
    return {
      qmin: tensor.dtype === "INT8" ? -128 : 0,
      qmax: tensor.dtype === "INT8" ? 127 : 255,
      scale: Number(tensor.scale_sample?.[0]),
      zeroPoint: Number(tensor.zero_point_sample?.[0]),
    };
  };
  const current = tensorContract(op.outputs[0]);
  return {
    input0: tensorContract(op.inputs[0]),
    input1: tensorContract(op.inputs[1]),
    current,
    candidate: { ...current, scale: Number(scenario.candidate_output_scale), zeroPoint: Number(scenario.candidate_output_zero_point) },
  };
}

function exportDistortionPair(q0, q1, contract) {
  const ideal = (q0 - contract.input0.zeroPoint) * contract.input0.scale
    + (q1 - contract.input1.zeroPoint) * contract.input1.scale;
  const represented = (output) => {
    const raw = roundTiesAway(ideal / output.scale) + output.zeroPoint;
    const code = Math.max(output.qmin, Math.min(output.qmax, raw));
    return (code - output.zeroPoint) * output.scale;
  };
  return { deltaSteps: (represented(contract.candidate) - represented(contract.current)) / contract.current.scale };
}

function exportMixRgb(from, to, ratio) {
  return from.map((value, index) => Math.round(value + (to[index] - value) * ratio));
}

function exportInfluenceClasses(analysis, row, contract) {
  const op = (analysis.ops || []).find((item) => item.index === row.op_index && item.name === "ADD");
  const tensorContract = (index) => {
    const tensor = (analysis.tensors || []).find((item) => item.index === index);
    const qmin = tensor?.dtype === "INT8" ? -128 : 0;
    const qmax = tensor?.dtype === "INT8" ? 127 : 255;
    return { qmin, qmax, scale: Number(tensor?.scale_sample?.[0]), zeroPoint: Number(tensor?.zero_point_sample?.[0]) };
  };
  const input0 = tensorContract(op.inputs[0]);
  const input1 = tensorContract(op.inputs[1]);
  const output = { ...tensorContract(op.outputs[0]), scale: Number(contract.output_scale), zeroPoint: Number(contract.output_zero_point) };
  const projected = (q0, q1) => {
    const real = (q0 - input0.zeroPoint) * input0.scale + (q1 - input1.zeroPoint) * input1.scale;
    const scaled = real / output.scale;
    const rounded = scaled >= 0 ? Math.floor(scaled + 0.5) : Math.ceil(scaled - 0.5);
    return Math.max(output.qmin, Math.min(output.qmax, rounded + output.zeroPoint));
  };
  const classes = new Uint8Array(255 * 255);
  let index = 0;
  for (let q0 = input0.qmin; q0 < input0.qmax; q0 += 1) {
    for (let q1 = input1.qmin; q1 < input1.qmax; q1 += 1) {
      const base = projected(q0, q1);
      const visible0 = projected(q0 + 1, q1) !== base;
      const visible1 = projected(q0, q1 + 1) !== base;
      classes[index] = visible0 ? (visible1 ? 0 : 1) : (visible1 ? 2 : 3);
      index += 1;
    }
  }
  return classes;
}

function renderRequantizationFidelityCanvas(analysis, filename) {
  const fidelity = analysis.requantization_fidelity || {};
  const byIndex = new Map((fidelity.ops || []).map((row) => [row.op_index, row]));
  const row = (fidelity.fidelity_ranking_op_indices || []).map((index) => byIndex.get(index)).find((item) => item?.assessment_status === "assessed")
    || (fidelity.ops || []).find((item) => item.assessment_status === "assessed");
  const { canvas, ctx, y } = createExportCanvas(
    filename,
    "Requantization Fidelity Lab",
    "Pinned TFLite Q0.31 multiplier encoding drift and conservative pre-clamp rounding bounds.",
    1180,
    760,
  );
  if (!row) {
    drawEmptyState(ctx, "No quantized convolution-family requantization contract was assessed.", 48, y);
    return canvas;
  }
  const plotX = 54;
  const plotY = y + 16;
  const plotWidth = 590;
  const plotHeight = 470;
  const channels = row.assessed_channel_count;
  const columns = Math.min(64, Math.max(4, Math.ceil(Math.sqrt(channels * 1.75))));
  const rows = Math.ceil(channels / columns);
  const gap = channels > 512 ? 1 : 2;
  const cellWidth = (plotWidth - gap * (columns - 1)) / columns;
  const cellHeight = (plotHeight - gap * (rows - 1)) / rows;
  for (let channel = 0; channel < channels; channel += 1) {
    const column = channel % columns;
    const rowIndex = Math.floor(channel / columns);
    const drift = Number(row.channel_encoding_drift_bound_codes[channel] || 0);
    const normalized = Math.max(0, Math.min(1, (Math.log10(drift + 1e-15) + 15) / 15));
    ctx.fillStyle = row.channel_default_pre_shift_int32_safe[channel]
      ? mixHex("#277e99", "#e2a634", normalized)
      : "#d63c52";
    ctx.fillRect(
      plotX + column * (cellWidth + gap),
      plotY + rowIndex * (cellHeight + gap),
      Math.max(1, cellWidth),
      Math.max(1, cellHeight),
    );
  }
  ctx.strokeStyle = "#dbe3e8";
  ctx.strokeRect(plotX - 1, plotY - 1, plotWidth + 2, plotHeight + 2);
  ctx.fillStyle = "#617083";
  ctx.font = "12px Inter, Arial, sans-serif";
  ctx.fillText(`${formatNumber(channels)} output channels · color = encoding-only output-code drift`, plotX, plotY + plotHeight + 24);
  [["1e-12", .2], ["1e-9", .4], ["1e-6", .6], ["1e-3", .8]].forEach(([label, amount], index) => {
    const legendX = plotX + index * 142;
    ctx.fillStyle = mixHex("#277e99", "#e2a634", amount);
    ctx.fillRect(legendX, plotY + plotHeight + 42, 16, 10);
    ctx.fillStyle = "#617083";
    ctx.fillText(label, legendX + 22, plotY + plotHeight + 51);
  });

  const detailX = 700;
  const detailWidth = 420;
  ctx.fillStyle = "#17222c";
  ctx.font = "600 18px Inter, Arial, sans-serif";
  ctx.fillText(`#${padOp(row.op_index)} ${row.op_name}`, detailX, plotY + 8);
  ctx.fillStyle = "#617083";
  ctx.font = "12px Inter, Arial, sans-serif";
  ctx.fillText(`${formatNumber(row.assessed_channel_count)} channels · ${row.weight_scale_mode.replaceAll("_", " ")}`, detailX, plotY + 30);
  const metrics = [
    ["Q0.31 shift range", `${signedShift(row.minimum_shift)} .. ${signedShift(row.maximum_shift)}`],
    ["Maximum relative error", Number(row.maximum_relative_multiplier_error).toExponential(6)],
    ["Maximum error ppm", Number(row.maximum_multiplier_error_ppm).toExponential(6)],
    ["Maximum encoding drift", `${Number(row.maximum_encoding_drift_bound_codes).toExponential(6)} codes`],
    ["Default double bound", `${Number(row.maximum_default_double_rounding_bound_codes).toFixed(9)} codes`],
    ["Single-rounding bound", `${Number(row.maximum_single_rounding_bound_codes).toFixed(9)} codes`],
    ["Default pre-shift overflow", `${formatNumber(row.default_pre_shift_overflow_channel_count)} channels`],
    ["Build encoding divergence", `${formatNumber(row.single_rounding_encoding_divergence_channel_count)} channels`],
  ];
  metrics.forEach(([label, value], index) => {
    const metricY = plotY + 72 + index * 48;
    ctx.fillStyle = "#617083";
    ctx.font = "11px Inter, Arial, sans-serif";
    ctx.fillText(label, detailX, metricY);
    ctx.fillStyle = "#17222c";
    ctx.font = "600 15px ui-monospace, SFMono-Regular, Consolas, monospace";
    ctx.fillText(value, detailX, metricY + 19, detailWidth);
  });
  const witness = row.worst_channel;
  if (witness) {
    ctx.fillStyle = "#617083";
    ctx.font = "11px Inter, Arial, sans-serif";
    ctx.fillText("Worst encoding witness", detailX, plotY + 468);
    ctx.fillStyle = "#17222c";
    ctx.font = "600 13px ui-monospace, SFMono-Regular, Consolas, monospace";
    ctx.fillText(`ch ${witness.channel_index} · q ${witness.quantized_multiplier} · shift ${signedShift(witness.shift)}`, detailX, plotY + 488, detailWidth);
  }
  ctx.fillStyle = "#617083";
  ctx.font = "10px ui-monospace, SFMono-Regular, Consolas, monospace";
  ctx.fillText(`${fidelity.schema} · ${fidelity.source_commit}`, detailX, plotY + 520, detailWidth);
  ctx.fillText(`ledger ${row.channel_ledger_sha256}`, detailX, plotY + 540, detailWidth);
  return canvas;
}

function signedShift(value) {
  return value == null ? "N/A" : `${Number(value) >= 0 ? "+" : ""}${value}`;
}

function renderAccumulatorAtlasCanvas(analysis, filename) {
  const atlas = analysis.accumulator_atlas || {};
  const byIndex = new Map((atlas.ops || []).map((row) => [row.op_index, row]));
  const row = (atlas.headroom_ranking_op_indices || []).map((index) => byIndex.get(index)).find((item) => item?.assessment_status === "assessed")
    || (atlas.ops || []).find((item) => item.assessment_status === "assessed");
  const { canvas, ctx, width, y } = createExportCanvas(
    filename,
    "Accumulator Headroom Atlas",
    "Exact legal-code envelope from stored centered weights and bias under pinned TFLite reference integer algebra.",
    1180,
    760,
  );
  if (!row) {
    drawEmptyState(ctx, "No constant 8-bit convolution-family accumulator was assessed.", 48, y);
    return canvas;
  }
  const plotX = 54;
  const plotY = y + 16;
  const plotWidth = 590;
  const plotHeight = 430;
  const channels = Number(row.assessed_channel_count || 0);
  const columns = Math.min(64, Math.max(4, Math.ceil(Math.sqrt(channels * 1.75))));
  const rows = Math.ceil(channels / columns);
  const gap = channels > 512 ? 1 : 2;
  const cellWidth = (plotWidth - gap * (columns - 1)) / columns;
  const cellHeight = (plotHeight - gap * (rows - 1)) / rows;
  for (let channel = 0; channel < channels; channel += 1) {
    const column = channel % columns;
    const rowIndex = Math.floor(channel / columns);
    const bits = Number(row.channel_required_signed_bits[channel] || 0);
    ctx.fillStyle = accumulatorBitColor(bits);
    ctx.fillRect(
      plotX + column * (cellWidth + gap),
      plotY + rowIndex * (cellHeight + gap),
      Math.max(1, cellWidth),
      Math.max(1, cellHeight),
    );
  }
  ctx.strokeStyle = "#536273";
  ctx.strokeRect(plotX - 1, plotY - 1, plotWidth + 2, plotHeight + 2);
  ctx.fillStyle = "#617083";
  ctx.font = "12px Inter, Arial, sans-serif";
  ctx.fillText(`${formatNumber(channels)} output channels · color = required signed bits`, plotX, plotY + plotHeight + 24);
  [["≤20 bits", 20], ["24 bits", 24], ["28 bits", 28], [">32 overflow", 33]].forEach(([label, bits], index) => {
    const legendX = plotX + index * 142;
    ctx.fillStyle = accumulatorBitColor(bits);
    ctx.fillRect(legendX, plotY + plotHeight + 42, 16, 10);
    ctx.fillStyle = "#617083";
    ctx.fillText(label, legendX + 22, plotY + plotHeight + 51);
  });

  const detailX = 700;
  const detailWidth = width - detailX - 48;
  ctx.fillStyle = "#17202c";
  ctx.font = "700 20px Inter, Arial, sans-serif";
  ctx.fillText(`#${padOp(row.op_index)} ${row.op_name}`, detailX, plotY + 8);
  ctx.fillStyle = "#617083";
  ctx.font = "12px Inter, Arial, sans-serif";
  ctx.fillText(`${formatNumber(row.assessed_channel_count)} channels × ${formatNumber(row.accumulation_terms_per_channel)} terms`, detailX, plotY + 30);
  const metrics = [
    ["Exact envelope maximum", formatDecimalInteger(row.maximum_absolute_accumulator_decimal)],
    ["INT32 utilization", formatPercent(row.maximum_int32_ratio)],
    ["Required width / headroom", `${row.maximum_required_signed_bits} / ${row.minimum_int32_headroom_bits} bits`],
    ["Metadata-only comparison", `${formatDecimalInteger(row.metadata_only_magnitude_bound_decimal)} (${formatPercent(row.metadata_only_int32_ratio)})`],
    ["Exact bound tightening", `${Number(row.exact_tightening_factor).toFixed(2)}×`],
    ["Overflow channels", `${formatNumber(row.int32_overflow_channel_count)} / ${formatNumber(row.assessed_channel_count)}`],
    ["Worst channel", row.worst_channel ? `${row.worst_channel.channel_index}: [${formatDecimalInteger(row.worst_channel.accumulator_envelope_min_decimal)}, ${formatDecimalInteger(row.worst_channel.accumulator_envelope_max_decimal)}]` : "N/A"],
    ["Stored bias", row.worst_channel ? formatDecimalInteger(row.worst_channel.bias_decimal) : "N/A"],
  ];
  metrics.forEach(([label, value], index) => {
    const metricY = plotY + 72 + index * 48;
    ctx.fillStyle = "#617083";
    ctx.font = "11px Inter, Arial, sans-serif";
    ctx.fillText(label, detailX, metricY);
    ctx.fillStyle = "#17202c";
    ctx.font = "700 13px Inter, Arial, sans-serif";
    ctx.fillText(value, detailX, metricY + 19, detailWidth);
    ctx.strokeStyle = "#d9e0ea";
    ctx.beginPath();
    ctx.moveTo(detailX, metricY + 29);
    ctx.lineTo(detailX + detailWidth, metricY + 29);
    ctx.stroke();
  });
  ctx.fillStyle = "#617083";
  ctx.font = "10px Inter, Arial, sans-serif";
  ctx.fillText(`${atlas.schema} · ${atlas.source_commit}`, detailX, plotY + 485, detailWidth);
  ctx.fillText(`ledger ${row.channel_ledger_sha256}`, detailX, plotY + 505, detailWidth);
  return canvas;
}

function accumulatorBitColor(bits) {
  if (bits > 32) return "#d63c52";
  const bounded = Math.max(0, Math.min(1, Number(bits || 0) / 32));
  if (bounded < 0.68) return mixHex("#1f8977", "#ddb136", bounded / 0.68);
  return mixHex("#ddb136", "#db4b47", (bounded - 0.68) / 0.32);
}

function formatDecimalInteger(value) {
  try { return BigInt(value).toLocaleString("en-US"); } catch { return String(value ?? "N/A"); }
}

function renderResidualLatticeCanvas(analysis, filename) {
  const lattice = analysis.quantization_lattice || {};
  const rowByIndex = new Map((lattice.residual_adds || []).map((row) => [row.op_index, row]));
  const row = (lattice.domain_escape_ranking_op_indices || []).map((index) => rowByIndex.get(index)).find((item) => item?.assessment_status === "assessed")
    || (lattice.residual_adds || []).find((item) => item.assessment_status === "assessed");
  const { canvas, ctx, width, y } = createExportCanvas(
    filename,
    "Residual Quantization Lattice",
    "Complete legal 8-bit input-code domain; endpoint escape is not observed activation frequency or task error.",
    1180,
    760,
  );
  if (!row) {
    drawEmptyState(ctx, "No per-tensor 8-bit residual ADD was assessed.", 48, y);
    return canvas;
  }
  const grid = Number(row.tile_grid_dimension || 16);
  const plotSize = 470;
  const cell = plotSize / grid;
  const plotX = 64;
  const plotY = y + 12;
  const values = row.tile_range_escape_pair_counts.map((value) => value / (row.tile_size_codes ** 2));
  values.forEach((value, index) => {
    ctx.fillStyle = latticeEscapeColor(value);
    const rowIndex = Math.floor(index / grid);
    const column = index % grid;
    ctx.fillRect(plotX + column * cell, plotY + (grid - 1 - rowIndex) * cell, Math.ceil(cell), Math.ceil(cell));
  });
  ctx.strokeStyle = "#536273";
  ctx.lineWidth = 1;
  ctx.strokeRect(plotX, plotY, plotSize, plotSize);
  ctx.fillStyle = "#536273";
  ctx.font = "13px Inter, Arial, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("input 1 code", plotX + plotSize / 2, plotY + plotSize + 28);
  ctx.save();
  ctx.translate(plotX - 28, plotY + plotSize / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("input 0 code", 0, 0);
  ctx.restore();
  ctx.textAlign = "left";
  ctx.fillStyle = "#197c78";
  ctx.fillRect(plotX, plotY + plotSize + 48, 16, 10);
  ctx.fillStyle = "#536273";
  ctx.font = "12px Inter, Arial, sans-serif";
  ctx.fillText("0% endpoint escape", plotX + 23, plotY + plotSize + 58);
  ctx.fillStyle = "#e15d3a";
  ctx.fillRect(plotX + 170, plotY + plotSize + 48, 16, 10);
  ctx.fillStyle = "#536273";
  ctx.fillText("100% endpoint escape", plotX + 193, plotY + plotSize + 58);

  const detailX = 600;
  const detailWidth = width - detailX - 48;
  ctx.fillStyle = "#17202c";
  ctx.font = "700 20px Inter, Arial, sans-serif";
  ctx.fillText(`#${padOp(row.op_index)} ADD`, detailX, plotY + 10);
  ctx.fillStyle = "#617083";
  ctx.font = "12px Inter, Arial, sans-serif";
  ctx.fillText(row.output_tensor_name || `T${row.output_tensor_index}`, detailX, plotY + 32);
  const metrics = [
    ["Endpoint escape", `${formatNumber(row.range_escape_pair_count)} / ${formatNumber(row.enumerated_code_pair_count)} (${formatPercent(row.range_escape_pair_ratio)})`],
    ["Rounded projection clamp", `${formatNumber(row.rounded_projection_clamp_pair_count)} (${formatPercent(row.rounded_projection_clamp_pair_ratio)})`],
    ["In-range rounding error", `${Number(row.mean_in_range_rounding_error_steps).toFixed(4)} mean / ${Number(row.maximum_in_range_rounding_error_steps).toFixed(4)} max output steps`],
    ["All-pair projection error", `${Number(row.mean_clamped_projection_error_steps).toFixed(4)} mean / ${Number(row.maximum_clamped_projection_error_steps).toFixed(4)} max output steps`],
    ["Input scales", row.input_scales.map((value) => Number(value).toPrecision(6)).join(" / ")],
    ["Output scale / zero-point", `${Number(row.output_scale).toPrecision(6)} / ${row.output_zero_point}`],
    ["Legal sum range", latticeInterval(row.legal_sum_real_range)],
    ["Output real range", latticeInterval(row.output_real_range)],
    ["Projected output codes", `${row.distinct_projected_output_code_count} / 256`],
    ["Fixed-zp containment", row.fixed_zero_point_containment
      ? `${Number(row.fixed_zero_point_containment.scale_ratio_to_current).toFixed(3)}x scale / ${formatNumber(row.fixed_zero_point_containment.rounded_projection_clamp_pair_count)} clamps`
      : "unavailable"],
    ["Global finest containment", row.globally_finest_containment
      ? `${Number(row.globally_finest_containment.scale_ratio_to_current).toFixed(3)}x scale / zp ${Number(row.globally_finest_containment.signed_zero_point_delta) >= 0 ? "+" : ""}${row.globally_finest_containment.signed_zero_point_delta} / ${formatNumber(row.globally_finest_containment.rounded_projection_clamp_pair_count)} clamps`
      : "unavailable"],
  ];
  metrics.forEach(([label, value], index) => {
    const metricY = plotY + 72 + index * 46;
    ctx.fillStyle = "#617083";
    ctx.font = "11px Inter, Arial, sans-serif";
    ctx.fillText(label, detailX, metricY);
    ctx.fillStyle = "#17202c";
    ctx.font = "700 13px Inter, Arial, sans-serif";
    ctx.fillText(value, detailX, metricY + 19, detailWidth);
    ctx.strokeStyle = "#d9e0ea";
    ctx.beginPath();
    ctx.moveTo(detailX, metricY + 29);
    ctx.lineTo(detailX + detailWidth, metricY + 29);
    ctx.stroke();
  });
  ctx.fillStyle = "#617083";
  ctx.font = "11px Inter, Arial, sans-serif";
  ctx.fillText(`${lattice.schema} / ${lattice.rounding_rule}`, detailX, plotY + 520, detailWidth);
  return canvas;
}

function latticeInterval(range) {
  return Array.isArray(range) ? `[${Number(range[0]).toPrecision(6)}, ${Number(range[1]).toPrecision(6)}]` : "not assessed";
}

function latticeEscapeColor(ratio) {
  const bounded = Math.max(0, Math.min(1, Number(ratio || 0)));
  if (bounded < 0.5) return mixHex("#197c78", "#d99a2b", bounded * 2);
  return mixHex("#d99a2b", "#e15d3a", (bounded - 0.5) * 2);
}

function mixHex(from, to, ratio) {
  const parse = (hex) => [1, 3, 5].map((offset) => Number.parseInt(hex.slice(offset, offset + 2), 16));
  const left = parse(from);
  const right = parse(to);
  return `rgb(${left.map((value, index) => Math.round(value + (right[index] - value) * ratio)).join(",")})`;
}

function createExportCanvas(filename, title, subtitle, width = 1180, height = 680) {
  return createExportCanvasShell(title, subtitle, width, height, filename);
}

function renderEvidenceTreemapCanvas(analysis, filename, kind) {
  const presentation = kind === "resource"
    ? buildResourceMapPresentation(analysis)
    : buildQuantizationExposurePresentation(
      analysis,
      {},
      serializedTensorPresentation(analysis),
      quantSummaryEvidence(analysis).opSignals,
    );
  const { canvas, ctx, width, y } = createExportCanvas(
    filename,
    presentation.title,
    `${presentation.metricLabel}; ${presentation.evidenceClass}. Area is quantitative and color is categorical.`,
    1180,
    760,
  );
  ctx.fillStyle = "#536273";
  ctx.font = "10px Inter, Arial, sans-serif";
  ctx.fillText(
    `${formatNumber(presentation.items.length)}/${formatNumber(presentation.assessedCount)} positive/assessed; ${formatNumber(presentation.zeroCount)} exact zero; ${formatNumber(presentation.unassessedCount)} not assessed; conservation ${presentation.conservationStatus}`,
    48,
    y + 17,
    width - 96,
  );
  if (presentation.status !== "assessed") {
    drawEmptyState(ctx, "The selected quantity has no positive assessed values; unavailable is not rendered as zero.", 48, y + 44);
    return canvas;
  }
  const x = 48;
  const top = y + 40;
  const mapWidth = width - 96;
  const mapHeight = 510;
  const layout = layoutGroupedTreemap(presentation.items, mapWidth, mapHeight);
  const tones = { good: "#dcefe8", cool: "#dfebef", warn: "#f4e7d4", risk: "#f4dfe0", violet: "#ece6ef", mixed: "#f4eccc", neutral: "#edf1ee" };
  for (const item of layout.tiles) {
    const rect = item.rect;
    ctx.fillStyle = tones[item.tone] || tones.neutral;
    ctx.fillRect(x + rect.x, top + rect.y, rect.w, rect.h);
    ctx.strokeStyle = "#718078";
    ctx.strokeRect(x + rect.x, top + rect.y, rect.w, rect.h);
    if (rect.w > 58 && rect.h > 24) {
      ctx.fillStyle = "#17211d";
      ctx.font = "700 8px Inter, Arial, sans-serif";
      ctx.fillText(item.shortLabel, x + rect.x + 4, top + rect.y + rect.h - 12, rect.w - 8);
    }
  }
  for (const group of layout.groups) {
    const rect = group.rect;
    ctx.strokeStyle = "#34423d";
    ctx.lineWidth = 2;
    ctx.strokeRect(x + rect.x, top + rect.y, rect.w, rect.h);
    ctx.lineWidth = 1;
    if (rect.w > 90 && rect.h > 20) {
      ctx.fillStyle = "rgba(255,254,250,0.9)";
      ctx.fillRect(x + rect.x + 2, top + rect.y + 2, Math.min(rect.w - 4, 220), 15);
      ctx.fillStyle = "#17211d";
      ctx.font = "700 8px Inter, Arial, sans-serif";
      ctx.fillText(group.label, x + rect.x + 6, top + rect.y + 12, Math.min(rect.w - 12, 210));
    }
  }
  ctx.fillStyle = "#536273";
  ctx.font = "9px Inter, Arial, sans-serif";
  ctx.fillText(presentation.boundary, 48, top + mapHeight + 22, width - 96);
  return canvas;
}

function renderMacDistributionCanvas(analysis, filename) {
  const distribution = macDistributionData(analysis);
  const coverage = distribution.coverageComplete
    ? "Complete deterministic MAC ledger."
    : `${distribution.assessedComputeOps ?? 0}/${distribution.computeOps ?? "?"} compute ops assessed; unresolved rows are excluded from this subtotal.`;
  const { canvas, ctx, width, y } = createExportCanvas(
    filename,
    distribution.coverageComplete ? "MAC Distribution" : "Assessed MAC Distribution",
    `Top operators by ${distribution.coverageComplete ? "complete MAC share" : "assessed MAC subtotal"}; color follows static roofline posture. ${coverage}`,
  );
  const { top, totalMacs, otherMacs } = distribution;
  const total = Math.max(1, totalMacs);
  const segments = top.map((op) => ({ label: `#${padOp(op.index)}`, value: Number(op.macs || 0), tone: boundTone(op.static_bound_guess), detail: op.name }));
  if (otherMacs > 0) segments.push({ label: "Other", value: otherMacs, tone: "neutral", detail: "Remaining ops" });
  drawFlame(ctx, segments, total, 48, y, width - 96, 104);
  drawRankList(ctx, top.slice(0, 8).map((op) => [
    `#${padOp(op.index)} ${op.name}`,
    `${formatNumber(op.macs)} MACs / ${formatPercent(Number(op.macs || 0) / total)}`,
    boundTone(op.static_bound_guess),
  ]), 48, y + 140, width - 96);
  return canvas;
}

function renderBottleneckCanvas(analysis, filename, targetProfile) {
  const { canvas, ctx, width, y } = createExportCanvas(
    filename,
    "Modeled Bottleneck Contribution",
    "Steady-state static estimate from roofline base, predicted partition-break setup, and fallback traffic; one-time packing is excluded.",
  );
  const { top, totalUs, otherUs } = bottleneckDistributionData(analysis, analysis.target_profile || targetProfile);
  const total = Math.max(1, totalUs);
  const segments = top.map(({ op, estimate }) => ({
    label: `#${padOp(op.index)}`,
    value: estimate.totalUs,
    tone: estimate.dominantTone,
    detail: estimate.dominantLabel,
  }));
  if (otherUs > 0) segments.push({ label: "Other", value: otherUs, tone: "neutral", detail: "Remaining estimate" });
  drawFlame(ctx, segments, total, 48, y, width - 96, 104);
  drawRankList(ctx, top.slice(0, 8).map(({ op, estimate }) => [
    `#${padOp(op.index)} ${op.name}`,
    `${formatUs(estimate.totalUs)} / ${estimate.dominantLabel}`,
    estimate.dominantTone,
  ]), 48, y + 140, width - 96);
  return canvas;
}

function renderTargetComparisonCanvas(filename, rows) {
  const { canvas, ctx, width, y } = createExportCanvas(
    filename,
    "Target Comparison",
    "Static target sensitivity across common deployment profiles.",
  );
  const cardWidth = (width - 120) / 2;
  rows.slice(0, 4).forEach((row, index) => {
    const x = 48 + (index % 2) * (cardWidth + 24);
    const cardY = y + Math.floor(index / 2) * 190;
    drawTargetCard(ctx, row, x, cardY, cardWidth, 160);
  });
  return canvas;
}

function renderChainFlowCanvas(analysis, filename) {
  const chains = Array.isArray(analysis.xnnpack_chains) ? analysis.xnnpack_chains : [];
  const height = Math.max(420, 230 + Math.ceil(chains.length / 4) * 120);
  const { canvas, ctx, width, y } = createExportCanvas(
    filename,
    "TFLite / XNNPACK Predicted Partition Flow",
    "Predicted delegate segments and partition-break markers. Blocks are scaled by MAC share where available.",
    1180,
    height,
  );
  const ops = analysis.ops || [];
  const breaks = ops.filter((op) => op.xnnpack_chain_break);
  if (!chains.length) {
    drawEmptyState(ctx, "No conditionally delegatable XNNPACK segment was predicted for this target.", 48, y);
    return canvas;
  }
  const sorted = [...chains].sort((a, b) => Number(a.first_op || 0) - Number(b.first_op || 0));
  sorted.slice(0, 16).forEach((chain, index) => {
    const col = index % 4;
    const row = Math.floor(index / 4);
    const x = 48 + col * 270;
    const cardY = y + row * 116;
    drawChainBlock(ctx, chain, x, cardY, 222, 82);
    const next = sorted[index + 1];
    const between = next ? breaks.filter((op) => op.index > Number(chain.last_op) && op.index < Number(next.first_op)) : [];
    if (between.length && col < 3) drawBreakPill(ctx, `${between.length} break`, x + 230, cardY + 26);
  });
  return canvas;
}

function renderQuantHeatmapCanvas(analysis, filename) {
  const source = analysis.ops || [];
  const evidence = quantSummaryEvidence(analysis);
  const columns = 34;
  const tile = 24;
  const rows = Math.ceil(Math.min(source.length, 238) / columns);
  const height = Math.max(360, 190 + rows * (tile + 6));
  const { canvas, ctx, y } = createExportCanvas(
    filename,
    "Quantization State & Risk Map",
    "All graph ops are shown. Warn/risk includes op-local checks and stored-kernel contract findings.",
    1180,
    height,
  );
  source.slice(0, 238).forEach((op, index) => {
    const x = 48 + (index % columns) * (tile + 6);
    const tileY = y + Math.floor(index / columns) * (tile + 6);
    ctx.fillStyle = colorForTone(evidence.opSignals.get(Number(op.index))?.tone || quantTileTone(op));
    roundRect(ctx, x, tileY, tile, tile, 5);
    ctx.fill();
    ctx.strokeStyle = "#d9e0ea";
    ctx.stroke();
    ctx.fillStyle = "#151a22";
    ctx.font = "8px Inter, Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(String(op.index), x + tile / 2, tileY + 15);
    ctx.textAlign = "left";
  });
  drawLegend(ctx, 48, y + rows * (tile + 6) + 30);
  return canvas;
}

function renderStageMemoryMixCanvas(analysis, filename) {
  const blockStages = analysis?.block_inventory?.status === "assessed"
    ? analysis.block_inventory.stages || []
    : [];
  const stages = blockStages.length
    ? blockStages.map((stage) => ({
        index: stage.index,
        key: stage.display_name || `stage-${stage.index}`,
        op_indices: stage.op_indices || [],
        mac_percent: stage.aggregates?.mac_percent ?? null,
        xnnpack_chain_breaks: stage.aggregates?.predicted_break_count || 0,
      }))
    : analysis.stages || [];
  const height = Math.max(420, 180 + stages.length * 42);
  const { canvas, ctx, width, y } = createExportCanvas(
    filename,
    "Stage Memory Mix",
    "Stage-level compute/mixed/memory pressure from static roofline classification.",
    1180,
    height,
  );
  const ops = analysis.ops || [];
  stages.forEach((stage, index) => {
    const opIndices = Array.isArray(stage.op_indices) ? new Set(stage.op_indices) : null;
    const stageOps = opIndices
      ? ops.filter((op) => opIndices.has(op.index))
      : ops.filter((op) => op.index >= Number(stage.first_op) && op.index <= Number(stage.last_op));
    drawStageMixRow(ctx, stage, stageOps, 48, y + index * 42, width - 96);
  });
  return canvas;
}
