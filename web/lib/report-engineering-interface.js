import { formatNumber, padOp } from "./format.js";
import { compareInterfaceContracts } from "./interface-contract.js";
import { buildInterfaceQuantizationContractLedger } from "./quantization-contract-summary.js";
import { code, markdownTable } from "./report-utils.js";

export function inputLayoutDetermination(analysis) {
  const contracts = analysis?.input_contracts || [];
  if (!contracts.length) return "not assessed; no structured input contract was emitted";
  return contracts.map((contract) => {
    const tensor = contract.name || `T${contract.tensor_index}`;
    const layout = contract.layout || "not determined";
    const source = contract.layout_source_op_index == null
      ? "no source op"
      : `#${padOp(contract.layout_source_op_index)} ${contract.layout_source_op_name || "UNKNOWN"}`;
    return `${tensor}: ${layout}; Evidence: ${contract.layout_evidence_class || "NOT_ASSESSABLE"}; status ${contract.layout_status || "not_assessed"}; ${source}`;
  }).join(" / ");
}

export function inputContractRiskSummary(analysis) {
  const risks = (analysis?.input_contracts || [])
    .flatMap((contract) => contract.risks || [])
    .filter(Boolean);
  if (risks.length) return risks.slice(0, 4).join(" / ");
  const notes = (analysis?.input_contracts || []).map((contract) => contract.range_note).filter(Boolean);
  if (notes.length) return notes.slice(0, 4).join(" / ");
  if (analysis?.inputs?.length && !analysis?.metadata_presence?.documented_preprocessing) {
    return "Color order, input range, resize policy, interpolation, and normalization are not embedded.";
  }
  return "not emitted";
}

export function inputNumericalContractSummary(analysis) {
  const contracts = analysis?.input_contracts || [];
  if (!contracts.length) return "not emitted";
  return contracts.map((contract) => {
    const hasRange = Number.isFinite(contract.expected_range_low) && Number.isFinite(contract.expected_range_high);
    return `${contract.name || `T${contract.tensor_index}`}: ${contract.tensor_numerical_contract_status || "not_assessed"}${hasRange ? `; real range [${Number(contract.expected_range_low).toExponential(4)}, ${Number(contract.expected_range_high).toExponential(4)}]` : "; scalar real range not emitted"}`;
  }).join(" / ");
}

export function sourcePreprocessingContractSummary(analysis) {
  const contracts = analysis?.input_contracts || [];
  if (!contracts.length) return analysis?.metadata_presence?.documented_preprocessing ? "documented in artifact metadata" : "not embedded in artifact";
  return contracts.map((contract) => `${contract.name || `T${contract.tensor_index}`}: ${contract.source_data_to_tensor_preprocessing_status || "not_embedded_in_artifact"}`).join(" / ");
}

export function inputContractEvidenceMarkdown(analysis) {
  const contracts = analysis?.input_contracts || [];
  if (!contracts.length) return "> No structured input tensor contract was emitted.";
  const rows = contracts.map((contract) => {
    const rangeLow = contract.expected_range_low;
    const rangeHigh = contract.expected_range_high;
    const layoutSourceIndex = contract.layout_source_op_index;
    const layoutSourceName = contract.layout_source_op_name;
    const channelAxis = contract.channel_axis;
    const channels = contract.channels;
    const range = Number.isFinite(rangeLow) && Number.isFinite(rangeHigh)
      ? `[${Number(rangeLow).toPrecision(9)}, ${Number(rangeHigh).toPrecision(9)}]`
      : "not emitted";
    const layoutSource = layoutSourceIndex == null
      ? "none"
      : `#${padOp(layoutSourceIndex)} ${layoutSourceName || "UNKNOWN"}`;
    const channel = channelAxis == null
      ? "not determined"
      : `axis ${formatNumber(channelAxis)} / ${channels == null ? "unknown" : formatNumber(channels)} channel(s)`;
    return [
      `T${formatNumber(contract.tensor_index)} ${code(contract.name || "-")}`,
      `${contract.dtype || "UNKNOWN"} ${code(JSON.stringify(contract.shape || []))}`,
      `${contract.layout || "not determined"}; ${contract.layout_evidence_class || "NOT_ASSESSABLE"}; ${contract.layout_status || "not_assessed"}; ${channel}; source ${layoutSource}`,
      `${contract.is_quantized ? "quantized" : "not quantized"}; ${contract.tensor_numerical_contract_status || "not_assessed"}; range ${range}; ${contract.range_note || "no range note"}`,
      contract.source_data_to_tensor_preprocessing_status || "not_assessed",
      (contract.risks || []).join(" / ") || "none",
      `${contract.schema || "schema not emitted"}; ${contract.layout_reason || "layout basis not emitted"}`,
    ];
  });
  return markdownTable(["Tensor", "ABI", "Layout contract", "Numerical contract", "Source preprocessing", "Risks", "Method / boundary"], rows);
}

export function interfaceQuantizationLedgerMarkdown(analysis, productionInterfaceContract = null) {
  const ledger = buildInterfaceQuantizationContractLedger(analysis);
  if (!ledger.parameter_count) return "> No graph interface parameters were parsed.";
  const boundary = ledger.boundary_contract || {};
  const boundaryStatus = (scope) => `${scope?.status || "not_declared"}; ${formatNumber(scope?.affine_quantized_parameter_count || 0)} affine, ${formatNumber(scope?.unquantized_parameter_count || 0)} unquantized, ${formatNumber(scope?.invalid_or_incomplete_parameter_count || 0)} invalid/incomplete`;
  const rows = ledger.parameters.map((parameter) => {
    const quantization = parameter.quantization;
    const domain = quantization.scalar_real_code_domain;
    return [
      `${parameter.direction} ${parameter.ordinal}`,
      `${parameter.dtype} ${code(JSON.stringify(parameter.shape))}`,
      quantization.status,
      quantization.granularity,
      quantization.status === "complete" ? `${quantization.scale_count} / ${quantization.zero_point_count}` : "N/A",
      quantization.status === "complete" ? `${compactContractVector(quantization.scales)} / ${compactContractVector(quantization.zero_points)}` : "N/A",
      domain ? `[${domain.real_min.toPrecision(7)}, ${domain.real_max.toPrecision(7)}]` : "N/A",
      quantization.cardinality_status,
      parameter.interface_contract_sha256.slice(0, 16),
    ];
  });
  const comparison = compareInterfaceContracts(ledger, productionInterfaceContract, analysis?.model_sha256);
  const mismatchRows = comparison.mismatches.map((item) => [
    item.parameter_id || "document",
    item.field,
    code(compactContractValue(item.expected)),
    code(compactContractValue(item.declared)),
    code(item.evidence_pointer || "-"),
  ]);
  return [
    `Ledger SHA-256: ${code(ledger.ledger_sha256)} (${ledger.hash_contract.method}). Complete affine ${formatNumber(ledger.quantized_parameter_count)}; unquantized ${formatNumber(ledger.unquantized_parameter_count)}; invalid/incomplete ${formatNumber(ledger.invalid_or_incomplete_parameter_count)}.`,
    markdownTable(["Boundary scope", "Declared storage contract"], [
      ["All external parameters", boundaryStatus(boundary)],
      ["Inputs", boundaryStatus(boundary.inputs)],
      ["Outputs", boundaryStatus(boundary.outputs)],
      ["Recorded facts", (boundary.recorded_facts || []).join(", ") || "not emitted"],
      ["Not established", (boundary.not_established_by_this_contract || []).join(", ") || "not emitted"],
    ]),
    markdownTable(["Parameter", "Dtype / shape", "Status", "Granularity", "Scale / ZP count", "Scale / ZP values", "Real code domain", "Cardinality", "ABI SHA-256"], rows),
    "### Production Interface Binding",
    markdownTable(["Field", "Value"], [
      ["Status / gate", `${comparison.status} / ${comparison.gate_result}`],
      ["Declared / expected parameters", `${formatNumber(comparison.declared_parameter_count)} / ${formatNumber(comparison.expected_parameter_count)}`],
      ["Field differences", formatNumber(comparison.mismatch_count)],
      ["Encoder / decoder implementation SHA-256", comparison.implementation_sha256 || "not bound"],
    ]),
    mismatchRows.length
      ? markdownTable(["Parameter", "Field", "Artifact", "Production declaration", "Pointer"], mismatchRows)
      : `> ${productionBindingSummary(comparison)}`,
    `> ${comparison.interpretation_boundary}`,
    `> ${ledger.interpretation_boundary}`,
  ].join("\n\n");
}

function productionBindingSummary(comparison) {
  if (comparison.status === "bound_exact_contract") return "The supplied production declaration matches every external tensor contract and binds its implementation digest.";
  if (comparison.status === "unbound") return "No production declaration was supplied; this boundary remains unbound.";
  if (comparison.status.startsWith("partial_")) return `The supplied declaration matches the assessed tensor fields but remains incomplete: ${comparison.status.replaceAll("_", " ")}.`;
  return `The supplied production declaration was assessed as ${comparison.status}.`;
}

function compactContractVector(values) {
  const source = Array.isArray(values) ? values : [];
  if (source.length <= 8) return code(JSON.stringify(source));
  return `${code(JSON.stringify(source.slice(0, 8)))} + ${formatNumber(source.length - 8)} value(s); full vector hash-bound in the interface ledger`;
}

function compactContractValue(value) {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text.length > 160 ? `${text.slice(0, 157)}...` : text;
}
