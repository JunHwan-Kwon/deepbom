export const MAC_COVERAGE_SCHEMA = "deepbom.mac_coverage.v1";

const WEIGHT_CONTAINERS = new Set(["gguf", "safetensors"]);

export function deriveMacCoverage(analysis = {}, quantizationStatus = null) {
  const format = String(analysis?.format || "unknown").toLowerCase();
  const explicit = analysis?.mac_assessment || null;
  const quant = quantizationStatus || analysis?.quantization_status || {};
  const explicitCompute = nonNegativeInteger(explicit?.compute_ops);
  const quantCompute = nonNegativeInteger(quant?.compute_ops);
  const graphCompute = graphComputeOperatorCount(analysis?.ops);
  const computeOps = explicitCompute ?? quantCompute ?? graphCompute;
  const explicitAssessed = nonNegativeInteger(explicit?.assessed_compute_ops);
  const explicitStatus = normalizedStatus(explicit?.status);

  if (WEIGHT_CONTAINERS.has(format)) {
    return coverage(format, "not_applicable_weight_container", 0, 0, "serialized_format_has_no_execution_graph");
  }

  if (explicitStatus) {
    const assessed = explicitAssessed ?? (completeStatus(explicitStatus) ? computeOps : null);
    return coverage(format, explicitStatus, computeOps, assessed,
      assessed == null ? "assessed_compute_operator_count_not_emitted" : null);
  }

  if (format === "tflite" && Array.isArray(analysis?.ops) && finite(analysis?.total_macs) != null) {
    const complete = analysis.ops.every((op) => finite(op?.macs) != null);
    return coverage(format, complete ? "assessed" : "not_assessed_incomplete_mac_ledger",
      computeOps, complete ? computeOps : null,
      complete ? null : "one_or_more_serialized_operators_lack_mac_evidence");
  }

  return coverage(format, "not_assessed", computeOps, explicitAssessed,
    "format_adapter_did_not_emit_a_complete_mac_assessment");
}

export function deriveQuantizedComputeAssessment(analysis = {}, quantizationStatus = null, macCoverage = null) {
  const format = String(analysis?.format || "unknown").toLowerCase();
  const quant = quantizationStatus || analysis?.quantization_status || {};
  const coverageValue = macCoverage || deriveMacCoverage(analysis, quant);
  const quantizedOps = nonNegativeInteger(quant?.quantized_compute_ops);
  const quantizedMacRatio = finite(quant?.quantized_compute_mac_percent);

  if (WEIGHT_CONTAINERS.has(format)) return {
    operator_status: "not_applicable_weight_container",
    operator_reason: "serialized_format_has_no_execution_graph",
    mac_status: "not_applicable_weight_container",
    mac_reason: "serialized_format_has_no_execution_graph",
  };
  if (quantizedMacRatio != null) return {
    operator_status: quantizedOps == null ? "not_assessed_quantized_operator_count_unavailable" : "assessed",
    operator_reason: quantizedOps == null ? "quantized_compute_operator_count_not_emitted" : null,
    mac_status: "assessed",
    mac_reason: null,
  };
  if (!coverageComplete(coverageValue)) return {
    operator_status: quantizedOps == null ? "not_assessed_quantized_operator_count_unavailable" : "assessed",
    operator_reason: quantizedOps == null ? "quantized_compute_operator_count_not_emitted" : null,
    mac_status: "not_assessed_mac_coverage_incomplete",
    mac_reason: coverageValue.reason || coverageValue.status,
  };
  if (format === "coreml" && quantizedOps == null) return {
    operator_status: "not_assessed_execution_precision_not_serialized",
    operator_reason: "serialized_weights_do_not_establish_executed_operator_precision",
    mac_status: "not_assessed_execution_precision_not_serialized",
    mac_reason: "serialized_weights_do_not_establish_which_compute_operators_execute_quantized",
  };
  return {
    operator_status: quantizedOps == null ? "not_assessed_quantized_operator_count_unavailable" : "assessed",
    operator_reason: quantizedOps == null ? "quantized_compute_operator_count_not_emitted" : null,
    mac_status: "not_assessed_quantization_contract_unavailable",
    mac_reason: "quantized_compute_mac_ledger_not_emitted",
  };
}

export function coverageComplete(value) {
  return value?.compute_ops != null
    && value?.assessed_compute_ops === value.compute_ops
    && completeStatus(value?.status);
}

function coverage(format, status, computeOps, assessedComputeOps, reason) {
  if (computeOps != null && assessedComputeOps != null && assessedComputeOps > computeOps) {
    throw new Error("MAC coverage assessed compute operators exceed the compute-operator denominator.");
  }
  return Object.freeze({
    schema: MAC_COVERAGE_SCHEMA,
    format,
    status,
    compute_ops: computeOps,
    assessed_compute_ops: assessedComputeOps,
    unassessed_compute_ops: computeOps == null || assessedComputeOps == null ? null : computeOps - assessedComputeOps,
    reason,
  });
}

function completeStatus(value) {
  const status = normalizedStatus(value);
  return Boolean(status) && !/not_assessed|incomplete|partial|unresolved|failed/.test(status)
    && /assessed|complete/.test(status);
}

function normalizedStatus(value) {
  return String(value || "").trim().toLowerCase().replaceAll("-", "_");
}

function nonNegativeInteger(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isSafeInteger(number) && number >= 0 ? number : null;
}

function finite(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function graphComputeOperatorCount(ops) {
  if (!Array.isArray(ops)) return null;
  return ops.reduce((count, op) => count + (finite(op?.macs) > 0 ? 1 : 0), 0);
}
