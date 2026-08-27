import { evaluateDynamicIntegerFormula } from "./dynamic-shape-cost.js";
import { evaluateOnnxDimensionExpression } from "./onnx-dimension-expression.js";

export const TENSORRT_PROFILE_COST_SCHEMA = "deepbom.tensorrt_optimization_profile_cost.v1";

export function buildTensorRtOptimizationProfileCost(analysis, buildProfile) {
  const contract = analysis?.dynamic_shape_cost_contract;
  const profiles = buildProfile?.optimization_profiles || [];
  if (!profiles.length) return base("not_applicable_no_optimization_profiles", []);
  if (!contract || !Array.isArray(contract.symbols)) return base("not_assessed_dynamic_cost_contract_missing", []);
  const scenarios = []; const conflicts = [];
  for (const profile of profiles) for (const point of ["min", "opt", "max"]) {
    const result = bindScenario(contract, profile, point);
    scenarios.push(result.scenario);
    conflicts.push(...result.conflicts);
  }
  return {
    ...base(conflicts.length ? "invalid_symbol_binding" : scenarios.every((row) => row.status === "exact_conditional") ? "assessed" : "partial", scenarios),
    conflict_count: conflicts.length,
    conflicts,
  };
}

function base(status, scenarios) {
  return {
    schema: TENSORRT_PROFILE_COST_SCHEMA,
    evidence_class: status.startsWith("not_") ? "NOT_ASSESSABLE" : "CONFIGURATION_BOUND_DERIVED",
    status,
    scenario_count: scenarios.length,
    scenarios,
    conflict_count: 0,
    conflicts: [],
    interpretation_boundary: "Each row is an exact conditional evaluation at one declared TensorRT min/opt/max profile point. The three points are not asserted to be global cost extrema. Values do not establish parser acceptance, engine tactics, fusion, workspace allocation, transfer, latency, or runtime memory.",
  };
}

function bindScenario(contract, profile, point) {
  const inputShapes = new Map(profile.inputs.map((input) => [input.name, input[point]]));
  const byId = new Map(); const byName = new Map(); const conflicts = [];
  for (const symbol of contract.symbols) {
    if (symbol.source === "onnx_derived_dimension_expression") continue;
    const values = (symbol.occurrences || []).map((occurrence) => inputShapes.get(occurrence.tensor_name)?.[occurrence.axis]).filter(Number.isSafeInteger);
    if (!values.length) continue;
    if (new Set(values).size !== 1) {
      conflicts.push({ profile_id: profile.id, profile_point: point, symbol_id: symbol.symbol_id, reason: "shared_dim_param_has_conflicting_profile_values", values });
      continue;
    }
    const value = BigInt(values[0]); byId.set(symbol.symbol_id, value);
    if (symbol.declared_name) byName.set(symbol.declared_name, value);
  }
  let progress = true;
  while (progress) {
    progress = false;
    for (const symbol of contract.symbols) {
      if (byId.has(symbol.symbol_id) || !symbol.expression_ir) continue;
      const value = evaluateOnnxDimensionExpression(symbol.expression_ir, byName);
      if (value != null) { byId.set(symbol.symbol_id, value); byName.set(symbol.declared_name, value); progress = true; }
    }
  }
  const residualSymbols = contract.symbols.filter((symbol) => !byId.has(symbol.symbol_id)).map((symbol) => symbol.symbol_id);
  const macs = evaluateDynamicIntegerFormula(contract.total_macs_formula, byId);
  const payloads = (contract.tensor_formulas || []).map((row) => {
    const elements = evaluateDynamicIntegerFormula(row.element_count_formula, byId);
    const bits = evaluateDynamicIntegerFormula(row.payload_bits_formula, byId);
    const bytes = evaluateDynamicIntegerFormula(row.payload_bytes_formula, byId) ?? (bits == null ? null : (bits + 7n) / 8n);
    return { tensor_index: row.tensor_index, tensor_name: row.tensor_name, elements_decimal: decimal(elements), payload_bytes_decimal: decimal(bytes) };
  });
  const liveValues = (contract.liveness?.candidates || []).map((row) => evaluateDynamicIntegerFormula(row.live_payload_formula, byId));
  const peak = liveValues.length && liveValues.every((value) => value != null) ? liveValues.reduce((a, b) => a > b ? a : b) : null;
  const exact = !conflicts.length && !residualSymbols.length && macs != null && peak != null && payloads.every((row) => row.payload_bytes_decimal != null);
  return {
    conflicts,
    scenario: {
      profile_id: profile.id,
      profile_point: point,
      evidence_class: "CONFIGURATION_BOUND_DERIVED",
      status: exact ? "exact_conditional" : "partial",
      symbol_assignments: Object.fromEntries([...byId].map(([key, value]) => [key, value.toString()])),
      residual_symbol_ids: residualSymbols,
      total_macs_decimal: decimal(macs),
      total_macs: safeNumber(macs),
      peak_live_payload_bytes_decimal: decimal(peak),
      peak_live_payload_bytes: safeNumber(peak),
      dynamic_tensor_payloads: payloads,
    },
  };
}

function decimal(value) { return typeof value === "bigint" ? value.toString() : null; }
function safeNumber(value) { return typeof value === "bigint" && value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null; }
