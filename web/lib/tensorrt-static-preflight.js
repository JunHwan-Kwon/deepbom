import { buildBackendPlacementProjection, BACKEND_PLACEMENT_STATES as S } from "./backend-placement-projection.js";
import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import { TENSORRT_SOURCE_METADATA } from "./tensorrt-source-metadata.js";
import { buildTensorRtOptimizationProfileCost } from "./tensorrt-profile-cost.js";

export const TENSORRT_BUILD_PROFILE_SCHEMA = "deepbom.tensorrt_build_profile.v1";
export const TENSORRT_STATIC_PREFLIGHT_SCHEMA = "deepbom.tensorrt_static_preflight.v1";
export const TENSORRT_PARSER_OBSERVATION_SCHEMA = "deepbom.tensorrt_parser_observation.v1";

const SHA256 = /^[a-f0-9]{64}$/;
const PATHS = new Set(["native_tensorrt", "ort_tensorrt_ep"]);
const NATIVE_PARSER_METHODS = new Set(["supportsModel", "supportsModelV2"]);

export function createTensorRtBuildProfile(configuration) {
  if (!configuration || typeof configuration !== "object" || Array.isArray(configuration)) {
    throw new Error("TensorRT build profile must be an object.");
  }
  const executionPath = String(configuration.execution_path || "");
  if (!PATHS.has(executionPath)) throw new Error("TensorRT execution_path must be native_tensorrt or ort_tensorrt_ep.");
  const precision = configuration.precision;
  if (!precision || typeof precision !== "object") throw new Error("TensorRT precision policy is required.");
  for (const key of ["tf32", "fp16", "bf16", "int8", "fp8"]) {
    if (typeof precision[key] !== "boolean") throw new Error(`TensorRT precision.${key} must be explicitly true or false.`);
  }
  const profile = {
    schema: TENSORRT_BUILD_PROFILE_SCHEMA,
    execution_path: executionPath,
    expected_tensorrt_version: nullableText(configuration.expected_tensorrt_version),
    expected_cuda_version: nullableText(configuration.expected_cuda_version),
    device_id: nonNegativeInteger(configuration.device_id, "device_id"),
    device_compute_capability: nullableText(configuration.device_compute_capability),
    precision: {
      tf32: precision.tf32,
      fp16: precision.fp16,
      bf16: precision.bf16,
      int8: precision.int8,
      fp8: precision.fp8,
    },
    workspace_limit_bytes: nullablePositiveInteger(configuration.workspace_limit_bytes, "workspace_limit_bytes"),
    builder_optimization_level: nullableBoundedInteger(configuration.builder_optimization_level, 0, 5, "builder_optimization_level"),
    dla_core: nullableNonNegativeInteger(configuration.dla_core, "dla_core"),
    allow_gpu_fallback: explicitBoolean(configuration.allow_gpu_fallback, "allow_gpu_fallback"),
    calibration_cache_sha256: nullableSha(configuration.calibration_cache_sha256, "calibration_cache_sha256"),
    plugins: normalizePlugins(configuration.plugins),
    optimization_profiles: normalizeOptimizationProfiles(configuration.optimization_profiles),
    ort_ep_options: executionPath === "ort_tensorrt_ep" ? normalizeOrtOptions(configuration.ort_ep_options) : null,
    source_basis: TENSORRT_SOURCE_METADATA,
  };
  profile.profile_sha256 = sha256TextHex(canonicalJson(profile));
  return profile;
}

export function buildTensorRtStaticPreflight(analysis, configuration = null, parserObservation = null) {
  const format = String(analysis?.format || "").toLowerCase();
  if (format !== "onnx") return notApplicable(analysis, format);
  if (!Array.isArray(analysis.ops) || !Array.isArray(analysis.tensors) || !Array.isArray(analysis.inputs)) {
    throw new Error("TensorRT preflight requires ONNX operator, tensor, and external-input ledgers.");
  }
  const profile = configuration ? createTensorRtBuildProfile(configuration) : null;
  const inputContracts = analysis.inputs.map(inputContract);
  const issues = profile ? validateBuildSemantics(analysis, profile, inputContracts) : [issue(
    "configuration_unbound", "BLOCKING", "TensorRT build profile is not bound.",
    "Bind an execution path, precision policy, device selection, optimization profiles, workspace policy, DLA fallback policy, plugins, and ORT EP options where applicable.",
  )];
  const observation = parserObservation ? validateParserObservation(analysis, profile, parserObservation) : null;
  const optimizationProfileCost = profile ? buildTensorRtOptimizationProfileCost(analysis, profile) : null;
  for (const conflict of optimizationProfileCost?.conflicts || []) issues.push(issue(
    "profile_symbol_binding_conflict", "BLOCKING",
    `Optimization profile ${conflict.profile_id} ${conflict.profile_point} assigns incompatible values to shared ONNX symbol ${conflict.symbol_id}.`,
    "Use one value for every occurrence of the same serialized dim_param at each profile point.",
  ));
  if (observation?.collector?.git_state === "dirty") {
    issues.push(issue("collector_source_dirty", "UNRESOLVED", "The parser collector was built from a dirty worktree.", "Use the collector binary/source-set SHA-256 for this capture and rebuild from a clean pinned tree for release evidence."));
  }
  const rows = observation ? observationRows(analysis.ops, observation) : analysis.ops.map((op, position) => ({
    op_index: opIndex(op, position),
    state: S.UNRESOLVED,
    reason_codes: [],
    unresolved_predicates: [profile?.execution_path === "ort_tensorrt_ep"
      ? "ort_tensorrt_get_capability_not_observed" : "tensorrt_native_parser_support_not_observed"],
  }));
  const projection = buildBackendPlacementProjection({
    analysis,
    profileId: profile?.execution_path || "tensorrt_unbound",
    label: profile?.execution_path === "ort_tensorrt_ep" ? "ONNX Runtime TensorRT EP" : "TensorRT native parser",
    evidenceClass: observation ? "PARSER_OBSERVED_CONFIGURATION_BOUND" : profile ? "CONFIGURATION_VALIDATED_PARSER_UNOBSERVED" : "CONFIGURATION_UNBOUND",
    rows,
    source: profile?.execution_path === "ort_tensorrt_ep"
      ? TENSORRT_SOURCE_METADATA.onnxruntime_tensorrt_ep : TENSORRT_SOURCE_METADATA.tensorrt,
    interpretationBoundary: observation
      ? "Parser subgraph acceptance was observed for the identity-bound TensorRT build profile. This is build/parser evidence, not an engine-build success, optimized engine layer assignment, runtime execution, kernel selection, transfer, or latency observation."
      : profile?.execution_path === "ort_tensorrt_ep"
        ? "ORT TensorRT EP support is intentionally unresolved until an identity-bound ORT capture observes GetCapability and original-op profile assignment for this artifact and provider configuration. Native parser evidence is not substituted."
        : "TensorRT native-parser support is intentionally unresolved until the pinned collector executes the parser capability API exposed by the selected SDK for this artifact and bound configuration. No generic op-name support table is substituted.",
  });
  const blocking = issues.filter((row) => row.severity === "BLOCKING");
  const unresolved = issues.filter((row) => row.severity === "UNRESOLVED");
  return {
    schema: TENSORRT_STATIC_PREFLIGHT_SCHEMA,
    method_version: "1.0.0",
    format: "onnx",
    artifact_sha256: String(analysis.model_sha256 || ""),
    evidence_class: observation ? "PARSER_OBSERVED/CONFIGURATION_BOUND/DERIVED" : profile ? "DERIVED_CONFIGURATION_PREFLIGHT" : "NOT_ASSESSABLE_CONFIGURATION_UNBOUND",
    status: blocking.length ? "blocked" : observation
      ? projection.state_counts.UNRESOLVED ? "parser_observed_partial" : projection.state_counts.DEFINITE_EXCLUSION ? "parser_observed_with_exclusions" : "parser_observed_all_supported"
      : "configuration_valid_parser_observation_required",
    build_profile: profile,
    input_contracts: inputContracts,
    issues,
    blocking_issue_count: blocking.length,
    unresolved_issue_count: unresolved.length,
    parser_observation: observation,
    optimization_profile_cost: optimizationProfileCost,
    projection,
    trust_boundary: {
      browser_engine_deserialization: "prohibited",
      browser_plan_execution: "prohibited",
      accepted_import: TENSORRT_PARSER_OBSERVATION_SCHEMA,
      reason: "TensorRT plan files are native executable runtime artifacts. DEEPBOM accepts identity-bound collector JSON and never deserializes an untrusted plan in the browser or public service.",
    },
    source_basis: TENSORRT_SOURCE_METADATA,
    interpretation_boundary: "Static preflight validates artifact-visible ONNX contracts and an explicit build profile. TensorRT parser acceptance must be observed in the selected native build; engine build, tactic selection, memory allocation, device execution, accuracy, and latency remain separate evidence classes.",
  };
}

export function validateParserObservation(analysis, profile, value) {
  if (!profile) throw new Error("TensorRT parser observation requires a bound build profile.");
  const collector = value?.collector;
  const expectedProfileFileSha = sha256TextHex(`${canonicalJson(profile)}\n`);
  if (!value || value.schema !== TENSORRT_PARSER_OBSERVATION_SCHEMA
    || String(value.artifact_sha256 || "").toLowerCase() !== String(analysis.model_sha256 || "").toLowerCase()
    || value.build_profile_sha256 !== profile.profile_sha256
    || value.build_profile_file_sha256 !== expectedProfileFileSha
    || canonicalJson(value.build_profile) !== canonicalJson(profile)
    || value.execution_path !== profile.execution_path
    || !String(value.tensorrt_version || "").trim()
    || !String(value.cuda_version || "").trim()
    || Number(value.device_id) !== profile.device_id
    || (profile.device_compute_capability != null
      && String(value.device_compute_capability || "") !== String(profile.device_compute_capability))
    || !String(value.device_identity || "").trim()
    || (profile.execution_path === "native_tensorrt"
      ? !NATIVE_PARSER_METHODS.has(value.api_method)
      : value.api_method !== "ORT_GetCapability_and_profile_assignment")
    || !Array.isArray(value.subgraphs) || !Array.isArray(value.errors)
    || !collector || !SHA256.test(String(collector.binary_sha256 || ""))
    || !SHA256.test(String(collector.source_set_sha256 || ""))
    || !String(collector.git_commit || "").trim()
    || !["clean", "dirty"].includes(String(collector.git_state || ""))) {
    throw new Error("TensorRT parser observation identity or evidence class is invalid.");
  }
  if (!versionMatches(value.tensorrt_version, profile.expected_tensorrt_version)
    || !versionMatches(value.cuda_version, profile.expected_cuda_version)) {
    throw new Error("TensorRT parser observation runtime version differs from the bound profile.");
  }
  validateObservedPlugins(value.plugins, profile.plugins);
  const validOps = new Set(analysis.ops.map(opIndex));
  const expectedSemantics = value.api_method === "supportsModel"
    ? "legacy_supported_collection_membership" : "per_subgraph_api_flag";
  if (value.subgraph_support_semantics !== expectedSemantics) {
    throw new Error("TensorRT parser observation subgraph semantics do not match its API generation.");
  }
  const seen = new Set();
  for (const [position, subgraph] of value.subgraphs.entries()) {
    if (Number(subgraph.subgraph_index) !== position || typeof subgraph.supported !== "boolean"
      || typeof subgraph.sdk_reported_flag !== "boolean"
      || !Array.isArray(subgraph.node_indices) || !subgraph.node_indices.length) {
      throw new Error(`TensorRT parser subgraph #${position} is invalid.`);
    }
    for (const node of subgraph.node_indices) {
      const index = Number(node);
      if (!Number.isSafeInteger(index) || !validOps.has(index) || seen.has(index)) {
        throw new Error(`TensorRT parser observation has an invalid or duplicate node #${node}.`);
      }
      seen.add(index);
    }
  }
  if (value.errors.some((row) => !Number.isSafeInteger(Number(row?.code)) || !String(row?.message || "").trim())) {
    throw new Error("TensorRT parser error ledger is invalid.");
  }
  const normalized = structuredCloneSafe(value);
  normalized.observed_node_count = seen.size;
  normalized.unobserved_node_count = analysis.ops.length - seen.size;
  normalized.coverage_status = seen.size === analysis.ops.length ? "complete" : "partial";
  return normalized;
}

function validateBuildSemantics(analysis, profile, inputs) {
  const issues = [];
  const inputByName = new Map(inputs.map((input) => [input.name, input]));
  const dynamicInputs = inputs.filter((input) => input.dynamic_dimension_indices.length || input.shape_status !== "static");
  if (dynamicInputs.length && !profile.optimization_profiles.length) {
    issues.push(issue("dynamic_profile_missing", "BLOCKING", `${dynamicInputs.length} dynamic or unresolved external input(s) require optimization-profile bounds.`, "Provide min/opt/max dimensions for every dynamic external input in every selected optimization profile."));
  }
  for (const candidate of profile.optimization_profiles) {
    const seen = new Set();
    for (const binding of candidate.inputs) {
      const input = inputByName.get(binding.name);
      if (!input) {
        issues.push(issue("profile_unknown_input", "BLOCKING", `Optimization profile ${candidate.id} references unknown input ${binding.name}.`, "Use the exact serialized ONNX external-input name."));
        continue;
      }
      if (seen.has(binding.name)) issues.push(issue("profile_duplicate_input", "BLOCKING", `Optimization profile ${candidate.id} duplicates ${binding.name}.`, "Emit one min/opt/max binding per input."));
      seen.add(binding.name);
      if (input.rank == null || [binding.min, binding.opt, binding.max].some((shape) => shape.length !== input.rank)) {
        issues.push(issue("profile_rank_mismatch", "BLOCKING", `Optimization profile ${candidate.id} rank for ${binding.name} differs from the ONNX contract.`, "Match every min/opt/max vector to the serialized input rank."));
        continue;
      }
      for (let axis = 0; axis < input.rank; axis += 1) {
        const min = binding.min[axis]; const opt = binding.opt[axis]; const max = binding.max[axis];
        if (!(min <= opt && opt <= max)) issues.push(issue("profile_non_monotonic_bounds", "BLOCKING", `Optimization profile ${candidate.id} ${binding.name}[${axis}] is not min <= opt <= max.`, "Correct the profile bounds."));
        const declared = input.shape[axis];
        if (Number.isSafeInteger(declared) && declared >= 0 && (min !== declared || opt !== declared || max !== declared)) {
          issues.push(issue("profile_static_dimension_changed", "BLOCKING", `Optimization profile ${candidate.id} changes static ${binding.name}[${axis}] = ${declared}.`, "Keep statically declared dimensions identical in min, opt, and max."));
        }
      }
    }
    for (const input of dynamicInputs) if (!seen.has(input.name)) {
      issues.push(issue("profile_dynamic_input_missing", "BLOCKING", `Optimization profile ${candidate.id} omits dynamic input ${input.name}.`, "Bind every dynamic external input in every optimization profile."));
    }
  }
  const quantizedSignal = Number(analysis.quantized_tensors || 0) > 0
    || analysis.ops.some((op) => ["QUANTIZELINEAR", "DEQUANTIZELINEAR"].includes(String(op.name || "").replace(/[^A-Za-z]/g, "").toUpperCase()));
  if (profile.precision.int8 && !quantizedSignal && !profile.calibration_cache_sha256) {
    issues.push(issue("int8_calibration_unbound", "BLOCKING", "INT8 is enabled but neither serialized quantization evidence nor a calibration-cache digest is bound.", "Bind an identity-hashed calibration cache or use an ONNX model with explicit quantization contracts."));
  }
  if ((profile.precision.fp16 || profile.precision.bf16 || profile.precision.fp8) && !profile.device_compute_capability) {
    issues.push(issue("precision_device_capability_unbound", "UNRESOLVED", "Reduced-precision policy is selected but device compute capability is not bound.", "Capture the selected CUDA device identity and capability in the native collector."));
  }
  if (profile.dla_core != null && profile.allow_gpu_fallback == null) {
    issues.push(issue("dla_fallback_policy_unbound", "BLOCKING", "A DLA core is selected without an explicit GPU fallback policy.", "Set allow_gpu_fallback true or false."));
  }
  if (!profile.expected_tensorrt_version || !profile.expected_cuda_version) {
    issues.push(issue("runtime_version_unbound", "UNRESOLVED", "Expected TensorRT and CUDA versions are not both bound.", "Bind expected versions and verify observed versions in the collector output."));
  }
  if (profile.workspace_limit_bytes == null) {
    issues.push(issue("workspace_policy_unbound", "UNRESOLVED", "Workspace limit is not bound.", "Set an explicit workspace limit; tactic availability and build feasibility can depend on it."));
  }
  if (Number(analysis.onnx_external_data?.incomplete_tensor_count || analysis.onnx_external_data?.verification_failed_tensor_count || 0) > 0) {
    issues.push(issue("external_data_incomplete", "BLOCKING", "ONNX external initializer data is incomplete or failed verification.", "Supply and hash-verify every external data component before parser capture."));
  }
  if (Number(analysis.onnx_domain_analysis?.registry_issue_count || 0) > 0 || Number(analysis.onnx_domain_analysis?.external_registry_count || 0) > 0) {
    issues.push(issue("custom_domain_resolution_unbound", "UNRESOLVED", "Custom or external-domain resolution remains outside the serialized standard registry.", "Bind required plugin libraries and observe parser errors/subgraphs in the selected build."));
  }
  return deduplicateIssues(issues);
}

function observationRows(ops, observation) {
  const byNode = new Map();
  for (const subgraph of observation.subgraphs) for (const node of subgraph.node_indices) byNode.set(Number(node), subgraph.supported);
  return ops.map((op, position) => {
    const index = opIndex(op, position);
    if (!byNode.has(index)) return { op_index: index, state: S.UNRESOLVED, reason_codes: [], unresolved_predicates: ["parser_subgraph_coverage_missing"] };
    return byNode.get(index)
      ? { op_index: index, state: S.CONDITIONALLY_ELIGIBLE, reason_codes: [], unresolved_predicates: ["engine_build_and_runtime_not_observed"] }
      : { op_index: index, state: S.DEFINITE_EXCLUSION, reason_codes: ["parser_observed_unsupported_subgraph"], unresolved_predicates: [] };
  });
}

function inputContract(tensor) {
  const shape = Array.isArray(tensor?.shape_signature) && tensor.shape_signature.length
    ? tensor.shape_signature.map(normalizeDimension) : Array.isArray(tensor?.shape) ? tensor.shape.map(normalizeDimension) : [];
  const shapeStatus = shape.length && shape.every((dim) => Number.isSafeInteger(dim) && dim >= 0)
    ? "static" : shape.length ? "dynamic_or_symbolic" : "not_assessed_shape_missing";
  return {
    tensor_index: Number.isSafeInteger(Number(tensor?.index)) ? Number(tensor.index) : null,
    name: String(tensor?.name || ""),
    dtype: String(tensor?.dtype || "UNKNOWN"),
    rank: shape.length || null,
    shape,
    shape_status: shapeStatus,
    dynamic_dimension_indices: shape.map((dim, index) => Number.isSafeInteger(dim) && dim >= 0 ? null : index).filter((value) => value != null),
    shape_tensor_status: "not_assessed_requires_parser_network_introspection",
  };
}

function normalizeOptimizationProfiles(values) {
  if (values == null) return [];
  if (!Array.isArray(values)) throw new Error("TensorRT optimization_profiles must be an array.");
  const ids = new Set();
  return values.map((profile, position) => {
    const id = String(profile?.id || `profile_${position}`);
    if (ids.has(id)) throw new Error(`TensorRT optimization profile id duplicates ${id}.`);
    ids.add(id);
    if (!Array.isArray(profile?.inputs)) throw new Error(`TensorRT optimization profile ${id} inputs must be an array.`);
    return { id, inputs: profile.inputs.map((input) => ({
      name: String(input?.name || ""),
      min: positiveShape(input?.min, `${id}.${input?.name}.min`),
      opt: positiveShape(input?.opt, `${id}.${input?.name}.opt`),
      max: positiveShape(input?.max, `${id}.${input?.name}.max`),
    })) };
  });
}

function normalizePlugins(values) {
  if (values == null) return [];
  if (!Array.isArray(values)) throw new Error("TensorRT plugins must be an array.");
  return values.map((plugin) => {
    const path = String(plugin?.path || "").trim();
    const sha256 = nullableSha(plugin?.sha256, `plugin ${path || "unknown"} sha256`);
    if (!path || !sha256) throw new Error("Every TensorRT plugin requires a path and SHA-256 digest.");
    return { path, sha256 };
  });
}

function validateObservedPlugins(observed, expected) {
  if (!Array.isArray(observed) || observed.length !== expected.length) {
    throw new Error("TensorRT parser observation plugin manifest differs from the bound profile.");
  }
  for (let index = 0; index < expected.length; index += 1) {
    const actual = observed[index];
    const wanted = expected[index];
    if (actual?.profile_path !== wanted.path || actual?.sha256 !== wanted.sha256
      || !Number.isSafeInteger(Number(actual?.byte_length)) || Number(actual.byte_length) < 0) {
      throw new Error("TensorRT parser observation plugin manifest differs from the bound profile.");
    }
  }
}

function versionMatches(observed, expected) {
  if (!expected) return true;
  const wanted = String(expected).trim();
  const actual = String(observed || "").trim();
  if (actual === wanted) return true;
  return (actual.match(/\d+(?:\.\d+){1,3}/g) || []).includes(wanted);
}

function normalizeOrtOptions(value) {
  if (!value || typeof value !== "object") throw new Error("ORT TensorRT EP path requires ort_ep_options.");
  return {
    provider_priority: positiveInteger(value.provider_priority, "provider_priority"),
    max_partition_iterations: positiveInteger(value.max_partition_iterations, "max_partition_iterations"),
    min_subgraph_size: positiveInteger(value.min_subgraph_size, "min_subgraph_size"),
    engine_cache_enable: explicitBoolean(value.engine_cache_enable, "engine_cache_enable"),
    timing_cache_enable: explicitBoolean(value.timing_cache_enable, "timing_cache_enable"),
    context_memory_sharing_enable: explicitBoolean(value.context_memory_sharing_enable, "context_memory_sharing_enable"),
  };
}

function notApplicable(analysis, format) {
  return {
    schema: TENSORRT_STATIC_PREFLIGHT_SCHEMA,
    method_version: "1.0.0",
    format: format || "unknown",
    artifact_sha256: String(analysis?.model_sha256 || ""),
    evidence_class: "NOT_APPLICABLE",
    status: "not_applicable_non_onnx",
    build_profile: null,
    input_contracts: [],
    issues: [],
    blocking_issue_count: 0,
    unresolved_issue_count: 0,
    parser_observation: null,
    optimization_profile_cost: null,
    projection: null,
    trust_boundary: { browser_engine_deserialization: "prohibited", browser_plan_execution: "prohibited", accepted_import: TENSORRT_PARSER_OBSERVATION_SCHEMA },
    source_basis: TENSORRT_SOURCE_METADATA,
    interpretation_boundary: "TensorRT ONNX parser and ORT TensorRT EP preflight applies to ONNX artifacts. GGUF and SafeTensors use separate TensorRT-LLM deployment contracts; TFLite and Core ML use their native backend contracts.",
  };
}

function issue(id, severity, observation, action) { return { id, severity, observation, action }; }
function deduplicateIssues(values) { return [...new Map(values.map((row) => [`${row.id}\0${row.observation}`, row])).values()]; }
function normalizeDimension(value) { const number = Number(value); return Number.isSafeInteger(number) ? number : null; }
function opIndex(op, position = 0) { return Number.isSafeInteger(Number(op?.index)) ? Number(op.index) : position; }
function nullableText(value) { const text = String(value ?? "").trim(); return text || null; }
function nullableSha(value, label) { if (value == null || value === "") return null; const text = String(value).toLowerCase(); if (!SHA256.test(text)) throw new Error(`TensorRT ${label} is not SHA-256.`); return text; }
function nonNegativeInteger(value, label) { const number = Number(value); if (!Number.isSafeInteger(number) || number < 0) throw new Error(`TensorRT ${label} must be a non-negative integer.`); return number; }
function positiveInteger(value, label) { const number = Number(value); if (!Number.isSafeInteger(number) || number <= 0) throw new Error(`TensorRT ${label} must be a positive integer.`); return number; }
function nullablePositiveInteger(value, label) { return value == null ? null : positiveInteger(value, label); }
function nullableNonNegativeInteger(value, label) { return value == null ? null : nonNegativeInteger(value, label); }
function nullableBoundedInteger(value, min, max, label) { if (value == null) return null; const number = Number(value); if (!Number.isSafeInteger(number) || number < min || number > max) throw new Error(`TensorRT ${label} must be an integer from ${min} to ${max}.`); return number; }
function explicitBoolean(value, label) { if (typeof value !== "boolean") throw new Error(`TensorRT ${label} must be explicitly true or false.`); return value; }
function positiveShape(value, label) { if (!Array.isArray(value) || !value.length || value.some((dim) => !Number.isSafeInteger(Number(dim)) || Number(dim) <= 0)) throw new Error(`TensorRT ${label} must contain positive integer dimensions.`); return value.map(Number); }
function structuredCloneSafe(value) { return JSON.parse(JSON.stringify(value)); }
