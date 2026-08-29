import { formatBytes, formatDrift, formatPercent, formatPercentRange, formatUs, padOp } from "./format.js";
import { opPrecisionLabel, predictedPartitionBoundaryInventory, predictedPartitionBoundaryPayloadForOp } from "./analysis.js";
import { decodeXnnpReason } from "./reason-codes.js";
import { deriveArenaRuntimeReconciliation, validateRuntimeMemoryEvidence } from "./runtime-memory.js";
import { TFLITE_PROFILE_INFO_SOURCE } from "./tflite-profile-info-adapter.js";
import { TFLITE_DELEGATE_RULEPACK_METADATA } from "./tflite-delegate-rulepack-metadata.js";
import { validateOrtSelectedBuildProviderBinding } from "./ort-selected-build-binding.js";
import { validateOrtBuildAttestation } from "./ort-build-attestation.js";
import { buildRuntimeBackendEvidenceLedger } from "./runtime-backend-evidence-ledger.js";

const RUNTIME_ASSIGNMENT_SCHEMA = "deepbom.runtime_assignment.v1.9";
const V18_RUNTIME_ASSIGNMENT_SCHEMA = "deepbom.runtime_assignment.v1.8";
const PREVIOUS_RUNTIME_ASSIGNMENT_SCHEMA = "deepbom.runtime_assignment.v1.7";
const PREVIOUS_SELECTOR_RUNTIME_ASSIGNMENT_SCHEMA = "deepbom.runtime_assignment.v1.6";
const OLDER_RUNTIME_ASSIGNMENT_SCHEMA = "deepbom.runtime_assignment.v1.5";
const EARLIER_RUNTIME_ASSIGNMENT_SCHEMA = "deepbom.runtime_assignment.v1.4";
const OLD_RUNTIME_ASSIGNMENT_SCHEMA = "deepbom.runtime_assignment.v1.3";
const ANCIENT_RUNTIME_ASSIGNMENT_SCHEMA = "deepbom.runtime_assignment.v1.2";
const HISTORIC_RUNTIME_ASSIGNMENT_SCHEMA = "deepbom.runtime_assignment.v1.1";
const LEGACY_RUNTIME_ASSIGNMENT_SCHEMA = "deepbom.runtime_assignment.v1";
const RUNTIME_ASSIGNMENT_COMPARISON_SCHEMA = "deepbom.runtime_assignment_comparison.v1.4";
const DURATION_SEMANTICS = new Set(["not_collected", "per_original_op_exclusive", "per_partition_total", "per_execution_plan_node_exclusive", "unspecified"]);
const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const ISO_TIMESTAMP_PATTERN = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d{1,9})?(?:Z|([+-])(\d{2}):(\d{2}))$/;
const TEMPLATE_PLACEHOLDER_PATTERN = /^REPLACE_WITH_/i;

function requiredText(value, field, maxLength = 200) {
  const text = String(value ?? "").trim();
  if (!text) throw new Error(`Runtime assignment ${field} is required.`);
  if (text.length > maxLength) throw new Error(`Runtime assignment ${field} exceeds ${maxLength} characters.`);
  return text;
}

function optionalText(value, field, maxLength = 500) {
  if (value == null) return null;
  const text = String(value).trim();
  if (text.length > maxLength) throw new Error(`Runtime assignment ${field} exceeds ${maxLength} characters.`);
  return text || null;
}

export function parseRuntimeAssignmentDocument(text, analysis, { fileSha256 = null } = {}) {
  let source;
  try {
    source = JSON.parse(String(text || ""));
  } catch {
    throw new Error("Runtime assignment file is not valid JSON.");
  }
  if (![RUNTIME_ASSIGNMENT_SCHEMA, V18_RUNTIME_ASSIGNMENT_SCHEMA, PREVIOUS_RUNTIME_ASSIGNMENT_SCHEMA, PREVIOUS_SELECTOR_RUNTIME_ASSIGNMENT_SCHEMA, OLDER_RUNTIME_ASSIGNMENT_SCHEMA, EARLIER_RUNTIME_ASSIGNMENT_SCHEMA, OLD_RUNTIME_ASSIGNMENT_SCHEMA, ANCIENT_RUNTIME_ASSIGNMENT_SCHEMA, HISTORIC_RUNTIME_ASSIGNMENT_SCHEMA, LEGACY_RUNTIME_ASSIGNMENT_SCHEMA].includes(source?.schema)) {
    throw new Error(`Runtime assignment schema must be ${RUNTIME_ASSIGNMENT_SCHEMA}; prior v1.x schemas remain accepted.`);
  }
  const artifactSha = String(source.artifact_sha256 || "").toLowerCase();
  const expectedSha = String(analysis?.model_sha256 || "").toLowerCase();
  if (!SHA256_PATTERN.test(expectedSha) || !SHA256_PATTERN.test(artifactSha) || artifactSha !== expectedSha) {
    throw new Error("Runtime assignment artifact_sha256 does not match the active model.");
  }
  const targetId = String(source.target_profile_id || "");
  if (!targetId || targetId !== String(analysis?.target_profile?.id || "")) {
    throw new Error("Runtime assignment target_profile_id does not match the active target profile.");
  }
  const targetProfileSha = String(source.target_profile_sha256 || "").toLowerCase();
  const expectedTargetProfileSha = String(analysis?.target_profile?.profile_sha256 || "").toLowerCase();
  if (!SHA256_PATTERN.test(expectedTargetProfileSha) || targetProfileSha !== expectedTargetProfileSha) {
    throw new Error("Runtime assignment target_profile_sha256 does not match the active target profile.");
  }
  if (!Array.isArray(source.assignments) || !source.assignments.length) {
    throw new Error("Runtime assignment JSON must contain a non-empty assignments array.");
  }
  if (source.assignments.length > 100000) throw new Error("Runtime assignment contains too many rows.");
  const opByIndex = new Map((analysis?.ops || []).map((op) => [op.index, op]));
  const seen = new Set();
  const assignments = source.assignments.map((item) => {
    const opIndex = Number(item?.op_index);
    const op = opByIndex.get(opIndex);
    if (!Number.isInteger(opIndex) || !op) throw new Error(`Runtime assignment op_index ${item?.op_index} is not present in the active graph.`);
    if (seen.has(opIndex)) throw new Error(`Runtime assignment contains duplicate op_index ${opIndex}.`);
    if (item.op_name && String(item.op_name) !== op.name) throw new Error(`Runtime assignment op_name does not match op #${opIndex}.`);
    const provider = requiredText(item.provider, `assignments[${opIndex}].provider`);
    if (item.delegated != null && typeof item.delegated !== "boolean") {
      throw new Error(`Runtime assignment op #${opIndex} delegated must be boolean or null.`);
    }
    const duration = item.duration_us == null ? null : Number(item.duration_us);
    if (duration != null && (!Number.isFinite(duration) || duration < 0)) {
      throw new Error(`Runtime assignment op #${opIndex} duration_us must be a non-negative finite number.`);
    }
    seen.add(opIndex);
    const sampleCount = item.sample_count == null ? null : Number(item.sample_count);
    const durationSum = item.duration_sum_us == null ? null : Number(item.duration_sum_us);
    if (sampleCount != null && (!Number.isSafeInteger(sampleCount) || sampleCount <= 0)) {
      throw new Error(`Runtime assignment op #${opIndex} sample_count must be a positive safe integer.`);
    }
    if (durationSum != null && (!Number.isFinite(durationSum) || durationSum < 0)) {
      throw new Error(`Runtime assignment op #${opIndex} duration_sum_us must be a non-negative finite number.`);
    }
    if ((sampleCount == null) !== (durationSum == null)) {
      throw new Error(`Runtime assignment op #${opIndex} sample_count and duration_sum_us must be provided together.`);
    }
    if (sampleCount != null && duration != null && !closeNumber(duration, durationSum / sampleCount)) {
      throw new Error(`Runtime assignment op #${opIndex} duration_us must equal duration_sum_us / sample_count.`);
    }
    const lowerings = validateLoweringObservations(item.lowerings, item, opIndex);
    const dispatches = validateDispatchObservations(item.dispatches, item, opIndex);
    return {
      op_index: opIndex,
      op_name: op.name,
      provider,
      delegated: item.delegated == null ? null : item.delegated,
      partition_id: optionalText(item.partition_id, `assignments[${opIndex}].partition_id`, 100),
      kernel: optionalText(item.kernel, `assignments[${opIndex}].kernel`, 300),
      kernel_id: optionalText(item.kernel_id, `assignments[${opIndex}].kernel_id`, 300),
      kernel_source_ref: optionalText(item.kernel_source_ref, `assignments[${opIndex}].kernel_source_ref`, 600),
      kernel_build_identifier_sha256: item.kernel_build_identifier_sha256 == null ? null : requiredSha(item.kernel_build_identifier_sha256, `assignments[${opIndex}].kernel_build_identifier_sha256`),
      lowering_id: optionalText(item.lowering_id, `assignments[${opIndex}].lowering_id`, 300),
      duration_us: duration,
      duration_sum_us: durationSum,
      sample_count: sampleCount,
      mapping_method: optionalText(item.mapping_method, `assignments[${opIndex}].mapping_method`, 120),
      runtime_node_index: item.runtime_node_index == null ? null : requiredNonNegativeInteger(item.runtime_node_index, `assignments[${opIndex}].runtime_node_index`),
      runtime_node_name: optionalText(item.runtime_node_name, `assignments[${opIndex}].runtime_node_name`, 300),
      graph_node_name: optionalText(item.graph_node_name, `assignments[${opIndex}].graph_node_name`, 300),
      lowerings,
      dispatches,
    };
  }).sort((left, right) => left.op_index - right.op_index);
  const durationSemantics = source.schema === LEGACY_RUNTIME_ASSIGNMENT_SCHEMA
    ? "unspecified"
    : requiredText(source.source?.duration_semantics, "source.duration_semantics", 80);
  if (!DURATION_SEMANTICS.has(durationSemantics)) {
    throw new Error(`Runtime assignment source.duration_semantics must be one of: ${[...DURATION_SEMANTICS].join(", ")}.`);
  }
  const hasDispatchInventory = assignments.some((item) => item.dispatches.length > 0);
  const dispatchSampleSemantics = [RUNTIME_ASSIGNMENT_SCHEMA, V18_RUNTIME_ASSIGNMENT_SCHEMA].includes(source.schema) && hasDispatchInventory
    ? requiredExactText(source.source?.dispatch_sample_semantics, "source.dispatch_sample_semantics", "unique_context_function_selection_per_process")
    : optionalText(source.source?.dispatch_sample_semantics, "source.dispatch_sample_semantics", 120)
      || (hasDispatchInventory ? "unspecified_legacy" : "not_applicable");
  const normalized = {
    schema: RUNTIME_ASSIGNMENT_SCHEMA,
    source_schema: source.schema,
    evidence_class: "OBSERVED_RUNTIME",
    artifact_sha256: artifactSha,
    target_profile_id: targetId,
    target_profile_sha256: targetProfileSha,
    runtime: {
      name: requiredRuntimeIdentity(source.runtime?.name, "runtime.name"),
      version: requiredRuntimeIdentity(source.runtime?.version, "runtime.version"),
      backend: requiredRuntimeIdentity(source.runtime?.backend, "runtime.backend"),
      build: requiredRuntimeIdentity(source.runtime?.build, "runtime.build", 500),
      binary_sha256: source.runtime?.binary_sha256 == null ? null : requiredSha(source.runtime.binary_sha256, "runtime.binary_sha256"),
      graph_optimization_level: optionalEnum(source.runtime?.graph_optimization_level, "runtime.graph_optimization_level", ["disabled", "basic", "extended", "all", "unknown"]),
      execution_mode: optionalEnum(source.runtime?.execution_mode, "runtime.execution_mode", ["sequential", "parallel", "unknown"]),
    },
    source: {
      kind: requiredText(source.source?.kind, "source.kind"),
      collected_at: validatedTimestamp(source.source?.collected_at, source.schema !== LEGACY_RUNTIME_ASSIGNMENT_SCHEMA),
      capture_id: optionalText(source.source?.capture_id, "source.capture_id", 160),
      capture_binding_semantics: optionalText(source.source?.capture_binding_semantics, "source.capture_binding_semantics", 120),
      assignment_semantics: source.schema === LEGACY_RUNTIME_ASSIGNMENT_SCHEMA
        ? "original_graph_op_assignment"
        : requiredExactText(source.source?.assignment_semantics, "source.assignment_semantics", "original_graph_op_assignment"),
      partition_semantics: source.schema === LEGACY_RUNTIME_ASSIGNMENT_SCHEMA
        ? "partition_id_identifies_runtime_partition_when_present"
        : requiredExactText(source.source?.partition_semantics, "source.partition_semantics", "partition_id_identifies_runtime_partition_when_present"),
      dispatch_sample_semantics: dispatchSampleSemantics,
      duration_semantics: durationSemantics,
      duration_statistic: optionalText(source.source?.duration_statistic, "source.duration_statistic", 120),
      profile_sha256: source.source?.profile_sha256 == null ? null : requiredSha(source.source.profile_sha256, "source.profile_sha256"),
      import_file_sha256: fileSha256 == null ? null : requiredSha(fileSha256, "source.import_file_sha256"),
      adapter: null,
      collector: null,
    },
    selector_context: null,
    tflite_delegate_build_inventory: null,
    runtime_memory: null,
    arena_reconciliation: null,
    assignment_count: assignments.length,
    graph_op_count: (analysis?.ops || []).length,
    coverage_ratio: assignments.length / Math.max(1, (analysis?.ops || []).length),
    assignments,
    interpretation_boundary: `Observed provider/kernel assignment for the artifact SHA-256, target-profile SHA-256, and declared runtime export; dispatch samples use ${dispatchSampleSemantics}. Logical boundary payload does not establish tensor-copy materialization. Duration totals require semantics and coverage checks appropriate to original-op or execution-plan-node evidence.`,
  };
  normalized.source.adapter = validateRuntimeAdapter(source.source?.adapter, normalized, analysis);
  normalized.source.collector = validateNativeCollector(source.source?.collector, normalized.source.kind);
  normalized.runtime_memory = validateRuntimeMemoryEvidence(source.runtime_memory, analysis, {
    sourceSchema: source.schema,
    collector: normalized.source.collector,
  });
  if (normalized.source.collector?.instrumentation?.arena_allocations && !normalized.runtime_memory) {
    throw new Error("Runtime collector declared arena allocation instrumentation but emitted no runtime_memory evidence.");
  }
  normalized.arena_reconciliation = deriveArenaRuntimeReconciliation(analysis, normalized.runtime_memory);
  normalized.selector_context = validateSelectorContext(source.selector_context, normalized, source.schema, analysis);
  normalized.tflite_delegate_build_inventory = validateTfliteDelegateBuildInventory(
    source.tflite_delegate_build_inventory,
    normalized,
    analysis,
  );
  normalized.selector_observation = deriveSelectorObservation(normalized);
  normalized.comparison = deriveRuntimeAssignmentComparison(analysis, normalized);
  return normalized;
}

function validateLoweringObservations(values, assignment, opIndex) {
  const fallback = assignment.lowering_id ? [{ lowering_id: assignment.lowering_id, runtime_node_id: null, observation_count: 1 }] : [];
  const source = values == null ? fallback : values;
  if (!Array.isArray(source) || source.length > 4096) throw new Error(`Runtime assignment op #${opIndex} lowerings must be an array with at most 4096 rows.`);
  const rows = source.map((item, index) => ({
    lowering_id: requiredText(item?.lowering_id, `assignments[${opIndex}].lowerings[${index}].lowering_id`, 300),
    runtime_node_id: item?.runtime_node_id == null ? null : requiredNonNegativeInteger(item.runtime_node_id, `assignments[${opIndex}].lowerings[${index}].runtime_node_id`),
    observation_count: requiredPositiveInteger(item?.observation_count ?? 1, `assignments[${opIndex}].lowerings[${index}].observation_count`),
  }));
  const keys = rows.map((item) => `${item.lowering_id}\0${item.runtime_node_id ?? ""}`);
  const sorted = [...rows].sort((left, right) => left.lowering_id.localeCompare(right.lowering_id)
    || compareOptionalInteger(left.runtime_node_id, right.runtime_node_id));
  if (new Set(keys).size !== keys.length || rows.some((item, index) => item !== sorted[index])) {
    throw new Error(`Runtime assignment op #${opIndex} lowerings must be unique and canonically sorted.`);
  }
  const unique = rows.length === 1 ? rows[0].lowering_id : null;
  if (unique !== (assignment.lowering_id == null ? null : String(assignment.lowering_id))) {
    throw new Error(`Runtime assignment op #${opIndex} singular lowering_id does not match the deterministic lowering inventory.`);
  }
  return rows;
}

function validateDispatchObservations(values, assignment, opIndex) {
  const singular = assignment.kernel == null ? [] : [{
    lowering_id: assignment.lowering_id,
    runtime_node_id: null,
    compute_invocation_id: null,
    kernel_id: assignment.kernel_id,
    kernel: assignment.kernel,
    kernel_source_ref: assignment.kernel_source_ref,
    kernel_build_identifier_sha256: assignment.kernel_build_identifier_sha256,
    duration_us: null,
    duration_sum_us: null,
    sample_count: 1,
  }];
  const source = values == null ? singular : values;
  if (!Array.isArray(source) || source.length > 16384) throw new Error(`Runtime assignment op #${opIndex} dispatches must be an array with at most 16384 rows.`);
  const rows = source.map((item, index) => {
    const sampleCount = requiredPositiveInteger(item?.sample_count, `assignments[${opIndex}].dispatches[${index}].sample_count`);
    const duration = item?.duration_us == null ? null : Number(item.duration_us);
    const durationSum = item?.duration_sum_us == null ? null : Number(item.duration_sum_us);
    if ((duration == null) !== (durationSum == null) || (duration != null && (!Number.isFinite(duration) || duration < 0 || !Number.isFinite(durationSum) || durationSum < 0 || !closeNumber(duration, durationSum / sampleCount)))) {
      throw new Error(`Runtime assignment op #${opIndex} dispatch timing must be a finite mean/sum pair consistent with sample_count.`);
    }
    return {
      lowering_id: requiredText(item?.lowering_id, `assignments[${opIndex}].dispatches[${index}].lowering_id`, 300),
      runtime_node_id: item?.runtime_node_id == null ? null : requiredNonNegativeInteger(item.runtime_node_id, `assignments[${opIndex}].dispatches[${index}].runtime_node_id`),
      compute_invocation_id: item?.compute_invocation_id == null ? null : requiredNonNegativeInteger(item.compute_invocation_id, `assignments[${opIndex}].dispatches[${index}].compute_invocation_id`),
      kernel_id: requiredText(item?.kernel_id, `assignments[${opIndex}].dispatches[${index}].kernel_id`, 300),
      kernel: requiredText(item?.kernel, `assignments[${opIndex}].dispatches[${index}].kernel`, 300),
      kernel_source_ref: requiredText(item?.kernel_source_ref, `assignments[${opIndex}].dispatches[${index}].kernel_source_ref`, 600),
      kernel_build_identifier_sha256: requiredSha(item?.kernel_build_identifier_sha256, `assignments[${opIndex}].dispatches[${index}].kernel_build_identifier_sha256`),
      duration_us: duration,
      duration_sum_us: durationSum,
      sample_count: sampleCount,
    };
  });
  const keys = rows.map((item) => [item.lowering_id, item.runtime_node_id ?? "", item.compute_invocation_id ?? "", item.kernel_id, item.kernel, item.kernel_source_ref, item.kernel_build_identifier_sha256].join("\0"));
  const sorted = [...rows].sort((left, right) => left.lowering_id.localeCompare(right.lowering_id)
    || compareOptionalInteger(left.runtime_node_id, right.runtime_node_id)
    || compareOptionalInteger(left.compute_invocation_id, right.compute_invocation_id)
    || left.kernel_id.localeCompare(right.kernel_id)
    || left.kernel.localeCompare(right.kernel)
    || left.kernel_source_ref.localeCompare(right.kernel_source_ref)
    || left.kernel_build_identifier_sha256.localeCompare(right.kernel_build_identifier_sha256));
  if (new Set(keys).size !== keys.length || rows.some((item, index) => item !== sorted[index])) {
    throw new Error(`Runtime assignment op #${opIndex} dispatches must be unique and canonically sorted.`);
  }
  const uniqueLowering = assignment.lowering_id == null ? null : String(assignment.lowering_id);
  const loweringInventoryIsSingular = assignment.lowerings == null
    ? uniqueLowering != null
    : Array.isArray(assignment.lowerings)
      && assignment.lowerings.length === 1
      && String(assignment.lowerings[0]?.lowering_id || "") === uniqueLowering;
  const unique = rows.length === 1
    && loweringInventoryIsSingular
    && rows[0].lowering_id === uniqueLowering
    ? rows[0]
    : null;
  for (const [field, value] of [["kernel_id", assignment.kernel_id], ["kernel", assignment.kernel], ["kernel_source_ref", assignment.kernel_source_ref], ["kernel_build_identifier_sha256", assignment.kernel_build_identifier_sha256]]) {
    if ((unique?.[field] ?? null) !== (value == null ? null : String(value))) throw new Error(`Runtime assignment op #${opIndex} singular ${field} does not match the deterministic dispatch inventory.`);
  }
  return rows;
}

function compareOptionalInteger(left, right) {
  if (left == null) return right == null ? 0 : -1;
  if (right == null) return 1;
  return left - right;
}

export function buildRuntimeAssignmentTemplate(analysis) {
  return {
    schema: RUNTIME_ASSIGNMENT_SCHEMA,
    artifact_sha256: analysis?.model_sha256 || "",
    target_profile_id: analysis?.target_profile?.id || "",
    target_profile_sha256: analysis?.target_profile?.profile_sha256 || "",
    runtime: {
      name: "REPLACE_WITH_RUNTIME_NAME",
      version: "REPLACE_WITH_RUNTIME_VERSION",
      backend: "REPLACE_WITH_BACKEND",
      build: "REPLACE_WITH_BUILD_ID_AND_RELEVANT_FLAGS",
      binary_sha256: null,
    },
    source: {
      kind: "interpreter_plan_export",
      collected_at: null,
      assignment_semantics: "original_graph_op_assignment",
      partition_semantics: "partition_id_identifies_runtime_partition_when_present",
      duration_semantics: "not_collected",
      collector: null,
    },
    selector_context: null,
    tflite_delegate_build_inventory: null,
    native_selector_capture_contract: {
      runtime_assignment_schema: RUNTIME_ASSIGNMENT_SCHEMA,
      source_kind: "deepbom_native_runtime_capture",
      collector_schema: "deepbom.native_runtime_collector.v1.1",
      selector_context_schema: "deepbom.runtime_selector_context.v1.1",
      tflite_delegate_build_inventory_schema: "deepbom.tflite_delegate_build_inventory.v1",
      selector_observation_schema: "deepbom.runtime_selector_observation.v1",
      runtime_memory_schema: "deepbom.runtime_memory.v1",
      arena_reconciliation_schema: "deepbom.arena_runtime_reconciliation.v1",
      required_build_definitions: ["DEEPBOM_RUNTIME_INSTRUMENTATION", "XNN_BUILD_ALL_MICROKERNELS", "XNN_ENABLE_ASSEMBLY"],
      optional_delegate_build_definitions: ["TFLITE_ENABLE_GPU", "TFLITE_ENABLE_NNAPI"],
      required_op_fields_for_microkernel: ["mapping_method=native_runtime_original_op_instrumentation", "lowerings[]", "dispatches[].lowering_id", "dispatches[].kernel_id", "dispatches[].kernel", "dispatches[].kernel_source_ref", "dispatches[].kernel_build_identifier_sha256"],
      closure_rule: "runtime_architecture_identity + compile_configuration + lowering_shape + runtime_dispatch must be present in one artifact/build/invocation-bound capture",
      attestation_boundary: "v1 requires attestation_status=not_attested; a browser-verifiable signature profile is not yet defined",
    },
    graph_ops: (analysis?.ops || []).map((op) => ({
      op_index: Number(op.index),
      op_name: op.name || "",
      input_tensor_ids: (op.inputs || []).map(Number),
      output_tensor_ids: (op.outputs || []).map(Number),
      predicted_delegated: Number(op.xnnpack_chain_id) >= 0,
      predicted_chain_id: Number(op.xnnpack_chain_id) >= 0 ? Number(op.xnnpack_chain_id) : null,
      reference_only: true,
    })),
    assignments: [],
    assignment_row_schema: {
      op_index: "integer; must exist in the active graph",
      op_name: "exact parsed op name",
      provider: "observed provider/delegate name",
      delegated: "boolean or null",
      partition_id: "string or null",
      kernel: "observed kernel symbol or null",
      kernel_id: "stable instrumented runtime kernel ID or null",
      kernel_source_ref: "pinned source reference for the observed kernel or null",
      kernel_build_identifier_sha256: "must equal selector_context.build.microkernel_build_identifier_sha256 when kernel is observed",
      lowering_id: "instrumented runtime lowering path ID or null",
      duration_us: "non-negative finite number or null",
    },
  };
}

function requiredExactText(value, field, expected) {
  const text = requiredText(value, field, 120);
  if (text !== expected) throw new Error(`Runtime assignment ${field} must be ${expected}.`);
  return text;
}

function requiredRuntimeIdentity(value, field, maxLength = 200) {
  const text = requiredText(value, field, maxLength);
  if (TEMPLATE_PLACEHOLDER_PATTERN.test(text)) throw new Error(`Runtime assignment ${field} must replace the template placeholder.`);
  return text;
}

function requiredSha(value, field) {
  const sha = String(value || "").toLowerCase();
  if (!SHA256_PATTERN.test(sha)) throw new Error(`Runtime assignment ${field} must be a lowercase SHA-256.`);
  return sha;
}

function requiredNonNegativeInteger(value, field) {
  const number = Number(value);
  if (!Number.isSafeInteger(number) || number < 0) throw new Error(`Runtime assignment ${field} must be a non-negative safe integer.`);
  return number;
}

function optionalEnum(value, field, allowed) {
  if (value == null || value === "") return null;
  const text = String(value);
  if (!allowed.includes(text)) throw new Error(`Runtime assignment ${field} must be one of: ${allowed.join(", ")}.`);
  return text;
}

function validateNativeCollector(collector, sourceKind) {
  if (collector == null) return null;
  if (sourceKind !== "deepbom_native_runtime_capture") throw new Error("Runtime assignment source.collector is valid only for deepbom_native_runtime_capture evidence.");
  if (!["deepbom.native_runtime_collector.v1", "deepbom.native_runtime_collector.v1.1"].includes(collector.schema)) throw new Error("Runtime assignment source.collector schema is unsupported.");
  if (collector.attestation_status !== "not_attested") throw new Error("Runtime collector attestation cannot be claimed without a browser-verifiable signature path.");
  if (typeof collector.instrumentation?.lowering_ids !== "boolean" || typeof collector.instrumentation?.microkernel_ids !== "boolean"
    || (collector.schema === "deepbom.native_runtime_collector.v1.1" && typeof collector.instrumentation?.arena_allocations !== "boolean")) {
    throw new Error("Runtime collector instrumentation capabilities must be explicit booleans.");
  }
  return {
    schema: collector.schema,
    name: requiredText(collector.name, "source.collector.name", 120),
    version: requiredText(collector.version, "source.collector.version", 120),
    source_commit: requiredText(collector.source_commit, "source.collector.source_commit", 200),
    binary_sha256: requiredSha(collector.binary_sha256, "source.collector.binary_sha256"),
    attestation_status: "not_attested",
    instrumentation: {
      lowering_ids: collector.instrumentation.lowering_ids,
      microkernel_ids: collector.instrumentation.microkernel_ids,
      arena_allocations: collector.schema === "deepbom.native_runtime_collector.v1.1" ? collector.instrumentation.arena_allocations : false,
    },
  };
}

function validateSelectorContext(context, normalized, sourceSchema, analysis) {
  const selectorClaims = normalized.assignments.some((item) => item.lowerings.length || item.dispatches.length || item.lowering_id || item.kernel || item.kernel_id || item.kernel_source_ref || item.kernel_build_identifier_sha256);
  if (context == null) {
    if (selectorClaims) throw new Error("Runtime selector claims require a v1.7 selector_context from an instrumented native capture.");
    return null;
  }
  if (![RUNTIME_ASSIGNMENT_SCHEMA, V18_RUNTIME_ASSIGNMENT_SCHEMA, PREVIOUS_RUNTIME_ASSIGNMENT_SCHEMA].includes(sourceSchema)) throw new Error("Runtime selector_context requires deepbom.runtime_assignment.v1.7, v1.8, or v1.9.");
  if (normalized.source.kind !== "deepbom_native_runtime_capture" || !normalized.source.collector) throw new Error("Runtime selector_context requires a declared DeepBOM native collector export.");
  if (!normalized.source.capture_id) throw new Error("Runtime selector_context requires source.capture_id.");
  if (!normalized.source.import_file_sha256) throw new Error("Runtime selector_context requires the imported canonical capture file SHA-256.");
  const expectedSelectorSchema = sourceSchema === RUNTIME_ASSIGNMENT_SCHEMA
    ? "deepbom.runtime_selector_context.v1.1"
    : "deepbom.runtime_selector_context.v1";
  if (context.schema !== expectedSelectorSchema || context.backend_library !== "XNNPACK") throw new Error("Runtime selector_context schema or backend library is unsupported.");
  if (!normalized.runtime.binary_sha256) throw new Error("Runtime selector_context requires runtime.binary_sha256.");

  const architecture = requiredText(context.device?.architecture, "selector_context.device.architecture", 80);
  const expectedArchitecture = targetArchitectureFamily(normalized.target_profile_id);
  if (expectedArchitecture && architecture !== expectedArchitecture) {
    throw new Error(`Runtime selector_context device.architecture must be ${expectedArchitecture} for target profile ${normalized.target_profile_id}.`);
  }
  const featureSource = optionalEnum(context.device?.cpu_feature_source, "selector_context.device.cpu_feature_source", ["native_os_api", "runtime_library_api", "rust_std_runtime_detection"]);
  const cpuFeatures = canonicalUniqueStrings(context.device?.cpu_features, "selector_context.device.cpu_features", 128, 80);
  if (!cpuFeatures.length || !featureSource) throw new Error("Runtime selector_context requires native-observed CPU features and their source.");

  const sourceCommit = String(context.build?.xnnpack_source_commit || "").toLowerCase();
  if (!/^[a-f0-9]{40}$/.test(sourceCommit)) throw new Error("Runtime selector_context build.xnnpack_source_commit must be a commit hash.");
  const tensorflowSourceCommit = context.build?.tensorflow_source_commit == null
    ? null
    : String(context.build.tensorflow_source_commit).toLowerCase();
  if (tensorflowSourceCommit != null && !/^[a-f0-9]{40}$/.test(tensorflowSourceCommit)) {
    throw new Error("Runtime selector_context build.tensorflow_source_commit must be a commit hash when present.");
  }
  const definitions = validateCompileDefinitions(context.build?.compile_definitions);
  for (const required of ["XNN_ENABLE_ASSEMBLY", "XNN_BUILD_ALL_MICROKERNELS"]) {
    const definition = definitions.find((item) => item.name === required);
    if (!definition) throw new Error(`Runtime selector_context compile_definitions must include ${required}.`);
    definition.value = normalizedBuildBoolean(definition.value, `selector_context.build.compile_definitions.${required}`);
  }
  const runtimeBinarySha = requiredSha(context.build?.runtime_binary_sha256, "selector_context.build.runtime_binary_sha256");
  if (runtimeBinarySha !== normalized.runtime.binary_sha256) throw new Error("Runtime selector_context build/runtime binary SHA-256 values differ.");
  const buildIdentifierSha = requiredSha(context.build?.microkernel_build_identifier_sha256, "selector_context.build.microkernel_build_identifier_sha256");
  const buildManifestSha = requiredSha(context.build?.build_manifest_sha256, "selector_context.build.build_manifest_sha256");

  const inputShapes = validateInvocationShapes(context.invocation?.inputs, analysis);
  const threadCount = requiredPositiveInteger(context.invocation?.thread_count, "selector_context.invocation.thread_count");
  const runtimeOptionsSha = requiredSha(context.invocation?.runtime_options_sha256, "selector_context.invocation.runtime_options_sha256");
  const resourcePartition = validateResourcePartition(context.invocation?.resource_partition, sourceSchema, threadCount);
  const collector = normalized.source.collector;
  for (const item of normalized.assignments) {
    const hasKernelClaim = Boolean(item.kernel || item.kernel_id || item.kernel_source_ref || item.kernel_build_identifier_sha256);
    if ((item.kernel == null) !== (item.kernel_id == null) || (item.kernel == null) !== (item.kernel_source_ref == null) || (item.kernel == null) !== (item.kernel_build_identifier_sha256 == null)) {
      throw new Error(`Runtime assignment op #${item.op_index} microkernel symbol, stable ID, source, and build identifier must be provided together.`);
    }
    if ((item.lowering_id || hasKernelClaim) && item.mapping_method !== "native_runtime_original_op_instrumentation") {
      throw new Error(`Runtime assignment op #${item.op_index} selector evidence requires native original-op instrumentation mapping.`);
    }
    if (item.lowering_id && !collector.instrumentation.lowering_ids) throw new Error(`Runtime assignment op #${item.op_index} lowering claim exceeds collector capabilities.`);
    if (hasKernelClaim) {
      if (!item.lowering_id) throw new Error(`Runtime assignment op #${item.op_index} microkernel evidence requires an observed lowering_id.`);
      if (!collector.instrumentation.microkernel_ids) throw new Error(`Runtime assignment op #${item.op_index} microkernel claim exceeds collector capabilities.`);
      if (item.kernel_build_identifier_sha256 !== buildIdentifierSha) throw new Error(`Runtime assignment op #${item.op_index} kernel/build identifier mismatch.`);
      if (!/^xnn_[A-Za-z0-9_]*ukernel[A-Za-z0-9_]*$/.test(item.kernel)) throw new Error(`Runtime assignment op #${item.op_index} kernel must be a concrete XNNPACK microkernel symbol.`);
      if (!/^[a-z0-9][a-z0-9._:-]{2,299}$/.test(item.kernel_id)) throw new Error(`Runtime assignment op #${item.op_index} kernel_id must be a stable machine identifier.`);
      if (!/^[a-z0-9][a-z0-9._:-]{2,299}$/.test(item.lowering_id)) throw new Error(`Runtime assignment op #${item.op_index} lowering_id must be a stable machine identifier.`);
      if (!xnnpackSourcePinnedToCommit(item.kernel_source_ref, sourceCommit)) throw new Error(`Runtime assignment op #${item.op_index} kernel source is not pinned to the captured XNNPACK commit.`);
    }
    if (item.lowerings.length && !collector.instrumentation.lowering_ids) throw new Error(`Runtime assignment op #${item.op_index} lowering inventory exceeds collector capabilities.`);
    if (item.dispatches.length && !collector.instrumentation.microkernel_ids) throw new Error(`Runtime assignment op #${item.op_index} dispatch inventory exceeds collector capabilities.`);
    for (const dispatch of item.dispatches) {
      if (dispatch.kernel_build_identifier_sha256 !== buildIdentifierSha) throw new Error(`Runtime assignment op #${item.op_index} dispatch/build identifier mismatch.`);
      if (!/^xnn_[A-Za-z0-9_]*ukernel[A-Za-z0-9_]*$/.test(dispatch.kernel)) throw new Error(`Runtime assignment op #${item.op_index} dispatch kernel must be a concrete XNNPACK microkernel symbol.`);
      if (!/^[a-z0-9][a-z0-9._:-]{2,299}$/.test(dispatch.kernel_id) || !/^[a-z0-9][a-z0-9._:-]{2,299}$/.test(dispatch.lowering_id)) {
        throw new Error(`Runtime assignment op #${item.op_index} dispatch IDs must be stable machine identifiers.`);
      }
      if (!xnnpackSourcePinnedToCommit(dispatch.kernel_source_ref, sourceCommit)) throw new Error(`Runtime assignment op #${item.op_index} dispatch source is not pinned to the captured XNNPACK commit.`);
    }
  }
  return {
    schema: context.schema,
    backend_library: "XNNPACK",
    device: {
      architecture,
      identity: requiredText(context.device?.identity, "selector_context.device.identity", 200),
      cpu_feature_source: featureSource,
      cpu_features: cpuFeatures,
    },
    build: {
      runtime_binary_sha256: runtimeBinarySha,
      tensorflow_source_commit: tensorflowSourceCommit,
      xnnpack_source_commit: sourceCommit,
      microkernel_build_identifier_sha256: buildIdentifierSha,
      build_manifest_sha256: buildManifestSha,
      compile_definitions: definitions,
    },
    invocation: {
      inputs: inputShapes,
      thread_count: threadCount,
      runtime_options_sha256: runtimeOptionsSha,
      resource_partition: resourcePartition,
    },
    interpretation_boundary: "CPU features, build identity, flags, and invocation shapes are declared by a binary-identified but not cryptographically attested native collector. Lowering and microkernel dimensions close only on op rows carrying complete instrumented identities.",
  };
}

function validateTfliteDelegateBuildInventory(inventory, normalized, analysis) {
  if (inventory == null) return null;
  if (String(analysis?.format || "").toLowerCase() !== "tflite") {
    throw new Error("TFLite delegate build inventory is not applicable to a non-TFLite artifact.");
  }
  if (!normalized.selector_context || inventory.schema !== "deepbom.tflite_delegate_build_inventory.v1"
    || inventory.evidence_class !== "DECLARED_BUILD_AND_RUNTIME_OPTION_INVENTORY") {
    throw new Error("TFLite delegate build inventory schema or evidence class is invalid.");
  }
  const artifactSha = requiredSha(inventory.artifact_sha256, "tflite_delegate_build_inventory.artifact_sha256");
  const runtimeBinarySha = requiredSha(inventory.runtime_binary_sha256, "tflite_delegate_build_inventory.runtime_binary_sha256");
  const buildManifestSha = requiredSha(inventory.build_manifest_sha256, "tflite_delegate_build_inventory.build_manifest_sha256");
  const tensorflowCommit = String(inventory.tensorflow_source_commit || "").toLowerCase();
  if (artifactSha !== normalized.artifact_sha256
    || runtimeBinarySha !== normalized.runtime.binary_sha256
    || buildManifestSha !== normalized.selector_context.build.build_manifest_sha256
    || tensorflowCommit !== TFLITE_DELEGATE_RULEPACK_METADATA.tensorflowCommit
    || normalized.selector_context.build.tensorflow_source_commit !== tensorflowCommit
    || analysis?.tflite_delegate_compatibility_evidence?.tensorflow_source_commit
      && analysis.tflite_delegate_compatibility_evidence.tensorflow_source_commit !== tensorflowCommit) {
    throw new Error("TFLite delegate build inventory artifact, binary, build manifest, or source binding does not match the active evidence.");
  }

  const definitions = new Map(normalized.selector_context.build.compile_definitions.map((item) => [item.name, item.value]));
  const options = inventory.build_options;
  if (!Array.isArray(options) || options.length !== 2
    || options.map((item) => item?.name).join("|") !== "TFLITE_ENABLE_GPU|TFLITE_ENABLE_NNAPI") {
    throw new Error("TFLite delegate build inventory must contain the canonical GPU and NNAPI CMake options.");
  }
  const cmakeSystemName = optionalText(inventory.cmake_system_name, "tflite_delegate_build_inventory.cmake_system_name", 80);
  const gpuEnabled = validateDelegateBuildOption(options[0], definitions, "TFLITE_ENABLE_GPU");
  const nnapiEnabled = validateDelegateBuildOption(options[1], definitions, "TFLITE_ENABLE_NNAPI");
  const expectedGpuStatus = gpuEnabled == null ? "not_declared"
    : gpuEnabled ? "enabled_by_declared_cmake_option" : "disabled_by_declared_cmake_option";
  const expectedNnapiStatus = nnapiEnabled == null ? "not_declared"
    : !nnapiEnabled ? "disabled_by_declared_cmake_option"
      : cmakeSystemName == null ? "unresolved_cmake_system_name"
        : cmakeSystemName.toLowerCase() === "android" ? "enabled_by_declared_cmake_option_and_android_gate"
          : "disabled_by_non_android_cmake_gate";
  if (options[0].effective_status !== expectedGpuStatus || options[1].effective_status !== expectedNnapiStatus) {
    throw new Error("TFLite delegate build inventory effective CMake status is inconsistent.");
  }

  const gpu = inventory.gpu;
  const flags = gpu?.experimental_flags == null ? null : requiredNonNegativeInteger(gpu.experimental_flags, "tflite_delegate_build_inventory.gpu.experimental_flags");
  const maxPartitions = gpu?.max_delegated_partitions == null ? null : requiredPositiveInteger(gpu.max_delegated_partitions, "tflite_delegate_build_inventory.gpu.max_delegated_partitions");
  const expectedQuantStatus = flags == null ? "not_declared"
    : flags & 1 ? "enabled_by_declared_runtime_option" : "disabled_by_declared_runtime_option";
  if (!gpu || gpu.compiled_status !== expectedGpuStatus || Number(gpu.quantized_model_flag_bit) !== 1
    || gpu.quantized_model_flag_status !== expectedQuantStatus
    || gpu.option_source !== "capture runtime-options.json") {
    throw new Error("TFLite GPU selected-build or runtime-option inventory is inconsistent.");
  }
  const nnapi = inventory.nnapi;
  const featureLevel = nnapi?.runtime_feature_level == null ? null : requiredPositiveInteger(nnapi.runtime_feature_level, "tflite_delegate_build_inventory.nnapi.runtime_feature_level");
  if (featureLevel != null && featureLevel < 27) throw new Error("TFLite NNAPI runtime feature level must be at least 27.");
  const acceleratorIdentity = optionalText(nnapi?.accelerator_identity, "tflite_delegate_build_inventory.nnapi.accelerator_identity", 300);
  const capabilitySource = optionalEnum(nnapi?.capability_source, "tflite_delegate_build_inventory.nnapi.capability_source", ["not_collected", "android_nnapi_runtime_query", "declared_capture_configuration"]);
  if (!nnapi || nnapi.compiled_status !== expectedNnapiStatus || !capabilitySource
    || ((featureLevel != null || acceleratorIdentity != null) && capabilitySource === "not_collected")) {
    throw new Error("TFLite NNAPI selected-build or capability inventory is inconsistent.");
  }
  validateTfliteDelegateBuildSources(inventory.source_files, tensorflowCommit);
  if (!String(inventory.interpretation_boundary || "").includes("do not establish")
    && !String(inventory.interpretation_boundary || "").includes("remain runtime observations")) {
    throw new Error("TFLite delegate build inventory interpretation boundary is missing.");
  }
  return {
    schema: inventory.schema,
    evidence_class: inventory.evidence_class,
    artifact_sha256: artifactSha,
    tensorflow_source_commit: tensorflowCommit,
    runtime_binary_sha256: runtimeBinarySha,
    build_manifest_sha256: buildManifestSha,
    cmake_system_name: cmakeSystemName,
    build_options: options.map((item) => ({
      name: item.name,
      declared_value: item.declared_value == null ? null : String(item.declared_value),
      normalized_enabled: item.normalized_enabled == null ? null : Boolean(item.normalized_enabled),
      effective_status: item.effective_status,
    })),
    gpu: {
      compiled_status: gpu.compiled_status,
      experimental_flags: flags,
      quantized_model_flag_bit: 1,
      quantized_model_flag_status: gpu.quantized_model_flag_status,
      max_delegated_partitions: maxPartitions,
      option_source: gpu.option_source,
    },
    nnapi: {
      compiled_status: nnapi.compiled_status,
      runtime_feature_level: featureLevel,
      accelerator_identity: acceleratorIdentity,
      capability_source: capabilitySource,
    },
    source_files: inventory.source_files.map((item) => ({ ...item })),
    interpretation_boundary: String(inventory.interpretation_boundary),
  };
}

function validateDelegateBuildOption(option, definitions, name) {
  if (!option || option.name !== name) throw new Error(`TFLite delegate build option ${name} is missing.`);
  const declared = definitions.get(name) ?? null;
  if ((option.declared_value == null ? null : String(option.declared_value)) !== declared) {
    throw new Error(`TFLite delegate build option ${name} differs from selector_context compile definitions.`);
  }
  const enabled = declared == null ? null : normalizedBuildBoolean(declared, `tflite_delegate_build_inventory.build_options.${name}`) === "1";
  if (option.normalized_enabled !== enabled) throw new Error(`TFLite delegate build option ${name} normalized state is invalid.`);
  return enabled;
}

function validateTfliteDelegateBuildSources(sources, commit) {
  const expectedIds = ["tflite_cmake_build_options", "tflite_gpu_delegate_options"];
  if (!Array.isArray(sources) || sources.length !== expectedIds.length
    || sources.map((item) => item?.id).join("|") !== expectedIds.join("|")) {
    throw new Error("TFLite delegate selected-build source ledger is incomplete.");
  }
  for (const source of sources) {
    const expected = TFLITE_DELEGATE_RULEPACK_METADATA.sources.find((item) => item.id === source.id);
    if (!expected || source.sha256 !== expected.sha256 || !String(source.source_ref || "").includes(commit)
      || !String(source.source_ref || "").endsWith(expected.path)) {
      throw new Error(`TFLite delegate selected-build source identity is invalid for ${source.id}.`);
    }
  }
}

function deriveSelectorObservation(normalized) {
  const context = normalized.selector_context;
  let loweringCount = 0;
  let microkernelCount = 0;
  let closedCount = 0;
  for (const item of normalized.assignments) {
    const dimensions = [];
    if (context) dimensions.push("runtime_architecture_identity", "compile_configuration");
    if (item.lowerings.length) {
      dimensions.push("lowering_shape");
      loweringCount += 1;
    }
    if (item.dispatches.length) {
      dimensions.push("runtime_dispatch");
      microkernelCount += 1;
    }
    if (dimensions.length === 4) closedCount += 1;
    item.resolved_selector_dimensions = dimensions;
    item.selector_evidence_class = item.dispatches.length
      ? "OBSERVED_MICROKERNEL"
      : item.lowerings.length ? "OBSERVED_LOWERING" : context ? "OBSERVED_RUNTIME_CONTEXT" : "NOT_COLLECTED";
  }
  return {
    schema: "deepbom.runtime_selector_observation.v1",
    status: !context ? "not_collected" : closedCount === normalized.graph_op_count ? "complete_graph" : closedCount ? "partial_graph" : "context_only",
    context_bound: Boolean(context),
    context_evidence_class: context ? "OBSERVED_RUNTIME_UNATTESTED_EXPORT" : "NOT_COLLECTED",
    collector_attestation_status: normalized.source.collector?.attestation_status || "not_present",
    graph_op_count: normalized.graph_op_count,
    assignment_count: normalized.assignment_count,
    lowering_observed_op_count: loweringCount,
    microkernel_observed_op_count: microkernelCount,
    selector_ambiguity_closed_op_count: closedCount,
    interpretation_boundary: "An op closes selector ambiguity only when architecture, compile configuration, lowering, and runtime dispatch are all bound in the same capture. Imported native exports are not cryptographically attested in this schema version.",
  };
}

function canonicalUniqueStrings(values, field, maxItems, maxLength) {
  if (!Array.isArray(values) || values.length > maxItems) throw new Error(`Runtime assignment ${field} must be an array with at most ${maxItems} items.`);
  const normalized = values.map((value, index) => requiredText(value, `${field}[${index}]`, maxLength));
  if (new Set(normalized).size !== normalized.length || JSON.stringify(normalized) !== JSON.stringify([...normalized].sort())) {
    throw new Error(`Runtime assignment ${field} must be unique and lexicographically sorted.`);
  }
  return normalized;
}

function validateResourcePartition(value, sourceSchema, threadCount) {
  if (value == null) return null;
  if (sourceSchema !== RUNTIME_ASSIGNMENT_SCHEMA) {
    throw new Error("Resource-partition evidence requires deepbom.runtime_assignment.v1.9.");
  }
  if (value.schema !== "deepbom.resource_partition_observation.v1"
    || value.evidence_class !== "OBSERVED_OS_RESOURCE_PARTITION") {
    throw new Error("Runtime resource-partition schema or evidence class is unsupported.");
  }
  const requested = canonicalCpuIds(value.requested_cpu_ids, "resource_partition.requested_cpu_ids");
  const effective = canonicalCpuIds(value.observed_effective_cpu_ids || [], "resource_partition.observed_effective_cpu_ids", true);
  const processors = canonicalCpuIds(value.observed_processor_ids || [], "resource_partition.observed_processor_ids", true);
  const online = canonicalCpuIds(value.online_cpu_ids || [], "resource_partition.online_cpu_ids", true);
  if (threadCount > requested.length) {
    throw new Error("Runtime thread count exceeds the requested CPU-set cardinality.");
  }
  if (online.length && !requested.every((cpu) => online.includes(cpu))) {
    throw new Error("Requested runtime CPUs are not all present in the observed online CPU set.");
  }
  if (!processors.every((cpu) => requested.includes(cpu))) {
    throw new Error("Observed runtime processors fall outside the requested CPU set.");
  }
  const affinityStatus = optionalEnum(value.affinity_status, "resource_partition.affinity_status", [
    "observed_all_sampled_threads_within_requested_set",
    "not_observed_or_outside_requested_set",
  ]);
  const isolationStatus = optionalEnum(value.exclusive_isolation_status, "resource_partition.exclusive_isolation_status", [
    "observed_cgroup_v2_isolated_partition",
    "not_observed_affinity_only",
  ]);
  if (affinityStatus !== "observed_all_sampled_threads_within_requested_set") {
    throw new Error("Imported resource-partition evidence does not establish sampled-thread affinity.");
  }
  const sampleCount = requiredPositiveInteger(value.sample_count, "resource_partition.sample_count");
  const maximumThreadCount = requiredPositiveInteger(value.maximum_observed_thread_count, "resource_partition.maximum_observed_thread_count");
  if (maximumThreadCount < threadCount) {
    throw new Error("Resource-partition sampling did not observe the declared runtime thread count.");
  }
  if (!Array.isArray(value.sampled_threads) || !value.sampled_threads.length) {
    throw new Error("Resource-partition evidence requires sampled thread masks.");
  }
  const sampledThreads = value.sampled_threads.map((row, index) => ({
    tid: requiredPositiveInteger(row?.tid, `resource_partition.sampled_threads[${index}].tid`),
    allowed_cpu_ids: canonicalCpuIds(row?.allowed_cpu_ids, `resource_partition.sampled_threads[${index}].allowed_cpu_ids`),
  }));
  if (sampledThreads.some((row) => !row.allowed_cpu_ids.every((cpu) => requested.includes(cpu)))) {
    throw new Error("A sampled thread mask falls outside the requested CPU set.");
  }
  if (sampledThreads.some((row, index) => index > 0 && sampledThreads[index - 1].tid >= row.tid)) {
    throw new Error("Resource-partition sampled threads must be unique and sorted by TID.");
  }
  const allowedUnion = [...new Set(sampledThreads.flatMap((row) => row.allowed_cpu_ids))].sort((left, right) => left - right);
  const declaredAllowedUnion = canonicalCpuIds(value.observed_allowed_cpu_ids_union, "resource_partition.observed_allowed_cpu_ids_union");
  if (JSON.stringify(allowedUnion) !== JSON.stringify(declaredAllowedUnion)
    || JSON.stringify(allowedUnion) !== JSON.stringify(requested)) {
    throw new Error("Sampled thread-mask union must exactly reproduce the requested CPU set.");
  }
  const isolationExpectation = optionalEnum(value.isolation_expectation, "resource_partition.isolation_expectation", ["affinity_only", "exclusive_cpuset"]);
  if (isolationStatus === "observed_cgroup_v2_isolated_partition"
    && (value.cgroup_v2_partition_state !== "isolated" || JSON.stringify(effective) !== JSON.stringify(requested))) {
    throw new Error("Exclusive cpuset status requires an isolated cgroup v2 partition whose effective CPUs exactly match the request.");
  }
  if (isolationExpectation === "exclusive_cpuset"
    && isolationStatus !== "observed_cgroup_v2_isolated_partition") {
    throw new Error("An exclusive_cpuset request requires an observed isolated cgroup v2 partition.");
  }
  return {
    schema: value.schema,
    evidence_class: value.evidence_class,
    observation_sha256: requiredSha(value.observation_sha256, "resource_partition.observation_sha256"),
    requested_cpu_ids: requested,
    affinity_mode: optionalEnum(value.affinity_mode, "resource_partition.affinity_mode", ["taskset_process_and_descendants"]),
    isolation_expectation: isolationExpectation,
    affinity_status: affinityStatus,
    exclusive_isolation_status: isolationStatus,
    sample_count: sampleCount,
    maximum_observed_thread_count: maximumThreadCount,
    sampled_threads: sampledThreads,
    observed_allowed_cpu_ids_union: allowedUnion,
    observed_processor_ids: processors,
    observed_effective_cpu_ids: effective,
    cgroup_v2_path: value.cgroup_v2_path == null ? null : requiredText(value.cgroup_v2_path, "resource_partition.cgroup_v2_path", 512),
    cgroup_v2_partition_state: value.cgroup_v2_partition_state == null ? null : requiredText(value.cgroup_v2_partition_state, "resource_partition.cgroup_v2_partition_state", 80),
    online_cpu_ids: online,
    kernel_command_line: value.kernel_command_line == null ? null : requiredText(value.kernel_command_line, "resource_partition.kernel_command_line", 8192),
    kernel_isolation_parameters: value.kernel_isolation_parameters && typeof value.kernel_isolation_parameters === "object"
      ? { ...value.kernel_isolation_parameters }
      : {},
    cpu_frequency_policy: Array.isArray(value.cpu_frequency_policy) ? value.cpu_frequency_policy.map((row) => ({ ...row })) : [],
    cache_shared_cpu_lists: Array.isArray(value.cache_shared_cpu_lists) ? value.cache_shared_cpu_lists.map((row) => ({ ...row })) : [],
    interpretation_boundary: requiredText(value.interpretation_boundary, "resource_partition.interpretation_boundary", 2048),
  };
}

function canonicalCpuIds(value, field, allowEmpty = false) {
  if (!Array.isArray(value) || (!allowEmpty && !value.length) || value.length > 4096
    || value.some((cpu) => !Number.isSafeInteger(cpu) || cpu < 0)
    || value.some((cpu, index) => index > 0 && value[index - 1] >= cpu)) {
    throw new Error(`Runtime assignment ${field} must be ${allowEmpty ? "an" : "a non-empty"} ascending unique CPU-ID array.`);
  }
  return [...value];
}

function validateCompileDefinitions(values) {
  if (!Array.isArray(values) || !values.length || values.length > 256) throw new Error("Runtime selector_context compile_definitions must be a non-empty array.");
  const definitions = values.map((item, index) => ({
    name: requiredText(item?.name, `selector_context.build.compile_definitions[${index}].name`, 120),
    value: requiredText(item?.value, `selector_context.build.compile_definitions[${index}].value`, 200),
  }));
  if (new Set(definitions.map((item) => item.name)).size !== definitions.length
    || JSON.stringify(definitions.map((item) => item.name)) !== JSON.stringify(definitions.map((item) => item.name).sort())) {
    throw new Error("Runtime selector_context compile_definitions must have unique, lexicographically sorted names.");
  }
  return definitions;
}

function normalizedBuildBoolean(value, field) {
  const normalized = String(value || "").trim().toUpperCase();
  if (["1", "ON", "TRUE", "YES"].includes(normalized)) return "1";
  if (["0", "OFF", "FALSE", "NO"].includes(normalized)) return "0";
  throw new Error(`Runtime assignment ${field} must be an explicit boolean build value.`);
}

function targetArchitectureFamily(targetProfileId) {
  const id = String(targetProfileId || "");
  if (id.startsWith("android_") || id.startsWith("rpi")) return "aarch64";
  if (id.startsWith("x86_")) return "x86_64";
  if (id.startsWith("wasm_")) return "wasm32";
  return "";
}

function xnnpackSourcePinnedToCommit(sourceRef, commit) {
  const value = String(sourceRef || "");
  if (value.includes("..")) return false;
  return value.startsWith(`google/XNNPACK@${commit}/src/`)
    || value.startsWith(`https://github.com/google/XNNPACK/blob/${commit}/src/`)
    || value.startsWith(`https://raw.githubusercontent.com/google/XNNPACK/${commit}/src/`);
}

function validateInvocationShapes(values, analysis) {
  const inputs = analysis?.inputs || [];
  if (!Array.isArray(values) || values.length !== inputs.length) throw new Error("Runtime selector_context invocation inputs must exactly cover model inputs.");
  return inputs.map((input) => {
    const row = values.find((item) => Number(item?.tensor_index) === Number(input.index));
    if (!row || String(row.name || "") !== String(input.name || "")) throw new Error(`Runtime selector_context invocation input T${input.index} identity mismatch.`);
    if (!Array.isArray(row.shape) || row.shape.length !== (input.shape || []).length || row.shape.some((dim) => !Number.isSafeInteger(Number(dim)) || Number(dim) <= 0)) {
      throw new Error(`Runtime selector_context invocation input T${input.index} requires a concrete positive shape.`);
    }
    for (let index = 0; index < row.shape.length; index += 1) {
      const declared = Number(input.shape?.[index] || -1);
      if (declared > 0 && Number(row.shape[index]) !== declared) throw new Error(`Runtime selector_context invocation input T${input.index} shape differs from a static artifact dimension.`);
    }
    return { tensor_index: Number(input.index), name: String(input.name || ""), shape: row.shape.map(Number) };
  });
}

function validateRuntimeAdapter(adapter, normalized, analysis) {
  if (adapter == null) {
    if (["onnxruntime_profile_json_adapter", "tflite_model_runtime_info_proto_adapter", "tflite_model_runtime_info_and_benchmark_profile_proto_adapter"].includes(normalized.source.kind)) {
      throw new Error("Runtime assignment source.adapter is required for adapted runtime evidence.");
    }
    return null;
  }
  if (normalized.source.kind === "onnxruntime_profile_json_adapter") return validateOrtRuntimeAdapter(adapter, normalized, analysis);
  if (["tflite_model_runtime_info_proto_adapter", "tflite_model_runtime_info_and_benchmark_profile_proto_adapter"].includes(normalized.source.kind)) return validateTfliteRuntimeInfoAdapter(adapter, normalized, analysis);
  throw new Error("Runtime assignment source.adapter is not valid for this evidence source kind.");
}

function validateOrtRuntimeAdapter(adapter, normalized, analysis) {
  if (!["deepbom.ort_profile_adapter.v1", "deepbom.ort_profile_adapter.v2", "deepbom.ort_profile_adapter.v2.1", "deepbom.ort_profile_adapter.v2.2"].includes(adapter.schema)) throw new Error("Runtime assignment source.adapter schema is unsupported.");
  if (String(analysis?.format || "").toLowerCase() !== "onnx") throw new Error("ONNX Runtime profile adapter evidence requires an ONNX artifact.");
  const expectedCommit = "microsoft/onnxruntime@8c546c37b43caaca1fa25db430dab94b901cf277";
  const expectedFile = "onnxruntime/core/framework/sequential_executor.cc";
  if (adapter.source_commit !== expectedCommit || adapter.source_file !== expectedFile) {
    throw new Error("Runtime assignment ONNX Runtime profile adapter source provenance is not pinned to the supported parser basis.");
  }
  if (!normalized.source.profile_sha256) throw new Error("Runtime assignment source.profile_sha256 is required for adapted profile evidence.");
  const counts = {
    source_event_count: requiredNonNegativeInteger(adapter.source_event_count, "source.adapter.source_event_count"),
    kernel_event_count: requiredNonNegativeInteger(adapter.kernel_event_count, "source.adapter.kernel_event_count"),
    mapped_kernel_event_count: requiredNonNegativeInteger(adapter.mapped_kernel_event_count, "source.adapter.mapped_kernel_event_count"),
    unresolved_runtime_node_count: requiredNonNegativeInteger(adapter.unresolved_runtime_node_count, "source.adapter.unresolved_runtime_node_count"),
    conflict_count: requiredNonNegativeInteger(adapter.conflict_count, "source.adapter.conflict_count"),
  };
  const methods = {};
  for (const assignment of normalized.assignments) {
    const op = (analysis?.ops || []).find((candidate) => Number(candidate.index) === assignment.op_index);
    if (!assignment.mapping_method || assignment.runtime_node_index == null || !assignment.runtime_node_name || assignment.sample_count == null || assignment.duration_sum_us == null) {
      throw new Error(`Runtime assignment adapted op #${assignment.op_index} is missing mapping or duration aggregation evidence.`);
    }
    if (assignment.partition_id != null || assignment.kernel != null) {
      throw new Error(`Runtime assignment adapted op #${assignment.op_index} cannot infer a partition ID or kernel symbol from an ORT profile event.`);
    }
    if (assignment.mapping_method === "exact_graph_node_name_and_op_type") {
      if (!op?.graph_node_name || op.graph_node_name !== assignment.runtime_node_name || op.name !== assignment.op_name) {
        throw new Error(`Runtime assignment adapted op #${assignment.op_index} exact node-name mapping does not match the parsed ONNX graph.`);
      }
    } else if (assignment.mapping_method === "optimization_disabled_unnamed_node_index_and_op_type") {
      if (normalized.runtime.graph_optimization_level !== "disabled" || op?.graph_node_name || assignment.runtime_node_index !== assignment.op_index || op?.name !== assignment.op_name) {
        throw new Error(`Runtime assignment adapted op #${assignment.op_index} index mapping is not valid for an unnamed, optimization-disabled ONNX node.`);
      }
    } else {
      throw new Error(`Runtime assignment adapted op #${assignment.op_index} mapping_method is unsupported.`);
    }
    methods[assignment.mapping_method] = (methods[assignment.mapping_method] || 0) + 1;
  }
  if (counts.mapped_kernel_event_count !== normalized.assignments.reduce((sum, item) => sum + Number(item.sample_count || 0), 0)) {
    throw new Error("Runtime assignment adapted mapped event count does not equal assignment sample counts.");
  }
  if (!sameCountMap(adapter.mapping_method_counts, methods)) {
    throw new Error("Runtime assignment adapted mapping method counts do not match assignment rows.");
  }
  const expectedCoverage = normalized.assignment_count / Math.max(1, normalized.graph_op_count);
  if (!closeNumber(Number(adapter.mapping_coverage_ratio), expectedCoverage)) {
    throw new Error("Runtime assignment adapted mapping coverage does not match assignment rows.");
  }
  const uniqueSamples = new Set(normalized.assignments.map((item) => item.sample_count));
  const additive = normalized.runtime.execution_mode === "sequential" && uniqueSamples.size === 1;
  if ((normalized.source.duration_semantics === "per_original_op_exclusive") !== additive) {
    throw new Error("Runtime assignment adapted duration semantics do not match execution mode and sample-count evidence.");
  }
  const runtimeTensorEvidence = validateOrtRuntimeTensorObservations(adapter, normalized);
  const nativeCapture = adapter.native_capture == null ? null : validateOrtNativeCaptureAdapter(adapter.native_capture, normalized, counts, analysis);
  return {
    schema: adapter.schema,
    runtime_version_basis: optionalText(adapter.runtime_version_basis, "source.adapter.runtime_version_basis", 120),
    source_commit: expectedCommit,
    source_file: expectedFile,
    source_ref: optionalText(adapter.source_ref, "source.adapter.source_ref", 500),
    event_fields: Array.isArray(adapter.event_fields) ? adapter.event_fields.slice(0, 16).map((item) => requiredText(item, "source.adapter.event_fields[]", 80)) : [],
    ...counts,
    mapping_method_counts: methods,
    mapping_coverage_ratio: expectedCoverage,
    ...runtimeTensorEvidence,
    native_capture: nativeCapture,
    interpretation_boundary: requiredText(adapter.interpretation_boundary, "source.adapter.interpretation_boundary", 1000),
  };
}

function validateOrtRuntimeTensorObservations(adapter, normalized) {
  const supportsTensorShapes = adapter.schema === "deepbom.ort_profile_adapter.v2.2";
  if (!supportsTensorShapes) {
    return {
      runtime_tensor_observation_count: 0,
      runtime_tensor_observation_conflict_count: 0,
      runtime_tensor_observation_not_exposed_count: normalized.assignment_count,
      runtime_tensor_observations: [],
    };
  }
  const declaredObserved = requiredNonNegativeInteger(adapter.runtime_tensor_observation_count, "source.adapter.runtime_tensor_observation_count");
  const declaredConflicts = requiredNonNegativeInteger(adapter.runtime_tensor_observation_conflict_count, "source.adapter.runtime_tensor_observation_conflict_count");
  const declaredNotExposed = requiredNonNegativeInteger(adapter.runtime_tensor_observation_not_exposed_count, "source.adapter.runtime_tensor_observation_not_exposed_count");
  if (!Array.isArray(adapter.runtime_tensor_observations) || adapter.runtime_tensor_observations.length > normalized.assignment_count) {
    throw new Error("Runtime assignment source.adapter.runtime_tensor_observations is invalid.");
  }
  const assignments = new Map(normalized.assignments.map((row) => [row.op_index, row]));
  const seen = new Set();
  const rows = adapter.runtime_tensor_observations.map((row, index) => {
    const label = `source.adapter.runtime_tensor_observations[${index}]`;
    const opIndex = requiredNonNegativeInteger(row?.op_index, `${label}.op_index`);
    const assignment = assignments.get(opIndex);
    if (!assignment || seen.has(opIndex)) throw new Error(`Runtime assignment ${label} does not identify one unique mapped op.`);
    seen.add(opIndex);
    const status = requiredText(row.status, `${label}.status`, 80);
    if (!["consistent", "conflict_repeated_events"].includes(status)) throw new Error(`Runtime assignment ${label}.status is unsupported.`);
    if (requiredText(row.op_name, `${label}.op_name`, 200) !== assignment.op_name
      || requiredNonNegativeInteger(row.runtime_node_index, `${label}.runtime_node_index`) !== assignment.runtime_node_index
      || requiredText(row.runtime_node_name, `${label}.runtime_node_name`, 300) !== assignment.runtime_node_name
      || requiredPositiveInteger(row.sample_count, `${label}.sample_count`) !== assignment.sample_count) {
      throw new Error(`Runtime assignment ${label} identity or sample count does not match its assignment row.`);
    }
    const inputTypeShapes = validateOrtTypeShapeRows(row.input_type_shapes, `${label}.input_type_shapes`);
    const outputTypeShapes = validateOrtTypeShapeRows(row.output_type_shapes, `${label}.output_type_shapes`);
    const variantCount = requiredNonNegativeInteger(row.observed_contract_variant_count, `${label}.observed_contract_variant_count`);
    if ((status === "consistent" && variantCount !== 1)
      || (status === "conflict_repeated_events" && (variantCount < 2 || inputTypeShapes.length || outputTypeShapes.length))) {
      throw new Error(`Runtime assignment ${label} variant count or conflict payload is inconsistent.`);
    }
    return {
      op_index: opIndex,
      op_name: assignment.op_name,
      runtime_node_index: assignment.runtime_node_index,
      runtime_node_name: assignment.runtime_node_name,
      sample_count: assignment.sample_count,
      status,
      input_type_shapes: inputTypeShapes,
      output_type_shapes: outputTypeShapes,
      ...validateOrtByteCountPair(row, "activation_size", label),
      ...validateOrtByteCountPair(row, "parameter_size", label),
      ...validateOrtByteCountPair(row, "output_size", label),
      observed_contract_variant_count: variantCount,
    };
  });
  const observedCount = rows.filter((row) => row.status === "consistent").length;
  const conflictCount = rows.filter((row) => row.status === "conflict_repeated_events").length;
  if (declaredObserved !== observedCount || declaredConflicts !== conflictCount
    || declaredNotExposed + rows.length !== normalized.assignment_count) {
    throw new Error("Runtime assignment ORT tensor-observation counts do not conserve mapped assignment rows.");
  }
  return {
    runtime_tensor_observation_count: observedCount,
    runtime_tensor_observation_conflict_count: conflictCount,
    runtime_tensor_observation_not_exposed_count: declaredNotExposed,
    runtime_tensor_observations: rows,
  };
}

function validateOrtTypeShapeRows(values, field) {
  if (!Array.isArray(values) || values.length > 100_000) throw new Error(`Runtime assignment ${field} is invalid.`);
  return values.map((row, index) => {
    const slot = requiredNonNegativeInteger(row?.slot, `${field}[${index}].slot`);
    if (slot !== index) throw new Error(`Runtime assignment ${field} slots must be contiguous and ordered.`);
    const ortType = requiredText(row.ort_type, `${field}[${index}].ort_type`, 80);
    const dtype = row.dtype == null ? null : requiredText(row.dtype, `${field}[${index}].dtype`, 80);
    if (!Array.isArray(row.shape) || row.shape.length > 64
      || row.shape.some((dimension) => !Number.isSafeInteger(Number(dimension)) || Number(dimension) < 0)) {
      throw new Error(`Runtime assignment ${field}[${index}].shape is invalid.`);
    }
    return { slot, ort_type: ortType, dtype, shape: row.shape.map(Number) };
  });
}

function validateOrtByteCountPair(row, prefix, label) {
  const decimalField = `${prefix}_bytes_decimal`;
  const numberField = `${prefix}_bytes`;
  if (row[decimalField] == null) {
    if (row[numberField] != null) throw new Error(`Runtime assignment ${label}.${numberField} lacks its exact decimal value.`);
    return { [numberField]: null, [decimalField]: null };
  }
  const decimal = String(row[decimalField]);
  if (!/^\d+$/.test(decimal)) throw new Error(`Runtime assignment ${label}.${decimalField} is invalid.`);
  const exact = BigInt(decimal);
  const expectedNumber = exact <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(exact) : null;
  if ((row[numberField] == null ? null : Number(row[numberField])) !== expectedNumber) {
    throw new Error(`Runtime assignment ${label}.${numberField} does not match ${decimalField}.`);
  }
  return { [numberField]: expectedNumber, [decimalField]: exact.toString() };
}

function validateOrtNativeCaptureAdapter(capture, normalized, counts, analysis) {
  if (capture?.schema !== "deepbom.ort_native_profile.v1.4"
    || normalized.source.capture_binding_semantics !== "BROWSER_VERIFIED_NATIVE_CAPTURE_ENVELOPE_ARTIFACT_CONTENT_SET_AND_PROFILE_SHA256"
    || !normalized.source.capture_id || capture.capture_id !== normalized.source.capture_id
    || capture.capture_content_sha256 == null || requiredSha(capture.capture_content_sha256, "source.adapter.native_capture.capture_content_sha256") !== capture.capture_content_sha256
    || capture.artifact_binding !== "BROWSER_VERIFIED_ACTIVE_ONNX_AND_EXTERNAL_DATA_CONTENT_SET_SHA256"
    || capture.profile_binding !== "BROWSER_VERIFIED_EMBEDDED_PROFILE_SHA256"
    || capture.envelope_binding !== "BROWSER_VERIFIED_CANONICAL_CONTENT_SHA256"
    || capture.runtime_identity_semantics !== "CAPTURE_COLLECTOR_OBSERVED_BROWSER_NOT_REHASHED"
    || capture.assignment_evidence_class !== "OBSERVED_RUNTIME") {
    throw new Error("Runtime assignment native ORT capture binding semantics are invalid.");
  }
  if (capture.artifact?.sha256 !== normalized.artifact_sha256
    || capture.profile?.sha256 !== normalized.source.profile_sha256
    || capture.profile?.source_event_count !== counts.source_event_count
    || capture.profile?.kernel_event_count !== counts.kernel_event_count) {
    throw new Error("Runtime assignment native ORT artifact/profile identity is inconsistent.");
  }
  validateOrtNativeArtifactContentSet(capture.artifact, analysis);
  const runtime = capture.runtime || {};
  const expectedPackageIntegrity = "sha512-OHl6PiOEOqxaLHL0N9eFrbzS7IGmu3BtJNH3RTEnRAheCIkfc3gjcjl4sGcjp9C22ZC9YTquDOxSdT/stBQ6BQ==";
  if (runtime.name !== "ONNX Runtime Node.js" || runtime.version !== "1.26.0" || runtime.package_name !== "onnxruntime-node"
    || !["NPM_PACKAGE_LOCK_ATTESTED", "SOURCE_BUILD_ATTESTED"].includes(runtime.distribution_identity)
    || (runtime.distribution_identity === "NPM_PACKAGE_LOCK_ATTESTED"
      ? runtime.package_integrity !== expectedPackageIntegrity || runtime.build_attestation != null
      : runtime.package_integrity != null || validateOrtBuildAttestation(runtime.build_attestation).attestation_sha256 !== runtime.build_attestation.attestation_sha256)
    || runtime.source_commit !== "8c546c37b43caaca1fa25db430dab94b901cf277"
    || runtime.node_napi !== "napi-v6" || runtime.primary_binary_sha256 !== normalized.runtime.binary_sha256
    || !requiredSha(runtime.package_manifest_sha256, "source.adapter.native_capture.runtime.package_manifest_sha256")
    || !requiredSha(runtime.binary_inventory_sha256, "source.adapter.native_capture.runtime.binary_inventory_sha256")
    || !Array.isArray(runtime.binary_inventory) || !runtime.binary_inventory.length) {
    throw new Error("Runtime assignment native ORT package/binary identity is invalid.");
  }
  for (const file of runtime.binary_inventory) {
    if (!safeRelativeEvidencePath(file?.path) || !Number.isSafeInteger(file?.byte_length) || file.byte_length < 1
      || !requiredSha(file?.sha256, "source.adapter.native_capture.runtime.binary_inventory[].sha256")) {
      throw new Error("Runtime assignment native ORT binary inventory row is invalid.");
    }
  }
  if (!runtime.binary_inventory.some((file) => file.path === runtime.primary_binary_path && file.sha256 === runtime.primary_binary_sha256)) {
    throw new Error("Runtime assignment native ORT primary binary is not present in the inventory.");
  }
  const backends = runtime.supported_backends;
  const requestedProviders = runtime.requested_execution_providers;
  if (!Array.isArray(backends) || !backends.length
    || backends.some((backend) => !backend?.name || typeof backend.bundled !== "boolean")
    || new Set(backends.map((backend) => backend.name)).size !== backends.length
    || JSON.stringify([...backends].sort((left, right) => left.name.localeCompare(right.name))) !== JSON.stringify(backends)
    || !requiredSha(runtime.supported_backends_sha256, "source.adapter.native_capture.runtime.supported_backends_sha256")
    || runtime.provider_inventory_status !== "OBSERVED_FROM_ORT_LIST_SUPPORTED_BACKENDS"
    || !["NOT_EXPOSED_BY_ONNXRUNTIME_NODE_API_NOT_INFERRED", "IMPORTED_CONFIG_NOT_BINARY_ATTESTED", "BUILD_INPUT_BINARY_ATTESTED"].includes(runtime.reduced_operator_inventory_status)
    || !Array.isArray(requestedProviders) || !requestedProviders.length || new Set(requestedProviders).size !== requestedProviders.length
    || requestedProviders.some((name) => !backends.some((backend) => backend.name === name))) {
    throw new Error("Runtime assignment native ORT selected-build provider inventory is invalid.");
  }
  validateOrtSelectedBuildProviderBinding(capture.selected_build_provider_binding, analysis, runtime);
  const expectedOptimization = capture.profile_role === "identity" ? "disabled" : capture.profile_role === "production" ? "all" : null;
  if (!expectedOptimization || normalized.runtime.graph_optimization_level !== expectedOptimization
    || normalized.runtime.execution_mode !== "sequential"
    || capture.invocation?.graph_optimization_level !== expectedOptimization || capture.invocation?.execution_mode !== "sequential") {
    throw new Error("Runtime assignment native ORT profile role/session options are invalid.");
  }
  if (!Array.isArray(capture.invocation.inputs) || !capture.invocation.inputs.length
    || capture.invocation.inputs.some((input) => !input?.name || !input.type || !Array.isArray(input.shape)
      || input.shape.some((dim) => !Number.isSafeInteger(dim) || dim < 1) || !requiredSha(input.data_sha256, "source.adapter.native_capture.invocation.inputs[].data_sha256"))) {
    throw new Error("Runtime assignment native ORT input inventory is invalid.");
  }
  validateOrtNativeOutputComparison(capture.paired_profile_output_comparison);
  validateOrtNativeRuntimeGraph(capture.paired_profile_runtime_graph);
  if (!Array.isArray(capture.output_observations) || !capture.output_observations.length
    || capture.output_observations.some((output) => !output?.name || !output.type || !Array.isArray(output.dims)
      || !requiredSha(output.first_run_sha256, "source.adapter.native_capture.output_observations[].first_run_sha256")
      || !requiredSha(output.last_run_sha256, "source.adapter.native_capture.output_observations[].last_run_sha256"))) {
    throw new Error("Runtime assignment native ORT output observations are invalid.");
  }
  return JSON.parse(JSON.stringify(capture));
}

function validateOrtNativeArtifactContentSet(artifact, analysis) {
  const active = analysis?.onnx_external_data || {};
  const tensorCount = Number(active.tensor_count || 0);
  if (!artifact || artifact.name !== analysis?.filename || artifact.byte_length !== analysis?.file_size
    || !requiredSha(artifact.content_set_sha256, "source.adapter.native_capture.artifact.content_set_sha256")
    || !String(analysis?.format || "").toLowerCase().includes("onnx")
    || !Number.isSafeInteger(tensorCount) || tensorCount < 0 || tensorCount !== (active.tensors || []).length
    || (tensorCount > 0 && (active.status !== "verified_payloads" || Number(active.verified_payload_count || 0) !== tensorCount))) {
    throw new Error("Runtime assignment native ORT artifact content-set identity is invalid for the active analysis.");
  }
  const files = (active.supplied_files || []).filter((file) => file.used === true).map((file) => ({
    path: safeRelativeEvidencePath(file.path),
    byte_length: file.byte_length,
    sha256: String(file.sha256 || "").toLowerCase(),
    sha1: String(file.sha1 || "").toLowerCase(),
  })).sort((left, right) => compareText(left.path, right.path));
  if (new Set(files.map((file) => file.path)).size !== files.length || files.some((file) => !file.path
    || !Number.isSafeInteger(file.byte_length) || file.byte_length < 0
    || !SHA256_PATTERN.test(file.sha256) || !/^[a-f0-9]{40}$/.test(file.sha1))) {
    throw new Error("Runtime assignment active ONNX external-data file ledger is invalid.");
  }
  const tensorRanges = (active.tensors || []).map((row) => ({
    scope: row.scope,
    tensor_role: row.tensor_role,
    tensor_name: row.tensor_name,
    location: safeRelativeEvidencePath(row.normalized_location || row.location),
    offset: row.offset,
    length: row.length,
    payload_bytes: row.payload_bytes,
    checksum: String(row.checksum || "").toLowerCase(),
    sidecar_sha256: row.sidecar_sha256,
  })).sort(compareExternalTensorRange);
  const expectedExternal = {
    schema: "deepbom.onnx_external_artifact_set.v1.2",
    status: tensorCount ? "verified_payloads" : "assessed_absent",
    tensor_count: tensorCount,
    verified_payload_bytes: Number(active.verified_payload_bytes || 0),
    tensor_ranges: tensorRanges,
    files,
    ledger_sha256: artifact.external_data?.ledger_sha256,
  };
  if (!requiredSha(expectedExternal.ledger_sha256, "source.adapter.native_capture.artifact.external_data.ledger_sha256")
    || JSON.stringify(artifact.external_data) !== JSON.stringify(expectedExternal)) {
    throw new Error("Runtime assignment native ORT external-data ledger does not match the active ONNX audit.");
  }
}

function validateOrtNativeRuntimeGraph(graph) {
  if (!graph || !["deepbom.ort_paired_runtime_graph.v1", "deepbom.ort_paired_runtime_graph.v1.1"].includes(graph.schema)
    || graph.production_original_graph_mapping_status !== "NOT_INFERRED_TRANSFORMED_RUNTIME_NODE_IDENTITY"
    || !Array.isArray(graph.profiles) || graph.profiles.length !== 2) {
    throw new Error("Runtime assignment native ORT paired runtime graph is invalid.");
  }
  const roles = new Set();
  for (const profile of graph.profiles) {
    if (!["identity", "production"].includes(profile.role) || roles.has(profile.role)
      || !requiredSha(profile.profile_sha256, "source.adapter.native_capture.runtime_graph.profile_sha256")
      || !Number.isSafeInteger(profile.kernel_event_count) || profile.kernel_event_count < 1
      || !Number.isSafeInteger(profile.runtime_node_count) || profile.runtime_node_count < 1
      || !Array.isArray(profile.nodes) || profile.nodes.length !== profile.runtime_node_count
      || profile.nodes.reduce((sum, node) => sum + Number(node.sample_count || 0), 0) !== profile.kernel_event_count) {
      throw new Error("Runtime assignment native ORT paired runtime graph profile is invalid.");
    }
    roles.add(profile.role);
    if (graph.schema === "deepbom.ort_paired_runtime_graph.v1.1"
      && (!Number.isSafeInteger(profile.invocation_run_count) || profile.invocation_run_count < 1)) {
      throw new Error("Runtime assignment native ORT paired runtime graph invocation count is invalid.");
    }
    for (const node of profile.nodes) {
      if (!node?.runtime_node_name || !Number.isSafeInteger(node.runtime_node_index) || node.runtime_node_index < 0 || !node.op_name || !node.provider
        || !Number.isSafeInteger(node.sample_count) || node.sample_count < 1
        || !Number.isFinite(node.duration_sum_us) || node.duration_sum_us < 0
        || !Number.isFinite(node.duration_mean_us) || node.duration_mean_us < 0
        || (graph.schema === "deepbom.ort_paired_runtime_graph.v1.1"
          && node.output_size_bytes_decimal != null && !/^\d+$/.test(node.output_size_bytes_decimal))) {
        throw new Error("Runtime assignment native ORT paired runtime graph node is invalid.");
      }
    }
  }
}

function validateOrtNativeOutputComparison(comparison) {
  if (!comparison || comparison.schema !== "deepbom.ort_profile_output_comparison.v1"
    || comparison.reference_profile_role !== "identity" || comparison.candidate_profile_role !== "production"
    || !["assessed", "partially_assessed"].includes(comparison.status)
    || typeof comparison.all_outputs_bitwise_equal !== "boolean" || !Array.isArray(comparison.outputs) || !comparison.outputs.length) {
    throw new Error("Runtime assignment native ORT paired-profile output comparison is invalid.");
  }
  for (const output of comparison.outputs) {
    if (!output?.name || !output.type || !Array.isArray(output.dims) || typeof output.bitwise_equal !== "boolean"
      || !requiredSha(output.identity_sha256, "source.adapter.native_capture.output_comparison.identity_sha256")
      || !requiredSha(output.production_sha256, "source.adapter.native_capture.output_comparison.production_sha256")) {
      throw new Error("Runtime assignment native ORT paired-profile output row is invalid.");
    }
    if (output.numeric_comparison_status === "assessed") {
      for (const field of ["max_abs_error", "mean_abs_error", "rms_error", "relative_l2_error"]) {
        if (!Number.isFinite(output[field]) || output[field] < 0) throw new Error(`Runtime assignment native ORT ${field} is invalid.`);
      }
      if (output.cosine_distance != null && !Number.isFinite(output.cosine_distance)) throw new Error("Runtime assignment native ORT cosine distance is invalid.");
    } else if (output.numeric_comparison_status !== "not_assessed_non_numeric_or_nonfinite") {
      throw new Error("Runtime assignment native ORT numeric comparison status is invalid.");
    }
  }
}

function safeRelativeEvidencePath(value) {
  const text = String(value || "").replaceAll("\\", "/").replace(/^(?:\.\/)+/, "");
  return text && !text.includes("\0") && !/^[a-z][a-z0-9+.-]*:/i.test(text) && !text.startsWith("/")
    && text.split("/").every((part) => part && part !== "." && part !== "..") ? text : null;
}

function compareExternalTensorRange(left, right) {
  return compareText(
    [left.scope, left.tensor_role, left.tensor_name, left.location, left.offset, left.length, left.payload_bytes, left.checksum, left.sidecar_sha256].join("\0"),
    [right.scope, right.tensor_role, right.tensor_name, right.location, right.offset, right.length, right.payload_bytes, right.checksum, right.sidecar_sha256].join("\0"),
  );
}

function compareText(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function validateTfliteRuntimeInfoAdapter(adapter, normalized, analysis) {
  const timed = adapter.schema === "deepbom.tflite_runtime_info_adapter.v2";
  if (!timed && adapter.schema !== "deepbom.tflite_runtime_info_adapter.v1") throw new Error("Runtime assignment TFLite runtime-info adapter schema is unsupported.");
  if (String(analysis?.format || "").toLowerCase() !== "tflite") throw new Error("TFLite runtime-info adapter evidence requires a TFLite artifact.");
  if (!normalized.source.profile_sha256) throw new Error("Runtime assignment source.profile_sha256 is required for adapted runtime evidence.");
  if (timed) {
    if (normalized.source.kind !== "tflite_model_runtime_info_and_benchmark_profile_proto_adapter"
      || normalized.source.duration_semantics !== "per_execution_plan_node_exclusive"
      || normalized.source.duration_statistic !== "execution_node_mean_per_run_from_one_event_per_primary_run; ancillary_sum_divided_by_common_primary_run_count"
      || !normalized.source.capture_id
      || normalized.source.capture_binding_semantics !== "DECLARED_BENCHMARK_INVOCATION_IDENTIFIER") {
      throw new Error("Runtime assignment combined TFLite timing source semantics are invalid.");
    }
  } else if (normalized.source.kind !== "tflite_model_runtime_info_proto_adapter" || normalized.source.duration_semantics !== "not_collected") {
    throw new Error("Runtime assignment ModelRuntimeDetails-only source semantics are invalid.");
  }
  const expectedCommit = "tensorflow/tensorflow@87bbf65b8d23d3f06912b1b2183587e1884bc45c";
  const expectedFile = "tensorflow/lite/profiling/model_runtime_info.cc";
  const expectedSourceSha = "9a6edf838fe149c54efe0700bcdc2faf58dd5343f1370538e91bf5ed8a0e11b6";
  const expectedProtoFile = "tensorflow/lite/profiling/proto/model_runtime_info.proto";
  const expectedProtoSha = "7829c3163339ce7dea01091a5154a06cb302f35573ad51d3af06a2a09b95a8fb";
  const expectedDelegationFile = "tensorflow/lite/optional_debug_tools.cc";
  const expectedDelegationSha = "607010c8f7aba721bd5d96f45cb08c55226a800e55946da708368d09d4545260";
  const expectedExportDriverFile = "tensorflow/lite/tools/benchmark/benchmark_tflite_model.cc";
  const expectedExportDriverSha = "e26a3e4e26300442b47d27e1b515ccbc85b34425625d60f637faa28973f7a8f7";
  if (adapter.source_commit !== expectedCommit || adapter.source_file !== expectedFile || adapter.source_sha256 !== expectedSourceSha
    || adapter.proto_file !== expectedProtoFile || adapter.proto_sha256 !== expectedProtoSha
    || adapter.delegation_metadata_file !== expectedDelegationFile || adapter.delegation_metadata_sha256 !== expectedDelegationSha
    || adapter.export_driver_file !== expectedExportDriverFile || adapter.export_driver_sha256 !== expectedExportDriverSha) {
    throw new Error("Runtime assignment TFLite runtime-info provenance is not pinned to the supported parser and proto basis.");
  }
  const counts = {
    source_byte_length: requiredPositiveInteger(adapter.source_byte_length, "source.adapter.source_byte_length"),
    subgraph_id: requiredNonNegativeInteger(adapter.subgraph_id, "source.adapter.subgraph_id"),
    original_node_count: requiredNonNegativeInteger(adapter.original_node_count, "source.adapter.original_node_count"),
    delegate_node_count: requiredNonNegativeInteger(adapter.delegate_node_count, "source.adapter.delegate_node_count"),
    execution_plan_node_count: requiredNonNegativeInteger(adapter.execution_plan_node_count, "source.adapter.execution_plan_node_count"),
    delegated_op_count: requiredNonNegativeInteger(adapter.delegated_op_count, "source.adapter.delegated_op_count"),
    nondelegated_op_count: requiredNonNegativeInteger(adapter.nondelegated_op_count, "source.adapter.nondelegated_op_count"),
  };
  if (counts.subgraph_id !== 0 || counts.original_node_count !== normalized.graph_op_count || normalized.assignment_count !== normalized.graph_op_count) {
    throw new Error("Runtime assignment TFLite runtime-info evidence must exactly cover primary-subgraph original ops.");
  }
  if (counts.delegated_op_count + counts.nondelegated_op_count !== normalized.graph_op_count
    || counts.execution_plan_node_count !== counts.delegate_node_count + counts.nondelegated_op_count) {
    throw new Error("Runtime assignment TFLite runtime-info node and execution-plan counts are inconsistent.");
  }
  const methods = {};
  const delegatedByPartition = new Map();
  let delegatedCount = 0;
  for (const assignment of normalized.assignments) {
    if ((!timed && (assignment.duration_us != null || assignment.duration_sum_us != null || assignment.sample_count != null)) || assignment.kernel != null) {
      throw new Error(`Runtime assignment TFLite runtime-info op #${assignment.op_index} claims timing or microkernel evidence absent from ModelRuntimeDetails.`);
    }
    if (timed && assignment.delegated === true && (assignment.duration_us != null || assignment.duration_sum_us != null || assignment.sample_count != null)) {
      throw new Error(`Runtime assignment TFLite delegated op #${assignment.op_index} duplicates partition or delegate-internal timing.`);
    }
    if (assignment.delegated) {
      delegatedCount += 1;
      if (assignment.mapping_method !== "runtime_info_original_node_id_and_symmetric_delegate_map"
        || assignment.runtime_node_index == null || !assignment.runtime_node_name
        || assignment.partition_id !== `subgraph:0/delegate_node:${assignment.runtime_node_index}`) {
        throw new Error(`Runtime assignment TFLite delegated op #${assignment.op_index} has invalid delegate mapping evidence.`);
      }
      if (!delegatedByPartition.has(assignment.partition_id)) delegatedByPartition.set(assignment.partition_id, []);
      delegatedByPartition.get(assignment.partition_id).push(assignment);
    } else if (assignment.delegated === false) {
      if (assignment.mapping_method !== "runtime_info_original_node_id_execution_plan"
        || assignment.partition_id != null || assignment.provider !== "TFLite non-delegated kernel"
        || assignment.runtime_node_index !== assignment.op_index || assignment.runtime_node_name !== assignment.op_name) {
        throw new Error(`Runtime assignment TFLite non-delegated op #${assignment.op_index} has invalid execution-plan mapping evidence.`);
      }
    } else {
      throw new Error(`Runtime assignment TFLite runtime-info op #${assignment.op_index} must have an observed delegated boolean.`);
    }
    methods[assignment.mapping_method] = (methods[assignment.mapping_method] || 0) + 1;
  }
  if (delegatedCount !== counts.delegated_op_count || normalized.graph_op_count - delegatedCount !== counts.nondelegated_op_count) {
    throw new Error("Runtime assignment TFLite runtime-info delegated-op counts do not match assignment rows.");
  }
  if (!sameCountMap(adapter.mapping_method_counts, methods) || !closeNumber(Number(adapter.mapping_coverage_ratio), 1)) {
    throw new Error("Runtime assignment TFLite runtime-info mapping coverage does not match assignment rows.");
  }
  if (!Array.isArray(adapter.partitions) || adapter.partitions.length !== counts.delegate_node_count) {
    throw new Error("Runtime assignment TFLite runtime-info partition inventory is incomplete.");
  }
  const partitionIds = new Set();
  const partitions = adapter.partitions.map((partition) => {
    const delegateNodeId = requiredNonNegativeInteger(partition.delegate_node_id, "source.adapter.partitions[].delegate_node_id");
    const partitionId = requiredText(partition.partition_id, "source.adapter.partitions[].partition_id", 100);
    const delegateName = requiredText(partition.delegate_name, "source.adapter.partitions[].delegate_name", 200);
    const runtimeNodeName = requiredText(partition.runtime_node_name, "source.adapter.partitions[].runtime_node_name", 300);
    if (partitionId !== `subgraph:0/delegate_node:${delegateNodeId}` || partitionIds.has(partitionId)) {
      throw new Error("Runtime assignment TFLite runtime-info partition IDs are invalid or duplicated.");
    }
    partitionIds.add(partitionId);
    const replaced = Array.isArray(partition.replaced_op_ids)
      ? partition.replaced_op_ids.map((value) => requiredNonNegativeInteger(value, "source.adapter.partitions[].replaced_op_ids[]")).sort((a, b) => a - b)
      : [];
    if (!replaced.length || new Set(replaced).size !== replaced.length) throw new Error(`Runtime assignment TFLite partition ${partitionId} has invalid replaced op IDs.`);
    const rows = (delegatedByPartition.get(partitionId) || []).sort((left, right) => left.op_index - right.op_index);
    if (rows.length !== replaced.length || rows.some((row, index) => row.op_index !== replaced[index]
      || row.provider !== delegateName || row.runtime_node_index !== delegateNodeId || row.runtime_node_name !== runtimeNodeName)) {
      throw new Error(`Runtime assignment TFLite partition ${partitionId} does not match delegated assignment rows.`);
    }
    return { partition_id: partitionId, delegate_node_id: delegateNodeId, delegate_name: delegateName, runtime_node_name: runtimeNodeName, replaced_op_ids: replaced };
  });
  if (delegatedByPartition.size !== partitions.length) throw new Error("Runtime assignment TFLite delegated rows reference an unlisted partition.");
  const topology = adapter.topology_binding;
  if (topology?.method !== "exact_original_op_id_name_and_input_output_tensor_ids"
    || Number(topology.matched_original_op_count) !== normalized.graph_op_count
    || Number(topology.graph_op_count) !== normalized.graph_op_count
    || topology.input_output_tensor_id_arrays_matched !== true
    || topology.source_artifact_sha256_embedded !== false
    || adapter.artifact_binding !== "active_artifact_exact_original_op_topology"
    || adapter.source_artifact_sha256_embedded !== false
    || adapter.runtime_identity_semantics !== "DECLARED"
    || adapter.assignment_evidence_class !== "OBSERVED_RUNTIME") {
    throw new Error("Runtime assignment TFLite runtime-info artifact binding or evidence semantics are invalid.");
  }
  const timingProfile = timed ? validateTfliteTimingProfile(adapter.timing_profile, normalized, partitions, counts) : null;
  return {
    schema: adapter.schema,
    runtime_version_basis: optionalText(adapter.runtime_version_basis, "source.adapter.runtime_version_basis", 200),
    source_commit: expectedCommit,
    source_file: expectedFile,
    source_ref: optionalText(adapter.source_ref, "source.adapter.source_ref", 500),
    source_sha256: expectedSourceSha,
    proto_file: expectedProtoFile,
    proto_ref: optionalText(adapter.proto_ref, "source.adapter.proto_ref", 500),
    proto_sha256: expectedProtoSha,
    delegation_metadata_file: expectedDelegationFile,
    delegation_metadata_ref: optionalText(adapter.delegation_metadata_ref, "source.adapter.delegation_metadata_ref", 500),
    delegation_metadata_sha256: expectedDelegationSha,
    export_driver_file: expectedExportDriverFile,
    export_driver_ref: optionalText(adapter.export_driver_ref, "source.adapter.export_driver_ref", 500),
    export_driver_sha256: expectedExportDriverSha,
    export_flag: optionalText(adapter.export_flag, "source.adapter.export_flag", 300),
    source_model_name: optionalText(adapter.source_model_name, "source.adapter.source_model_name", 500),
    ...counts,
    mapping_method_counts: methods,
    mapping_coverage_ratio: 1,
    topology_binding: {
      method: topology.method,
      matched_original_op_count: normalized.graph_op_count,
      graph_op_count: normalized.graph_op_count,
      input_output_tensor_id_arrays_matched: true,
      source_artifact_sha256_embedded: false,
    },
    partitions,
    artifact_binding: adapter.artifact_binding,
    source_artifact_sha256_embedded: false,
    runtime_identity_semantics: "DECLARED",
    assignment_evidence_class: "OBSERVED_RUNTIME",
    ...(timingProfile ? { timing_profile: timingProfile } : {}),
    interpretation_boundary: requiredText(adapter.interpretation_boundary, "source.adapter.interpretation_boundary", 1000),
  };
}

function validateTfliteTimingProfile(source, normalized, partitions, counts) {
  if (source?.schema !== "deepbom.tflite_benchmark_profile_adapter.v1") throw new Error("Runtime assignment TFLite timing adapter schema is unsupported.");
  for (const [field, expected] of Object.entries(TFLITE_PROFILE_INFO_SOURCE)) {
    if (source[field] !== expected) throw new Error(`Runtime assignment TFLite timing source provenance drifted at ${field}.`);
  }
  const profileSha256 = requiredSha(source.profile_sha256, "source.adapter.timing_profile.profile_sha256");
  const sourceByteLength = requiredPositiveInteger(source.source_byte_length, "source.adapter.timing_profile.source_byte_length");
  const collectedAt = validatedTimestamp(source.collected_at, true);
  const captureId = requiredText(source.capture_id, "source.adapter.timing_profile.capture_id", 160);
  if (captureId !== normalized.source.capture_id || source.capture_binding_semantics !== "DECLARED_SAME_BENCHMARK_INVOCATION") {
    throw new Error("Runtime assignment TFLite timing capture binding does not match the runtime plan.");
  }
  const executionNodes = validateTimingRows(source.execution_nodes, "execution_nodes", true);
  const internalEvents = validateTimingRows(source.delegate_internal_events, "delegate_internal_events", false);
  const otherEvents = validateTimingRows(source.other_primary_events, "other_primary_events", false);
  if (internalEvents.some((row) => !["primary_subgraph_delegate_profiled_event", "delegate_internal_section_event"].includes(row.node_kind) || !row.name.startsWith("Delegate/"))
    || otherEvents.some((row) => row.node_kind !== "unmapped_primary_subgraph_event")
    || executionNodes.some((row) => row.name.startsWith("Delegate/"))) {
    throw new Error("Runtime assignment TFLite timing event groups or delegate-internal names are inconsistent.");
  }
  const primaryOrders = [...executionNodes, ...internalEvents.filter((row) => row.node_kind === "primary_subgraph_delegate_profiled_event"), ...otherEvents].map((row) => row.run_order);
  if (new Set(primaryOrders).size !== primaryOrders.length) throw new Error("Runtime assignment TFLite primary-profile run orders are not unique.");
  for (const row of executionNodes) {
    const expectedRunCount = row.formatter_times_called_integer_average === 1 ? row.event_sample_count : null;
    const expectedStatus = expectedRunCount == null
      ? "not_derivable_execution_node_calls_per_run_not_one"
      : "derived_primary_execution_node_one_event_per_run";
    if (row.run_count !== expectedRunCount || row.run_count_derivation_status !== expectedStatus) {
      throw new Error(`Runtime assignment TFLite execution-node run count is inconsistent for ${row.name}.`);
    }
  }
  const commonRunCount = commonTimingRunCount(executionNodes);
  for (const row of [...internalEvents, ...otherEvents]) {
    const expectedStatus = commonRunCount == null
      ? "not_derivable_no_common_primary_execution_run_count"
      : "derived_from_common_primary_execution_run_count";
    if (row.run_count !== commonRunCount || row.run_count_derivation_status !== expectedStatus
      || (commonRunCount != null && Math.floor(row.event_sample_count / commonRunCount) !== row.formatter_times_called_integer_average)) {
      throw new Error(`Runtime assignment TFLite ancillary-event run count is inconsistent for ${row.name}.`);
    }
  }
  const executionById = new Map();
  const nondelegatedByNode = new Map(normalized.assignments.filter((row) => row.delegated === false).map((row) => [row.runtime_node_index, row]));
  const partitionByNode = new Map(partitions.map((row) => [row.delegate_node_id, row]));
  let originalCount = 0;
  let partitionCount = 0;
  for (const row of executionNodes) {
    if (executionById.has(row.runtime_node_index)) throw new Error(`Runtime assignment TFLite timing duplicates execution node ${row.runtime_node_index}.`);
    executionById.set(row.runtime_node_index, row);
    if (row.node_kind === "original_op") {
      const assignment = nondelegatedByNode.get(row.runtime_node_index);
      if (!assignment || row.op_index !== assignment.op_index || row.partition_id != null || row.provider !== assignment.provider
        || (row.node_type !== assignment.op_name && !row.node_type.startsWith(`${assignment.op_name}/`))) {
        throw new Error(`Runtime assignment TFLite timing original node ${row.runtime_node_index} does not match the execution plan.`);
      }
      originalCount += 1;
    } else if (row.node_kind === "delegate_partition") {
      const partition = partitionByNode.get(row.runtime_node_index);
      if (!partition || row.op_index != null || row.partition_id !== partition.partition_id || row.provider !== partition.delegate_name
        || (row.node_type !== partition.runtime_node_name && !row.node_type.startsWith(`${partition.runtime_node_name}/`))) {
        throw new Error(`Runtime assignment TFLite timing delegate node ${row.runtime_node_index} does not match the partition inventory.`);
      }
      partitionCount += 1;
    } else {
      throw new Error(`Runtime assignment TFLite timing execution node ${row.runtime_node_index} has invalid node_kind.`);
    }
  }
  for (const assignment of normalized.assignments.filter((row) => row.delegated === false)) {
    const timing = executionById.get(assignment.runtime_node_index);
    if (!timing) {
      if (assignment.duration_us != null || assignment.duration_sum_us != null || assignment.sample_count != null) {
        throw new Error(`Runtime assignment TFLite op #${assignment.op_index} has timing without a mapped execution-node row.`);
      }
    } else if (timing.run_count == null) {
      if (assignment.duration_us != null || assignment.duration_sum_us != null || assignment.sample_count != null) {
        throw new Error(`Runtime assignment TFLite op #${assignment.op_index} has per-run timing when run count is not derivable.`);
      }
    } else if (!closeNumber(assignment.duration_us, timing.mean_per_run_us)
      || !closeNumber(assignment.duration_sum_us, timing.sum_us)
      || assignment.sample_count !== timing.run_count) {
      throw new Error(`Runtime assignment TFLite op #${assignment.op_index} timing does not match its execution-node row.`);
    }
  }
  const executionPlanCount = requiredNonNegativeInteger(source.execution_plan_node_count, "source.adapter.timing_profile.execution_plan_node_count");
  const mappedCount = requiredNonNegativeInteger(source.mapped_execution_node_count, "source.adapter.timing_profile.mapped_execution_node_count");
  if (executionPlanCount !== counts.execution_plan_node_count || mappedCount !== executionNodes.length
    || requiredNonNegativeInteger(source.original_execution_node_timing_count, "source.adapter.timing_profile.original_execution_node_timing_count") !== originalCount
    || requiredNonNegativeInteger(source.delegate_partition_timing_count, "source.adapter.timing_profile.delegate_partition_timing_count") !== partitionCount
    || requiredNonNegativeInteger(source.delegate_internal_event_count, "source.adapter.timing_profile.delegate_internal_event_count") !== internalEvents.length
    || requiredNonNegativeInteger(source.other_primary_event_count, "source.adapter.timing_profile.other_primary_event_count") !== otherEvents.length) {
    throw new Error("Runtime assignment TFLite timing inventory counts are inconsistent.");
  }
  const coverage = mappedCount / Math.max(1, executionPlanCount);
  if (!closeNumber(Number(source.execution_plan_coverage_ratio), coverage)) throw new Error("Runtime assignment TFLite timing coverage ratio is inconsistent.");
  const complete = mappedCount === executionPlanCount && commonRunCount != null;
  if ((source.common_run_count == null ? null : requiredPositiveInteger(source.common_run_count, "source.adapter.timing_profile.common_run_count")) !== commonRunCount) {
    throw new Error("Runtime assignment TFLite timing common run count is inconsistent.");
  }
  const expected = {
    execution_node_total_us: complete ? timingSum(executionNodes) : null,
    mapped_execution_node_subtotal_us: commonRunCount == null ? null : timingSum(executionNodes),
    cpu_execution_node_subtotal_us: commonTimingRunCount(executionNodes.filter((row) => row.node_kind === "original_op")) == null ? null : timingSum(executionNodes.filter((row) => row.node_kind === "original_op")),
    delegate_partition_subtotal_us: commonTimingRunCount(executionNodes.filter((row) => row.node_kind === "delegate_partition")) == null ? null : timingSum(executionNodes.filter((row) => row.node_kind === "delegate_partition")),
    primary_delegate_profiled_subtotal_us: commonTimingRunCount(internalEvents.filter((row) => row.node_kind === "primary_subgraph_delegate_profiled_event")) == null ? null : timingSum(internalEvents.filter((row) => row.node_kind === "primary_subgraph_delegate_profiled_event")),
    delegate_internal_section_subtotal_us: commonTimingRunCount(internalEvents.filter((row) => row.node_kind === "delegate_internal_section_event")) == null ? null : timingSum(internalEvents.filter((row) => row.node_kind === "delegate_internal_section_event")),
    delegate_internal_profiled_subtotal_us: commonTimingRunCount(internalEvents) == null ? null : timingSum(internalEvents),
  };
  for (const [field, value] of Object.entries(expected)) {
    if (!sameNullableNumber(source[field], value)) throw new Error(`Runtime assignment TFLite timing ${field} is inconsistent.`);
  }
  const totalStatus = complete ? "assessed_complete_execution_plan" : mappedCount ? "partial_execution_plan_coverage" : "not_assessed_no_execution_node_rows";
  if (source.total_status !== totalStatus) throw new Error("Runtime assignment TFLite timing total status is inconsistent.");
  const runtimeSubgraphCount = requiredPositiveInteger(source.runtime_subgraph_count, "source.adapter.timing_profile.runtime_subgraph_count");
  const nonprimarySubgraphCount = requiredNonNegativeInteger(source.unassessed_nonprimary_subgraph_count, "source.adapter.timing_profile.unassessed_nonprimary_subgraph_count");
  if (runtimeSubgraphCount !== nonprimarySubgraphCount + 1) throw new Error("Runtime assignment TFLite timing subgraph counts are inconsistent.");
  return {
    schema: source.schema,
    ...TFLITE_PROFILE_INFO_SOURCE,
    profile_sha256: profileSha256,
    source_byte_length: sourceByteLength,
    source_model_name: optionalText(source.source_model_name, "source.adapter.timing_profile.source_model_name", 500),
    collected_at: collectedAt,
    capture_id: captureId,
    capture_binding_semantics: source.capture_binding_semantics,
    primary_subgraph_index: requiredZero(source.primary_subgraph_index, "source.adapter.timing_profile.primary_subgraph_index"),
    runtime_subgraph_count: runtimeSubgraphCount,
    unassessed_nonprimary_subgraph_count: nonprimarySubgraphCount,
    execution_plan_node_count: executionPlanCount,
    mapped_execution_node_count: mappedCount,
    execution_plan_coverage_ratio: coverage,
    original_execution_node_timing_count: originalCount,
    delegate_partition_timing_count: partitionCount,
    delegate_internal_event_count: internalEvents.length,
    other_primary_event_count: otherEvents.length,
    common_run_count: commonRunCount,
    ...expected,
    total_status: totalStatus,
    execution_nodes: executionNodes,
    delegate_internal_events: internalEvents,
    other_primary_events: otherEvents,
    interpretation_boundary: requiredText(source.interpretation_boundary, "source.adapter.timing_profile.interpretation_boundary", 1200),
  };
}

function validateTimingRows(rows, field, executionNodes) {
  if (!Array.isArray(rows) || rows.length > 1_000_000) throw new Error(`Runtime assignment TFLite timing ${field} must be a bounded array.`);
  const orders = new Set();
  return rows.map((source, index) => {
    const label = `source.adapter.timing_profile.${field}[${index}]`;
    const row = {
      node_kind: requiredText(source.node_kind, `${label}.node_kind`, 80),
      node_type: requiredText(source.node_type, `${label}.node_type`, 300),
      name: requiredText(source.name, `${label}.name`, 1000),
      run_order: requiredNonNegativeInteger(source.run_order, `${label}.run_order`),
      formatter_times_called_integer_average: requiredNonNegativeInteger(source.formatter_times_called_integer_average, `${label}.formatter_times_called_integer_average`),
      event_sample_count: requiredPositiveInteger(source.event_sample_count, `${label}.event_sample_count`),
      run_count: source.run_count == null ? null : requiredPositiveInteger(source.run_count, `${label}.run_count`),
      run_count_derivation_status: requiredText(source.run_count_derivation_status, `${label}.run_count_derivation_status`, 80),
      first_us: nonNegativeNumber(source.first_us, `${label}.first_us`),
      last_us: nonNegativeNumber(source.last_us, `${label}.last_us`),
      min_us: nonNegativeNumber(source.min_us, `${label}.min_us`),
      max_us: nonNegativeNumber(source.max_us, `${label}.max_us`),
      sum_us: nonNegativeNumber(source.sum_us, `${label}.sum_us`),
      mean_per_event_us: nonNegativeNumber(source.mean_per_event_us, `${label}.mean_per_event_us`),
      mean_per_run_us: source.mean_per_run_us == null ? null : nonNegativeNumber(source.mean_per_run_us, `${label}.mean_per_run_us`),
      stddev_us: nonNegativeNumber(source.stddev_us, `${label}.stddev_us`),
      variance_us2: nonNegativeNumber(source.variance_us2, `${label}.variance_us2`),
    };
    if (orders.has(row.run_order)) throw new Error(`Runtime assignment TFLite timing ${field} contains duplicate run_order ${row.run_order}.`);
    orders.add(row.run_order);
    if (row.min_us > row.max_us || row.first_us < row.min_us || row.first_us > row.max_us || row.last_us < row.min_us || row.last_us > row.max_us
      || !closeNumber(row.mean_per_event_us, row.sum_us / row.event_sample_count)
      || (row.run_count == null ? row.mean_per_run_us != null : !closeNumber(row.mean_per_run_us, row.sum_us / row.run_count))) {
      throw new Error(`Runtime assignment TFLite timing ${field}[${index}] statistics are inconsistent.`);
    }
    if (executionNodes) {
      row.runtime_node_index = requiredNonNegativeInteger(source.runtime_node_index, `${label}.runtime_node_index`);
      row.op_index = source.op_index == null ? null : requiredNonNegativeInteger(source.op_index, `${label}.op_index`);
      row.partition_id = optionalText(source.partition_id, `${label}.partition_id`, 100);
      row.provider = requiredText(source.provider, `${label}.provider`, 200);
    } else if (source.delegate_name != null) {
      row.delegate_name = optionalText(source.delegate_name, `${label}.delegate_name`, 200);
    }
    return row;
  });
}

function commonTimingRunCount(rows) {
  if (!rows.length || rows.some((row) => row.run_count == null)) return null;
  const counts = new Set(rows.map((row) => row.run_count));
  return counts.size === 1 ? rows[0].run_count : null;
}

function timingSum(rows) { return rows.reduce((total, row) => total + row.mean_per_run_us, 0); }

function sameNullableNumber(left, right) {
  if (left == null || right == null) return left == null && right == null;
  return closeNumber(Number(left), Number(right));
}

function nonNegativeNumber(value, field) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) throw new Error(`Runtime assignment ${field} must be a non-negative finite number.`);
  return number;
}

function requiredZero(value, field) {
  const number = requiredNonNegativeInteger(value, field);
  if (number !== 0) throw new Error(`Runtime assignment ${field} must be zero.`);
  return number;
}

function requiredPositiveInteger(value, field) {
  const number = Number(value);
  if (!Number.isSafeInteger(number) || number <= 0) throw new Error(`Runtime assignment ${field} must be a positive safe integer.`);
  return number;
}

function sameCountMap(left, right) {
  const leftKeys = Object.keys(left || {}).sort();
  const rightKeys = Object.keys(right || {}).sort();
  return leftKeys.length === rightKeys.length && leftKeys.every((key, index) => key === rightKeys[index] && Number(left[key]) === Number(right[key]));
}

function closeNumber(left, right) {
  if (!Number.isFinite(left) || !Number.isFinite(right)) return false;
  return Math.abs(left - right) <= Math.max(1e-9, Math.abs(right) * 1e-9);
}

function validatedTimestamp(value, required) {
  if (value == null || value === "") {
    if (required) throw new Error("Runtime assignment source.collected_at is required for schema v1.1.");
    return null;
  }
  const text = requiredText(value, "source.collected_at", 64);
  const match = ISO_TIMESTAMP_PATTERN.exec(text);
  const [, yearText, monthText, dayText, hourText, minuteText, secondText, , offsetHourText = "0", offsetMinuteText = "0"] = match || [];
  const [year, month, day, hour, minute, second, offsetHour, offsetMinute] = [yearText, monthText, dayText, hourText, minuteText, secondText, offsetHourText, offsetMinuteText].map(Number);
  const maxDay = Number.isInteger(year) && Number.isInteger(month) && month >= 1 && month <= 12
    ? new Date(Date.UTC(year, month, 0)).getUTCDate()
    : 0;
  const validFields = Boolean(match)
    && day >= 1 && day <= maxDay
    && hour >= 0 && hour <= 23
    && minute >= 0 && minute <= 59
    && second >= 0 && second <= 59
    && offsetHour >= 0 && offsetHour <= 23
    && offsetMinute >= 0 && offsetMinute <= 59;
  if (!validFields || !Number.isFinite(Date.parse(text))) {
    throw new Error("Runtime assignment source.collected_at must be an ISO-8601 timestamp with a timezone.");
  }
  return text;
}

export function deriveRuntimeAssignmentComparison(analysis, runtimeEvidence) {
  const ops = analysis?.ops || [];
  const predictionApplicable = String(analysis?.format || "tflite").toLowerCase() === "tflite";
  const runtimeByOp = new Map((runtimeEvidence?.assignments || []).map((item) => [Number(item.op_index), item]));
  const opComparisons = ops.map((op) => compareOpAssignment(op, runtimeByOp.get(Number(op.index)), predictionApplicable));
  const observedAssignments = opComparisons.filter((item) => item.observed_delegated != null);
  const assessed = opComparisons.filter((item) => item.predicted_delegated != null && item.observed_delegated != null);
  const mismatches = assessed.filter((item) => !item.matches_prediction);
  const matchCount = assessed.length - mismatches.length;
  const overpredicted = mismatches.filter((item) => item.classification === "overpredicted_delegation");
  const underpredicted = mismatches.filter((item) => item.classification === "underpredicted_delegation");
  const graphEdges = deriveRuntimeBoundaryComparisons(analysis, runtimeByOp, predictionApplicable);
  const boundaryAssessed = graphEdges.filter((edge) => edge.predicted_boundary != null && edge.observed_boundary != null);
  const boundaryMismatches = boundaryAssessed.filter((edge) => edge.predicted_boundary !== edge.observed_boundary);
  const predictedBoundaries = graphEdges.filter((edge) => edge.predicted_boundary === true);
  const observedBoundaries = graphEdges.filter((edge) => edge.observed_boundary === true);
  const predictedBoundaryPayload = summarizeBoundaryPayload(predictedBoundaries);
  const observedBoundaryPayload = summarizeBoundaryPayload(observedBoundaries);
  const mac = summarizeAssignmentMacs(opComparisons, ops);
  const timingProfile = runtimeEvidence?.source?.adapter?.timing_profile || null;
  const duration = summarizeAssignmentDuration(opComparisons, runtimeEvidence?.source?.duration_semantics, ops.length, timingProfile);
  const partitions = deriveObservedPartitions(ops, runtimeByOp, runtimeEvidence?.source?.duration_semantics, predictionApplicable, timingProfile);
  const placementStatus = !predictionApplicable ? "not_applicable" : assessed.length === ops.length ? "assessed" : assessed.length ? "partial" : "not_assessed";
  const boundaryStatus = !predictionApplicable ? "not_applicable" : boundaryAssessed.length === graphEdges.length ? "assessed" : boundaryAssessed.length ? "partial" : "not_assessed";
  return {
    schema: RUNTIME_ASSIGNMENT_COMPARISON_SCHEMA,
    evidence_class: "DERIVED_FROM_OBSERVED_RUNTIME",
    status: !predictionApplicable ? "observed_assignment_only" : placementStatus === "assessed" && boundaryStatus === "assessed" ? "assessed" : placementStatus === "not_assessed" ? "not_assessed" : "partial",
    prediction_applicability: predictionApplicable ? "tflite_xnnpack_static_prediction" : "not_applicable_for_onnx_execution_provider_assignment",
    source_adapter_schema: runtimeEvidence?.source?.adapter?.schema || null,
    artifact_sha256: runtimeEvidence?.artifact_sha256 || null,
    target_profile_id: runtimeEvidence?.target_profile_id || null,
    target_profile_sha256: runtimeEvidence?.target_profile_sha256 || null,
    graph_op_count: ops.length,
    assignment_row_count: runtimeEvidence?.assignments?.length || 0,
    placement_assessment: {
      status: placementStatus,
      prediction_applicability: predictionApplicable ? "applicable" : "not_applicable",
      observed_assignment_count: observedAssignments.length,
      observed_assignment_coverage_ratio: ratioOrNull(observedAssignments.length, ops.length),
      assessed_op_count: assessed.length,
      unassessed_op_count: ops.length - assessed.length,
      coverage_ratio: ratioOrNull(assessed.length, ops.length),
      match_count: matchCount,
      mismatch_count: mismatches.length,
      match_ratio: ratioOrNull(matchCount, assessed.length),
      predicted_delegated_op_count: predictionApplicable ? assessed.filter((item) => item.predicted_delegated).length : null,
      observed_delegated_op_count: observedAssignments.filter((item) => item.observed_delegated).length,
      overpredicted_delegation_count: overpredicted.length,
      underpredicted_delegation_count: underpredicted.length,
      confusion_matrix: {
        predicted_delegate_observed_delegate: predictionApplicable ? assessed.filter((item) => item.predicted_delegated && item.observed_delegated).length : null,
        predicted_delegate_observed_cpu: predictionApplicable ? overpredicted.length : null,
        predicted_cpu_observed_delegate: predictionApplicable ? underpredicted.length : null,
        predicted_cpu_observed_cpu: predictionApplicable ? assessed.filter((item) => !item.predicted_delegated && !item.observed_delegated).length : null,
      },
    },
    mac_comparison: mac,
    duration_comparison: duration,
    observed_partitions: partitions,
    boundary_comparison: {
      status: boundaryStatus,
      prediction_applicability: predictionApplicable ? "applicable" : "not_applicable",
      observed_relation_edge_count: graphEdges.filter((edge) => edge.observed_boundary != null).length,
      graph_edge_count: graphEdges.length,
      assessed_edge_count: boundaryAssessed.length,
      unassessed_edge_count: graphEdges.length - boundaryAssessed.length,
      match_count: boundaryAssessed.length - boundaryMismatches.length,
      mismatch_count: boundaryMismatches.length,
      match_ratio: ratioOrNull(boundaryAssessed.length - boundaryMismatches.length, boundaryAssessed.length),
      overpredicted_boundary_count: boundaryMismatches.filter((edge) => edge.predicted_boundary).length,
      underpredicted_boundary_count: boundaryMismatches.filter((edge) => !edge.predicted_boundary).length,
    },
    predicted_boundary_inventory: {
      schema: "deepbom.predicted_runtime_boundary_summary.v1",
      status: predictionApplicable ? "assessed" : "not_applicable",
      assignment_evidence_class: predictionApplicable ? "PREDICTED" : "NOT_APPLICABLE",
      payload_evidence_class: "DERIVED",
      edge_count: predictedBoundaries.length,
      assessed_edge_payload_bytes: predictedBoundaryPayload.assessed_edge_payload_bytes,
      summed_edge_payload_bytes: predictedBoundaryPayload.summed_edge_payload_bytes,
      unique_tensor_count: predictedBoundaryPayload.unique_tensor_count,
      assessed_unique_tensor_payload_bytes: predictedBoundaryPayload.assessed_unique_tensor_payload_bytes,
      unique_tensor_payload_bytes: predictedBoundaryPayload.unique_tensor_payload_bytes,
      materialization_status: "NOT_ASSESSED",
    },
    observed_boundary_inventory: {
      schema: "deepbom.observed_partition_boundary_edges.v1",
      evidence_class: "DERIVED_FROM_OBSERVED_RUNTIME",
      edge_count: observedBoundaries.length,
      assessed_relation_edge_count: graphEdges.filter((edge) => edge.observed_boundary != null).length,
      unassessed_relation_edge_count: graphEdges.filter((edge) => edge.observed_boundary == null).length,
      assessed_edge_payload_bytes: observedBoundaryPayload.assessed_edge_payload_bytes,
      summed_edge_payload_bytes: observedBoundaryPayload.summed_edge_payload_bytes,
      unique_tensor_count: observedBoundaryPayload.unique_tensor_count,
      assessed_unique_tensor_payload_bytes: observedBoundaryPayload.assessed_unique_tensor_payload_bytes,
      unique_tensor_payload_bytes: observedBoundaryPayload.unique_tensor_payload_bytes,
      materialization_status: "NOT_ASSESSED",
      edges: observedBoundaries,
      interpretation_boundary: "Execution-domain or explicit runtime-partition edges derived from imported original-op assignments. Payload is declared logical tensor size; copy/alias/materialization and transition latency are not established.",
    },
    boundary_comparisons: graphEdges,
    mismatches,
    boundary_differences: boundaryMismatches,
    op_comparisons: opComparisons,
    interpretation_boundary: predictionApplicable
      ? "Prediction agreement is evaluated only where delegated is boolean. Delegate-to-delegate boundary relation requires distinct observed providers or explicit partition IDs; missing partition IDs remain unassessed. Static XNNPACK prediction and imported runtime/build provenance remain separately visible."
      : "ONNX execution-provider assignment has no TFLite XNNPACK static prediction. Provider rows and provider/CPU transition edges are observed or derived from imported runtime evidence; prediction agreement and predicted boundaries are not applicable.",
  };
}

function compareOpAssignment(op, runtime, predictionApplicable) {
  const predicted = predictionApplicable ? Number(op?.xnnpack_chain_id) >= 0 : null;
  const observed = typeof runtime?.delegated === "boolean" ? runtime.delegated : null;
  const matches = observed == null || predicted == null ? null : observed === predicted;
  return {
    op_index: Number(op.index),
    op_name: op.name || "",
    predicted_delegated: predicted,
    predicted_domain: predicted == null ? "NOT_APPLICABLE" : predicted ? `XNNPACK:C${op.xnnpack_chain_id}` : "TFLITE_CPU",
    observed_delegated: observed,
    observed_provider: runtime?.provider || null,
    observed_partition_id: runtime?.partition_id ?? null,
    observed_kernel: runtime?.kernel || null,
    observed_kernel_id: runtime?.kernel_id || null,
    observed_lowering_id: runtime?.lowering_id || null,
    selector_evidence_class: runtime?.selector_evidence_class || "NOT_COLLECTED",
    resolved_selector_dimensions: runtime?.resolved_selector_dimensions || [],
    duration_us: runtime?.duration_us ?? null,
    macs: opMacs(op),
    estimated_logical_bytes: finiteNonNegative(op?.estimated_bytes),
    matches_prediction: matches,
    classification: observed == null
      ? "not_assessed"
      : predicted == null
        ? "observed_provider_no_static_prediction"
      : matches
        ? predicted ? "matched_delegated" : "matched_cpu"
        : predicted ? "overpredicted_delegation" : "underpredicted_delegation",
  };
}

function deriveRuntimeBoundaryComparisons(analysis, runtimeByOp, predictionApplicable) {
  const tensors = new Map((analysis?.tensors || []).map((tensor) => [Number(tensor.index), tensor]));
  const producers = new Map();
  for (const op of analysis?.ops || []) for (const index of op.outputs || []) if (Number(index) >= 0) producers.set(Number(index), op);
  const edges = [];
  for (const consumer of analysis?.ops || []) {
    for (const tensorIndex of new Set((consumer.inputs || []).map(Number).filter((index) => index >= 0))) {
      const producer = producers.get(tensorIndex);
      const tensor = tensors.get(tensorIndex);
      if (!producer || !tensor || tensor.constant_buffer) continue;
      const predictedProducer = predictionApplicable ? Number(producer.xnnpack_chain_id) >= 0 : null;
      const predictedConsumer = predictionApplicable ? Number(consumer.xnnpack_chain_id) >= 0 : null;
      const predictedBoundary = predictionApplicable ? predictedDomainKey(producer) !== predictedDomainKey(consumer) : null;
      const observed = observedBoundary(runtimeByOp.get(Number(producer.index)), runtimeByOp.get(Number(consumer.index)));
      edges.push({
        tensor_index: tensorIndex,
        tensor_name: tensor.name || "",
        tensor_shape: tensor.shape || [],
        tensor_dtype: tensor.dtype || "",
        payload_bytes: deterministicTensorPayloadBytes(tensor),
        producer_op_index: Number(producer.index),
        producer_op_name: producer.name || "",
        consumer_op_index: Number(consumer.index),
        consumer_op_name: consumer.name || "",
        predicted_producer_domain: predictionApplicable ? predictedDomainKey(producer) : "NOT_APPLICABLE",
        predicted_consumer_domain: predictionApplicable ? predictedDomainKey(consumer) : "NOT_APPLICABLE",
        predicted_boundary: predictedBoundary,
        observed_producer_domain: observed.producer_domain,
        observed_consumer_domain: observed.consumer_domain,
        observed_boundary: observed.boundary,
        observed_relation_reason: observed.reason,
        observed_direction: observed.boundary === true ? boundaryDirection(observed.producer_delegated, observed.consumer_delegated) : null,
        classification: observed.boundary == null
          ? "not_assessed"
          : predictedBoundary == null
            ? observed.boundary ? "observed_boundary_no_static_prediction" : "observed_same_domain_no_static_prediction"
          : observed.boundary === predictedBoundary
            ? predictedBoundary ? "matched_boundary" : "matched_no_boundary"
            : predictedBoundary ? "overpredicted_boundary" : "underpredicted_boundary",
        predicted_direction: predictedBoundary === true ? boundaryDirection(predictedProducer, predictedConsumer) : null,
        materialization_status: "NOT_ASSESSED",
      });
    }
  }
  return edges.sort((left, right) => `${left.producer_op_index}:${left.consumer_op_index}:${left.tensor_index}`.localeCompare(`${right.producer_op_index}:${right.consumer_op_index}:${right.tensor_index}`));
}

function observedBoundary(producer, consumer) {
  const producerDelegated = typeof producer?.delegated === "boolean" ? producer.delegated : null;
  const consumerDelegated = typeof consumer?.delegated === "boolean" ? consumer.delegated : null;
  const base = {
    producer_delegated: producerDelegated,
    consumer_delegated: consumerDelegated,
    producer_domain: observedDomainKey(producer),
    consumer_domain: observedDomainKey(consumer),
  };
  if (producerDelegated == null || consumerDelegated == null) return { ...base, boundary: null, reason: "producer_or_consumer_assignment_unassessed" };
  if (producerDelegated !== consumerDelegated) return { ...base, boundary: true, reason: "delegate_cpu_domain_transition_observed" };
  if (!producerDelegated) return { ...base, boundary: false, reason: "both_original_ops_observed_on_cpu" };
  if (producer.provider !== consumer.provider) return { ...base, boundary: true, reason: "delegated_provider_transition_observed" };
  if (producer.partition_id != null && consumer.partition_id != null) {
    const boundary = String(producer.partition_id) !== String(consumer.partition_id);
    return { ...base, boundary, reason: boundary ? "distinct_runtime_partition_ids_observed" : "same_runtime_partition_id_observed" };
  }
  return { ...base, boundary: null, reason: "delegate_partition_relation_requires_partition_ids" };
}

function deriveObservedPartitions(ops, runtimeByOp, durationSemantics, allowImplicitPartitions, timingProfile = null) {
  const segments = [];
  let active = null;
  const flush = () => { if (active) segments.push(active); active = null; };
  for (const op of ops) {
    const runtime = runtimeByOp.get(Number(op.index));
    if (runtime?.delegated !== true) { flush(); continue; }
    const explicit = runtime.partition_id != null;
    const key = explicit ? `${runtime.provider}\u0000${runtime.partition_id}` : `implicit\u0000${runtime.provider}`;
    if (!active || active.key !== key) {
      flush();
      active = { key, provider: runtime.provider, partition_id: runtime.partition_id ?? null, explicit_partition_id: explicit, first_op: Number(op.index), last_op: Number(op.index), op_count: 0, op_indices: [], assessed_macs: 0, macs: 0, macs_complete: true, assessed_duration_us: 0, duration_row_count: 0 };
    }
    active.last_op = Number(op.index);
    active.op_count += 1;
    active.op_indices.push(Number(op.index));
    const macs = opMacs(op);
    if (macs == null) active.macs_complete = false;
    else active.assessed_macs += macs;
    if (runtime.duration_us != null) { active.assessed_duration_us += Number(runtime.duration_us); active.duration_row_count += 1; }
  }
  flush();
  const providerSegments = deriveObservedProviderSegments(ops, runtimeByOp, durationSemantics);
  for (const item of segments) {
    item.macs = item.macs_complete ? item.assessed_macs : null;
    item.duration_us = durationSemantics === "per_original_op_exclusive" && item.duration_row_count === item.op_count ? item.assessed_duration_us : null;
    delete item.macs_complete;
  }
  const explicitGroups = new Map();
  for (const segment of segments.filter((item) => item.explicit_partition_id)) {
    let partition = explicitGroups.get(segment.key);
    if (!partition) {
      partition = { ...segment, op_indices: [], op_count: 0, assessed_macs: 0, macs: 0, assessed_duration_us: 0, duration_row_count: 0, contiguous_segment_count: 0, macs_complete: true, duration_complete: true };
      explicitGroups.set(segment.key, partition);
    }
    partition.first_op = Math.min(partition.first_op, segment.first_op);
    partition.last_op = Math.max(partition.last_op, segment.last_op);
    partition.op_count += segment.op_count;
    partition.op_indices.push(...segment.op_indices);
    partition.assessed_macs += segment.assessed_macs;
    partition.macs_complete &&= segment.macs != null;
    partition.assessed_duration_us += segment.assessed_duration_us;
    partition.duration_row_count += segment.duration_row_count;
    partition.duration_complete &&= segment.duration_us != null;
    partition.contiguous_segment_count += 1;
  }
  const explicitPartitions = [...explicitGroups.values()].map((partition) => {
    partition.op_indices.sort((left, right) => left - right);
    partition.macs = partition.macs_complete ? partition.assessed_macs : null;
    partition.duration_us = partition.duration_complete ? partition.assessed_duration_us : null;
    delete partition.key;
    delete partition.macs_complete;
    delete partition.duration_complete;
    return partition;
  });
  const timedPartitions = new Map((timingProfile?.execution_nodes || [])
    .filter((row) => row.node_kind === "delegate_partition" && row.partition_id != null)
    .map((row) => [String(row.partition_id), row]));
  for (const partition of explicitPartitions) {
    const timing = timedPartitions.get(String(partition.partition_id));
    if (timing) {
      partition.duration_us = timing.mean_per_run_us;
      partition.duration_sample_count = timing.run_count;
      partition.duration_source = "observed_execution_plan_delegate_node";
    }
  }
  const implicitPartitions = allowImplicitPartitions
    ? segments.filter((item) => !item.explicit_partition_id).map((item) => ({ ...item, contiguous_segment_count: 1 }))
    : [];
  for (const partition of implicitPartitions) delete partition.key;
  const partitions = [...explicitPartitions, ...implicitPartitions].sort((left, right) => left.first_op - right.first_op);
  const explicitCount = explicitPartitions.length;
  return {
    status: partitions.length
      ? "assessed_from_imported_rows"
      : providerSegments.length && !allowImplicitPartitions ? "not_assessed_no_partition_ids" : "not_assessed",
    partition_count: partitions.length,
    explicit_partition_count: explicitCount,
    implicit_contiguous_partition_count: allowImplicitPartitions ? partitions.length - explicitCount : 0,
    noncontiguous_partition_id_count: explicitPartitions.filter((partition) => partition.contiguous_segment_count > 1).length,
    provider_segment_count: providerSegments.length,
    provider_segments: providerSegments,
    partitions,
  };
}

function deriveObservedProviderSegments(ops, runtimeByOp, durationSemantics) {
  const segments = [];
  let active = null;
  const flush = () => { if (active) segments.push(active); active = null; };
  for (const op of ops) {
    const runtime = runtimeByOp.get(Number(op.index));
    if (typeof runtime?.delegated !== "boolean") { flush(); continue; }
    const key = `${runtime.provider}\u0000${runtime.delegated}`;
    if (!active || active.key !== key) {
      flush();
      active = { key, provider: runtime.provider, delegated: runtime.delegated, first_op: Number(op.index), last_op: Number(op.index), op_count: 0, op_indices: [], assessed_duration_us: 0, duration_row_count: 0 };
    }
    active.last_op = Number(op.index);
    active.op_count += 1;
    active.op_indices.push(Number(op.index));
    if (runtime.duration_us != null) { active.assessed_duration_us += Number(runtime.duration_us); active.duration_row_count += 1; }
  }
  flush();
  for (const item of segments) {
    item.duration_us = durationSemantics === "per_original_op_exclusive" && item.duration_row_count === item.op_count ? item.assessed_duration_us : null;
    delete item.key;
  }
  return segments;
}

function summarizeAssignmentMacs(rows, ops) {
  const graphMacs = ops.map(opMacs).filter((value) => value != null);
  const classified = rows.filter((item) => item.observed_delegated != null);
  const assessed = classified.filter((item) => item.macs != null);
  const assessedTotal = safeSum(assessed.map((item) => item.macs));
  const graphTotal = safeSum(graphMacs);
  const predictionAssessed = assessed.filter((item) => item.predicted_delegated != null);
  const mismatch = predictionAssessed.filter((item) => item.matches_prediction === false);
  return {
    status: !classified.length ? "not_assessed" : !predictionAssessed.length ? "observed_assignment_only" : assessed.length === classified.length ? "assessed_for_classified_ops" : "partial",
    assessed_op_count: assessed.length,
    assessed_macs: assessedTotal,
    graph_assessed_macs: graphTotal,
    coverage_ratio: graphTotal > 0 ? assessedTotal / graphTotal : null,
    predicted_delegated_macs: predictionAssessed.length ? safeSum(predictionAssessed.filter((item) => item.predicted_delegated).map((item) => item.macs)) : null,
    observed_delegated_macs: safeSum(assessed.filter((item) => item.observed_delegated).map((item) => item.macs)),
    mismatch_macs: predictionAssessed.length ? safeSum(mismatch.map((item) => item.macs)) : null,
    mismatch_mac_ratio: predictionAssessed.length && assessedTotal > 0 ? safeSum(mismatch.map((item) => item.macs)) / assessedTotal : null,
  };
}

function summarizeAssignmentDuration(rows, semantics, graphOpCount, timingProfile = null) {
  if (semantics === "per_execution_plan_node_exclusive" && timingProfile) {
    const total = timingProfile.execution_node_total_us;
    return {
      status: total != null ? "assessed" : timingProfile.mapped_execution_node_count ? "partial" : "not_assessed",
      duration_semantics: semantics,
      duration_row_count: timingProfile.original_execution_node_timing_count,
      execution_plan_node_count: timingProfile.execution_plan_node_count,
      mapped_execution_node_count: timingProfile.mapped_execution_node_count,
      assessed_duration_us: timingProfile.mapped_execution_node_subtotal_us,
      total_duration_us: total,
      cpu_execution_node_subtotal_us: timingProfile.cpu_execution_node_subtotal_us,
      delegate_partition_subtotal_us: timingProfile.delegate_partition_subtotal_us,
      primary_delegate_profiled_subtotal_us: timingProfile.primary_delegate_profiled_subtotal_us,
      delegate_internal_section_subtotal_us: timingProfile.delegate_internal_section_subtotal_us,
      delegate_internal_profiled_subtotal_us: timingProfile.delegate_internal_profiled_subtotal_us,
      mismatch_assessed_duration_us: null,
      mismatch_duration_us: null,
      reason: total != null
        ? "Every imported execution-plan node has an exclusive timing row with a common derived run count; delegate partitions are counted once and their duration is not copied to replaced original ops."
        : "Only mapped execution-plan rows are subtotaled. Delegate-internal events are reported separately and are not treated as partition or graph totals.",
    };
  }
  const additive = semantics === "per_original_op_exclusive";
  const withDuration = rows.filter((item) => item.duration_us != null);
  const predictionAssessed = rows.some((item) => item.predicted_delegated != null);
  const mismatches = rows.filter((item) => item.matches_prediction === false);
  const mismatchWithDuration = mismatches.filter((item) => item.duration_us != null);
  return {
    status: !additive ? "not_assessed" : withDuration.length === graphOpCount ? "assessed" : withDuration.length ? "partial" : "not_assessed",
    duration_semantics: semantics || "unspecified",
    duration_row_count: withDuration.length,
    assessed_duration_us: additive ? safeSum(withDuration.map((item) => item.duration_us)) : null,
    total_duration_us: additive && withDuration.length === graphOpCount ? safeSum(withDuration.map((item) => item.duration_us)) : null,
    mismatch_assessed_duration_us: additive && predictionAssessed ? safeSum(mismatchWithDuration.map((item) => item.duration_us)) : null,
    mismatch_duration_us: additive && predictionAssessed && mismatchWithDuration.length === mismatches.length ? safeSum(mismatchWithDuration.map((item) => item.duration_us)) : null,
    reason: additive ? "Imported per-original-op exclusive durations are additive; a total is emitted only at full graph coverage." : "Duration rows are not summed because source.duration_semantics does not declare additive per-original-op exclusive timing.",
  };
}

function summarizeBoundaryPayload(edges) {
  const assessed = edges.filter((edge) => edge.payload_bytes != null);
  const unique = new Map();
  for (const edge of edges) if (!unique.has(edge.tensor_index)) unique.set(edge.tensor_index, edge.payload_bytes);
  const assessedUnique = [...unique.values()].filter((value) => value != null);
  return {
    assessed_edge_payload_bytes: safeIntegerSum(assessed.map((edge) => edge.payload_bytes)),
    summed_edge_payload_bytes: assessed.length === edges.length ? safeIntegerSum(assessed.map((edge) => edge.payload_bytes)) : null,
    unique_tensor_count: unique.size,
    assessed_unique_tensor_payload_bytes: safeIntegerSum(assessedUnique),
    unique_tensor_payload_bytes: assessedUnique.length === unique.size ? safeIntegerSum(assessedUnique) : null,
  };
}

function deterministicTensorPayloadBytes(tensor) {
  const shape = tensor?.shape || [];
  const signature = tensor?.shape_signature || [];
  if ([...shape, ...signature].some((dim) => !Number.isInteger(Number(dim)) || Number(dim) < 0)) return null;
  const elements = shape.reduce((product, dim) => product * Number(dim), 1);
  if (!Number.isSafeInteger(elements)) return null;
  const bits = {
    UINT2: 2, INT2: 2, UINT4: 4, INT4: 4, FLOAT4E2M1: 4,
    BOOL: 8, INT8: 8, UINT8: 8, FLOAT8E4M3FN: 8, FLOAT8E4M3FNUZ: 8,
    FLOAT8E5M2: 8, FLOAT8E5M2FNUZ: 8, FLOAT8E8M0: 8,
    FLOAT16: 16, BFLOAT16: 16, INT16: 16, UINT16: 16,
    FLOAT32: 32, INT32: 32, UINT32: 32,
    FLOAT64: 64, INT64: 64, UINT64: 64, COMPLEX64: 64, COMPLEX128: 128,
  }[String(tensor?.dtype || "").toUpperCase()];
  const bytes = bits == null || elements > Math.floor(Number.MAX_SAFE_INTEGER / bits) ? null : Math.ceil(elements * bits / 8);
  return Number.isSafeInteger(bytes) ? bytes : null;
}

function predictedDomainKey(op) {
  return Number(op?.xnnpack_chain_id) >= 0 ? `XNNPACK:C${op.xnnpack_chain_id}` : "TFLITE_CPU";
}

function observedDomainKey(runtime) {
  if (typeof runtime?.delegated !== "boolean") return null;
  if (!runtime.delegated) return `CPU:${runtime.provider}`;
  return `${runtime.provider}:P${runtime.partition_id ?? "unknown"}`;
}

function boundaryDirection(producerDelegated, consumerDelegated) {
  if (producerDelegated && consumerDelegated) return "delegate_partition_to_delegate_partition";
  return producerDelegated ? "delegate_to_cpu" : "cpu_to_delegate";
}

function opMacs(op) {
  return op?.macs_status === "not_assessed" ? null : finiteNonNegative(op?.macs);
}

function finiteNonNegative(value) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : null;
}

function safeSum(values) {
  let total = 0;
  for (const value of values) {
    total += Number(value);
    if (!Number.isFinite(total)) return null;
  }
  return total;
}

function safeIntegerSum(values) {
  const total = safeSum(values);
  return Number.isSafeInteger(total) ? total : null;
}

function ratioOrNull(numerator, denominator) {
  return denominator > 0 ? numerator / denominator : null;
}

export function renderKernelInspector({
  analysis,
  body,
  summary,
  comparisonPanel,
  boundaryList,
  status,
  query = "",
  filter = "all",
  runtimeEvidence = null,
  onSelect = () => {},
  onLoadSourceEvidence = null,
} = {}) {
  if (!body || !summary || !status) return;
  const target = analysis?.target_profile || {};
  const format = String(analysis?.format || "tflite").toLowerCase();
  const predictionApplicable = format === "tflite";
  const coreMlPlan = runtimeEvidence?.schema === "deepbom.coreml_compute_plan.v1" ? runtimeEvidence : null;
  const runtimeRows = coreMlPlan ? coreMlPlan.structure.rows.map((item) => ({
    ...item,
    provider: item.preferred_compute_device || "not determined",
    kernel: "MLComputePlan anticipated device/cost",
    kernel_id: "plan",
    evidence_kind: "coreml_compute_plan",
  })) : runtimeEvidence?.assignments || [];
  const runtimeByOp = new Map(runtimeRows.map((item) => [item.op_index, item]));
  const comparison = runtimeEvidence?.comparison || null;
  const adapter = runtimeEvidence?.source?.adapter || null;
  const comparisonByOp = new Map((comparison?.op_comparisons || []).map((item) => [item.op_index, item]));
  const boundaryInventory = predictionApplicable ? predictedPartitionBoundaryInventory(analysis) : null;
  const boundaryOpIndices = new Set([
    ...(boundaryInventory?.edges || []).flatMap((edge) => [edge.producer_op_index, edge.consumer_op_index]),
    ...(comparison?.observed_boundary_inventory?.edges || []).flatMap((edge) => [edge.producer_op_index, edge.consumer_op_index]),
  ]);
  const mismatchOpIndices = new Set((comparison?.mismatches || []).map((item) => item.op_index));
  const panel = body.closest("#kernelInspectorPanel");
  for (const button of panel?.querySelectorAll('[data-kernel-filter="selector"], [data-kernel-filter="tail"], [data-kernel-filter="packing"]') || []) {
    button.hidden = !predictionApplicable;
  }
  const normalizedQuery = String(query || "").trim().toLowerCase();
  const allOps = analysis?.ops || [];
  const ortProviders = analysis?.ort_compatibility_evidence?.execution_providers || [];
  const selectorAssessment = String(analysis?.xnnpack_selector_assessment_status || "not_reported");
  const rows = allOps.filter((op) => {
    if (filter === "boundary" && !boundaryOpIndices.has(op.index)) return false;
    if (filter === "mismatch" && !mismatchOpIndices.has(op.index)) return false;
    if (filter === "selector" && !["SOURCE_ENUMERATED_CANDIDATE_SET", "SOURCE_ENUMERATED_NO_MATCH"].includes(op.xnnpack_kernel_evidence_class)) return false;
    if (filter === "tail" && op.channel_alignment_status !== "misaligned") return false;
    if (filter === "packing" && op.weight_packing_risk !== "warn") return false;
    if (!normalizedQuery) return true;
    const runtime = runtimeByOp.get(op.index);
    const compared = comparisonByOp.get(op.index);
    const candidateSearch = (op.xnnpack_kernel_candidates || []).flatMap((candidate) => [candidate.family, candidate.architecture_condition, candidate.compile_condition, candidate.runtime_condition, candidate.source_ref, candidate.source_file_sha256, candidate.alignment_multiple, candidate.padded_output_channels, candidate.inactive_output_channels, candidate.inactive_lane_ratio]);
    const selectorFacts = op.selector_artifact_facts || {};
    const ortSearch = ortEpRowsForOp(analysis, op.index).flatMap((item) => [item.execution_provider, item.status, item.imported_opset, item.resolved_schema_version, item.documented_condition, item.schema_source_ref, item.schema_source_sha256]);
    return [op.index, op.name, op.xnnpack_kernel_candidate, op.xnnpack_kernel_source, op.xnnpack_build_requirement, op.xnnpack_reason, op.xnnpack_break_class, op.xnnpack_chain_id, op.no_match_reason_code, ...(op.unresolved_selector_dimensions || []), ...Object.values(selectorFacts), runtime?.provider, runtime?.kernel, compared?.classification, ...candidateSearch, ...ortSearch]
      .some((value) => String(value ?? "").toLowerCase().includes(normalizedQuery));
  });
  const buildRequirements = [...new Set(rows.map((op) => op.xnnpack_build_requirement).filter(Boolean))];
  const summaryText = predictionApplicable
    ? `Planning profile: ${target.label || target.id || "target"} | ${target.architecture || "architecture unspecified"} | SIMD ${target.simd_width_bits || "?"}-bit | INT8/FP16/FP32 lanes ${target.int8_lanes || "?"}/${target.fp16_lanes || "?"}/${target.fp32_lanes || "?"} | selector ${selectorAssessment} | ${rows.length}/${allOps.length} ops`
    : format === "onnx"
      ? `ONNX Runtime provider evidence | original graph ${allOps.length} op(s) | ${ortProviders.length} pinned source EP profile(s) | mapped runtime rows ${runtimeEvidence?.assignment_count || 0} | actual placement and microkernel not statically claimed | ${rows.length}/${allOps.length} ops shown`
      : `Core ML serialized operation evidence | ${allOps.length} op(s) | MLComputePlan ${coreMlPlan ? `${coreMlPlan.structure.operation_count} identity-bound estimate row(s)` : "not imported"} | execution placement and microkernel not observed | ${rows.length}/${allOps.length} ops shown`;
  renderSelectorDecisionLedger(summary, summaryText, analysis, predictionApplicable, onSelect, onLoadSourceEvidence);
  if (predictionApplicable) renderTfliteDelegateSourceLedger(summary, analysis, runtimeEvidence, onSelect);
  const boundaryStatus = boundaryInventory
    ? ` Static-predicted internal boundary inventory: ${boundaryInventory.edge_count} edge(s), ${boundaryInventory.unique_tensor_count} unique tensor(s), ${boundaryInventory.summed_edge_payload_bytes == null ? `${formatBytes(boundaryInventory.assessed_edge_payload_bytes || 0)} assessed partial logical payload` : `${formatBytes(boundaryInventory.summed_edge_payload_bytes)} summed logical edge payload`}; runtime materialization unconfirmed.`
    : predictionApplicable
      ? " Predicted boundary-edge inventory not emitted."
      : format === "onnx"
        ? comparison ? ` Observed provider-transition inventory: ${comparison.observed_boundary_inventory?.edge_count || 0} edge(s); runtime materialization unconfirmed.` : " Provider-transition inventory requires imported runtime evidence."
        : " MLComputePlan does not report materialized inter-device tensor transfers; graph-edge movement remains unobserved.";
  renderAssignmentComparison(comparisonPanel, comparison, comparison?.predicted_boundary_inventory || boundaryInventory, onSelect, adapter?.timing_profile || null, adapter?.native_capture || null, buildRuntimeBackendEvidenceLedger(runtimeEvidence));
  renderBoundaryInventory(boundaryList, boundaryInventory, comparison, filter, onSelect);
  const selectorStatus = !predictionApplicable
    ? ` XNNPACK selector enumeration is not applicable to this ${format === "onnx" ? "ONNX execution-provider" : "Core ML compute-plan"} view.`
    : selectorAssessment === "complete"
    ? " Protected pinned-source selector evidence is loaded; candidate configurations still do not identify the executed runtime microkernel."
    : selectorAssessment === "not_loaded"
      ? " Source-backed selector evidence is not loaded; open static profile hints are HEURISTIC and contain no source-backed tile claim."
      : " Pinned-source selector enumeration is not available for this planning profile.";
  const delegateBuildStatus = runtimeEvidence?.tflite_delegate_build_inventory
    ? ` TFLite selected-build inventory: GPU ${runtimeEvidence.tflite_delegate_build_inventory.gpu.compiled_status}, quant option ${runtimeEvidence.tflite_delegate_build_inventory.gpu.quantized_model_flag_status}, NNAPI ${runtimeEvidence.tflite_delegate_build_inventory.nnapi.compiled_status}; placement remains unobserved.`
    : analysis?.tflite_delegate_compatibility_evidence
      ? " TFLite GPU/NNAPI source candidates are loaded; selected-build inventory is not imported."
      : "";
  const staticKernelBoundary = !predictionApplicable
    ? format === "coreml"
      ? coreMlPlan ? "MLComputePlan anticipated device usage and relative costs are imported; fusion, lowering, selected kernels, memory allocation, and executed placement remain unobserved." : "Core ML runtime specialization, compute-device plan, fusion, lowering, and microkernel selection are not statically inferred."
      : ortProviders.length
      ? "Pinned ONNX OpSchema and ORT kernel-registration candidates are loaded; type/attribute eligibility, GetCapability placement, and microkernel selection remain unobserved."
      : "ONNX execution-provider placement and microkernel selection are not statically predicted."
    : selectorAssessment === "complete"
    ? "Protected source-enumerated kernel candidates are static evidence, not runtime confirmation."
    : "Public kernel profile hints are HEURISTIC and are not runtime confirmation.";
  const adapterStatus = !adapter ? "" : adapter.schema?.startsWith("deepbom.tflite_runtime_info_adapter.v")
    ? ` TFLite runtime plan ${(runtimeEvidence.source.profile_sha256 || "").slice(0, 16)}; exact topology ${adapter.topology_binding?.matched_original_op_count || 0}/${adapter.topology_binding?.graph_op_count || 0}, ${adapter.delegate_node_count || 0} explicit delegate partition(s), ${adapter.execution_plan_node_count || 0} execution node(s); source artifact SHA-256 absent.${adapter.timing_profile ? ` Timing profile ${adapter.timing_profile.profile_sha256.slice(0, 16)} maps ${adapter.timing_profile.mapped_execution_node_count}/${adapter.timing_profile.execution_plan_node_count} execution node(s); graph total ${adapter.timing_profile.execution_node_total_us == null ? "withheld" : formatUs(adapter.timing_profile.execution_node_total_us)}, delegate-internal events ${adapter.timing_profile.delegate_internal_event_count}.` : " Timing and executed microkernel not exposed."}`
    : ` ORT profile ${(runtimeEvidence.source.profile_sha256 || "").slice(0, 16)}; mapped ${adapter.mapped_kernel_event_count}/${adapter.kernel_event_count} kernel event(s), unresolved ${adapter.unresolved_runtime_node_count}, conflicts ${adapter.conflict_count}; partitions and microkernel symbols not inferred.${adapter.native_capture ? ` Pinned native ${adapter.native_capture.profile_role} capture ${adapter.native_capture.capture_id}; production transformed graph ${(adapter.native_capture.paired_profile_runtime_graph?.profiles || []).find((item) => item.role === "production")?.runtime_node_count || 0} observed node(s), original mapping not inferred.` : ""}`;
  status.textContent = (coreMlPlan
    ? `${coreMlPlan.evidence_class}: ${coreMlPlan.structure.operation_count}/${allOps.length} operation rows bound to compiled model ${coreMlPlan.runtime.compiled_model_content_sha256.slice(0, 16)}... under ${coreMlPlan.configuration.compute_units}; preferred/supported devices and cost weights are estimates, not execution observations.`
    : runtimeEvidence
    ? `${runtimeEvidence.evidence_class} assignments: declared ${runtimeEvidence.runtime.name} ${runtimeEvidence.runtime.version} / ${runtimeEvidence.runtime.backend}; ${runtimeEvidence.assignment_count}/${runtimeEvidence.graph_op_count} rows (${formatPercent(runtimeEvidence.coverage_ratio)}) bound by artifact SHA-256 and target-profile SHA-256.${adapterStatus}${comparison ? predictionApplicable
      ? ` Placement match ${percentOrNA(comparison.placement_assessment?.match_ratio)} across ${comparison.placement_assessment?.assessed_op_count || 0} classified op(s); ${comparison.placement_assessment?.mismatch_count || 0} mismatch(es).`
      : ` Observed provider assignment coverage ${percentOrNA(comparison.placement_assessment?.observed_assignment_coverage_ratio)}; ${comparison.boundary_comparison?.observed_relation_edge_count || 0}/${comparison.boundary_comparison?.graph_edge_count || 0} graph-edge relation(s) assessed; static EP prediction not applicable.` : ""}`
    : `No runtime assignment evidence imported. ${staticKernelBoundary}${buildRequirements.length ? ` Required runtime build: ${buildRequirements.join(" / ")}.` : ""}`) + boundaryStatus;
  status.textContent += selectorStatus;
  status.textContent += delegateBuildStatus;
  const renderedRows = rows.slice(0, 1000).map((op) => kernelRow(op, analysis, target, runtimeByOp.get(op.index), comparisonByOp.get(op.index), onSelect));
  if (filter === "selector" && renderedRows.length === 0) renderedRows.push(selectorEmptyRow(selectorAssessment));
  if (filter === "mismatch" && renderedRows.length === 0) renderedRows.push(runtimeMismatchEmptyRow(predictionApplicable));
  body.replaceChildren(...renderedRows);
}

function renderSelectorDecisionLedger(container, summaryText, analysis, predictionApplicable, onSelect, onLoadSourceEvidence) {
  const context = document.createElement("p");
  context.className = "kernel-inspector-context";
  context.textContent = summaryText;
  if (!predictionApplicable) {
    if (String(analysis?.format || "").toLowerCase() === "onnx") renderOrtSourceLedger(container, context, analysis, onSelect, onLoadSourceEvidence);
    else container.replaceChildren(context);
    return;
  }
  if (analysis?.xnnpack_selector_assessment_status !== "complete") {
    container.replaceChildren(context);
    return;
  }
  const provenance = analysis.xnnpack_selector_evidence_provenance || {};
  const details = document.createElement("details");
  details.className = "kernel-selector-ledger";
  details.open = true;
  const heading = document.createElement("summary");
  heading.textContent = "Source selector decision ledger";
  const metrics = document.createElement("div");
  metrics.className = "kernel-selector-ledger-metrics";
  const worstOps = (provenance.worst_case_tail_op_indices || []).map((index) => `#${padOp(index)}`).join(", ") || "none";
  for (const [label, value, tone = ""] of [
    ["Assessed", provenance.assessed_op_count || 0],
    ["Unique", provenance.unique_candidate_op_count || 0, "good"],
    ["Ambiguous", provenance.ambiguous_candidate_op_count || 0, Number(provenance.ambiguous_candidate_op_count || 0) ? "warn" : "good"],
    ["No match", provenance.no_match_op_count || 0, Number(provenance.no_match_op_count || 0) ? "warn" : "good"],
    ["Configurations", provenance.candidate_configuration_count || 0],
    ["Tail assessed", provenance.tail_assessed_op_count || 0],
    ["Worst tail", `${formatPercent(provenance.worst_case_tail_ratio || 0)} / ${worstOps}`, Number(provenance.worst_case_tail_ratio || 0) ? "warn" : "good"],
    ["Unresolved gates", `${provenance.unresolved_selector_dimension_count || 0} / ${provenance.unresolved_selector_op_count || 0} ops`, Number(provenance.unresolved_selector_dimension_count || 0) ? "warn" : "good"],
  ]) {
    const metric = document.createElement("div");
    metric.className = `kernel-selector-ledger-metric ${tone}`.trim();
    const name = document.createElement("span");
    name.textContent = label;
    const number = document.createElement("strong");
    number.textContent = String(value);
    metric.append(name, number);
    metrics.append(metric);
  }
  const hotspots = document.createElement("div");
  hotspots.className = "kernel-selector-hotspots";
  const rows = (analysis.ops || [])
    .filter((op) => String(op.xnnpack_kernel_evidence_class || "").startsWith("SOURCE_ENUMERATED"))
    .sort((left, right) => Number(right.channel_tail_overhead_percent_max || 0) - Number(left.channel_tail_overhead_percent_max || 0)
      || (right.xnnpack_kernel_candidates || []).length - (left.xnnpack_kernel_candidates || []).length
      || Number(left.index) - Number(right.index))
    .slice(0, 8);
  for (const op of rows) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "kernel-selector-hotspot";
    const identity = document.createElement("span");
    identity.textContent = `#${padOp(op.index)} ${op.name}`;
    const value = document.createElement("strong");
    value.textContent = (op.xnnpack_kernel_candidates || []).length
      ? `${formatPercent(op.channel_tail_overhead_percent_max || 0)} tail max`
      : "source no match";
    const detail = document.createElement("small");
    detail.textContent = `${(op.xnnpack_kernel_candidates || []).length} candidate(s) | C=${op.selector_artifact_facts?.output_channels || 0} | ${(op.unresolved_selector_dimensions || []).length} unresolved`;
    button.append(identity, value, detail);
    button.addEventListener("click", () => onSelect(op.index));
    hotspots.append(button);
  }
  details.append(heading, metrics, hotspots);
  container.replaceChildren(context, details);
}

function renderTfliteDelegateSourceLedger(container, analysis, runtimeEvidence, onSelect) {
  const evidence = analysis?.tflite_delegate_compatibility_evidence;
  if (!evidence) return;
  const details = document.createElement("details");
  details.className = "kernel-selector-ledger tflite-delegate-ledger";
  const heading = document.createElement("summary");
  heading.textContent = "TFLite GPU / NNAPI source candidates";
  const context = document.createElement("p");
  context.className = "kernel-selector-ledger-context";
  context.textContent = `TensorFlow ${evidence.tensorflow_source_commit.slice(0, 12)}... / rulepack ${evidence.rulepack_sha256.slice(0, 12)}.... Registration plus artifact precheck is not selected-build support or runtime assignment.`;
  const metrics = document.createElement("div");
  metrics.className = "kernel-selector-ledger-metrics";
  for (const profile of evidence.profiles || []) {
    for (const [label, value, tone] of [
      [`${profile.id === "tflite_gpu" ? "GPU" : "NNAPI"} candidates`, profile.source_candidate_after_artifact_precheck_count, ""],
      [`${profile.id === "tflite_gpu" ? "GPU" : "NNAPI"} exclusions`, profile.definite_exclusion_count, Number(profile.definite_exclusion_count || 0) ? "warn" : "good"],
    ]) {
      const metric = document.createElement("div");
      metric.className = `tflite-delegate-ledger-metric ${tone}`.trim();
      const name = document.createElement("span");
      name.textContent = label;
      const number = document.createElement("strong");
      number.textContent = String(value);
      metric.append(name, number);
      metrics.append(metric);
    }
  }
  const inventory = runtimeEvidence?.tflite_delegate_build_inventory || null;
  const build = document.createElement("p");
  build.className = "kernel-selector-ledger-context";
  build.textContent = inventory
    ? `Selected build: GPU ${inventory.gpu.compiled_status}; quant flag ${inventory.gpu.quantized_model_flag_status}; NNAPI ${inventory.nnapi.compiled_status}; feature level ${inventory.nnapi.runtime_feature_level ?? "not collected"}; accelerator ${inventory.nnapi.accelerator_identity || "not collected"}.`
    : "Selected build: not imported. GPU backend initialization, quant option, NNAPI effective Android gate, feature level, and accelerator identity remain unresolved.";
  const hotspots = document.createElement("div");
  hotspots.className = "kernel-selector-hotspots";
  const excluded = (evidence.profiles || []).flatMap((profile) => (profile.rows || [])
    .filter((row) => row.artifact_precheck_status === "definite_exclusion")
    .map((row) => ({ ...row, profile: profile.id })))
    .sort((left, right) => left.op_index - right.op_index || left.profile.localeCompare(right.profile))
    .slice(0, 8);
  for (const row of excluded) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "kernel-selector-hotspot";
    const identity = document.createElement("span");
    identity.textContent = `#${padOp(row.op_index)} ${row.op_name}`;
    const value = document.createElement("strong");
    value.textContent = row.profile === "tflite_gpu" ? "GPU excluded" : "NNAPI excluded";
    const detail = document.createElement("small");
    detail.textContent = (row.definite_exclusion_reasons || []).join("; ");
    button.append(identity, value, detail);
    button.addEventListener("click", () => onSelect(row.op_index));
    hotspots.append(button);
  }
  details.append(heading, context, metrics, build);
  if (excluded.length) details.append(hotspots);
  container.append(details);
}

function renderOrtSourceLedger(container, context, analysis, onSelect, onLoadSourceEvidence) {
  const providers = analysis?.ort_compatibility_evidence?.execution_providers || [];
  const inventory = analysis?.ort_compatibility_evidence?.source_condition_inventory || null;
  if (!providers.length) {
    const action = document.createElement("button");
    action.type = "button";
    action.className = "secondary-action";
    action.textContent = "Load source-backed EP analysis";
    action.disabled = typeof onLoadSourceEvidence !== "function";
    action.addEventListener("click", async () => {
      action.disabled = true;
      action.setAttribute("aria-busy", "true");
      await onLoadSourceEvidence?.();
      const focusTarget = container.querySelector(".kernel-selector-ledger > summary") || action;
      focusTarget.focus();
      action.removeAttribute("aria-busy");
    });
    container.replaceChildren(context, action);
    return;
  }
  const details = document.createElement("details");
  details.className = "kernel-selector-ledger";
  details.open = true;
  const heading = document.createElement("summary");
  heading.textContent = "ORT source compatibility ledger";
  const inventoryLine = document.createElement("p");
  inventoryLine.className = "kernel-selector-ledger-context";
  inventoryLine.textContent = inventory
    ? `${inventory.source_rule_count} pinned source rules / ${inventory.cpu_registration_variant_count} CPU type registration variants / ${inventory.machine_condition_count} machine-evaluated artifact conditions / ${inventory.versioned_scalar_schema_default_binding_count || 0} versioned scalar schema-default bindings / ${inventory.unresolved_source_fragment_count} source fragments unresolved. Remaining candidates are not support or assignment.`
    : "Source-condition extractor inventory is unavailable.";
  const metrics = document.createElement("div");
  metrics.className = "kernel-selector-ledger-metrics";
  for (const ep of providers) {
    const metric = document.createElement("div");
    metric.className = `kernel-selector-ledger-metric ${ep.artifact_precheck_definite_fail_op_count || ep.artifact_precheck_unresolved_op_count || ep.schema_kernel_version_no_match_count || ep.schema_version_unresolved_count || ep.source_rule_missing_count ? "warn" : "good"}`;
    const name = document.createElement("span");
    name.textContent = ep.execution_provider;
    const number = document.createElement("strong");
    number.textContent = `${ep.source_candidate_after_artifact_precheck_count}/${ep.assessed_op_count}`;
    const detail = document.createElement("small");
    detail.textContent = `source ${ep.schema_kernel_version_match_count}; pass/fail/unresolved ${ep.artifact_precheck_pass_op_count}/${ep.artifact_precheck_definite_fail_op_count}/${ep.artifact_precheck_unresolved_op_count}`;
    metric.append(name, number, detail);
    metrics.append(metric);
  }
  const hotspots = document.createElement("div");
  hotspots.className = "kernel-selector-hotspots";
  const rows = (analysis.ops || []).map((op) => ({ op, epRows: ortEpRowsForOp(analysis, op.index) }))
    .filter(({ epRows }) => epRows.some((row) => !row.source_candidate_after_artifact_precheck || row.artifact_precheck_status === "ARTIFACT_PRECHECK_UNRESOLVED"))
    .sort((left, right) => Number(right.epRows.some((row) => row.definite_source_exclusion)) - Number(left.epRows.some((row) => row.definite_source_exclusion)) || left.op.index - right.op.index)
    .slice(0, 8);
  for (const { op, epRows } of rows) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "kernel-selector-hotspot";
    const identity = document.createElement("span");
    identity.textContent = `#${padOp(op.index)} ${op.name}`;
    const value = document.createElement("strong");
    value.textContent = `${epRows.filter((row) => row.source_candidate_after_artifact_precheck).length}/${epRows.length} narrowed candidates`;
    const detail = document.createElement("small");
    detail.textContent = epRows.map((row) => `${row.execution_provider}:schema${row.resolved_schema_version ?? "?"}:${row.artifact_precheck_status}:${row.artifact_condition_fail_count || 0}F/${row.artifact_condition_unresolved_count || 0}U`).join(" | ");
    button.append(identity, value, detail);
    button.addEventListener("click", () => onSelect(op.index));
    hotspots.append(button);
  }
  details.append(heading, inventoryLine, metrics);
  if (rows.length) details.append(hotspots);
  container.replaceChildren(context, details);
}

function runtimeMismatchEmptyRow(predictionApplicable) {
  const row = document.createElement("tr");
  row.className = "kernel-empty-row";
  const cell = document.createElement("td");
  cell.colSpan = 9;
  cell.textContent = predictionApplicable
    ? "No imported runtime assignment differs from the applicable static prediction."
    : "Static execution-provider placement prediction is not defined for ONNX; mismatch classification is not applicable. Use All to inspect observed provider rows.";
  row.append(cell);
  return row;
}

function selectorEmptyRow(assessment) {
  const row = document.createElement("tr");
  row.className = "kernel-empty-row";
  const cell = document.createElement("td");
  cell.colSpan = 9;
  cell.textContent = assessment === "complete"
    ? "Protected selector assessment is complete; no unresolved candidate set or source-enumerated no-match remains in this view."
    : assessment === "not_loaded"
      ? "Source-backed selector evidence is not loaded. Run the controlled DEEPBOM module to enumerate pinned-source configuration candidates."
      : "Pinned-source selector enumeration is not available for this planning profile.";
  row.append(cell);
  return row;
}

function renderAssignmentComparison(container, comparison, predictedInventory, onSelect, timingProfile = null, nativeOrtCapture = null, backendLedger = null) {
  if (!container) return;
  container.hidden = !comparison;
  if (!comparison) { container.replaceChildren(); return; }
  const placement = comparison.placement_assessment || {};
  const boundaries = comparison.boundary_comparison || {};
  const observed = comparison.observed_boundary_inventory || {};
  const partitions = comparison.observed_partitions || {};
  const predictionApplicable = comparison.prediction_applicability === "tflite_xnnpack_static_prediction";
  const metrics = document.createElement("div");
  metrics.className = "runtime-comparison-metrics";
  const metricRows = predictionApplicable ? [
    ["Placement match", percentOrNA(placement.match_ratio), placement.mismatch_count ? "warn" : "good"],
    ["Classified ops", `${placement.assessed_op_count || 0}/${comparison.graph_op_count || 0}`, placement.unassessed_op_count ? "warn" : "good"],
    ["Conditionally delegatable / observed delegated", `${placement.predicted_delegated_op_count || 0} / ${placement.observed_delegated_op_count || 0}`, placement.mismatch_count ? "warn" : "neutral"],
    ["Boundary match", percentOrNA(boundaries.match_ratio), boundaries.mismatch_count ? "warn" : "good"],
    ["Observed partitions", String(partitions.partition_count || 0), partitions.noncontiguous_partition_id_count ? "warn" : "neutral"],
    ["Interface logical payload, predicted / observed", `${predictedInventory?.summed_edge_payload_bytes == null ? `${formatBytes(predictedInventory?.assessed_edge_payload_bytes || 0)} partial` : formatBytes(predictedInventory.summed_edge_payload_bytes)} / ${observed.summed_edge_payload_bytes == null ? `${formatBytes(observed.assessed_edge_payload_bytes || 0)} partial` : formatBytes(observed.summed_edge_payload_bytes)}`, "neutral"],
  ] : [
    ["Provider coverage", percentOrNA(placement.observed_assignment_coverage_ratio), placement.observed_assignment_count === comparison.graph_op_count ? "good" : "warn"],
    ["Mapped original ops", `${placement.observed_assignment_count || 0}/${comparison.graph_op_count || 0}`, placement.observed_assignment_count === comparison.graph_op_count ? "good" : "warn"],
    ["Provider-assigned / CPU", `${placement.observed_delegated_op_count || 0} / ${(placement.observed_assignment_count || 0) - (placement.observed_delegated_op_count || 0)}`, "neutral"],
    ["Observed relation edges", `${boundaries.observed_relation_edge_count || 0}/${boundaries.graph_edge_count || 0}`, "neutral"],
    ["Provider segments", String(partitions.provider_segment_count || 0), "neutral"],
    ["Observed transition payload", observed.summed_edge_payload_bytes == null ? `${formatBytes(observed.assessed_edge_payload_bytes || 0)} partial` : formatBytes(observed.summed_edge_payload_bytes), "neutral"],
  ];
  const duration = comparison.duration_comparison || {};
  if (timingProfile) {
    metricRows.push(
      ["Execution timing coverage", `${timingProfile.mapped_execution_node_count}/${timingProfile.execution_plan_node_count}`, timingProfile.execution_node_total_us == null ? "warn" : "good"],
      ["Execution-plan total", timingProfile.execution_node_total_us == null ? "Withheld" : formatUs(timingProfile.execution_node_total_us), timingProfile.execution_node_total_us == null ? "warn" : "good"],
      ["CPU / partition subtotal", `${duration.cpu_execution_node_subtotal_us == null ? "N/A" : formatUs(duration.cpu_execution_node_subtotal_us)} / ${duration.delegate_partition_subtotal_us == null ? "N/A" : formatUs(duration.delegate_partition_subtotal_us)}`, "neutral"],
      ["Primary delegate-profiled subtotal", duration.primary_delegate_profiled_subtotal_us == null ? "N/A" : `${formatUs(duration.primary_delegate_profiled_subtotal_us)} unassigned`, "neutral"],
      ["Nested delegate-internal subtotal", duration.delegate_internal_section_subtotal_us == null ? "N/A" : `${formatUs(duration.delegate_internal_section_subtotal_us)} non-additive`, "neutral"],
    );
  }
  for (const [label, value, tone] of metricRows) metrics.append(comparisonMetric(label, value, tone));
  const tracks = document.createElement("div");
  tracks.className = "runtime-comparison-tracks";
  if (predictionApplicable) tracks.append(assignmentTrack("Predicted eligibility", comparison.op_comparisons || [], (item) => item.predicted_delegated ? "delegate" : "cpu", onSelect));
  tracks.append(assignmentTrack(predictionApplicable ? "Observed" : "Provider", comparison.op_comparisons || [], (item) => item.observed_delegated == null ? "unknown" : item.observed_delegated ? "delegate" : "cpu", onSelect, predictionApplicable));
  const mismatchList = document.createElement("div");
  mismatchList.className = "runtime-mismatch-list";
  for (const item of (comparison.mismatches || []).slice(0, 24)) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "runtime-mismatch-chip";
    button.textContent = `#${padOp(item.op_index)} ${item.op_name}: ${item.classification}`;
    button.addEventListener("click", () => onSelect(item.op_index));
    mismatchList.append(button);
  }
  if ((comparison.mismatches || []).length > 24) {
    const more = document.createElement("span");
    more.className = "runtime-mismatch-more";
    more.textContent = `+${comparison.mismatches.length - 24} more`;
    mismatchList.append(more);
  }
  const timingList = timingProfile ? runtimeTimingHotspots(timingProfile, onSelect) : null;
  const nativeOrtGraph = nativeOrtCapture ? nativeOrtRuntimeGraph(nativeOrtCapture) : null;
  const backendEvidence = backendLedger ? runtimeBackendEvidenceSection(backendLedger) : null;
  container.replaceChildren(metrics, tracks, ...(backendEvidence ? [backendEvidence] : []), ...(nativeOrtGraph ? [nativeOrtGraph] : []), ...(timingList ? [timingList] : []), mismatchList);
}

function runtimeBackendEvidenceSection(ledger) {
  const section = document.createElement("section");
  section.className = "runtime-backend-ledger";
  const heading = document.createElement("h3");
  heading.textContent = "Selected runtime backend evidence";
  const note = document.createElement("p");
  note.textContent = "Build inclusion, capability acceptance, original-op assignment, and execution are independent. Empty later stages are not treated as rejection.";
  const grid = document.createElement("div");
  grid.className = "runtime-backend-ledger-grid";
  for (const row of ledger.providers) {
    const item = document.createElement("article");
    item.className = "runtime-backend-ledger-row";
    const title = document.createElement("strong");
    title.textContent = row.label;
    item.append(title,
      backendStage("Build", row.configured_inclusion.status, row.configured_inclusion.evidence_class),
      backendStage("Capability", row.capability_acceptance.status, row.capability_acceptance.evidence_class),
      backendStage("Assignment", `${row.assignment.status}${row.assignment.assigned_original_op_count ? ` (${row.assignment.assigned_original_op_count})` : ""}`, row.assignment.evidence_class),
      backendStage("Execution", `${row.execution.status}${row.execution.executed_original_op_count ? ` (${row.execution.executed_original_op_count})` : ""}`, row.execution.evidence_class));
    grid.append(item);
  }
  const digest = document.createElement("small");
  digest.textContent = `Ledger SHA-256 ${ledger.ledger_sha256}`;
  section.append(heading, note, grid, digest);
  return section;
}

function backendStage(label, status, evidenceClass) {
  const stage = document.createElement("span");
  stage.className = `runtime-backend-stage ${/observed|accepted/.test(status) ? "observed" : status === "not_assessed" || status === "not_observed" ? "missing" : "neutral"}`;
  const name = document.createElement("b");
  name.textContent = label;
  const value = document.createElement("span");
  value.textContent = String(status).replaceAll("_", " ");
  value.title = evidenceClass;
  stage.append(name, value);
  return stage;
}

function nativeOrtRuntimeGraph(capture) {
  const graph = capture.paired_profile_runtime_graph;
  const production = (graph?.profiles || []).find((item) => item.role === "production");
  if (!production) return null;
  const section = document.createElement("section");
  section.className = "runtime-timing-inventory native-ort-runtime-graph";
  const heading = document.createElement("h3");
  heading.textContent = "Observed production ORT runtime graph";
  const note = document.createElement("p");
  note.textContent = `${production.runtime_node_count} optimized/fused runtime node(s), ${production.kernel_event_count} kernel event(s), optimization ${production.graph_optimization_level}. Provider and duration are observed; original-op mapping is not inferred.`;
  const list = document.createElement("div");
  list.className = "runtime-timing-list";
  for (const node of production.nodes || []) {
    const item = document.createElement("div");
    item.className = "runtime-timing-row native-ort-node";
    const identity = document.createElement("span");
    identity.textContent = `#${node.runtime_node_index} ${node.runtime_node_name} (${node.op_name})`;
    identity.title = identity.textContent;
    const provider = document.createElement("strong");
    provider.textContent = node.provider;
    const detail = document.createElement("small");
    detail.textContent = `${formatUs(node.duration_mean_us)} mean / ${node.sample_count} sample(s)`;
    item.append(identity, provider, detail);
    list.append(item);
  }
  const output = capture.paired_profile_output_comparison;
  const outputNote = document.createElement("p");
  outputNote.className = "native-ort-output-comparison";
  outputNote.textContent = output?.outputs?.map((row) => row.numeric_comparison_status === "assessed"
    ? `${row.name}: max abs ${formatDrift(row.max_abs_error)}, RMS ${formatDrift(row.rms_error)}, relative L2 ${formatDrift(row.relative_l2_error)}, cosine distance ${formatDrift(row.cosine_distance)}`
    : `${row.name}: ${row.numeric_comparison_status}`).join(" / ") || "Paired output comparison not assessed.";
  section.append(heading, note, list, outputNote);
  return section;
}

function runtimeTimingHotspots(timingProfile, onSelect) {
  const section = document.createElement("section");
  section.className = "runtime-timing-inventory";
  const heading = document.createElement("h3");
  heading.textContent = "Observed latency hotspots";
  const note = document.createElement("p");
  note.textContent = "Execution-node rows are additive only at complete common-run coverage. Primary delegate-profiled events remain unassigned; nested delegate-section events are non-additive to partition totals.";
  const list = document.createElement("div");
  list.className = "runtime-timing-list";
  const rows = [
    ...(timingProfile.execution_nodes || []).map((row) => ({ ...row, evidence_group: row.node_kind })),
    ...(timingProfile.delegate_internal_events || []).map((row) => ({ ...row, evidence_group: "delegate_internal" })),
  ].sort((left, right) => Number(right.mean_per_run_us ?? -1) - Number(left.mean_per_run_us ?? -1)).slice(0, 16);
  for (const row of rows) {
    const item = document.createElement(row.op_index == null ? "div" : "button");
    if (row.op_index != null) {
      item.type = "button";
      item.addEventListener("click", () => onSelect(row.op_index));
    }
    item.className = `runtime-timing-row ${row.evidence_group}`;
    const identity = document.createElement("span");
    identity.textContent = row.evidence_group === "original_op"
      ? `#${padOp(row.op_index)} ${row.node_type}`
      : row.evidence_group === "delegate_partition"
        ? `${row.partition_id} ${row.provider}`
        : row.node_kind === "primary_subgraph_delegate_profiled_event"
          ? `Primary delegate-profiled ${row.node_type}`
          : `Nested delegate internal ${row.node_type}`;
    const duration = document.createElement("strong");
    duration.textContent = row.mean_per_run_us == null ? "N/A" : formatUs(row.mean_per_run_us);
    const detail = document.createElement("small");
    detail.textContent = row.run_count == null
      ? `${row.event_sample_count} event(s); run count not derivable`
      : `${row.run_count} run(s), formatter times_called ${row.formatter_times_called_integer_average} (integer average)`;
    item.append(identity, duration, detail);
    list.append(item);
  }
  section.append(heading, note, list);
  return section;
}

function comparisonMetric(label, value, tone) {
  const item = document.createElement("div");
  item.className = `runtime-comparison-metric ${tone}`;
  const name = document.createElement("span");
  name.textContent = label;
  const number = document.createElement("strong");
  number.textContent = value;
  item.append(name, number);
  return item;
}

function assignmentTrack(label, rows, stateFor, onSelect, markMismatch = false) {
  const row = document.createElement("div");
  row.className = "runtime-assignment-track-row";
  const name = document.createElement("span");
  name.textContent = label;
  const track = document.createElement("div");
  track.className = "runtime-assignment-track";
  for (const item of rows) {
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = `runtime-assignment-cell ${stateFor(item)}${markMismatch && item.matches_prediction === false ? " mismatch" : ""}`;
    cell.title = `#${item.op_index} ${item.op_name}: ${label.toLowerCase()} ${stateFor(item)}${item.classification ? ` / ${item.classification}` : ""}`;
    cell.setAttribute("aria-label", cell.title);
    cell.addEventListener("click", () => onSelect(item.op_index));
    track.append(cell);
  }
  row.append(name, track);
  return row;
}

function renderBoundaryInventory(container, inventory, comparison, filter, onSelect) {
  if (!container) return;
  const predictedEdges = inventory?.edges || [];
  const observedEdges = comparison?.observed_boundary_inventory?.edges || [];
  const differences = comparison?.boundary_differences || [];
  container.hidden = filter !== "boundary" || (!predictedEdges.length && !observedEdges.length);
  if (container.hidden) {
    container.replaceChildren();
    return;
  }
  const predictionApplicable = comparison?.prediction_applicability !== "not_applicable_for_onnx_execution_provider_assignment";
  container.replaceChildren(
    ...(predictionApplicable ? boundarySection("Predicted internal execution-domain edges", `${inventory?.assignment_evidence_class || "PREDICTED"} assignment / ${inventory?.payload_evidence_class || "DERIVED"} logical payload`, predictedEdges, false, onSelect) : []),
    ...boundarySection("Observed internal execution-domain edges", "DERIVED_FROM_OBSERVED_RUNTIME / logical payload / materialization not assessed", observedEdges, true, onSelect),
    ...(differences.length ? boundarySection("Prediction boundary deltas", `${differences.length} graph edge mismatch(es)`, differences, true, onSelect) : []),
  );
}

function boundarySection(titleText, evidenceText, edges, observed, onSelect) {
  if (!edges.length) return [];
  const heading = document.createElement("div");
  heading.className = "kernel-boundary-heading";
  const title = document.createElement("strong");
  title.textContent = titleText;
  const evidence = document.createElement("span");
  evidence.textContent = evidenceText;
  heading.append(title, evidence);
  const rows = edges.map((edge) => {
    const button = document.createElement("button");
    button.className = "kernel-boundary-edge";
    button.type = "button";
    const identity = document.createElement("span");
    identity.className = "kernel-boundary-identity";
    const tensor = document.createElement("strong");
    tensor.textContent = `T${edge.tensor_index} ${edge.tensor_name || "unnamed tensor"}`;
    const transition = document.createElement("span");
    const producerDomain = observed ? edge.observed_producer_domain : edge.producer_domain;
    const consumerDomain = observed ? edge.observed_consumer_domain : edge.consumer_domain;
    transition.textContent = `#${padOp(edge.producer_op_index)} ${edge.producer_op_name} [${producerDomain || "unassessed"}] -> #${padOp(edge.consumer_op_index)} ${edge.consumer_op_name} [${consumerDomain || "unassessed"}]`;
    identity.append(tensor, transition);

    const metadata = document.createElement("span");
    metadata.className = "kernel-boundary-metadata";
    const payload = document.createElement("strong");
    payload.textContent = edge.payload_bytes == null ? "payload not assessed" : formatBytes(edge.payload_bytes);
    const shape = Array.isArray(edge.tensor_shape) ? `[${edge.tensor_shape.join("x") || "scalar"}]` : "shape unknown";
    const detail = document.createElement("span");
    detail.textContent = `${shape} ${edge.tensor_dtype || "UNKNOWN"} / ${edge.classification || edge.direction || "boundary"} / materialization ${edge.materialization_status || "not_assessed"}`;
    metadata.append(payload, detail);
    button.append(identity, metadata);
    button.setAttribute("aria-label", `Inspect consumer op ${edge.consumer_op_index} for boundary tensor ${edge.tensor_index}`);
    button.addEventListener("click", () => onSelect(edge.consumer_op_index));
    return button;
  });
  return [heading, ...rows];
}

function kernelRow(op, analysis, target, runtime, compared, onSelect) {
  const tr = document.createElement("tr");
  tr.className = "clickable-row";
  if (compared?.matches_prediction === false) tr.classList.add("runtime-assignment-mismatch");
  const format = String(analysis?.format || "tflite").toLowerCase();
  const predictionApplicable = format === "tflite";
  const coreMl = format === "coreml";
  const path = op.quantized_compute_path ? `${opPrecisionLabel(op, analysis)} compute` : op.quantized_path ? "quant signal" : `${opPrecisionLabel(op, analysis)} path`;
  const alignmentMultiples = op.xnnpack_kernel_alignment_multiples || [];
  const alignmentLabel = alignmentMultiples.length ? alignmentMultiples.map((value) => `x${value}`).join("/") : `x${op.channel_alignment_multiple || "?"}`;
  const tailMin = Number(op.channel_tail_overhead_percent_min ?? op.channel_tail_overhead_percent ?? 0);
  const tailMax = Number(op.channel_tail_overhead_percent_max ?? op.channel_tail_overhead_percent ?? 0);
  const tail = op.output_channels > 0 && (alignmentMultiples.length || op.channel_alignment_multiple > 0)
    ? `C=${op.output_channels} / ${alignmentLabel}; modeled occupancy ${formatPercentRange(1 - tailMax, 1 - tailMin)}`
    : "not applicable";
  const predicted = !predictionApplicable
    ? coreMl ? "serialized operation; native placement unobserved" : "static EP assignment not assessed"
    : op.xnnpack_chain_break
    ? `boundary: ${op.xnnpack_break_class || "unknown"}`
    : op.xnnpack_chain_id >= 0 ? `conditionally delegatable C${op.xnnpack_chain_id}` : "predicted fallback";
  const boundaryPayload = predictionApplicable ? predictedPartitionBoundaryPayloadForOp(analysis, op.index) : { edge_count: 0 };
  const packing = !predictionApplicable
    ? coreMl ? "runtime specialization not exposed" : "not assessed for ONNX EP"
    : Number(op.weight_packing_overhead_us || 0) > 0
    ? `${formatUs(op.weight_packing_overhead_us)} ESTIMATED`
    : "not applicable";
  const runtimeText = runtime
    ? runtime.evidence_kind === "coreml_compute_plan"
      ? `anticipated ${runtime.preferred_compute_device || "not determined"}; supported ${runtime.supported_compute_devices.join(" / ") || "not determined"}; cost ${runtime.estimated_cost_weight == null ? "not determined" : Number(runtime.estimated_cost_weight).toFixed(9)}; NOT EXECUTED`
      : `${runtime.provider}${runtime.partition_id != null ? ` / P${runtime.partition_id}` : ""}${runtime.lowering_id ? ` / lowering ${runtime.lowering_id}` : ""}${runtime.kernel ? ` / ${runtime.kernel_id || "kernel"}:${runtime.kernel}` : ""}${runtime.duration_us != null ? ` / ${formatUs(runtime.duration_us)}` : ""}${compared ? ` / ${compared.classification}` : ""}`
    : "not imported";
  const sourceEvidence = op.xnnpack_kernel_evidence_class || "HEURISTIC_PROFILE";
  const evidence = !predictionApplicable
    ? coreMl
      ? runtime ? "COREML_COMPUTE_PLAN_ESTIMATE / execution NOT_OBSERVED / microkernel NOT_ASSESSABLE" : "SERIALIZED_COREML_OPERATION / compute plan NOT_IMPORTED / execution NOT_OBSERVED"
      : runtime ? "OBSERVED_RUNTIME / source EP compatibility separate / microkernel NOT_ASSESSABLE" : ortEpRowsForOp(analysis, op.index).length ? "SOURCE_AND_ARTIFACT_PRECHECK_ONLY / runtime assignment NOT_OBSERVED" : "runtime provider NOT_ASSESSED / static EP prediction NOT_ASSESSABLE"
    : runtime ? `OBSERVED_RUNTIME + ${sourceEvidence}` : `PREDICTED + ${sourceEvidence} / ESTIMATED`;
  const candidates = op.xnnpack_kernel_candidates || [];
  const tile = !predictionApplicable ? coreMl ? "not exposed by MLComputePlan" : "not exposed by imported profile" : kernelCandidateTilesLabel(candidates) || "tile not source-enumerated";
  const ortRows = format === "onnx" ? ortEpRowsForOp(analysis, op.index) : [];
  const candidateLabel = !predictionApplicable
    ? coreMl
      ? `${op.mil_operation_type || op.name} serialized Core ML operation`
      : ortRows.length
      ? ortRows.map((row) => {
        const issue = (row.artifact_conditions || []).find((condition) => condition.status !== "PASS");
        return `${row.execution_provider}:opset ${row.imported_opset ?? "?"} -> schema v${row.resolved_schema_version ?? "?"} ${row.artifact_precheck_status.toLowerCase()}${issue ? ` (${issue.condition_id}: ${issue.observed || issue.reason})` : ""}`;
      }).join(" / ")
      : "ORT EP kernel selection not statically modeled"
    : candidates.length
    ? `${candidates.length} configuration${candidates.length === 1 ? "" : "s"}: ${candidates[0].family}${candidates.length > 1 ? ` +${candidates.length - 1}` : ""}`
    : op.xnnpack_kernel_candidate || op.target_microkernel_hint || target.xnnpack_kernel_family || "-";
  appendCells(tr, [
    `#${padOp(op.index)}`,
    `${op.name} | ${path}`,
    `${candidateLabel} | ${predictionApplicable ? compactKernelSource(candidates[0]?.source_ref || op.xnnpack_kernel_source) : coreMl ? "pinned Core ML schema; runtime plan separate" : ortRows.length ? "pinned ORT source tables" : "runtime profile only"}`,
    `${tile} | ${predictionApplicable ? tail : "occupancy not assessed"}`,
    predictionApplicable ? `${predicted} | ${op.xnnpack_reason || "no rule code"}: ${decodeXnnpReason(op.xnnpack_reason || "") || "no rule explanation"}` : predicted,
    !predictionApplicable
      ? coreMl ? "runtime transfer materialization not exposed" : "see observed transition inventory"
      : !boundaryPayload.edge_count
        ? "-"
      : boundaryPayload.unassessed_edge_count
        ? `${formatBytes(boundaryPayload.assessed_bytes)} assessed / ${boundaryPayload.unassessed_edge_count} unknown edge(s)`
        : `${formatBytes(boundaryPayload.assessed_bytes)} / ${boundaryPayload.edge_count} edge(s)`,
    packing,
    runtimeText,
    `${evidence} | ${coreMl && runtime ? `artifact ${(analysis.model_sha256 || "").slice(0, 12) || "bound"}` : `profile ${(target.profile_sha256 || "").slice(0, 12) || "unbound"}`}`,
  ]);
  tr.tabIndex = 0;
  tr.setAttribute("role", "button");
  tr.setAttribute("aria-label", `Inspect op ${op.index} ${op.name}`);
  tr.addEventListener("click", () => onSelect(op.index));
  tr.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    onSelect(op.index);
  });
  return tr;
}

function ortEpRowsForOp(analysis, opIndex) {
  return (analysis?.ort_compatibility_evidence?.execution_providers || []).flatMap((ep) => {
    const row = (ep.ops || []).find((item) => Number(item.op_index) === Number(opIndex));
    return row ? [{ ...row, execution_provider: ep.execution_provider }] : [];
  });
}

function kernelCandidateTilesLabel(candidates = []) {
  const labels = [...new Set(candidates.map((candidate) => {
    if (Number(candidate.tile_nr || 0) > 0) return `MR<=${candidate.tile_mr || "?"} x NR${candidate.tile_nr}`;
    if (Number(candidate.channel_tile || 0) > 0) return `${candidate.primary_tile || "?"}p${candidate.channel_tile}c`;
    return null;
  }).filter(Boolean))];
  return labels.join(" / ");
}

function percentOrNA(value) {
  return value == null ? "N/A" : formatPercent(value);
}

function compactKernelSource(source) {
  if (!source) return "profile-only";
  const match = String(source).match(/google\/XNNPACK@([a-f0-9]+)\/src\/configs\/(.+)$/i);
  return match ? `XNNPACK@${match[1].slice(0, 12)}/${match[2]}` : String(source);
}

function appendCells(row, values) {
  for (const value of values) {
    const cell = document.createElement("td");
    cell.textContent = value;
    row.append(cell);
  }
}
