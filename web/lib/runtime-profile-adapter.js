import { sha256Hex } from "./hash.js";
import { canonicalJson } from "./report-utils.js";
import { validateOrtBuildAttestation } from "./ort-build-attestation.js";
import { buildOrtSelectedBuildProviderBinding } from "./ort-selected-build-binding.js";
import { parseOrtReducedOperatorConfig } from "./ort-reduced-operator-config.js";

const ORT_PROFILE_ADAPTER_SCHEMA = "deepbom.ort_profile_adapter.v2.2";
const ORT_NATIVE_PROFILE_SCHEMA = "deepbom.ort_native_profile.v1.4";
const ORT_RUNTIME_VERSION = "1.26.0";
const ORT_SOURCE_COMMIT = "8c546c37b43caaca1fa25db430dab94b901cf277";
const ORT_PACKAGE_INTEGRITY = "sha512-OHl6PiOEOqxaLHL0N9eFrbzS7IGmu3BtJNH3RTEnRAheCIkfc3gjcjl4sGcjp9C22ZC9YTquDOxSdT/stBQ6BQ==";
const ORT_KERNEL_SUFFIX = "_kernel_time";
const MAX_PROFILE_EVENTS = 1_000_000;
const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const ORT_PROFILE_DTYPE_MAP = Object.freeze({
  float: "FLOAT32",
  double: "FLOAT64",
  float16: "FLOAT16",
  bfloat16: "BFLOAT16",
  bool: "BOOL",
  string: "STRING",
  int8: "INT8",
  int8_t: "INT8",
  uint8: "UINT8",
  uint8_t: "UINT8",
  int16: "INT16",
  int16_t: "INT16",
  uint16: "UINT16",
  uint16_t: "UINT16",
  int32: "INT32",
  int32_t: "INT32",
  uint32: "UINT32",
  uint32_t: "UINT32",
  int64: "INT64",
  int64_t: "INT64",
  uint64: "UINT64",
  uint64_t: "UINT64",
  complex64: "COMPLEX64",
  complex128: "COMPLEX128",
});

export const ORT_PROFILE_SOURCE = Object.freeze({
  runtime_version_basis: "ONNX Runtime 1.26.0",
  source_commit: "microsoft/onnxruntime@8c546c37b43caaca1fa25db430dab94b901cf277",
  source_file: "onnxruntime/core/framework/sequential_executor.cc",
  source_ref: "https://github.com/microsoft/onnxruntime/blob/8c546c37b43caaca1fa25db430dab94b901cf277/onnxruntime/core/framework/sequential_executor.cc",
  event_fields: [
    "args.op_name", "args.provider", "args.node_index", "dur", "name",
    "args.input_type_shape", "args.output_type_shape", "args.activation_size",
    "args.parameter_size", "args.output_size",
  ],
});

export function isCanonicalRuntimeAssignment(source) {
  return /^deepbom\.runtime_assignment\.v1(?:\.\d+)?$/.test(String(source?.schema || ""));
}

export function parseRuntimeProfileSource(text, analysis) {
  let source;
  try {
    source = JSON.parse(String(text || ""));
  } catch {
    throw new Error("Runtime evidence file is not valid JSON.");
  }
  if (isCanonicalRuntimeAssignment(source)) return { kind: "canonical", source };
  if (String(analysis?.format || "").toLowerCase() !== "onnx") {
    throw new Error("Raw ONNX Runtime profiles can only be mapped to an active ONNX artifact.");
  }
  const nativeCaptureEnvelope = source?.schema === ORT_NATIVE_PROFILE_SCHEMA ? source : null;
  if (nativeCaptureEnvelope) {
    if (typeof nativeCaptureEnvelope.profile?.json !== "string") throw new Error("Native ORT capture envelope does not contain embedded profile JSON.");
    try { source = JSON.parse(nativeCaptureEnvelope.profile.json); } catch { throw new Error("Native ORT embedded profile is not valid JSON."); }
  }
  const events = Array.isArray(source) ? source : source?.traceEvents;
  if (!Array.isArray(events)) {
    throw new Error("Runtime evidence must be a DeepBOM assignment document or ONNX Runtime Chrome-trace JSON.");
  }
  if (events.length > MAX_PROFILE_EVENTS) {
    throw new Error(`ONNX Runtime profile exceeds ${MAX_PROFILE_EVENTS.toLocaleString("en-US")} events.`);
  }
  const runtimeEvents = events.flatMap((event, sourceIndex) => {
    if (!event || typeof event !== "object" || String(event.cat || "").toLowerCase() !== "node") return [];
    const eventName = String(event.name || "");
    const args = event.args && typeof event.args === "object" ? event.args : {};
    const opName = String(args.op_name || "").trim();
    const provider = String(args.provider || "").trim();
    const runtimeNodeIndex = strictNonNegativeInteger(args.node_index);
    const durationUs = Number(event.dur);
    if (!eventName.endsWith(ORT_KERNEL_SUFFIX) || !opName || !provider || runtimeNodeIndex == null) return [];
    if (!Number.isFinite(durationUs) || durationUs < 0) return [];
    const runtimeTensorObservation = parseOrtRuntimeTensorObservation(args, sourceIndex);
    return [{
      source_event_index: sourceIndex,
      runtime_node_name: eventName.slice(0, -ORT_KERNEL_SUFFIX.length),
      runtime_node_index: runtimeNodeIndex,
      op_name: opName,
      provider,
      duration_us: durationUs,
      runtime_tensor_observation: runtimeTensorObservation,
    }];
  });
  if (!runtimeEvents.length) {
    throw new Error("ONNX Runtime profile contains no valid Node/*_kernel_time events with provider, op_name, node_index, and duration.");
  }
  return {
    kind: "onnxruntime_profile",
    schema: ORT_PROFILE_ADAPTER_SCHEMA,
    source_event_count: events.length,
    kernel_event_count: runtimeEvents.length,
    runtime_events: runtimeEvents,
    providers: [...new Set(runtimeEvents.map((event) => event.provider))].sort(),
    source_basis: ORT_PROFILE_SOURCE,
    native_capture_envelope: nativeCaptureEnvelope,
  };
}

export async function verifyOrtNativeCaptureProfile(profile, analysis) {
  const envelope = profile?.native_capture_envelope;
  if (!envelope) return null;
  if (profile.kind !== "onnxruntime_profile" || envelope.schema !== ORT_NATIVE_PROFILE_SCHEMA) throw new Error("Native ORT capture profile is invalid.");
  const { capture_content_sha256: declaredContentSha, ...content } = envelope;
  const calculatedContentSha = await sha256Hex(new TextEncoder().encode(canonicalJson(content)));
  if (!SHA256_PATTERN.test(String(declaredContentSha || "")) || declaredContentSha !== calculatedContentSha) throw new Error("Native ORT capture envelope content SHA-256 is invalid.");
  const profileSha256 = await sha256Hex(new TextEncoder().encode(envelope.profile.json));
  if (profileSha256 !== envelope.profile.sha256) throw new Error("Native ORT embedded profile SHA-256 is invalid.");
  if (!SHA256_PATTERN.test(String(envelope.artifact?.sha256 || "")) || envelope.artifact.sha256 !== analysis?.model_sha256) {
    throw new Error("Native ORT capture is not bound to the active ONNX artifact SHA-256.");
  }
  await verifyNativeOrtArtifactContentSet(envelope.artifact, analysis);
  const runtime = envelope.runtime || {};
  if (runtime.name !== "ONNX Runtime Node.js" || runtime.version !== ORT_RUNTIME_VERSION
    || runtime.package_name !== "onnxruntime-node"
    || !["NPM_PACKAGE_LOCK_ATTESTED", "SOURCE_BUILD_ATTESTED"].includes(runtime.distribution_identity)
    || (runtime.distribution_identity === "NPM_PACKAGE_LOCK_ATTESTED"
      ? runtime.package_integrity !== ORT_PACKAGE_INTEGRITY || runtime.build_attestation != null
      : runtime.package_integrity != null)
    || runtime.source_commit !== ORT_SOURCE_COMMIT || runtime.node_napi !== "napi-v6"
    || !SHA256_PATTERN.test(String(runtime.package_manifest_sha256 || ""))
    || !SHA256_PATTERN.test(String(runtime.primary_binary_sha256 || ""))
    || !SHA256_PATTERN.test(String(runtime.binary_inventory_sha256 || ""))) {
    throw new Error("Native ORT capture runtime/package identity does not match the pinned collector contract.");
  }
  if (!Array.isArray(runtime.binary_inventory) || !runtime.binary_inventory.length
    || runtime.binary_inventory.some((file) => !safeRelativePath(file?.path) || !Number.isSafeInteger(file?.byte_length) || file.byte_length < 1 || !SHA256_PATTERN.test(String(file?.sha256 || "")))) {
    throw new Error("Native ORT capture binary inventory is invalid.");
  }
  const inventorySha = await sha256Hex(new TextEncoder().encode(canonicalJson(runtime.binary_inventory)));
  if (inventorySha !== runtime.binary_inventory_sha256
    || !runtime.binary_inventory.some((file) => file.path === runtime.primary_binary_path && file.sha256 === runtime.primary_binary_sha256)) {
    throw new Error("Native ORT capture binary inventory digest or primary binary binding is invalid.");
  }
  const supportedBackends = runtime.supported_backends;
  const requestedProviders = runtime.requested_execution_providers;
  if (!Array.isArray(supportedBackends) || !supportedBackends.length
    || supportedBackends.some((backend) => !backend?.name || typeof backend.bundled !== "boolean")
    || new Set(supportedBackends.map((backend) => backend.name)).size !== supportedBackends.length
    || canonicalJson([...supportedBackends].sort((left, right) => left.name.localeCompare(right.name))) !== canonicalJson(supportedBackends)
    || !SHA256_PATTERN.test(String(runtime.supported_backends_sha256 || ""))
    || runtime.supported_backends_sha256 !== await sha256Hex(new TextEncoder().encode(canonicalJson(supportedBackends)))
    || runtime.provider_inventory_status !== "OBSERVED_FROM_ORT_LIST_SUPPORTED_BACKENDS"
    || !["NOT_EXPOSED_BY_ONNXRUNTIME_NODE_API_NOT_INFERRED", "IMPORTED_CONFIG_NOT_BINARY_ATTESTED", "BUILD_INPUT_BINARY_ATTESTED"].includes(runtime.reduced_operator_inventory_status)
    || !Array.isArray(requestedProviders) || !requestedProviders.length || new Set(requestedProviders).size !== requestedProviders.length
    || requestedProviders.some((name) => !supportedBackends.some((backend) => backend.name === name))) {
    throw new Error("Native ORT selected-build provider inventory is invalid.");
  }
  if (runtime.distribution_identity === "SOURCE_BUILD_ATTESTED") {
    const attestation = validateOrtBuildAttestation(runtime.build_attestation);
    if (attestation.runtime_package.package_manifest_sha256 !== runtime.package_manifest_sha256
      || attestation.runtime_package.binary_inventory_sha256 !== runtime.binary_inventory_sha256
      || attestation.runtime_package.primary_binary_sha256 !== runtime.primary_binary_sha256
      || canonicalJson(attestation.runtime_package.binary_inventory) !== canonicalJson(runtime.binary_inventory)) {
      throw new Error("Native ORT source-build attestation does not bind the imported runtime package inventory.");
    }
  }
  await validateReducedOperatorConfigIdentity(runtime.reduced_operator_config, runtime.reduced_operator_inventory_status);
  const role = String(envelope.profile_role || "");
  const expectedOptimization = role === "identity" ? "disabled" : role === "production" ? "all" : null;
  if (!expectedOptimization || envelope.invocation?.graph_optimization_level !== expectedOptimization || envelope.invocation?.execution_mode !== "sequential") {
    throw new Error("Native ORT capture role/session options are inconsistent.");
  }
  if (envelope.profile.source_event_count !== profile.source_event_count
    || envelope.profile.kernel_event_count !== profile.kernel_event_count
    || canonicalJson(envelope.profile.observed_providers) !== canonicalJson(profile.providers)) {
    throw new Error("Native ORT capture event/provider inventory is inconsistent.");
  }
  if (!Array.isArray(envelope.invocation.inputs) || !envelope.invocation.inputs.length
    || envelope.invocation.inputs.some((input) => !input?.name || !input.type || !Array.isArray(input.shape)
      || input.shape.some((dim) => !Number.isSafeInteger(dim) || dim < 1) || !SHA256_PATTERN.test(String(input.data_sha256 || "")))) {
    throw new Error("Native ORT capture input inventory is invalid.");
  }
  validateNativeOutputEvidence(envelope);
  const collectedAt = String(envelope.collection?.collected_at || "");
  if (!Number.isFinite(Date.parse(collectedAt))) throw new Error("Native ORT capture timestamp is invalid.");
  const selectedBuildProviderBinding = buildOrtSelectedBuildProviderBinding(analysis, runtime);
  const build = [
    `onnxruntime-node@${ORT_RUNTIME_VERSION}`,
    `npm-integrity ${ORT_PACKAGE_INTEGRITY}`,
    `source ${ORT_SOURCE_COMMIT}`,
    `${runtime.platform}/${runtime.arch} ${runtime.node_napi}`,
    `EP-order ${requestedProviders.join(",")}`,
    `binary-inventory ${runtime.binary_inventory_sha256}`,
    `artifact-content-set ${envelope.artifact.content_set_sha256}`,
    `profile-role ${role}`,
  ].join("; ");
  return {
    profileSha256,
    metadata: {
      runtimeVersion: ORT_RUNTIME_VERSION,
      backend: profile.providers.join(" + "),
      runtimeBuild: build,
      binarySha256: runtime.primary_binary_sha256,
      graphOptimizationLevel: expectedOptimization,
      executionMode: "sequential",
      collectedAt,
      captureId: String(envelope.capture_id || ""),
    },
    evidence: {
      schema: ORT_NATIVE_PROFILE_SCHEMA,
      capture_id: String(envelope.capture_id || ""),
      capture_content_sha256: declaredContentSha,
      profile_role: role,
      mapping_intent: String(envelope.mapping_intent || ""),
      artifact: structuredCloneSafe(envelope.artifact),
      runtime: structuredCloneSafe(runtime),
      invocation: structuredCloneSafe(envelope.invocation),
      collection: structuredCloneSafe(envelope.collection),
      profile: {
        sha256: profileSha256,
        source_event_count: profile.source_event_count,
        kernel_event_count: profile.kernel_event_count,
        observed_providers: [...profile.providers],
      },
      output_observations: structuredCloneSafe(envelope.output_observations),
      paired_profile_output_comparison: structuredCloneSafe(envelope.paired_profile_output_comparison),
      paired_profile_runtime_graph: structuredCloneSafe(envelope.paired_profile_runtime_graph),
      selected_build_provider_binding: selectedBuildProviderBinding,
      artifact_binding: "BROWSER_VERIFIED_ACTIVE_ONNX_AND_EXTERNAL_DATA_CONTENT_SET_SHA256",
      profile_binding: "BROWSER_VERIFIED_EMBEDDED_PROFILE_SHA256",
      envelope_binding: "BROWSER_VERIFIED_CANONICAL_CONTENT_SHA256",
      runtime_identity_semantics: "CAPTURE_COLLECTOR_OBSERVED_BROWSER_NOT_REHASHED",
      assignment_evidence_class: "OBSERVED_RUNTIME",
      interpretation_boundary: String(envelope.evidence_boundary || ""),
    },
  };
}

async function validateReducedOperatorConfigIdentity(identity, status) {
  if (status === "NOT_EXPOSED_BY_ONNXRUNTIME_NODE_API_NOT_INFERRED") {
    if (identity != null) throw new Error("Native ORT reduced-operator config contradicts its inventory status.");
    return;
  }
  if (!identity || identity.schema !== "deepbom.ort_reduced_operator_config_identity.v1"
    || !identity.source_name || /[\\/]/.test(identity.source_name)
    || !SHA256_PATTERN.test(String(identity.source_sha256 || ""))
    || !SHA256_PATTERN.test(String(identity.normalized_sha256 || ""))
    || (status === "BUILD_INPUT_BINARY_ATTESTED"
      ? identity.binary_binding_status !== "ATTESTED_OBSERVED_BUILD_INPUT_BOUND_TO_SELECTED_BINARY_INVENTORY"
      : identity.binary_binding_status !== "NOT_ATTESTED_CONFIG_INPUT_NOT_OBSERVED_FROM_SELECTED_BINARY")
    || typeof identity.source_text !== "string") {
    throw new Error("Native ORT reduced-operator config identity is invalid.");
  }
  const sourceSha = await sha256Hex(new TextEncoder().encode(identity.source_text));
  const reparsed = parseOrtReducedOperatorConfig(identity.source_text);
  const normalizedSha = await sha256Hex(new TextEncoder().encode(canonicalJson(reparsed)));
  if (sourceSha !== identity.source_sha256 || normalizedSha !== identity.normalized_sha256
    || canonicalJson(reparsed) !== canonicalJson(identity.normalized_config)) {
    throw new Error("Native ORT reduced-operator config digest or normalization is invalid.");
  }
}

export function previewOrtProfileMapping(profile, analysis, metadata = {}) {
  if (profile?.kind !== "onnxruntime_profile") throw new Error("Expected a parsed ONNX Runtime profile.");
  const optimizationLevel = normalizedMode(metadata.graphOptimizationLevel, ["disabled", "basic", "extended", "all", "unknown"], "unknown");
  const opByName = new Map();
  for (const op of analysis?.ops || []) {
    const graphName = String(op.graph_node_name || "");
    if (!graphName) continue;
    const key = nodeKey(graphName, op.name);
    if (!opByName.has(key)) opByName.set(key, []);
    opByName.get(key).push(op);
  }
  const eventGroups = groupProfileEvents(profile.runtime_events);
  const proposed = [];
  const unresolved = [];
  for (const group of eventGroups) {
    const exactNameMatches = opByName.get(nodeKey(group.runtime_node_name, group.op_name)) || [];
    let op = exactNameMatches.length === 1 ? exactNameMatches[0] : null;
    let mappingMethod = op ? "exact_graph_node_name_and_op_type" : null;
    if (!op && optimizationLevel === "disabled") {
      const indexed = (analysis?.ops || []).find((candidate) => Number(candidate.index) === group.runtime_node_index);
      if (indexed && !String(indexed.graph_node_name || "") && indexed.name === group.op_name) {
        op = indexed;
        mappingMethod = "optimization_disabled_unnamed_node_index_and_op_type";
      }
    }
    if (!op) {
      unresolved.push({
        runtime_node_name: group.runtime_node_name,
        runtime_node_index: group.runtime_node_index,
        op_name: group.op_name,
        provider: group.provider,
        sample_count: group.sample_count,
        reason: exactNameMatches.length > 1
          ? "AMBIGUOUS_GRAPH_NODE_NAME"
          : optimizationLevel === "disabled" ? "NO_EXACT_OR_SAFE_INDEX_MATCH" : "NO_EXACT_GRAPH_NODE_NAME_MATCH",
      });
      continue;
    }
    proposed.push({
      op_index: Number(op.index),
      op_name: op.name,
      graph_node_name: String(op.graph_node_name || ""),
      provider: group.provider,
      delegated: !isOrtCpuProvider(group.provider),
      partition_id: null,
      kernel: null,
      duration_us: group.duration_sum_us / group.sample_count,
      duration_sum_us: group.duration_sum_us,
      sample_count: group.sample_count,
      mapping_method: mappingMethod,
      runtime_node_index: group.runtime_node_index,
      runtime_node_name: group.runtime_node_name,
      runtime_tensor_observation: group.runtime_tensor_observation,
    });
  }
  const assignments = [];
  const conflicts = [];
  for (const [opIndex, rows] of groupBy(proposed, (row) => row.op_index)) {
    const identities = new Set(rows.map((row) => `${row.provider}\0${row.runtime_node_index}\0${row.runtime_node_name}`));
    if (identities.size !== 1) {
      conflicts.push({
        op_index: opIndex,
        op_name: rows[0]?.op_name || "UNKNOWN",
        reason: "MULTIPLE_RUNTIME_IDENTITIES_FOR_ORIGINAL_OP",
        runtime_identities: [...identities],
      });
      continue;
    }
    const durationSum = rows.reduce((sum, row) => sum + row.duration_sum_us, 0);
    const sampleCount = rows.reduce((sum, row) => sum + row.sample_count, 0);
    assignments.push({
      ...rows[0],
      duration_us: durationSum / sampleCount,
      duration_sum_us: durationSum,
      sample_count: sampleCount,
    });
  }
  assignments.sort((left, right) => left.op_index - right.op_index);
  const runtimeTensorObservations = assignments
    .filter((row) => row.runtime_tensor_observation?.status !== "not_exposed")
    .map((row) => ({
      op_index: row.op_index,
      op_name: row.op_name,
      runtime_node_index: row.runtime_node_index,
      runtime_node_name: row.runtime_node_name,
      sample_count: row.sample_count,
      ...row.runtime_tensor_observation,
    }));
  const publicAssignments = assignments.map(({ runtime_tensor_observation: _observation, ...row }) => row);
  const sampleCounts = new Set(assignments.map((row) => row.sample_count));
  const executionMode = normalizedMode(metadata.executionMode, ["sequential", "parallel", "unknown"], "unknown");
  const durationsAdditive = executionMode === "sequential" && sampleCounts.size === 1;
  const methodCounts = countBy(assignments.map((row) => row.mapping_method));
  return {
    schema: ORT_PROFILE_ADAPTER_SCHEMA,
    graph_optimization_level: optimizationLevel,
    execution_mode: executionMode,
    assignment_count: assignments.length,
    graph_op_count: (analysis?.ops || []).length,
    mapping_coverage_ratio: assignments.length / Math.max(1, (analysis?.ops || []).length),
    kernel_event_count: profile.kernel_event_count,
    mapped_kernel_event_count: assignments.reduce((sum, row) => sum + row.sample_count, 0),
    unresolved_runtime_node_count: unresolved.length,
    conflict_count: conflicts.length,
    mapping_method_counts: methodCounts,
    durations_additive: durationsAdditive,
    duration_semantics: durationsAdditive ? "per_original_op_exclusive" : "unspecified",
    duration_statistic: "arithmetic_mean_per_profile_event",
    runtime_tensor_observation_count: runtimeTensorObservations.filter((row) => row.status === "consistent").length,
    runtime_tensor_observation_conflict_count: runtimeTensorObservations.filter((row) => row.status === "conflict_repeated_events").length,
    runtime_tensor_observation_not_exposed_count: assignments.length - runtimeTensorObservations.length,
    runtime_tensor_observations: runtimeTensorObservations,
    assignments: publicAssignments,
    unresolved_runtime_nodes: unresolved.slice(0, 256),
    conflicts: conflicts.slice(0, 256),
    interpretation_boundary: "Provider assignment is accepted only through a unique original ONNX node-name/op-type match, or for unnamed nodes through index/op-type identity when graph optimization is explicitly disabled. Fused, renamed, ambiguous, and conflicting runtime nodes remain unresolved.",
  };
}

export function buildOrtRuntimeAssignmentDocument(profile, analysis, metadata) {
  const preview = previewOrtProfileMapping(profile, analysis, metadata);
  const nativeRuntimeGraphOnly = !preview.assignment_count
    && metadata?.nativeCaptureEvidence
    && preview.kernel_event_count > 0
    && preview.unresolved_runtime_node_count > 0;
  if (!preview.assignment_count && !nativeRuntimeGraphOnly) {
    throw new Error("No ONNX Runtime profile node can be bound deterministically to the active graph.");
  }
  const profileSha = String(metadata?.profileSha256 || "").toLowerCase();
  if (!SHA256_PATTERN.test(profileSha)) throw new Error("ONNX Runtime profile SHA-256 is required.");
  const collectedAt = String(metadata?.collectedAt || "").trim();
  if (!collectedAt) throw new Error("ONNX Runtime profile collection time is required.");
  return {
    schema: "deepbom.runtime_assignment.v1.10",
    artifact_sha256: analysis?.model_sha256 || "",
    target_profile_id: analysis?.target_profile?.id || "",
    target_profile_sha256: analysis?.target_profile?.profile_sha256 || "",
    runtime: {
      name: "ONNX Runtime",
      version: requiredMetadata(metadata?.runtimeVersion, "runtime version"),
      backend: requiredMetadata(metadata?.backend || profile.providers.join(" + "), "runtime backend"),
      build: requiredMetadata(metadata?.runtimeBuild, "runtime build and flags", 500),
      binary_sha256: metadata?.binarySha256 || null,
      graph_optimization_level: preview.graph_optimization_level,
      execution_mode: preview.execution_mode,
    },
    source: {
      kind: "onnxruntime_profile_json_adapter",
      collected_at: collectedAt,
      capture_id: metadata?.captureId || null,
      capture_binding_semantics: metadata?.nativeCaptureEvidence
        ? "BROWSER_VERIFIED_NATIVE_CAPTURE_ENVELOPE_ARTIFACT_CONTENT_SET_AND_PROFILE_SHA256"
        : null,
      assignment_semantics: nativeRuntimeGraphOnly
        ? "runtime_graph_observed_original_graph_mapping_unresolved"
        : "original_graph_op_assignment",
      partition_semantics: "partition_id_identifies_runtime_partition_when_present",
      duration_semantics: preview.duration_semantics,
      duration_statistic: preview.duration_statistic,
      profile_sha256: profileSha,
      adapter: {
        schema: ORT_PROFILE_ADAPTER_SCHEMA,
        ...ORT_PROFILE_SOURCE,
        source_event_count: profile.source_event_count,
        kernel_event_count: preview.kernel_event_count,
        mapped_kernel_event_count: preview.mapped_kernel_event_count,
        unresolved_runtime_node_count: preview.unresolved_runtime_node_count,
        conflict_count: preview.conflict_count,
        mapping_method_counts: preview.mapping_method_counts,
        mapping_coverage_ratio: preview.mapping_coverage_ratio,
        runtime_tensor_observation_count: preview.runtime_tensor_observation_count,
        runtime_tensor_observation_conflict_count: preview.runtime_tensor_observation_conflict_count,
        runtime_tensor_observation_not_exposed_count: preview.runtime_tensor_observation_not_exposed_count,
        runtime_tensor_observations: preview.runtime_tensor_observations,
        native_capture: metadata?.nativeCaptureEvidence || null,
        interpretation_boundary: preview.interpretation_boundary,
      },
    },
    assignments: preview.assignments,
  };
}

function validateNativeOutputEvidence(envelope) {
  if (!Array.isArray(envelope.output_observations) || !envelope.output_observations.length
    || envelope.output_observations.some((output) => !output?.name || !output.type || !Array.isArray(output.dims)
      || !SHA256_PATTERN.test(String(output.first_run_sha256 || "")) || !SHA256_PATTERN.test(String(output.last_run_sha256 || "")))) {
    throw new Error("Native ORT output observations are invalid.");
  }
  const comparison = envelope.paired_profile_output_comparison;
  if (!comparison || comparison.schema !== "deepbom.ort_profile_output_comparison.v1"
    || comparison.reference_profile_role !== "identity" || comparison.candidate_profile_role !== "production"
    || !["assessed", "partially_assessed"].includes(comparison.status)
    || typeof comparison.all_outputs_bitwise_equal !== "boolean" || !Array.isArray(comparison.outputs) || !comparison.outputs.length) {
    throw new Error("Native ORT paired-profile output comparison is invalid.");
  }
  for (const output of comparison.outputs) {
    if (!output?.name || !output.type || !Array.isArray(output.dims) || typeof output.bitwise_equal !== "boolean"
      || !SHA256_PATTERN.test(String(output.identity_sha256 || "")) || !SHA256_PATTERN.test(String(output.production_sha256 || ""))) {
      throw new Error("Native ORT paired-profile output comparison row is invalid.");
    }
    if (output.numeric_comparison_status === "assessed") {
      for (const field of ["max_abs_error", "mean_abs_error", "rms_error", "relative_l2_error"]) {
        if (!Number.isFinite(output[field]) || output[field] < 0) throw new Error(`Native ORT ${field} is invalid.`);
      }
      if (output.cosine_distance != null && !Number.isFinite(output.cosine_distance)) throw new Error("Native ORT cosine distance is invalid.");
    } else if (output.numeric_comparison_status !== "not_assessed_non_numeric_or_nonfinite") {
      throw new Error("Native ORT numeric comparison status is invalid.");
    }
  }
  const runtimeGraph = envelope.paired_profile_runtime_graph;
  if (!runtimeGraph || !["deepbom.ort_paired_runtime_graph.v1", "deepbom.ort_paired_runtime_graph.v1.1"].includes(runtimeGraph.schema)
    || runtimeGraph.production_original_graph_mapping_status !== "NOT_INFERRED_TRANSFORMED_RUNTIME_NODE_IDENTITY"
    || !Array.isArray(runtimeGraph.profiles) || runtimeGraph.profiles.length !== 2) {
    throw new Error("Native ORT paired runtime graph is invalid.");
  }
  const roles = new Set();
  for (const runtimeProfile of runtimeGraph.profiles) {
    if (!['identity', 'production'].includes(runtimeProfile.role) || roles.has(runtimeProfile.role)
      || !SHA256_PATTERN.test(String(runtimeProfile.profile_sha256 || ""))
      || !Number.isSafeInteger(runtimeProfile.kernel_event_count) || runtimeProfile.kernel_event_count < 1
      || !Number.isSafeInteger(runtimeProfile.runtime_node_count) || runtimeProfile.runtime_node_count < 1
      || !Array.isArray(runtimeProfile.nodes) || runtimeProfile.nodes.length !== runtimeProfile.runtime_node_count
      || runtimeProfile.nodes.reduce((sum, node) => sum + Number(node.sample_count || 0), 0) !== runtimeProfile.kernel_event_count) {
      throw new Error("Native ORT paired runtime graph profile is invalid.");
    }
    roles.add(runtimeProfile.role);
    if (runtimeGraph.schema === "deepbom.ort_paired_runtime_graph.v1.1"
      && (!Number.isSafeInteger(runtimeProfile.invocation_run_count) || runtimeProfile.invocation_run_count < 1)) {
      throw new Error("Native ORT paired runtime graph invocation count is invalid.");
    }
    for (const node of runtimeProfile.nodes) {
      if (!node?.runtime_node_name || !Number.isSafeInteger(node.runtime_node_index) || node.runtime_node_index < 0 || !node.op_name || !node.provider
        || !Number.isSafeInteger(node.sample_count) || node.sample_count < 1
        || !Number.isFinite(node.duration_sum_us) || node.duration_sum_us < 0
        || !Number.isFinite(node.duration_mean_us) || node.duration_mean_us < 0
        || (runtimeGraph.schema === "deepbom.ort_paired_runtime_graph.v1.1"
          && node.output_size_bytes_decimal != null && !/^\d+$/.test(node.output_size_bytes_decimal))) {
        throw new Error("Native ORT paired runtime graph node is invalid.");
      }
    }
  }
}

function safeRelativePath(value) {
  const text = String(value || "").replaceAll("\\", "/").replace(/^(?:\.\/)+/, "");
  return text && !text.includes("\0") && !/^[a-z][a-z0-9+.-]*:/i.test(text) && !text.startsWith("/")
    && text.split("/").every((part) => part && part !== "." && part !== "..") ? text : null;
}

async function verifyNativeOrtArtifactContentSet(artifact, analysis) {
  const active = analysis?.onnx_external_data || {};
  const tensorCount = Number(active.tensor_count || 0);
  if (!Number.isSafeInteger(tensorCount) || tensorCount < 0 || tensorCount !== (active.tensors || []).length
    || (tensorCount && (active.status !== "verified_payloads" || Number(active.verified_payload_count || 0) !== tensorCount))) {
    throw new Error("Native ORT capture external data cannot be matched because the active ONNX sidecar ledger is incomplete.");
  }
  const files = (active.supplied_files || []).filter((file) => file.used === true).map((file) => ({
    path: safeRelativePath(file.path),
    byte_length: file.byte_length,
    sha256: String(file.sha256 || "").toLowerCase(),
    sha1: String(file.sha1 || "").toLowerCase(),
  })).sort((left, right) => compareText(left.path, right.path));
  if (files.some((file) => !file.path || !Number.isSafeInteger(file.byte_length) || file.byte_length < 0
    || !SHA256_PATTERN.test(file.sha256) || !/^[a-f0-9]{40}$/.test(file.sha1))) {
    throw new Error("Native ORT capture active external-data file ledger is invalid.");
  }
  const tensorRanges = (active.tensors || []).map((row) => ({
    scope: row.scope,
    tensor_role: row.tensor_role,
    tensor_name: row.tensor_name,
    location: safeRelativePath(row.normalized_location || row.location),
    offset: row.offset,
    length: row.length,
    payload_bytes: row.payload_bytes,
    checksum: String(row.checksum || "").toLowerCase(),
    sidecar_sha256: row.sidecar_sha256,
  })).sort(compareTensorRange);
  const filesByPath = new Map(files.map((file) => [file.path, file]));
  const verifiedPayloadBytes = tensorRanges.reduce((total, row) => total + Number(row.payload_bytes || 0), 0);
  if (tensorRanges.some((row) => {
    const file = filesByPath.get(row.location);
    return !row.scope || ![
      "graph_initializer", "node_attribute_tensor", "function_default_attribute_tensor",
      "graph_sparse_initializer_values", "graph_sparse_initializer_indices",
      "node_attribute_sparse_tensor_values", "node_attribute_sparse_tensor_indices",
      "function_default_attribute_sparse_tensor_values", "function_default_attribute_sparse_tensor_indices",
    ].includes(row.tensor_role)
      || !row.location || !file || !Number.isSafeInteger(row.offset) || row.offset < 0
      || !Number.isSafeInteger(row.payload_bytes) || row.payload_bytes < 0
      || !Number.isSafeInteger(row.offset + row.payload_bytes) || row.offset + row.payload_bytes > file.byte_length
      || (row.length == null ? row.payload_bytes !== file.byte_length - row.offset : row.length !== row.payload_bytes)
      || row.sidecar_sha256 !== file.sha256 || (row.checksum && row.checksum !== file.sha1);
  }) || verifiedPayloadBytes !== Number(active.verified_payload_bytes || 0)) {
    throw new Error("Native ORT capture active external-data tensor-range ledger is invalid.");
  }
  const externalData = {
    schema: "deepbom.onnx_external_artifact_set.v1.2",
    status: tensorCount ? "verified_payloads" : "assessed_absent",
    tensor_count: tensorCount,
    verified_payload_bytes: Number(active.verified_payload_bytes || 0),
    tensor_ranges: tensorRanges,
    files,
  };
  externalData.ledger_sha256 = await sha256Hex(new TextEncoder().encode(canonicalJson(externalData)));
  if (canonicalJson(artifact.external_data) !== canonicalJson(externalData)) {
    throw new Error("Native ORT capture external-data file/range ledger does not match the active ONNX audit.");
  }
  const contentSet = {
    byte_length: Number(artifact.byte_length),
    sha256: artifact.sha256,
    external_data: externalData,
  };
  const contentSetSha256 = await sha256Hex(new TextEncoder().encode(canonicalJson(contentSet)));
  if (Number(artifact.byte_length) !== Number(analysis?.file_size || 0)
    || !SHA256_PATTERN.test(String(artifact.content_set_sha256 || ""))
    || artifact.content_set_sha256 !== contentSetSha256) {
    throw new Error("Native ORT capture artifact content-set SHA-256 is invalid or does not match the active ONNX package.");
  }
}

function compareText(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function compareTensorRange(left, right) {
  return compareText(
    [left.scope, left.tensor_role, left.tensor_name, left.location, left.offset, left.length, left.payload_bytes, left.checksum, left.sidecar_sha256].join("\0"),
    [right.scope, right.tensor_role, right.tensor_name, right.location, right.offset, right.length, right.payload_bytes, right.checksum, right.sidecar_sha256].join("\0"),
  );
}

function structuredCloneSafe(value) {
  return value == null ? null : JSON.parse(JSON.stringify(value));
}

function groupProfileEvents(events) {
  const groups = new Map();
  for (const event of events) {
    const key = [event.runtime_node_name, event.runtime_node_index, event.op_name, event.provider].join("\0");
    const current = groups.get(key) || {
      ...event,
      sample_count: 0,
      duration_sum_us: 0,
      runtime_tensor_observation_variants: new Map(),
    };
    current.sample_count += 1;
    current.duration_sum_us += event.duration_us;
    const observationKey = canonicalJson(event.runtime_tensor_observation);
    if (!current.runtime_tensor_observation_variants.has(observationKey)) {
      current.runtime_tensor_observation_variants.set(observationKey, event.runtime_tensor_observation);
    }
    groups.set(key, current);
  }
  return [...groups.values()].map((group) => {
    const variants = [...group.runtime_tensor_observation_variants.values()];
    const { runtime_tensor_observation_variants: _variants, ...rest } = group;
    return {
      ...rest,
      runtime_tensor_observation: variants.length === 1
        ? variants[0]
        : {
          status: "conflict_repeated_events",
          input_type_shapes: [],
          output_type_shapes: [],
          activation_size_bytes: null,
          activation_size_bytes_decimal: null,
          parameter_size_bytes: null,
          parameter_size_bytes_decimal: null,
          output_size_bytes: null,
          output_size_bytes_decimal: null,
          observed_contract_variant_count: variants.length,
        },
    };
  });
}

function parseOrtRuntimeTensorObservation(args, sourceIndex) {
  const shapeFieldsPresent = Object.prototype.hasOwnProperty.call(args, "input_type_shape")
    || Object.prototype.hasOwnProperty.call(args, "output_type_shape");
  const sizeFieldsPresent = ["activation_size", "parameter_size", "output_size"]
    .some((field) => Object.prototype.hasOwnProperty.call(args, field));
  if (!shapeFieldsPresent && !sizeFieldsPresent) {
    return {
      status: "not_exposed",
      input_type_shapes: [],
      output_type_shapes: [],
      activation_size_bytes: null,
      activation_size_bytes_decimal: null,
      parameter_size_bytes: null,
      parameter_size_bytes_decimal: null,
      output_size_bytes: null,
      output_size_bytes_decimal: null,
      observed_contract_variant_count: 0,
    };
  }
  if (!Object.prototype.hasOwnProperty.call(args, "input_type_shape")
    || !Object.prototype.hasOwnProperty.call(args, "output_type_shape")) {
    throw new Error(`ONNX Runtime profile event ${sourceIndex} exposes an incomplete input/output type-shape pair.`);
  }
  const activationSize = parseOrtByteCount(args.activation_size, `event ${sourceIndex} activation_size`);
  const parameterSize = parseOrtByteCount(args.parameter_size, `event ${sourceIndex} parameter_size`);
  const outputSize = parseOrtByteCount(args.output_size, `event ${sourceIndex} output_size`);
  return {
    status: "consistent",
    input_type_shapes: parseOrtTypeShapeArray(args.input_type_shape, `event ${sourceIndex} input_type_shape`),
    output_type_shapes: parseOrtTypeShapeArray(args.output_type_shape, `event ${sourceIndex} output_type_shape`),
    activation_size_bytes: activationSize.number,
    activation_size_bytes_decimal: activationSize.decimal,
    parameter_size_bytes: parameterSize.number,
    parameter_size_bytes_decimal: parameterSize.decimal,
    output_size_bytes: outputSize.number,
    output_size_bytes_decimal: outputSize.decimal,
    observed_contract_variant_count: 1,
  };
}

function parseOrtTypeShapeArray(value, field) {
  let source = value;
  if (typeof source === "string") {
    try { source = JSON.parse(source); } catch { throw new Error(`ONNX Runtime ${field} is not valid JSON.`); }
  }
  if (!Array.isArray(source) || source.length > 100_000) {
    throw new Error(`ONNX Runtime ${field} must be an array with at most 100000 entries.`);
  }
  return source.map((entry, slot) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      throw new Error(`ONNX Runtime ${field}[${slot}] must be a one-key type/shape object.`);
    }
    const keys = Object.keys(entry);
    if (keys.length !== 1 || !/^[A-Za-z][A-Za-z0-9_]*$/.test(keys[0])) {
      throw new Error(`ONNX Runtime ${field}[${slot}] must contain exactly one bounded type key.`);
    }
    const ortType = keys[0];
    const shape = entry[ortType];
    if (!Array.isArray(shape) || shape.length > 64
      || shape.some((dimension) => !Number.isSafeInteger(Number(dimension)) || Number(dimension) < 0)) {
      throw new Error(`ONNX Runtime ${field}[${slot}] contains an invalid concrete tensor shape.`);
    }
    return {
      slot,
      ort_type: ortType,
      dtype: ORT_PROFILE_DTYPE_MAP[ortType.toLowerCase()] || null,
      shape: shape.map(Number),
    };
  });
}

function parseOrtByteCount(value, field) {
  if (value == null) return { number: null, decimal: null };
  const text = typeof value === "number" ? String(value) : String(value).trim();
  if (!/^\d+$/.test(text)) throw new Error(`ONNX Runtime ${field} must be a non-negative decimal integer.`);
  const exact = BigInt(text);
  return {
    number: exact <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(exact) : null,
    decimal: exact.toString(),
  };
}

function groupBy(values, keyFor) {
  const groups = new Map();
  for (const value of values) {
    const key = keyFor(value);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(value);
  }
  return groups;
}

function countBy(values) {
  const counts = {};
  for (const value of values) counts[value] = (counts[value] || 0) + 1;
  return counts;
}

function strictNonNegativeInteger(value) {
  if (typeof value === "string" && !/^\d+$/.test(value)) return null;
  const number = Number(value);
  return Number.isSafeInteger(number) && number >= 0 ? number : null;
}

function nodeKey(name, opName) {
  return `${name}\0${opName}`;
}

function normalizedMode(value, allowed, fallback) {
  const text = String(value || fallback).toLowerCase();
  return allowed.includes(text) ? text : fallback;
}

function isOrtCpuProvider(provider) {
  return String(provider).toLowerCase() === "cpuexecutionprovider";
}

function requiredMetadata(value, field, maxLength = 200) {
  const text = String(value || "").trim();
  if (!text) throw new Error(`ONNX Runtime ${field} is required.`);
  if (text.length > maxLength) throw new Error(`ONNX Runtime ${field} exceeds ${maxLength} characters.`);
  return text;
}
