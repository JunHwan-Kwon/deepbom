function closeEnough(actual, expected, tolerance = 1e-9) {
  if (actual == null || expected == null) return actual == null && expected == null;
  return Number.isFinite(Number(actual))
    && Number.isFinite(Number(expected))
    && Math.abs(Number(actual) - Number(expected)) <= tolerance * Math.max(1, Math.abs(Number(expected)));
}

function sameCounts(actual, expected) {
  const left = actual && typeof actual === "object" ? actual : {};
  const right = expected && typeof expected === "object" ? expected : {};
  const keys = [...new Set([...Object.keys(left), ...Object.keys(right)])].sort();
  return keys.every((key) => Number(left[key] || 0) === Number(right[key] || 0));
}

export function assessOrtRuntimeAdapter({ adapter, runtimeAssignment, staticAnalysis }) {
  const rows = runtimeAssignment?.assignments || [];
  const methodCounts = rows.reduce((counts, item) => {
    if (item.mapping_method) counts[item.mapping_method] = (counts[item.mapping_method] || 0) + 1;
    return counts;
  }, {});
  const mappingCountsMatch = sameCounts(adapter?.mapping_method_counts, methodCounts);
  const rawProfileBound = /^[a-f0-9]{64}$/.test(runtimeAssignment?.source?.profile_sha256 || "");
  const isOrtAdapter = ["deepbom.ort_profile_adapter.v2.1", "deepbom.ort_profile_adapter.v2.2"].includes(adapter?.schema);
  const mappedEvents = rows.reduce((sum, item) => sum + Number(item.sample_count || 0), 0);
  const mappingsValid = rows.every((item) => {
    const op = (staticAnalysis?.ops || []).find((candidate) => candidate.index === item.op_index && candidate.name === item.op_name);
    const durationValid = Number.isSafeInteger(item.sample_count) && item.sample_count > 0
      && Number.isFinite(item.duration_sum_us) && item.duration_sum_us >= 0
      && closeEnough(item.duration_us, item.duration_sum_us / item.sample_count);
    if (!op || !durationValid || item.partition_id != null || item.kernel != null) return false;
    if (item.mapping_method === "exact_graph_node_name_and_op_type") {
      return Boolean(op.graph_node_name) && op.graph_node_name === item.runtime_node_name;
    }
    return item.mapping_method === "optimization_disabled_unnamed_node_index_and_op_type"
      && runtimeAssignment?.runtime?.graph_optimization_level === "disabled"
      && !op.graph_node_name
      && item.runtime_node_index === item.op_index;
  });

  const nativeCapture = adapter?.native_capture || null;
  const selectedBuild = nativeCapture?.selected_build_provider_binding || null;
  const selectedRuntime = nativeCapture?.runtime || null;
  const compatibility = staticAnalysis?.ort_compatibility_evidence || null;
  const inventory = compatibility?.source_condition_inventory?.execution_providers || [];
  const backends = selectedRuntime?.supported_backends || [];
  const selectedBuildValid = !nativeCapture || (selectedBuild?.schema === "deepbom.ort_selected_build_provider_binding.v1.2"
    && selectedBuild.evidence_class === "OBSERVED_BUILD_INVENTORY_PLUS_SOURCE_PROFILE_CROSS_REFERENCE"
    && selectedBuild.runtime_source_commit === selectedRuntime.source_commit
    && selectedBuild.rulepack_source_commit === compatibility?.source_commit
    && selectedBuild.source_commit_match === true
    && selectedBuild.supported_backends_sha256 === selectedRuntime.supported_backends_sha256
    && selectedBuild.provider_inventory_status === "OBSERVED_FROM_ORT_LIST_SUPPORTED_BACKENDS"
    && ["NOT_EXPOSED_BY_ONNXRUNTIME_NODE_API_NOT_INFERRED", "IMPORTED_CONFIG_NOT_BINARY_ATTESTED", "BUILD_INPUT_BINARY_ATTESTED"].includes(selectedBuild.reduced_operator_inventory_status)
    && (["IMPORTED_CONFIG_NOT_BINARY_ATTESTED", "BUILD_INPUT_BINARY_ATTESTED"].includes(selectedBuild.reduced_operator_inventory_status)
      ? selectedBuild.reduced_operator_config_identity?.source_sha256 === selectedRuntime.reduced_operator_config?.source_sha256
        && selectedBuild.reduced_operator_config_identity?.normalized_sha256 === selectedRuntime.reduced_operator_config?.normalized_sha256
        && selectedBuild.reduced_operator_config_identity?.binary_binding_status === (selectedBuild.reduced_operator_inventory_status === "BUILD_INPUT_BINARY_ATTESTED"
          ? "ATTESTED_OBSERVED_BUILD_INPUT_BOUND_TO_SELECTED_BINARY_INVENTORY"
          : "NOT_ATTESTED_CONFIG_INPUT_NOT_OBSERVED_FROM_SELECTED_BINARY")
        && selectedBuild.reduced_operator_assessment?.schema === "deepbom.ort_reduced_operator_assessment.v1"
      : selectedBuild.reduced_operator_config_identity == null && selectedBuild.reduced_operator_assessment == null)
    && (selectedRuntime.distribution_identity === "SOURCE_BUILD_ATTESTED"
      ? selectedBuild.source_build_attestation?.attestation_sha256 === selectedRuntime.build_attestation?.attestation_sha256
        && selectedBuild.source_build_attestation?.binary_inventory_sha256 === selectedRuntime.binary_inventory_sha256
      : selectedBuild.source_build_attestation == null)
    && selectedBuild.bindings?.length === backends.length
    && selectedBuild.bindings.every((row, index) => row.backend_name === backends[index]?.name
      && row.bundled === backends[index]?.bundled
      && (row.source_profile == null
        ? row.source_profile_rule_count == null && row.binding_status.includes("WITHOUT_STATIC_SOURCE_PROFILE")
        : row.source_profile_rule_count === inventory.find((item) => item.execution_provider === row.source_profile)?.source_rule_count
          && row.binding_status.includes("WITH_PINNED_SOURCE_PROFILE")))
    && String(selectedBuild.interpretation_boundary || "").includes("not proof"));
  const tensorShapeEvidenceValid = adapter?.schema !== "deepbom.ort_profile_adapter.v2.2"
    || validateTensorShapeEvidence(adapter, rows);

  return Object.freeze({
    adapter_counts_match: mappingCountsMatch,
    raw_profile_bound: rawProfileBound,
    is_ort_adapter: isOrtAdapter,
    mapped_event_count: mappedEvents,
    mappings_valid: mappingsValid,
    selected_build_valid: selectedBuildValid,
    tensor_shape_evidence_valid: tensorShapeEvidenceValid,
    valid: isOrtAdapter
      && runtimeAssignment?.source?.kind === "onnxruntime_profile_json_adapter"
      && rawProfileBound
      && adapter.source_commit === "microsoft/onnxruntime@8c546c37b43caaca1fa25db430dab94b901cf277"
      && adapter.source_file === "onnxruntime/core/framework/sequential_executor.cc"
      && mappedEvents === adapter.mapped_kernel_event_count
      && closeEnough(adapter.mapping_coverage_ratio, rows.length / Math.max(1, staticAnalysis?.ops?.length || 0))
      && mappingCountsMatch
      && mappingsValid
      && selectedBuildValid
      && tensorShapeEvidenceValid,
  });
}

function validateTensorShapeEvidence(adapter, assignments) {
  const rows = adapter.runtime_tensor_observations;
  if (!Array.isArray(rows)) return false;
  const assignmentByOp = new Map(assignments.map((row) => [row.op_index, row]));
  const seen = new Set();
  const structurallyValid = rows.every((row) => {
    const assignment = assignmentByOp.get(row?.op_index);
    if (!assignment || seen.has(row.op_index)) return false;
    seen.add(row.op_index);
    const statusValid = row.status === "consistent"
      ? row.observed_contract_variant_count === 1
      : row.status === "conflict_repeated_events"
        && row.observed_contract_variant_count >= 2
        && row.input_type_shapes?.length === 0 && row.output_type_shapes?.length === 0;
    return statusValid
      && row.op_name === assignment.op_name
      && row.runtime_node_index === assignment.runtime_node_index
      && row.runtime_node_name === assignment.runtime_node_name
      && row.sample_count === assignment.sample_count
      && validTypeShapes(row.input_type_shapes)
      && validTypeShapes(row.output_type_shapes)
      && ["activation_size", "parameter_size", "output_size"].every((prefix) => validExactBytePair(row, prefix));
  });
  const observed = rows.filter((row) => row.status === "consistent").length;
  const conflicts = rows.filter((row) => row.status === "conflict_repeated_events").length;
  return structurallyValid
    && observed === adapter.runtime_tensor_observation_count
    && conflicts === adapter.runtime_tensor_observation_conflict_count
    && rows.length + Number(adapter.runtime_tensor_observation_not_exposed_count) === assignments.length;
}

function validTypeShapes(rows) {
  return Array.isArray(rows) && rows.every((row, index) => row?.slot === index
    && typeof row.ort_type === "string" && row.ort_type.length > 0
    && (row.dtype == null || typeof row.dtype === "string")
    && Array.isArray(row.shape) && row.shape.length <= 64
    && row.shape.every((dimension) => Number.isSafeInteger(dimension) && dimension >= 0));
}

function validExactBytePair(row, prefix) {
  const exact = row[`${prefix}_bytes_decimal`];
  const number = row[`${prefix}_bytes`];
  if (exact == null) return number == null;
  if (!/^\d+$/.test(String(exact))) return false;
  const value = BigInt(exact);
  return number === (value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null);
}
