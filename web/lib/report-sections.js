import { formatBytes, formatDrift, formatNumber, formatPercent, maxBy, padOp, score100 } from "./format.js";
import { RUNTIME_COMPATIBILITY_EVIDENCE_LABEL } from "./report-metadata.js";
import { code, markdownTable } from "./report-utils.js";
import { buildRuntimeBackendEvidenceLedger } from "./runtime-backend-evidence-ledger.js";
import { buildRuntimeDataMovementEvidence } from "./runtime-data-movement-evidence.js";
import { buildRuntimeEvidenceSidecar } from "./runtime-evidence-sidecar.js";
import { buildOnnxRuntimeShapeBinding } from "./onnx-runtime-shape-binding.js";

function formatSignedBytes(value, notAssessed = "NOT_ASSESSED") {
  if (value == null || !Number.isFinite(Number(value))) return notAssessed;
  const number = Number(value);
  return `${number > 0 ? "+" : number < 0 ? "-" : ""}${formatBytes(Math.abs(number))}`;
}

export function rooflineMixSummary(analysis) {
  const ops = analysis?.ops || [];
  const assessed = ops.filter((op) => op.intensity_status == null || op.intensity_status === "assessed");
  const total = Math.max(assessed.length, 1);
  const compute = assessed.filter((op) => op.static_bound_guess === "compute-bound").length;
  const mixed = assessed.filter((op) => op.static_bound_guess === "mixed").length;
  const memory = assessed.filter((op) => op.static_bound_guess === "memory-bound").length;
  return `high-intensity ${formatPercent(compute / total)} / mixed-intensity ${formatPercent(mixed / total)} / low-intensity ${formatPercent(memory / total)} (${assessed.length}/${ops.length} op(s) assessed; ${ops.length - assessed.length} N/A)`;
}

export function l1PressureSummary(analysis) {
  const top = maxBy(analysis?.ops || [], (op) => Number(op.row_working_set_ratio || 0));
  if (!top || !Number(top.row_working_set_ratio || 0)) return "No row working-set signal available.";
  return `max #${padOp(top.index)} ${top.name}: ${formatBytes(top.row_working_set_bytes || 0)} / ${Number(top.row_working_set_ratio || 0).toFixed(2)}x L1`;
}

export function evidenceClassLegend() {
  return markdownTable(["Class", "Meaning"], [
    ["OBSERVED", "Read directly from the selected model artifact or browser/runtime environment."],
    ["DERIVED", "Calculated deterministically from observed graph, tensor, or runtime metadata."],
    ["MEASURED_SYNTHETIC", "Measured in this browser with synthetic or prepared local inputs."],
    ["HEURISTIC", "Composite triage index or label built from bounded analyzer signals; no pass/fail or deployment-readiness meaning."],
    ["ESTIMATED", "Modeled from target profile, static performance assumptions, cache/roofline heuristics, or byte estimates."],
    ["PREDICTED", "Predicted from runtime/delegate rulepack rather than confirmed delegate logs."],
    ["PROXY", "Indirect stability, topology, or weight-space signal used to prioritize follow-up validation."],
    ["NOT_ASSESSABLE", "Cannot be assessed from the deployment artifact alone."],
  ]);
}

export function runtimeEnvironmentMarkdown(runtimeEvidence = {}, runtimeBenchmarkResults = [], analysis = null) {
  const backends = runtimeBenchmarkResults.length
    ? runtimeBenchmarkResults.map((row) => `${row.backend}:${row.ok ? "ok" : "failed"}`).join(" / ")
    : "no benchmark run";
  const inputBasis = [...new Set(runtimeBenchmarkResults.map((row) => row.input_basis).filter(Boolean))].join(" / ") || "-";
  const importedRuntimeEvidence = runtimeEvidence.runtimeAssignmentEvidence || null;
  const normalizedSidecar = analysis && importedRuntimeEvidence
    ? buildRuntimeEvidenceSidecar(analysis, importedRuntimeEvidence)
    : null;
  const ggufEnvironment = importedRuntimeEvidence?.schema === "deepbom.gguf_runtime_environment.v2" ? importedRuntimeEvidence : null;
  const coreMlComputePlan = importedRuntimeEvidence?.schema === "deepbom.coreml_compute_plan.v1" ? importedRuntimeEvidence : null;
  const assignment = ggufEnvironment || coreMlComputePlan ? null : importedRuntimeEvidence;
  const adapter = assignment?.source?.adapter || null;
  const tfliteRuntimeInfo = adapter?.schema?.startsWith("deepbom.tflite_runtime_info_adapter.v");
  const nativeOrtCapture = adapter?.native_capture || null;
  const nativeOrtSelectedBuild = nativeOrtCapture?.selected_build_provider_binding || null;
  const nativeOrtOutputComparison = nativeOrtCapture?.paired_profile_output_comparison || null;
  const nativeOrtProduction = (nativeOrtCapture?.paired_profile_runtime_graph?.profiles || []).find((item) => item.role === "production") || null;
  const runtimeBackendLedger = buildRuntimeBackendEvidenceLedger(assignment);
  const runtimeDataMovement = buildRuntimeDataMovementEvidence(assignment);
  const onnxRuntimeShapeBinding = assignment && String(analysis?.format || "").toLowerCase() === "onnx"
    ? buildOnnxRuntimeShapeBinding(analysis, assignment)
    : null;
  const timingProfile = tfliteRuntimeInfo ? adapter?.timing_profile || null : null;
  const selectorObservation = assignment?.selector_observation || null;
  const selectorContext = assignment?.selector_context || null;
  const resourcePartition = selectorContext?.invocation?.resource_partition || null;
  const tfliteDelegateBuild = assignment?.tflite_delegate_build_inventory || null;
  const runtimeMemory = assignment?.runtime_memory || null;
  const arenaReconciliation = assignment?.arena_reconciliation || null;
  const adapterSummary = !adapter ? "not used" : tfliteRuntimeInfo
    ? `${adapter.schema}; runtime-plan protobuf SHA-256 ${assignment.source.profile_sha256}; exact topology ${adapter.topology_binding?.matched_original_op_count || 0}/${adapter.topology_binding?.graph_op_count || 0} original op(s); ${adapter.delegate_node_count || 0} delegate partition(s), ${adapter.execution_plan_node_count || 0} execution-plan node(s)${timingProfile ? `; timing protobuf SHA-256 ${timingProfile.profile_sha256}, mapped ${timingProfile.mapped_execution_node_count}/${timingProfile.execution_plan_node_count} execution node(s)` : ""}`
    : `${adapter.schema}; profile SHA-256 ${assignment.source.profile_sha256}; mapped ${adapter.mapped_kernel_event_count}/${adapter.kernel_event_count} kernel event(s) to ${assignment.assignment_count}/${assignment.graph_op_count} original op(s); unresolved ${adapter.unresolved_runtime_node_count}, mapping conflicts ${adapter.conflict_count}; internal type-shape ${adapter.runtime_tensor_observation_count || 0}, repeated-shape conflicts ${adapter.runtime_tensor_observation_conflict_count || 0}, not exposed ${adapter.runtime_tensor_observation_not_exposed_count ?? assignment.assignment_count}${nativeOrtCapture ? `; pinned native ${nativeOrtCapture.profile_role} capture ${nativeOrtCapture.capture_id}` : ""}`;
  const identityProvenance = !assignment ? "not imported" : tfliteRuntimeInfo
    ? `runtime version/build/collection time/capture ID are DECLARED; execution plan, delegate names, and symmetric replacement map are OBSERVED_RUNTIME; timing statistics are OBSERVED_RUNTIME when imported; source artifact SHA-256 embedded false; binary SHA-256 ${assignment.runtime?.binary_sha256 || "not provided"}`
    : nativeOrtCapture
      ? `provider/node/duration rows and collector-recorded package/binary inventory are OBSERVED_RUNTIME; envelope, active ONNX plus external-data content set, and embedded profile hashes were browser-verified; native binaries were not browser-rehashed; primary binary SHA-256 ${assignment.runtime?.binary_sha256}`
      : `${adapter ? "runtime version/build/preparation are DECLARED; provider/node/duration rows are OBSERVED_RUNTIME" : "runtime identity is declared by the imported assignment source"}; binary SHA-256 ${assignment.runtime?.binary_sha256 || "not provided"}`;
  const environment = markdownTable(["Field", "Value"], [
    ["Browser bucket", runtimeEvidence.browserBucket || "unknown"],
    ["SharedArrayBuffer", yesNo(runtimeEvidence.sharedArrayBufferAvailable)],
    ["WebGPU exposed", yesNo(runtimeEvidence.webgpuAvailable)],
    ["WebNN exposed", yesNo(runtimeEvidence.webnnAvailable)],
    ["Backends observed", backends],
    ["Imported runtime assignment", assignment ? `OBSERVED_RUNTIME; ${assignment.runtime?.name || "runtime"} ${assignment.runtime?.version || ""} / ${assignment.runtime?.backend || ""}; ${assignment.assignment_count}/${assignment.graph_op_count} op(s), artifact SHA-256 and target-profile SHA-256 bound` : "not imported; delegate/provider assignment remains predicted or not assessable"],
    ["Imported runtime environment", ggufEnvironment ? `OBSERVED_INSTRUMENTED_RUNTIME; ${ggufEnvironment.runtime.repository}@${ggufEnvironment.runtime.source_commit}; binary SHA-256 ${ggufEnvironment.runtime.binary_sha256}; scheduler graphs ${ggufEnvironment.compute_graph?.graph_count || 0}; dispatched ${ggufEnvironment.compute_graph?.dispatched_graph_count || 0}` : "not imported"],
    ["Imported Core ML compute plan", coreMlComputePlan ? `COREML_COMPUTE_PLAN_ESTIMATE; ${coreMlComputePlan.structure.operation_count} operation row(s); compute units ${coreMlComputePlan.configuration.compute_units}; execution ${coreMlComputePlan.execution_status}` : "not imported"],
    ["Cross-format runtime sidecar", normalizedSidecar ? `${normalizedSidecar.schema}; SHA-256 ${normalizedSidecar.sidecar_sha256}; source ${normalizedSidecar.source_contract.schema} ${normalizedSidecar.source_contract.evidence_sha256}` : "not emitted; requires imported runtime evidence and the active artifact identity"],
    ["Runtime identity provenance", identityProvenance],
    ["TFLite capture binding", tfliteRuntimeInfo ? `${assignment.source.capture_binding_semantics || "not declared"}; capture ID ${assignment.source.capture_id || "not provided"}${timingProfile ? `; timing attachment ${timingProfile.capture_binding_semantics}` : ""}` : "not applicable"],
    ["Native ORT capture binding", nativeOrtCapture ? `${assignment.source.capture_binding_semantics}; capture ID ${nativeOrtCapture.capture_id}; role ${nativeOrtCapture.profile_role}; envelope SHA-256 ${nativeOrtCapture.capture_content_sha256}; artifact content-set SHA-256 ${nativeOrtCapture.artifact.content_set_sha256}; external files ${nativeOrtCapture.artifact.external_data.files.length}, tensor ranges ${nativeOrtCapture.artifact.external_data.tensor_count}; binary inventory SHA-256 ${nativeOrtCapture.runtime.binary_inventory_sha256}` : "not applicable"],
    ["Native ORT selected build", nativeOrtSelectedBuild ? `${nativeOrtSelectedBuild.provider_inventory_status}; backends ${nativeOrtSelectedBuild.bindings.map((row) => `${row.backend_name}:${row.bundled ? "bundled" : "available"}`).join(" / ")}; backend inventory SHA-256 ${nativeOrtSelectedBuild.supported_backends_sha256}; reduced operator inventory ${nativeOrtSelectedBuild.reduced_operator_inventory_status}` : "not applicable"],
    ["Native ORT production runtime graph", nativeOrtProduction ? `${nativeOrtProduction.runtime_node_count} optimized/fused runtime node(s); ${nativeOrtProduction.kernel_event_count} kernel event(s); providers ${nativeOrtProduction.observed_providers.join(" / ")}; profile SHA-256 ${nativeOrtProduction.profile_sha256}; original-op mapping NOT_INFERRED` : "not applicable"],
    ["Native ORT paired output", nativeOrtOutputComparison ? `${nativeOrtOutputComparison.all_outputs_bitwise_equal ? "bitwise equal" : "not bitwise equal"}; ${nativeOrtOutputComparison.outputs.map((output) => output.numeric_comparison_status === "assessed" ? `${output.name} max_abs ${formatDrift(output.max_abs_error)}, RMS ${formatDrift(output.rms_error)}, relative-L2 ${formatDrift(output.relative_l2_error)}, cosine ${formatDrift(output.cosine_distance)}` : `${output.name} ${output.numeric_comparison_status}`).join(" / ")}` : "not applicable"],
    ["Runtime graph preparation", assignment ? tfliteRuntimeInfo ? "not represented in ModelRuntimeDetails; no preparation state inferred" : `${nativeOrtCapture ? "COLLECTOR_OBSERVED" : "DECLARED"} optimization ${assignment.runtime?.graph_optimization_level || "not declared"}; execution ${assignment.runtime?.execution_mode || "not declared"}` : "not imported"],
    ["Runtime profile adapter", adapterSummary],
    ["Native selector context", selectorContext ? `${selectorContext.device.architecture} / ${selectorContext.device.cpu_features.join(", ")}; XNNPACK ${selectorContext.build.xnnpack_source_commit}; runtime binary ${selectorContext.build.runtime_binary_sha256}; build ID ${selectorContext.build.microkernel_build_identifier_sha256}; collector attestation ${selectorObservation.collector_attestation_status}` : "not imported"],
    ["CPU affinity / isolation observation", resourcePartition
      ? `${resourcePartition.exclusive_isolation_status === "observed_cgroup_v2_isolated_partition" ? "OBSERVED_OS_RESOURCE_PARTITION" : "OBSERVED_OS_AFFINITY"}; requested [${resourcePartition.requested_cpu_ids.join(", ")}]; effective [${resourcePartition.observed_effective_cpu_ids.join(", ")}]; processors [${resourcePartition.observed_processor_ids.join(", ")}]; ${resourcePartition.sample_count} sample(s), max ${resourcePartition.maximum_observed_thread_count} thread(s); observation SHA-256 ${resourcePartition.observation_sha256}`
      : "not imported; static core-allocation scenarios do not establish scheduler affinity or exclusive CPU isolation"],
    ["TFLite GPU / NNAPI selected build", tfliteDelegateBuild ? `DECLARED_HASH_BOUND; GPU ${tfliteDelegateBuild.gpu.compiled_status}; quant option ${tfliteDelegateBuild.gpu.quantized_model_flag_status}; NNAPI ${tfliteDelegateBuild.nnapi.compiled_status}; assignment not established` : "not imported"],
    ["Lowering / microkernel observation", selectorObservation ? `${selectorObservation.lowering_observed_op_count} lowering op(s); ${selectorObservation.microkernel_observed_op_count} microkernel op(s); ambiguity closed for ${selectorObservation.selector_ambiguity_closed_op_count}/${selectorObservation.graph_op_count}; ${selectorObservation.status}` : "not collected"],
    ["Observed TFLite arena memory", runtimeMemory ? `${runtimeMemory.evidence_class}; ${runtimeMemory.snapshot_count} post-commit snapshot(s); peak ${formatBytes(runtimeMemory.peak_combined_arena_bytes)}, final ${formatBytes(runtimeMemory.final_combined_arena_bytes)}; ledger SHA-256 ${runtimeMemory.allocation_ledger_sha256}` : "not collected"],
    ["Arena projection reconciliation", arenaReconciliation ? `${arenaReconciliation.status}; projected ${arenaReconciliation.projected_combined_arena_bytes == null ? "not assessed" : formatBytes(arenaReconciliation.projected_combined_arena_bytes)}, observed peak ${formatBytes(arenaReconciliation.observed_peak_combined_arena_bytes)}, delta ${formatSignedBytes(arenaReconciliation.peak_delta_bytes, "not assessed")}; runtime-only ${arenaReconciliation.runtime_only_allocation_count}, missing observed ${arenaReconciliation.missing_observed_allocation_count}` : "not assessed"],
    ["Dispatch sample semantics", assignment?.source?.dispatch_sample_semantics || "not applicable"],
    ["Runtime identity mapping", adapter ? Object.entries(adapter.mapping_method_counts || {}).map(([method, count]) => `${method}:${count}`).join(" / ") || "none" : "not applicable"],
    ["Runtime profile parser basis", adapter ? tfliteRuntimeInfo ? `${adapter.source_commit} / ${adapter.source_file} SHA-256 ${adapter.source_sha256}; ${adapter.proto_file} SHA-256 ${adapter.proto_sha256}; ${adapter.delegation_metadata_file} SHA-256 ${adapter.delegation_metadata_sha256}; ${adapter.export_driver_file} SHA-256 ${adapter.export_driver_sha256}${timingProfile ? `; timing ${timingProfile.proto_file} SHA-256 ${timingProfile.proto_sha256}; ${timingProfile.summarizer_file} SHA-256 ${timingProfile.summarizer_sha256}; ${timingProfile.formatter_file} SHA-256 ${timingProfile.formatter_sha256}; ${timingProfile.profiler_api_file} SHA-256 ${timingProfile.profiler_api_sha256}; ${timingProfile.subgraph_file} SHA-256 ${timingProfile.subgraph_sha256}; ${timingProfile.xnnpack_delegate_file} SHA-256 ${timingProfile.xnnpack_delegate_sha256}` : ""}` : `${adapter.source_commit} / ${adapter.source_file}` : "not applicable"],
    ["TFLite timing aggregation", timingProfile ? `${timingProfile.total_status}; execution nodes ${timingProfile.mapped_execution_node_count}/${timingProfile.execution_plan_node_count}; total ${timingProfile.execution_node_total_us == null ? "withheld" : `${Number(timingProfile.execution_node_total_us).toFixed(3)} us`}; primary delegate-profiled ${timingProfile.primary_delegate_profiled_subtotal_us == null ? "not assessed" : `${Number(timingProfile.primary_delegate_profiled_subtotal_us).toFixed(3)} us, unassigned`}; nested delegate-internal ${timingProfile.delegate_internal_section_subtotal_us == null ? "not assessed" : `${Number(timingProfile.delegate_internal_section_subtotal_us).toFixed(3)} us, non-additive`}` : tfliteRuntimeInfo ? "not imported" : "not applicable"],
    ["Benchmark input basis", inputBasis],
    ["Timing protocol", "Compile/setup is tracked separately. First (cold) is exactly one invocation before any warmup. Declared warmup runs follow and are excluded. Every reported p50/p90/p95/p99/mean and Steady p50 sample is measured after warmup."],
    ["Reproducibility note", "Re-run after changing browser, runtime assets, target profile, device, power state, or prepared input."],
  ]);
  return [
    environment,
    ggufEnvironment ? ggufRuntimeEnvironmentMarkdown(ggufEnvironment) : "",
    coreMlComputePlan ? coreMlComputePlanMarkdown(coreMlComputePlan) : "",
    assignment?.comparison ? runtimeAssignmentComparisonMarkdown(assignment.comparison) : "",
    nativeOrtSelectedBuild ? ortSelectedBuildBindingMarkdown(nativeOrtSelectedBuild) : "",
    runtimeBackendLedger ? runtimeBackendEvidenceLedgerMarkdown(runtimeBackendLedger) : "",
    runtimeDataMovement ? runtimeDataMovementEvidenceMarkdown(runtimeDataMovement) : "",
    onnxRuntimeShapeBinding ? onnxRuntimeShapeBindingMarkdown(onnxRuntimeShapeBinding) : "",
    tfliteDelegateBuild ? tfliteDelegateBuildInventoryMarkdown(tfliteDelegateBuild) : "",
    arenaReconciliation ? runtimeArenaReconciliationMarkdown(runtimeMemory, arenaReconciliation) : "",
  ].filter(Boolean).join("\n\n");
}

function runtimeDataMovementEvidenceMarkdown(evidence) {
  const rows = (evidence.copy_nodes || []).map((row) => [
    `${row.runtime_node_name} (#${row.runtime_node_index})`,
    row.op_name,
    row.direction,
    row.provider,
    row.sample_count,
    row.output_payload_bytes ? `${row.output_payload_bytes.decimal} B` : "NOT ASSESSED",
  ]);
  return [
    "### Runtime Copy-Node Data Movement",
    "",
    `Status: ${code(evidence.status)}; profiled copy nodes ${formatNumber(evidence.observed_copy_node_count)}; copy events ${formatNumber(evidence.observed_copy_event_count)}.`,
    `Observed logical copy payload across captured events: ${evidence.observed_copy_event_payload_bytes ? `${evidence.observed_copy_event_payload_bytes.decimal} B` : "NOT ASSESSED"}; per invocation: ${evidence.observed_copy_payload_per_invocation_bytes ? `${evidence.observed_copy_payload_per_invocation_bytes.decimal} B` : "NOT ASSESSED"}.`,
    rows.length ? markdownTable(["Runtime node", "Op", "Direction", "Provider", "Samples", "Output payload / event"], rows) : "No profiled copy-node row is present for this captured configuration.",
    `> Physical transfer bytes: ${code(evidence.physical_transfer_status)}. ${evidence.interpretation_boundary}`,
  ].join("\n");
}

function onnxRuntimeShapeBindingMarkdown(evidence) {
  const tensorRows = (evidence.observed_tensors || []).slice(0, 32).map((row) => [
    `T${row.tensor_index}`,
    code(row.tensor_name),
    row.dtype || "not mapped",
    row.executed_shape.join("x") || "scalar",
    row.payload_bytes_decimal == null ? "not assessed" : formatBytes(Number(row.payload_bytes_decimal)),
    row.observation_sources.join(" / "),
  ]);
  const residualRows = (evidence.remaining_mac_residuals || []).slice(0, 32).map((row) => [
    `#${padOp(row.op_index)}`,
    row.op_name,
    row.runtime_reason || row.static_reason || "not assessed",
  ]);
  return [
    "### ONNX Runtime Internal Shape And Cost Binding",
    markdownTable(["Field", "Value"], [
      ["Status / evidence", `${evidence.status} / ${evidence.evidence_class}`],
      ["Observed tensor contracts", `${evidence.observed_tensor_count} unique; interface ${evidence.observed_interface_tensor_count}; internal ${evidence.observed_internal_tensor_count}`],
      ["Observed logical payload", evidence.observed_tensor_payload_bytes_decimal == null ? `${evidence.observed_tensor_payload_assessed_count}/${evidence.observed_tensor_count} tensor(s) assessed` : `${evidence.observed_tensor_payload_bytes == null ? `${evidence.observed_tensor_payload_bytes_decimal} B exact` : formatBytes(evidence.observed_tensor_payload_bytes)} across ${evidence.observed_tensor_count} unique tensor(s)`],
      ["Dynamic symbols", `${evidence.bound_symbol_count}/${evidence.symbol_count} bound; ${evidence.unbound_formula_symbol_count} formula symbol(s) unresolved`],
      ["MAC closure", `${evidence.runtime_closed_mac_op_count} op(s) newly closed; ${evidence.remaining_unassessed_mac_op_count} remain; runtime-bound assessed subtotal ${evidence.runtime_bound_assessed_macs_decimal}`],
      ["Complete runtime-bound MAC total", evidence.runtime_bound_complete_macs_decimal ?? "not assessed"],
      ["Conflicts / exclusions", `${evidence.conflict_count} / ${evidence.exclusion_count}`],
    ]),
    tensorRows.length ? markdownTable(["Tensor", "Name", "Dtype", "Executed shape", "Logical payload", "Observation source"], tensorRows) : "No runtime tensor contract was accepted.",
    residualRows.length ? markdownTable(["Op", "Type", "Remaining reason"], residualRows) : "All supported MAC-bearing signatures are assessable under this runtime shape binding.",
    `> ${evidence.interpretation_boundary}`,
  ].join("\n\n");
}

function runtimeBackendEvidenceLedgerMarkdown(evidence) {
  return [
    "### Selected Runtime Backend Evidence Ledger",
    markdownTable(["Backend", "Configured inclusion", "Capability acceptance", "Original-op assignment", "Execution"], evidence.providers.map((row) => [
      row.label,
      `${row.configured_inclusion.status} (${row.configured_inclusion.evidence_class})`,
      `${row.capability_acceptance.status} (${row.capability_acceptance.evidence_class})`,
      `${row.assignment.status}; ${formatNumber(row.assignment.assigned_original_op_count)} op(s)`,
      `${row.execution.status}; ${formatNumber(row.execution.executed_original_op_count)} op(s), ${formatNumber(row.execution.event_sample_count)} event sample(s)`,
    ])),
    `Ledger SHA-256: ${code(evidence.ledger_sha256)}`,
    `> ${evidence.interpretation_boundary}`,
  ].join("\n\n");
}

function ortSelectedBuildBindingMarkdown(evidence) {
  const config = evidence.reduced_operator_config_identity;
  const reduced = evidence.reduced_operator_assessment;
  const attestation = evidence.source_build_attestation;
  return [
    "### ONNX Runtime Selected-Build Provider Binding",
    markdownTable(["Backend", "Bundled", "Pinned source profile", "Source rules", "Binding status"], evidence.bindings.map((row) => [
      row.backend_name,
      yesNo(row.bundled),
      row.source_profile || "not available",
      row.source_profile_rule_count ?? "not available",
      row.binding_status,
    ])),
    `Source profiles not listed by this selected build: ${evidence.source_profiles_not_listed_by_selected_build.join(" / ") || "none"}.`,
    config ? markdownTable(["Reduced-build config field", "Value"], [
      ["Source / SHA-256", `${config.source_name} / ${config.source_sha256}`],
      ["Normalized SHA-256", config.normalized_sha256],
      ["Binary binding", config.binary_binding_status],
      ["Artifact operator compatibility", `${reduced.status}; included ${reduced.included_node_count}/${reduced.assessed_node_count}; excluded ${reduced.excluded_node_count}; unresolved identity ${reduced.unresolved_node_count}; unresolved type reduction ${reduced.type_reduction_unresolved_node_count}`],
    ]) : "Reduced-operator config: not imported; selected package does not expose a compiled-op inventory.",
    attestation ? markdownTable(["Source-build attestation field", "Value"], [
      ["Schema / distribution", `${attestation.schema} / ${attestation.distribution_identity}`],
      ["Attestation SHA-256", attestation.attestation_sha256],
      ["Package manifest SHA-256", attestation.package_manifest_sha256],
      ["Binary inventory / primary SHA-256", `${attestation.binary_inventory_sha256} / ${attestation.primary_binary_sha256}`],
      ["Reduced config build-input SHA-256", attestation.reduced_operator_config_sha256 || "not a reduced build"],
    ]) : "Source-build attestation: not imported; package lock and observed binaries identify the selected prebuilt distribution.",
    reduced?.rows?.some((row) => row.status !== "included_exact_imported_opset" && row.status !== "included_all_operators_directive")
      ? markdownTable(["Scope", "Operator", "Config status", "Type-reduction status"], reduced.rows
        .filter((row) => row.status !== "included_exact_imported_opset" && row.status !== "included_all_operators_directive")
        .slice(0, 32)
        .map((row) => [row.scope, `${row.domain}:${row.imported_opset}:${row.op_name}`, row.status, row.type_reduction_status])) : "",
    `> ${evidence.interpretation_boundary}`,
  ].join("\n\n");
}

function tfliteDelegateBuildInventoryMarkdown(evidence) {
  return [
    "### TFLite GPU / NNAPI Selected-Build Inventory (DECLARED_HASH_BOUND/NOT_ASSIGNMENT)",
    markdownTable(["Selected-build field", "Value"], [
      ["Schema / evidence", `${evidence.schema} / ${evidence.evidence_class}`],
      ["Artifact / runtime binary SHA-256", `${evidence.artifact_sha256} / ${evidence.runtime_binary_sha256}`],
      ["Build manifest SHA-256", evidence.build_manifest_sha256],
      ["TensorFlow source commit", evidence.tensorflow_source_commit],
      ["CMake system", evidence.cmake_system_name || "not declared"],
      ["GPU compiled / quant flag / max partitions", `${evidence.gpu.compiled_status} / ${evidence.gpu.quantized_model_flag_status} / ${evidence.gpu.max_delegated_partitions ?? "not declared"}`],
      ["NNAPI compiled / feature / accelerator", `${evidence.nnapi.compiled_status} / ${evidence.nnapi.runtime_feature_level ?? "not collected"} / ${evidence.nnapi.accelerator_identity || "not collected"}`],
      ["NNAPI capability source", evidence.nnapi.capability_source],
    ]),
    markdownTable(["CMake option", "Declared", "Normalized enabled", "Effective status"], (evidence.build_options || []).map((option) => [
      option.name,
      option.declared_value ?? "not declared",
      option.normalized_enabled == null ? "not declared" : yesNo(option.normalized_enabled),
      option.effective_status,
    ])),
    markdownTable(["Pinned selected-build source", "SHA-256", "Reference"], (evidence.source_files || []).map((source) => [source.id, source.sha256, source.source_ref])),
    `> ${evidence.interpretation_boundary}`,
  ].join("\n\n");
}

function coreMlComputePlanMarkdown(evidence) {
  const rows = evidence.structure.rows.map((row) => [
    `#${String(row.op_index).padStart(3, "0")}`,
    row.operator_type,
    code(row.identity),
    row.preferred_compute_device || "not determined",
    row.supported_compute_devices.join(" / ") || "not determined",
    row.estimated_cost_weight == null ? "not determined" : Number(row.estimated_cost_weight).toFixed(9),
  ]);
  return [
    "### Core ML MLComputePlan Estimate",
    markdownTable(["Field", "Bound evidence"], [
      ["Evidence class", `${evidence.evidence_class}; ${evidence.execution_status}`],
      ["Artifact / compiled model", `${evidence.artifact.sha256} / ${evidence.runtime.compiled_model_content_sha256}`],
      ["Runtime source", `coremltools ${evidence.runtime.coremltools_version}; compute_plan.py SHA-256 ${evidence.runtime.coremltools_compute_plan_source_sha256}; ${evidence.runtime.pinned_source_alignment}`],
      ["Host identity", `${evidence.runtime.hardware_model}; macOS ${evidence.runtime.macos_version} (${evidence.runtime.os_build}); ${evidence.runtime.architecture}; Python ${evidence.runtime.python_version}`],
      ["Available compute devices", evidence.runtime.available_compute_devices.map((device) => `${device.type} x${device.instance_count}${device.total_core_count == null ? "" : ` (${device.total_core_count} cores)`}`).join(" / ")],
      ["Configuration", `compute units ${evidence.configuration.compute_units}; function ${evidence.configuration.function_name || "model-level"}`],
      ["Identity coverage", `${evidence.summary.mapped_operation_count}/${evidence.structure.operation_count}; ${evidence.structure.kind}`],
      ["Preferred device counts", Object.entries(evidence.summary.preferred_compute_device_counts).map(([name, count]) => `${name}:${count}`).join(" / ") || "none"],
      ["Estimated cost coverage", `${evidence.summary.estimated_cost_operation_count}/${evidence.structure.operation_count}; emitted-weight sum ${Number(evidence.summary.estimated_cost_weight_sum).toFixed(9)}`],
      ["Collector", `${evidence.capture.collector.name} ${evidence.capture.collector.version}; source SHA-256 ${evidence.capture.collector.source_sha256}; capture ${evidence.capture.capture_id}; ${evidence.capture.collected_at}`],
      ["Claim boundary", evidence.boundary],
    ]),
    rows.length ? markdownTable(["Op", "Type", "Identity", "Preferred", "Supported", "Estimated cost weight"], rows) : "No compute-plan operation rows were imported.",
  ].join("\n\n");
}

function ggufRuntimeEnvironmentMarkdown(evidence) {
  const buildOptions = Object.entries(evidence.build.options)
    .map(([name, enabled]) => `${name}=${enabled ? "ON" : "OFF"}`)
    .join(" / ");
  const graph = evidence.compute_graph || null;
  const graphRows = (graph?.graphs || []).map((item) => [
    item.sequence,
    item.graph_uid_decimal,
    `${item.original_node_count} / ${item.scheduled_node_count} / ${item.scheduler_inserted_node_count}`,
    item.split_count,
    `${item.original_backend_transition_edge_count} / ${item.scheduled_backend_transition_edge_count}`,
    item.dispatch_statuses.join(" / ") || "not dispatched",
    item.graph_sha256,
  ]);
  const splitRows = (graph?.graphs || []).flatMap((item) => item.splits.map((split) => [
    `${item.sequence}:${split.split_id}`,
    `[${split.start_node_index}, ${split.end_node_index})`,
    split.backend,
    split.input_count,
    split.inputs.join(" / ") || "none",
  ]));
  return [
    "### GGUF Instrumented Runtime And Scheduler Graph",
    markdownTable(["Field", "Bound evidence"], [
      ["Evidence class", `${evidence.evidence_class}; ${evidence.runtime_identity_status}; graph assignment ${evidence.graph_assignment_status}`],
      ["Artifact", `GGUF SHA-256 ${evidence.artifact.sha256}`],
      ["Runtime source", `${evidence.runtime.repository}@${evidence.runtime.source_commit}; ${evidence.runtime.source_alignment}`],
      ["Runtime binary", `SHA-256 ${evidence.runtime.binary_sha256}`],
      ["CMake cache", `SHA-256 ${evidence.build.cmake_cache_sha256}`],
      ["Selected-build attestation", `file SHA-256 ${evidence.build.attestation.file_sha256}; canonical SHA-256 ${evidence.build.attestation.canonical_sha256}; ${evidence.build.attestation.document.build.cmake_generator || "generator not named"}; ${evidence.build.attestation.document.build.build_type}; ${evidence.build.attestation.document.build.parallel_jobs} build job(s); timeout ${evidence.build.attestation.document.build.timeout_ms} ms`],
      ["Scheduler instrumentation", `${evidence.instrumentation.patch_id}; patch SHA-256 ${evidence.instrumentation.patch_sha256}; protocol ${evidence.instrumentation.trace_protocol}`],
      ["Build options", buildOptions],
      ["Compiled backends", evidence.build.compiled_backend_profile_ids.join(" / ") || "none"],
      ["Requested execution", `${evidence.selection.requested_backend_label}; context ${evidence.selection.context_size}; batch ${evidence.selection.batch_size}; ubatch ${evidence.selection.ubatch_size}; GPU layers ${evidence.selection.gpu_layers}; ${evidence.selection.selection_evidence_class}`],
      ["Device", `${evidence.device.platform} / ${evidence.device.architecture}; CPU features ${evidence.device.cpu_features.join(", ") || "not collected"}; accelerators ${evidence.device.accelerator_inventory.join(" / ") || "not observed"}`],
      ["Collector", `${evidence.capture.collector.name} ${evidence.capture.collector.version}; capture ${evidence.capture.capture_id}; ${evidence.capture.collected_at}`],
      ["Observed process status", `backend inventory ${evidence.observations.backend_inventory_status}; model load ${evidence.observations.model_load_status}; inference ${evidence.observations.inference_status}; selected backend ${evidence.observations.selected_backend_observation}`],
      ["Process witness", evidence.observations.inference_status === "not_run" ? "not run" : `exit ${evidence.observations.process_exit_code}; elapsed ${Number(evidence.observations.elapsed_ms).toFixed(3)} ms; stdout SHA-256 ${evidence.observations.stdout_sha256}; stderr SHA-256 ${evidence.observations.stderr_sha256}`],
      ["Scheduler graph witness", graph ? `${graph.dispatched_graph_count}/${graph.graph_count} graph call(s) dispatched; dispatch status success/failure ${graph.successful_dispatch_count}/${graph.failed_dispatch_count}; original ${graph.original_node_count} node(s); scheduled ${graph.scheduled_node_count}; inserted ${graph.scheduler_inserted_node_count}; splits ${graph.split_count}; backend transition edges original/scheduled ${graph.original_backend_transition_edge_count}/${graph.scheduled_backend_transition_edge_count}; unresolved source edges ${graph.original_backend_unresolved_source_edge_count}/${graph.scheduled_backend_unresolved_source_edge_count}; evidence SHA-256 ${graph.evidence_sha256}` : "not captured"],
      ["Conclusion", evidence.compatibility_conclusion],
      ["Claim boundary", evidence.boundary],
    ]),
    graphRows.length ? markdownTable(["Graph", "GGML UID", "Original / scheduled / inserted nodes", "Splits", "Original / scheduled backend transitions", "Dispatch status", "Graph SHA-256"], graphRows) : "",
    splitRows.length ? markdownTable(["Graph:split", "Original node range", "Assigned backend", "Crossing inputs", "Input references"], splitRows) : "",
    graph ? `> ${graph.interpretation_boundary}` : "",
  ].join("\n\n");
}

export function runtimeArenaReconciliationMarkdown(runtimeMemory, reconciliation) {
  if (!runtimeMemory || !reconciliation) return "";
  const delta = reconciliation.peak_delta_bytes;
  const ratio = reconciliation.peak_to_projection_ratio;
  const summary = markdownTable(["Arena allocation evidence", "Value"], [
    ["Evidence", `${runtimeMemory.evidence_class}; ${reconciliation.evidence_class}; ${reconciliation.status}`],
    ["Pinned TensorFlow source", `tensorflow/tensorflow@${runtimeMemory.tensorflow_source_commit}`],
    ["Observed snapshots", `${formatNumber(runtimeMemory.snapshot_count)}; final snapshot ${formatNumber(reconciliation.runtime_snapshot_id)}`],
    ["Observed peak / final", `${formatBytes(runtimeMemory.peak_combined_arena_bytes)} / ${formatBytes(runtimeMemory.final_combined_arena_bytes)}`],
    ["Declared-shape projection", reconciliation.projected_combined_arena_bytes == null ? "NOT_ASSESSED" : formatBytes(reconciliation.projected_combined_arena_bytes)],
    ["Peak delta / ratio", delta == null ? "NOT_ASSESSED" : `${formatSignedBytes(delta)} / ${ratio == null ? "N/A" : `${ratio.toFixed(6)}x`}`],
    ["Root allocations", `projected ${formatNumber(reconciliation.projected_root_allocation_count)} / observed ${formatNumber(reconciliation.observed_root_allocation_count)} / joined ${formatNumber(reconciliation.matched_allocation_count)}`],
    ["Runtime-only / missing observed", `${formatNumber(reconciliation.runtime_only_allocation_count)} / ${formatNumber(reconciliation.missing_observed_allocation_count)}`],
    ["Prepare-time runtime temporaries", `${formatNumber(reconciliation.runtime_temporary_allocation_count)} allocation(s); ${formatBytes(reconciliation.runtime_temporary_interval_bytes)}`],
    ["Size / offset / alias differences", `${formatNumber(reconciliation.size_mismatch_count)} / ${formatNumber(reconciliation.offset_mismatch_count)} / ${formatNumber(reconciliation.alias_mismatch_count)}`],
    ["Allocation ledger SHA-256", runtimeMemory.allocation_ledger_sha256],
  ]);
  const allDifferences = (reconciliation.allocation_rows || [])
    .filter((row) => !row.projected_present || !row.observed_present || row.size_match === false || row.arena_match === false || row.offset_match === false);
  const differences = allDifferences.slice(0, 64)
    .map((row) => [
      `T${row.tensor_index} ${row.tensor_name || ""}`,
      row.artifact_tensor ? "artifact" : "runtime temporary",
      row.projected_present ? `${row.projected_arena} / ${formatBytes(row.projected_size_bytes)} @ ${formatBytes(row.projected_offset_bytes)}` : "not projected",
      row.observed_present ? `${row.observed_arena} / ${formatBytes(row.observed_size_bytes)} @ ${formatBytes(row.observed_offset_bytes)}` : "not observed",
      row.size_delta_bytes == null ? "N/A" : formatSignedBytes(row.size_delta_bytes),
    ]);
  return [
    "### Static Projection Vs Observed TFLite Arena",
    summary,
    differences.length ? markdownTable(["Tensor", "Origin", "Projected", "Observed", "Size delta"], differences) : "Every joined allocation matched arena, size, and offset in the final observed snapshot.",
    allDifferences.length > 64
      ? `> Difference table truncated: 64 / ${formatNumber(allDifferences.length)} difference rows shown; complete rows remain in engineering evidence.`
      : "",
    `> ${reconciliation.interpretation_boundary}`,
  ].filter(Boolean).join("\n\n");
}

export function benchmarkInputContractText(input = {}) {
  const shape = (value, absent) => Array.isArray(value) ? `[${value.length ? value.join("x") : "scalar"}]` : absent;
  return `${input.input_name}:${input.runtime_dtype}; artifact ${shape(input.declared_shape, "unavailable")} / signature ${shape(input.artifact_shape_signature, "not declared")} / runtime ${shape(input.runtime_declared_shape, "not separately observed")} / executed ${shape(input.executed_shape, "unavailable")} / ${input.element_count} elements / basis ${input.basis}`;
}

export function benchmarkOutputContractText(output = {}) {
  return benchmarkInputContractText({ ...output, input_name: output.output_name }).replace("basis observed_runtime_output", "basis observed runtime output");
}

export function benchmarkExternalDataBindingText(binding = null) {
  if (!binding) return "external data runtime binding not recorded";
  if (binding.status === "not_applicable") return `${binding.schema}; external data not applicable`;
  const files = (binding.files || []).map((file) => `${file.path}:${file.byte_length}B:SHA-256 ${file.sha256}`).join(" / ");
  return `${binding.schema}; ${binding.status}; ${binding.tensor_count} tensor range(s); ${binding.file_count} file(s); ${binding.total_file_bytes} B; ${files}`;
}

export function benchmarkSampleSeriesText(row = {}) {
  return (row.measured_samples_ms || []).map((value) => Number(value).toString()).join(", ");
}

export function benchmarkNoiseSummaryText(row = {}) {
  const noise = row.noise_diagnostics || {};
  return `trimmed p50 ${Number(noise.trimmedP50).toString()} ms; trimmed mean ${Number(noise.trimmedMean).toString()} ms; outliers ${noise.outlierCount}; isolated spikes ${noise.gcSpikeCount}; OLS slope ${Number(noise.trendSlope).toString()} ms/run; ${noise.trendLabel}`;
}

export function benchmarkSampleLedgerMarkdown(rows = []) {
  const measured = rows.filter((row) => row?.ok === true);
  if (!measured.length) return "";
  return [
    "### Raw Measured Latency Samples",
    "> Samples are milliseconds in execution order after warmup. Percentiles use nearest-rank index ceil(n*q)-1 after sorting; mean and population variance/stddev divide by n; CV is population stddev/mean. Noise diagnostics use mean + 2.5 population SD, isolated-spike adjacency, OLS slope in ms/run, and 10% two-sided trimming.",
    markdownTable(["Backend", "Statistics/noise methods", "Count", "Noise diagnostics", "Exact measured samples (ms)"], measured.map((row) => [row.backend, `${row.statistics_method} / ${row.noise_method}`, row.measured_samples_ms.length, benchmarkNoiseSummaryText(row), benchmarkSampleSeriesText(row)])),
  ].join("\n\n");
}

export function runtimeAssignmentComparisonMarkdown(comparison) {
  const placement = comparison?.placement_assessment || {};
  const mac = comparison?.mac_comparison || {};
  const duration = comparison?.duration_comparison || {};
  const boundaries = comparison?.boundary_comparison || {};
  const predicted = comparison?.predicted_boundary_inventory || {};
  const observed = comparison?.observed_boundary_inventory || {};
  const partitions = comparison?.observed_partitions || {};
  const percent = (value) => value == null ? "N/A" : formatPercent(value);
  const predictionApplicable = comparison?.prediction_applicability === "tflite_xnnpack_static_prediction";
  const observedPayload = observed.summed_edge_payload_bytes == null
    ? `PARTIAL: ${formatBytes(observed.assessed_edge_payload_bytes || 0)} across ${formatNumber(observed.edge_count || 0)} observed transition edge(s)`
    : `${formatBytes(observed.summed_edge_payload_bytes)} across ${formatNumber(observed.edge_count || 0)} observed transition edge(s); ${formatBytes(observed.unique_tensor_payload_bytes || 0)} unique tensor payload`;
  const summaryRows = predictionApplicable ? [
    ["Evidence", `${comparison.evidence_class || "DERIVED_FROM_OBSERVED_RUNTIME"}; ${comparison.status || "partial"}`],
    ["Placement coverage", `${formatNumber(placement.assessed_op_count || 0)} / ${formatNumber(comparison.graph_op_count || 0)} op(s) (${percent(placement.coverage_ratio)})`],
    ["Placement agreement", `${formatNumber(placement.match_count || 0)} / ${formatNumber(placement.assessed_op_count || 0)} (${percent(placement.match_ratio)}); overpredicted ${formatNumber(placement.overpredicted_delegation_count || 0)}, underpredicted ${formatNumber(placement.underpredicted_delegation_count || 0)}`],
    ["Conditionally delegatable / observed delegated ops", `${formatNumber(placement.predicted_delegated_op_count || 0)} / ${formatNumber(placement.observed_delegated_op_count || 0)} within classified rows`],
    ["MAC-weighted mismatch", `${formatNumber(mac.mismatch_macs || 0)} MACs / ${percent(mac.mismatch_mac_ratio)} of MAC-assessed classified rows; coverage ${percent(mac.coverage_ratio)}`],
    ["Boundary agreement", `${formatNumber(boundaries.match_count || 0)} / ${formatNumber(boundaries.assessed_edge_count || 0)} assessed graph edge(s) (${percent(boundaries.match_ratio)}); overpredicted ${formatNumber(boundaries.overpredicted_boundary_count || 0)}, underpredicted ${formatNumber(boundaries.underpredicted_boundary_count || 0)}`],
    ["Observed runtime partitions", `${formatNumber(partitions.partition_count || 0)}; explicit ${formatNumber(partitions.explicit_partition_count || 0)}, implicit contiguous ${formatNumber(partitions.implicit_contiguous_partition_count || 0)}, noncontiguous ID reuse ${formatNumber(partitions.noncontiguous_partition_id_count || 0)}`],
    ["Predicted interface logical payload", predicted.summed_edge_payload_bytes == null ? `PARTIAL: ${formatBytes(predicted.assessed_edge_payload_bytes || 0)} across ${formatNumber(predicted.edge_count || 0)} predicted boundary edge(s)` : `${formatBytes(predicted.summed_edge_payload_bytes)} across ${formatNumber(predicted.edge_count || 0)} predicted boundary edge(s); ${formatBytes(predicted.unique_tensor_payload_bytes || 0)} unique tensor payload`],
    ["Observed interface logical payload", observedPayload],
  ] : [
    ["Evidence", `${comparison.evidence_class || "DERIVED_FROM_OBSERVED_RUNTIME"}; observed provider assignment only`],
    ["Provider assignment coverage", `${formatNumber(placement.observed_assignment_count || 0)} / ${formatNumber(comparison.graph_op_count || 0)} original op(s) (${percent(placement.observed_assignment_coverage_ratio)})`],
    ["Provider-assigned / CPU ops", `${formatNumber(placement.observed_delegated_op_count || 0)} / ${formatNumber((placement.observed_assignment_count || 0) - (placement.observed_delegated_op_count || 0))}`],
    ["Static EP prediction", "NOT_APPLICABLE: no ONNX execution-provider rulepack was applied"],
    ["Observed graph-edge relations", `${formatNumber(boundaries.observed_relation_edge_count || 0)} / ${formatNumber(boundaries.graph_edge_count || 0)} edge(s); prediction agreement N/A`],
    ["Observed provider-transition logical payload", observedPayload],
    ["Runtime partitions", partitions.status === "not_assessed_no_partition_ids" ? `NOT_ASSESSED: profile exposed no partition IDs; ${formatNumber(partitions.provider_segment_count || 0)} contiguous provider segment(s) reported separately` : `${formatNumber(partitions.partition_count || 0)} explicit runtime partition(s)`],
  ];
  summaryRows.push(["Duration aggregation", `${duration.status || "not_assessed"}; semantics ${duration.duration_semantics || "unspecified"}; ${duration.total_duration_us == null ? "total not emitted" : `${Number(duration.total_duration_us).toFixed(3)} us total`}`]);
  if (duration.duration_semantics === "per_execution_plan_node_exclusive") {
    summaryRows.push(
      ["Execution-node timing coverage", `${formatNumber(duration.mapped_execution_node_count || 0)} / ${formatNumber(duration.execution_plan_node_count || 0)}`],
      ["CPU execution-node subtotal", duration.cpu_execution_node_subtotal_us == null ? "N/A" : `${Number(duration.cpu_execution_node_subtotal_us).toFixed(3)} us`],
      ["Delegate partition subtotal", duration.delegate_partition_subtotal_us == null ? "N/A" : `${Number(duration.delegate_partition_subtotal_us).toFixed(3)} us`],
      ["Primary delegate-profiled subtotal", duration.primary_delegate_profiled_subtotal_us == null ? "N/A" : `${Number(duration.primary_delegate_profiled_subtotal_us).toFixed(3)} us; delegate-internal IDs, not assigned to original ops or partitions`],
      ["Nested delegate-internal subtotal", duration.delegate_internal_section_subtotal_us == null ? "N/A" : `${Number(duration.delegate_internal_section_subtotal_us).toFixed(3)} us; not added to partition totals`],
    );
  }
  const summary = markdownTable([predictionApplicable ? "Runtime assignment comparison" : "Runtime provider evidence", "Value"], summaryRows);
  const mismatchRows = (comparison?.mismatches || []).slice(0, 64).map((item) => [
    `#${padOp(item.op_index)} ${item.op_name}`,
    item.predicted_domain,
    item.observed_delegated == null ? "not assessed" : item.observed_delegated ? `delegate: ${item.observed_provider || "provider unknown"}${item.observed_partition_id != null ? ` / P${item.observed_partition_id}` : ""}` : `CPU: ${item.observed_provider || "provider unknown"}`,
    item.classification,
    item.macs == null ? "N/A" : formatNumber(item.macs),
    item.duration_us == null ? "N/A" : `${Number(item.duration_us).toFixed(3)} us`,
  ]);
  const boundaryRows = (comparison?.boundary_differences || []).slice(0, 64).map((edge) => [
    `T${edge.tensor_index} ${edge.tensor_name || "-"}`,
    `#${padOp(edge.producer_op_index)} ${edge.producer_op_name} -> #${padOp(edge.consumer_op_index)} ${edge.consumer_op_name}`,
    `${edge.predicted_producer_domain} -> ${edge.predicted_consumer_domain}`,
    `${edge.observed_producer_domain || "not assessed"} -> ${edge.observed_consumer_domain || "not assessed"}`,
    edge.classification,
    edge.payload_bytes == null ? "N/A" : formatBytes(edge.payload_bytes),
  ]);
  return [
    predictionApplicable ? "### Predicted Vs Observed Runtime Assignment" : "### Observed Runtime Provider Assignment",
    summary,
    predictionApplicable
      ? mismatchRows.length ? markdownTable(["Op", "Predicted", "Observed", "Classification", "MACs", "Duration"], mismatchRows) : "No classified op-placement mismatch was observed."
      : "Static provider placement prediction was not performed; mismatch classification is not applicable.",
    (comparison?.mismatches || []).length > 64 ? `> Mismatch table truncated: 64 / ${formatNumber(comparison.mismatches.length)} rows shown; complete rows remain in engineering evidence.` : "",
    predictionApplicable
      ? boundaryRows.length ? markdownTable(["Tensor", "Graph edge", "Predicted domains", "Observed domains", "Classification", "Logical payload"], boundaryRows) : "No assessed graph-edge boundary mismatch was observed."
      : "Observed provider-transition edges are reported as runtime evidence without a predicted-boundary comparison.",
    (comparison?.boundary_differences || []).length > 64 ? `> Boundary-delta table truncated: 64 / ${formatNumber(comparison.boundary_differences.length)} rows shown; complete rows remain in engineering evidence.` : "",
    comparison?.source_adapter_schema?.startsWith("deepbom.tflite_runtime_info_adapter.v")
      ? "> Imported TFLite ModelRuntimeDetails confirms execution-plan placement and explicit delegate partitions when present. BenchmarkProfilingData timing is included only when separately imported and capture-bound; delegate-internal rows are never relabeled as original-op or partition timing. It does not expose executed microkernel symbols or tensor-copy materialization. Logical interface payload is DERIVED from declared tensor shape and dtype; aliasing, synchronization, and transition latency remain not assessed."
      : "> Observed assignment confirms provider mapping only. A null partition ID or kernel means the imported source did not expose it; neither is inferred from an ONNX Runtime node event. Logical interface payload is DERIVED from declared tensor shape and dtype; runtime copy materialization, aliasing, synchronization, and transition latency remain not assessed.",
  ].filter(Boolean).join("\n\n");
}

export function runtimeEvidenceMarkdown({
  runtimeBenchmarkResults = [],
  runtimeBasinResult = null,
  runtimeEvidence = {},
} = {}) {
  const consequence = runtimeEvidence.preprocessingConsequenceResult || runtimeEvidence.preprocessing_consequence_atlas || null;
  const calibration = runtimeEvidence.calibrationValidationResult || runtimeEvidence.representative_dataset_validation || null;
  const rows = runtimeBenchmarkResults.map((row) => [
    row.backend,
    row.ok ? "MEASURED_SYNTHETIC" : "ATTEMPTED",
    row.ok ? "ok" : "failed",
    row.compile_ms == null ? "-" : `${Number(row.compile_ms).toFixed(2)} ms`,
    row.first_run_ms == null ? "-" : `${Number(row.first_run_ms).toFixed(2)} ms`,
    row.stats?.p50 == null ? "-" : `${Number(row.stats.p50).toFixed(2)} ms`,
    row.stats?.p90 == null ? "-" : `${Number(row.stats.p90).toFixed(2)} ms`,
    row.stats?.p95 == null ? "-" : `${Number(row.stats.p95).toFixed(2)} ms`,
    row.stats?.p99 == null ? "-" : `${Number(row.stats.p99).toFixed(2)} ms`,
    row.stats?.mean == null ? "-" : `${Number(row.stats.mean).toFixed(2)} ms`,
    row.steady_stats?.p50 == null ? "-" : `${Number(row.steady_stats.p50).toFixed(2)} ms`,
    row.stats?.cv == null ? "-" : formatPercent(row.stats.cv),
    row.ok ? [
      `${row.output_count} outputs`,
      ...(row.output_contracts || []).map(benchmarkOutputContractText),
      row.input_basis,
      ...(row.input_contracts || []).map(benchmarkInputContractText),
      row.external_data_runtime_binding ? benchmarkExternalDataBindingText(row.external_data_runtime_binding) : "",
      row.onnx_runtime_shape_binding
        ? `ONNX shape binding ${row.onnx_runtime_shape_binding.status}; ${row.onnx_runtime_shape_binding.bound_symbol_count}/${row.onnx_runtime_shape_binding.symbol_count} symbols bound; total MACs ${row.onnx_runtime_shape_binding.evaluated_total_macs_decimal ?? "not evaluated"}; conflicts ${row.onnx_runtime_shape_binding.conflict_count}`
        : "",
      row.timing_method || "timing method not recorded",
      row.phase_counts ? `cold ${row.phase_counts.cold_first_runs} / warmup ${row.phase_counts.warmup_runs} / measured ${row.phase_counts.measured_runs}` : "phase counts not recorded",
      row.p99_evidence?.detail || "p99 sample adequacy not recorded",
      row.output_digest ? `digest ${row.output_digest.slice(0, 16)}` : "",
    ].filter(Boolean).join(" / ") : row.error || "-",
  ]);
  const basinRows = runtimeBasinResult?.attempted_backends?.map((item) => [
    item.backend,
    item.ok ? "complete" : "failed",
    item.run_ms == null ? "-" : `${Number(item.run_ms).toFixed(2)} ms`,
    item.drift?.max_abs == null ? "-" : `${formatDrift(item.drift.max_abs)} ${runtimeBasinResult.output_profile?.unit || ""}`,
    item.interpretation || "-",
  ]) || [];
  return [
    runtimeEnvironmentMarkdown(runtimeEvidence, runtimeBenchmarkResults),
    "",
    runtimeBenchmarkResults.length
      ? markdownTable(["Backend", "Evidence", "Status", "Compile", "First", "p50", "p90", "p95", "p99", "Mean", "Steady p50", "Jitter", "Notes"], rows)
      : "No repeated Runtime Benchmark has been run in this browser session.",
    benchmarkSampleLedgerMarkdown(runtimeBenchmarkResults),
    "",
    runtimeBasinResult
      ? markdownTable(["Backend", "Status", "Run", "Max drift", "Interpretation"], basinRows)
      : `No ${RUNTIME_COMPATIBILITY_EVIDENCE_LABEL} module result is available.`,
    "",
    preprocessingConsequenceMarkdown(consequence),
    "",
    calibrationValidationMarkdown(calibration),
  ].join("\n");
}

export function calibrationValidationMarkdown(ledger, headingLevel = "###") {
  if (!ledger) return `${headingLevel} Representative Dataset Validation\nNot imported in this browser session; no representative-data saturation, reference drift, or repeated-run determinism claim is made.`;
  const endpoint = ledger.input_endpoint_saturation || {};
  const drift = ledger.reference_output_drift || {};
  const repeat = ledger.repeat_nondeterminism || {};
  const rows = (ledger.samples || []).map((sample) => {
    const referenceMax = maximumFinite((sample.reference_comparisons || []).map((row) => row.maximum_absolute_difference));
    const repeatMax = maximumFinite((sample.repeat_comparisons || []).map((row) => row.maximum_absolute_difference));
    return [
      sample.sample_id,
      sample.run_count,
      sample.input_endpoint_saturation?.endpoint_ratio == null ? "N/A" : formatPercent(sample.input_endpoint_saturation.endpoint_ratio),
      referenceMax == null ? "not provided" : formatDrift(referenceMax),
      repeatMax == null ? "not assessed" : formatDrift(repeatMax),
    ];
  });
  return [
    `${headingLevel} Representative Dataset Validation (DERIVED_FROM_HASH_BOUND_CAPTURED_DATASET)`,
    `Dataset ${code(`${ledger.dataset?.id}@${ledger.dataset?.version}`)}; manifest SHA-256 ${code(ledger.dataset?.manifest_sha256)}; runtime ${code(`${ledger.runtime?.name}@${ledger.runtime?.version}/${ledger.runtime?.backend}`)}; source capture SHA-256 ${code(ledger.source_capture_sha256)}; ledger SHA-256 ${code(ledger.ledger_sha256)}.`,
    `Integer interface endpoint saturation: ${endpoint.status}; ${formatNumber(endpoint.endpoint_count || 0)} / ${formatNumber(endpoint.assessed_value_count || 0)} assessed values${endpoint.endpoint_ratio == null ? "" : ` (${formatPercent(endpoint.endpoint_ratio)})`}. Reference drift: ${drift.status}; ${formatNumber(drift.changed_value_count || 0)} / ${formatNumber(drift.compared_value_count || 0)} values changed; maximum absolute difference ${drift.maximum_absolute_difference == null ? "N/A" : formatDrift(drift.maximum_absolute_difference)}. Repeat nondeterminism: ${repeat.status}; ${formatNumber(repeat.changed_value_count || 0)} / ${formatNumber(repeat.compared_value_count || 0)} values changed; maximum absolute difference ${repeat.maximum_absolute_difference == null ? "N/A" : formatDrift(repeat.maximum_absolute_difference)}.`,
    "",
    markdownTable(["Sample", "Runs", "Input endpoints", "Reference max |delta|", "Repeat max |delta|"], rows),
    "",
    ledger.interpretation_boundary,
  ].join("\n");
}

function maximumFinite(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  return finite.length ? Math.max(...finite) : null;
}

export function preprocessingConsequenceMarkdown(consequence, headingLevel = "###") {
  if (!consequence) return `${headingLevel} Preprocessing Consequence Atlas\nNot run in this browser session; no output consequence is asserted for the static preprocessing counterfactuals.`;
  const rows = (consequence.candidates || []).map((row) => [
    row.contract_id,
    row.source_exact_tensor_realization ? "exact source" : "minimum-error source",
    row.input_changed_element_count,
    row.output_changed_element_count,
    row.output_maximum_absolute_difference,
    row.first_output_top1_changed ? `${row.baseline_first_output_top1_index} -> ${row.first_output_top1_index}` : String(row.first_output_top1_index),
    row.output_tensor_set_sha256.slice(0, 16),
  ]);
  return [
    `${headingLevel} Preprocessing Consequence Atlas (MEASURED_SYNTHETIC)`,
    `Schema ${code(consequence.schema)}; method ${code(consequence.method_version)}; runtime ${code(`${consequence.runtime?.name}@${consequence.runtime?.version}/${consequence.runtime?.backend}`)}; portfolio SHA-256 ${code(consequence.portfolio_ledger_sha256)}.`,
    `Two byte-compared executions were captured for the tensor-ABI witness and each of ${formatNumber(consequence.candidate_count)} counterfactual contract inputs. They formed ${formatNumber(consequence.unique_input_tensor_count)} unique input tensor(s) and ${formatNumber(consequence.unique_output_tensor_set_count)} unique output tensor set(s); ${formatNumber(consequence.output_changed_candidate_count)} candidate(s) changed at least one output value and ${formatNumber(consequence.top1_changed_candidate_count)} changed the raw first-output argmax index.`,
    "",
    markdownTable(["Contract", "Input basis", "Input changed", "Output changed", "Max |delta|", "Raw top-1", "Output SHA"], rows),
    "",
    consequence.interpretation_boundary,
  ].join("\n");
}

function yesNo(value) {
  return value ? "yes" : "no";
}

export function weightIndicatorMarkdown({
  deepBomResult = null,
  perturbationResult = null,
  deployCurvatureResult = null,
} = {}) {
  return markdownTable(["Indicator", "Evidence", "Result", "Boundary"], [
    ["DEEPBOM weight/topology proxy", "PROXY", deepBomResult ? (deepBomResult.score_assessment?.status === "ASSESSED" ? `basin=${score100(deepBomResult.basin_proxy_score)}, topology=${score100(deepBomResult.topology_stress_score)}` : `NOT_ASSESSABLE: ${deepBomResult.score_assessment?.reason || "required score inputs unavailable"}`) : "not run", "Not a generalization proof."],
    ["Input perturbation drift", "MEASURED_SYNTHETIC", perturbationResult?.drift ? `max=${formatDrift(perturbationResult.drift.max_abs)} ${perturbationResult.output_profile?.unit || ""}` : "not run", "Synthetic/prepared local inputs only."],
    ["Weight perturbation", "MEASURED_SYNTHETIC", perturbationResult?.weight_perturbation ? `${perturbationResult.weight_perturbation.touched_tensors} tensor(s)` : "not available/not run", "Local model-byte copy; raw weights not exported."],
    ["Research deployment-sensitivity proxy", "MEASURED_SYNTHETIC", deployCurvatureResult?.basin ? `${Number(deployCurvatureResult.basin.score || 0).toFixed(1)} / 100` : "not run", "Deployment-function local finite difference; not Hessian/PAC-Bayes."],
  ]);
}
