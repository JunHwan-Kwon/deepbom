import { ANALYZER_METADATA } from "./report-metadata.js";
import { validateFindingEvidenceBindings } from "./finding-contract.js";
import { validateQuantizationLattice } from "./quantization-lattice.js";
import { validateAccumulatorReachabilityShape } from "./accumulator-reachability.js";
import {
  reconstructNumericalAbiPropagation,
  validateNumericalAbiPropagationAgainstReconstruction,
  validateNumericalAbiPropagationShape,
} from "./numerical-abi-propagation.js";
import {
  buildInputWitnessTensor,
  validateInputCounterexampleShape,
} from "./input-counterexample.js";
import { buildMetricCoverageEntries, validateMetricCoverageManifest } from "./metric-coverage.js";
import { BROWSER_BENCHMARK_TIMING_METHOD } from "./runtime.js";
import { benchmarkExternalDataBindingText, benchmarkInputContractText, benchmarkNoiseSummaryText, benchmarkOutputContractText, benchmarkSampleSeriesText } from "./report-sections.js";
import { assessOrtRuntimeAdapter } from "./report-conformance-ort-runtime.js";
import { buildOnnxRuntimeShapeBinding } from "./onnx-runtime-shape-binding.js";
import { buildRuntimeDataMovementEvidence } from "./runtime-data-movement-evidence.js";
import { registerTfliteStorageConformance } from "./report-conformance-tflite-storage.js";
import { registerCoreMlSerializedConformance } from "./report-conformance-coreml.js";
import {
  registerGgufSerializedConformance,
  registerSafeTensorsSerializedConformance,
} from "./report-conformance-serialized-containers.js";
import { verifyDynamicShapeCostEvidence } from "./dynamic-shape-cost-conformance.js";
import { registerExecutionPlacementConformance } from "./report-conformance-execution-placement.js";
import { registerOnnxConformance } from "./report-conformance-onnx.js";
import { validateOnDeviceLlmContract } from "./on-device-llm-contract.js";
import { buildExecuTorchSelectedBuildBinding } from "./executorch-build-binding.js";
import { canonicalJson } from "./report-utils.js";
import {
  buildQuantResearchCoverage,
  QUANT_RESEARCH_COVERAGE_SCHEMA,
} from "./quant-research-applicability.js";
import {
  compactMlBomEvidencePointerValid,
  formatIntegerForConformance,
} from "./report-conformance-common.js";
import {
  arenaAllocationsDoNotConflict,
  boundaryEdgeKey,
  collectAssessmentMetricFailures,
  deriveArenaProjection,
  deriveCpuIslandRanges,
  deriveDeclaredLiveness,
  deriveDelegationChanges,
  deriveDelegationRepairGraph,
  deriveDelegationState,
  deriveMovementPayload,
  derivePredictedBoundaryEdges,
  deriveRuntimeBoundaryComparisonEdges,
  deriveRuntimeOpComparisons,
  deriveRuntimePartitionSummary,
  independentCommonRunSubtotal,
  nullableClose,
  reconstructRuntimeArenaEvidence,
  sameArray,
  sameDelegationChanges,
  sameDelegationSummary,
  sameRuntimeBoundarySet,
  sameRuntimePartitionInventory,
  summarizeRuntimeBoundaryComparison,
  summarizeRuntimeBoundaryPayload,
  summarizeRuntimePlacement,
  runtimeBoundaryKey,
  validBenchmarkInputContract,
  validBenchmarkOutputContract,
  validBenchmarkStatistics,
  validIsoTimestamp,
  validOnnxBenchmarkExternalDataBinding,
} from "./report-conformance-runtime-helpers.js";

const QUANT_RESEARCH_CONFORMANCE_PREFIXES = new Map([
  ["CF-ACCUMULATOR-", "accumulator_atlas"],
  ["CF-REQUANT-", "requantization_fidelity"],
  ["CF-WITNESS-", "kernel_extremum_witness"],
  ["CF-VITALITY-", "channel_vitality"],
  ["CF-ROUND-", "rounding_equivalence"],
  ["CF-REACH-", "accumulator_reachability"],
  ["CF-ABI-", "numerical_abi_propagation"],
  ["CF-IW-", "input_counterexample"],
  ["CF-PR-", "preprocessing_realizability"],
  ["CF-PC-", "preprocessing_consequence"],
  ["CF-LATTICE-", "quantization_lattice"],
  ["CF-MIGRATION-", "contract_migration"],
  ["CF-STEP-", "residual_step_response"],
  ["CF-DISTORTION-", "residual_contract_distortion"],
]);

const TFLITE_INPUT_LAYOUT_OPS = new Set([
  "CONV_2D", "DEPTHWISE_CONV_2D", "AVERAGE_POOL_2D", "MAX_POOL_2D",
  "RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR",
]);
const ONNX_INPUT_LAYOUT_OPS = new Set([
  "AveragePool", "BatchNormalization", "Conv", "ConvInteger", "ConvTranspose", "GlobalAveragePool",
  "GlobalLpPool", "GlobalMaxPool", "InstanceNormalization", "LpPool", "MaxPool",
  "QLinearConv",
]);

function inputCodeRange(dtype) {
  return ({
    UINT8: [0, 255], INT8: [-128, 127], UINT16: [0, 65_535], INT16: [-32_768, 32_767],
    INT32: [-2_147_483_648, 2_147_483_647], INT4: [-8, 7], UINT4: [0, 15], UINT2: [0, 3], INT2: [-2, 1],
  })[String(dtype || "").toUpperCase()] || null;
}

function independentlyReconstructInputContract(tensor, ops, onnx) {
  const shape = Array.isArray(tensor?.shape) ? [...tensor.shape] : [];
  const semanticOps = onnx ? ONNX_INPUT_LAYOUT_OPS : TFLITE_INPUT_LAYOUT_OPS;
  const semanticConsumer = shape.length === 4
    ? (ops || []).find((op) => op?.inputs?.[0] === tensor.index
      && (!onnx || op.standard_domain === true)
      && semanticOps.has(String(op.name || "")))
    : null;
  const layout = semanticConsumer ? onnx ? "NCHW" : "NHWC" : null;
  const channelAxis = semanticConsumer ? onnx ? 1 : 3 : null;
  const channels = channelAxis == null || !Number.isInteger(shape[channelAxis]) ? null : shape[channelAxis];
  const layoutStatus = shape.length !== 4
    ? "not_applicable_non_4d_input"
    : semanticConsumer ? `derived_${layout.toLowerCase()}_from_direct_consumer_semantics`
      : "not_assessed_no_direct_layout_semantic_consumer";
  const layoutEvidenceClass = shape.length !== 4 ? "NOT_APPLICABLE" : semanticConsumer ? "DERIVED" : "NOT_ASSESSABLE";
  const scaleCount = Number(tensor?.quant_scales || 0);
  const zeroPointCount = Number(tensor?.quant_zero_points || 0);
  const isQuantized = scaleCount > 0;
  const bounds = inputCodeRange(tensor?.dtype);
  const scale = tensor?.scale_sample?.[0];
  const zeroPoint = tensor?.zero_point_sample?.[0];
  let numericalStatus = "not_embedded_in_artifact";
  let low = null;
  let high = null;
  if (isQuantized && (scaleCount !== 1 || zeroPointCount !== 1)) {
    numericalStatus = "not_assessed_non_scalar_input_quantization";
  } else if (isQuantized && !bounds) {
    numericalStatus = "not_assessed_unsupported_quantized_input_dtype";
  } else if (isQuantized && (!Number.isFinite(scale) || scale <= 0 || !Number.isSafeInteger(zeroPoint)
    || zeroPoint < bounds[0] || zeroPoint > bounds[1])) {
    numericalStatus = "invalid_or_incomplete_quantization_metadata";
  } else if (isQuantized) {
    numericalStatus = "known_from_artifact_quantization_metadata";
    low = scale * (bounds[0] - zeroPoint);
    high = scale * (bounds[1] - zeroPoint);
  }
  const risks = [];
  if (tensor?.dtype === "FLOAT32" && !isQuantized) risks.push("FLOAT32 normalization range is unknown and must match the training pipeline exactly.");
  if (!layout && shape.length === 4) risks.push("Input layout and channel axis are not determined by the supported direct-consumer semantics.");
  if (channels === 3) risks.push("Channel order (RGB vs BGR) cannot be determined from graph structure; a mismatch may silently degrade task performance.");
  return {
    schema: "deepbom.input_tensor_contract.v1",
    tensor_index: tensor?.index,
    name: tensor?.name || "",
    shape,
    dtype: tensor?.dtype || "UNKNOWN",
    is_quantized: isQuantized,
    expected_range_low: low,
    expected_range_high: high,
    tensor_numerical_contract_status: numericalStatus,
    source_data_to_tensor_preprocessing_status: "not_embedded_in_artifact",
    layout,
    layout_status: layoutStatus,
    layout_evidence_class: layoutEvidenceClass,
    layout_source_op_index: semanticConsumer?.index ?? null,
    layout_source_op_name: semanticConsumer?.name ?? null,
    channel_axis: channelAxis,
    channels,
    risks,
  };
}

function finiteOrNullEqual(actual, expected) {
  if (expected == null) return actual == null;
  if (!Number.isFinite(actual)) return false;
  return Math.abs(actual - expected) <= Math.max(1e-12, Math.abs(expected) * 1e-12);
}

function inputContractMatchesReconstruction(contract, expected) {
  return contract?.schema === expected.schema
    && contract.tensor_index === expected.tensor_index
    && contract.name === expected.name
    && JSON.stringify(contract.shape) === JSON.stringify(expected.shape)
    && contract.dtype === expected.dtype
    && contract.is_quantized === expected.is_quantized
    && finiteOrNullEqual(contract.expected_range_low, expected.expected_range_low)
    && finiteOrNullEqual(contract.expected_range_high, expected.expected_range_high)
    && contract.tensor_numerical_contract_status === expected.tensor_numerical_contract_status
    && contract.source_data_to_tensor_preprocessing_status === expected.source_data_to_tensor_preprocessing_status
    && (contract.layout ?? null) === expected.layout
    && contract.layout_status === expected.layout_status
    && contract.layout_evidence_class === expected.layout_evidence_class
    && (contract.layout_source_op_index ?? null) === expected.layout_source_op_index
    && (contract.layout_source_op_name ?? null) === expected.layout_source_op_name
    && (contract.channel_axis ?? null) === expected.channel_axis
    && (contract.channels ?? null) === expected.channels
    && JSON.stringify(contract.risks || []) === JSON.stringify(expected.risks)
    && typeof contract.range_note === "string" && contract.range_note.length > 0
    && typeof contract.layout_reason === "string" && contract.layout_reason.length > 0;
}

function mlBomInputContractRows(contracts) {
  return (contracts || []).map((contract) => ({
    schema: contract.schema || "not_emitted", tensor_index: contract.tensor_index, name: contract.name || "",
    dtype: contract.dtype || "UNKNOWN", shape: contract.shape || [], layout: contract.layout ?? null,
    layout_status: contract.layout_status || "not_assessed", layout_evidence_class: contract.layout_evidence_class || "NOT_ASSESSABLE",
    layout_source_op_index: contract.layout_source_op_index ?? null, layout_source_op_name: contract.layout_source_op_name ?? null,
    channel_axis: contract.channel_axis ?? null, channels: contract.channels ?? null,
    tensor_numerical_contract_status: contract.tensor_numerical_contract_status || "not_assessed",
    expected_range_low: contract.expected_range_low ?? null, expected_range_high: contract.expected_range_high ?? null,
    source_data_to_tensor_preprocessing_status: contract.source_data_to_tensor_preprocessing_status || "not_assessed",
    risks: contract.risks || [],
  }));
}

function registerSerializedRuntimeConformance({ format, staticAnalysis, runtimeResults, reportText, check }) {
  const runtimeAssignment = runtimeResults?.runtime_assignment || null;
  if (format === "gguf") {
    const runtimeEnvironment = runtimeResults?.runtime_environment || null;
    check("CF-RUNTIME-ENV-001", !runtimeEnvironment || (runtimeEnvironment.schema === "deepbom.gguf_runtime_environment.v2"
      && runtimeEnvironment.artifact?.sha256 === staticAnalysis?.model_sha256
      && /^[a-f0-9]{64}$/.test(runtimeEnvironment.runtime?.binary_sha256 || "")
      && /^[a-f0-9]{64}$/.test(runtimeEnvironment.build?.cmake_cache_sha256 || "")
      && runtimeEnvironment.runtime_identity_status === "bound"
      && runtimeEnvironment.graph_assignment_status === "observed_generated_scheduler_graphs"
      && runtimeEnvironment.compute_graph?.dispatched_graph_count > 0
      && /^[a-f0-9]{64}$/.test(runtimeEnvironment.build?.attestation?.canonical_sha256 || "")
      && runtimeEnvironment.build?.attestation?.document?.binary?.sha256 === runtimeEnvironment.runtime?.binary_sha256
      && runtimeEnvironment.build?.compiled_backend_profile_ids?.includes(runtimeEnvironment.selection?.requested_backend_profile_id)),
    "Imported GGUF runtime evidence must bind the active artifact, pinned instrumented build, binary, complete backend inventory, requested backend, and at least one dispatched scheduler graph.", ["/evidence/runtime_results/runtime_environment", "/evidence/static_analysis"]);
    check("CF-RUNTIME-ENV-002", !runtimeEnvironment || (!runtimeAssignment
      && reportText.includes("GGUF Instrumented Runtime And Scheduler Graph")
      && reportText.includes(runtimeEnvironment.runtime.binary_sha256)
      && reportText.includes(runtimeEnvironment.compute_graph?.evidence_sha256 || "missing")),
    "GGUF runtime scheduler evidence must remain separate from serialized graph analysis and disclose binary plus graph identities in the engineering report.", ["/evidence/runtime_results/runtime_environment", "/evidence/runtime_results/runtime_assignment", "/engineering_report.md"]);
    return;
  }
  if (format !== "coreml") return;
  const plan = runtimeResults?.coreml_compute_plan || null;
  const functionNames = Object.keys(staticAnalysis?.coreml?.ml_program?.functions || {});
  const expectedFunction = staticAnalysis?.coreml?.description?.default_function_name
    || (functionNames.includes("main") ? "main" : functionNames.length === 1 ? functionNames[0] : null);
  const availableDevices = new Set((plan?.runtime?.available_compute_devices || []).map((device) => device.type));
  check("CF-RUNTIME-CML-001", !plan || (plan.schema === "deepbom.coreml_compute_plan.v1"
    && plan.artifact?.sha256 === staticAnalysis?.model_sha256
    && /^[a-f0-9]{64}$/.test(plan.runtime?.compiled_model_content_sha256 || "")
    && /^[a-f0-9]{64}$/.test(plan.runtime?.coremltools_compute_plan_source_sha256 || "")
    && /^[a-f0-9]{64}$/.test(plan.capture?.collector?.source_sha256 || "")
    && plan.runtime?.platform_system === "Darwin"
    && [plan.runtime?.macos_version, plan.runtime?.os_build, plan.runtime?.hardware_model, plan.runtime?.python_version].every((value) => String(value || "").trim())
    && availableDevices.size === plan.runtime?.available_compute_devices?.length
    && availableDevices.size > 0
    && plan.runtime.available_compute_devices.every((device) => Number.isSafeInteger(device.instance_count) && device.instance_count > 0
      && (device.type !== "NEURAL_ENGINE" || Number.isSafeInteger(device.total_core_count) && device.total_core_count > 0))
    && (plan.structure?.kind === "program" ? plan.configuration?.function_name === expectedFunction : plan.configuration?.function_name == null)
    && plan.execution_status === "not_observed_compute_plan_only"
    && plan.structure?.rows?.length === staticAnalysis?.ops?.length
    && plan.structure.rows.every((row, index) => row.op_index === index
      && String(row.identity) === String(staticAnalysis.ops[index]?.coreml_layer_name || "")
      && String(row.operator_type).replaceAll(/[^a-z0-9]/gi, "").toLowerCase() === String(staticAnalysis.ops[index]?.mil_operation_type || staticAnalysis.ops[index]?.name || "").replaceAll(/[^a-z0-9]/gi, "").toLowerCase()
      && (!row.preferred_compute_device || row.supported_compute_devices?.includes(row.preferred_compute_device))
      && (!row.preferred_compute_device || availableDevices.has(row.preferred_compute_device))
      && (row.supported_compute_devices || []).every((device) => availableDevices.has(device))
      && (row.estimated_cost_weight == null || Number(row.estimated_cost_weight) >= 0 && Number(row.estimated_cost_weight) <= 1))),
  "Core ML compute-plan evidence must bind artifact, compiled model, host, collector, function, and every decoded operation without claiming execution.", ["/evidence/runtime_results/coreml_compute_plan", "/evidence/static_analysis/ops"]);
  check("CF-RUNTIME-CML-002", !plan || (!runtimeAssignment
    && reportText.includes("Core ML MLComputePlan Estimate")
    && reportText.includes("not_observed_compute_plan_only")
    && reportText.includes(plan.runtime.compiled_model_content_sha256)
    && reportText.includes(plan.runtime.os_build)
    && reportText.includes(plan.capture.collector.source_sha256)),
  "Core ML compute-plan evidence must remain separate from runtime assignment and preserve its estimate-only boundary and host/collector identity in the report.", ["/evidence/runtime_results/coreml_compute_plan", "/evidence/runtime_results/runtime_assignment", "/engineering_report.md"]);
}

function buildSerializedArtifactConformanceReport({
  analysis,
  staticAnalysis,
  quantization,
  runtimeResults,
  executionPlacement,
  findingsRegister,
  engineeringReport,
  metricCoverage,
  evidenceRoot,
} = {}) {
  const failures = [];
  const checks = [];
  const format = String(analysis?.format || staticAnalysis?.format || "").toLowerCase();
  const check = (id, condition, message, pointers = []) => {
    checks.push({ id, severity: "critical", status: condition ? "pass" : "fail", evidence_json_pointers: pointers });
    if (!condition) failures.push({ id, severity: "critical", message, evidence_json_pointers: pointers });
  };
  const expectedStaticSchema = format === "coreml"
    ? ANALYZER_METADATA.schemas.coreMlStaticAnalysis
    : format === "executorch"
      ? ANALYZER_METADATA.schemas.execuTorchStaticAnalysis
    : ANALYZER_METADATA.schemas.serializedContainerAnalysis;
  const reportText = String(engineeringReport || "");
  const tensors = Array.isArray(staticAnalysis?.tensors) ? staticAnalysis.tensors : [];
  const ops = Array.isArray(staticAnalysis?.ops) ? staticAnalysis.ops : [];
  const sha = String(staticAnalysis?.model_sha256 || "");
  check("CF-SERIALIZED-SCHEMA-001", ["gguf", "safetensors", "coreml", "executorch"].includes(format)
    && staticAnalysis?.schema === expectedStaticSchema,
  "Serialized-artifact static-analysis schema does not match the format registry.", ["/evidence/static_analysis/schema"]);
  check("CF-SERIALIZED-SCHEMA-002", quantization?.schema === ANALYZER_METADATA.schemas.quantizationEvidence,
    "Serialized-artifact quantization evidence schema does not match analyzer metadata.", ["/evidence/quantization/schema"]);
  registerExecutionPlacementConformance({
    check, id: "CF-SERIALIZED-PLACEMENT-001", analysis: staticAnalysis,
    executionPlacement, engineeringReport: reportText, serialized: true,
  });
  const publicRedacted = staticAnalysis?.filename === "ARTIFACT-001"
    && sha === ""
    && reportText.includes("`redacted`");
  check("CF-SERIALIZED-IDENTITY-001", (publicRedacted
    || (/^[0-9a-f]{64}$/i.test(sha) && reportText.includes(sha)))
    && Number(staticAnalysis?.file_size ?? staticAnalysis?.file_size_bytes) > 0,
  "Serialized-artifact report is not bound to a non-empty artifact byte length and SHA-256.", ["/evidence/static_analysis/model_sha256", "/engineering_report.md"]);
  check("CF-SERIALIZED-TARGET-001", !staticAnalysis?.target_profile
    && reportText.includes("artifact-only; no CPU, GPU, NPU, execution provider, or target profile selected or inferred"),
  "Serialized-artifact report inherited or implied an execution-target profile.", ["/evidence/static_analysis", "/engineering_report.md"]);
  const bundle = staticAnalysis?.artifact_bundle || null;
  if (bundle) {
    const files = Array.isArray(bundle.files) ? bundle.files : [];
    const paths = files.map((file) => file.path);
    const publicBundleRedaction = publicRedacted
      && bundle.public_redaction?.schema === "deepbom.public_artifact_bundle_redaction.v1"
      && bundle.public_redaction?.artifact_identity_removed === true
      && bundle.public_redaction?.canonical_bundle_sha256_removed === true
      && bundle.public_redaction?.member_content_sha256_removed === true
      && bundle.bundle_sha256 === ""
      && files.every((file) => file.sha256 === "");
    const privateBundleIdentity = /^[0-9a-f]{64}$/i.test(String(bundle.bundle_sha256 || ""))
      && bundle.bundle_sha256 === sha
      && files.every((file) => /^[0-9a-f]{64}$/i.test(String(file.sha256 || "")));
    check("CF-SERIALIZED-BUNDLE-001", (publicBundleRedaction || privateBundleIdentity)
      && files.length > 0
      && new Set(paths).size === paths.length
      && files.every((file) => typeof file.path === "string" && file.path.length > 0
        && Number.isSafeInteger(file.byte_length) && file.byte_length >= 0
        && file.required === true),
    "Serialized package identity, canonical member hash manifest, or explicit public-redaction projection is inconsistent.", ["/evidence/static_analysis/artifact_bundle"]);
  }
  check("CF-SERIALIZED-TENSOR-001", Number(staticAnalysis?.tensor_count) === tensors.length
    && tensors.every((tensor, index) => tensor?.index === index),
  "Serialized tensor count or deterministic index ownership is inconsistent.", ["/evidence/static_analysis/tensor_count", "/evidence/static_analysis/tensors"]);

  if (format === "gguf") {
    registerGgufSerializedConformance({ staticAnalysis, tensors, ops, reportText, check });
  } else if (format === "safetensors") {
    registerSafeTensorsSerializedConformance({ staticAnalysis, tensors, ops, reportText, check });
  } else if (format === "executorch") registerExecuTorchSerializedConformance({ staticAnalysis, tensors, ops, reportText, check });
  else registerCoreMlSerializedConformance({ staticAnalysis, ops, reportText, check });
  registerSerializedRuntimeConformance({ format, staticAnalysis, runtimeResults, reportText, check });
  if (["gguf", "safetensors"].includes(format)) {
    const llmValidation = validateOnDeviceLlmContract(staticAnalysis);
    check("CF-LLM-CONTRACT-001", llmValidation.valid && reportText.includes("## On-device LLM Evidence Contract"),
      llmValidation.errors[0] || "On-device LLM architecture, sidecar, state-scenario, or claim-boundary contract is invalid.",
      ["/evidence/static_analysis/on_device_llm", "/engineering_report.md"]);
  }

  const normalizedFindings = findingsRegister?.findings || [];
  const findingBindings = validateFindingEvidenceBindings(normalizedFindings, evidenceRoot || {});
  check("CF-SERIALIZED-FINDING-001", findingBindings.schema_error_count === 0
    && findingBindings.unresolved_pointer_count === 0,
  "Serialized-artifact finding schema or evidence-pointer binding is invalid.", ["/evidence/findings_register/findings"]);
  const metricFailures = validateMetricCoverageManifest(metricCoverage, engineeringReport);
  check("CF-SERIALIZED-COVERAGE-001", metricCoverage?.schema === ANALYZER_METADATA.schemas.metricCoverageManifest
    && metricCoverage?.coverage_status !== "fail"
    && metricCoverage?.decision_coverage?.status !== "fail"
    && metricFailures.length === 0,
  metricFailures[0] || "Serialized-artifact metric and decision coverage is incomplete or unbound.", ["/evidence/metric_coverage_manifest", "/engineering_report.md"]);
  return {
    schema: ANALYZER_METADATA.schemas.conformanceReport,
    status: failures.length ? "fail" : "pass",
    checked_invariants: checks.length,
    failed_invariants: failures.length,
    critical_failed_invariants: failures.length,
    release_export_allowed: failures.length === 0,
    development_export_allowed: true,
    development_watermark_required: failures.length > 0,
    finding_evidence_pointer_validation: findingBindings,
    checks,
    failures,
    scope: "Format-aware serialized-artifact invariants for GGUF, SafeTensors, Core ML, and ExecuTorch graph/data/numerical evidence plus the declared execution-placement evidence boundary. Runtime-trace validity remains owned by its format-specific importer; latency, un-serialized framework state, and task quality remain outside this conformance scope.",
  };
}

function registerExecuTorchSerializedConformance({ staticAnalysis, tensors, ops, reportText, check }) {
  const pte = staticAnalysis.executorch_container === "pte";
  const ptd = staticAnalysis.executorch_container === "ptd";
  const contract = pte ? staticAnalysis.executorch_program : staticAnalysis.executorch_flat_tensor;
  const source = contract?.source || {};
  const segments = Array.isArray(contract?.segments) ? contract.segments : [];
  check("CF-EXECUTORCH-SOURCE-001", (pte || ptd)
    && source.repository === "pytorch/executorch"
    && /^[0-9a-f]{40}$/i.test(String(source.commit || ""))
    && /^[0-9a-f]{64}$/i.test(String(source.scalar_type_schema_sha256 || ""))
    && (pte ? contract.identifier === "ET12" && /^[0-9a-f]{64}$/i.test(String(source.program_schema_sha256 || ""))
      : contract.identifier === "FT01" && /^[0-9a-f]{64}$/i.test(String(source.flat_tensor_schema_sha256 || ""))),
  "ExecuTorch wire interpretation is not bound to ET12/FT01 and pinned schema content digests.", ["/evidence/static_analysis/executorch_program", "/evidence/static_analysis/executorch_flat_tensor"]);
  check("CF-EXECUTORCH-SEGMENT-001", segments.every((segment, index) => {
    try {
      const offset = BigInt(segment.offset);
      const size = BigInt(segment.size);
      const absolute = BigInt(segment.absolute_offset);
      const end = BigInt(segment.absolute_end);
      const base = BigInt(contract.extended_header?.segment_base_offset || "0");
      return segment.index === index && offset >= 0n && size >= 0n && absolute === base + offset && end === absolute + size && end <= BigInt(staticAnalysis.file_size);
    } catch { return false; }
  }), "ExecuTorch segment range conservation failed.", ["/evidence/static_analysis/size_breakdown", `/evidence/static_analysis/${pte ? "executorch_program" : "executorch_flat_tensor"}/segments`]);
  if (pte) {
    const plans = Array.isArray(contract.plans) ? contract.plans : [];
    const planInstructions = plans.reduce((sum, plan) => sum + Number(plan.instruction_count || 0), 0);
    const kernelInstructions = plans.reduce((sum, plan) => sum + Number(plan.kernel_instruction_count || 0), 0);
    const delegateInstructions = plans.reduce((sum, plan) => sum + Number(plan.delegate_instruction_count || 0), 0);
    const unknownComputeInstructions = Number(staticAnalysis.mac_assessment?.unknown_compute_instruction_count || 0);
    const nonComputeInstructions = plans.reduce((sum, plan) => sum + Number(plan.non_compute_instruction_count || 0), 0);
    const planTensors = plans.reduce((sum, plan) => sum + Number(plan.tensor_count || 0), 0);
    const plannedBytes = plans.reduce((sum, plan) => sum + BigInt(plan.non_const_memory_bytes_decimal || "0"), 0n);
    check("CF-EXECUTORCH-PROGRAM-001", Number(staticAnalysis.subgraphs) === plans.length
      && Number(staticAnalysis.operator_count) === ops.length && planInstructions === ops.length
      && kernelInstructions + delegateInstructions + nonComputeInstructions === planInstructions
      && Number(staticAnalysis.tensor_count) === tensors.length && planTensors === tensors.length
      && ops.every((op, index) => op.index === index && Number.isInteger(op.plan_index) && Number.isInteger(op.instruction_index))
      && plannedBytes.toString() === String(staticAnalysis.tensor_liveness?.planned_non_const_memory_decimal || ""),
    "ExecuTorch plan, instruction, tensor, or AOT memory totals do not conserve the decoded ET12 inventory.", ["/evidence/static_analysis/executorch_program/plans", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors", "/evidence/static_analysis/tensor_liveness"]);
    const signatureSource = contract.operator_signature_registry || {};
    check("CF-EXECUTORCH-SIGNATURE-SOURCE-001", signatureSource.schema === "deepbom.executorch_operator_signature_source.v1"
      && signatureSource.portable_operator_count === 209
      && signatureSource.executorch?.commit === source.commit
      && /^[a-f0-9]{64}$/.test(String(signatureSource.executorch?.portable_functions_sha256 || ""))
      && /^[a-f0-9]{40}$/.test(String(signatureSource.pytorch?.commit || ""))
      && /^[a-f0-9]{64}$/.test(String(signatureSource.pytorch?.native_functions_sha256 || ""))
      && Number(staticAnalysis.mac_assessment?.source_bound_kernel_instruction_count || 0) <= kernelInstructions
      && delegateInstructions === Number(staticAnalysis.mac_assessment?.delegate_instruction_count || 0),
    "ExecuTorch KernelCall direction or nominal-MAC evidence is not bound to the pinned portable/PyTorch signature sources.", ["/evidence/static_analysis/executorch_program/operator_signature_registry", "/evidence/static_analysis/mac_assessment"]);
    const buildBinding = contract.selected_build_binding || {};
    let reconstructedBinding = null;
    try {
      reconstructedBinding = buildExecuTorchSelectedBuildBinding(
        contract.delegates || [],
        ops,
        buildBinding.selected_build || null,
        buildBinding.selected_build_input || null,
      );
    } catch { reconstructedBinding = null; }
    check("CF-EXECUTORCH-BUILD-BINDING-001", reconstructedBinding != null
      && canonicalJson(reconstructedBinding) === canonicalJson(buildBinding)
      && buildBinding.source_registry?.commit === source.commit
      && buildBinding.source_registry?.backend_count === 7
      && /^[a-f0-9]{64}$/.test(String(buildBinding.source_registry?.registry_sha256 || ""))
      && (!buildBinding.selected_build_input || (reportText.includes(String(buildBinding.selected_build_input.path))
        && reportText.includes(String(buildBinding.selected_build_input.file_sha256))))
      && buildBinding.status !== "CONTRADICTION_SELECTED_BUILD_CANNOT_SATISFY_SERIALIZED_PROGRAM",
    "ExecuTorch backend source registry or selected-build attestation does not reconstruct, or the selected build contradicts the serialized program.", ["/evidence/static_analysis/executorch_program/selected_build_binding"]);
    const processedPayloads = Array.isArray(contract.processed_backend_payloads) ? contract.processed_backend_payloads : [];
    const delegates = Array.isArray(contract.delegates) ? contract.delegates : [];
    check("CF-EXECUTORCH-PROCESSED-PAYLOAD-001", processedPayloads.length === delegates.length
      && processedPayloads.every((payload) => {
        const delegate = delegates.find((row) => row.plan_index === payload.plan_index && row.index === payload.delegate_index);
        return delegate && delegate.backend_id === payload.backend_id
          && delegate.processed_location === payload.processed_location && delegate.processed_index === payload.processed_index
          && Number.isSafeInteger(payload.byte_length) && payload.byte_length >= 0
          && /^[a-f0-9]{64}$/.test(String(payload.sha256 || ""))
          && !String(payload.structural_status || "").startsWith("CONTRADICTION_");
      }),
    "ExecuTorch processed backend payload identities, ranges, hashes, or source-described FlatBuffer envelopes do not conserve.", ["/evidence/static_analysis/executorch_program/processed_backend_payloads"]);
    const macContractValid = unknownComputeInstructions > 0
      ? staticAnalysis.total_macs == null && staticAnalysis.mac_assessment?.complete === false
        && Number(staticAnalysis.mac_assessment?.unknown_compute_instruction_count) === unknownComputeInstructions
      : /^(?:0|[1-9]\d*)$/.test(String(staticAnalysis.total_macs_decimal || ""))
        && staticAnalysis.mac_assessment?.complete === true
        && Number(staticAnalysis.mac_assessment?.unknown_compute_instruction_count) === 0;
    const graphBoundary = String(contract.graph_boundary || "");
    const selectedBuildBoundary = String(buildBinding.interpretation_boundary || "");
    check("CF-EXECUTORCH-BOUNDARY-001", macContractValid
      && graphBoundary.length > 0 && reportText.includes(graphBoundary)
      && selectedBuildBoundary.length > 0 && reportText.includes(selectedBuildBoundary)
      && reportText.includes("Matching portable KernelCall direction is source-bound"),
    "ExecuTorch report promoted unbound operator semantics or serialized delegate calls beyond their evidence boundary.", ["/evidence/static_analysis/mac_assessment", "/engineering_report.md"]);
    const external = contract.external_tensor_data || {};
    const bindings = Array.isArray(external.bindings) ? external.bindings : [];
    const matched = bindings.filter((row) => row.status === "matched");
    const verifiedBytes = matched.reduce((sum, row) => sum + BigInt(row.logical_bytes_decimal || "0"), 0n);
    check("CF-EXECUTORCH-EXTERNAL-001", Number(external.resolved_name_count || 0) === bindings.length
      && Number(external.verified_contract_count || 0) === matched.length
      && String(external.verified_logical_bytes_decimal || "0") === verifiedBytes.toString()
      && bindings.every((row) => row.status === "matched" ? !(row.reasons || []).length : (row.reasons || []).length > 0),
    "ExecuTorch PTD external tensor bindings do not conserve exact contract counts or logical bytes.", ["/evidence/static_analysis/executorch_program/external_tensor_data", "/engineering_report.md"]);
  } else {
    check("CF-EXECUTORCH-FLATTENSOR-001", ops.length === 0 && Number(staticAnalysis.operator_count) === 0
      && Number(staticAnalysis.tensor_count) === tensors.length
      && tensors.every((tensor, index) => tensor.index === index && Number.isInteger(tensor.segment_index)),
    "ExecuTorch FT01 data was promoted to an execution graph or its named-data inventory is inconsistent.", ["/evidence/static_analysis/executorch_flat_tensor", "/evidence/static_analysis/tensors", "/evidence/static_analysis/ops"]);
  }
}

export function buildConformanceReport({
  analysis,
  staticAnalysis,
  quantization,
  findingsRegister,
  runtimeResults,
  executionPlacement,
  securityPosture,
  mlBomDocument,
  engineeringReport,
  metricCoverage,
  evidenceRoot,
} = {}) {
  if (["gguf", "safetensors", "coreml", "executorch"].includes(String(analysis?.format || "").toLowerCase())) {
    return buildSerializedArtifactConformanceReport({
      analysis,
      staticAnalysis,
      quantization,
      runtimeResults,
      executionPlacement,
      findingsRegister,
      engineeringReport,
      metricCoverage,
      evidenceRoot,
    });
  }
  const failures = [];
  const checks = [];
  const onnx = String(analysis?.format || "tflite").toLowerCase() === "onnx";
  const quantResearchCoverage = onnx ? null : buildQuantResearchCoverage(analysis);
  const quantResearchLabById = new Map((quantResearchCoverage?.labs || []).map((row) => [row.id, row]));
  const labIdForCheck = (id) => {
    const prefix = [...QUANT_RESEARCH_CONFORMANCE_PREFIXES.keys()].find((candidate) => id.startsWith(candidate));
    return prefix ? QUANT_RESEARCH_CONFORMANCE_PREFIXES.get(prefix) : null;
  };
  const check = (id, condition, message, pointers = [], severity = "critical") => {
    const labId = labIdForCheck(id);
    const lab = labId ? quantResearchLabById.get(labId) : null;
    if (lab?.status === "not_applicable") {
      checks.push({
        id,
        severity,
        status: "pass",
        applicability: "not_applicable",
        applicability_reason_code: lab.reason_code,
        evidence_json_pointers: ["/evidence/static_analysis/quant_research_coverage"],
      });
      return;
    }
    checks.push({ id, severity, status: condition ? "pass" : "fail", evidence_json_pointers: pointers });
    if (!condition) failures.push({ id, severity, message, evidence_json_pointers: pointers });
  };
  const engineeringReportText = String(engineeringReport || "");
  const compactMlBomEvidence = compactMlBomEvidencePointerValid(mlBomDocument);

  check("CF-SCHEMA-001", staticAnalysis?.schema === ANALYZER_METADATA.schemas.staticAnalysis, "Static-analysis schema does not match analyzer metadata.", ["/evidence/static_analysis/schema"]);
  check("CF-SCHEMA-002", quantization?.schema === ANALYZER_METADATA.schemas.quantizationEvidence, "Quantization-evidence schema does not match analyzer metadata.", ["/evidence/quantization/schema"]);
  registerExecutionPlacementConformance({
    check, id: "CF-PLACEMENT-001", analysis: staticAnalysis,
    executionPlacement, engineeringReport: engineeringReportText,
  });
  if (!onnx) {
    const staticCoverage = staticAnalysis?.quant_research_coverage;
    const quantizationCoverage = quantization?.quant_research_coverage;
    const statusCount = Number(staticCoverage?.assessed_lab_count || 0)
      + Number(staticCoverage?.partial_lab_count || 0)
      + Number(staticCoverage?.not_assessed_lab_count || 0)
      + Number(staticCoverage?.not_applicable_lab_count || 0);
    check("CF-QR-COVERAGE-001", staticCoverage?.schema === QUANT_RESEARCH_COVERAGE_SCHEMA
      && Number(staticCoverage.lab_count) === 15
      && (staticCoverage.labs || []).length === 15
      && statusCount === 15
      && Number(staticCoverage.class_supported_lab_count || 0) + Number(staticCoverage.class_excluded_lab_count || 0) === 15,
    "Quant research artifact class, lab denominator, or status conservation is invalid.", ["/evidence/static_analysis/quant_research_coverage"]);
    check("CF-QR-COVERAGE-002", JSON.stringify(staticCoverage) === JSON.stringify(quantResearchCoverage)
      && JSON.stringify(quantizationCoverage) === JSON.stringify(quantResearchCoverage),
    "Quant research coverage does not reconstruct or differs across static and quantization evidence.", ["/evidence/static_analysis/quant_research_coverage", "/evidence/quantization/quant_research_coverage"]);
    check("CF-QR-COVERAGE-003", engineeringReportText.includes("## Quantization Research Coverage (DERIVED)")
      && engineeringReportText.includes(quantResearchCoverage.artifact_class_label)
      && engineeringReportText.includes(quantResearchCoverage.scan_denominator_policy),
    "Engineering Report does not preserve the artifact class and scan-denominator policy.", ["/evidence/static_analysis/quant_research_coverage", "/engineering_report.md"]);

    registerTfliteStorageConformance({ check, staticAnalysis, engineeringReportText, mlBomDocument });
  }
  check("CF-FINDING-001", findingsRegister?.authoritative_action_source === "findings", "Normalized findings must be the sole authoritative action source.", ["/evidence/findings_register/authoritative_action_source"]);
  const rawSignals = findingsRegister?.raw_analyzer_signals || [];
  check("CF-FINDING-002", rawSignals.every((item) => item.authoritative === false && item.classification === "raw_analyzer_signal" && !("severity" in item) && !("impact" in item) && !("actions" in item)), "Raw analyzer signals contain authoritative severity, impact, or action semantics.", ["/evidence/findings_register/raw_analyzer_signals"]);
  const normalizedFindings = findingsRegister?.findings || [];
  const pointerRoot = evidenceRoot || {
    analyzer_metadata: ANALYZER_METADATA,
    evidence: {
      static_analysis: staticAnalysis,
      quantization,
      runtime_results: runtimeResults,
      execution_placement: executionPlacement,
      security_posture: securityPosture,
      findings_register: findingsRegister,
      mlbom_cyclonedx: mlBomDocument,
    },
  };
  const findingBindings = validateFindingEvidenceBindings(normalizedFindings, pointerRoot);
  check("CF-FINDING-004", findingBindings.schema_error_count === 0,
    findingBindings.schema_errors[0]?.message || "Every authoritative finding must carry a unique ID, enumerated priority, declared evidence confidence, source rule, method, and non-empty unique JSON Pointer list.",
    ["/evidence/findings_register/findings"]);
  check("CF-FINDING-005", findingBindings.unresolved_pointer_count === 0,
    findingBindings.unresolved_pointers[0]?.message || "Every authoritative finding JSON Pointer resolves against the exact exported evidence document.",
    ["/evidence/findings_register/findings", "/evidence"]);
  check("CF-FINDING-006", !evidenceRoot || normalizedFindings.every((finding) => String(engineeringReport || "").includes(finding.finding_id)
    && String(engineeringReport || "").includes(finding.source_rule_id)
    && (finding.evidence_json_pointers || []).every((pointer) => String(engineeringReport || "").includes(pointer))),
  "Engineering Report must expose every authoritative finding without truncation, including source rule and evidence pointer bindings.", ["/evidence/findings_register/findings", "/engineering_report.md"]);

  const metricFailures = collectAssessmentMetricFailures(staticAnalysis);
  check("CF-STATUS-001", metricFailures.length === 0, metricFailures[0] || "Assessment metric status/value contract failed.", ["/evidence/static_analysis"]);
  const metricCoverageFailures = validateMetricCoverageManifest(metricCoverage, engineeringReport);
  check("CF-COVERAGE-001", metricCoverage?.schema === ANALYZER_METADATA.schemas.metricCoverageManifest
    && metricCoverageFailures.length === 0, metricCoverageFailures[0] || "Metric coverage manifest is valid and fully bound.", ["/evidence/metric_coverage_manifest", "/engineering_report.md"]);
  const dynamicShapeVerification = verifyDynamicShapeCostEvidence({
    analysis: staticAnalysis,
    engineeringReport,
    mlBomDocument,
    findingsRegister,
  });
  check("CF-DYNAMIC-001", dynamicShapeVerification.contract_present
    && dynamicShapeVerification.static_non_applicability_valid,
  "Dynamic-shape cost evidence is missing, has the wrong format/schema/status, or misclassifies a static model.", ["/evidence/static_analysis/dynamic_shape_cost_contract"]);
  check("CF-DYNAMIC-002", dynamicShapeVerification.symbols_valid
    && dynamicShapeVerification.tensor_formulas_valid,
  "Dynamic dimension symbols or tensor payload formulas differ from independent tensor-shape reconstruction.", ["/evidence/static_analysis/dynamic_shape_cost_contract", "/evidence/static_analysis/tensors"]);
  check("CF-DYNAMIC-003", dynamicShapeVerification.op_formulas_valid
    && dynamicShapeVerification.total_macs_valid,
  "Dynamic compute-op or total-MAC polynomials differ from independent operator-shape reconstruction.", ["/evidence/static_analysis/dynamic_shape_cost_contract", "/evidence/static_analysis/ops"]);
  check("CF-DYNAMIC-004", dynamicShapeVerification.liveness_valid,
    "Symbolic live-set candidates or peak dominance differ from independent graph-lifetime reconstruction.", ["/evidence/static_analysis/dynamic_shape_cost_contract/liveness", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"]);
  check("CF-DYNAMIC-005", dynamicShapeVerification.report_valid,
    "Engineering Report does not preserve the dynamic-shape schema, status, formulas, blockers, and peak-selection boundary.", ["/evidence/static_analysis/dynamic_shape_cost_contract", "/engineering_report.md"]);
  check("CF-DYNAMIC-006", dynamicShapeVerification.mlbom_valid
    && dynamicShapeVerification.finding_valid,
  "ML-BOM or authoritative finding does not preserve the exact dynamic-shape cost contract.", ["/evidence/static_analysis/dynamic_shape_cost_contract", "/evidence/mlbom_cyclonedx", "/evidence/findings_register/findings"]);
  const comparableMetricEntries = (entries = []) => entries.map(({ pointer_binding_status: _pointerBindingStatus, ...entry }) => entry);
  const expectedMetricEntries = buildMetricCoverageEntries(analysis, {
    findings: normalizedFindings,
    runtimeResults,
  });
  check("CF-COVERAGE-002", JSON.stringify(comparableMetricEntries(metricCoverage?.entries)) === JSON.stringify(comparableMetricEntries(expectedMetricEntries)),
    "Metric-family applicability, status, evidence class, method, or route does not reconstruct from the analysis and runtime evidence.", ["/evidence/metric_coverage_manifest/entries", "/evidence/static_analysis", "/evidence/runtime_results"]);

  const staticInputs = staticAnalysis?.inputs || [];
  const staticOutputs = staticAnalysis?.outputs || [];
  const inputTensorIndices = staticAnalysis?.input_tensor_indices || [];
  const outputTensorIndices = staticAnalysis?.output_tensor_indices || [];
  const inputContracts = staticAnalysis?.input_contracts || [];
  const reconstructedInputContracts = staticInputs.map((tensor) => independentlyReconstructInputContract(tensor, staticAnalysis?.ops || [], onnx));
  check("CF-IO-001", inputTensorIndices.length === staticInputs.length
    && outputTensorIndices.length === staticOutputs.length
    && staticInputs.every((tensor, index) => tensor.index === inputTensorIndices[index])
    && staticOutputs.every((tensor, index) => tensor.index === outputTensorIndices[index]),
  "Graph input/output tensor indices must conserve the ordered artifact ABI inventory.", ["/evidence/static_analysis/inputs", "/evidence/static_analysis/outputs", "/evidence/static_analysis/input_tensor_indices", "/evidence/static_analysis/output_tensor_indices"]);
  check("CF-IO-002", inputContracts.length === reconstructedInputContracts.length
    && inputContracts.every((contract, index) => inputContractMatchesReconstruction(contract, reconstructedInputContracts[index])),
  "Input layout, channel axis, scalar quantized range, or preprocessing boundary does not independently reconstruct from the tensor ABI and direct consumer semantics.", ["/evidence/static_analysis/input_contracts", "/evidence/static_analysis/inputs", "/evidence/static_analysis/ops"]);
  check("CF-IO-003", inputContracts.every((contract) => engineeringReportText.includes(contract.schema)
    && engineeringReportText.includes(contract.layout_status)
    && engineeringReportText.includes(contract.tensor_numerical_contract_status)
    && engineeringReportText.includes(contract.source_data_to_tensor_preprocessing_status)
    && engineeringReportText.includes(contract.range_note)
    && engineeringReportText.includes(contract.layout_reason)
    && (contract.risks || []).every((risk) => engineeringReportText.includes(risk))),
  "Engineering Report must preserve every structured input tensor layout, range, preprocessing, and risk contract.", ["/evidence/static_analysis/input_contracts", "/engineering_report.md"]);
  const commonMlBomProperties = mlBomDocument?.properties || [];
  const commonMlBomValue = (name) => commonMlBomProperties.find((item) => item.name === name)?.value;
  check("CF-IO-004", compactMlBomEvidence || commonMlBomValue("deepbom:model:inputTensorContractEvidence") === JSON.stringify(mlBomInputContractRows(inputContracts))
    && commonMlBomValue("deepbom:model:inputTensorContractCount") === String(inputContracts.length)
    && commonMlBomValue("deepbom:model:inputLayoutDerivedCount") === String(inputContracts.filter((contract) => contract.layout_evidence_class === "DERIVED").length)
    && commonMlBomValue("deepbom:model:inputLayoutUnassessedCount") === String(inputContracts.filter((contract) => contract.layout_evidence_class === "NOT_ASSESSABLE").length),
  "ML-BOM must preserve the complete structured input contract and its derived/unassessed layout counts.", ["/evidence/static_analysis/input_contracts", "/evidence/mlbom_cyclonedx/properties"]);

  const benchmarks = runtimeResults?.benchmark_results || [];
  const successfulBenchmarks = benchmarks.filter((row) => row?.ok === true);
  check("CF-RUNTIME-BENCH-001", successfulBenchmarks.every((row) => String(row.backend || "").trim()
    && Number.isInteger(row.warmup) && row.warmup >= 0
    && Number.isInteger(row.runs) && row.runs > 0
    && validIsoTimestamp(row.generated_at)
    && row.timing_method === BROWSER_BENCHMARK_TIMING_METHOD
    && row.phase_counts?.cold_first_runs === 1
    && row.phase_counts?.warmup_runs === row.warmup
    && row.phase_counts?.measured_runs === row.runs
    && Number.isFinite(row.first_run_ms) && row.first_run_ms >= 0
    && Number.isFinite(row.compile_ms) && row.compile_ms >= 0
    && validBenchmarkStatistics(row)
    && /^[0-9a-f]{64}$/.test(row.output_digest || "")
    && Array.isArray(row.input_contracts) && row.input_contracts.length > 0
    && row.input_basis === [...new Set(row.input_contracts.map((input) => input.basis))].join(" / ")
    && row.input_contracts.every(validBenchmarkInputContract)
    && Number.isSafeInteger(row.output_count) && row.output_count > 0
    && Array.isArray(row.output_contracts) && row.output_contracts.length === row.output_count
    && row.output_contracts.every(validBenchmarkOutputContract)
    && (!onnx || validOnnxBenchmarkExternalDataBinding(row.external_data_runtime_binding, staticAnalysis))), "Successful browser benchmarks must bind true cold-first phases, exact run counts, complete executed input/output contracts, and the exact ONNX external-data file set when applicable.", ["/evidence/runtime_results/benchmark_results"]);
  check("CF-RUNTIME-BENCH-002", !successfulBenchmarks.length || (engineeringReportText.includes(BROWSER_BENCHMARK_TIMING_METHOD)
    && successfulBenchmarks.every((row) => row.input_contracts.every((input) => engineeringReportText.includes(benchmarkInputContractText(input)))
      && row.output_contracts.every((output) => engineeringReportText.includes(benchmarkOutputContractText(output)))
      && engineeringReportText.includes(row.statistics_method)
      && engineeringReportText.includes(row.noise_method)
      && (!onnx || engineeringReportText.includes(benchmarkExternalDataBindingText(row.external_data_runtime_binding)))
      && engineeringReportText.includes(benchmarkNoiseSummaryText(row))
      && engineeringReportText.includes(benchmarkSampleSeriesText(row)))), "Engineering report must preserve timing, exact samples, statistics method, and every input/output contract for successful runs.", ["/evidence/runtime_results/benchmark_results", "/engineering_report.md"]);

  if (onnx) {
    registerOnnxConformance({ staticAnalysis, findingsRegister, mlBomDocument, engineeringReport, engineeringReportText, compactMlBomEvidence, check });
  } else {
    const metadata = staticAnalysis?.metadata_presence || {};
    const inputProcessUnits = metadata.input_process_units || [];
    const outputAssociatedFiles = metadata.output_associated_files || [];
    const packedAssociatedFiles = metadata.packed_associated_files || [];
    const packedByName = new Map(packedAssociatedFiles.map((file) => [file.name, file]));
    const recognizedProcessUnits = inputProcessUnits.filter((unit) => unit.status === "assessed").length;
    const invalidProcessUnits = inputProcessUnits.filter((unit) => unit.status === "invalid_options").length;
    const unsupportedProcessUnits = inputProcessUnits.filter((unit) => unit.status === "unsupported_options_type").length;
    const outputLabelFiles = outputAssociatedFiles.filter((file) => [2, 3].includes(Number(file.file_type_code))).length;
    const payloadVerifiedFiles = packedAssociatedFiles.filter((file) => file.payload_status === "verified").length;
    const payloadUnsupportedFiles = packedAssociatedFiles.filter((file) => ["unsupported_compression_method", "encrypted_not_supported", "decoded_size_limit_exceeded"].includes(file.payload_status)).length;
    const payloadInvalidFiles = packedAssociatedFiles.length - payloadVerifiedFiles - payloadUnsupportedFiles;
    const payloadRowsValid = packedAssociatedFiles.every((file) => file.payload_status !== "verified" || (
      file.decoded_bytes === file.uncompressed_bytes
      && file.crc32_verified === true
      && /^[a-f0-9]{64}$/.test(file.payload_sha256 || "")
      && [0, 8].includes(Number(file.compression_method))
    ));
    const outputBindingValid = (file) => {
      const packed = packedByName.get(file.name);
      return file.packed_status !== "verified_payload" || Boolean(packed
        && packed.payload_status === "verified"
        && file.payload_status === "verified"
        && file.payload_sha256 === packed.payload_sha256
        && file.payload_bytes === packed.decoded_bytes
        && file.crc32_verified === true);
    };
    const axisCardinalityStatus = (file) => {
      if (Number(file.file_type_code) !== 2 || file.packed_status !== "verified_payload"
        || file.text_encoding_status !== "valid_utf8" || Number(file.label_entry_count || 0) === 0) return null;
      const shape = file.output_shape || [];
      const labelCount = Number(file.label_entry_count);
      const dynamic = shape.some((dimension) => Number(dimension) < 0);
      const axes = shape.map((dimension, axis) => ({ dimension: Number(dimension), axis }))
        .filter(({ dimension, axis }) => !(axis === 0 && shape.length > 1 && dimension === 1)
          && dimension > 0 && dimension === labelCount)
        .map(({ axis }) => axis);
      const status = axes.length === 1 && !dynamic ? "verified_unique_axis_match"
        : axes.length === 1 ? "unresolved_dynamic_shape_with_known_axis_match"
          : axes.length === 0 && dynamic ? "unresolved_dynamic_shape_no_known_axis_match"
            : axes.length === 0 ? "mismatch_no_output_axis"
              : "ambiguous_multiple_output_axes";
      return { status, axes };
    };
    const labelRowsValid = outputAssociatedFiles.every((file) => {
      const derived = axisCardinalityStatus(file);
      return !derived || (file.cardinality_status === derived.status
        && JSON.stringify(file.matching_output_axes || []) === JSON.stringify(derived.axes));
    });
    const semanticLabelVerified = (file) => [2, 3].includes(Number(file.file_type_code))
      && file.packed_status === "verified_payload"
      && file.text_encoding_status === "valid_utf8"
      && Number(file.label_entry_count || 0) > 0
      && (Number(file.file_type_code) === 3 || file.cardinality_status === "verified_unique_axis_match");
    const verifiedOutputAssociatedFiles = outputAssociatedFiles.filter((file) => file.packed_status === "verified_payload" && outputBindingValid(file)).length;
    const missingOutputAssociatedFiles = outputAssociatedFiles.filter((file) => file.packed_status === "missing_from_archive").length;
    const verifiedOutputLabelFiles = outputAssociatedFiles.filter(semanticLabelVerified).length;
    const missingOutputLabelFiles = outputAssociatedFiles.filter((file) => [2, 3].includes(Number(file.file_type_code)) && file.packed_status === "missing_from_archive").length;
    const invalidOutputLabelFiles = outputLabelFiles - verifiedOutputLabelFiles - missingOutputLabelFiles;
    const verifiedOutput0LabelFiles = outputAssociatedFiles.filter((file) => Number(file.output_ordinal) === 0 && semanticLabelVerified(file)).length;
    const axisLabelFiles = outputAssociatedFiles.filter((file) => Number(file.file_type_code) === 2);
    const labelCardinalityMatches = axisLabelFiles.filter((file) => file.cardinality_status === "verified_unique_axis_match").length;
    const labelCardinalityMismatches = axisLabelFiles.filter((file) => file.cardinality_status === "mismatch_no_output_axis").length;
    const labelCardinalityAmbiguous = axisLabelFiles.filter((file) => file.cardinality_status === "ambiguous_multiple_output_axes").length;
    const labelCardinalityUnresolved = axisLabelFiles.length - labelCardinalityMatches - labelCardinalityMismatches - labelCardinalityAmbiguous;
    const expectedPreprocessingStatus = !metadata.has_model_metadata
      ? "absent_no_model_metadata"
      : metadata.status !== "parsed" ? "not_assessed_metadata_parse_failure"
        : inputProcessUnits.length === 0 ? "absent_no_explicit_input_process_units"
          : recognizedProcessUnits === inputProcessUnits.length && invalidProcessUnits === 0 && unsupportedProcessUnits === 0
            ? "assessed_explicit_input_process_units" : "partial_invalid_or_unsupported_input_process_units";
    check("CF-METADATA-001", metadata.schema === ANALYZER_METADATA.schemas.artifactMetadata
      && metadata.format === "tflite"
      && Number(metadata.input_process_unit_count || 0) === inputProcessUnits.length
      && Number(metadata.recognized_input_process_unit_count || 0) === recognizedProcessUnits
      && Number(metadata.invalid_input_process_unit_count || 0) === invalidProcessUnits
      && Number(metadata.unrecognized_input_process_unit_count || 0) === unsupportedProcessUnits
      && Number(metadata.output_associated_file_count || 0) === outputAssociatedFiles.length
      && Number(metadata.output_label_file_count || 0) === outputLabelFiles
      && Number(metadata.packed_associated_file_count || 0) === packedAssociatedFiles.length
      && payloadRowsValid
      && outputAssociatedFiles.every(outputBindingValid)
      && labelRowsValid
      && Number(metadata.payload_verified_file_count || 0) === payloadVerifiedFiles
      && Number(metadata.payload_invalid_file_count || 0) === payloadInvalidFiles
      && Number(metadata.payload_unsupported_file_count || 0) === payloadUnsupportedFiles
      && Number(metadata.verified_output_associated_file_count || 0) === verifiedOutputAssociatedFiles
      && Number(metadata.missing_output_associated_file_count || 0) === missingOutputAssociatedFiles
      && Number(metadata.verified_output_label_file_count || 0) === verifiedOutputLabelFiles
      && Number(metadata.missing_output_label_file_count || 0) === missingOutputLabelFiles
      && Number(metadata.invalid_output_label_file_count || 0) === invalidOutputLabelFiles
      && Number(metadata.verified_output0_label_file_count || 0) === verifiedOutput0LabelFiles
      && Number(metadata.label_cardinality_match_count || 0) === labelCardinalityMatches
      && Number(metadata.label_cardinality_mismatch_count || 0) === labelCardinalityMismatches
      && Number(metadata.label_cardinality_ambiguous_count || 0) === labelCardinalityAmbiguous
      && Number(metadata.label_cardinality_unresolved_count || 0) === labelCardinalityUnresolved
      && metadata.preprocessing_contract_status === expectedPreprocessingStatus
      && metadata.documented_preprocessing === (expectedPreprocessingStatus === "assessed_explicit_input_process_units")
      && metadata.output_semantics_documented === (metadata.status === "parsed" && verifiedOutput0LabelFiles > 0),
    "TFLite metadata ledger must derive preprocessing and output semantics from parsed M001 process-unit and associated-file records, not generic metadata presence.", ["/evidence/static_analysis/metadata_presence"]);
    check("CF-METADATA-002", engineeringReportText.includes(metadata.schema)
      && engineeringReportText.includes(metadata.preprocessing_contract_status)
      && engineeringReportText.includes(metadata.associated_file_archive_status)
      && engineeringReportText.includes(`${Number(metadata.payload_verified_file_count || 0).toLocaleString("en-US")} verified`)
      && (!metadata.metadata_schema_identifier || engineeringReportText.includes(metadata.metadata_schema_identifier))
      && inputProcessUnits.slice(0, 24).every((unit) => engineeringReportText.includes(unit.options_type))
      && outputAssociatedFiles.slice(0, 24).every((file) => !file.payload_sha256 || engineeringReportText.includes(file.payload_sha256)),
    "Engineering report must preserve the TFLite metadata schema, process-unit status, and parsed contract rows.", ["/evidence/static_analysis/metadata_presence", "/engineering_report.md"]);
    const frontier = staticAnalysis?.deployment_frontier || null;
    const frontierReport = String(engineeringReport || "");
    check("CF-FRONTIER-001", frontier
      ? frontier.schema === ANALYZER_METADATA.schemas.deploymentFrontier
        && frontier.artifact_sha256 === staticAnalysis.model_sha256
        && frontier.target_count >= 2
        && frontier.op_count === (staticAnalysis.ops || []).length
        && frontier.robust_coverage?.minimum_union_coverage + 1e-12 >= frontier.robust_coverage?.threshold
        && frontier.target_divergence?.pairs?.every((pair) => pair.drivers?.length === frontier.op_count
          && Math.abs(Number(pair.attribution_sum || 0) - Number(pair.normalized_jensen_shannon_divergence || 0)) <= 1e-10)
        && frontierReport.includes("## Deployment Frontier (DERIVED/ESTIMATED)")
        && frontierReport.includes("### Maximum-Pair Divergence Attribution (DERIVED)")
      : frontierReport.includes("## Deployment Frontier (NOT_ASSESSED)"), "TFLite deployment frontier must be fully bound when emitted, or explicitly reported as NOT_ASSESSED.", ["/evidence/static_analysis/deployment_frontier", "/engineering_report.md"]);
    const deploymentDelta = staticAnalysis?.deployment_delta || null;
    check("CF-DELTA-001", deploymentDelta
      ? deploymentDelta.schema === ANALYZER_METADATA.schemas.deploymentDelta
        && deploymentDelta.candidate?.sha256 === staticAnalysis.model_sha256
        && deploymentDelta.target_count >= 2
        && deploymentDelta.target_deltas?.length === deploymentDelta.target_count
        && deploymentDelta.alignment_rows?.filter((row) => row.baseline_op_index != null).length === deploymentDelta.baseline?.operator_count
        && deploymentDelta.alignment_rows?.filter((row) => row.candidate_op_index != null).length === deploymentDelta.candidate?.operator_count
        && deploymentDelta.target_deltas.every((target) => Math.abs(Number(target.driver_delta_sum_us) - Number(target.signed_delta_us)) <= Math.max(1e-9, Math.abs(Number(target.signed_delta_us)) * 1e-10))
        && frontierReport.includes("## Deployment Delta (DERIVED/ESTIMATED)")
      : frontierReport.includes("## Deployment Delta (NOT_ASSESSED)"), "TFLite deployment delta must conserve every bound target ledger and appear in the engineering report, or be explicitly reported as NOT_ASSESSED.", ["/evidence/static_analysis/deployment_delta", "/engineering_report.md"]);
    check("CF-QUANT-001", quantization?.quantization_contract_checks?.schema === ANALYZER_METADATA.schemas.quantizationContractChecks, "TFLite quantization contract checks are missing from structured evidence.", ["/evidence/quantization/quantization_contract_checks"]);
    check("CF-QUANT-002", String(engineeringReport || "").includes(ANALYZER_METADATA.schemas.quantizationContractChecks), "Engineering report does not bind its quantization table to the structured contract-check schema.", ["/engineering_report.md"]);
    const contracts = quantization?.quantization_contract_checks || {};
    check("CF-QUANT-003", contracts.status === contracts.contract_integrity_status
      && ["pass", "fail", "not_applicable"].includes(contracts.contract_integrity_status)
      && ["clear", "review", "not_applicable"].includes(contracts.quantization_design_review_status), "TFLite contract integrity and quantization design review statuses must be distinct and schema-valid.", ["/evidence/quantization/quantization_contract_checks"]);
    const accumulatorAtlas = staticAnalysis?.accumulator_atlas || null;
    const accumulatorReport = String(engineeringReport || "");
    check("CF-ACCUMULATOR-001", Boolean(accumulatorAtlas)
      && accumulatorAtlas.schema === ANALYZER_METADATA.schemas.accumulatorAtlas
      && quantization?.accumulator_headroom_atlas?.schema === ANALYZER_METADATA.schemas.accumulatorAtlas
      && JSON.stringify(quantization.accumulator_headroom_atlas) === JSON.stringify(accumulatorAtlas), "Accumulator-atlas evidence is missing, schema-invalid, or inconsistent across static and quantization evidence.", ["/evidence/static_analysis/accumulator_atlas", "/evidence/quantization/accumulator_headroom_atlas"]);
    const assessedAccumulatorRows = (accumulatorAtlas?.ops || []).filter((row) => row.assessment_status === "assessed");
    check("CF-ACCUMULATOR-002", Boolean(accumulatorAtlas)
      && accumulatorAtlas.assessed_channel_count === assessedAccumulatorRows.reduce((total, row) => total + Number(row.assessed_channel_count || 0), 0)
      && accumulatorAtlas.stored_bias_channel_count === assessedAccumulatorRows.reduce((total, row) => total + Number(row.stored_bias_channel_count || 0), 0)
      && accumulatorAtlas.int32_overflow_channel_count === assessedAccumulatorRows.reduce((total, row) => total + Number(row.int32_overflow_channel_count || 0), 0)
      && accumulatorAtlas.bias_half_range_exceedance_channel_count === assessedAccumulatorRows.reduce((total, row) => total + Number(row.bias_half_range_exceedance_channel_count || 0), 0)
      && accumulatorAtlas.bias_half_range_guard_adjacent_channel_count === assessedAccumulatorRows.reduce((total, row) => total + Number(row.bias_half_range_guard_adjacent_channel_count || 0), 0)
      && accumulatorAtlas.bias_half_range_material_exceedance_channel_count === assessedAccumulatorRows.reduce((total, row) => total + Number(row.bias_half_range_material_exceedance_channel_count || 0), 0)
      && (accumulatorAtlas.required_signed_bits_histogram || []).reduce((total, count) => total + Number(count || 0), 0) === accumulatorAtlas.assessed_channel_count
      && assessedAccumulatorRows.every((row) => row.channel_accumulator_envelope_min_decimals?.length === row.assessed_channel_count
        && row.channel_accumulator_envelope_max_decimals?.length === row.assessed_channel_count
        && row.channel_post_bias_min_decimals?.length === row.assessed_channel_count
        && row.channel_post_bias_max_decimals?.length === row.assessed_channel_count
        && row.channel_required_signed_bits?.length === row.assessed_channel_count
        && row.bias_half_range_exceedance_channel_indices?.length === row.bias_half_range_exceedance_channel_count
        && row.bias_half_range_material_exceedance_channel_indices?.length === row.bias_half_range_material_exceedance_channel_count
        && row.bias_half_range_guard_adjacent_channel_count + row.bias_half_range_material_exceedance_channel_count === row.bias_half_range_exceedance_channel_count
        && row.required_signed_bits_histogram?.reduce((total, count) => total + Number(count || 0), 0) === row.assessed_channel_count
        && /^[a-f0-9]{64}$/.test(row.channel_ledger_sha256 || "")), "Accumulator-atlas channel ledgers, bit histograms, or digest bindings are internally inconsistent.", ["/evidence/static_analysis/accumulator_atlas"]);
    check("CF-ACCUMULATOR-003", accumulatorReport.includes("## Accumulator Headroom Lab (DERIVED EXACT INTEGER DOMAIN)")
      && accumulatorReport.includes(ANALYZER_METADATA.schemas.accumulatorAtlas)
      && (accumulatorAtlas?.headroom_ranking_op_indices || []).slice(0, 32).every((index) => accumulatorReport.includes(`#${String(index).padStart(3, "0")}`))
      && accumulatorReport.includes("not an observed activation distribution")
      && accumulatorReport.includes(accumulatorAtlas?.source_commit || "missing-source-commit"), "Engineering report does not preserve the accumulator schema, ranked rows, source commit, or interpretation boundary.", ["/evidence/static_analysis/accumulator_atlas", "/engineering_report.md"]);
    const requantizationFidelity = staticAnalysis?.requantization_fidelity || null;
    check("CF-REQUANT-001", Boolean(requantizationFidelity)
      && requantizationFidelity.schema === ANALYZER_METADATA.schemas.requantizationFidelity
      && quantization?.requantization_fidelity?.schema === ANALYZER_METADATA.schemas.requantizationFidelity
      && JSON.stringify(quantization.requantization_fidelity) === JSON.stringify(requantizationFidelity), "Requantization-fidelity evidence is missing, schema-invalid, or inconsistent across static and quantization evidence.", ["/evidence/static_analysis/requantization_fidelity", "/evidence/quantization/requantization_fidelity"]);
    const assessedRequantizationRows = (requantizationFidelity?.ops || []).filter((row) => row.assessment_status === "assessed");
    check("CF-REQUANT-002", Boolean(requantizationFidelity)
      && requantizationFidelity.assessed_channel_count === assessedRequantizationRows.reduce((total, row) => total + Number(row.assessed_channel_count || 0), 0)
      && requantizationFidelity.fixed_point_bound_channel_count === assessedRequantizationRows.reduce((total, row) => total + Number(row.fixed_point_bound_channel_count || 0), 0)
      && (requantizationFidelity.shift_histogram || []).reduce((total, bin) => total + Number(bin.channel_count || 0), 0) === requantizationFidelity.assessed_channel_count
      && assessedRequantizationRows.every((row) => row.channel_real_multipliers?.length === row.assessed_channel_count
        && row.channel_quantized_multipliers?.length === row.assessed_channel_count
        && row.channel_shifts?.length === row.assessed_channel_count
        && row.channel_encoding_drift_bound_codes?.length === row.assessed_channel_count
        && row.channel_default_double_rounding_bound_codes?.length === row.assessed_channel_count
        && row.channel_single_rounding_bound_codes?.length === row.assessed_channel_count
        && /^[a-f0-9]{64}$/.test(row.channel_ledger_sha256 || "")), "Requantization channel arrays, shift histogram, or digest bindings are internally inconsistent.", ["/evidence/static_analysis/requantization_fidelity"]);
    check("CF-REQUANT-003", accumulatorReport.includes("## Requantization Fidelity Lab (DERIVED PINNED Q0.31 DOMAIN)")
      && accumulatorReport.includes(ANALYZER_METADATA.schemas.requantizationFidelity)
      && (requantizationFidelity?.fidelity_ranking_op_indices || []).slice(0, 32).every((index) => accumulatorReport.includes(`#${String(index).padStart(3, "0")}`))
      && accumulatorReport.includes("does not identify the runtime's TFLITE_SINGLE_ROUNDING compile flag")
      && accumulatorReport.includes(requantizationFidelity?.source_commit || "missing-source-commit"), "Engineering report does not preserve the requantization schema, ranked rows, source commit, or build/runtime boundary.", ["/evidence/static_analysis/requantization_fidelity", "/engineering_report.md"]);
    const kernelWitness = staticAnalysis?.kernel_extremum_witness || null;
    const quantizationKernelWitness = quantization?.kernel_extremum_witness || null;
    check("CF-WITNESS-001", Boolean(kernelWitness)
      && kernelWitness.schema === ANALYZER_METADATA.schemas.kernelExtremumWitness
      && quantizationKernelWitness?.schema === ANALYZER_METADATA.schemas.kernelExtremumWitness
      && JSON.stringify(quantizationKernelWitness) === JSON.stringify(kernelWitness), "Kernel-witness evidence is missing, schema-invalid, or inconsistent across static and quantization evidence.", ["/evidence/static_analysis/kernel_extremum_witness", "/evidence/quantization/kernel_extremum_witness"]);
    const assessedWitnessRows = (kernelWitness?.ops || []).filter((row) => row.assessment_status === "assessed");
    const witnessChannels = assessedWitnessRows.flatMap((row) => [...(row.top_channels || []), ...(row.worst_channel ? [row.worst_channel] : [])]);
    check("CF-WITNESS-002", Boolean(kernelWitness)
      && Number(kernelWitness.assessed_op_count || 0) === assessedWitnessRows.length
      && Number(kernelWitness.assessed_channel_count || 0) === assessedWitnessRows.reduce((total, row) => total + Number(row.assessed_channel_count || 0), 0)
      && Number(kernelWitness.fixed_point_assessed_channel_count || 0) === assessedWitnessRows.reduce((total, row) => total + Number(row.fixed_point_assessed_channel_count || 0), 0)
      && Number(kernelWitness.witness_assignment_count || 0) === assessedWitnessRows.reduce((total, row) => total + Number(row.witness_assignment_count || 0), 0)
      && Number(kernelWitness.fixed_point_endpoint_evaluation_count || 0) === assessedWitnessRows.reduce((total, row) => total + Number(row.fixed_point_endpoint_evaluation_count || 0), 0)
      && Number(kernelWitness.build_mode_divergent_endpoint_count || 0) === assessedWitnessRows.reduce((total, row) => total + Number(row.build_mode_divergent_endpoint_count || 0), 0)
      && assessedWitnessRows.every((row) => Number(row.witness_assignment_count || 0) === Number(row.assessed_channel_count || 0) * Number(row.accumulation_terms_per_channel || 0) * 2
        && Number(row.fixed_point_endpoint_evaluation_count || 0) === Number(row.fixed_point_assessed_channel_count || 0) * 4
        && /^[a-f0-9]{64}$/.test(row.witness_ledger_sha256 || "")
        && (row.top_channels || []).every((channel) => Number(channel.term_count || 0) === Number(row.accumulation_terms_per_channel || 0)))
      && witnessChannels.every((channel) => /^[a-f0-9]{64}$/.test(channel.witness_pattern_sha256 || "")
        && Number(channel.positive_centered_weight_count || 0) + Number(channel.negative_centered_weight_count || 0) + Number(channel.zero_centered_weight_count || 0) === Number(channel.term_count || 0)
        && [channel.minimum, channel.maximum].every((endpoint) => (endpoint?.witness_code_histogram || []).reduce((total, item) => total + Number(item.count || 0), 0) === Number(channel.term_count || 0)))
      && (kernelWitness.source_references || []).length >= 6
      && kernelWitness.source_references.every((source) => /^[a-f0-9]{64}$/.test(source.sha256 || "") && String(source.url || "").includes(kernelWitness.source_commit)), "Kernel-witness assignment, endpoint-execution, sign, histogram, source, or digest conservation is internally inconsistent.", ["/evidence/static_analysis/kernel_extremum_witness"]);
    check("CF-WITNESS-003", accumulatorReport.includes("## Quantized Kernel Witness Lab (DERIVED EXACT SYNTHETIC RECEPTIVE-FIELD DOMAIN)")
      && accumulatorReport.includes(ANALYZER_METADATA.schemas.kernelExtremumWitness)
      && accumulatorReport.includes(formatIntegerForConformance(kernelWitness?.witness_assignment_count))
      && accumulatorReport.includes(formatIntegerForConformance(kernelWitness?.fixed_point_endpoint_evaluation_count))
      && accumulatorReport.includes("not a full-model input")
      && accumulatorReport.includes(kernelWitness?.source_commit || "missing-source-commit"), "Engineering report does not preserve the kernel-witness schema, exact counts, source commit, or synthetic-domain boundary.", ["/evidence/static_analysis/kernel_extremum_witness", "/engineering_report.md"]);
    const witnessMlBomProperties = [...(mlBomDocument?.metadata?.component?.properties || []), ...(mlBomDocument?.properties || [])];
    const witnessMlBomValue = (name) => witnessMlBomProperties.find((item) => item.name === name)?.value;
    const witnessFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-QNT-0111");
    const witnessHasBuildDelta = Number(kernelWitness?.build_mode_divergent_endpoint_count || 0) > 0;
    const witnessMlBomValid = compactMlBomEvidence || (witnessMlBomValue("deepbom:model:kernelExtremumWitnessSchema") === String(kernelWitness?.schema)
      && witnessMlBomValue("deepbom:model:kernelWitnessAssignments") === String(kernelWitness?.witness_assignment_count)
      && witnessMlBomValue("deepbom:model:kernelWitnessEndpointExecutions") === String(kernelWitness?.fixed_point_endpoint_evaluation_count)
      && witnessMlBomValue("deepbom:model:kernelWitnessBuildModeDifferences") === String(kernelWitness?.build_mode_divergent_endpoint_count)
      && witnessMlBomValue("deepbom:model:kernelWitnessSourceCommit") === String(kernelWitness?.source_commit));
    check("CF-WITNESS-004", witnessMlBomValid && (witnessHasBuildDelta
        ? witnessFinding?.evidence_class === "DERIVED"
          && witnessFinding?.technical_priority === "Informational"
          && (witnessFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/kernel_extremum_witness")
        : !witnessFinding), "ML-BOM or conditional authoritative finding does not preserve kernel-witness evidence.", ["/evidence/static_analysis/kernel_extremum_witness", "/evidence/mlbom_cyclonedx", "/evidence/findings_register/findings"]);
    const channelVitality = staticAnalysis?.channel_vitality || null;
    const quantizationChannelVitality = quantization?.channel_vitality || null;
    check("CF-VITALITY-001", Boolean(channelVitality)
      && channelVitality.schema === ANALYZER_METADATA.schemas.channelVitality
      && quantizationChannelVitality?.schema === ANALYZER_METADATA.schemas.channelVitality
      && JSON.stringify(quantizationChannelVitality) === JSON.stringify(channelVitality), "Channel-vitality evidence is missing, schema-invalid, or inconsistent across static and quantization evidence.", ["/evidence/static_analysis/channel_vitality", "/evidence/quantization/channel_vitality"]);
    const assessedVitalityRows = (channelVitality?.ops || []).filter((row) => row.assessment_status === "assessed");
    const vitalitySum = (key) => assessedVitalityRows.reduce((total, row) => total + Number(row[key] || 0), 0);
    const vitalityRowsConserve = assessedVitalityRows.every((row) => {
      const count = Number(row.assessed_channel_count || 0);
      const arrays = [
        row.default_minimum_output_codes,
        row.default_maximum_output_codes,
        row.single_minimum_output_codes,
        row.single_maximum_output_codes,
        row.post_bias_sign_codes,
        row.default_constant_reason_codes,
        row.single_constant_reason_codes,
      ];
      if (!arrays.every((values) => Array.isArray(values) && values.length === count)) return false;
      let fixed = 0;
      let defaultConstant = 0;
      let singleConstant = 0;
      let dualConstant = 0;
      let modeDependent = 0;
      for (let index = 0; index < count; index += 1) {
        const defaultReason = Number(row.default_constant_reason_codes[index]);
        const singleReason = Number(row.single_constant_reason_codes[index]);
        if (![0, 1, 2, 3, 4].includes(defaultReason) || ![0, 1, 2, 3, 4].includes(singleReason)) return false;
        if (![-1, 0, 1].includes(Number(row.post_bias_sign_codes[index]))) return false;
        const endpointValues = [row.default_minimum_output_codes[index], row.default_maximum_output_codes[index], row.single_minimum_output_codes[index], row.single_maximum_output_codes[index]];
        if (endpointValues.every((value) => value != null)) fixed += 1;
        if (defaultReason !== 0) defaultConstant += 1;
        if (singleReason !== 0) singleConstant += 1;
        if (defaultReason !== 0 && singleReason !== 0) dualConstant += 1;
        if ((defaultReason === 0) !== (singleReason === 0)) modeDependent += 1;
      }
      return fixed === Number(row.fixed_point_assessed_channel_count || 0)
        && defaultConstant === Number(row.default_constant_output_channel_count || 0)
        && singleConstant === Number(row.single_constant_output_channel_count || 0)
        && dualConstant === Number(row.dual_mode_constant_output_channel_count || 0)
        && modeDependent === Number(row.mode_dependent_constant_output_channel_count || 0)
        && /^[a-f0-9]{64}$/.test(row.source_witness_ledger_sha256 || "")
        && /^[a-f0-9]{64}$/.test(row.vitality_ledger_sha256 || "");
    });
    check("CF-VITALITY-002", Boolean(channelVitality)
      && Number(channelVitality.assessed_op_count || 0) === assessedVitalityRows.length
      && Number(channelVitality.assessed_channel_count || 0) === vitalitySum("assessed_channel_count")
      && Number(channelVitality.fixed_point_assessed_channel_count || 0) === vitalitySum("fixed_point_assessed_channel_count")
      && Number(channelVitality.dual_mode_constant_output_channel_count || 0) === vitalitySum("dual_mode_constant_output_channel_count")
      && Number(channelVitality.nonconstant_accumulator_dual_mode_constant_channel_count || 0) === vitalitySum("nonconstant_accumulator_dual_mode_constant_channel_count")
      && Number(channelVitality.mode_dependent_constant_output_channel_count || 0) === vitalitySum("mode_dependent_constant_output_channel_count")
      && (channelVitality.span_histogram || []).reduce((total, bin) => total + Number(bin.default_channel_count || 0), 0) === Number(channelVitality.fixed_point_assessed_channel_count || 0)
      && (channelVitality.span_histogram || []).reduce((total, bin) => total + Number(bin.single_rounding_channel_count || 0), 0) === Number(channelVitality.fixed_point_assessed_channel_count || 0)
      && vitalityRowsConserve
      && (channelVitality.source_references || []).length >= 4
      && channelVitality.source_references.every((source) => /^[a-f0-9]{64}$/.test(source.sha256 || "") && String(source.url || "").includes(channelVitality.source_commit)), "Channel-vitality arrays, classifications, histogram, source bindings, or digest syntax are internally inconsistent.", ["/evidence/static_analysis/channel_vitality"]);
    check("CF-VITALITY-003", accumulatorReport.includes("## Quantized Channel Vitality Atlas (DERIVED EXACT MONOTONE ENDPOINT PROOF)")
      && accumulatorReport.includes(ANALYZER_METADATA.schemas.channelVitality)
      && accumulatorReport.includes(formatIntegerForConformance(channelVitality?.assessed_channel_count))
      && accumulatorReport.includes(formatIntegerForConformance(channelVitality?.nonconstant_accumulator_dual_mode_constant_channel_count))
      && accumulatorReport.includes("not an exact reachable-code count")
      && accumulatorReport.includes(channelVitality?.source_commit || "missing-source-commit"), "Engineering report does not preserve the channel-vitality schema, exact counts, proof boundary, or source commit.", ["/evidence/static_analysis/channel_vitality", "/engineering_report.md"]);
    const vitalityFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-QNT-0112");
    const vitalityHasVariableCollapse = Number(channelVitality?.nonconstant_accumulator_dual_mode_constant_channel_count || 0) > 0;
    const vitalityMlBomValid = compactMlBomEvidence || (witnessMlBomValue("deepbom:model:channelVitalitySchema") === String(channelVitality?.schema)
      && witnessMlBomValue("deepbom:model:channelVitalityAssessedChannels") === String(channelVitality?.assessed_channel_count)
      && witnessMlBomValue("deepbom:model:channelVitalityDualModeConstantChannels") === String(channelVitality?.dual_mode_constant_output_channel_count)
      && witnessMlBomValue("deepbom:model:channelVitalityVariableAccumulatorConstantChannels") === String(channelVitality?.nonconstant_accumulator_dual_mode_constant_channel_count)
      && witnessMlBomValue("deepbom:model:channelVitalityModeDependentConstantChannels") === String(channelVitality?.mode_dependent_constant_output_channel_count)
      && witnessMlBomValue("deepbom:model:channelVitalitySourceCommit") === String(channelVitality?.source_commit));
    check("CF-VITALITY-004", vitalityMlBomValid && (vitalityHasVariableCollapse
        ? vitalityFinding?.evidence_class === "DERIVED"
          && vitalityFinding?.technical_priority === "High"
          && (vitalityFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/channel_vitality")
        : !vitalityFinding), "ML-BOM or conditional authoritative finding does not preserve channel-vitality evidence.", ["/evidence/static_analysis/channel_vitality", "/evidence/mlbom_cyclonedx", "/evidence/findings_register/findings"]);
    const roundingEquivalence = staticAnalysis?.rounding_equivalence || null;
    const quantizationRoundingEquivalence = quantization?.rounding_equivalence || null;
    check("CF-ROUND-001", Boolean(roundingEquivalence)
      && roundingEquivalence.schema === ANALYZER_METADATA.schemas.roundingEquivalence
      && quantizationRoundingEquivalence?.schema === ANALYZER_METADATA.schemas.roundingEquivalence
      && JSON.stringify(quantizationRoundingEquivalence) === JSON.stringify(roundingEquivalence), "Rounding-equivalence evidence is missing, schema-invalid, or inconsistent across static and quantization evidence.", ["/evidence/static_analysis/rounding_equivalence", "/evidence/quantization/rounding_equivalence"]);
    const assessedEquivalenceRows = (roundingEquivalence?.ops || []).filter((row) => row.assessment_status === "assessed");
    const decimalSum = (values) => values.reduce((total, value) => total + BigInt(value || "0"), 0n);
    const rowDecimalSum = (key) => assessedEquivalenceRows.reduce((total, row) => total + BigInt(row[key] || "0"), 0n);
    const equivalenceRowsConserve = assessedEquivalenceRows.every((row) => {
      const count = Number(row.assessed_channel_count || 0);
      const arrays = [
        row.channel_interval_state_counts_decimal,
        row.channel_divergent_state_counts_decimal,
        row.channel_default_lower_state_counts_decimal,
        row.channel_default_higher_state_counts_decimal,
        row.channel_pair_segment_counts,
        row.channel_divergent_region_counts,
        row.channel_maximum_absolute_output_deltas,
        row.channel_first_divergent_accumulators_decimal,
        row.channel_first_default_output_codes,
        row.channel_first_single_output_codes,
      ];
      if (!arrays.every((values) => Array.isArray(values) && values.length === count)) return false;
      const divergent = row.channel_divergent_state_counts_decimal.map((value) => BigInt(value || "0"));
      return decimalSum(row.channel_interval_state_counts_decimal) === BigInt(row.interval_state_count_decimal || "0")
        && decimalSum(row.channel_divergent_state_counts_decimal) === BigInt(row.divergent_state_count_decimal || "0")
        && decimalSum(row.channel_default_lower_state_counts_decimal) === BigInt(row.default_lower_state_count_decimal || "0")
        && decimalSum(row.channel_default_higher_state_counts_decimal) === BigInt(row.default_higher_state_count_decimal || "0")
        && BigInt(row.default_lower_state_count_decimal || "0") + BigInt(row.default_higher_state_count_decimal || "0") === BigInt(row.divergent_state_count_decimal || "0")
        && divergent.filter((value) => value === 0n).length === Number(row.equivalent_channel_count || 0)
        && divergent.filter((value) => value > 0n).length === Number(row.divergent_channel_count || 0)
        && row.channel_pair_segment_counts.every((value) => Number.isInteger(value) && value >= 1 && value <= 511)
        && row.channel_divergent_region_counts.every((value, index) => Number.isInteger(value) && value >= 0 && value <= Number(row.channel_pair_segment_counts[index]))
        && row.channel_maximum_absolute_output_deltas.every((value, index) => Number.isSafeInteger(value) && value >= 0
          && ((divergent[index] === 0n && value === 0 && row.channel_first_divergent_accumulators_decimal[index] == null)
            || (divergent[index] > 0n && value > 0 && row.channel_first_divergent_accumulators_decimal[index] != null
              && row.channel_first_default_output_codes[index] !== row.channel_first_single_output_codes[index])))
        && /^[a-f0-9]{64}$/.test(row.source_witness_ledger_sha256 || "")
        && /^[a-f0-9]{64}$/.test(row.source_requantization_ledger_sha256 || "")
        && /^[a-f0-9]{64}$/.test(row.equivalence_ledger_sha256 || "");
    });
    check("CF-ROUND-002", Boolean(roundingEquivalence)
      && Number(roundingEquivalence.assessed_op_count || 0) === assessedEquivalenceRows.length
      && Number(roundingEquivalence.assessed_channel_count || 0) === assessedEquivalenceRows.reduce((total, row) => total + Number(row.assessed_channel_count || 0), 0)
      && Number(roundingEquivalence.equivalent_channel_count || 0) === assessedEquivalenceRows.reduce((total, row) => total + Number(row.equivalent_channel_count || 0), 0)
      && Number(roundingEquivalence.divergent_channel_count || 0) === assessedEquivalenceRows.reduce((total, row) => total + Number(row.divergent_channel_count || 0), 0)
      && BigInt(roundingEquivalence.interval_state_count_decimal || "0") === rowDecimalSum("interval_state_count_decimal")
      && BigInt(roundingEquivalence.divergent_state_count_decimal || "0") === rowDecimalSum("divergent_state_count_decimal")
      && BigInt(roundingEquivalence.default_lower_state_count_decimal || "0") === rowDecimalSum("default_lower_state_count_decimal")
      && BigInt(roundingEquivalence.default_higher_state_count_decimal || "0") === rowDecimalSum("default_higher_state_count_decimal")
      && Number(roundingEquivalence.pair_segment_count || 0) === assessedEquivalenceRows.reduce((total, row) => total + row.channel_pair_segment_counts.reduce((sum, value) => sum + Number(value), 0), 0)
      && Number(roundingEquivalence.divergent_region_count || 0) === assessedEquivalenceRows.reduce((total, row) => total + row.channel_divergent_region_counts.reduce((sum, value) => sum + Number(value), 0), 0)
      && (roundingEquivalence.divergence_histogram || []).reduce((total, bin) => total + Number(bin.channel_count || 0), 0) === Number(roundingEquivalence.assessed_channel_count || 0)
      && decimalSum((roundingEquivalence.divergence_histogram || []).map((bin) => bin.interval_state_count_decimal)) === BigInt(roundingEquivalence.interval_state_count_decimal || "0")
      && decimalSum((roundingEquivalence.divergence_histogram || []).map((bin) => bin.divergent_state_count_decimal)) === BigInt(roundingEquivalence.divergent_state_count_decimal || "0")
      && equivalenceRowsConserve
      && (roundingEquivalence.source_references || []).length >= 3
      && roundingEquivalence.source_references.every((source) => /^[a-f0-9]{64}$/.test(source.sha256 || "") && String(source.url || "").includes(roundingEquivalence.source_commit)), "Rounding-equivalence arrays, interval-state conservation, segment bound, source bindings, or digest syntax are internally inconsistent.", ["/evidence/static_analysis/rounding_equivalence"]);
    check("CF-ROUND-003", accumulatorReport.includes("## Fixed-Point Rounding Equivalence Lab (DERIVED EXACT CLOSED-INTERVAL CERTIFICATE)")
      && accumulatorReport.includes(ANALYZER_METADATA.schemas.roundingEquivalence)
      && accumulatorReport.includes(formatIntegerForConformance(roundingEquivalence?.interval_state_count_decimal))
      && accumulatorReport.includes(formatIntegerForConformance(roundingEquivalence?.divergent_state_count_decimal))
      && accumulatorReport.includes("interior integers can be unreachable")
      && accumulatorReport.includes(roundingEquivalence?.source_commit || "missing-source-commit"), "Engineering report does not preserve the rounding-equivalence schema, exact state counts, interval-hull boundary, or source commit.", ["/evidence/static_analysis/rounding_equivalence", "/engineering_report.md"]);
    const equivalenceFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-QNT-0113");
    const hasRoundingDivergence = Number(roundingEquivalence?.divergent_channel_count || 0) > 0;
    const roundingMlBomValid = compactMlBomEvidence || (witnessMlBomValue("deepbom:model:roundingEquivalenceSchema") === String(roundingEquivalence?.schema)
      && witnessMlBomValue("deepbom:model:roundingEquivalenceAssessedChannels") === String(roundingEquivalence?.assessed_channel_count)
      && witnessMlBomValue("deepbom:model:roundingEquivalenceEquivalentChannels") === String(roundingEquivalence?.equivalent_channel_count)
      && witnessMlBomValue("deepbom:model:roundingEquivalenceDivergentChannels") === String(roundingEquivalence?.divergent_channel_count)
      && witnessMlBomValue("deepbom:model:roundingEquivalenceIntervalStates") === String(roundingEquivalence?.interval_state_count_decimal)
      && witnessMlBomValue("deepbom:model:roundingEquivalenceDivergentStates") === String(roundingEquivalence?.divergent_state_count_decimal)
      && witnessMlBomValue("deepbom:model:roundingEquivalenceSourceCommit") === String(roundingEquivalence?.source_commit));
    check("CF-ROUND-004", roundingMlBomValid && (hasRoundingDivergence
        ? equivalenceFinding?.evidence_class === "DERIVED"
          && equivalenceFinding?.technical_priority === "Informational"
          && (equivalenceFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/rounding_equivalence")
        : !equivalenceFinding), "ML-BOM or conditional authoritative finding does not preserve rounding-equivalence evidence.", ["/evidence/static_analysis/rounding_equivalence", "/evidence/mlbom_cyclonedx", "/evidence/findings_register/findings"]);
    const accumulatorReachability = staticAnalysis?.accumulator_reachability || null;
    const quantizationAccumulatorReachability = quantization?.accumulator_reachability || null;
    let reachabilityShapeValidated = false;
    try {
      validateAccumulatorReachabilityShape(accumulatorReachability);
      reachabilityShapeValidated = true;
    } catch {
      reachabilityShapeValidated = false;
    }
    check("CF-REACH-001", Boolean(accumulatorReachability)
      && accumulatorReachability.schema === ANALYZER_METADATA.schemas.accumulatorReachability
      && quantizationAccumulatorReachability?.schema === ANALYZER_METADATA.schemas.accumulatorReachability
      && JSON.stringify(quantizationAccumulatorReachability) === JSON.stringify(accumulatorReachability)
      && reachabilityShapeValidated, "Accumulator-reachability evidence is missing, schema-invalid, inconsistent across exports, or fails its state partition.", ["/evidence/static_analysis/accumulator_reachability", "/evidence/quantization/accumulator_reachability"]);
    const reachabilityRows = (accumulatorReachability?.ops || []).filter((row) => row.assessment_status === "assessed");
    const reachabilityRowsConserve = reachabilityRows.every((row) => {
      const count = Number(row.assessed_channel_count || 0);
      const arrays = [
        row.channel_lattice_gcds,
        row.channel_proof_statuses,
        row.channel_certified_reachable_state_counts_decimal,
        row.channel_provably_unreachable_state_counts_decimal,
        row.channel_unresolved_state_counts_decimal,
        row.channel_exact_reachable_divergent_state_counts_decimal,
        row.channel_provably_unreachable_divergent_state_counts_decimal,
        row.channel_unresolved_divergent_state_counts_decimal,
        row.channel_first_exact_reachable_divergent_accumulators_decimal,
      ];
      if (!arrays.every((values) => Array.isArray(values) && values.length === count)) return false;
      const sumArray = (values) => values.reduce((total, value) => total + BigInt(value || "0"), 0n);
      const certified = sumArray(row.channel_certified_reachable_state_counts_decimal);
      const excluded = sumArray(row.channel_provably_unreachable_state_counts_decimal);
      const unresolved = sumArray(row.channel_unresolved_state_counts_decimal);
      const exactDivergent = sumArray(row.channel_exact_reachable_divergent_state_counts_decimal);
      const excludedDivergent = sumArray(row.channel_provably_unreachable_divergent_state_counts_decimal);
      const unresolvedDivergent = sumArray(row.channel_unresolved_divergent_state_counts_decimal);
      const proofCount = (status) => row.channel_proof_statuses.filter((value) => value === status).length;
      return certified === BigInt(row.certified_reachable_state_count_decimal || "0")
        && excluded === BigInt(row.provably_unreachable_state_count_decimal || "0")
        && unresolved === BigInt(row.unresolved_state_count_decimal || "0")
        && certified + excluded + unresolved === BigInt(row.interval_state_count_decimal || "0")
        && exactDivergent === BigInt(row.exact_reachable_divergent_state_count_decimal || "0")
        && excludedDivergent === BigInt(row.provably_unreachable_divergent_state_count_decimal || "0")
        && unresolvedDivergent === BigInt(row.unresolved_divergent_state_count_decimal || "0")
        && exactDivergent + excludedDivergent + unresolvedDivergent === BigInt(row.interval_divergent_state_count_decimal || "0")
        && proofCount("complete_integer_interval") === Number(row.complete_integer_interval_channel_count || 0)
        && proofCount("complete_modular_lattice") === Number(row.complete_modular_lattice_channel_count || 0)
        && proofCount("partial_endpoint_bands") === Number(row.partial_band_channel_count || 0)
        && proofCount("singleton") === Number(row.singleton_channel_count || 0)
        && row.channel_lattice_gcds.every((value) => Number.isSafeInteger(value) && value >= 0)
        && /^[a-f0-9]{64}$/.test(row.source_witness_ledger_sha256 || "")
        && /^[a-f0-9]{64}$/.test(row.source_rounding_equivalence_ledger_sha256 || "")
        && /^[a-f0-9]{64}$/.test(row.reachability_ledger_sha256 || "");
    });
    const reachabilityRowSum = (key) => reachabilityRows.reduce((total, row) => total + BigInt(row[key] || "0"), 0n);
    check("CF-REACH-002", Boolean(accumulatorReachability)
      && Number(accumulatorReachability.assessed_op_count || 0) === reachabilityRows.length
      && Number(accumulatorReachability.assessed_channel_count || 0) === reachabilityRows.reduce((total, row) => total + Number(row.assessed_channel_count || 0), 0)
      && BigInt(accumulatorReachability.interval_state_count_decimal || "0") === reachabilityRowSum("interval_state_count_decimal")
      && BigInt(accumulatorReachability.certified_reachable_state_count_decimal || "0") === reachabilityRowSum("certified_reachable_state_count_decimal")
      && BigInt(accumulatorReachability.provably_unreachable_state_count_decimal || "0") === reachabilityRowSum("provably_unreachable_state_count_decimal")
      && BigInt(accumulatorReachability.unresolved_state_count_decimal || "0") === reachabilityRowSum("unresolved_state_count_decimal")
      && BigInt(accumulatorReachability.interval_divergent_state_count_decimal || "0") === reachabilityRowSum("interval_divergent_state_count_decimal")
      && BigInt(accumulatorReachability.exact_reachable_divergent_state_count_decimal || "0") === reachabilityRowSum("exact_reachable_divergent_state_count_decimal")
      && BigInt(accumulatorReachability.provably_unreachable_divergent_state_count_decimal || "0") === reachabilityRowSum("provably_unreachable_divergent_state_count_decimal")
      && BigInt(accumulatorReachability.unresolved_divergent_state_count_decimal || "0") === reachabilityRowSum("unresolved_divergent_state_count_decimal")
      && reachabilityRowsConserve
      && (accumulatorReachability.source_references || []).length >= 1
      && accumulatorReachability.source_references.every((source) => /^[a-f0-9]{64}$/.test(source.sha256 || "") && String(source.url || "").includes(accumulatorReachability.source_commit)), "Accumulator-reachability channel arrays, exact state partitions, source bindings, or digest syntax are internally inconsistent.", ["/evidence/static_analysis/accumulator_reachability"]);
    check("CF-REACH-003", accumulatorReport.includes("## Accumulator Reachability Lattice (DERIVED EXACT KERNEL-LOCAL CERTIFICATE)")
      && accumulatorReport.includes(ANALYZER_METADATA.schemas.accumulatorReachability)
      && accumulatorReport.includes(formatIntegerForConformance(accumulatorReachability?.exact_reachable_divergent_state_count_decimal))
      && accumulatorReport.includes(formatIntegerForConformance(accumulatorReachability?.provably_unreachable_divergent_state_count_decimal))
      && accumulatorReport.includes(formatIntegerForConformance(accumulatorReachability?.unresolved_divergent_state_count_decimal))
      && accumulatorReport.includes("full-model-input reachability")
      && accumulatorReport.includes(accumulatorReachability?.source_commit || "missing-source-commit"), "Engineering report does not preserve the reachability schema, exact divergent partition, interpretation boundary, or source commit.", ["/evidence/static_analysis/accumulator_reachability", "/engineering_report.md"]);
    const reachabilityFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-QNT-0115");
    const hasExactReachableDivergence = BigInt(accumulatorReachability?.exact_reachable_divergent_state_count_decimal || "0") > 0n;
    const reachabilityMlBomValid = compactMlBomEvidence || (witnessMlBomValue("deepbom:model:accumulatorReachabilitySchema") === String(accumulatorReachability?.schema)
      && witnessMlBomValue("deepbom:model:accumulatorReachabilityAssessedChannels") === String(accumulatorReachability?.assessed_channel_count)
      && witnessMlBomValue("deepbom:model:accumulatorReachabilityCompleteIntegerChannels") === String(accumulatorReachability?.complete_integer_interval_channel_count)
      && witnessMlBomValue("deepbom:model:accumulatorReachabilityCompleteModularChannels") === String(accumulatorReachability?.complete_modular_lattice_channel_count)
      && witnessMlBomValue("deepbom:model:accumulatorReachabilityExactDivergentStates") === String(accumulatorReachability?.exact_reachable_divergent_state_count_decimal)
      && witnessMlBomValue("deepbom:model:accumulatorReachabilityExcludedDivergentStates") === String(accumulatorReachability?.provably_unreachable_divergent_state_count_decimal)
      && witnessMlBomValue("deepbom:model:accumulatorReachabilityUnresolvedDivergentStates") === String(accumulatorReachability?.unresolved_divergent_state_count_decimal)
      && witnessMlBomValue("deepbom:model:accumulatorReachabilitySourceCommit") === String(accumulatorReachability?.source_commit));
    check("CF-REACH-004", reachabilityMlBomValid && (hasExactReachableDivergence
        ? reachabilityFinding?.evidence_class === "DERIVED"
          && reachabilityFinding?.technical_priority === "Informational"
          && (reachabilityFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/accumulator_reachability")
        : !reachabilityFinding), "ML-BOM or conditional authoritative finding does not preserve accumulator-reachability evidence.", ["/evidence/static_analysis/accumulator_reachability", "/evidence/mlbom_cyclonedx", "/evidence/findings_register/findings"]);
    const abiPropagation = staticAnalysis?.numerical_abi_propagation || null;
    const quantizationAbiPropagation = quantization?.numerical_abi_propagation || null;
    let abiReconstructionValidated = false;
    try {
      validateNumericalAbiPropagationShape(abiPropagation);
      validateNumericalAbiPropagationAgainstReconstruction(abiPropagation, reconstructNumericalAbiPropagation(staticAnalysis));
      abiReconstructionValidated = true;
    } catch {
      abiReconstructionValidated = false;
    }
    check("CF-ABI-001", Boolean(abiPropagation)
      && abiPropagation.schema === ANALYZER_METADATA.schemas.numericalAbiPropagation
      && quantizationAbiPropagation?.schema === ANALYZER_METADATA.schemas.numericalAbiPropagation
      && JSON.stringify(quantizationAbiPropagation) === JSON.stringify(abiPropagation)
      && abiReconstructionValidated, "Numerical ABI propagation evidence is missing, schema-invalid, inconsistent across exports, or differs from independent graph reconstruction.", ["/evidence/static_analysis/numerical_abi_propagation", "/evidence/quantization/numerical_abi_propagation", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"]);
    const abiSources = abiPropagation?.sources || [];
    const abiDivergentSources = abiSources.filter((source) => source.assessment_status === "propagates_structurally");
    const abiExactSources = abiDivergentSources.filter((source) => source.local_reachability_status === "exact_local_counterexample");
    const abiBoundaryEdges = (abiPropagation?.unique_predicted_boundary_edge_indices || []).map((index) => abiPropagation?.graph_edges?.[index]).filter(Boolean);
    const abiExactBoundaryEdges = (abiPropagation?.exact_unique_predicted_boundary_edge_indices || []).map((index) => abiPropagation?.graph_edges?.[index]).filter(Boolean);
    const abiPayloadInventory = abiDivergentSources.reduce((total, source) => total + BigInt(source.assessed_boundary_logical_payload_bytes || 0), 0n);
    const abiRouteConservation = abiDivergentSources.every((source) => {
      const routeCounts = (source.model_output_paths || []).map((path) => path.exact_graph_route_count_decimal);
      return source.route_count_status === "assessed_acyclic"
        && routeCounts.every((value) => value != null)
        && routeCounts.reduce((total, value) => total + BigInt(value), 0n) === BigInt(source.exact_model_output_graph_route_count_decimal || "0")
        && /^[a-f0-9]{64}$/.test(source.source_equivalence_ledger_sha256 || "")
        && /^[a-f0-9]{64}$/.test(source.source_reachability_ledger_sha256 || "")
        && BigInt(source.exact_reachable_divergent_state_count_decimal || "0")
          + BigInt(source.provably_unreachable_divergent_state_count_decimal || "0")
          + BigInt(source.unresolved_divergent_state_count_decimal || "0") === BigInt(source.divergent_state_count_decimal || "0")
        && /^[a-f0-9]{64}$/.test(source.propagation_ledger_sha256 || "");
    });
    check("CF-ABI-002", Boolean(abiPropagation)
      && abiSources.length === Number(abiPropagation.candidate_source_op_count || 0)
      && abiDivergentSources.length === Number(abiPropagation.divergent_source_op_count || 0)
      && abiExactSources.length === Number(abiPropagation.exact_local_counterexample_source_op_count || 0)
      && abiExactSources.filter((source) => source.reachable_model_output_tensor_count > 0).length === Number(abiPropagation.exact_output_reachable_source_op_count || 0)
      && abiDivergentSources.reduce((total, source) => total + BigInt(source.divergent_state_count_decimal || "0"), 0n) === BigInt(abiPropagation.interval_divergent_state_count_decimal || "0")
      && abiDivergentSources.reduce((total, source) => total + BigInt(source.exact_reachable_divergent_state_count_decimal || "0"), 0n) === BigInt(abiPropagation.exact_local_divergent_state_count_decimal || "0")
      && abiDivergentSources.reduce((total, source) => total + BigInt(source.provably_unreachable_divergent_state_count_decimal || "0"), 0n) === BigInt(abiPropagation.residue_excluded_divergent_state_count_decimal || "0")
      && abiDivergentSources.reduce((total, source) => total + BigInt(source.unresolved_divergent_state_count_decimal || "0"), 0n) === BigInt(abiPropagation.unresolved_divergent_state_count_decimal || "0")
      && abiDivergentSources.reduce((total, source) => total + Number(source.corridor_edge_count || 0), 0) === Number(abiPropagation.source_corridor_edge_instance_count || 0)
      && abiDivergentSources.reduce((total, source) => total + Number(source.predicted_boundary_edge_count || 0), 0) === Number(abiPropagation.source_boundary_edge_instance_count || 0)
      && abiPayloadInventory === BigInt(abiPropagation.assessed_source_boundary_edge_instance_payload_bytes_decimal || "0")
      && abiBoundaryEdges.length === Number(abiPropagation.unique_predicted_boundary_edge_count || 0)
      && abiBoundaryEdges.every((edge) => edge.predicted_boundary)
      && (abiBoundaryEdges.some((edge) => edge.logical_payload_bytes == null)
        ? abiPropagation.unique_predicted_boundary_logical_payload_bytes == null
        : abiBoundaryEdges.reduce((total, edge) => total + Number(edge.logical_payload_bytes), 0) === Number(abiPropagation.unique_predicted_boundary_logical_payload_bytes))
      && abiExactSources.reduce((total, source) => total + Number(source.corridor_edge_count || 0), 0) === Number(abiPropagation.exact_source_corridor_edge_instance_count || 0)
      && abiExactBoundaryEdges.length === Number(abiPropagation.exact_unique_predicted_boundary_edge_count || 0)
      && abiExactBoundaryEdges.every((edge) => edge.predicted_boundary)
      && (abiExactBoundaryEdges.some((edge) => edge.logical_payload_bytes == null)
        ? abiPropagation.exact_unique_predicted_boundary_logical_payload_bytes == null
        : abiExactBoundaryEdges.reduce((total, edge) => total + Number(edge.logical_payload_bytes), 0) === Number(abiPropagation.exact_unique_predicted_boundary_logical_payload_bytes))
      && abiRouteConservation
      && abiPropagation.graph_cycle_status === "acyclic"
      && /^[a-f0-9]{64}$/.test(abiPropagation.graph_ledger_sha256 || ""), "Numerical ABI corridor, boundary, route, or digest ledgers do not conserve their exact source and graph inventories.", ["/evidence/static_analysis/numerical_abi_propagation"]);
    check("CF-ABI-003", accumulatorReport.includes("## Numerical ABI Propagation Atlas (DERIVED EXACT LOCAL SOURCE + STRUCTURAL CORRIDOR)")
      && accumulatorReport.includes(ANALYZER_METADATA.schemas.numericalAbiPropagation)
      && accumulatorReport.includes(formatIntegerForConformance(abiPropagation?.source_corridor_edge_instance_count))
      && accumulatorReport.includes(formatIntegerForConformance(abiPropagation?.maximum_model_output_graph_route_count_decimal))
      && accumulatorReport.includes(formatIntegerForConformance(abiPropagation?.exact_local_divergent_state_count_decimal))
      && accumulatorReport.includes(formatIntegerForConformance(abiPropagation?.residue_excluded_divergent_state_count_decimal))
      && accumulatorReport.includes(formatIntegerForConformance(abiPropagation?.unresolved_divergent_state_count_decimal))
      && accumulatorReport.includes(abiPropagation?.graph_ledger_sha256 || "missing-graph-ledger")
      && accumulatorReport.includes("repeated source exposure inventory, not physical runtime traffic")
      && accumulatorReport.includes("full-model input realizes the local assignment"), "Engineering report does not preserve the Numerical ABI schema, exact local-state partition, route/corridor counts, graph digest, repeated-inventory distinction, or interpretation boundary.", ["/evidence/static_analysis/numerical_abi_propagation", "/engineering_report.md"]);
    const abiFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-QNT-0114");
    const hasOutputReachableAbiSource = Number(abiPropagation?.exact_output_reachable_source_op_count || 0) > 0;
    const abiMlBomValid = compactMlBomEvidence || (witnessMlBomValue("deepbom:model:numericalAbiPropagationSchema") === String(abiPropagation?.schema)
      && witnessMlBomValue("deepbom:model:numericalAbiPropagationDivergentSources") === String(abiPropagation?.divergent_source_op_count)
      && witnessMlBomValue("deepbom:model:numericalAbiPropagationOutputReachableSources") === String(abiPropagation?.output_reachable_source_op_count)
      && witnessMlBomValue("deepbom:model:numericalAbiPropagationExactLocalSources") === String(abiPropagation?.exact_local_counterexample_source_op_count)
      && witnessMlBomValue("deepbom:model:numericalAbiPropagationExactLocalDivergentStates") === String(abiPropagation?.exact_local_divergent_state_count_decimal)
      && witnessMlBomValue("deepbom:model:numericalAbiPropagationResidueExcludedDivergentStates") === String(abiPropagation?.residue_excluded_divergent_state_count_decimal)
      && witnessMlBomValue("deepbom:model:numericalAbiPropagationUnresolvedDivergentStates") === String(abiPropagation?.unresolved_divergent_state_count_decimal)
      && witnessMlBomValue("deepbom:model:numericalAbiPropagationGraphEdges") === String(abiPropagation?.graph_edge_count)
      && witnessMlBomValue("deepbom:model:numericalAbiPropagationCorridorEdgeInstances") === String(abiPropagation?.source_corridor_edge_instance_count)
      && witnessMlBomValue("deepbom:model:numericalAbiPropagationMaximumRoutes") === String(abiPropagation?.maximum_model_output_graph_route_count_decimal ?? "not_assessed")
      && witnessMlBomValue("deepbom:model:numericalAbiPropagationGraphLedgerSha256") === String(abiPropagation?.graph_ledger_sha256));
    check("CF-ABI-004", abiMlBomValid && (hasOutputReachableAbiSource
        ? abiFinding?.evidence_class === "DERIVED"
          && abiFinding?.technical_priority === "Informational"
          && (abiFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/numerical_abi_propagation")
        : !abiFinding), "ML-BOM or conditional authoritative finding does not preserve Numerical ABI propagation evidence.", ["/evidence/static_analysis/numerical_abi_propagation", "/evidence/mlbom_cyclonedx", "/evidence/findings_register/findings"]);
    const inputCounterexample = staticAnalysis?.input_counterexample || null;
    const quantizationInputCounterexample = quantization?.input_counterexample || null;
    const inputWitness = inputCounterexample?.witnesses?.[0] || null;
    const expectsInputCounterexample = Object.prototype.hasOwnProperty.call(analysis || {}, "input_counterexample");
    let inputCounterexampleShapeValid = false;
    try {
      validateInputCounterexampleShape(inputCounterexample);
      for (const witness of inputCounterexample.witnesses || []) buildInputWitnessTensor(witness);
      inputCounterexampleShapeValid = true;
    } catch {
      inputCounterexampleShapeValid = false;
    }
    check("CF-IW-001", !expectsInputCounterexample || (Boolean(inputCounterexample)
      && inputCounterexample.schema === ANALYZER_METADATA.schemas.inputCounterexample
      && quantizationInputCounterexample?.schema === ANALYZER_METADATA.schemas.inputCounterexample
      && JSON.stringify(quantizationInputCounterexample) === JSON.stringify(inputCounterexample)
      && inputCounterexampleShapeValid), "Model-input counterexample evidence is missing, schema-invalid, inconsistent across exports, or cannot reconstruct its sparse full tensor.", ["/evidence/static_analysis/input_counterexample", "/evidence/quantization/input_counterexample"]);
    const inputSources = inputCounterexample?.sources || [];
    const constructiveSources = inputSources.filter((source) => source.classification === "tensor_abi_constructive");
    const upstreamSources = inputSources.filter((source) => source.classification === "upstream_activation_constraint_unresolved");
    const hasConstructiveInput = constructiveSources.length > 0;
    const inputDot = (inputWitness?.terms || []).reduce((total, term) => total + BigInt(term.term_product_decimal || "0"), 0n);
    const linkedInputSource = inputWitness ? inputSources.find((source) => source.op_index === inputWitness.source_op_index) : null;
    check("CF-IW-002", !expectsInputCounterexample || (Boolean(inputCounterexample)
      && inputSources.length === Number(inputCounterexample.exact_local_source_op_count || 0)
      && constructiveSources.length === Number(inputCounterexample.tensor_abi_constructive_source_op_count || 0)
      && upstreamSources.length === Number(inputCounterexample.upstream_activation_unresolved_source_op_count || 0)
      && constructiveSources.reduce((total, source) => total + Number(source.exact_reachable_divergent_channel_count || 0), 0) === Number(inputCounterexample.tensor_abi_constructive_channel_count || 0)
      && constructiveSources.reduce((total, source) => total + BigInt(source.exact_reachable_divergent_state_count_decimal || "0"), 0n) === BigInt(inputCounterexample.tensor_abi_constructive_divergent_state_count_decimal || "0")
      && (inputCounterexample.witnesses || []).length === Number(inputCounterexample.representative_witness_count || 0)
      && (hasConstructiveInput
        ? Boolean(inputWitness && linkedInputSource)
          && inputDot === BigInt(inputWitness.dot_product_decimal || "0")
          && inputDot + BigInt(inputWitness.bias_decimal || "0") === BigInt(inputWitness.post_bias_accumulator_decimal || "0")
          && Number(inputWitness.default_output_code) - Number(inputWitness.single_rounding_output_code) === Number(inputWitness.output_code_delta)
          && inputWitness.default_output_code !== inputWitness.single_rounding_output_code
          && linkedInputSource.representative_witness_ledger_sha256 === inputWitness.witness_ledger_sha256
          && /^[a-f0-9]{64}$/.test(inputWitness.full_model_input_tensor_sha256 || "")
          && /^[a-f0-9]{64}$/.test(inputWitness.witness_ledger_sha256 || "")
        : inputWitness == null && Number(inputCounterexample.representative_witness_count || 0) === 0)
      && /^[a-f0-9]{64}$/.test(inputCounterexample.portfolio_ledger_sha256 || "")), "Model-input counterexample counts, exact arithmetic, source joins, or SHA-256 ledgers do not conserve.", ["/evidence/static_analysis/input_counterexample"]);
    const inputReportHeading = inputCounterexample?.status === "not_applicable"
      ? "## Model Input Tensor ABI Witness (NOT_APPLICABLE)"
      : hasConstructiveInput ? "## Model Input Tensor ABI Witness (DERIVED CONSTRUCTIVE EXISTENCE CERTIFICATE)"
        : "## Model Input Tensor ABI Witness (DERIVED SOURCE CLASSIFICATION; NO CONSTRUCTIVE WITNESS)";
    check("CF-IW-003", !expectsInputCounterexample || (accumulatorReport.includes(inputReportHeading)
      && accumulatorReport.includes(ANALYZER_METADATA.schemas.inputCounterexample)
      && accumulatorReport.includes(formatIntegerForConformance(inputCounterexample?.tensor_abi_constructive_divergent_state_count_decimal))
      && (hasConstructiveInput
        ? accumulatorReport.includes(formatIntegerForConformance(inputWitness?.model_input_element_count))
          && accumulatorReport.includes(inputWitness?.full_model_input_tensor_sha256 || "missing-input-tensor-sha")
          && accumulatorReport.includes(inputWitness?.witness_ledger_sha256 || "missing-witness-ledger")
          && accumulatorReport.includes("exact at the model tensor ABI")
          && accumulatorReport.includes("does not prove a declared model output changes")
        : accumulatorReport.includes("No complete model-input tensor witness was constructed."))), "Engineering report does not preserve the complete input construction, arithmetic, digests, proof scope, or interpretation boundary.", ["/evidence/static_analysis/input_counterexample", "/engineering_report.md"]);
    const inputFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-QNT-0116");
    const inputCounterexampleMlBomValid = compactMlBomEvidence || (witnessMlBomValue("deepbom:model:inputCounterexampleSchema") === String(inputCounterexample?.schema)
      && witnessMlBomValue("deepbom:model:inputCounterexampleExactLocalSources") === String(inputCounterexample?.exact_local_source_op_count)
      && witnessMlBomValue("deepbom:model:inputCounterexampleConstructiveSources") === String(inputCounterexample?.tensor_abi_constructive_source_op_count)
      && witnessMlBomValue("deepbom:model:inputCounterexampleConstructiveChannels") === String(inputCounterexample?.tensor_abi_constructive_channel_count)
      && witnessMlBomValue("deepbom:model:inputCounterexampleConstructiveDivergentStates") === String(inputCounterexample?.tensor_abi_constructive_divergent_state_count_decimal)
      && witnessMlBomValue("deepbom:model:inputCounterexamplePortfolioLedgerSha256") === String(inputCounterexample?.portfolio_ledger_sha256)
      && (hasConstructiveInput
        ? witnessMlBomValue("deepbom:model:inputCounterexampleFullTensorSha256") === String(inputWitness?.full_model_input_tensor_sha256)
          && witnessMlBomValue("deepbom:model:inputCounterexampleWitnessLedgerSha256") === String(inputWitness?.witness_ledger_sha256)
        : witnessMlBomValue("deepbom:model:inputCounterexampleFullTensorSha256") === "not_assessed"
          && witnessMlBomValue("deepbom:model:inputCounterexampleWitnessLedgerSha256") === "not_assessed"));
    const inputCounterexampleFindingValid = hasConstructiveInput
      ? inputFinding?.evidence_class === "DERIVED"
          && inputFinding?.technical_priority === "Medium"
          && (inputFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/input_counterexample")
      : !inputFinding;
    check("CF-IW-004", !expectsInputCounterexample || (inputCounterexampleMlBomValid && inputCounterexampleFindingValid), "ML-BOM or conditional authoritative finding does not preserve the model-input counterexample evidence.", ["/evidence/static_analysis/input_counterexample", "/evidence/mlbom_cyclonedx", "/evidence/findings_register/findings"]);
    const preprocessing = staticAnalysis?.preprocessing_realizability || null;
    const quantizationPreprocessing = quantization?.preprocessing_realizability || null;
    const expectsPreprocessing = Object.prototype.hasOwnProperty.call(analysis || {}, "preprocessing_realizability");
    const preprocessingCandidates = preprocessing?.candidates || [];
    const preprocessingAssessed = preprocessingCandidates.filter((candidate) => candidate.status === "assessed");
    const preprocessingExact = preprocessingAssessed.filter((candidate) => candidate.exact_tensor_realization);
    const preprocessingNonExact = preprocessingAssessed.filter((candidate) => !candidate.exact_tensor_realization);
    check("CF-PR-001", !expectsPreprocessing || (Boolean(preprocessing)
      && preprocessing.schema === ANALYZER_METADATA.schemas.preprocessingRealizability
      && quantizationPreprocessing?.schema === ANALYZER_METADATA.schemas.preprocessingRealizability
      && JSON.stringify(quantizationPreprocessing) === JSON.stringify(preprocessing)
      && preprocessing.source_input_counterexample_portfolio_sha256 === inputCounterexample?.portfolio_ledger_sha256), "Preprocessing realizability evidence is missing, schema-invalid, inconsistent across exports, or detached from the input witness portfolio.", ["/evidence/static_analysis/preprocessing_realizability", "/evidence/quantization/preprocessing_realizability", "/evidence/static_analysis/input_counterexample"]);
    check("CF-PR-002", !expectsPreprocessing || (preprocessingCandidates.length === Number(preprocessing?.candidate_evaluation_count || 0)
      && preprocessingAssessed.length === Number(preprocessing?.assessed_candidate_count || 0)
      && preprocessingExact.length === Number(preprocessing?.exact_tensor_realization_candidate_count || 0)
      && preprocessingNonExact.length === Number(preprocessing?.non_exact_candidate_count || 0)
      && preprocessingExact.every((candidate) => candidate.exact_tensor_element_count === candidate.witness_tensor_element_count
        && candidate.unrealizable_tensor_element_count === 0
        && candidate.minimum_total_absolute_tensor_code_error_decimal === "0"
        && candidate.exact_rgb_fixture_sha256 === candidate.nearest_rgb_fixture_sha256)
      && preprocessingCandidates.every((candidate) => (candidate.channel_maps || []).every((map) => (map.pixel_to_tensor_codes || []).length === 256
        && map.reachable_tensor_code_count + map.tensor_code_hole_count === 256)
        && /^[a-f0-9]{64}$/.test(candidate.nearest_rgb_fixture_sha256 || "")
        && /^[a-f0-9]{64}$/.test(candidate.candidate_ledger_sha256 || ""))
      && /^[a-f0-9]{64}$/.test(preprocessing?.portfolio_ledger_sha256 || "")), "Preprocessing finite-domain LUT, exact/non-exact partition, fixture, or digest conservation failed.", ["/evidence/static_analysis/preprocessing_realizability"]);
    check("CF-PR-003", !expectsPreprocessing || (accumulatorReport.includes("## Pixel-to-Tensor Contract Lab (DERIVED EXHAUSTIVE COUNTERFACTUAL MATRIX)")
      && accumulatorReport.includes(ANALYZER_METADATA.schemas.preprocessingRealizability)
      && accumulatorReport.includes(formatIntegerForConformance(preprocessing?.exact_tensor_realization_candidate_count))
      && accumulatorReport.includes(formatIntegerForConformance(preprocessing?.best_non_exact_unrealizable_element_count))
      && accumulatorReport.includes(preprocessing?.portfolio_ledger_sha256 || "missing-preprocessing-ledger")
      && accumulatorReport.includes("explicit counterfactual contracts, not observations of the production application")), "Engineering report does not preserve the preprocessing contract matrix, finite-domain counts, digest, or interpretation boundary.", ["/evidence/static_analysis/preprocessing_realizability", "/engineering_report.md"]);
    const preprocessingFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-QNT-0117");
    const hasMixedPreprocessingOutcome = preprocessingExact.length > 0 && preprocessingNonExact.length > 0;
    const preprocessingMlBomValid = compactMlBomEvidence || (witnessMlBomValue("deepbom:model:preprocessingRealizabilitySchema") === String(preprocessing?.schema)
      && witnessMlBomValue("deepbom:model:preprocessingRealizabilityCandidateEvaluations") === String(preprocessing?.candidate_evaluation_count)
      && witnessMlBomValue("deepbom:model:preprocessingRealizabilityExactCandidates") === String(preprocessing?.exact_tensor_realization_candidate_count)
      && witnessMlBomValue("deepbom:model:preprocessingRealizabilityNonExactCandidates") === String(preprocessing?.non_exact_candidate_count)
      && witnessMlBomValue("deepbom:model:preprocessingRealizabilityPortfolioLedgerSha256") === String(preprocessing?.portfolio_ledger_sha256));
    check("CF-PR-004", !expectsPreprocessing || (preprocessingMlBomValid && (hasMixedPreprocessingOutcome
        ? preprocessingFinding?.evidence_class === "DERIVED"
          && preprocessingFinding?.technical_priority === "Medium"
          && (preprocessingFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/preprocessing_realizability")
        : !preprocessingFinding)), "ML-BOM or conditional authoritative finding does not preserve preprocessing realizability evidence.", ["/evidence/static_analysis/preprocessing_realizability", "/evidence/mlbom_cyclonedx", "/evidence/findings_register/findings"]);
    const preprocessingConsequence = runtimeResults?.preprocessing_consequence_atlas || null;
    const expectsPreprocessingConsequence = preprocessingConsequence != null;
    const consequenceCandidates = preprocessingConsequence?.candidates || [];
    const consequenceInputClasses = preprocessingConsequence?.input_equivalence_classes || [];
    const consequenceOutputClasses = preprocessingConsequence?.output_equivalence_classes || [];
    check("CF-PC-001", !expectsPreprocessingConsequence || (preprocessingConsequence.schema === ANALYZER_METADATA.schemas.preprocessingConsequence
      && preprocessingConsequence.evidence_class === "MEASURED_SYNTHETIC"
      && preprocessingConsequence.status === "assessed"
      && preprocessingConsequence.artifact_sha256 === analysis?.model_sha256
      && preprocessingConsequence.source_preprocessing_portfolio_sha256 === preprocessing?.portfolio_ledger_sha256
      && preprocessingConsequence.source_input_witness_tensor_sha256 === inputWitness?.full_model_input_tensor_sha256), "Preprocessing consequence evidence is schema-invalid or detached from the artifact, finite-domain portfolio, or tensor witness.", ["/evidence/runtime_results/preprocessing_consequence_atlas", "/evidence/static_analysis/preprocessing_realizability", "/evidence/static_analysis/input_counterexample"]);
    check("CF-PC-002", !expectsPreprocessingConsequence || (consequenceCandidates.length === Number(preprocessingConsequence.candidate_count || 0)
      && consequenceCandidates.length === preprocessingCandidates.length
      && consequenceCandidates.every((row, index) => row.contract_id === preprocessingCandidates[index]?.contract_id
        && row.source_candidate_ledger_sha256 === preprocessingCandidates[index]?.candidate_ledger_sha256
        && row.deterministic_replay === true
        && /^[a-f0-9]{64}$/.test(row.input_tensor_sha256 || "")
        && /^[a-f0-9]{64}$/.test(row.output_tensor_set_sha256 || "")
        && /^[a-f0-9]{64}$/.test(row.candidate_ledger_sha256 || ""))
      && consequenceInputClasses.length === Number(preprocessingConsequence.unique_input_tensor_count || 0)
      && consequenceOutputClasses.length === Number(preprocessingConsequence.unique_output_tensor_set_count || 0)
      && consequenceInputClasses.reduce((total, row) => total + Number(row.candidate_count || 0), 0) === consequenceCandidates.length
      && consequenceOutputClasses.reduce((total, row) => total + Number(row.candidate_count || 0), 0) === consequenceCandidates.length
      && preprocessingConsequence.execution_contract?.captured_repetitions_per_input === 2
      && preprocessingConsequence.execution_contract?.deterministic_replay_required === true
      && preprocessingConsequence.exact_contract_output_conservation === true
      && /^[a-f0-9]{64}$/.test(preprocessingConsequence.portfolio_ledger_sha256 || "")), "Preprocessing consequence replay counts, deterministic captures, equivalence classes, or SHA-256 ledgers do not conserve.", ["/evidence/runtime_results/preprocessing_consequence_atlas"]);
    check("CF-PC-003", !expectsPreprocessingConsequence || (accumulatorReport.includes("## Preprocessing Consequence Atlas (MEASURED_SYNTHETIC)")
      && accumulatorReport.includes(preprocessingConsequence.schema)
      && accumulatorReport.includes(preprocessingConsequence.portfolio_ledger_sha256)
      && accumulatorReport.includes(formatIntegerForConformance(preprocessingConsequence.unique_input_tensor_count))
      && accumulatorReport.includes(formatIntegerForConformance(preprocessingConsequence.unique_output_tensor_set_count))
      && accumulatorReport.includes("does not observe the production decoder")), "Engineering report does not preserve the measured preprocessing consequence matrix, digest, or interpretation boundary.", ["/evidence/runtime_results/preprocessing_consequence_atlas", "/engineering_report.md"]);
    const consequenceFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-RUN-0003");
    const consequenceChanged = Number(preprocessingConsequence?.output_changed_candidate_count || 0) > 0;
    const preprocessingConsequenceMlBomValid = compactMlBomEvidence || (witnessMlBomValue("deepbom:runtime:preprocessingConsequenceSchema") === String(preprocessingConsequence?.schema)
      && witnessMlBomValue("deepbom:runtime:preprocessingConsequenceCandidateReplays") === String(preprocessingConsequence.candidate_count)
      && witnessMlBomValue("deepbom:runtime:preprocessingConsequenceUniqueInputs") === String(preprocessingConsequence.unique_input_tensor_count)
      && witnessMlBomValue("deepbom:runtime:preprocessingConsequenceUniqueOutputs") === String(preprocessingConsequence.unique_output_tensor_set_count)
      && witnessMlBomValue("deepbom:runtime:preprocessingConsequenceOutputChangedCandidates") === String(preprocessingConsequence.output_changed_candidate_count)
      && witnessMlBomValue("deepbom:runtime:preprocessingConsequencePortfolioLedgerSha256") === String(preprocessingConsequence.portfolio_ledger_sha256));
    check("CF-PC-004", !expectsPreprocessingConsequence || (preprocessingConsequenceMlBomValid && (consequenceChanged
        ? consequenceFinding?.evidence_class === "MEASURED_SYNTHETIC"
          && consequenceFinding?.technical_priority === "Medium"
          && (consequenceFinding.evidence_json_pointers || []).includes("/evidence/runtime_results/preprocessing_consequence_atlas")
        : !consequenceFinding)), "ML-BOM or conditional authoritative finding does not preserve preprocessing consequence runtime evidence.", ["/evidence/runtime_results/preprocessing_consequence_atlas", "/evidence/mlbom_cyclonedx", "/evidence/findings_register/findings"]);
    const lattice = staticAnalysis?.quantization_lattice || null;
    let latticeValidated = false;
    try {
      latticeValidated = Boolean(lattice && validateQuantizationLattice(lattice, staticAnalysis));
    } catch {
      latticeValidated = false;
    }
    check("CF-LATTICE-001", Boolean(lattice)
      && lattice.schema === ANALYZER_METADATA.schemas.quantizationLattice
      && quantization?.residual_quantization_lattice?.schema === ANALYZER_METADATA.schemas.quantizationLattice
      && JSON.stringify(quantization.residual_quantization_lattice) === JSON.stringify(lattice), "Residual quantization-lattice evidence is missing, schema-invalid, or inconsistent across static and quantization evidence.", ["/evidence/static_analysis/quantization_lattice", "/evidence/quantization/residual_quantization_lattice"]);
    check("CF-LATTICE-002", latticeValidated, "Residual quantization-lattice rows do not reproduce from the artifact tensor contracts and exhaustive code-pair enumeration.", ["/evidence/static_analysis/quantization_lattice", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"]);
    const latticeReport = String(engineeringReport || "");
    check("CF-LATTICE-003", latticeReport.includes("## Quantization Lattice Lab (DERIVED EXHAUSTIVE DOMAIN)")
      && latticeReport.includes(ANALYZER_METADATA.schemas.quantizationLattice)
      && (lattice?.domain_escape_ranking_op_indices || []).slice(0, 32).every((index) => latticeReport.includes(`#${String(index).padStart(3, "0")} ADD`))
      && latticeReport.includes("not over observed activation values or their probability distribution")
      && latticeReport.includes("counterfactual output contracts")
      && latticeReport.includes("min_binary64"), "Engineering report does not preserve the residual-lattice schema, ranked rows, containment design, or interpretation boundary.", ["/evidence/static_analysis/quantization_lattice", "/engineering_report.md"]);
    const migration = staticAnalysis?.contract_migration || null;
    const quantizationMigration = quantization?.residual_contract_migration || null;
    check("CF-MIGRATION-001", Boolean(migration)
      && migration.schema === ANALYZER_METADATA.schemas.contractMigration
      && quantizationMigration?.schema === ANALYZER_METADATA.schemas.contractMigration
      && JSON.stringify(quantizationMigration) === JSON.stringify(migration), "Contract-migration evidence is missing, schema-invalid, or inconsistent across static and quantization evidence.", ["/evidence/static_analysis/contract_migration", "/evidence/quantization/residual_contract_migration"]);
    const migrationRows = migration?.migrations || [];
    const migrationScenarios = migrationRows.flatMap((row) => row.scenarios || []);
    const migrationConsumers = migrationRows.flatMap((row) => row.direct_consumers || []);
    const migrationKernelConsumers = migrationScenarios.flatMap((scenario) => scenario.kernel_consumers || []);
    const migrationAddConsumers = migrationScenarios.flatMap((scenario) => scenario.add_consumers || []);
    const latticeByOp = new Map((lattice?.residual_adds || []).map((row) => [Number(row.op_index), row]));
    const candidateFor = (row, design) => design === "fixed_zero_point_minimum_containment"
      ? row?.fixed_zero_point_containment : row?.globally_finest_containment;
    check("CF-MIGRATION-002", Boolean(migration)
      && migrationRows.length === migration.residual_contract_count
      && migrationScenarios.length === migration.candidate_scenario_count
      && migrationConsumers.length === migration.direct_consumer_count
      && migrationConsumers.reduce((total, consumer) => total + Number(consumer.input_slots?.length || 0), 0) === migration.direct_consumer_edge_count
      && migrationScenarios.reduce((total, scenario) => total + Number(scenario.assessed_consumer_count || 0), 0) === migration.assessed_consumer_scenario_count
      && migrationScenarios.reduce((total, scenario) => total + Number(scenario.unassessed_consumer_count || 0), 0) === migration.unassessed_consumer_scenario_count
      && migrationScenarios.reduce((total, scenario) => total + Number(scenario.assessed_kernel_channel_count || 0), 0) === migration.assessed_kernel_channel_scenario_count
      && migrationKernelConsumers.filter((consumer) => consumer.assessment_status === "assessed").every((consumer) => consumer.channel_current_quantized_multipliers?.length === consumer.assessed_channel_count
        && consumer.channel_candidate_quantized_multipliers?.length === consumer.assessed_channel_count
        && consumer.channel_current_shifts?.length === consumer.assessed_channel_count
        && consumer.channel_candidate_shifts?.length === consumer.assessed_channel_count
        && consumer.channel_current_bias_codes?.length === consumer.assessed_channel_count
        && consumer.channel_candidate_bias_code_decimals?.length === consumer.assessed_channel_count
        && consumer.channel_bias_rebase_error_current_steps?.length === consumer.assessed_channel_count
        && /^[a-f0-9]{64}$/.test(consumer.channel_ledger_sha256 || ""))
      && migrationAddConsumers.filter((consumer) => consumer.assessment_status === "assessed").every((consumer) => consumer.current_parameters?.left_shift === 20
        && consumer.candidate_parameters?.left_shift === 20
        && consumer.current_parameters?.input_multipliers?.length === 2
        && consumer.candidate_parameters?.input_multipliers?.length === 2)
      && migrationRows.every((row) => {
        const latticeRow = latticeByOp.get(Number(row.source_add_op_index));
        return latticeRow && Number(row.output_tensor_index) === Number(latticeRow.output_tensor_index)
          && row.scenarios.every((scenario) => {
            const candidate = candidateFor(latticeRow, scenario.design);
            return candidate && Number(candidate.output_scale) === Number(scenario.candidate_output_scale)
              && Number(candidate.output_zero_point) === Number(scenario.candidate_output_zero_point);
          });
      })
      && (migration.source_references || []).length === 3
      && migration.source_references.every((source) => /^[a-f0-9]{64}$/.test(source.sha256 || "")
        && String(source.url || "").includes(migration.source_commit)), "Contract-migration counts, channel arrays, ADD parameters, lattice bindings, or source identities are internally inconsistent.", ["/evidence/static_analysis/contract_migration", "/evidence/static_analysis/quantization_lattice"]);
    check("CF-MIGRATION-003", latticeReport.includes("## Contract Migration Impact Lab (DERIVED COUNTERFACTUAL RE-EXPORT)")
      && latticeReport.includes(ANALYZER_METADATA.schemas.contractMigration)
      && migrationRows.every((row) => latticeReport.includes(`#${String(row.source_add_op_index).padStart(3, "0")} ADD`))
      && latticeReport.includes("counterfactual re-export impact analysis")
      && latticeReport.includes("structural behavior-impact radius only")
      && latticeReport.includes("bias_int_new=round_ties_away")
      && latticeReport.includes(migration?.source_commit || "missing-source-commit"), "Engineering report does not preserve the migration schema, complete residual portfolio, formulas, source commit, or interpretation boundary.", ["/evidence/static_analysis/contract_migration", "/engineering_report.md"]);
    const migrationMlBomProperties = [...(mlBomDocument?.metadata?.component?.properties || []), ...(mlBomDocument?.properties || [])];
    const migrationMlBomValue = (name) => migrationMlBomProperties.find((item) => item.name === name)?.value;
    check("CF-MIGRATION-004", compactMlBomEvidence || (migrationMlBomValue("deepbom:model:contractMigrationSchema") === String(migration?.schema)
      && migrationMlBomValue("deepbom:model:contractMigrationKernelChannelScenarios") === String(migration?.assessed_kernel_channel_scenario_count)
      && migrationMlBomValue("deepbom:model:contractMigrationChangedBiasCodes") === String(migration?.bias_code_changed_channel_scenario_count)
      && migrationMlBomValue("deepbom:model:contractMigrationSourceCommit") === String(migration?.source_commit)), "ML-BOM contract-migration properties do not match structured static evidence.", ["/evidence/static_analysis/contract_migration", "/evidence/mlbom_cyclonedx"]);
    const stepResponse = staticAnalysis?.residual_step_response || null;
    const quantizationStepResponse = quantization?.residual_step_response || null;
    check("CF-STEP-001", Boolean(stepResponse)
      && stepResponse.schema === ANALYZER_METADATA.schemas.residualStepResponse
      && quantizationStepResponse?.schema === ANALYZER_METADATA.schemas.residualStepResponse
      && JSON.stringify(quantizationStepResponse) === JSON.stringify(stepResponse), "Residual step-response evidence is missing, schema-invalid, or inconsistent across static and quantization evidence.", ["/evidence/static_analysis/residual_step_response", "/evidence/quantization/residual_step_response"]);
    const stepContracts = (stepResponse?.residual_adds || []).flatMap((row) => row.contracts || []);
    const stepRowsByOp = new Map((stepResponse?.residual_adds || []).map((row) => [Number(row.op_index), row]));
    const stepLatticeByOp = new Map((staticAnalysis?.quantization_lattice?.residual_adds || []).map((row) => [Number(row.op_index), row]));
    const stepContractsConserve = stepContracts.every((contract) => {
      const branches = contract.branch_responses || [];
      const branchTransitions = branches.reduce((total, branch) => total + Number(branch.transition_count || 0), 0);
      const branchVisible = branches.reduce((total, branch) => total + Number(branch.visible_transition_count || 0), 0);
      const branchSilent = branches.reduce((total, branch) => total + Number(branch.silent_transition_count || 0), 0);
      const jointClasses = Number(contract.both_branches_visible_cell_count || 0)
        + Number(contract.input_0_only_visible_cell_count || 0)
        + Number(contract.input_1_only_visible_cell_count || 0)
        + Number(contract.neither_branch_visible_cell_count || 0);
      return branches.length === 2
        && branches.every((branch) => Number(branch.visible_transition_count || 0) + Number(branch.silent_transition_count || 0) === Number(branch.transition_count || 0))
        && branchTransitions === Number(contract.total_transition_count || 0)
        && branchVisible === Number(contract.visible_transition_count || 0)
        && branchSilent === Number(contract.silent_transition_count || 0)
        && branchVisible + branchSilent === branchTransitions
        && jointClasses === Number(contract.joint_interior_cell_count || 0)
        && /^[a-f0-9]{64}$/.test(contract.transition_ledger_sha256 || "")
        && (contract.tile_joint_cell_counts || []).every((count, index) => count === Number(contract.tile_both_branches_visible_counts?.[index] || 0)
          + Number(contract.tile_input_0_only_visible_counts?.[index] || 0)
          + Number(contract.tile_input_1_only_visible_counts?.[index] || 0)
          + Number(contract.tile_neither_branch_visible_counts?.[index] || 0));
    });
    const stepContractsBindLattice = [...stepRowsByOp].every(([opIndex, row]) => {
      if (row.assessment_status !== "assessed") return !(row.contracts || []).length;
      const latticeRow = stepLatticeByOp.get(opIndex);
      const current = (row.contracts || []).find((contract) => contract.design === "current_artifact_contract");
      const fixed = (row.contracts || []).find((contract) => contract.design === "fixed_zero_point_minimum_containment");
      const global = (row.contracts || []).find((contract) => contract.design === "globally_finest_minimum_containment");
      return latticeRow && current && fixed && global
        && Number(current.output_scale) === Number(latticeRow.output_scale)
        && Number(current.output_zero_point) === Number(latticeRow.output_zero_point)
        && Number(fixed.output_scale) === Number(latticeRow.fixed_zero_point_containment?.output_scale)
        && Number(fixed.output_zero_point) === Number(latticeRow.fixed_zero_point_containment?.output_zero_point)
        && Number(global.output_scale) === Number(latticeRow.globally_finest_containment?.output_scale)
        && Number(global.output_zero_point) === Number(latticeRow.globally_finest_containment?.output_zero_point);
    });
    check("CF-STEP-002", stepContracts.length === stepResponse?.contract_response_count
      && stepContractsConserve
      && stepContractsBindLattice
      && stepContracts.every((contract) => /^[a-f0-9]{64}$/.test(contract.transition_ledger_sha256 || "")
        && (contract.tile_joint_cell_counts || []).every((count, index) => count === Number(contract.tile_both_branches_visible_counts?.[index] || 0)
          + Number(contract.tile_input_0_only_visible_counts?.[index] || 0)
          + Number(contract.tile_input_1_only_visible_counts?.[index] || 0)
          + Number(contract.tile_neither_branch_visible_counts?.[index] || 0))), "Residual step-response counts, branch arrays, influence tiles, lattice contracts, or transition ledger identities are internally inconsistent.", ["/evidence/static_analysis/residual_step_response", "/evidence/static_analysis/quantization_lattice"]);
    const stepReport = String(engineeringReport || "");
    check("CF-STEP-003", stepReport.includes("## Residual Step Response Lab (DERIVED EXHAUSTIVE ADJACENT-CODE DOMAIN)")
      && stepReport.includes(ANALYZER_METADATA.schemas.residualStepResponse)
      && stepReport.includes(formatIntegerForConformance(stepResponse?.total_transition_count))
      && stepReport.includes("not an observed activation distribution")
      && stepReport.includes("does not prove a runtime branch is inactive"), "Engineering report does not preserve the exhaustive step-response schema, counts, or interpretation boundary.", ["/evidence/static_analysis/residual_step_response", "/engineering_report.md"]);
    const stepFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-QNT-0109");
    const stepMlBomValue = (name) => migrationMlBomProperties.find((item) => item.name === name)?.value;
    const stepMlBomValid = compactMlBomEvidence || (stepMlBomValue("deepbom:model:residualStepResponseSchema") === String(stepResponse?.schema)
      && stepMlBomValue("deepbom:model:residualStepResponseTransitions") === String(stepResponse?.total_transition_count)
      && stepMlBomValue("deepbom:model:residualStepResponseContainmentAddedSilent") === String(stepResponse?.containment_additional_silent_transition_count));
    check("CF-STEP-004", stepMlBomValid
      && (!stepResponse?.assessed_add_count || (stepFinding?.evidence_class === "DERIVED"
        && (stepFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/residual_step_response"))), "ML-BOM or authoritative finding does not preserve the residual step-response evidence.", ["/evidence/static_analysis/residual_step_response", "/evidence/mlbom_cyclonedx", "/evidence/findings_register/findings"]);
    const distortion = staticAnalysis?.residual_contract_distortion || null;
    const quantizationDistortion = quantization?.residual_contract_distortion || null;
    check("CF-DISTORTION-001", Boolean(distortion)
      && distortion.schema === ANALYZER_METADATA.schemas.residualContractDistortion
      && quantizationDistortion?.schema === ANALYZER_METADATA.schemas.residualContractDistortion
      && JSON.stringify(quantizationDistortion) === JSON.stringify(distortion), "Residual contract-distortion evidence is missing, schema-invalid, or inconsistent across static and quantization evidence.", ["/evidence/static_analysis/residual_contract_distortion", "/evidence/quantization/residual_contract_distortion"]);
    const distortionRows = distortion?.residual_adds || [];
    const distortionScenarios = distortionRows.flatMap((row) => row.scenarios || []);
    const distortionLatticeByOp = new Map((staticAnalysis?.quantization_lattice?.residual_adds || []).map((row) => [Number(row.op_index), row]));
    const distortionScenariosConserve = distortionScenarios.every((scenario) => Number(scenario.enumerated_pair_count || 0) === Number(scenario.same_represented_value_pair_count || 0) + Number(scenario.changed_represented_value_pair_count || 0)
      && Number(scenario.enumerated_pair_count || 0) === Number(scenario.ideal_error_improved_pair_count || 0) + Number(scenario.ideal_error_worsened_pair_count || 0) + Number(scenario.ideal_error_equal_within_tolerance_pair_count || 0)
      && Number(scenario.current_clamped_pair_count || 0) === Number(scenario.rescued_current_clamp_pair_count || 0) + Number(scenario.persistent_clamp_pair_count || 0)
      && Number(scenario.candidate_clamped_pair_count || 0) === Number(scenario.introduced_clamp_pair_count || 0) + Number(scenario.persistent_clamp_pair_count || 0)
      && (scenario.absolute_delta_histogram_counts || []).reduce((total, count) => total + Number(count || 0), 0) === Number(scenario.enumerated_pair_count || 0)
      && (scenario.tile_pair_counts || []).length === 256
      && (scenario.tile_pair_counts || []).every((count, index) => Number(count || 0) === Number(scenario.tile_ideal_error_improved_pair_counts?.[index] || 0) + Number(scenario.tile_ideal_error_worsened_pair_counts?.[index] || 0) + Number(scenario.tile_ideal_error_equal_pair_counts?.[index] || 0))
      && /^[a-f0-9]{64}$/.test(scenario.pair_ledger_sha256 || ""));
    const distortionContractsBindLattice = distortionRows.every((row) => {
      if (row.assessment_status !== "assessed") return !(row.scenarios || []).length;
      const latticeRow = distortionLatticeByOp.get(Number(row.op_index));
      return latticeRow && (row.scenarios || []).every((scenario) => {
        const candidate = candidateFor(latticeRow, scenario.design);
        return candidate && Number(candidate.output_scale) === Number(scenario.candidate_output_scale)
          && Number(candidate.output_zero_point) === Number(scenario.candidate_output_zero_point);
      });
    });
    check("CF-DISTORTION-002", distortionScenarios.length === Number(distortion?.scenario_count || 0)
      && distortionScenariosConserve
      && distortionContractsBindLattice
      && distortionScenarios.reduce((total, scenario) => total + Number(scenario.enumerated_pair_count || 0), 0) === Number(distortion?.total_enumerated_pair_count || 0)
      && distortionScenarios.reduce((total, scenario) => total + Number(scenario.rescued_current_clamp_pair_count || 0), 0) === Number(distortion?.rescued_current_clamp_pair_instance_count || 0)
      && distortionScenarios.reduce((total, scenario) => total + Number(scenario.ideal_error_improved_pair_count || 0), 0) === Number(distortion?.ideal_error_improved_pair_count || 0)
      && distortionScenarios.reduce((total, scenario) => total + Number(scenario.ideal_error_worsened_pair_count || 0), 0) === Number(distortion?.ideal_error_worsened_pair_count || 0), "Residual contract-distortion pair counts, clamp states, error classes, tiles, lattice bindings, or ledger identities are internally inconsistent.", ["/evidence/static_analysis/residual_contract_distortion", "/evidence/static_analysis/quantization_lattice"]);
    const distortionReport = String(engineeringReport || "");
    check("CF-DISTORTION-003", distortionReport.includes("## Residual Contract Distortion Atlas (DERIVED EXHAUSTIVE COUNTERFACTUAL DOMAIN)")
      && distortionReport.includes(ANALYZER_METADATA.schemas.residualContractDistortion)
      && distortionReport.includes(formatIntegerForConformance(distortion?.total_enumerated_pair_count))
      && distortionReport.includes("not an observed activation distribution")
      && distortionReport.includes("complete re-export"), "Engineering report does not preserve the distortion schema, exact pair count, or interpretation boundary.", ["/evidence/static_analysis/residual_contract_distortion", "/engineering_report.md"]);
    const distortionFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-QNT-0110");
    const distortionMlBomValid = compactMlBomEvidence || (stepMlBomValue("deepbom:model:residualContractDistortionSchema") === String(distortion?.schema)
      && stepMlBomValue("deepbom:model:residualContractDistortionPairs") === String(distortion?.total_enumerated_pair_count)
      && stepMlBomValue("deepbom:model:residualContractDistortionRescuedClamps") === String(distortion?.rescued_current_clamp_pair_instance_count));
    check("CF-DISTORTION-004", distortionMlBomValid
      && (!distortion?.assessed_add_count || (distortionFinding?.evidence_class === "DERIVED"
        && (distortionFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/residual_contract_distortion"))), "ML-BOM or authoritative finding does not preserve residual contract-distortion evidence.", ["/evidence/static_analysis/residual_contract_distortion", "/evidence/mlbom_cyclonedx", "/evidence/findings_register/findings"]);
    check("CF-IOC-001", (contracts.io_dequantization?.inputs || []).every((item) => item.tensor_numerical_contract_status === "known_from_artifact_quantization_metadata"
      && item.source_data_to_tensor_preprocessing_status === "not_embedded_in_artifact"), "Input tensor numerical range and source-data preprocessing status must remain separate.", ["/evidence/quantization/quantization_contract_checks/io_dequantization/inputs"]);
    const tensorsByIndex = new Map((staticAnalysis?.tensors || []).map((tensor) => [tensor.index, tensor]));
    const sourceCandidateOps = (staticAnalysis?.ops || []).filter((op) => String(op.xnnpack_kernel_evidence_class || "").startsWith("SOURCE_ENUMERATED_CANDIDATE"));
    const sourceNoMatchOps = (staticAnalysis?.ops || []).filter((op) => op.xnnpack_kernel_evidence_class === "SOURCE_ENUMERATED_NO_MATCH");
    const selectorEligibleOps = (staticAnalysis?.ops || []).filter((op) => ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"].includes(op.name));
    const selectorAssessment = String(staticAnalysis?.xnnpack_selector_assessment_status || "not_reported");
    check("CF-KERNEL-000", selectorAssessment === "complete"
      ? staticAnalysis?.xnnpack_selector_evidence_schema === ANALYZER_METADATA.schemas.xnnpackSelectorEvidence
        && staticAnalysis?.xnnpack_selector_evidence_access === "research"
        && sourceCandidateOps.length + sourceNoMatchOps.length === selectorEligibleOps.length
      : ["not_loaded", "not_available_for_profile"].includes(selectorAssessment)
        && sourceCandidateOps.length === 0
        && sourceNoMatchOps.length === 0
        && (staticAnalysis?.ops || []).every((op) => !(op.xnnpack_kernel_candidates || []).length), "Selector assessment status, access tier, schema, and per-op source evidence must agree.", ["/evidence/static_analysis/xnnpack_selector_assessment_status", "/evidence/static_analysis/ops"]);
    const alternateDelegates = staticAnalysis?.tflite_delegate_compatibility_evidence || null;
    const alternateProfiles = alternateDelegates?.profiles || [];
    const alternateRequirements = alternateDelegates?.build_requirements || [];
    check("CF-TFL-DELEGATE-001", alternateDelegates
      ? alternateDelegates.schema === ANALYZER_METADATA.schemas.tfliteDelegateSourceRulepack
        && alternateDelegates.assessment_status === "assessed_source_candidates_with_unresolved_runtime_predicates"
        && alternateDelegates.evidence_class === "SOURCE_PINNED/DERIVED_PARTIAL"
        && /^[a-f0-9]{64}$/.test(alternateDelegates.rulepack_sha256 || "")
        && /^[a-f0-9]{40}$/.test(alternateDelegates.tensorflow_source_commit || "")
        && Number(alternateDelegates.graph_op_count) === (staticAnalysis?.ops || []).length
        && alternateProfiles.length === 2
        && alternateProfiles.map((profile) => profile.id).join("|") === "tflite_gpu|tflite_nnapi"
      : String(engineeringReport || "").includes("## TFLite GPU and NNAPI Source Compatibility (NOT_ASSESSED)"), "TFLite GPU/NNAPI source compatibility must be schema-bound or explicitly not assessed.", ["/evidence/static_analysis/tflite_delegate_compatibility_evidence", "/engineering_report.md"]);
    check("CF-TFL-DELEGATE-002", !alternateDelegates || alternateProfiles.every((profile) => {
      const rows = profile.rows || [];
      const candidates = rows.filter((row) => row.artifact_precheck_status === "source_candidate_partial");
      const exclusions = rows.filter((row) => row.artifact_precheck_status === "definite_exclusion");
      return rows.length === (staticAnalysis?.ops || []).length
        && profile.op_count_conservation_status === "complete"
        && candidates.length === Number(profile.source_candidate_after_artifact_precheck_count)
        && exclusions.length === Number(profile.definite_exclusion_count)
        && candidates.length + exclusions.length === rows.length
        && candidates.every((row) => !(row.definite_exclusion_reasons || []).length && (row.unresolved_predicates || []).length)
        && exclusions.every((row) => (row.definite_exclusion_reasons || []).length && !(row.unresolved_predicates || []).length)
        && rows.every((row, index) => Number(row.op_index) === Number(staticAnalysis.ops[index]?.index ?? index)
          && row.op_name === staticAnalysis.ops[index]?.name
          && Number(row.op_version) === Number(staticAnalysis.ops[index]?.version ?? 1));
    }), "TFLite GPU/NNAPI source candidates must conserve and bind every graph op without converting unresolved predicates into support claims.", ["/evidence/static_analysis/tflite_delegate_compatibility_evidence/profiles", "/evidence/static_analysis/ops"]);
    check("CF-TFL-DELEGATE-003", !alternateDelegates || ((alternateDelegates.source_files || []).length === 5
      && (alternateDelegates.source_files || []).every((source) => /^[a-f0-9]{64}$/.test(source.sha256 || "")
        && String(source.source_ref || "").includes(alternateDelegates.tensorflow_source_commit)
        && String(engineeringReport || "").includes(source.source_ref)
        && String(engineeringReport || "").includes(source.sha256))
      && alternateRequirements.length === 4
      && alternateRequirements.every((requirement) => requirement.binding_status === (Number(requirement.affected_source_candidate_op_count || 0) > 0
        ? "pending_runtime_build_evidence" : "not_applicable_no_affected_source_candidates"))
      && String(engineeringReport || "").includes(alternateDelegates.interpretation_boundary)), "Engineering report must preserve every TFLite delegate source digest, pending build requirement, and interpretation boundary.", ["/evidence/static_analysis/tflite_delegate_compatibility_evidence", "/engineering_report.md"]);
    check("CF-KERNEL-001", sourceCandidateOps.every((op) => op.xnnpack_kernel_candidate
      && op.xnnpack_kernel_source?.includes("google/XNNPACK@")
      && op.xnnpack_build_requirement
      && Array.isArray(op.xnnpack_kernel_candidates)
      && op.xnnpack_kernel_candidates.length > 0
      && op.xnnpack_kernel_candidates.every((candidate) => candidate.family
        && Number(candidate.tile_nr || candidate.channel_tile || 0) > 0
        && candidate.architecture_condition
        && candidate.compile_condition
        && candidate.runtime_condition
        && candidate.source_ref?.includes("google/XNNPACK@"))), "Source-enumerated kernel candidate sets must include candidate tiles, selector conditions, pinned source locations, and build requirements.", ["/evidence/static_analysis/ops"]);
    check("CF-KERNEL-002", sourceCandidateOps.every((op) => {
      const inputDtype = String(tensorsByIndex.get(op.inputs?.[0])?.dtype || "").toUpperCase();
      return inputDtype !== "UINT8" || (op.xnnpack_kernel_candidates || []).every((candidate) => !String(candidate.family).includes("NEON DOT"));
    }), "Legacy UINT8/QU8 kernels must not be classified as signed NEON DOT candidates.", ["/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"]);
    check("CF-KERNEL-003", !sourceCandidateOps.length || (String(engineeringReport || "").includes("## Pinned XNNPACK Kernel Candidates")
      && sourceCandidateOps.every((op) => (op.xnnpack_kernel_candidates || []).every((candidate) => String(engineeringReport || "").includes(candidate.source_ref)))), "Engineering report must render every pinned kernel-candidate source represented by structured evidence.", ["/evidence/static_analysis/ops", "/engineering_report.md"]);
    check("CF-KERNEL-004", sourceCandidateOps.every((op) => {
      const expected = [...new Set((op.xnnpack_kernel_candidates || []).map((candidate) => Number(candidate.tile_nr || candidate.channel_tile || 0)).filter((value) => value > 0))].sort((a, b) => a - b);
      const actual = [...(op.xnnpack_kernel_alignment_multiples || [])].map(Number).sort((a, b) => a - b);
      return JSON.stringify(actual) === JSON.stringify(expected)
        && (op.xnnpack_kernel_candidates || []).every((candidate) => {
          const expectedSha = String(candidate.source_ref || "").includes("/gemm-config.c")
            ? ANALYZER_METADATA.rulepackProvenance.xnnpackGemmConfigSha256
            : ANALYZER_METADATA.rulepackProvenance.xnnpackDwconvConfigSha256;
          return /^[a-f0-9]{64}$/.test(candidate.source_file_sha256 || "") && candidate.source_file_sha256 === expectedSha;
        });
    }), "Candidate-set alignment multiples and pinned source-file SHA-256 values must be complete and internally consistent.", ["/evidence/static_analysis/ops"]);
    check("CF-KERNEL-005", sourceNoMatchOps.every((op) => Array.isArray(op.xnnpack_kernel_candidates) && op.xnnpack_kernel_candidates.length === 0 && String(op.xnnpack_kernel_selector_status || "").includes("no matching configuration")), "Source-enumerated no-match rows must not contain fabricated candidates and must disclose their limited inference boundary.", ["/evidence/static_analysis/ops"]);
    const selectorRows = [...sourceCandidateOps, ...sourceNoMatchOps].sort((left, right) => Number(left.index) - Number(right.index));
    const candidateUnresolved = ["runtime_architecture_identity", "compile_configuration", "lowering_shape", "runtime_dispatch"];
    const noMatchUnresolved = ["unenumerated_lowering_paths"];
    check("CF-KERNEL-006", selectorRows.every((op) => {
      const activation = tensorsByIndex.get(op.inputs?.[0]);
      const weights = tensorsByIndex.get(op.inputs?.[1]);
      const shape = Array.isArray(weights?.shape) ? weights.shape : [];
      const kernelProduct = Number(shape[1]) * Number(shape[2]);
      const kernelArea = shape.length === 4 && Number.isSafeInteger(Number(shape[1])) && Number(shape[1]) > 0
        && Number.isSafeInteger(Number(shape[2])) && Number(shape[2]) > 0 && Number.isSafeInteger(kernelProduct) ? kernelProduct : 0;
      const channels = Number.isSafeInteger(Number(op.output_channels)) && Number(op.output_channels) > 0 ? Number(op.output_channels) : 0;
      const facts = op.selector_artifact_facts || {};
      const kernelRelevant = ["CONV_2D", "DEPTHWISE_CONV_2D"].includes(op.name);
      const expectedFacts = String(facts.activation_dtype || "") === String(activation?.dtype || "")
        && Number(facts.kernel_area) === kernelArea
        && facts.kernel_area_status === (kernelRelevant ? (kernelArea > 0 ? "assessed" : "not_assessed") : "not_applicable")
        && facts.per_channel_weights === (Number(weights?.quant_scales || 0) > 1)
        && Number(facts.output_channels) === channels
        && facts.output_channels_status === (channels > 0 ? "assessed" : "not_assessed");
      const expectedUnresolved = (op.xnnpack_kernel_candidates || []).length ? candidateUnresolved : noMatchUnresolved;
      const expectedReason = (op.xnnpack_kernel_candidates || []).length ? "NOT_APPLICABLE_CANDIDATES_REMAIN"
        : !facts.activation_dtype ? "ACTIVATION_DTYPE_UNAVAILABLE"
          : kernelRelevant && kernelArea === 0 ? "KERNEL_AREA_UNAVAILABLE"
            : "NO_ENUMERATED_CONFIGURATION_AFTER_ARTIFACT_SELECTORS";
      return expectedFacts
        && JSON.stringify(op.unresolved_selector_dimensions || []) === JSON.stringify(expectedUnresolved)
        && op.no_match_reason_code === expectedReason
        && (op.xnnpack_kernel_candidates || []).every((candidate) => {
          const alignment = Math.max(Number(candidate.tile_nr || 0), Number(candidate.channel_tile || 0));
          const padded = channels > 0 ? Math.ceil(channels / alignment) * alignment : 0;
          const inactive = padded > 0 ? padded - channels : 0;
          const ratio = padded > 0 ? inactive / padded : 0;
          return Number(candidate.alignment_multiple) === alignment
            && candidate.tail_projection_status === (channels > 0 ? "assessed" : "not_assessed_output_channels_unavailable")
            && Number(candidate.padded_output_channels) === padded
            && Number(candidate.inactive_output_channels) === inactive
            && Math.abs(Number(candidate.inactive_lane_ratio) - ratio) <= 1e-9;
        });
    }), "Selector artifact facts, unresolved dimensions, reason codes, or candidate-specific tail arithmetic are inconsistent with graph evidence.", ["/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"]);
    const selectorProvenance = staticAnalysis?.xnnpack_selector_evidence_provenance || {};
    const tailRows = sourceCandidateOps.filter((op) => op.selector_artifact_facts?.output_channels_status === "assessed");
    const worstTail = tailRows.length ? Math.max(...tailRows.map((op) => Number(op.channel_tail_overhead_percent_max || 0))) : 0;
    const worstTailIndices = tailRows.filter((op) => Math.abs(Number(op.channel_tail_overhead_percent_max || 0) - worstTail) <= 1e-9).map((op) => Number(op.index));
    check("CF-KERNEL-007", selectorAssessment !== "complete" || (
      Number(selectorProvenance.assessed_op_count) === selectorRows.length
      && Number(selectorProvenance.candidate_op_count) === sourceCandidateOps.length
      && Number(selectorProvenance.candidate_configuration_count) === sourceCandidateOps.reduce((sum, op) => sum + op.xnnpack_kernel_candidates.length, 0)
      && Number(selectorProvenance.unique_candidate_op_count) === sourceCandidateOps.filter((op) => op.xnnpack_kernel_candidates.length === 1).length
      && Number(selectorProvenance.ambiguous_candidate_op_count) === sourceCandidateOps.filter((op) => op.xnnpack_kernel_candidates.length > 1).length
      && Number(selectorProvenance.no_match_op_count) === sourceNoMatchOps.length
      && Number(selectorProvenance.tail_assessed_op_count) === tailRows.length
      && Math.abs(Number(selectorProvenance.worst_case_tail_ratio) - worstTail) <= 1e-9
      && JSON.stringify(selectorProvenance.worst_case_tail_op_indices || []) === JSON.stringify(worstTailIndices)
      && Number(selectorProvenance.unresolved_selector_op_count) === selectorRows.filter((op) => op.unresolved_selector_dimensions?.length).length
      && Number(selectorProvenance.unresolved_selector_dimension_count) === selectorRows.reduce((sum, op) => sum + (op.unresolved_selector_dimensions || []).length, 0)
    ), "Selector decision-ledger summary does not reproduce the verified per-op evidence.", ["/evidence/static_analysis/xnnpack_selector_evidence_provenance", "/evidence/static_analysis/ops"]);
    check("CF-KERNEL-008", selectorAssessment !== "complete" || (String(engineeringReport || "").includes("Decision-ledger metric")
      && selectorRows.every((op) => {
        const report = String(engineeringReport || "");
        if (!(op.xnnpack_kernel_candidates || []).length) return report.includes(op.no_match_reason_code) && (op.unresolved_selector_dimensions || []).every((item) => report.includes(item));
        return op.xnnpack_kernel_candidates.every((candidate) => report.includes(candidate.family)
          && report.includes(`padded ${candidate.padded_output_channels}`)
          && report.includes(`inactive ${candidate.inactive_output_channels}`))
          && (op.unresolved_selector_dimensions || []).every((item) => report.includes(item));
      })), "Engineering report does not render the complete selector decision ledger.", ["/evidence/static_analysis/ops", "/engineering_report.md"]);
    check("CF-DELEGATION-004", (staticAnalysis?.ops || []).filter((op) => op.xnnpack_supported).every((op) => op.xnnpack_build_requirement), "Every conditionally delegatable op must disclose its runtime build requirement.", ["/evidence/static_analysis/ops"]);
    const boundaryInventory = staticAnalysis?.predicted_partition_boundaries || {};
    const expectedBoundaryEdges = derivePredictedBoundaryEdges(staticAnalysis);
    const expectedBoundaryByKey = new Map(expectedBoundaryEdges.map((edge) => [boundaryEdgeKey(edge), edge]));
    const emittedBoundaryEdges = boundaryInventory.edges || [];
    check("CF-DELEGATION-005", boundaryInventory.schema === "deepbom.predicted_partition_boundary_edges.v1.1"
      && boundaryInventory.assignment_evidence_class === "PREDICTED"
      && boundaryInventory.payload_evidence_class === "DERIVED"
      && boundaryInventory.edge_count === emittedBoundaryEdges.length
      && boundaryInventory.edge_count === expectedBoundaryEdges.length, "Predicted partition-boundary inventory schema or edge count does not match graph connectivity.", ["/evidence/static_analysis/predicted_partition_boundaries"]);
    check("CF-DELEGATION-006", emittedBoundaryEdges.every((edge) => {
      const expected = expectedBoundaryByKey.get(boundaryEdgeKey(edge));
      return Boolean(expected)
        && edge.producer_domain === expected.producer_domain
        && edge.consumer_domain === expected.consumer_domain
        && edge.direction === expected.direction
        && edge.tensor_dtype === expected.tensor_dtype
        && sameArray(edge.tensor_shape, expected.tensor_shape)
        && edge.payload_bytes === expected.payload_bytes
        && edge.payload_status === expected.payload_status
        && edge.payload_binding === expected.payload_binding
        && edge.materialization_status === "NOT_ASSESSABLE_FROM_STATIC_ARTIFACT";
    }), "A predicted partition-boundary edge does not match its tensor, producer/consumer domains, or deterministic payload.", ["/evidence/static_analysis/predicted_partition_boundaries/edges", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"]);
    const expectedEdgeBytes = expectedBoundaryEdges.filter((edge) => edge.payload_bytes != null).reduce((sum, edge) => sum + edge.payload_bytes, 0);
    const expectedUnique = new Map(expectedBoundaryEdges.map((edge) => [edge.tensor_index, edge.payload_bytes]));
    const expectedUniqueBytes = [...expectedUnique.values()].filter((value) => value != null).reduce((sum, value) => sum + value, 0);
    const completePayload = expectedBoundaryEdges.every((edge) => edge.payload_bytes != null);
    const expectedBindings = new Set(expectedBoundaryEdges.filter((edge) => edge.payload_bytes != null).map((edge) => edge.payload_binding));
    const expectedInventoryBinding = !completePayload ? "partial" : expectedBindings.size === 0 ? "none" : expectedBindings.size === 1 ? [...expectedBindings][0] : "mixed";
    check("CF-DELEGATION-007", boundaryInventory.assessed_edge_payload_bytes === expectedEdgeBytes
      && boundaryInventory.summed_edge_payload_bytes === (completePayload ? expectedEdgeBytes : null)
      && boundaryInventory.payload_binding === expectedInventoryBinding
      && boundaryInventory.unique_tensor_count === expectedUnique.size
      && boundaryInventory.assessed_unique_tensor_payload_bytes === expectedUniqueBytes
      && boundaryInventory.unique_tensor_payload_bytes === ([...expectedUnique.values()].every((value) => value != null) ? expectedUniqueBytes : null), "Predicted partition-boundary aggregate bytes do not match the per-edge inventory.", ["/evidence/static_analysis/predicted_partition_boundaries"]);
    const boundaryReport = String(engineeringReport || "");
    const displayedBoundaryEdges = emittedBoundaryEdges.slice(0, 24);
    check("CF-DELEGATION-008", boundaryReport.includes("### Predicted Partition Boundary Edges (PREDICTED assignment / DERIVED payload)")
      && displayedBoundaryEdges.every((edge) => boundaryReport.includes(`T${edge.tensor_index}`) && boundaryReport.includes(`${edge.producer_domain} -> ${edge.consumer_domain}`))
      && (emittedBoundaryEdges.length <= displayedBoundaryEdges.length
        || boundaryReport.includes(`${formatIntegerForConformance(emittedBoundaryEdges.length - displayedBoundaryEdges.length)} additional edge(s) remain in structured evidence`)),
    "Engineering report does not render the displayed predicted partition-boundary edges or disclose structured-evidence truncation.", ["/evidence/static_analysis/predicted_partition_boundaries", "/engineering_report.md"]);

    const repair = staticAnalysis?.delegation_repair || null;
    const repairOps = staticAnalysis?.ops || [];
    const repairToggles = repair?.toggles || [];
    const segmentCounts = repairOps.reduce((state, op) => {
      const delegated = Number(op.xnnpack_chain_id) >= 0;
      if (state.previous !== delegated) {
        if (delegated) state.delegate += 1;
        else state.cpu += 1;
      }
      state.previous = delegated;
      return state;
    }, { previous: null, delegate: 0, cpu: 0 });
    check("CF-REPAIR-001", !repair || (repair.schema === ANALYZER_METADATA.schemas.delegationRepair
      && repair.artifact_sha256 === staticAnalysis?.model_sha256
      && repair.target_id === staticAnalysis?.target_profile?.id
      && repair.target_profile_sha256 === staticAnalysis?.target_profile?.profile_sha256
      && repair.operator_count === repairOps.length
      && repairToggles.length === repairOps.length), "Delegation-repair schema, artifact/target binding, or complete op coverage is invalid.", ["/evidence/static_analysis/delegation_repair"]);
    check("CF-REPAIR-002", !repair || (repairToggles.every((row, index) => row.op_index === repairOps[index]?.index
      && row.op_name === repairOps[index]?.name)
      && new Set(repairToggles.map((row) => row.op_index)).size === repairOps.length
      && repair.baseline?.delegate_segment_count === segmentCounts.delegate
      && repair.baseline?.cpu_segment_count === segmentCounts.cpu
      && repair.baseline?.boundary_edge_count === boundaryInventory.edge_count
      && repair.baseline?.assessed_edge_payload_bytes === boundaryInventory.assessed_edge_payload_bytes
      && repair.baseline?.summed_edge_payload_bytes === boundaryInventory.summed_edge_payload_bytes), "Delegation-repair baseline segments, op identities, or boundary totals do not match the canonical static analysis.", ["/evidence/static_analysis/delegation_repair", "/evidence/static_analysis/predicted_partition_boundaries"]);
    const repairRows = repairToggles.filter((row) => row.repair_opportunity);
    const fragilityRows = repairToggles.filter((row) => row.fragmentation_risk);
    const repairIslands = repair?.cpu_islands || [];
    check("CF-REPAIR-003", !repair || (repair.repair_opportunity_count === repairRows.length
      && repair.fragmentation_risk_count === fragilityRows.length
      && repair.repair_ranking_op_indices?.length === repairRows.length
      && repair.fragility_ranking_op_indices?.length === fragilityRows.length
      && repair.cpu_island_count === repairIslands.length
      && repair.full_segment_repair_count === repairIslands.filter((island) => island.full_segment_repair).length
      && repair.group_only_repair_count === repairIslands.filter((island) => island.group_only_repair).length
      && Array.isArray(repair.export_interventions)
      && Array.isArray(repair.runtime_build_risks)
      && Array.isArray(repair.singleton_delegate_segments)
      && repair.cpu_island_ranking_indices?.length === repairIslands.length), "Delegation-repair summary counts and ranking coverage are inconsistent with toggle or CPU-island rows.", ["/evidence/static_analysis/delegation_repair"]);
    check("CF-REPAIR-004", repair
      ? boundaryReport.includes("## Delegation Repair Lab (PREDICTED/DERIVED COUNTERFACTUAL)")
        && boundaryReport.includes("### Predicted CPU-Island Assignment Portfolio")
        && boundaryReport.includes("### Actionable Export Interventions")
        && boundaryReport.includes("### Runtime Build Configuration Risks")
        && (repair.repair_ranking_op_indices || []).slice(0, 16).every((index) => {
          const row = repairToggles.find((item) => item.op_index === index);
          return row && boundaryReport.includes(`#${String(row.op_index).padStart(3, "0")} ${row.op_name}`);
        })
        && (repair.cpu_island_ranking_indices || []).slice(0, 16).every((index) => {
          const island = repairIslands.find((item) => item.island_index === index);
          return island && boundaryReport.includes(`island ${island.island_index}: #${String(island.first_op_index).padStart(3, "0")}-#${String(island.last_op_index).padStart(3, "0")}`);
        })
      : boundaryReport.includes("## Delegation Repair Lab (NOT_ASSESSED)"), "Engineering report does not preserve the delegation-repair assessment state, ranked repair rows, and complete CPU-island portfolio.", ["/evidence/static_analysis/delegation_repair", "/engineering_report.md"]);
    const repairGraph = deriveDelegationRepairGraph(staticAnalysis);
    const repairBaseline = deriveDelegationState(repairGraph, repairGraph.baselineAssignments);
    const expectedCpuIslandRanges = deriveCpuIslandRanges(repairGraph.baselineAssignments);
    check("CF-REPAIR-005", !repair || (expectedCpuIslandRanges.length === repairIslands.length
      && repairIslands.every((island, position) => {
        const [start, end] = expectedCpuIslandRanges[position] || [];
        const members = repairOps.slice(start, end + 1);
        const assignments = [...repairGraph.baselineAssignments];
        assignments.fill(true, start, end + 1);
        const counterfactual = deriveDelegationState(repairGraph, assignments);
        const changes = deriveDelegationChanges(repairBaseline.boundaries, counterfactual.boundaries);
        const incident = [...repairBaseline.boundaries.values()].filter((edge) => (edge.producerPosition >= start && edge.producerPosition <= end) || (edge.consumerPosition >= start && edge.consumerPosition <= end));
        const incidentPayload = incident.every((edge) => edge.payloadBytes != null) ? incident.reduce((sum, edge) => sum + edge.payloadBytes, 0) : null;
        return island.island_index === position + 1
          && island.execution_position_start === start
          && island.execution_position_end === end
          && island.first_op_index === members[0]?.index
          && island.last_op_index === members.at(-1)?.index
          && JSON.stringify(island.op_indices || []) === JSON.stringify(members.map((op) => op.index))
          && JSON.stringify(island.op_names || []) === JSON.stringify(members.map((op) => op.name))
          && island.baseline_incident_boundary_edge_count === incident.length
          && (island.baseline_incident_boundary_payload_bytes ?? null) === incidentPayload
          && sameDelegationSummary(island.counterfactual, counterfactual.summary)
          && sameDelegationChanges(island.edge_changes || [], changes);
      })), "Delegation-repair CPU-island membership, incident payload, counterfactual summary, or changed-edge ledger differs from an independent graph reconstruction.", ["/evidence/static_analysis/delegation_repair/cpu_islands", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"]);

    const arenaPlan = staticAnalysis?.tensor_arena_plan || {};
    const expectedArena = deriveArenaProjection(staticAnalysis);
    const expectedArenaRef = ANALYZER_METADATA.rulepackProvenance.arenaPlannerRef;
    check("CF-ARENA-001", arenaPlan.schema === "deepbom.tflite_arena_plan_projection.v1"
      && arenaPlan.evidence_class === "DERIVED"
      && `tensorflow/tensorflow@${arenaPlan.source_commit}` === expectedArenaRef
      && Number(arenaPlan.tensor_alignment_bytes) === Number(ANALYZER_METADATA.rulepackProvenance.arenaTensorAlignmentBytes), "Arena projection schema, evidence class, source commit, or alignment does not match pinned analyzer provenance.", ["/evidence/static_analysis/tensor_arena_plan", "/evidence/static_analysis/analyzer_metadata"]);
    check("CF-ARENA-002", arenaPlan.status === expectedArena.status
      && arenaPlan.non_persistent_arena_bytes === expectedArena.nonPersistentBytes
      && arenaPlan.persistent_arena_bytes === expectedArena.persistentBytes
      && arenaPlan.combined_arena_bytes === expectedArena.combinedBytes
      && Number(arenaPlan.root_allocation_count) === expectedArena.candidates.length
      && Number(arenaPlan.shared_tensor_count) === expectedArena.aliases.length
      && Number(arenaPlan.source_comparator_tie_group_count) === expectedArena.sourceComparatorTieGroups
      && Number(arenaPlan.source_comparator_tied_tensor_count) === expectedArena.sourceComparatorTiedTensors
      && Boolean(arenaPlan.source_comparator_fully_orders_projection) === (expectedArena.sourceComparatorTieGroups === 0), "Arena aggregate bytes, root count, alias count, source-comparator ties, or assessment status differs from the independent graph reconstruction.", ["/evidence/static_analysis/tensor_arena_plan", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"]);
    const emittedRoots = new Map((arenaPlan.allocations || []).filter((allocation) => allocation.allocation_status === "allocated").map((allocation) => [Number(allocation.tensor_index), allocation]));
    check("CF-ARENA-003", expectedArena.candidates.every((candidate) => {
      const emitted = emittedRoots.get(candidate.tensorIndex);
      return Boolean(emitted)
        && Number(emitted.size_bytes) === candidate.sizeBytes
        && Number(emitted.offset_bytes) === candidate.offsetBytes
        && Number(emitted.first_node) === candidate.firstNode
        && (emitted.last_node == null ? null : Number(emitted.last_node)) === candidate.lastNode
        && emitted.arena === candidate.arena;
    }) && expectedArena.aliases.every((alias) => (arenaPlan.aliases || []).some((emitted) => Number(emitted.tensor_index) === alias.tensorIndex
      && Number(emitted.shared_with_tensor_index) === alias.rootTensorIndex
      && Number(emitted.op_index) === alias.opIndex
      && Boolean(emitted.data_unmodified) === alias.dataUnmodified)), "Arena root offsets, lifetimes, sizes, or in-place aliases differ from the independent source-equivalent reconstruction.", ["/evidence/static_analysis/tensor_arena_plan/allocations", "/evidence/static_analysis/tensor_arena_plan/aliases"]);
    check("CF-ARENA-004", arenaAllocationsDoNotConflict([...emittedRoots.values()]), "Arena allocations with overlapping execution lifetimes overlap in memory.", ["/evidence/static_analysis/tensor_arena_plan/allocations"]);
    const arenaReport = String(engineeringReport || "");
    const byteNumber = new Intl.NumberFormat("en-US");
    const reportArenaRoots = (arenaPlan.allocations || [])
      .filter((allocation) => allocation.allocation_status === "allocated")
      .sort((left, right) => Number(right.size_bytes || 0) - Number(left.size_bytes || 0) || Number(left.tensor_index) - Number(right.tensor_index))
      .slice(0, 32);
    check("CF-ARENA-005", arenaReport.includes("## TFLite ArenaPlanner Declared-Shape Projection")
      && (arenaPlan.combined_arena_bytes == null || arenaReport.includes(`(${byteNumber.format(arenaPlan.combined_arena_bytes)} B)`))
      && reportArenaRoots.every((allocation) => arenaReport.includes(`T${allocation.tensor_index} \``)), "Engineering report does not render the structured ArenaPlanner projection and root allocation inventory.", ["/evidence/static_analysis/tensor_arena_plan", "/engineering_report.md"]);
    const mlBomProperties = [...(mlBomDocument?.metadata?.component?.properties || []), ...(mlBomDocument?.properties || [])];
    const mlBomValue = (name) => mlBomProperties.find((item) => item.name === name)?.value;
    check("CF-ARENA-006", compactMlBomEvidence || mlBomValue("deepbom:model:arenaProjectionStatus") === String(arenaPlan.status)
      && mlBomValue("deepbom:model:arenaCombinedBytes") === String(arenaPlan.combined_arena_bytes ?? "not_assessed")
      && mlBomValue("deepbom:model:arenaPlannerSourceCommit") === String(arenaPlan.source_commit), "ML-BOM arena projection properties do not match structured static evidence.", ["/evidence/static_analysis/tensor_arena_plan", "/evidence/mlbom_cyclonedx"]);
    const live = staticAnalysis?.tensor_liveness || {};
    const movement = staticAnalysis?.movement_analysis || {};
    const expectedLive = deriveDeclaredLiveness(staticAnalysis);
    check("CF-MEMORY-001", Boolean(live.assessed) === expectedLive.assessed
      && live.peak_bytes === expectedLive.peakBytes
      && live.peak_at_op === expectedLive.peakAtOp
      && live.peak_at_op_name === expectedLive.peakAtOpName
      && (live.assessed === false ? live.peak_bytes === null : Number.isFinite(Number(live.peak_bytes))), "Declared-shape liveness status, peak bytes, or peak operator differs from independent graph reconstruction.", ["/evidence/static_analysis/tensor_liveness", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"]);
    const expectedMovement = deriveMovementPayload(staticAnalysis);
    check("CF-MEMORY-002", movement.status === expectedMovement.status
      && movement.total_movement_bytes === expectedMovement.totalBytes
      && movement.xnn_break_movement_bytes === expectedMovement.breakBytes
      && Number(movement.assessed_movement_bytes) === expectedMovement.assessedBytes
      && Number(movement.assessed_xnn_break_movement_bytes) === expectedMovement.assessedBreakBytes
      && Number(movement.movement_op_count) === expectedMovement.opCount
      && Math.abs(Number(movement.movement_op_ratio || 0) - expectedMovement.opRatio) < 1e-15, "Movement payload bytes, predicted-break bytes, op count, or partial-assessment status differs from independent graph reconstruction.", ["/evidence/static_analysis/movement_analysis", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"]);
  }

  const runtime = securityPosture?.execution_integrity || {};
  check("CF-RUNTIME-001", runtime.runtime_execution_status !== "not_run" || (runtime.runtime_warnings === null && runtime.conclusion === "not_assessed"), "Runtime not_run state must use null warnings and a not_assessed conclusion.", ["/evidence/security_posture/execution_integrity"]);
  check("CF-RUNTIME-001A", runtime.runtime_execution_status !== "attempted_failed" || (runtime.runtime_warnings === null && runtime.conclusion === "not_assessed"), "A failed runtime attempt must not be promoted to measured execution or warning-absence evidence.", ["/evidence/security_posture/execution_integrity", "/evidence/runtime_results/benchmark_results"]);
  const runtimeAssignment = runtimeResults?.runtime_assignment || null;
  check("CF-RUNTIME-002", !runtimeAssignment || (runtimeAssignment.schema === ANALYZER_METADATA.schemas.runtimeAssignment
    && runtimeAssignment.evidence_class === "OBSERVED_RUNTIME"
    && runtimeAssignment.artifact_sha256 === staticAnalysis?.model_sha256
    && runtimeAssignment.target_profile_id === staticAnalysis?.target_profile?.id
    && runtimeAssignment.target_profile_sha256 === staticAnalysis?.target_profile?.profile_sha256
    && runtimeAssignment.source?.assignment_semantics === "original_graph_op_assignment"
    && runtimeAssignment.source?.partition_semantics === "partition_id_identifies_runtime_partition_when_present"
    && (runtimeAssignment.assignments || []).every((item) => staticAnalysis?.ops?.some((op) => op.index === item.op_index && op.name === item.op_name))), "Imported runtime assignment must be bound to the current artifact, target profile, and parsed op table.", ["/evidence/runtime_results/runtime_assignment", "/evidence/static_analysis"]);
  check("CF-RUNTIME-003", !runtimeAssignment || String(engineeringReport || "").includes("Imported runtime assignment | OBSERVED_RUNTIME"), "Engineering report must disclose imported runtime assignment evidence.", ["/evidence/runtime_results/runtime_assignment", "/engineering_report.md"]);
  if (runtimeAssignment) {
    const comparison = runtimeAssignment.comparison || {};
    const predictionApplicable = String(staticAnalysis?.format || "tflite").toLowerCase() === "tflite";
    const expectedOps = deriveRuntimeOpComparisons(staticAnalysis, runtimeAssignment, predictionApplicable);
    const expectedPlacement = summarizeRuntimePlacement(expectedOps, staticAnalysis?.ops?.length || 0);
    const expectedEdges = deriveRuntimeBoundaryComparisonEdges(staticAnalysis, runtimeAssignment, predictionApplicable);
    const expectedBoundary = summarizeRuntimeBoundaryComparison(expectedEdges);
    const expectedPredictedBoundaryPayload = summarizeRuntimeBoundaryPayload(expectedBoundary.predicted);
    const expectedBoundaryPayload = summarizeRuntimeBoundaryPayload(expectedBoundary.observed);
    const emittedOps = comparison.op_comparisons || [];
    const emittedEdges = comparison.boundary_comparisons || [];
    check("CF-RUNTIME-004", comparison.schema === ANALYZER_METADATA.schemas.runtimeAssignmentComparison
      && comparison.evidence_class === "DERIVED_FROM_OBSERVED_RUNTIME"
      && comparison.prediction_applicability === (predictionApplicable ? "tflite_xnnpack_static_prediction" : "not_applicable_for_onnx_execution_provider_assignment")
      && comparison.source_adapter_schema === (runtimeAssignment.source?.adapter?.schema || null)
      && comparison.artifact_sha256 === runtimeAssignment.artifact_sha256
      && comparison.target_profile_sha256 === runtimeAssignment.target_profile_sha256, "Runtime assignment comparison schema or artifact/target binding is invalid.", ["/evidence/runtime_results/runtime_assignment/comparison"]);
    check("CF-RUNTIME-005", emittedOps.length === expectedOps.length && expectedOps.every((expected, index) => {
      const actual = emittedOps[index];
      return actual?.op_index === expected.op_index
        && actual?.op_name === expected.op_name
        && actual?.predicted_delegated === expected.predicted_delegated
        && actual?.predicted_domain === expected.predicted_domain
        && actual?.observed_delegated === expected.observed_delegated
        && actual?.observed_provider === expected.observed_provider
        && actual?.matches_prediction === expected.matches_prediction
        && actual?.classification === expected.classification;
    }), "Predicted-vs-observed op assignment rows differ from independent reconstruction.", ["/evidence/runtime_results/runtime_assignment/comparison/op_comparisons", "/evidence/static_analysis/ops"]);
    check("CF-RUNTIME-006", comparison.placement_assessment?.prediction_applicability === (predictionApplicable ? "applicable" : "not_applicable")
      && comparison.placement_assessment?.observed_assignment_count === expectedPlacement.observed
      && nullableClose(comparison.placement_assessment?.observed_assignment_coverage_ratio, expectedPlacement.observedCoverageRatio)
      && comparison.placement_assessment?.assessed_op_count === expectedPlacement.assessed
      && comparison.placement_assessment?.unassessed_op_count === expectedPlacement.unassessed
      && comparison.placement_assessment?.match_count === expectedPlacement.matches
      && comparison.placement_assessment?.mismatch_count === expectedPlacement.mismatches
      && comparison.placement_assessment?.overpredicted_delegation_count === expectedPlacement.overpredicted
      && comparison.placement_assessment?.underpredicted_delegation_count === expectedPlacement.underpredicted
      && nullableClose(comparison.placement_assessment?.match_ratio, expectedPlacement.matchRatio), "Runtime placement aggregate counts or ratio differ from independent op rows.", ["/evidence/runtime_results/runtime_assignment/comparison/placement_assessment"]);
    check("CF-RUNTIME-007", emittedEdges.length === expectedEdges.length && expectedEdges.every((expected, index) => {
      const actual = emittedEdges[index];
      return runtimeBoundaryKey(actual) === runtimeBoundaryKey(expected)
        && actual?.predicted_boundary === expected.predicted_boundary
        && actual?.observed_boundary === expected.observed_boundary
        && actual?.predicted_producer_domain === expected.predicted_producer_domain
        && actual?.predicted_consumer_domain === expected.predicted_consumer_domain
        && actual?.classification === expected.classification
        && actual?.payload_bytes === expected.payload_bytes;
    }), "Runtime graph-edge boundary comparison differs from independent connectivity and assignment reconstruction.", ["/evidence/runtime_results/runtime_assignment/comparison/boundary_comparisons", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"]);
    check("CF-RUNTIME-008", comparison.boundary_comparison?.prediction_applicability === (predictionApplicable ? "applicable" : "not_applicable")
      && comparison.boundary_comparison?.observed_relation_edge_count === expectedBoundary.observedRelations
      && comparison.boundary_comparison?.assessed_edge_count === expectedBoundary.assessed
      && comparison.boundary_comparison?.unassessed_edge_count === expectedBoundary.unassessed
      && comparison.boundary_comparison?.match_count === expectedBoundary.matches
      && comparison.boundary_comparison?.mismatch_count === expectedBoundary.mismatches
      && comparison.predicted_boundary_inventory?.schema === ANALYZER_METADATA.schemas.predictedRuntimeBoundarySummary
      && comparison.predicted_boundary_inventory?.status === (predictionApplicable ? "assessed" : "not_applicable")
      && comparison.predicted_boundary_inventory?.assignment_evidence_class === (predictionApplicable ? "PREDICTED" : "NOT_APPLICABLE")
      && comparison.predicted_boundary_inventory?.edge_count === expectedBoundary.predicted.length
      && comparison.predicted_boundary_inventory?.assessed_edge_payload_bytes === expectedPredictedBoundaryPayload.assessedBytes
      && comparison.predicted_boundary_inventory?.summed_edge_payload_bytes === expectedPredictedBoundaryPayload.totalBytes
      && comparison.predicted_boundary_inventory?.unique_tensor_count === expectedPredictedBoundaryPayload.uniqueCount
      && comparison.predicted_boundary_inventory?.unique_tensor_payload_bytes === expectedPredictedBoundaryPayload.uniqueBytes
      && comparison.observed_boundary_inventory?.schema === ANALYZER_METADATA.schemas.observedPartitionBoundaryEdges
      && comparison.observed_boundary_inventory?.edge_count === expectedBoundary.observed.length
      && comparison.observed_boundary_inventory?.assessed_relation_edge_count === expectedBoundary.observedRelations
      && comparison.observed_boundary_inventory?.unassessed_relation_edge_count === expectedEdges.length - expectedBoundary.observedRelations
      && comparison.observed_boundary_inventory?.assessed_edge_payload_bytes === expectedBoundaryPayload.assessedBytes
      && comparison.observed_boundary_inventory?.summed_edge_payload_bytes === expectedBoundaryPayload.totalBytes
      && comparison.observed_boundary_inventory?.unique_tensor_count === expectedBoundaryPayload.uniqueCount
      && comparison.observed_boundary_inventory?.unique_tensor_payload_bytes === expectedBoundaryPayload.uniqueBytes
      && sameRuntimeBoundarySet(comparison.observed_boundary_inventory?.edges || [], expectedBoundary.observed), "Runtime boundary aggregates or observed edge inventory differ from independently reconstructed graph edges.", ["/evidence/runtime_results/runtime_assignment/comparison/boundary_comparison", "/evidence/runtime_results/runtime_assignment/comparison/observed_boundary_inventory"]);
    const runtimeHeading = predictionApplicable ? "### Predicted Vs Observed Runtime Assignment" : "### Observed Runtime Provider Assignment";
    check("CF-RUNTIME-009", String(engineeringReport || "").includes(runtimeHeading)
      && (comparison.mismatches || []).slice(0, 64).every((item) => String(engineeringReport || "").includes(`#${String(item.op_index).padStart(3, "0")} ${item.op_name}`)), "Engineering report does not render the runtime assignment comparison and mismatch rows.", ["/evidence/runtime_results/runtime_assignment/comparison", "/engineering_report.md"]);
    const additiveDuration = runtimeAssignment.source?.duration_semantics === "per_original_op_exclusive";
    const executionNodeDuration = runtimeAssignment.source?.duration_semantics === "per_execution_plan_node_exclusive";
    check("CF-RUNTIME-010", (additiveDuration || executionNodeDuration || (comparison.duration_comparison?.assessed_duration_us === null
      && comparison.duration_comparison?.total_duration_us === null
      && comparison.duration_comparison?.mismatch_duration_us === null))
      && (predictionApplicable || (comparison.duration_comparison?.mismatch_assessed_duration_us === null
        && comparison.duration_comparison?.mismatch_duration_us === null
        && comparison.mac_comparison?.mismatch_macs === null
        && comparison.mac_comparison?.mismatch_mac_ratio === null)), "Runtime durations were aggregated without additive semantics or prediction-mismatch metrics were emitted where static prediction is not applicable.", ["/evidence/runtime_results/runtime_assignment/source/duration_semantics", "/evidence/runtime_results/runtime_assignment/comparison/duration_comparison"]);
    const runtimeMismatchFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-DEL-0003");
    const hasRuntimeMismatch = Number(comparison.placement_assessment?.mismatch_count || 0) > 0 || Number(comparison.boundary_comparison?.mismatch_count || 0) > 0;
    check("CF-RUNTIME-011", !hasRuntimeMismatch || (runtimeMismatchFinding?.evidence_class === "DERIVED_FROM_OBSERVED_RUNTIME"
      && (runtimeMismatchFinding.evidence_json_pointers || []).includes("/evidence/runtime_results/runtime_assignment/comparison")), "Observed runtime assignment mismatch is missing from the authoritative findings register.", ["/evidence/runtime_results/runtime_assignment/comparison", "/evidence/findings_register/findings"]);
    const expectedPartitions = deriveRuntimePartitionSummary(staticAnalysis, runtimeAssignment, predictionApplicable);
    const emittedPartitions = comparison.observed_partitions || {};
    check("CF-RUNTIME-014", emittedPartitions.status === expectedPartitions.status
      && emittedPartitions.partition_count === expectedPartitions.partitionCount
      && emittedPartitions.explicit_partition_count === expectedPartitions.explicitCount
      && emittedPartitions.implicit_contiguous_partition_count === expectedPartitions.implicitCount
      && emittedPartitions.noncontiguous_partition_id_count === expectedPartitions.noncontiguousCount
      && emittedPartitions.provider_segment_count === expectedPartitions.providerSegmentCount
      && sameRuntimePartitionInventory(emittedPartitions.partitions || [], expectedPartitions.partitions), "Observed runtime partition counts or explicit partition-ID grouping differ from independent assignment-row reconstruction.", ["/evidence/runtime_results/runtime_assignment/comparison/observed_partitions", "/evidence/runtime_results/runtime_assignment/assignments"]);
    const selectorObservation = runtimeAssignment.selector_observation || {};
    const selectorContext = runtimeAssignment.selector_context || null;
    const loweringRows = (runtimeAssignment.assignments || []).filter((item) => item.lowering_id);
    const kernelRows = (runtimeAssignment.assignments || []).filter((item) => item.kernel);
    const closedRows = (runtimeAssignment.assignments || []).filter((item) => (item.resolved_selector_dimensions || []).length === 4);
    check("CF-RUNTIME-016", selectorObservation.schema === "deepbom.runtime_selector_observation.v1"
      && selectorObservation.context_bound === Boolean(selectorContext)
      && selectorObservation.lowering_observed_op_count === loweringRows.length
      && selectorObservation.microkernel_observed_op_count === kernelRows.length
      && selectorObservation.selector_ambiguity_closed_op_count === closedRows.length
      && (selectorContext ? runtimeAssignment.source?.kind === "deepbom_native_runtime_capture"
        && ["deepbom.native_runtime_collector.v1", "deepbom.native_runtime_collector.v1.1"].includes(runtimeAssignment.source?.collector?.schema)
        && (runtimeAssignment.source?.collector?.schema !== "deepbom.native_runtime_collector.v1.1"
          || ["lowering_ids", "microkernel_ids", "arena_allocations"].every((field) => typeof runtimeAssignment.source?.collector?.instrumentation?.[field] === "boolean"))
        && runtimeAssignment.source?.collector?.attestation_status === "not_attested"
        && selectorContext.build?.runtime_binary_sha256 === runtimeAssignment.runtime?.binary_sha256
        && /^[a-f0-9]{64}$/.test(runtimeAssignment.source?.import_file_sha256 || "")
        && kernelRows.every((item) => item.selector_evidence_class === "OBSERVED_MICROKERNEL"
          && item.kernel_id && item.kernel_source_ref
          && item.kernel_build_identifier_sha256 === selectorContext.build?.microkernel_build_identifier_sha256)
        : loweringRows.length === 0 && kernelRows.length === 0 && selectorObservation.status === "not_collected"), "Runtime selector observation counts, collector boundary, binary/build binding, or complete microkernel identities are inconsistent.", ["/evidence/runtime_results/runtime_assignment/selector_context", "/evidence/runtime_results/runtime_assignment/selector_observation", "/evidence/runtime_results/runtime_assignment/assignments"]);
    check("CF-RUNTIME-017", !selectorContext || (String(engineeringReport || "").includes("Native selector context")
      && String(engineeringReport || "").includes(`ambiguity closed for ${closedRows.length}/${runtimeAssignment.graph_op_count}`)
      && String(engineeringReport || "").includes("collector attestation not_attested")), "Engineering report does not preserve the native selector context and unattested collector boundary.", ["/evidence/runtime_results/runtime_assignment/selector_observation", "/engineering_report.md"]);
    const delegateBuildInventory = runtimeAssignment.tflite_delegate_build_inventory || null;
    check("CF-RUNTIME-022", !delegateBuildInventory || (String(staticAnalysis?.format || "").toLowerCase() === "tflite"
      && delegateBuildInventory.schema === "deepbom.tflite_delegate_build_inventory.v1"
      && delegateBuildInventory.artifact_sha256 === runtimeAssignment.artifact_sha256
      && delegateBuildInventory.runtime_binary_sha256 === runtimeAssignment.runtime.binary_sha256
      && delegateBuildInventory.build_manifest_sha256 === selectorContext?.build?.build_manifest_sha256
      && delegateBuildInventory.tensorflow_source_commit === selectorContext?.build?.tensorflow_source_commit
      && (!staticAnalysis?.tflite_delegate_compatibility_evidence
        || delegateBuildInventory.tensorflow_source_commit === staticAnalysis.tflite_delegate_compatibility_evidence.tensorflow_source_commit)
      && (delegateBuildInventory.build_options || []).map((option) => option.name).join("|") === "TFLITE_ENABLE_GPU|TFLITE_ENABLE_NNAPI"
      && (delegateBuildInventory.source_files || []).length === 2
      && (delegateBuildInventory.source_files || []).every((source) => /^[a-f0-9]{64}$/.test(source.sha256 || "")
        && String(engineeringReport || "").includes(source.source_ref)
        && String(engineeringReport || "").includes(source.sha256))
      && String(engineeringReport || "").includes("### TFLite GPU / NNAPI Selected-Build Inventory (DECLARED_HASH_BOUND/NOT_ASSIGNMENT)")
      && String(engineeringReport || "").includes(delegateBuildInventory.interpretation_boundary)), "TFLite delegate selected-build inventory must remain artifact/binary/build/source bound and report its non-assignment boundary.", ["/evidence/runtime_results/runtime_assignment/tflite_delegate_build_inventory", "/evidence/static_analysis/tflite_delegate_compatibility_evidence", "/engineering_report.md"]);
    const runtimeArena = reconstructRuntimeArenaEvidence(staticAnalysis, runtimeAssignment);
    check("CF-RUNTIME-018", runtimeArena.valid, runtimeArena.errors[0] || "Runtime arena snapshots and reconciliation match an independent reconstruction.", ["/evidence/runtime_results/runtime_assignment/runtime_memory", "/evidence/runtime_results/runtime_assignment/arena_reconciliation", "/evidence/static_analysis/tensor_arena_plan"]);
    if (runtimeArena.present) {
      const memory = runtimeAssignment.runtime_memory;
      const expectedArena = runtimeArena.reconciliation;
      check("CF-RUNTIME-019", String(engineeringReport || "").includes("### Static Projection Vs Observed TFLite Arena")
        && String(engineeringReport || "").includes(memory.allocation_ledger_sha256)
        && String(engineeringReport || "").includes(`tensorflow/tensorflow@${memory.tensorflow_source_commit}`)
        && String(engineeringReport || "").includes(`Prepare-time runtime temporaries | ${expectedArena.runtime_temporary_allocation_count} allocation(s)`), "Engineering report does not preserve the observed arena identity, reconciliation, and runtime-temporary summary.", ["/evidence/runtime_results/runtime_assignment/runtime_memory", "/evidence/runtime_results/runtime_assignment/arena_reconciliation", "/engineering_report.md"]);
      const runtimeMlBomProperties = mlBomDocument?.metadata?.component?.properties || [];
      const runtimeMlBomValue = (name) => runtimeMlBomProperties.find((item) => item.name === name)?.value;
      check("CF-RUNTIME-020", compactMlBomEvidence || (runtimeMlBomValue("deepbom:runtime:tfliteArenaMemorySchema") === String(memory.schema)
        && runtimeMlBomValue("deepbom:runtime:tfliteArenaTensorflowSourceCommit") === String(memory.tensorflow_source_commit)
        && runtimeMlBomValue("deepbom:runtime:tfliteArenaPeakCombinedBytes") === String(memory.peak_combined_arena_bytes)
        && runtimeMlBomValue("deepbom:runtime:tfliteArenaFinalCombinedBytes") === String(memory.final_combined_arena_bytes)
        && runtimeMlBomValue("deepbom:runtime:tfliteArenaAllocationLedgerSha256") === String(memory.allocation_ledger_sha256)
        && runtimeMlBomValue("deepbom:runtime:arenaReconciliationSchema") === String(expectedArena.schema)
        && runtimeMlBomValue("deepbom:runtime:arenaPeakDeltaBytes") === String(expectedArena.peak_delta_bytes ?? "not_assessed")
        && runtimeMlBomValue("deepbom:runtime:arenaRuntimeOnlyAllocations") === String(expectedArena.runtime_only_allocation_count)
        && runtimeMlBomValue("deepbom:runtime:arenaPrepareTemporaryIntervalBytes") === String(expectedArena.runtime_temporary_interval_bytes)), "ML-BOM runtime arena properties do not match independently reconstructed evidence.", ["/evidence/runtime_results/runtime_assignment/runtime_memory", "/evidence/runtime_results/runtime_assignment/arena_reconciliation", "/evidence/mlbom_cyclonedx"]);
      const arenaFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-MEM-0002");
      check("CF-RUNTIME-021", runtimeArena.hasDifference
        ? arenaFinding?.evidence_class === "DERIVED_FROM_OBSERVED_RUNTIME"
          && (arenaFinding.evidence_json_pointers || []).includes("/evidence/runtime_results/runtime_assignment/arena_reconciliation")
        : arenaFinding == null, "Runtime arena differences are missing from, or spuriously added to, the authoritative findings register.", ["/evidence/runtime_results/runtime_assignment/arena_reconciliation", "/evidence/findings_register/findings"]);
    }
    const adapter = runtimeAssignment.source?.adapter || null;
    const adapterRows = runtimeAssignment.assignments || [];
    const ortRuntimeAssessment = assessOrtRuntimeAdapter({ adapter, runtimeAssignment, staticAnalysis });
    const adapterCountsMatch = ortRuntimeAssessment.adapter_counts_match;
    const rawProfileBound = ortRuntimeAssessment.raw_profile_bound;
    const ortAdapter = ortRuntimeAssessment.is_ort_adapter;
    const tfliteAdapter = adapter?.schema?.startsWith("deepbom.tflite_runtime_info_adapter.v");
    const timedTfliteAdapter = adapter?.schema === "deepbom.tflite_runtime_info_adapter.v2";
    const ortValid = ortRuntimeAssessment.valid;
    const emittedOnnxShapeBinding = runtimeResults?.onnx_runtime_shape_binding || null;
    const expectedOnnxShapeBinding = String(staticAnalysis?.format || "").toLowerCase() === "onnx" && runtimeAssignment
      ? buildOnnxRuntimeShapeBinding(staticAnalysis, runtimeAssignment)
      : null;
    check("CF-RUNTIME-023", JSON.stringify(emittedOnnxShapeBinding) === JSON.stringify(expectedOnnxShapeBinding),
      "ONNX runtime internal shape/MAC closure does not reconstruct from the imported assignment and static tensor ledger.",
      ["/evidence/runtime_results/onnx_runtime_shape_binding", "/evidence/runtime_results/runtime_assignment", "/evidence/static_analysis/tensors", "/evidence/static_analysis/ops"]);
    check("CF-RUNTIME-024", !emittedOnnxShapeBinding || (String(engineeringReport || "").includes("### ONNX Runtime Internal Shape And Cost Binding")
      && String(engineeringReport || "").includes(`${emittedOnnxShapeBinding.runtime_closed_mac_op_count} op(s) newly closed`)
      && String(engineeringReport || "").includes(emittedOnnxShapeBinding.interpretation_boundary)),
    "Engineering report does not preserve runtime-bound ONNX internal shape/MAC closure and its physical-memory boundary.",
    ["/evidence/runtime_results/onnx_runtime_shape_binding", "/engineering_report.md"]);
    const emittedDataMovement = runtimeResults?.runtime_data_movement_evidence || null;
    const expectedDataMovement = buildRuntimeDataMovementEvidence(runtimeAssignment);
    check("CF-RUNTIME-025", JSON.stringify(emittedDataMovement) === JSON.stringify(expectedDataMovement),
      "Runtime copy-node data-movement evidence does not reconstruct from the imported native ORT capture.",
      ["/evidence/runtime_results/runtime_data_movement_evidence", "/evidence/runtime_results/runtime_assignment"]);
    check("CF-RUNTIME-026", !emittedDataMovement || (String(engineeringReport || "").includes("### Runtime Copy-Node Data Movement")
      && String(engineeringReport || "").includes(emittedDataMovement.physical_transfer_status)
      && String(engineeringReport || "").includes(emittedDataMovement.interpretation_boundary)),
    "Engineering report does not preserve the observed copy-node payload and physical-transfer claim boundary.",
    ["/evidence/runtime_results/runtime_data_movement_evidence", "/engineering_report.md"]);
    const tflitePartitions = Array.isArray(adapter?.partitions) ? adapter.partitions : [];
    const tfliteMappingsValid = adapterRows.every((item) => {
      const op = (staticAnalysis?.ops || []).find((candidate) => candidate.index === item.op_index && candidate.name === item.op_name);
      if (!op || item.kernel != null) return false;
      if (item.delegated === true) {
        if (item.duration_us != null || item.duration_sum_us != null || item.sample_count != null) return false;
        const partition = tflitePartitions.find((candidate) => candidate.partition_id === item.partition_id);
        return item.mapping_method === "runtime_info_original_node_id_and_symmetric_delegate_map"
          && item.partition_id === `subgraph:0/delegate_node:${item.runtime_node_index}`
          && partition?.delegate_node_id === item.runtime_node_index
          && partition?.delegate_name === item.provider
          && (partition?.replaced_op_ids || []).includes(item.op_index);
      }
      const timing = timedTfliteAdapter ? (adapter.timing_profile?.execution_nodes || []).find((candidate) => candidate.node_kind === "original_op" && candidate.runtime_node_index === item.runtime_node_index) : null;
      const durationValid = timing?.run_count == null
        ? item.duration_us == null && item.duration_sum_us == null && item.sample_count == null
        : timing
        ? nullableClose(item.duration_us, timing.mean_per_run_us) && nullableClose(item.duration_sum_us, timing.sum_us) && item.sample_count === timing.run_count
        : item.duration_us == null && item.duration_sum_us == null && item.sample_count == null;
      return durationValid && item.delegated === false
        && item.mapping_method === "runtime_info_original_node_id_execution_plan"
        && item.partition_id == null
        && item.provider === "TFLite non-delegated kernel"
        && item.runtime_node_index === item.op_index
        && item.runtime_node_name === item.op_name;
    });
    const delegatedRows = adapterRows.filter((item) => item.delegated === true);
    const nondelegatedRows = adapterRows.filter((item) => item.delegated === false);
    const timingRows = adapter?.timing_profile?.execution_nodes || [];
    const timingInternalRows = adapter?.timing_profile?.delegate_internal_events || [];
    const timingRunCounts = new Set(timingRows.map((item) => item.run_count));
    const timingRowsValid = timingRows.every((item) => item.formatter_times_called_integer_average === 1
      ? item.run_count === item.event_sample_count
        && item.run_count_derivation_status === "derived_primary_execution_node_one_event_per_run"
        && nullableClose(item.mean_per_run_us, item.sum_us / item.run_count)
      : item.run_count == null
        && item.run_count_derivation_status === "not_derivable_execution_node_calls_per_run_not_one"
        && item.mean_per_run_us == null);
    const timingComplete = timingRows.length === adapter?.execution_plan_node_count && timingRowsValid
      && timingRows.every((item) => Number.isSafeInteger(item.run_count) && item.run_count > 0) && timingRunCounts.size === 1;
    const independentlySummedTiming = timingRows.reduce((sum, item) => sum + Number(item.mean_per_run_us || 0), 0);
    const primaryDelegateTiming = timingInternalRows.filter((item) => item.node_kind === "primary_subgraph_delegate_profiled_event");
    const nestedDelegateTiming = timingInternalRows.filter((item) => item.node_kind === "delegate_internal_section_event");
    const independentPrimaryDelegateSubtotal = independentCommonRunSubtotal(primaryDelegateTiming);
    const independentNestedDelegateSubtotal = independentCommonRunSubtotal(nestedDelegateTiming);
    const independentDelegateProfiledSubtotal = independentCommonRunSubtotal(timingInternalRows);
    const internalRunBindingValid = timingInternalRows.every((item) => item.run_count == null
      ? item.run_count_derivation_status === "not_derivable_no_common_primary_execution_run_count" && item.mean_per_run_us == null
      : item.run_count === adapter.timing_profile?.common_run_count
        && item.run_count_derivation_status === "derived_from_common_primary_execution_run_count"
        && item.formatter_times_called_integer_average === Math.floor(item.event_sample_count / item.run_count)
        && nullableClose(item.mean_per_run_us, item.sum_us / item.run_count));
    const tfliteTimingValid = !timedTfliteAdapter || (adapter.timing_profile?.schema === "deepbom.tflite_benchmark_profile_adapter.v1"
      && /^[a-f0-9]{64}$/.test(adapter.timing_profile?.profile_sha256 || "")
      && adapter.timing_profile?.capture_id === runtimeAssignment.source?.capture_id
      && adapter.timing_profile?.capture_binding_semantics === "DECLARED_SAME_BENCHMARK_INVOCATION"
      && adapter.timing_profile?.proto_sha256 === "dd286650b64ae913cd9db25baa87a1527835fce4201ad407716b4af4f7dafc69"
      && adapter.timing_profile?.summarizer_sha256 === "de7010e97438b7e233e8ec80c8bb19fcbe435475d4f2cca5ccb8bbc03e4d0f91"
      && adapter.timing_profile?.formatter_sha256 === "cda02354a7a4c3f1c87263206fd13a64f3c3aae56270e0339a59e4aeae82491f"
      && adapter.timing_profile?.mapped_execution_node_count === timingRows.length
      && timingRowsValid
      && internalRunBindingValid
      && nullableClose(adapter.timing_profile?.execution_plan_coverage_ratio, timingRows.length / Math.max(1, adapter.execution_plan_node_count))
      && nullableClose(adapter.timing_profile?.execution_node_total_us, timingComplete ? independentlySummedTiming : null)
      && nullableClose(adapter.timing_profile?.primary_delegate_profiled_subtotal_us, independentPrimaryDelegateSubtotal)
      && nullableClose(adapter.timing_profile?.delegate_internal_section_subtotal_us, independentNestedDelegateSubtotal)
      && nullableClose(adapter.timing_profile?.delegate_internal_profiled_subtotal_us, independentDelegateProfiledSubtotal)
      && nullableClose(comparison.duration_comparison?.total_duration_us, timingComplete ? independentlySummedTiming : null)
      && adapterRows.filter((item) => item.delegated === true).every((item) => item.duration_us == null && item.duration_sum_us == null && item.sample_count == null));
    const tfliteValid = tfliteAdapter
      && runtimeAssignment.source?.kind === (timedTfliteAdapter ? "tflite_model_runtime_info_and_benchmark_profile_proto_adapter" : "tflite_model_runtime_info_proto_adapter")
      && rawProfileBound
      && adapter.source_commit === "tensorflow/tensorflow@87bbf65b8d23d3f06912b1b2183587e1884bc45c"
      && adapter.source_file === "tensorflow/lite/profiling/model_runtime_info.cc"
      && adapter.source_sha256 === "9a6edf838fe149c54efe0700bcdc2faf58dd5343f1370538e91bf5ed8a0e11b6"
      && adapter.proto_file === "tensorflow/lite/profiling/proto/model_runtime_info.proto"
      && adapter.proto_sha256 === "7829c3163339ce7dea01091a5154a06cb302f35573ad51d3af06a2a09b95a8fb"
      && adapter.delegation_metadata_file === "tensorflow/lite/optional_debug_tools.cc"
      && adapter.delegation_metadata_sha256 === "607010c8f7aba721bd5d96f45cb08c55226a800e55946da708368d09d4545260"
      && adapter.export_driver_file === "tensorflow/lite/tools/benchmark/benchmark_tflite_model.cc"
      && adapter.export_driver_sha256 === "e26a3e4e26300442b47d27e1b515ccbc85b34425625d60f637faa28973f7a8f7"
      && adapter.source_artifact_sha256_embedded === false
      && adapter.artifact_binding === "active_artifact_exact_original_op_topology"
      && adapter.topology_binding?.method === "exact_original_op_id_name_and_input_output_tensor_ids"
      && adapter.topology_binding?.matched_original_op_count === (staticAnalysis?.ops?.length || 0)
      && adapter.original_node_count === (staticAnalysis?.ops?.length || 0)
      && adapter.delegated_op_count === delegatedRows.length
      && adapter.nondelegated_op_count === nondelegatedRows.length
      && adapter.delegate_node_count === tflitePartitions.length
      && adapter.execution_plan_node_count === tflitePartitions.length + nondelegatedRows.length
      && nullableClose(adapter.mapping_coverage_ratio, 1)
      && adapterCountsMatch
      && tfliteMappingsValid
      && tfliteTimingValid;
    check("CF-RUNTIME-012", !adapter || ortValid || tfliteValid, "Adapted runtime evidence identity, mapping, execution-plan counts, duration limits, or pinned source provenance is inconsistent with the parsed graph.", ["/evidence/runtime_results/runtime_assignment/source/adapter", "/evidence/static_analysis/ops"]);
    const ortReportValid = !ortAdapter || (String(engineeringReport || "").includes(`profile SHA-256 ${runtimeAssignment.source.profile_sha256}`)
      && String(engineeringReport || "").includes(adapter.source_commit)
      && String(engineeringReport || "").includes(`mapped ${adapter.mapped_kernel_event_count}/${adapter.kernel_event_count} kernel event(s)`)
      && (!adapter.native_capture
        ? String(engineeringReport || "").includes("runtime version/build/preparation are DECLARED; provider/node/duration rows are OBSERVED_RUNTIME")
        : String(engineeringReport || "").includes(`artifact content-set SHA-256 ${adapter.native_capture.artifact.content_set_sha256}`)
        && String(engineeringReport || "").includes(`external files ${adapter.native_capture.artifact.external_data.files.length}, tensor ranges ${adapter.native_capture.artifact.external_data.tensor_count}`)
        && String(engineeringReport || "").includes("active ONNX plus external-data content set")
        && String(engineeringReport || "").includes("### ONNX Runtime Selected-Build Provider Binding")
        && String(engineeringReport || "").includes(adapter.native_capture.selected_build_provider_binding.supported_backends_sha256)
        && String(engineeringReport || "").includes(adapter.native_capture.selected_build_provider_binding.interpretation_boundary)));
    const tfliteReportValid = !tfliteAdapter || (String(engineeringReport || "").includes(`runtime-plan protobuf SHA-256 ${runtimeAssignment.source.profile_sha256}`)
      && String(engineeringReport || "").includes(adapter.source_commit)
      && String(engineeringReport || "").includes(`exact topology ${adapter.topology_binding?.matched_original_op_count}/${adapter.topology_binding?.graph_op_count} original op(s)`)
      && String(engineeringReport || "").includes("source artifact SHA-256 embedded false")
      && String(engineeringReport || "").includes("executed microkernel symbols")
      && (!timedTfliteAdapter || (String(engineeringReport || "").includes(`timing protobuf SHA-256 ${adapter.timing_profile.profile_sha256}`)
        && String(engineeringReport || "").includes(`execution nodes ${adapter.timing_profile.mapped_execution_node_count}/${adapter.timing_profile.execution_plan_node_count}`)
        && String(engineeringReport || "").includes("separate, not added to partition totals"))));
    check("CF-RUNTIME-013", !adapter || (ortReportValid && tfliteReportValid), "Engineering report does not disclose adapted runtime source digest, mapping coverage, parser basis, and declared-vs-observed provenance.", ["/evidence/runtime_results/runtime_assignment/source/adapter", "/engineering_report.md"]);
    check("CF-RUNTIME-015", !timedTfliteAdapter || tfliteTimingValid, "TFLite timing totals, execution-node coverage, capture binding, or delegated-op non-duplication differ from independent reconstruction.", ["/evidence/runtime_results/runtime_assignment/source/adapter/timing_profile", "/evidence/runtime_results/runtime_assignment/comparison/duration_comparison"]);
  }

  const criticalFailures = failures.filter((item) => item.severity === "critical");
  return {
    schema: ANALYZER_METADATA.schemas.conformanceReport,
    status: failures.length ? "fail" : "pass",
    checked_invariants: checks.length,
    failed_invariants: failures.length,
    critical_failed_invariants: criticalFailures.length,
    release_export_allowed: criticalFailures.length === 0,
    development_export_allowed: true,
    development_watermark_required: criticalFailures.length > 0,
    finding_evidence_pointer_validation: findingBindings,
    checks,
    failures,
    scope: "Cross-output semantic invariants for the engineering report, static analysis, quantization evidence, findings register, security posture, and ML-BOM. Bundle privacy and package-member integrity are validated by the package manifest/verifier contract after envelope construction.",
  };
}

function sameCountObject(left, right) {
  const leftKeys = Object.keys(left || {}).sort();
  const rightKeys = Object.keys(right || {}).sort();
  return leftKeys.length === rightKeys.length
    && leftKeys.every((key, index) => key === rightKeys[index] && Number(left[key]) === Number(right[key]));
}

export function assertConformance(report) {
  if (report?.status !== "pass") {
    const first = report?.failures?.[0];
    throw new Error(`Evidence conformance failed${first?.id ? ` (${first.id})` : ""}: ${first?.message || "unknown invariant"}`);
  }
  return report;
}
