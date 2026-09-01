import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import { buildTfliteAdditionalSourceProfiles } from "./tflite-accelerator-source-profiles.js";

export const ACCELERATOR_BINDING_SCHEMA = "deepbom.accelerator_binding.v1";
export const ACCELERATOR_EVIDENCE_STAGES = Object.freeze([
  "serialized_artifact",
  "source_eligibility",
  "selected_build",
  "compiled_plan",
  "observed_assignment",
  "measured_execution",
]);

const SHA256 = /^[a-f0-9]{64}$/;
const PROVIDERS = new Set(["apple", "google", "nvidia", "qualcomm", "android", "onnxruntime", "tensorflow"]);
const DEVICE_CLASSES = new Set(["gpu", "npu", "tpu", "heterogeneous_accelerator"]);
const BINDING_SOURCES = new Set([
  "source_rulepack",
  "selected_build_attestation",
  "compiled_plan_import",
  "runtime_evidence_import",
  "observed_host_profile",
]);

export function buildAcceleratorBinding(input) {
  const body = normalizeBinding(input);
  return Object.freeze({ ...body, binding_sha256: sha256TextHex(canonicalJson(body)) });
}

export function validateAcceleratorBinding(value) {
  const copy = JSON.parse(JSON.stringify(value || null));
  const declared = String(copy?.binding_sha256 || "").toLowerCase();
  if (copy) delete copy.binding_sha256;
  const body = normalizeBinding(copy);
  if (!validSha(declared) || declared !== sha256TextHex(canonicalJson(body))) {
    throw new Error("Accelerator binding SHA-256 is invalid.");
  }
  return Object.freeze({ ...body, binding_sha256: declared });
}

export function buildCoreMlAcceleratorBinding(analysis, computePlan) {
  if (computePlan?.schema !== "deepbom.coreml_compute_plan.v1"
    || computePlan?.artifact?.sha256 !== analysis?.model_sha256
    || !validSha(computePlan?.normalized_manifest_sha256)) {
    throw new Error("Core ML accelerator binding requires an identity-bound normalized MLComputePlan.");
  }
  return buildAcceleratorBinding({
    profile_id: `apple_coreml_${String(computePlan.configuration?.compute_units || "all").toLowerCase()}`,
    provider: "apple",
    backend: "coreml",
    device_class: "heterogeneous_accelerator",
    binding_source: "compiled_plan_import",
    evidence_stage: "compiled_plan",
    artifact_sha256: analysis.model_sha256,
    source_rulepack_sha256: computePlan.runtime?.coremltools_compute_plan_source_sha256 || null,
    selected_build_sha256: null,
    compiled_plan_sha256: computePlan.normalized_manifest_sha256,
    runtime_trace_sha256: null,
    configuration: {
      compute_units: computePlan.configuration?.compute_units || null,
      function_name: computePlan.configuration?.function_name || null,
      compiled_model_content_sha256: computePlan.runtime?.compiled_model_content_sha256 || null,
      macos_version: computePlan.runtime?.macos_version || null,
      os_build: computePlan.runtime?.os_build || null,
      hardware_model: computePlan.runtime?.hardware_model || null,
    },
    coverage: {
      assessed_operation_count: computePlan.summary?.mapped_operation_count ?? null,
      preferred_compute_device_counts: computePlan.summary?.preferred_compute_device_counts || {},
      unresolved_operation_count: computePlan.summary?.unresolved_device_usage_count ?? null,
    },
    claims: claimsForStage("compiled_plan"),
    interpretation_boundary: "This binding records MLComputePlan anticipated device usage for one compiled Core ML model and compute-unit configuration. It does not establish executed CPU, GPU, or Neural Engine assignment, physical transfer, latency, or task correctness.",
  });
}

export function buildNvidiaHostAcceleratorBinding(analysis, profileBinding) {
  if (profileBinding?.schema !== "deepbom.accelerator_profile_binding.v1"
    || !validSha(profileBinding?.binding_sha256) || !validSha(profileBinding?.profile_sha256)) {
    throw new Error("NVIDIA accelerator binding requires a validated host profile binding.");
  }
  return buildAcceleratorBinding({
    profile_id: `nvidia_cuda_device_${profileBinding.selected_device?.index ?? 0}`,
    provider: "nvidia",
    backend: "cuda",
    device_class: "gpu",
    binding_source: "observed_host_profile",
    evidence_stage: "serialized_artifact",
    artifact_sha256: requireSha(analysis?.model_sha256, "artifact SHA-256"),
    source_rulepack_sha256: null,
    selected_build_sha256: null,
    compiled_plan_sha256: null,
    runtime_trace_sha256: null,
    host_profile_sha256: profileBinding.profile_sha256,
    configuration: {
      device_index: profileBinding.selected_device?.index ?? null,
      compute_capability: profileBinding.selected_device?.compute_capability || null,
      driver_version: profileBinding.selected_device?.driver_version || null,
      physical_vram_bytes: profileBinding.selected_device?.memory_total_bytes || null,
    },
    coverage: { selected_build: "not_bound", assignment: "not_observed", execution: "not_measured" },
    claims: claimsForStage("serialized_artifact"),
    interpretation_boundary: "The NVIDIA host profile is observed, but the selected CUDA, TensorRT, or ONNX Runtime build is not bound. No provider inclusion, layer assignment, execution, latency, or fit claim is emitted.",
  });
}

export function buildEdgeTpuAcceleratorBinding(analysis, compilerEvidence) {
  if (compilerEvidence?.schema !== "deepbom.edgetpu_compiler_evidence.v1"
    || compilerEvidence?.artifact_sha256 !== analysis?.model_sha256
    || !validSha(compilerEvidence?.evidence_sha256)) {
    throw new Error("Edge TPU accelerator binding requires validated compiler evidence for the active artifact.");
  }
  return buildAcceleratorBinding({
    profile_id: "google_edgetpu_compiler",
    provider: "google",
    backend: "edgetpu_compiler",
    device_class: "tpu",
    binding_source: "compiled_plan_import",
    evidence_stage: "compiled_plan",
    artifact_sha256: analysis.model_sha256,
    source_rulepack_sha256: null,
    selected_build_sha256: compilerEvidence.compiler.binary_sha256,
    compiled_plan_sha256: compilerEvidence.compiled_artifact_sha256,
    runtime_trace_sha256: null,
    configuration: {
      compiler_version: compilerEvidence.compiler.version,
      invocation_sha256: compilerEvidence.invocation_sha256,
      compiler_report_sha256: compilerEvidence.compiler_report_sha256,
    },
    coverage: compilerEvidence.summary,
    claims: claimsForStage("compiled_plan"),
    interpretation_boundary: "The Edge TPU compiler report binds mapped and unmapped serialized TFLite operations to one compiler binary, invocation, and compiled artifact. It is not an observed device assignment, transfer, latency, or successful execution trace.",
  });
}

export function buildLiteRtQualcommAcceleratorBinding(analysis, compilerEvidence) {
  if (compilerEvidence?.schema !== "deepbom.litert_qualcomm_compiler_dispatch_evidence.v1"
    || compilerEvidence?.artifact_sha256 !== analysis?.model_sha256
    || !validSha(compilerEvidence?.evidence_sha256)) {
    throw new Error("LiteRT Qualcomm accelerator binding requires validated compiler evidence for the active artifact.");
  }
  return buildAcceleratorBinding({
    profile_id: "litert_qualcomm_qnn_compiled_plan",
    provider: "qualcomm",
    backend: "litert_qnn",
    device_class: "npu",
    binding_source: "compiled_plan_import",
    evidence_stage: "compiled_plan",
    artifact_sha256: analysis.model_sha256,
    source_rulepack_sha256: compilerEvidence.source.rulepack_sha256,
    selected_build_sha256: compilerEvidence.compiler.binary_sha256,
    compiled_plan_sha256: compilerEvidence.compiled_plan_sha256,
    runtime_trace_sha256: null,
    configuration: {
      litert_commit: compilerEvidence.source.litert_commit,
      compiler_name: compilerEvidence.compiler.name,
      compiler_version: compilerEvidence.compiler.version,
      invocation_sha256: compilerEvidence.invocation_sha256,
      dispatch_status: compilerEvidence.dispatch.status,
      dispatch_runtime_binary_sha256: compilerEvidence.dispatch.runtime_binary_sha256,
      dispatch_evidence_sha256: compilerEvidence.dispatch.evidence_sha256,
    },
    coverage: compilerEvidence.summary,
    claims: claimsForStage("compiled_plan"),
    interpretation_boundary: compilerEvidence.interpretation_boundary,
  });
}

export function buildTfliteAdditionalAcceleratorBindings(analysis) {
  if (String(analysis?.format || "").toLowerCase() !== "tflite" || !validSha(analysis?.model_sha256)) return [];
  return buildTfliteAdditionalSourceProfiles(analysis).map((profile) => {
    const coreMl = profile.profile_id === "tflite_coreml_delegate";
    return buildAcceleratorBinding({
      profile_id: profile.profile_id,
      provider: coreMl ? "apple" : "qualcomm",
      backend: coreMl ? "tflite_coreml_delegate" : "litert_qnn",
      device_class: coreMl ? "heterogeneous_accelerator" : "npu",
      binding_source: "source_rulepack",
      evidence_stage: "source_eligibility",
      artifact_sha256: analysis.model_sha256,
      source_rulepack_sha256: profile.source.rulepack_sha256,
      selected_build_sha256: null,
      compiled_plan_sha256: null,
      runtime_trace_sha256: null,
      configuration: {
        source_repository: profile.source.repository,
        source_commit: profile.source.commit,
        source_path: profile.source.support_path,
      },
      coverage: {
        assessed_operation_count: profile.op_count,
        source_candidate_operation_count: profile.state_counts.CONDITIONALLY_ELIGIBLE,
        definite_exclusion_count: profile.state_counts.DEFINITE_EXCLUSION,
        unresolved_operation_count: profile.state_counts.UNRESOLVED,
      },
      claims: claimsForStage("source_eligibility"),
      interpretation_boundary: profile.interpretation_boundary,
    });
  });
}

export function buildTensorRtAcceleratorBinding(analysis, preflight) {
  const profile = preflight?.build_profile;
  const parser = preflight?.parser_observation;
  const inspector = preflight?.engine_inspector_evidence;
  if (!profile || (!parser && !inspector)) return null;
  const stage = inspector ? "compiled_plan" : "selected_build";
  return buildAcceleratorBinding({
    profile_id: profile.execution_path === "ort_tensorrt_ep" ? "onnxruntime_tensorrt_ep" : "nvidia_tensorrt_native",
    provider: "nvidia",
    backend: profile.execution_path === "ort_tensorrt_ep" ? "ort_tensorrt_ep" : "tensorrt",
    device_class: profile.dla_core == null ? "gpu" : "heterogeneous_accelerator",
    binding_source: inspector ? "compiled_plan_import" : "selected_build_attestation",
    evidence_stage: stage,
    artifact_sha256: requireSha(analysis?.model_sha256, "artifact SHA-256"),
    source_rulepack_sha256: null,
    selected_build_sha256: parser?.collector?.binary_sha256 || null,
    compiled_plan_sha256: inspector?.engine?.sha256 || null,
    runtime_trace_sha256: null,
    configuration: {
      build_profile_sha256: profile.profile_sha256,
      execution_path: profile.execution_path,
      precision: profile.precision,
      device_id: profile.device_id,
      device_compute_capability: profile.device_compute_capability,
      dla_core: profile.dla_core,
      allow_gpu_fallback: profile.allow_gpu_fallback,
    },
    coverage: {
      parser_observation_status: parser ? "observed" : "not_bound",
      engine_inspector_status: inspector?.status || "not_bound",
      graph_operation_count: analysis?.ops?.length ?? null,
      projection_state_counts: preflight?.projection?.state_counts || {},
    },
    claims: claimsForStage(stage),
    interpretation_boundary: inspector
      ? "The serialized TensorRT engine and inspector metadata are bound to the ONNX artifact and build profile. Selected tactics are engine metadata, not timing, physical transfer, memory allocation, task accuracy, or executed original-op assignment."
      : "TensorRT parser acceptance was observed for the selected binary and build profile. Engine build, optimized layers, tactics, execution, transfer, latency, and task accuracy remain unobserved.",
  });
}

export function buildRuntimeAcceleratorBindings(analysis, runtimeEvidence) {
  const runtime = runtimeEvidence?.runtimeAssignmentEvidence || runtimeEvidence?.runtime_assignment || runtimeEvidence;
  const assignments = Array.isArray(runtime?.assignments) ? runtime.assignments : [];
  if (!assignments.length || String(runtime?.artifact_sha256 || "").toLowerCase() !== String(analysis?.model_sha256 || "").toLowerCase()) return [];
  const providers = new Map();
  for (const row of assignments) {
    const identity = providerIdentity(row?.provider);
    if (!identity) continue;
    if (!providers.has(identity.backend)) providers.set(identity.backend, { identity, rows: [] });
    providers.get(identity.backend).rows.push(row);
  }
  const runtimeTraceSha256 = sha256TextHex(canonicalJson(runtime));
  return [...providers.values()].map(({ identity, rows }) => {
    const measured = rows.some(hasNonNegativeTiming);
    return buildAcceleratorBinding({
    profile_id: `${identity.provider}_${identity.backend}`,
    provider: identity.provider,
    backend: identity.backend,
    device_class: identity.device_class,
    binding_source: "runtime_evidence_import",
    evidence_stage: measured ? "measured_execution" : "observed_assignment",
    artifact_sha256: analysis.model_sha256,
    source_rulepack_sha256: null,
    selected_build_sha256: runtime.runtime_binary_sha256 || runtime.binary_sha256 || null,
    compiled_plan_sha256: null,
    runtime_trace_sha256: runtimeTraceSha256,
    configuration: { runtime_name: runtime.runtime?.name || runtime.runtime_name || null },
    coverage: { observed_assignment_count: rows.length, graph_operation_count: analysis?.ops?.length ?? null },
    claims: claimsForStage(measured ? "measured_execution" : "observed_assignment"),
    interpretation_boundary: "Provider assignment is imported from identity-bound runtime evidence. Timing is claimed only when additive per-row timing fields are present; physical transfer and unreported runtime internals remain outside this binding.",
    });
  });
}

export function buildTfliteAcceleratorBindings(analysis, runtimeEvidence = null) {
  const source = analysis?.tflite_delegate_compatibility_evidence;
  if (source?.schema !== "deepbom.tflite_delegate_source_rulepack.v1" || !validSha(source?.rulepack_sha256)) return [];
  const inventory = runtimeEvidence?.tflite_delegate_build_inventory || null;
  return (source.profiles || []).map((profile) => {
    const gpu = profile.id === "tflite_gpu";
    const selected = gpu
      ? inventory?.gpu?.compiled_status === "enabled_by_declared_cmake_option"
      : inventory?.nnapi?.compiled_status === "enabled_by_declared_cmake_option_and_android_gate";
    const stage = selected && validSha(inventory?.build_manifest_sha256) ? "selected_build" : "source_eligibility";
    return buildAcceleratorBinding({
      profile_id: profile.id,
      provider: gpu ? "tensorflow" : "android",
      backend: profile.id,
      device_class: gpu ? "gpu" : "npu",
      binding_source: stage === "selected_build" ? "selected_build_attestation" : "source_rulepack",
      evidence_stage: stage,
      artifact_sha256: requireSha(analysis?.model_sha256, "artifact SHA-256"),
      source_rulepack_sha256: source.rulepack_sha256,
      selected_build_sha256: stage === "selected_build" ? inventory.build_manifest_sha256 : null,
      compiled_plan_sha256: null,
      runtime_trace_sha256: null,
      configuration: gpu ? {
        compiled_status: inventory?.gpu?.compiled_status || "not_bound",
        quantized_model_flag_status: inventory?.gpu?.quantized_model_flag_status || "not_bound",
        experimental_flags: inventory?.gpu?.experimental_flags ?? null,
      } : {
        compiled_status: inventory?.nnapi?.compiled_status || "not_bound",
        runtime_feature_level: inventory?.nnapi?.runtime_feature_level ?? null,
        accelerator_identity: inventory?.nnapi?.accelerator_identity || null,
        capability_source: inventory?.nnapi?.capability_source || "not_collected",
      },
      coverage: {
        assessed_operation_count: profile.assessed_graph_op_count ?? analysis?.ops?.length ?? null,
        source_candidate_operation_count: profile.source_candidate_after_artifact_precheck_count ?? null,
        definite_exclusion_count: profile.definite_exclusion_count ?? null,
      },
      claims: claimsForStage(stage),
      interpretation_boundary: stage === "selected_build"
        ? "The delegate source profile and selected-build inventory are bound. Device acceptance, partition assignment, physical transfer, execution, and latency remain unobserved."
        : "Pinned source registration and artifact-visible prechecks identify candidates only. Selected-build inclusion, device acceptance, partition assignment, execution, and latency remain unobserved.",
    });
  });
}

export function buildOrtAcceleratorBindings(analysis, runtimeEvidence = null) {
  const compatibility = analysis?.ort_compatibility_evidence;
  if (!compatibility || compatibility.assessment_status !== "complete") return [];
  const providers = Array.isArray(compatibility.execution_providers) ? compatibility.execution_providers : [];
  const selected = runtimeEvidence?.source?.adapter?.native_capture?.selected_build_provider_binding
    || runtimeEvidence?.selected_build_provider_binding
    || null;
  const selectedByProfile = new Map((selected?.bindings || [])
    .filter((row) => row?.bundled === true && row?.source_profile)
    .map((row) => [row.source_profile, row]));
  const selectedBuildSha256 = selected?.source_build_attestation?.attestation_sha256 || null;
  const rows = [];
  for (const source of providers) {
    const identity = ortProviderIdentity(source?.execution_provider);
    if (!identity || !validSha(source?.source_sha256)) continue;
    const coverage = {
      assessed_operation_count: source.assessed_op_count ?? analysis?.ops?.length ?? null,
      source_candidate_operation_count: source.source_candidate_after_artifact_precheck_count ?? null,
      definite_exclusion_count: source.artifact_precheck_definite_fail_op_count ?? null,
      unresolved_operation_count: source.artifact_precheck_unresolved_op_count ?? null,
    };
    rows.push(buildAcceleratorBinding({
      profile_id: `ort_${source.execution_provider}`,
      ...identity,
      binding_source: "source_rulepack",
      evidence_stage: "source_eligibility",
      artifact_sha256: requireSha(analysis?.model_sha256, "artifact SHA-256"),
      source_rulepack_sha256: source.source_sha256,
      selected_build_sha256: null,
      compiled_plan_sha256: null,
      runtime_trace_sha256: null,
      configuration: {
        ort_source_commit: compatibility.source_commit || null,
        execution_provider: source.execution_provider,
        source_id: source.source_id || null,
      },
      coverage,
      claims: claimsForStage("source_eligibility"),
      interpretation_boundary: "Pinned ONNX Runtime provider registrations and artifact-visible prechecks establish source eligibility only. Selected-build inclusion, GetCapability partitioning, optimized assignment, physical transfer, execution, latency, and task correctness remain unobserved.",
    }));
    const selectedRow = selectedByProfile.get(source.execution_provider);
    if (!selectedRow || !validSha(selectedBuildSha256)) continue;
    rows.push(buildAcceleratorBinding({
      profile_id: `ort_${source.execution_provider}`,
      ...identity,
      binding_source: "selected_build_attestation",
      evidence_stage: "selected_build",
      artifact_sha256: requireSha(analysis?.model_sha256, "artifact SHA-256"),
      source_rulepack_sha256: source.source_sha256,
      selected_build_sha256: selectedBuildSha256,
      compiled_plan_sha256: null,
      runtime_trace_sha256: null,
      configuration: {
        ort_source_commit: compatibility.source_commit || null,
        execution_provider: source.execution_provider,
        backend_name: selectedRow.backend_name,
        supported_backends_sha256: selected.supported_backends_sha256 || null,
      },
      coverage,
      claims: claimsForStage("selected_build"),
      interpretation_boundary: "The selected ONNX Runtime build attestation includes this provider and is cross-referenced to its pinned source profile. GetCapability acceptance, graph transforms, optimized assignment, physical transfer, execution, latency, and task correctness remain unobserved.",
    }));
  }
  return rows;
}

export function collectAcceleratorBindings(analysis, runtimeEvidence = null, artifactSha256 = null) {
  const boundAnalysis = validSha(artifactSha256)
    ? { ...analysis, model_sha256: artifactSha256 }
    : analysis;
  const rows = [...(Array.isArray(boundAnalysis?.accelerator_bindings) ? boundAnalysis.accelerator_bindings : [])];
  if (!validSha(boundAnalysis?.model_sha256)) return mergeAcceleratorBindings(rows);
  if (boundAnalysis?.accelerator_profile_binding) rows.push(buildNvidiaHostAcceleratorBinding(boundAnalysis, boundAnalysis.accelerator_profile_binding));
  const coreMlPlan = boundAnalysis?.coreml_compute_plan
    || (runtimeEvidence?.schema === "deepbom.coreml_compute_plan.v1" ? runtimeEvidence : null)
    || runtimeEvidence?.coreml_compute_plan || null;
  if (coreMlPlan) rows.push(buildCoreMlAcceleratorBinding(boundAnalysis, coreMlPlan));
  if (boundAnalysis?.edgetpu_compiler_evidence) rows.push(buildEdgeTpuAcceleratorBinding(boundAnalysis, boundAnalysis.edgetpu_compiler_evidence));
  if (boundAnalysis?.litert_qualcomm_evidence) rows.push(buildLiteRtQualcommAcceleratorBinding(boundAnalysis, boundAnalysis.litert_qualcomm_evidence));
  const tensorRt = buildTensorRtAcceleratorBinding(boundAnalysis, boundAnalysis?.tensorrt_static_preflight);
  if (tensorRt) rows.push(tensorRt);
  rows.push(...buildTfliteAcceleratorBindings(boundAnalysis, runtimeEvidence));
  rows.push(...buildTfliteAdditionalAcceleratorBindings(boundAnalysis));
  rows.push(...buildOrtAcceleratorBindings(boundAnalysis, runtimeEvidence));
  rows.push(...buildRuntimeAcceleratorBindings(boundAnalysis, runtimeEvidence));
  return mergeAcceleratorBindings(rows);
}

export function mergeAcceleratorBindings(...groups) {
  const byIdentity = new Map();
  for (const value of groups.flat(Infinity).filter(Boolean)) {
    const binding = validateAcceleratorBinding(value);
    const key = `${binding.profile_id}\0${binding.evidence_stage}\0${binding.binding_sha256}`;
    byIdentity.set(key, binding);
  }
  return [...byIdentity.values()].sort((left, right) => left.profile_id.localeCompare(right.profile_id)
    || ACCELERATOR_EVIDENCE_STAGES.indexOf(left.evidence_stage) - ACCELERATOR_EVIDENCE_STAGES.indexOf(right.evidence_stage)
    || left.binding_sha256.localeCompare(right.binding_sha256));
}

function normalizeBinding(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Accelerator binding must be an object.");
  const stage = requiredEnum(value.evidence_stage, ACCELERATOR_EVIDENCE_STAGES, "accelerator evidence stage");
  const body = {
    schema: ACCELERATOR_BINDING_SCHEMA,
    profile_id: requiredToken(value.profile_id, "accelerator profile id"),
    provider: requiredEnum(value.provider, PROVIDERS, "accelerator provider"),
    backend: requiredToken(value.backend, "accelerator backend"),
    device_class: requiredEnum(value.device_class, DEVICE_CLASSES, "accelerator device class"),
    binding_source: requiredEnum(value.binding_source, BINDING_SOURCES, "accelerator binding source"),
    evidence_stage: stage,
    artifact_sha256: requireSha(value.artifact_sha256, "artifact SHA-256"),
    source_rulepack_sha256: optionalSha(value.source_rulepack_sha256, "source rulepack SHA-256"),
    selected_build_sha256: optionalSha(value.selected_build_sha256, "selected build SHA-256"),
    compiled_plan_sha256: optionalSha(value.compiled_plan_sha256, "compiled plan SHA-256"),
    runtime_trace_sha256: optionalSha(value.runtime_trace_sha256, "runtime trace SHA-256"),
    host_profile_sha256: optionalSha(value.host_profile_sha256, "host profile SHA-256"),
    configuration: normalizePlainObject(value.configuration, "accelerator configuration"),
    coverage: normalizePlainObject(value.coverage, "accelerator coverage"),
    claims: normalizeClaims(value.claims, stage),
    interpretation_boundary: requiredText(value.interpretation_boundary, "accelerator interpretation boundary"),
  };
  if (stage === "source_eligibility" && !body.source_rulepack_sha256) throw new Error("Source-eligibility binding requires source_rulepack_sha256.");
  if (stage === "selected_build" && !body.selected_build_sha256) throw new Error("Selected-build binding requires selected_build_sha256.");
  if (stage === "compiled_plan" && !body.compiled_plan_sha256) throw new Error("Compiled-plan binding requires compiled_plan_sha256.");
  if (["observed_assignment", "measured_execution"].includes(stage) && !body.runtime_trace_sha256) {
    throw new Error(`${stage} binding requires runtime_trace_sha256.`);
  }
  return body;
}

function claimsForStage(stage) {
  const rank = ACCELERATOR_EVIDENCE_STAGES.indexOf(stage);
  return {
    source_eligibility: rank >= 1,
    selected_build: rank >= 2,
    compiled_plan: rank >= 3,
    observed_assignment: rank >= 4,
    measured_execution: rank >= 5,
  };
}

function normalizeClaims(value, stage) {
  const expected = claimsForStage(stage);
  for (const key of Object.keys(expected)) {
    if (value?.[key] !== expected[key]) throw new Error(`Accelerator claim ${key} is inconsistent with evidence stage ${stage}.`);
  }
  return expected;
}

function providerIdentity(value) {
  const text = String(value || "").toLowerCase();
  if (/tensorrt/.test(text)) return { provider: "nvidia", backend: "tensorrt", device_class: "gpu" };
  if (/cuda/.test(text)) return { provider: "nvidia", backend: "cuda", device_class: "gpu" };
  if (/qnn/.test(text)) return { provider: "qualcomm", backend: "qnn", device_class: "npu" };
  if (/nnapi/.test(text)) return { provider: "android", backend: "nnapi", device_class: "npu" };
  if (/coreml/.test(text)) return { provider: "apple", backend: "coreml", device_class: "heterogeneous_accelerator" };
  if (/directml/.test(text)) return { provider: "onnxruntime", backend: "directml", device_class: "gpu" };
  if (/webgpu/.test(text)) return { provider: "onnxruntime", backend: "webgpu", device_class: "gpu" };
  if (/webnn/.test(text)) return { provider: "onnxruntime", backend: "webnn", device_class: "heterogeneous_accelerator" };
  if (/gpu/.test(text)) return { provider: "tensorflow", backend: "tflite_gpu", device_class: "gpu" };
  return null;
}

function ortProviderIdentity(value) {
  return {
    cuda: { provider: "nvidia", backend: "ort_cuda", device_class: "gpu" },
    directml: { provider: "onnxruntime", backend: "ort_directml", device_class: "gpu" },
    webgpu: { provider: "onnxruntime", backend: "ort_webgpu", device_class: "gpu" },
    webnn: { provider: "onnxruntime", backend: "ort_webnn", device_class: "heterogeneous_accelerator" },
    qnn: { provider: "qualcomm", backend: "ort_qnn", device_class: "npu" },
    coreml: { provider: "apple", backend: "ort_coreml", device_class: "heterogeneous_accelerator" },
    nnapi: { provider: "android", backend: "ort_nnapi", device_class: "npu" },
  }[String(value || "").toLowerCase()] || null;
}

function hasNonNegativeTiming(row) {
  return [row?.duration_us, row?.duration_sum_us].some((value) => {
    if (value == null || value === "" || typeof value === "boolean") return false;
    const numeric = Number(value);
    return Number.isFinite(numeric) && numeric >= 0;
  });
}

function normalizePlainObject(value, label) {
  if (value == null) return {};
  if (typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object.`);
  return JSON.parse(canonicalJson(value));
}

function requiredEnum(value, allowed, label) {
  const text = String(value || "").trim();
  const present = typeof allowed?.has === "function" ? allowed.has(text) : Array.isArray(allowed) && allowed.includes(text);
  if (!present) throw new Error(`${label} is invalid.`);
  return text;
}

function requiredToken(value, label) {
  const text = String(value || "").trim();
  if (!/^[a-z0-9][a-z0-9_.-]{0,127}$/.test(text)) throw new Error(`${label} is invalid.`);
  return text;
}

function requiredText(value, label) {
  const text = String(value || "").trim();
  if (!text || text.length > 2000) throw new Error(`${label} is invalid.`);
  return text;
}

function requireSha(value, label) {
  const text = String(value || "").toLowerCase();
  if (!validSha(text)) throw new Error(`${label} is invalid.`);
  return text;
}

function optionalSha(value, label) { return value == null || value === "" ? null : requireSha(value, label); }
function validSha(value) { return typeof value === "string" && SHA256.test(value); }
