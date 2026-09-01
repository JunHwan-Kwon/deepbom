import { buildFindingsRegister } from "./report-findings.js";
import { buildInterfaceQuantizationContractLedger } from "./quantization-contract-summary.js";
import { canonicalJson, normalizeJsonContractValue } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import { validateArtifactSet } from "./artifact-set.js";
import { validateNvidiaAcceleratorProfileBinding } from "./accelerator-profile-binding.js";
import { collectAcceleratorBindings, validateAcceleratorBinding } from "./accelerator-binding.js";
import { validateCpuCostTargetBinding } from "./cpu-target-binding.js";
import { deriveMacCoverage } from "./mac-coverage.js";

export const ARTIFACT_EVIDENCE_SCHEMA = "deepbom.artifact_evidence_envelope.v1";
export const EVIDENCE_CLASSES = Object.freeze([
  "OBSERVED",
  "SOURCE_BACKED",
  "DERIVED",
  "DERIVED_WITH_HEURISTIC_THRESHOLD",
  "PREDICTED",
  "ESTIMATED",
  "DECLARED_UNVERIFIED",
  "MEASURED",
  "NOT_ASSESSABLE",
  "NOT_APPLICABLE",
]);

const SHA256 = /^[a-f0-9]{64}$/i;
const FORMAT_CAPABILITIES = Object.freeze({
  tflite: [
    "artifact_identity", "graph", "interfaces", "tensor_payloads", "affine_quantization",
    "metadata", "associated_files", "runtime_floor", "static_cost", "tflite_arena",
    "xnnpack_contracts", "quantization_proofs", "artifact_byte_integrity",
  ],
  onnx: [
    "artifact_identity", "graph", "interfaces", "tensor_payloads", "affine_quantization",
    "metadata", "external_data", "runtime_floor", "static_cost", "opset_contracts",
    "recursive_types", "control_flow", "onnx_runtime_contracts", "llm_serialized_graph",
    "tensorrt_static_preflight",
  ],
  gguf: [
    "artifact_identity", "tensor_inventory", "tensor_payloads", "block_quantization", "metadata", "runtime_requirements",
    "llm_architecture", "llm_tokenizer", "llm_layer_storage", "llm_state_projection", "llm_compute_scenario", "llm_memory_feasibility", "llm_static_memory_placement", "llm_runtime_contract", "medical_deployment_declaration",
  ],
  safetensors: [
    "artifact_identity", "tensor_inventory", "tensor_payloads", "tensor_payload_ranges", "metadata",
    "llm_architecture", "llm_tokenizer", "llm_layer_storage", "llm_state_projection", "llm_compute_scenario", "llm_memory_feasibility", "llm_static_memory_placement", "llm_runtime_contract", "medical_deployment_declaration",
  ],
  coreml: [
    "artifact_identity", "package_inventory", "graph", "interfaces", "tensor_payloads", "affine_quantization",
    "metadata", "runtime_requirements", "static_cost",
  ],
});

function text(value) {
  return String(value ?? "").trim();
}

function sha256(value) {
  const digest = text(value).toLowerCase();
  return SHA256.test(digest) ? digest : null;
}

function finite(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function evidenceClass(value, fallback = "NOT_ASSESSABLE") {
  const normalized = text(value).toUpperCase();
  if (EVIDENCE_CLASSES.includes(normalized)) return normalized;
  if (normalized.includes("NOT_ASSESSABLE")) return "NOT_ASSESSABLE";
  if (normalized.includes("NOT_APPLICABLE")) return "NOT_APPLICABLE";
  if (normalized.includes("MEASURED")) return "MEASURED";
  if (normalized.includes("PREDICTED")) return "PREDICTED";
  if (normalized.includes("ESTIMATED")) return "ESTIMATED";
  if (normalized.includes("SOURCE_BACKED") || normalized.includes("SOURCE-BASED") || normalized.includes("SOURCE_PINNED")) return "SOURCE_BACKED";
  if (normalized.includes("DECLARED")) return "DECLARED_UNVERIFIED";
  if (normalized.includes("HEURISTIC")) return "DERIVED_WITH_HEURISTIC_THRESHOLD";
  if (normalized.includes("DERIVED")) return "DERIVED";
  if (normalized.includes("OBSERVED")) return "OBSERVED";
  return fallback;
}

function observedMetadata(analysis) {
  const metadata = analysis?.metadata_presence || {};
  return {
    status: text(metadata.status || "not_assessed"),
    evidence_class: metadata.status === "assessed" ? "OBSERVED" : "NOT_ASSESSABLE",
    producer_name: text(metadata.producer_name) || null,
    producer_version: text(metadata.producer_version) || null,
    model_version: text(metadata.metadata_model_version || metadata.model_version) || null,
    author: text(metadata.metadata_author) || null,
    license: text(metadata.metadata_license) || null,
    description: text(metadata.metadata_model_description || metadata.description) || null,
    preprocessing_contract_status: text(metadata.preprocessing_contract_status || "not_declared"),
    output_semantics_documented: metadata.output_semantics_documented === true,
  };
}

function onnxExternalFiles(analysis) {
  const external = analysis?.onnx_external_data || {};
  const references = external.tensors || [];
  return (external.supplied_files || [])
    .filter((file) => file?.used === true && sha256(file.sha256))
    .map((file) => {
      const path = text(file.path);
      const boundReferences = references.filter((row) => text(row.sidecar_path) === path);
      const statuses = unique(boundReferences.map((row) => text(row.payload_status)));
      const fullyVerified = boundReferences.length > 0 && boundReferences.every((row) => row.payload_status === "verified");
      return {
        role: "external_weights",
        path,
        byte_length: finite(file.byte_length),
        sha256: sha256(file.sha256),
        sha1: text(file.sha1) || null,
        required: true,
        evidence_class: "OBSERVED",
        verification_status: fullyVerified
          ? "payload_range_cardinality_and_hash_verified"
          : `payload_verification_incomplete:${statuses.join(",") || "reference_status_unavailable"}`,
      };
    });
}

function tfliteAssociatedFiles(analysis) {
  const metadata = analysis?.metadata_presence || {};
  const files = metadata.packed_associated_files || [];
  return files
    .filter((file) => sha256(file?.payload_sha256))
    .map((file) => ({
      role: "associated_file",
      path: text(file.name || file.path),
      byte_length: finite(file.decoded_size ?? file.uncompressed_size),
      sha256: sha256(file.payload_sha256),
      sha1: null,
      required: (metadata.output_associated_files || []).some((item) => item?.name === file?.name),
      evidence_class: "OBSERVED",
      verification_status: text(file.status || file.payload_status || "archive_payload_verified"),
    }));
}

function suppliedBundleFiles(analysis) {
  return (analysis?.artifact_bundle?.files || [])
    .filter((file) => sha256(file?.sha256))
    .map((file) => ({
      role: text(file.role || "supporting_file"),
      path: text(file.path || file.name),
      byte_length: finite(file.byte_length ?? file.size),
      sha256: sha256(file.sha256),
      sha1: null,
      required: file.required !== false,
      evidence_class: evidenceClass(file.evidence_class, "OBSERVED"),
      verification_status: text(file.verification_status || "hash_verified"),
    }));
}

function externalFiles(analysis) {
  const files = [...onnxExternalFiles(analysis), ...tfliteAssociatedFiles(analysis), ...suppliedBundleFiles(analysis)];
  const seen = new Set();
  return files.filter((file) => {
    const key = `${file.sha256}:${file.path}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function capabilityManifest(analysis, format) {
  const available = new Set(FORMAT_CAPABILITIES[format] || ["artifact_identity"]);
  const states = new Map();
  const check = (id, value, complete = true) => {
    if (!available.has(id) || states.has(id)) return;
    states.set(id, value == null ? "unavailable" : complete ? "assessed" : "partial");
  };
  const checkStatus = (id, value, status) => {
    if (!available.has(id) || states.has(id)) return;
    states.set(id, assessmentState(status, value));
  };
  check("artifact_identity", analysis);
  check("graph", analysis?.ops && analysis?.tensors);
  check("interfaces", analysis?.inputs && analysis?.outputs);
  if (format === "onnx") {
    const weights = analysis?.weight_integrity;
    check("tensor_payloads", weights || analysis?.onnx_external_data_structure_binding,
      weights?.status === "assessed" && weights?.coverage_status === "complete");
  } else {
    check("tensor_payloads", analysis?.weight_integrity || analysis?.tensor_numerical_integrity || analysis?.tensors,
      analysis?.tensor_numerical_integrity ? analysis.tensor_numerical_integrity.status === "assessed"
        : format === "coreml" ? analysis?.weight_integrity?.status === "assessed" : true);
  }
  check("affine_quantization", analysis?.tensors);
  check("metadata", analysis?.metadata_presence);
  if (format === "onnx" && Number(analysis?.onnx_external_data?.tensor_count || 0) === 0) {
    check("external_data", analysis?.onnx_external_data, true);
  } else if (analysis?.onnx_external_data_structure_binding) {
    check("external_data", analysis.onnx_external_data_structure_binding, false);
  } else {
    checkStatus("external_data", analysis?.onnx_external_data, analysis?.onnx_external_data?.status);
  }
  check("associated_files", analysis?.metadata_presence, analysis?.metadata_presence?.associated_file_archive_status !== "partial");
  checkStatus("runtime_floor", analysis?.runtime_compat || analysis?.ort_compatibility_evidence,
    analysis?.runtime_compat?.status || analysis?.runtime_compat?.assessment_status || analysis?.ort_compatibility_evidence?.status);
  check("static_cost", analysis?.mac_assessment || (format === "tflite" && finite(analysis?.total_macs) != null ? analysis.ops : null),
    format === "coreml" ? analysis?.mac_assessment?.status === "assessed_all_decoded_compute_ops" && finite(analysis?.total_macs) != null : true);
  checkStatus("tflite_arena", analysis?.tensor_arena_plan, analysis?.tensor_arena_plan?.status);
  checkStatus("xnnpack_contracts", analysis?.xnnpack_selector_assessment_status, analysis?.xnnpack_selector_assessment_status);
  checkStatus("quantization_proofs", analysis?.quant_research_coverage || analysis?.channel_vitality,
    analysis?.quant_research_coverage?.status || analysis?.channel_vitality?.status);
  checkStatus("artifact_byte_integrity", analysis?.artifact_byte_integrity, analysis?.artifact_byte_integrity?.status);
  checkStatus("opset_contracts", analysis?.onnx_domain_analysis, analysis?.onnx_domain_analysis?.status);
  checkStatus("recursive_types", analysis?.onnx_type_proto_contract, analysis?.onnx_type_proto_contract?.status);
  checkStatus("control_flow", analysis?.onnx_shape_inference, analysis?.onnx_shape_inference?.status);
  checkStatus("onnx_runtime_contracts", analysis?.ort_compatibility_evidence, analysis?.ort_compatibility_evidence?.status);
  check("tensor_inventory", analysis?.tensors || analysis?.tensor_inventory);
  checkStatus("block_quantization", analysis?.gguf?.quantization || analysis?.quantization_status,
    analysis?.gguf?.quantization?.status || analysis?.quantization_status?.status);
  check("metadata", analysis?.metadata_presence || analysis?.metadata);
  check("package_inventory", analysis?.artifact_bundle);
  checkStatus("runtime_requirements", analysis?.runtime_compat || analysis?.runtime_requirements || analysis?.gguf?.backend_compatibility,
    analysis?.runtime_compat?.status || analysis?.runtime_requirements?.status || analysis?.gguf?.backend_compatibility?.status);
  check("tensor_payload_ranges", analysis?.tensor_inventory);
  checkStatus("llm_architecture", analysis?.on_device_llm?.architecture, analysis?.on_device_llm?.status);
  checkStatus("llm_tokenizer", analysis?.on_device_llm?.tokenizer, analysis?.on_device_llm?.tokenizer?.status);
  checkStatus("llm_layer_storage", analysis?.on_device_llm?.storage?.layer_storage, analysis?.on_device_llm?.storage?.layer_storage?.status);
  checkStatus("llm_state_projection", analysis?.on_device_llm?.state?.kv_projection, analysis?.on_device_llm?.state?.kv_projection?.status);
  checkStatus("llm_recurrent_state_projection", analysis?.on_device_llm?.state?.recurrent_projection, analysis?.on_device_llm?.state?.recurrent_projection?.status);
  checkStatus("llm_compute_scenario", analysis?.on_device_llm?.compute?.projection, analysis?.on_device_llm?.compute?.projection?.status);
  checkStatus("llm_memory_feasibility", analysis?.on_device_llm?.memory_feasibility, analysis?.on_device_llm?.memory_feasibility?.status);
  checkStatus("llm_static_memory_placement", analysis?.on_device_llm?.static_memory_placement, analysis?.on_device_llm?.static_memory_placement?.status);
  checkStatus("llm_runtime_contract", analysis?.on_device_llm?.runtime_contract, analysis?.on_device_llm?.runtime_contract?.status);
  checkStatus("llm_serialized_graph", analysis?.on_device_llm?.serialized_graph, analysis?.on_device_llm?.serialized_graph?.status);
  checkStatus("tensorrt_static_preflight", analysis?.tensorrt_static_preflight, analysis?.tensorrt_static_preflight?.status);
  checkStatus("tensorrt_llm_static_contract", analysis?.on_device_llm?.tensorrt_llm, analysis?.on_device_llm?.tensorrt_llm?.status);
  checkStatus("medical_deployment_declaration", analysis?.on_device_llm?.medical_ai_claim_boundary, "not_declared_in_artifact");
  for (const id of available) if (!states.has(id)) states.set(id, "unavailable");
  const select = (state) => [...states].filter(([, value]) => value === state).map(([id]) => id);
  return {
    schema: "deepbom.artifact_capability_manifest.v1",
    format,
    declared_capabilities: [...available],
    assessed: select("assessed"),
    partial: select("partial"),
    unavailable: select("unavailable"),
    conservation: {
      declared: available.size,
      classified: states.size,
      valid: available.size === states.size,
    },
  };
}

function assessmentState(status, value) {
  if (value == null) return "unavailable";
  const normalized = text(status).toLowerCase().replaceAll("-", "_");
  if (!normalized || /not_assessed|not_declared|not_selected|unavailable|unbound|external|unknown/.test(normalized)) return "unavailable";
  if (/invalid|failed|partial|incomplete|unresolved|unsupported/.test(normalized)) return "partial";
  if (/assessed|complete|observed|verified|source_candidate|available|decoded|bound/.test(normalized)) return "assessed";
  return "partial";
}

function findingRows(analysis, options) {
  let findings = Array.isArray(options?.findings) ? options.findings : null;
  if (!findings) {
    try {
      findings = buildFindingsRegister(analysis, {
        runtimeEvidence: options?.runtimeEvidence || null,
        analyzerMetadata: options?.analyzerMetadata,
      });
    } catch {
      findings = [];
    }
  }
  return findings.map((finding, index) => ({
    id: text(finding.finding_id || finding.id || finding.code || `finding-${index + 1}`),
    title: text(finding.title || finding.name || "Static analysis finding"),
    severity: text(finding.technical_priority || finding.severity || "info").toLowerCase(),
    status: text(finding.status || "open"),
    evidence_class: evidenceClass(finding.evidence_class || finding.evidenceClass, "DERIVED"),
    source_evidence_class: text(finding.evidence_class || finding.evidenceClass) || null,
    summary: text(finding.observation || finding.summary || finding.description || finding.detail),
    interpretation: text(finding.interpretation) || null,
    recommendation: text(finding.recommendation) || null,
    source_pointers: unique([
      finding.evidence_json_pointer,
      ...(finding.evidence_json_pointers || []),
      ...(finding.evidence_pointers || []),
    ].map(text)),
    rule_id: text(finding.source_rule_id || finding.rule_id || finding.ruleId) || null,
  }));
}

function formatExtensionSummary(analysis, format) {
  if (format === "tflite") return {
    schema_version: analysis?.schema_version ?? null,
    subgraph_count: analysis?.subgraphs ?? null,
    metadata_schema_identifier: analysis?.metadata_presence?.metadata_schema_identifier || null,
    arena_contract_schema: analysis?.tensor_arena_plan?.schema || null,
    xnnpack_rulepack_schema: analysis?.xnnpack_delegate_assessment?.schema || null,
    artifact_byte_integrity: analysis?.artifact_byte_integrity || null,
  };
  if (format === "onnx") return {
    ir_version: analysis?.onnx_ir_version ?? null,
    opsets: analysis?.opsets || [],
    producer: analysis?.producer || analysis?.metadata_presence?.producer_name || null,
    domain_contract_schema: analysis?.onnx_domain_analysis?.schema || null,
    shape_contract_schema: analysis?.onnx_shape_inference?.schema || null,
    quantization_binding_schema: analysis?.onnx_quantization_binding?.schema || null,
    external_data_structure_binding: analysis?.onnx_external_data_structure_binding || null,
    on_device_llm: analysis?.on_device_llm || null,
    tensorrt_static_preflight: analysis?.tensorrt_static_preflight || null,
  };
  const base = analysis?.format_extensions?.[format] || analysis?.[format] || null;
  if (["gguf", "safetensors"].includes(format)) return {
    ...base,
    tensor_numerical_integrity: analysis?.tensor_numerical_integrity || null,
    on_device_llm: analysis?.on_device_llm || null,
  };
  if (format === "coreml") return {
    ...base,
    weight_integrity: analysis?.weight_integrity || null,
    package_blob_integrity: analysis?.coreml_blob_integrity || null,
    tensor_liveness: analysis?.tensor_liveness || null,
    size_breakdown: analysis?.size_breakdown || null,
    mac_assessment: analysis?.mac_assessment || null,
  };
  if (format === "executorch") return {
    container: analysis?.executorch_container || null,
    version: analysis?.version ?? null,
    program: analysis?.executorch_program || null,
    flat_tensor: analysis?.executorch_flat_tensor || null,
    planned_memory: analysis?.tensor_liveness || null,
    size_breakdown: analysis?.size_breakdown || null,
    mac_assessment: analysis?.mac_assessment || null,
  };
  return base;
}

export function buildArtifactEvidenceEnvelope(analysis = {}, options = {}) {
  const format = text(analysis.format || options.format || "unknown").toLowerCase();
  const macCoverage = deriveMacCoverage(analysis);
  const interfaces = buildInterfaceQuantizationContractLedger(analysis);
  const files = externalFiles(analysis);
  const findings = findingRows(analysis, options);
  const identity = {
    filename: text(analysis.filename || options.filename || "model"),
    format,
    sha256: sha256(options.hash || analysis.model_sha256),
    hash_basis: text(analysis?.artifact_bundle?.hash_basis || "artifact_file_bytes_sha256"),
    byte_length: finite(options.fileSizeBytes ?? analysis.file_size_bytes ?? analysis.file_size),
    schema_or_opset: text(options.schemaOrOpset || analysis.schema_or_opset) || null,
  };
  const body = {
    schema: ARTIFACT_EVIDENCE_SCHEMA,
    generated_at: text(options.generatedAt) || null,
    identity,
    artifact_set: analysis?.artifact_set || null,
    cpu_cost_target_binding: analysis?.cpu_cost_target_binding || null,
    accelerator_profile_binding: analysis?.accelerator_profile_binding || null,
    accelerator_bindings: collectAcceleratorBindings(
      analysis,
      options.runtimeEvidence || options.runtimeAssignmentEvidence || null,
      identity.sha256,
    ),
    policy_identity: analysis?.policy_identity || null,
    capabilities: capabilityManifest(analysis, format),
    interfaces,
    graph: {
      operator_count: ["gguf", "safetensors"].includes(format) ? null : finite(analysis.operator_count ?? analysis.ops?.length),
      tensor_count: finite(analysis.tensor_count ?? analysis.tensors?.length),
      total_macs: finite(analysis.total_macs),
      mac_assessment_status: macCoverage.status,
      mac_coverage: macCoverage,
    },
    external_files: files,
    metadata: observedMetadata(analysis),
    findings,
    format_extensions: { [format]: formatExtensionSummary(analysis, format) },
    provenance: options.provenance || null,
    evidence_boundary: "Serialized artifact facts and deterministic static derivations. Declarations and runtime measurements are accepted only through separately identified inputs.",
  };
  const normalizedBody = normalizeJsonContractValue(body);
  return { ...normalizedBody, envelope_sha256: sha256TextHex(canonicalJson(normalizedBody)) };
}

export function validateArtifactEvidenceEnvelope(envelope) {
  const errors = [];
  if (envelope?.schema !== ARTIFACT_EVIDENCE_SCHEMA) errors.push("schema_mismatch");
  if (!text(envelope?.identity?.format) || envelope.identity.format === "unknown") errors.push("format_unbound");
  if (envelope?.identity?.sha256 && !sha256(envelope.identity.sha256)) errors.push("invalid_artifact_sha256");
  if (envelope?.artifact_set) {
    try { validateArtifactSet(envelope.artifact_set); }
    catch { errors.push("invalid_artifact_set"); }
  }
  if (envelope?.cpu_cost_target_binding) {
    try { validateCpuCostTargetBinding(envelope.cpu_cost_target_binding); }
    catch { errors.push("invalid_cpu_cost_target_binding"); }
  }
  if (envelope?.accelerator_profile_binding) {
    try { validateNvidiaAcceleratorProfileBinding(envelope.accelerator_profile_binding); }
    catch { errors.push("invalid_accelerator_profile_binding"); }
  }
  if (!Array.isArray(envelope?.accelerator_bindings)) errors.push("accelerator_bindings_missing");
  for (const binding of envelope?.accelerator_bindings || []) {
    try { validateAcceleratorBinding(binding); }
    catch { errors.push(`invalid_accelerator_binding:${text(binding?.profile_id) || "unknown"}`); }
  }
  if (envelope?.policy_identity && !validPolicyIdentity(envelope.policy_identity)) errors.push("invalid_policy_identity");
  if (!Array.isArray(envelope?.interfaces?.parameters)) errors.push("interface_ledger_missing");
  for (const file of envelope?.external_files || []) {
    if (!sha256(file.sha256)) errors.push(`invalid_external_file_sha256:${file.path}`);
    if (!text(file.path)) errors.push("external_file_path_missing");
  }
  for (const finding of envelope?.findings || []) {
    if (!EVIDENCE_CLASSES.includes(finding.evidence_class)) errors.push(`invalid_evidence_class:${finding.id}`);
  }
  const expected = { ...envelope };
  delete expected.envelope_sha256;
  if (envelope?.envelope_sha256 !== sha256TextHex(canonicalJson(expected))) errors.push("envelope_sha256_mismatch");
  return { valid: errors.length === 0, errors };
}

function validPolicyIdentity(value) {
  return value?.schema === "deepbom.review_policy.v1"
    && sha256(value?.policy_sha256)
    && sha256(value?.source_file_sha256)
    && ["observe", "enforce"].includes(value?.mode);
}
