import {
  modelQuantizationStatus,
  quantizationScopeExplanation,
} from "./analysis.js";
import { buildFindingsRegister } from "./report-findings.js";
import { collectArtifactIntegrity } from "./report-integrity.js";
import { ANALYZER_METADATA } from "./report-metadata.js";
import { buildInterfaceQuantizationContractLedger } from "./quantization-contract-summary.js";
import { sha256TextHex } from "./sha256-sync.js";
import { buildArtifactEvidenceEnvelope } from "./artifact-evidence-envelope.js";
import { compareInterfaceContracts } from "./interface-contract.js";
import { buildCycloneDx20ParameterContractPreview } from "./cyclonedx-20-preview.js";
import { coreMlFloorLabel } from "./coreml-deployment-contract.js";
import {
  analyzerContentVersion,
  analyzerBomRef as canonicalAnalyzerBomRef,
  artifactBomRef,
  artifactContentVersion,
  cycloneDxSerialNumber,
} from "./cyclonedx-identity.js";
import { artifactComponentMetadata } from "./cyclonedx-component-metadata.js";
import { safeTensorsQuantizationPropertyEntries } from "./safetensors-quantization-export.js";
import {
  interfaceCorpusValidationExternalReference,
  interfaceCorpusValidationProperties,
} from "./corpus-validation-provenance.js";
import {
  bindTfliteBuildRequirement,
  bindTfliteDelegateRequirement,
} from "./tflite-build-configuration-binding.js";
import { tensorRtCycloneDxPropertyEntries } from "./tensorrt-cyclonedx-properties.js";
import { resolveArtifactIrContext } from "./artifact-ir-context.js";

const CYCLONEDX_SCHEMA = "http://cyclonedx.org/schema/bom-1.7.schema.json";
const LITERT_INT8_SPEC = "https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/quantization_spec";
const DEFAULT_DOCUMENT_AUTHOR = Object.freeze({
  name: "Jun-Hwan Kwon",
  email: "",
  orcid: "https://orcid.org/0000-0002-6464-3895",
});
const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const ORCID_PATTERN = /^(?:https:\/\/orcid\.org\/)?(\d{4}-\d{4}-\d{4}-[\dX]{4})$/i;
const TFLITE_KERNEL_OPS = new Set(["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "TRANSPOSE_CONV"]);
const CURRENT_INT8_PROFILE_OPS = new Set(["CONV_2D", "DEPTHWISE_CONV_2D"]);

export const DEPLOYMENT_CONTRACT_FILES = Object.freeze({
  cyclonedx: "deepbom_cyclonedx_evidence.cdx.json",
  cyclonedx20Preview: "deepbom_cyclonedx_2_0_parameter_contract.preview.cdx.json",
  artifactEnvelope: "deepbom_artifact_evidence_envelope.json",
  artifactIr: "deepbom_artifact_ir.json",
  interfaceContracts: "deepbom_interface_contracts.json",
  formulation: "deepbom_observed_formulation.cdx.json",
  runtime: "deepbom_runtime_requirements.json",
  missingFields: "deepbom_missing_provenance_fields.json",
});

function text(value) {
  return String(value ?? "").trim();
}

function finite(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function integer(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isSafeInteger(number) ? number : null;
}

function present(value) {
  return value !== undefined && value !== null && !(typeof value === "string" && !value.trim());
}

function property(name, value) {
  if (!present(value)) return null;
  return { name, value: typeof value === "string" ? value : String(value) };
}

function properties(entries) {
  return entries.map(([name, value]) => property(name, value)).filter(Boolean);
}

function acceleratorBindingPropertyEntries(bindings) {
  const rows = Array.isArray(bindings) ? bindings : [];
  return [
    ["deepbom:accelerator:bindingCount", rows.length],
    ["deepbom:accelerator:bindingDetailPointer", "deepbom_artifact_evidence_envelope.json#/accelerator_bindings"],
  ];
}

function llmCycloneDxPropertyEntries(analysis) {
  const llm = analysis?.on_device_llm;
  if (llm?.schema !== "deepbom.on_device_llm_contract.v2") return [];
  const cliScenario = analysis?.llm_token_budget_scenario || analysis?.cli_context_scenario;
  return [
    ["deepbom:model:llmContractSchema", llm.schema],
    ["deepbom:model:llmContractStatus", llm.status],
    ["deepbom:model:llmEvidenceClass", llm.evidence_class],
    ["deepbom:model:llmArchitecture", llm.architecture?.family],
    ["deepbom:model:llmArchitectureKind", llm.architecture?.kind],
    ["deepbom:model:llmSerializedGraphStatus", llm.serialized_graph?.status],
    ["deepbom:model:llmSerializedGraphSignatureSha256", llm.serialized_graph?.graph_signature_sha256],
    ["deepbom:model:llmExplicitTransformerOperatorCount", llm.serialized_graph?.explicit_operator_count],
    ["deepbom:model:llmExternalStateCandidateCount", llm.serialized_graph?.external_state_candidate_count],
    ["deepbom:model:tensorRtLlmContractSchema", llm.tensorrt_llm?.status !== "not_selected" ? llm.tensorrt_llm?.schema : null],
    ["deepbom:model:tensorRtLlmContractStatus", llm.tensorrt_llm?.status !== "not_selected" ? llm.tensorrt_llm?.status : null],
    ["deepbom:model:tensorRtLlmContractSha256", llm.tensorrt_llm?.contract_sha256],
    ["deepbom:model:tensorRtLlmEngineConfigSha256", llm.tensorrt_llm?.engine_config?.sha256],
    ["deepbom:model:tensorRtLlmArtifactBindingStatus", llm.tensorrt_llm?.artifact_binding?.status],
    ["deepbom:model:tensorRtLlmSourceArtifactSha256", llm.tensorrt_llm?.artifact_binding?.source_artifact_sha256],
    ["deepbom:model:tensorRtLlmWorldSize", llm.tensorrt_llm?.parallelism?.world_size],
    ["deepbom:model:tensorRtLlmTensorParallelSize", llm.tensorrt_llm?.parallelism?.tensor_parallel_size],
    ["deepbom:model:tensorRtLlmPipelineParallelSize", llm.tensorrt_llm?.parallelism?.pipeline_parallel_size],
    ["deepbom:model:tensorRtLlmContextParallelSize", llm.tensorrt_llm?.parallelism?.context_parallel_size],
    ["deepbom:model:tensorRtLlmLogicalKvBytes", llm.tensorrt_llm?.kv_cache_scenario?.logical_bytes?.decimal],
    ["deepbom:model:llmContextLength", llm.architecture?.context_length],
    ["deepbom:model:llmVocabularySize", llm.architecture?.vocabulary_size],
    ["deepbom:model:llmSerializedParameterCount", llm.storage?.serialized_parameter_count_decimal],
    ["deepbom:model:llmSerializedTensorBytes", llm.storage?.serialized_tensor_bytes_decimal],
    ["deepbom:model:llmEncodingSignatureSchema", llm.storage?.encoding_signature_schema],
    ["deepbom:model:llmEncodingInventorySha256", llm.storage?.encoding_inventory_sha256],
    ["deepbom:model:llmTensorEncodingAssignmentSha256", llm.storage?.tensor_encoding_assignment_sha256],
    ["deepbom:model:llmLayerStorageSchema", llm.storage?.layer_storage?.schema],
    ["deepbom:model:llmLayerStorageStatus", llm.storage?.layer_storage?.status],
    ["deepbom:model:llmObservedLayerCount", llm.storage?.layer_storage?.observed_layer_count],
    ["deepbom:model:llmLayerSerializedBytes", llm.storage?.layer_storage?.layer_bytes?.decimal],
    ["deepbom:model:llmNonLayerSerializedBytes", llm.storage?.layer_storage?.non_layer_bytes?.decimal],
    ["deepbom:model:llmKvElementsPerTokenPerBatch", llm.state?.kv_projection?.elements_per_token_per_batch?.decimal],
    ["deepbom:model:llmRecurrentStateElementsPerBatch", llm.state?.recurrent_projection?.recurrent_state_elements_all_layers_per_batch?.decimal],
    ["deepbom:model:llmMoeExpertCount", llm.architecture?.moe?.expert_count],
    ["deepbom:model:llmMoeActiveExpertsPerToken", llm.architecture?.moe?.active_expert_count_per_token],
    ["deepbom:model:llmTokenizerStatus", llm.tokenizer?.status],
    ["deepbom:model:llmChatTemplateStatus", llm.tokenizer?.chat_template?.status],
    ["deepbom:model:llmChatTemplateSha256", llm.tokenizer?.chat_template?.sha256],
    ["deepbom:model:llmRuntimeBindingStatus", llm.runtime_contract?.status],
    ["deepbom:model:llmRuntimeManifestSchema", llm.runtime_contract?.source ? llm.runtime_contract.schema : null],
    ["deepbom:model:llmRuntimeManifestSha256", llm.runtime_contract?.source_sha256],
    ["deepbom:model:llmRuntimeEvidenceClass", llm.runtime_contract?.evidence_class],
    ["deepbom:model:llmRuntimeAcceleratorLayerCount", llm.runtime_contract?.layer_placement?.accelerator_layer_count],
    ["deepbom:model:llmRuntimeResidentStateBytes", llm.runtime_contract?.state_cache?.resident_bytes?.decimal],
    ["deepbom:model:llmRuntimeWorkingMemoryCoverage", llm.runtime_contract?.working_memory?.coverage],
    ["deepbom:model:llmRuntimeAccountedWorkingMemoryBytes", llm.runtime_contract?.working_memory?.accounted_nonweight_runtime_bytes?.decimal],
    ["deepbom:model:llmMemoryFeasibilitySchema", llm.memory_feasibility?.schema],
    ["deepbom:model:llmMemoryFeasibilityStatus", llm.memory_feasibility?.status],
    ["deepbom:model:llmStaticMemoryCapacityScope", llm.memory_feasibility?.capacity_scope],
    ["deepbom:model:llmStaticMemoryResidencyAssumption", llm.memory_feasibility?.residency_assumption],
    ["deepbom:model:llmSerializedWeightFloorBytes", llm.memory_feasibility?.serialized_weight_floor_bytes?.decimal],
    ["deepbom:model:llmMinimumStaticMemoryLowerBoundBytes", llm.memory_feasibility?.minimum_static_lower_bound_bytes?.decimal],
    ["deepbom:model:llmMaximumStaticMemoryLowerBoundBytes", llm.memory_feasibility?.maximum_static_lower_bound_bytes?.decimal],
    ["deepbom:model:llmRuntimePrimaryResidentLowerBoundBytes", llm.memory_feasibility?.runtime_primary_residency?.primary_resident_lower_bound_bytes?.decimal],
    ["deepbom:model:llmRuntimePrimaryAllocatedLowerBoundBytes", llm.memory_feasibility?.runtime_primary_residency?.primary_allocated_lower_bound_bytes?.decimal],
    ["deepbom:model:llmRuntimePrimaryAllocatedAccountedBytes", llm.memory_feasibility?.runtime_primary_residency?.primary_allocated_accounted_bytes?.decimal],
    ["deepbom:model:llmRuntimeAllocationAccountingStatus", llm.memory_feasibility?.runtime_primary_residency?.allocation_accounting_status],
    ["deepbom:model:llmMemoryFitClaim", llm.memory_feasibility?.fit_claim],
    ["deepbom:model:llmStaticPoolPlacementSchema", llm.static_memory_placement?.schema],
    ["deepbom:model:llmStaticPoolPlacementStatus", llm.static_memory_placement?.status],
    ["deepbom:model:llmStaticPoolProfileSha256", llm.static_memory_placement?.normalized_profile_sha256],
    ["deepbom:model:llmStaticPoolCandidateCount", llm.static_memory_placement?.candidate_count],
    ["deepbom:model:llmStaticPoolNotDisprovenCandidateCount", llm.static_memory_placement?.lower_bound_not_exceeding_candidate_count],
    ["deepbom:model:llmStaticPoolMaximumAcceleratorLayersNotDisproven", llm.static_memory_placement?.maximum_accelerator_layer_count_not_disproven],
    ["deepbom:model:llmStaticPoolFitClaim", llm.static_memory_placement?.fit_claim],
    ["deepbom:model:llmCliScenarioSchema", cliScenario?.schema],
    ["deepbom:model:llmCliScenarioContextLength", cliScenario?.context_length],
    ["deepbom:model:llmCliScenarioTextTokenCount", cliScenario?.token_budget?.text_tokens ?? cliScenario?.text_context_length],
    ["deepbom:model:llmCliScenarioImageCount", cliScenario?.token_budget?.image_count ?? cliScenario?.image_count],
    ["deepbom:model:llmCliScenarioTokensPerImage", cliScenario?.token_budget?.tokens_per_image ?? cliScenario?.tokens_per_image],
    ["deepbom:model:llmCliScenarioImageTokenCount", cliScenario?.token_budget?.image_tokens?.decimal ?? cliScenario?.image_token_count],
    ["deepbom:model:llmCliScenarioTotalContextTokens", cliScenario?.token_budget?.total_context_tokens?.decimal ?? cliScenario?.context_length],
    ["deepbom:model:llmCliScenarioContextAssessment", cliScenario?.serialized_context_contract?.assessment],
    ["deepbom:model:llmCliScenarioSha256", cliScenario?.scenario_sha256],
    ["deepbom:model:llmCliScenarioBatchSize", cliScenario?.batch_size],
    ["deepbom:model:llmCliScenarioStateStorageBits", cliScenario?.state_storage_bits],
    ["deepbom:model:llmCliScenarioMemoryStatus", cliScenario?.memory_feasibility?.status],
    ["deepbom:model:llmCliScenarioResidencyAssumption", cliScenario?.memory_feasibility?.residency_assumption],
    ["deepbom:model:llmCliScenarioStaticLowerBoundBytes", cliScenario?.memory_feasibility?.static_lower_bound_bytes?.decimal],
    ["deepbom:model:llmCliScenarioDeclaredCapacityBytes", cliScenario?.memory_feasibility?.declared_capacity_bytes?.decimal],
    ["deepbom:model:llmCliScenarioMemoryFitClaim", cliScenario?.memory_feasibility?.fit_claim],
    ["deepbom:model:medicalAiClaimBoundary", llm.medical_ai_claim_boundary?.status],
    ["deepbom:model:medicalAiDeclarationSha256", llm.medical_ai_claim_boundary?.declaration?.sha256],
    ["deepbom:model:medicalAiDeclarationCoverage", llm.medical_ai_claim_boundary?.declaration?.coverage ? `${llm.medical_ai_claim_boundary.declaration.coverage.declared}/${llm.medical_ai_claim_boundary.declaration.coverage.required}` : null],
    ["deepbom:model:llmEvidencePointer", "/evidence/static_analysis/on_device_llm"],
  ];
}

function firstValue(source, paths) {
  for (const path of paths) {
    let value = source;
    for (const part of path.split(".")) value = value?.[part];
    if (present(value)) return value;
  }
  return null;
}

function propertyMap(items) {
  return new Map((items || []).map((item) => [text(item?.name), text(item?.value)]));
}

function normalizedSha256(value) {
  const digest = text(value).toLowerCase();
  return SHA256_PATTERN.test(digest) ? digest : "";
}

function normalizedOrcid(value) {
  const match = text(value).match(ORCID_PATTERN);
  return match ? `https://orcid.org/${match[1].toUpperCase()}` : "";
}

function artifactSha256(analysis, mlBomDocument, hash) {
  return [
    hash,
    analysis?.model_sha256,
    mlBomDocument?.metadata?.component?.hashes?.find((item) => item?.alg === "SHA-256")?.content,
  ].map(normalizedSha256).find(Boolean) || "";
}

function declaredModelVersion(analysis, component) {
  const candidate = text(
    analysis?.metadata_presence?.metadata_model_version
      || analysis?.metadata_presence?.model_version
      || component?.version,
  );
  return candidate && !/^(unknown|unbound|not[_ -]?declared|n\/a)$/i.test(candidate) ? candidate : null;
}

function artifactIdentity(analysis, options = {}) {
  const mlBom = options.mlBomDocument || {};
  const component = mlBom.metadata?.component || {};
  const sha256 = artifactSha256(analysis, mlBom, options.hash);
  const byteLength = finite(options.fileSizeBytes ?? analysis?.file_size_bytes);
  const integrity = collectArtifactIntegrity(analysis, { sha256 }, byteLength || 0);
  const suppliedSchemaOrOpset = propertyMap(component.properties).get("mlbom:model:schemaOrOpset") || null;
  const schemaOrOpset = integrity.schema_or_opset && !/\bunknown\b/i.test(integrity.schema_or_opset)
    ? integrity.schema_or_opset
    : suppliedSchemaOrOpset || integrity.schema_or_opset || null;
  const componentBomRef = sha256
    ? artifactBomRef(sha256, analysis?.filename)
    : text(component["bom-ref"]) || artifactBomRef("", analysis?.filename || "unbound");
  return {
    name: text(analysis?.filename || component.name || "model"),
    format: text(analysis?.format || "unknown").toLowerCase(),
    sha256,
    hash_basis: text(analysis?.artifact_bundle?.hash_basis || "artifact_file_bytes_sha256"),
    byte_length: byteLength != null && byteLength >= 0 ? byteLength : null,
    schema_or_opset: schemaOrOpset,
    declared_version: declaredModelVersion(analysis, component),
    component_bom_ref: componentBomRef,
  };
}

function tensorContract(parameter) {
  const quantization = parameter.quantization;
  return {
    tensor_index: parameter.tensor_index,
    name: parameter.tensor_name,
    dtype: parameter.dtype,
    shape: parameter.shape,
    status: quantization.status,
    scheme: quantization.scheme,
    granularity: quantization.granularity,
    quantization: quantization.status === "not_quantized" ? "none" : quantization.granularity.replaceAll("_", "-"),
    quantized_dimension: quantization.axis,
    serialized_quantized_dimension: quantization.quantized_dimension,
    scale_count: quantization.scale_count,
    scale_values_complete: quantization.scale_values_complete,
    scales: quantization.scales,
    zero_point_count: quantization.zero_point_count,
    zero_point_values_complete: quantization.zero_point_values_complete,
    zero_points: quantization.zero_points,
    cardinality_status: quantization.cardinality_status,
    cardinality_reason: quantization.cardinality_reason,
    scalar_real_code_domain: quantization.scalar_real_code_domain,
    affine_contract_sha256: quantization.contract_sha256,
    serialized_quantization_sha256: quantization.serialized_quantization_sha256,
    interface_contract_sha256: parameter.interface_contract_sha256,
  };
}

function modelCardIo(contract) {
  const shape = contract.shape.length ? `[${contract.shape.join(",")}]` : "shape=unbound";
  const quant = contract.quantization === "none"
    ? "unquantized"
    : `${contract.quantization}; scale_count=${contract.scale_count}; zero_point_count=${contract.zero_point_count}${contract.scales?.length === 1 ? `; scale=${contract.scales[0]}` : ""}${contract.zero_points?.length === 1 ? `; zero_point=${contract.zero_points[0]}` : ""}`;
  return { format: `${contract.dtype} ${shape}; ${quant}` };
}

function runtimeFloor(analysis) {
  const format = text(analysis?.format || "tflite").toLowerCase();
  if (format === "onnx") {
    const floor = analysis?.ort_compatibility_evidence?.runtime_floor || {};
    return {
      runtime: "ONNX Runtime",
      declared_minimum_version: null,
      derived_minimum_version: text(floor.minimum_ort_version || analysis?.runtime_compat?.derived_min_runtime_version) || null,
      minimum_version: text(floor.minimum_ort_version || analysis?.runtime_compat?.effective_min_runtime_version) || null,
      status: text(floor.status || analysis?.runtime_compat?.runtime_floor_status || "not_assessed"),
      evidence_class: text(floor.evidence_class || analysis?.runtime_compat?.runtime_floor_evidence_class || "NOT_ASSESSED"),
      basis: floor.basis || analysis?.runtime_compat?.runtime_version_basis || null,
      standard_component_minimum_version: text(floor.standard_minimum_ort_version) || null,
      contrib_component_minimum_version: text(floor.contrib_minimum_ort_version) || null,
      source_backed_external_domains: [...(floor.source_backed_external_domains || [])],
      unresolved_domains: [...(floor.unresolved_domains || [])],
      contrib_operator_floors: (floor.contrib_operator_floors || []).map((row) => ({
        domain: row.domain,
        op_name: row.op_name,
        imported_opset: row.imported_opset,
        minimum_ort_version: row.minimum_ort_version,
        source_ref: row.source_ref,
        source_sha256: row.source_sha256,
        evidence_class: row.evidence_class,
      })),
    };
  }
  if (format === "coreml") {
    const floor = analysis?.coreml?.deployment_floor || null;
    const specificationVersion = integer(floor?.declared_specification_version ?? analysis?.coreml?.specification_version);
    return {
      runtime: "Core ML",
      declared_minimum_version: specificationVersion == null ? null : `specification ${specificationVersion}`,
      derived_minimum_version: floor?.observed_feature_minimum_specification_version == null ? null : `specification ${floor.observed_feature_minimum_specification_version}`,
      minimum_version: floor?.declared_load_floor ? coreMlFloorLabel(floor.declared_load_floor) : null,
      status: floor?.status || "not_assessed_os_runtime_floor",
      evidence_class: floor?.evidence_class || "NOT_ASSESSED",
      basis: floor?.declared_load_floor
        ? `Pinned Core ML Model.proto maps serialized specification ${specificationVersion} to ${coreMlFloorLabel(floor.declared_load_floor)}. Observed artifact features independently require specification ${floor.observed_feature_minimum_specification_version}; the declared version remains the load floor.`
        : "Core ML specification version is absent from the pinned OS availability table; no OS floor is inferred.",
    };
  }
  if (format === "gguf") {
    return {
      runtime: "GGUF-compatible runtime (unbound)",
      declared_minimum_version: null,
      derived_minimum_version: null,
      minimum_version: null,
      status: "not_assessed_runtime_implementation_unbound",
      evidence_class: "NOT_ASSESSED",
      basis: `GGUF v${analysis?.gguf?.version || "unknown"} is observed, but a loader/runtime implementation and build are not selected by the artifact.`,
    };
  }
  if (format === "safetensors") {
    return {
      runtime: "SafeTensors-compatible loader (unbound)",
      declared_minimum_version: null,
      derived_minimum_version: null,
      minimum_version: null,
      status: "not_assessed_runtime_implementation_unbound",
      evidence_class: "NOT_ASSESSED",
      basis: "SafeTensors is a weight container; the execution framework, loader version, graph, and runtime are not selected by the artifact.",
    };
  }
  if (format === "executorch") {
    return {
      runtime: "ExecuTorch runtime (build unbound)",
      declared_minimum_version: `serialized ${String(analysis?.executorch_container || "artifact").toUpperCase()} schema ${analysis?.version ?? "unknown"}`,
      derived_minimum_version: null,
      minimum_version: null,
      status: "not_assessed_runtime_build_unbound",
      evidence_class: "SOURCE_PINNED_ARTIFACT_OBSERVED",
      basis: analysis?.runtime_compat?.runtime_version_basis || "ET12/FT01 wire identity is observed, but no schema-version-to-runtime-release floor matrix is asserted.",
    };
  }
  if (format !== "tflite") {
    return {
      runtime: `${format || "unknown"} runtime (unbound)`,
      declared_minimum_version: null,
      derived_minimum_version: null,
      minimum_version: null,
      status: "not_assessed_runtime_implementation_unbound",
      evidence_class: "NOT_ASSESSED",
      basis: "No format-specific runtime-floor rule is implemented for this artifact.",
    };
  }
  const floor = analysis?.runtime_compat || {};
  return {
    runtime: "TensorFlow Lite / LiteRT",
    declared_minimum_version: text(floor.min_runtime_version) || null,
    derived_minimum_version: text(floor.derived_min_runtime_version) || null,
    minimum_version: text(floor.effective_min_runtime_version || floor.derived_min_runtime_version || floor.min_runtime_version) || null,
    status: text(floor.runtime_floor_status || (floor.effective_min_runtime_version ? "assessed" : "not_assessed")),
    evidence_class: text(floor.runtime_floor_evidence_class || "DERIVED"),
    basis: floor.runtime_version_basis || null,
  };
}

function quantizationSummary(analysis) {
  const status = modelQuantizationStatus(analysis);
  const scope = quantizationScopeExplanation(analysis);
  const declaredTensorTypes = Array.isArray(analysis?.tensor_types) ? analysis.tensor_types : [];
  const tensorTypeCounts = declaredTensorTypes.length
    ? declaredTensorTypes
    : Object.entries((analysis?.tensors || []).reduce((counts, tensor) => {
      const dtype = text(tensor?.dtype || "UNKNOWN").toUpperCase();
      counts[dtype] = Number(counts[dtype] || 0) + 1;
      return counts;
    }, {})).map(([name, count]) => ({ name, count }));
  const quantizedTensorCount = Math.max(
    Number(analysis?.quantized_tensors || 0),
    Number(status.block_quantized_tensor_count || 0),
  );
  return {
    classification: status.classification || "unknown",
    label: status.label || "Unknown",
    full_integer: Boolean(status.full_integer),
    classification_basis: "External I/O dtype, complete internal tensor dtype inventory, every MAC-bearing compute op's activation dtype path, and serialized Q/DQ inventory.",
    tensor_dtype_inventory: Object.fromEntries(tensorTypeCounts.map((row) => [String(row.name || "UNKNOWN"), Number(row.count || 0)])),
    int8_tensor_count: Number(status.int8_tensors || 0),
    uint8_tensor_count: Number(status.uint8_tensors || 0),
    float_tensor_count: Number(status.float_tensors || 0),
    serialized_quantize_operator_count: Number(status.quantize_ops || 0),
    serialized_dequantize_operator_count: Number(status.dequantize_ops || 0),
    quantized_tensor_count: quantizedTensorCount,
    block_quantized_tensor_count: Number(status.block_quantized_tensor_count || 0),
    scalar_encoded_tensor_count: Number(status.scalar_encoded_tensor_count || 0),
    per_axis_tensor_count: Number(analysis?.per_channel_tensors || 0),
    quantized_compute_mac_ratio: status.quantized_compute_mac_percent == null
      ? null
      : Number(status.quantized_compute_mac_percent),
    activation_path: {
      inventory_basis: "all_graph_operators",
      all_operator_denominator: Number(scope.all_ops_denominator || 0),
      states: (scope.op_state_counts || []).map((item) => ({
        state: item.name,
        operator_count: Number(item.count || 0),
      })),
      compute_scope: {
        basis: "MAC-bearing compute operators",
        denominator: Number(scope.compute_ops_denominator || 0),
        quantized_operators: Number(scope.quantized_compute_ops || 0),
        quantized_mac_ratio: scope.quantized_compute_mac_percent == null
          ? null
          : Number(scope.quantized_compute_mac_percent),
      },
      explanation: scope.explanation,
    },
  };
}

function analyzerMetadata(options = {}) {
  return options.analyzerMetadata && typeof options.analyzerMetadata === "object"
    ? { ...ANALYZER_METADATA, ...options.analyzerMetadata }
    : ANALYZER_METADATA;
}

function analyzerProvenance(options = {}) {
  const metadata = analyzerMetadata(options);
  return {
    analyzer: metadata.name,
    analyzer_semantic_version: metadata.semanticVersion,
    analyzer_build_version: metadata.version,
    analyzer_build_commit: metadata.buildCommit,
    analyzer_build_source_state: metadata.buildSourceState,
    analyzer_bundle_content_sha256: normalizedSha256(metadata.buildContentSha256) || null,
    analyzer_bundle_content_hash_method: metadata.buildContentHashMethod,
    rulepack_version: metadata.rulepackVersion,
    rulepack_sha256: normalizedSha256(metadata.rulepackSha256) || null,
    rulepack_hash_basis: metadata.rulepackHashBasis,
  };
}

function toolMetadata(options = {}) {
  const metadata = analyzerMetadata(options);
  const provenance = analyzerProvenance(options);
  return {
    components: [{
      type: "application",
      name: metadata.name,
      version: analyzerContentVersion(metadata.semanticVersion, metadata.buildCommit, metadata.buildContentSha256),
      "bom-ref": analyzerBomRef(options),
      ...(provenance.analyzer_bundle_content_sha256
        ? { hashes: [{ alg: "SHA-256", content: provenance.analyzer_bundle_content_sha256 }] }
        : {}),
      externalReferences: [interfaceCorpusValidationExternalReference()],
      properties: properties([
        ["deepbom:analyzer:buildVersion", provenance.analyzer_build_version],
        ["deepbom:analyzer:semanticVersion", provenance.analyzer_semantic_version],
        ["deepbom:analyzer:buildCommit", provenance.analyzer_build_commit],
        ["deepbom:analyzer:buildSourceState", provenance.analyzer_build_source_state],
        ["deepbom:analyzer:bundleContentHashMethod", provenance.analyzer_bundle_content_hash_method],
        ["deepbom:analyzer:rulepackVersion", provenance.rulepack_version],
        ["deepbom:analyzer:rulepackSha256", provenance.rulepack_sha256],
        ...interfaceCorpusValidationProperties(),
      ]),
    }],
  };
}

function analyzerBomRef(options = {}) {
  const metadata = analyzerMetadata(options);
  return canonicalAnalyzerBomRef(metadata.name, metadata.semanticVersion, metadata.buildCommit);
}

function findingsSummary(analysis, options = {}) {
  let findings = Array.isArray(options.findings) ? options.findings : null;
  if (!findings) {
    try {
      findings = buildFindingsRegister(analysis, {
        runtimeEvidence: options.runtimeEvidence || options.runtimeAssignmentEvidence || null,
        analyzerMetadata: options.analyzerMetadata,
      });
    } catch {
      findings = null;
    }
  }
  const details = analysis?.weight_integrity?.zero_kernel_slice_details || [];
  const maxInt32Utilization = details
    .flatMap((item) => item.bias_int32_utilization_sample || [])
    .map(finite)
    .filter((value) => value != null)
    .reduce((maximum, value) => Math.max(maximum, value), -Infinity);
  const vitality = analysis?.channel_vitality || {};
  return {
    status: findings ? "assessed" : "not_assessed",
    finding_count: findings?.length ?? null,
    high_severity_count: findings
      ? findings.filter((item) => text(item?.technical_priority).toLowerCase() === "high").length
      : null,
    high_severity_findings: findings
      ? findings
          .filter((item) => text(item?.technical_priority).toLowerCase() === "high")
          .map((item) => ({
            finding_id: item.finding_id,
            title: item.title,
            category: item.category,
            evidence_class: item.evidence_class,
          }))
      : null,
    constant_output_channels: integer(vitality.dual_mode_constant_output_channel_count),
    exact_zero_kernel_slices: integer(analysis?.weight_integrity?.exact_zero_kernel_slice_count),
    build_mode_dependent_channels: integer(vitality.mode_dependent_constant_output_channel_count),
    max_int32_utilization_ratio: Number.isFinite(maxInt32Utilization) ? maxInt32Utilization : null,
  };
}

function kernelBindings(analysis) {
  const tensorByIndex = new Map((analysis?.tensors || []).map((tensor) => [Number(tensor.index), tensor]));
  return (analysis?.ops || []).flatMap((op) => {
    const name = text(op?.name).toUpperCase();
    if (!TFLITE_KERNEL_OPS.has(name)) return [];
    const weightIndex = integer(op?.inputs?.[1]);
    return [{
      op_index: integer(op?.index),
      op_name: name,
      weight_tensor_index: weightIndex,
      weight: weightIndex == null ? null : tensorByIndex.get(weightIndex) || null,
    }];
  });
}

function quantizationFingerprint(analysis) {
  const bindings = kernelBindings(analysis);
  const legacyUint8AsymmetricKernelTensorIds = new Set();
  let perTensorDepthwiseOps = 0;
  let perAxisKernelTensors = 0;
  const perAxisIds = new Set();
  for (const binding of bindings) {
    const weight = binding.weight;
    if (!weight) continue;
    const scales = Number(weight.quant_scales || weight.scale_sample?.length || 0);
    const zeroPoints = weight.zero_point_sample || [];
    if (text(weight.dtype).toUpperCase() === "UINT8" && zeroPoints.some((value) => Number(value) !== 0)) {
      legacyUint8AsymmetricKernelTensorIds.add(Number(weight.index));
    }
    if (binding.op_name === "DEPTHWISE_CONV_2D" && scales === 1) perTensorDepthwiseOps += 1;
    if (scales > 1) perAxisIds.add(Number(weight.index));
  }
  perAxisKernelTensors = perAxisIds.size;
  return {
    asymmetric_uint8_kernel_tensor_count: legacyUint8AsymmetricKernelTensorIds.size,
    per_tensor_depthwise_op_count: perTensorDepthwiseOps,
    per_axis_kernel_tensor_count: perAxisKernelTensors,
    conversion_metadata_status: text(analysis?.metadata_presence?.conversion_metadata_status || "not_present"),
    artifact_description: text(analysis?.metadata_presence?.description) || null,
    maximum_observed_op_version: integer(analysis?.runtime_compat?.max_op_version),
    derived_runtime_floor: text(analysis?.runtime_compat?.derived_min_runtime_version) || null,
  };
}

function assessLiteRtInt8Conformance(analysis) {
  if (text(analysis?.format || "tflite").toLowerCase() !== "tflite") {
    return {
      profile: "litert_current_int8_operator_profile",
      status: "not_applicable",
      evidence_class: "NOT_APPLICABLE",
      assessed_operator_count: 0,
      violation_count: 0,
      violation_codes: [],
    };
  }
  const bindings = kernelBindings(analysis).filter((item) => CURRENT_INT8_PROFILE_OPS.has(item.op_name));
  if (!bindings.length) {
    return {
      profile: "litert_current_int8_operator_profile",
      status: "not_applicable_no_conv_family_operator",
      evidence_class: "NOT_APPLICABLE",
      assessed_operator_count: 0,
      violation_count: 0,
      violation_codes: [],
      source: LITERT_INT8_SPEC,
    };
  }
  const violations = [];
  let assessed = 0;
  let unassessed = 0;
  for (const binding of bindings) {
    const weight = binding.weight;
    if (!weight) {
      unassessed += 1;
      continue;
    }
    assessed += 1;
    const dtype = text(weight.dtype).toUpperCase();
    const scaleCount = Number(weight.quant_scales || weight.scale_sample?.length || 0);
    const zeroPointCount = Number(weight.quant_zero_points || weight.zero_point_sample?.length || 0);
    const zeroPoints = (weight.zero_point_sample || []).map(Number);
    const prefix = { op_index: binding.op_index, op_name: binding.op_name, weight_tensor_index: binding.weight_tensor_index };
    if (dtype !== "INT8") {
      violations.push({ ...prefix, code: "weight_dtype_expected_int8", observed: dtype || "UNKNOWN", expected: "INT8" });
    }
    if (scaleCount <= 1) {
      violations.push({ ...prefix, code: "weight_granularity_per_tensor_expected_per_axis", observed_scale_count: scaleCount, expected: "per-axis" });
    }
    if (!zeroPointCount || zeroPoints.length !== zeroPointCount) {
      violations.push({ ...prefix, code: "weight_zero_point_vector_not_fully_observed", observed_count: zeroPoints.length, declared_count: zeroPointCount });
    } else if (zeroPoints.some((value) => value !== 0)) {
      violations.push({
        ...prefix,
        code: "weight_zero_point_nonzero_expected_zero",
        observed_min: Math.min(...zeroPoints),
        observed_max: Math.max(...zeroPoints),
        expected: 0,
      });
    }
  }
  const violationCodes = [...new Set(violations.map((item) => item.code))].sort();
  return {
    profile: "litert_current_int8_operator_profile",
    profile_scope: "Current LiteRT INT8 CONV_2D and DEPTHWISE_CONV_2D weight contracts",
    status: violations.length
      ? "nonconformant"
      : unassessed
        ? "partial"
        : "conformant_for_assessed_operators",
    evidence_class: violations.length ? "OBSERVED_ARTIFACT_CONTRACT_VIOLATION" : unassessed ? "PARTIAL" : "DERIVED",
    candidate_operator_count: bindings.length,
    assessed_operator_count: assessed,
    unassessed_operator_count: unassessed,
    violation_count: violations.length,
    violation_codes: violationCodes,
    violations,
    legacy_qu8_artifact: violations.some((item) => item.code === "weight_dtype_expected_int8" && item.observed === "UINT8"),
    source: LITERT_INT8_SPEC,
    interpretation_boundary: "A legacy UINT8 asymmetric artifact can remain executable through a separately enabled legacy QU8 runtime path while being nonconformant to the current LiteRT INT8 operator profile.",
  };
}

function preprocessingContract(analysis) {
  const evidence = analysis?.preprocessing_realizability;
  if (!evidence) {
    return {
      status: "not_assessed",
      evidence_class: "NOT_ASSESSED",
      exact_contract_ids: [],
      candidate_count: 0,
    };
  }
  return {
    status: text(evidence.status || "not_assessed"),
    evidence_class: text(evidence.evidence_class || "DERIVED"),
    assessment_kind: text(evidence.assessment_kind) || null,
    candidate_count: Number(evidence.candidate_evaluation_count || 0),
    exact_contract_ids: [...new Set(evidence.exact_contract_ids || [])],
    exact_candidate_count: Number(evidence.exact_tensor_realization_candidate_count || 0),
    non_exact_candidate_count: Number(evidence.non_exact_candidate_count || 0),
    best_non_exact_contract_id: text(evidence.best_non_exact_contract_id) || null,
    best_non_exact_unrealizable_element_count: integer(evidence.best_non_exact_unrealizable_element_count),
    portfolio_ledger_sha256: normalizedSha256(evidence.portfolio_ledger_sha256) || null,
    interpretation_boundary: evidence.interpretation_boundary || null,
  };
}

function externalReference(type, url, comment, sha256 = "") {
  const digest = normalizedSha256(sha256);
  return {
    type,
    url,
    ...(comment ? { comment } : {}),
    ...(digest ? { hashes: [{ alg: "SHA-256", content: digest }] } : {}),
  };
}

function siblingHash(options, filename) {
  return normalizedSha256(options?.externalDocumentHashes?.[filename]);
}

function bundleReferenceComment(description, hashBound) {
  return `${description} Relative URL resolves inside the Deployment Contract Pack.${hashBound
    ? " SHA-256 binds the downloaded JSON member bytes."
    : " Integrity is bound by deepbom_contract_pack_manifest.json; no circular sibling hash is embedded here."}`;
}

function optionalExternalReference(options, key, type, fallbackUrl, comment) {
  const url = text(options?.[`${key}Url`]);
  const hash = options?.[`${key}Sha256`];
  return url || hash ? [externalReference(type, url || fallbackUrl, comment, hash)] : [];
}

function supportingComponents(envelope) {
  return (envelope?.external_files || []).map((file) => ({
    type: "file",
    name: file.path,
    "bom-ref": `deepbom-file-sha256-${file.sha256}-${sha256TextHex(file.path).slice(0, 12)}`,
    scope: file.required ? "required" : "optional",
    hashes: [
      { alg: "SHA-256", content: file.sha256 },
      ...(file.sha1 ? [{ alg: "SHA-1", content: file.sha1 }] : []),
    ],
    properties: properties([
      ["deepbom:file:role", file.role],
      ["deepbom:file:byteLength", file.byte_length],
      ["deepbom:file:required", file.required],
      ["deepbom:file:evidenceClass", file.evidence_class],
      ["deepbom:file:verificationStatus", file.verification_status],
    ]),
  }));
}

function dependencyGraph(modelBomRef, components) {
  return [
    { ref: modelBomRef, dependsOn: components.map((component) => component["bom-ref"]) },
    ...components.map((component) => ({ ref: component["bom-ref"], dependsOn: [] })),
  ];
}

function evidenceDeclarations(envelope, identity, generatedAt, options = {}) {
  const metadata = analyzerMetadata(options);
  const evidenceRef = `${identity.component_bom_ref}:evidence:artifact-envelope`;
  const claims = (envelope?.findings || []).map((finding) => ({
    "bom-ref": `${identity.component_bom_ref}:claim:${encodeURIComponent(finding.id)}`,
    target: identity.component_bom_ref,
    predicate: `${finding.title}: ${finding.summary || "See bound evidence."}`,
    reasoning: `Evidence class ${finding.evidence_class}; source class ${finding.source_evidence_class || "not declared"}. ${finding.interpretation || ""}`.trim(),
    evidence: [evidenceRef],
  }));
  return {
    claims,
    evidence: [{
      "bom-ref": evidenceRef,
      description: "Hash-bound artifact evidence envelope produced by deterministic static analysis.",
      data: [{
        name: "Artifact Evidence Envelope",
        contents: { url: DEPLOYMENT_CONTRACT_FILES.artifactEnvelope },
      }],
      created: generatedAt,
      author: { name: metadata.name },
    }],
  };
}

function evidenceCitations(generatedAt, options = {}) {
  return [{
    "bom-ref": `deepbom-citation-${sha256TextHex(generatedAt).slice(0, 16)}`,
    pointers: ["/metadata/component", "/components", "/dependencies", "/declarations"],
    timestamp: generatedAt,
    attributedTo: analyzerBomRef(options),
    note: "Artifact identity, dependency edges, and static claims were projected by the referenced analyzer from the hash-bound evidence envelope.",
  }];
}

function cycloneDxSubject(analysis, options = {}, externalReferences = []) {
  const metadata = analyzerMetadata(options);
  const identity = artifactIdentity(analysis, options);
  const quantization = quantizationSummary(analysis);
  const floor = runtimeFloor(analysis);
  const interfaceLedger = options.artifactEvidenceEnvelope?.interfaces
    || buildInterfaceQuantizationContractLedger(analysis);
  const inputs = interfaceLedger.parameters.filter((row) => row.direction === "input").map(tensorContract);
  const outputs = interfaceLedger.parameters.filter((row) => row.direction === "output").map(tensorContract);
  const primaryInput = inputs[0] || {};
  const findings = findingsSummary(analysis, options);
  const conformance = assessLiteRtInt8Conformance(analysis);
  const preprocessing = preprocessingContract(analysis);
  const contentVersion = artifactContentVersion(identity.sha256, identity.declared_version);
  const llmContainer = ["gguf", "safetensors"].includes(identity.format) && analysis?.on_device_llm?.schema === "deepbom.on_device_llm_contract.v2";
  const quantizationEntries = llmContainer ? [
    ["deepbom:model:storageEncodingClassification", quantization.classification],
    ["deepbom:model:storageEncodingBasis", identity.format === "gguf"
      ? "Source-pinned GGML block encodings and exact serialized ranges; no execution graph, affine activation contract, or Q/DQ placement is inferred."
      : "SafeTensors dtype, shape, and exact serialized ranges; no execution graph, affine activation contract, or Q/DQ placement is inferred."],
  ] : [
    ["deepbom:model:quantizationClassification", quantization.classification],
    ["deepbom:model:activationPath", JSON.stringify(quantization.activation_path)],
    ["deepbom:model:fullIntegerQuantized", quantization.full_integer],
    ["deepbom:model:quantizationClassificationBasis", quantization.classification_basis],
    ["deepbom:model:int8TensorCount", quantization.int8_tensor_count],
    ["deepbom:model:uint8TensorCount", quantization.uint8_tensor_count],
    ["deepbom:model:floatTensorCount", quantization.float_tensor_count],
    ["deepbom:model:serializedQuantizeOperatorCount", quantization.serialized_quantize_operator_count],
    ["deepbom:model:serializedDequantizeOperatorCount", quantization.serialized_dequantize_operator_count],
    ["deepbom:model:perAxisTensorCount", quantization.per_axis_tensor_count],
    ["deepbom:model:perAxisQuantizationPresent", quantization.per_axis_tensor_count > 0],
  ];
  const component = {
    type: "machine-learning-model",
    name: identity.name,
    ...artifactComponentMetadata(analysis),
    ...(contentVersion.version ? { version: contentVersion.version } : {}),
    "bom-ref": identity.component_bom_ref,
    ...(identity.sha256 ? { hashes: [{ alg: "SHA-256", content: identity.sha256 }] } : {}),
    modelCard: {
      "bom-ref": `${identity.component_bom_ref}:model-card`,
      modelParameters: {
        inputs: inputs.map(modelCardIo),
        outputs: outputs.map(modelCardIo),
      },
      properties: properties([
        ["deepbom:modelCard:ioContractSchema", "deepbom.tensor_io_contract.v1"],
        ["deepbom:modelCard:interfaceContractLedger", DEPLOYMENT_CONTRACT_FILES.interfaceContracts],
        ["deepbom:modelCard:interfaceContractLedgerSha256", interfaceLedger.ledger_sha256],
      ]),
    },
    properties: properties([
      ["deepbom:contract:schema", "deepbom.cyclonedx_evidence_profile.v1.3"],
      ["deepbom:model:artifactIrSchema", options.artifactIr?.schema],
      ["deepbom:model:artifactIrSha256", options.artifactIr?.artifact_ir_sha256],
      ["deepbom:model:artifactIrLocation", options.artifactIrLocation],
      ["deepbom:model:format", identity.format],
      ["deepbom:model:versionBasis", contentVersion.basis],
      ["deepbom:model:schemaOrOpset", identity.schema_or_opset],
      ["deepbom:model:fileSizeBytes", identity.byte_length],
      ["deepbom:model:hashBasis", identity.hash_basis],
      ...quantizationEntries,
      ...safeTensorsQuantizationPropertyEntries(analysis),
      ["deepbom:model:tensorDtypeInventory", JSON.stringify(quantization.tensor_dtype_inventory)],
      ...llmCycloneDxPropertyEntries(analysis),
      ...tensorRtCycloneDxPropertyEntries(analysis),
      ["deepbom:model:completeAffineInterfaceCount", interfaceLedger.quantized_parameter_count],
      ["deepbom:model:unquantizedInterfaceCount", interfaceLedger.unquantized_parameter_count],
      ["deepbom:model:invalidOrIncompleteInterfaceCount", interfaceLedger.invalid_or_incomplete_parameter_count],
      ["deepbom:model:primaryInputScale", primaryInput.scales?.length === 1 ? primaryInput.scales[0] : null],
      ["deepbom:model:primaryInputZeroPoint", primaryInput.zero_points?.length === 1 ? primaryInput.zero_points[0] : null],
      ["deepbom:model:tfliteSparseStorageSchema", analysis?.tflite_sparse_storage_contract?.schema],
      ["deepbom:model:tfliteSparseTensorCount", analysis?.tflite_sparse_storage_contract?.sparse_tensor_count],
      ["deepbom:model:tfliteSparseLogicalElements", analysis?.tflite_sparse_storage_contract?.logical_element_count],
      ["deepbom:model:tfliteSparseStoredElements", analysis?.tflite_sparse_storage_contract?.stored_element_count],
      ["deepbom:model:tfliteSparseImplicitZeroElements", analysis?.tflite_sparse_storage_contract?.implicit_zero_element_count],
      ["deepbom:model:tfliteSparseSerializedValueBytes", analysis?.tflite_sparse_storage_contract?.serialized_value_bytes],
      ["deepbom:model:tfliteSubgraphInventorySchema", analysis?.tflite_subgraph_inventory?.schema],
      ["deepbom:model:tfliteSubgraphCount", analysis?.tflite_subgraph_inventory?.subgraph_count],
      ["deepbom:model:tfliteSerializedOperatorCount", analysis?.tflite_subgraph_inventory?.serialized_operator_count],
      ["deepbom:model:tflitePrimaryOperatorCount", analysis?.tflite_subgraph_inventory?.primary_operator_count],
      ["deepbom:model:tfliteControlFlowReferenceCount", analysis?.tflite_subgraph_inventory?.control_flow_reference_count],
      ["deepbom:model:tfliteControlFlowContractCount", analysis?.tflite_subgraph_inventory?.control_flow_contract_count],
      ["deepbom:model:tfliteControlFlowContractPartialCount", analysis?.tflite_subgraph_inventory?.partial_control_flow_contract_count],
      ["deepbom:model:tfliteSubgraphIntrinsicCostSchema", analysis?.tflite_subgraph_inventory?.rows?.[0]?.intrinsic_cost?.schema],
      ["deepbom:model:tfliteSubgraphIntrinsicAssessedCount", analysis?.tflite_subgraph_inventory?.rows?.filter?.((row) => String(row?.intrinsic_cost?.status || "").startsWith("assessed")).length],
      ["deepbom:model:tfliteSubgraphIntrinsicPartialCount", analysis?.tflite_subgraph_inventory?.rows?.filter?.((row) => row?.intrinsic_cost?.status === "partial").length],
      ["deepbom:model:tfliteSubgraphDeepAnalysisSchema", analysis?.tflite_subgraph_deep_analysis?.schema],
      ["deepbom:model:tfliteSubgraphDeepAssessedCount", analysis?.tflite_subgraph_deep_analysis?.assessed_subgraph_count],
      ["deepbom:model:tfliteSubgraphDeepPredictedDelegateCount", analysis?.tflite_subgraph_deep_analysis?.rows?.reduce?.((sum, row) => sum + Number(row?.delegate?.predicted_delegated_operator_count || 0), 0)],
      ["deepbom:model:tfliteSubgraphDeepPredictedFallbackCount", analysis?.tflite_subgraph_deep_analysis?.rows?.reduce?.((sum, row) => sum + Number(row?.delegate?.predicted_fallback_operator_count || 0), 0)],
      ["deepbom:finding:assessmentStatus", findings.status],
      ["deepbom:finding:constantOutputChannels", findings.constant_output_channels],
      ["deepbom:finding:exactZeroKernelSlices", findings.exact_zero_kernel_slices],
      ["deepbom:finding:buildModeDependentChannels", findings.build_mode_dependent_channels],
      ["deepbom:finding:maxInt32UtilizationRatio", findings.max_int32_utilization_ratio],
      ["deepbom:finding:highSeverityCount", findings.high_severity_count],
      ["deepbom:finding:highSeverityFindings", findings.high_severity_findings ? JSON.stringify(findings.high_severity_findings) : null],
      ["deepbom:conformance:profile", conformance.profile],
      ["deepbom:conformance:profileScope", conformance.profile_scope],
      ["deepbom:conformance:status", conformance.status],
      ["deepbom:conformance:violationCodes", JSON.stringify(conformance.violation_codes)],
      ["deepbom:conformance:violationCount", conformance.violation_count],
      ["deepbom:conformance:legacyQu8Artifact", conformance.legacy_qu8_artifact],
      ["deepbom:conformance:source", conformance.source],
      ["deepbom:conformance:interpretationBoundary", conformance.interpretation_boundary],
      ["deepbom:preprocessing:assessmentStatus", preprocessing.status],
      ["deepbom:preprocessing:exactContractIds", JSON.stringify(preprocessing.exact_contract_ids)],
      ["deepbom:preprocessing:portfolioLedgerSha256", preprocessing.portfolio_ledger_sha256],
      ["deepbom:runtime:name", floor.runtime],
      ["deepbom:runtime:necessaryFloor", floor.minimum_version],
      ["deepbom:runtime:floorStatus", floor.status],
      ["deepbom:runtime:floorEvidenceClass", floor.evidence_class],
      ["deepbom:runtime:requirementsManifest", DEPLOYMENT_CONTRACT_FILES.runtime],
      ["deepbom:analyzer:semanticVersion", metadata.semanticVersion],
      ["deepbom:analyzer:buildVersion", metadata.version],
      ["deepbom:analyzer:rulepackVersion", metadata.rulepackVersion],
      ["deepbom:analyzer:rulepackSha256", metadata.rulepackSha256],
    ]),
    externalReferences,
  };
  return {
    identity,
    component,
    quantization,
    floor,
    inputs,
    outputs,
    findings,
    conformance,
    preprocessing,
  };
}

function cycloneDxEnvelope(subject, generatedAt, options = {}, formulation = null, documentProperties = [], additions = {}, serialProfile = "cyclonedx-document") {
  const author = options.author || {};
  const authorOrcid = normalizedOrcid(Object.hasOwn(author, "orcid") ? author.orcid : DEFAULT_DOCUMENT_AUTHOR.orcid);
  const authorName = Object.hasOwn(author, "name") ? text(author.name) : DEFAULT_DOCUMENT_AUTHOR.name;
  const authorEmail = Object.hasOwn(author, "email") ? text(author.email) : DEFAULT_DOCUMENT_AUTHOR.email;
  const authorProperties = properties([
    ["deepbom:documentAuthor:orcid", authorOrcid],
  ]);
  return {
    $schema: CYCLONEDX_SCHEMA,
    bomFormat: "CycloneDX",
    specVersion: "1.7",
    serialNumber: cycloneDxSerialNumber({
      artifactSha256: subject?.hashes?.find((item) => item?.alg === "SHA-256")?.content,
      generatedAt,
      profile: serialProfile,
    }),
    version: 1,
    metadata: {
      timestamp: generatedAt,
      lifecycles: [{ phase: "post-build" }],
      tools: toolMetadata(options),
      authors: [{
        ...(authorOrcid ? { "bom-ref": authorOrcid } : {}),
        name: authorName,
        ...(authorEmail ? { email: authorEmail } : {}),
      }],
      component: subject,
    },
    ...(formulation ? { formulation } : {}),
    ...additions,
    ...((authorProperties.length || documentProperties.length)
      ? { properties: [...authorProperties, ...documentProperties] }
      : {}),
  };
}

export function buildCycloneDxEvidenceDocument(analysis, options = {}) {
  const generatedAt = options.generatedAt || new Date().toISOString();
  const irIdentity = artifactIdentity(analysis, options);
  const artifactIr = SHA256_PATTERN.test(irIdentity.sha256) ? resolveArtifactIrContext(analysis, {
    filename: irIdentity.name,
    format: irIdentity.format,
    sha256: irIdentity.sha256,
    size: irIdentity.byte_length ?? 0,
    artifact_set_sha256: analysis?.artifact_set?.artifact_set_sha256 || null,
  }, {
    artifactIrContext: options.artifactIrContext || null,
    artifactIr: options.artifactIr || null,
    runtimeEvidence: options.runtimeEvidence || options.runtimeAssignmentEvidence || null,
  })?.artifact_ir || null : null;
  const artifactEnvelope = options.artifactEvidenceEnvelope || buildArtifactEvidenceEnvelope(analysis, {
    ...options,
    generatedAt,
    provenance: analyzerProvenance(options),
  });
  const envelopeHash = siblingHash(options, DEPLOYMENT_CONTRACT_FILES.artifactEnvelope);
  const artifactIrHash = siblingHash(options, DEPLOYMENT_CONTRACT_FILES.artifactIr);
  const interfaceHash = siblingHash(options, DEPLOYMENT_CONTRACT_FILES.interfaceContracts);
  const runtimeHash = siblingHash(options, DEPLOYMENT_CONTRACT_FILES.runtime);
  const missingHash = siblingHash(options, DEPLOYMENT_CONTRACT_FILES.missingFields);
  const formulationHash = siblingHash(options, DEPLOYMENT_CONTRACT_FILES.formulation);
  const cycloneDx20PreviewHash = siblingHash(options, DEPLOYMENT_CONTRACT_FILES.cyclonedx20Preview);
  const artifactIrMemberAvailable = Boolean(options.artifactIr && artifactIrHash);
  const references = [
    ...optionalExternalReference(options, "engineeringEvidence", "evidence", "engineering_evidence.json", "External Engineering Bundle evidence ledger."),
    ...optionalExternalReference(options, "engineeringReport", "quality-metrics", "engineering_report.md", "External human-readable Engineering Report."),
    externalReference("evidence", DEPLOYMENT_CONTRACT_FILES.artifactEnvelope, bundleReferenceComment("Canonical artifact evidence envelope.", Boolean(envelopeHash)), envelopeHash),
    ...(artifactIrMemberAvailable ? [externalReference("evidence", DEPLOYMENT_CONTRACT_FILES.artifactIr, bundleReferenceComment("Canonical Artifact Evidence IR with graph, storage, architecture, quantization, and overlay separation.", true), artifactIrHash)] : []),
    externalReference("evidence", DEPLOYMENT_CONTRACT_FILES.interfaceContracts, bundleReferenceComment("Canonical external tensor numerical-contract ledger.", Boolean(interfaceHash)), interfaceHash),
    externalReference("evidence", DEPLOYMENT_CONTRACT_FILES.cyclonedx20Preview, bundleReferenceComment("Commit-pinned CycloneDX 2.0 non-conformant proposal fixture; the pinned draft schema closure is unavailable and this is not a 2.0 conformance claim.", Boolean(cycloneDx20PreviewHash)), cycloneDx20PreviewHash),
    externalReference("formulation", DEPLOYMENT_CONTRACT_FILES.formulation, bundleReferenceComment("Artifact-observed formulation and declaration comparison.", Boolean(formulationHash)), formulationHash),
    externalReference("configuration", DEPLOYMENT_CONTRACT_FILES.runtime, bundleReferenceComment("Machine-readable runtime requirement manifest.", Boolean(runtimeHash)), runtimeHash),
    externalReference("evidence", DEPLOYMENT_CONTRACT_FILES.missingFields, bundleReferenceComment("Machine-readable release-lineage field gap specification.", Boolean(missingHash)), missingHash),
  ];
  const subject = cycloneDxSubject(analysis, {
    ...options,
    artifactEvidenceEnvelope: artifactEnvelope,
    artifactIr,
    artifactIrLocation: artifactIrMemberAvailable ? DEPLOYMENT_CONTRACT_FILES.artifactIr : null,
  }, references);
  const components = supportingComponents(artifactEnvelope);
  return cycloneDxEnvelope(subject.component, generatedAt, options, null, properties([
    ["deepbom:profile", "artifact-numerical-and-runtime-contract-evidence"],
    ["deepbom:artifactEvidenceEnvelopeSchema", artifactEnvelope.schema],
    ["deepbom:artifactEvidenceEnvelopeSha256", artifactEnvelope.envelope_sha256],
    ["deepbom:artifactIrSchema", artifactIr?.schema],
    ["deepbom:artifactIrSha256", artifactIr?.artifact_ir_sha256],
    ["deepbom:model:cpuCostTargetBindingSource", artifactEnvelope.cpu_cost_target_binding?.binding_source],
    ["deepbom:model:cpuCostTargetHostObserved", artifactEnvelope.cpu_cost_target_binding?.host_observed],
    ["deepbom:model:cpuCostTargetProfileId", artifactEnvelope.cpu_cost_target_binding?.profile_id],
    ["deepbom:model:cpuCostTargetProfileSha256", artifactEnvelope.cpu_cost_target_binding?.profile_sha256],
    ...acceleratorBindingPropertyEntries(artifactEnvelope.accelerator_bindings),
    ["deepbom:review:policySchema", artifactEnvelope.policy_identity?.schema],
    ["deepbom:review:policySha256", artifactEnvelope.policy_identity?.policy_sha256],
    ["deepbom:review:policyMode", artifactEnvelope.policy_identity?.mode],
    ["deepbom:evidenceBoundary", ["gguf", "safetensors"].includes(String(analysis?.format || "").toLowerCase())
      ? "Deployment artifact identity, exact tensor storage, declared LLM architecture fields, hash-bound selected repository sidecars, conditional state/compute scenarios, and lower-bound-only memory feasibility where complete. Runtime-private memory, complete allocation or assignment, prompt behavior, task accuracy, clinical validity or utility, safety/effectiveness, and release readiness remain separately bound."
      : "Deployment artifact identity, graph totals, complete internal tensor dtype inventory, graph-level quantization state, deterministic static derivations, and serialized external I/O contracts. Runtime assignment, source-data preprocessing, task accuracy, clinical performance, and release readiness remain separately bound."],
    ["deepbom:bundle:relativeReferencePolicy", "Resolve relative externalReferences against the root of the Deployment Contract Pack."],
  ]), {
    ...(components.length ? { components } : {}),
    dependencies: dependencyGraph(subject.identity.component_bom_ref, components),
    declarations: evidenceDeclarations(artifactEnvelope, subject.identity, generatedAt, options),
    citations: evidenceCitations(generatedAt, options),
  }, "cyclonedx-evidence");
}

function observedFormulation(analysis) {
  const metadata = analysis?.metadata_presence || {};
  const quantization = quantizationSummary(analysis);
  const fingerprint = quantizationFingerprint(analysis);
  const description = text(metadata.description);
  const producerName = text(metadata.producer_name || analysis?.producer);
  const producerVersion = text(metadata.producer_version);
  const converterVersion = text(metadata.converter_tensorflow_version);
  const evidence = [];
  if (/toco/i.test(description)) {
    evidence.push({
      code: "artifact_description_toco",
      evidence_class: "OBSERVED",
      source_pointer: "/metadata_presence/description",
      value: description,
    });
  }
  if (fingerprint.asymmetric_uint8_kernel_tensor_count > 0) {
    evidence.push({
      code: "asymmetric_uint8_kernel_tensors",
      evidence_class: "DERIVED",
      source_pointer: "/tensors + /ops",
      value: fingerprint.asymmetric_uint8_kernel_tensor_count,
    });
  }
  if (fingerprint.per_tensor_depthwise_op_count > 0) {
    evidence.push({
      code: "per_tensor_depthwise_operators",
      evidence_class: "DERIVED",
      source_pointer: "/tensors + /ops",
      value: fingerprint.per_tensor_depthwise_op_count,
    });
  }
  if (fingerprint.per_axis_kernel_tensor_count === 0 && kernelBindings(analysis).length > 0) {
    evidence.push({
      code: "no_per_axis_kernel_tensors",
      evidence_class: "DERIVED",
      source_pointer: "/tensors + /ops",
      value: 0,
    });
  }
  if (fingerprint.conversion_metadata_status === "not_present") {
    evidence.push({
      code: "conversion_metadata_absent",
      evidence_class: "OBSERVED",
      source_pointer: "/metadata_presence/conversion_metadata_status",
      value: "not_present",
    });
  }
  let converterFamily = "unresolved";
  let familyEvidenceClass = "NOT_ASSESSABLE";
  if (/toco/i.test(description) && evidence.filter((item) => item.code !== "artifact_description_toco").length >= 2) {
    converterFamily = "tensorflow-toco-legacy";
    familyEvidenceClass = "CONVERGENT_ARTIFACT_FINGERPRINT";
  } else if (/toco/i.test(description)) {
    converterFamily = "tensorflow-toco-legacy";
    familyEvidenceClass = "DERIVED_FROM_OBSERVED_ARTIFACT_DESCRIPTION";
  } else if (converterVersion) {
    converterFamily = "tensorflow-lite-converter";
    familyEvidenceClass = "OBSERVED_CONVERSION_METADATA_GENERIC_FAMILY";
  } else if (producerName) {
    converterFamily = producerName;
    familyEvidenceClass = "OBSERVED_PRODUCER_DECLARATION";
  }
  const frameworkName = converterVersion
    ? "TensorFlow"
    : producerName || (converterFamily === "tensorflow-toco-legacy" ? "TensorFlow" : null);
  const frameworkEvidenceClass = converterVersion
    ? "OBSERVED_CONVERSION_METADATA"
    : producerName
      ? "OBSERVED_PRODUCER_DECLARATION"
      : converterFamily === "tensorflow-toco-legacy"
        ? "DERIVED_FROM_CONVERGENT_CONVERTER_FAMILY"
        : "NOT_ASSESSABLE";
  return {
    converter_family: converterFamily,
    converter_family_evidence_class: familyEvidenceClass,
    converter_version: converterVersion || producerVersion || null,
    export_framework: frameworkName,
    export_framework_evidence_class: frameworkEvidenceClass,
    export_framework_version: converterVersion || producerVersion || null,
    artifact_description: description || null,
    conversion_metadata_status: text(metadata.conversion_metadata_status || "not_present"),
    conversion_metadata_entry_count: Number(metadata.conversion_metadata_entry_count || 0),
    quantization_classification: quantization.classification,
    quantization_evidence_class: "DERIVED_FROM_ARTIFACT_TENSOR_AND_OPERATOR_CONTRACTS",
    structural_fingerprint: fingerprint,
    evidence,
    interpretation_boundary: "The fingerprint can identify a converter family but not an exact converter binary, source commit, build configuration, or execution environment.",
  };
}

function parseDeclaredPayload(value) {
  if (!present(value)) return null;
  if (typeof value === "object") return value;
  try {
    const parsed = JSON.parse(String(value));
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function normalizeDeclaredFormulation(source) {
  if (!source || typeof source !== "object" || Array.isArray(source)) return null;
  if (source.bomFormat === "CycloneDX" && Array.isArray(source.formulation)) {
    for (const formula of source.formulation) {
      const map = propertyMap(formula?.properties);
      if (map.get("deepbom:formulation:kind") !== "declared") continue;
      const payload = parseDeclaredPayload(map.get("deepbom:formulation:declaredPayload"));
      if (payload) return payload;
      return {
        converter_family: map.get("deepbom:formulation:converterFamily") || null,
        converter_version: map.get("deepbom:formulation:converterVersion") || null,
        export_framework: map.get("deepbom:formulation:exportFramework") || null,
        export_framework_version: map.get("deepbom:formulation:exportFrameworkVersion") || null,
      };
    }
    return null;
  }
  return source;
}

function formulationComparison(declared, observed) {
  if (!declared) {
    return {
      status: "not_assessable_declared_formulation_not_provided",
      comparable_field_count: 0,
      mismatch_count: null,
      mismatches: null,
    };
  }
  const pairs = [
    ["converter_family", firstValue(declared, ["converter_family", "converter.family", "converter.name"]), observed.converter_family],
    ["converter_version", firstValue(declared, ["converter_version", "converter.version"]), observed.converter_version],
    ["export_framework", firstValue(declared, ["export_framework", "framework.name"]), observed.export_framework],
    ["export_framework_version", firstValue(declared, ["export_framework_version", "framework.version"]), observed.export_framework_version],
  ].filter(([, declaredValue, observedValue]) => present(declaredValue) && present(observedValue));
  if (!pairs.length) {
    return {
      status: "not_assessable_no_comparable_fields",
      comparable_field_count: 0,
      mismatch_count: null,
      mismatches: null,
    };
  }
  const mismatches = pairs
    .filter(([, declaredValue, observedValue]) => text(declaredValue).toLowerCase() !== text(observedValue).toLowerCase())
    .map(([field, declaredValue, observedValue]) => ({
      field,
      declared: text(declaredValue),
      observed: text(observedValue),
    }));
  return {
    status: mismatches.length ? "mismatch" : "match",
    comparable_field_count: pairs.length,
    mismatch_count: mismatches.length,
    mismatches,
  };
}

export function buildObservedFormulationDocument(analysis, options = {}) {
  const generatedAt = options.generatedAt || new Date().toISOString();
  const declared = normalizeDeclaredFormulation(options.declaredFormulation || analysis?.declared_formulation || null);
  const observed = observedFormulation(analysis);
  const comparison = formulationComparison(declared, observed);
  const evidenceHash = siblingHash(options, DEPLOYMENT_CONTRACT_FILES.cyclonedx);
  const runtimeHash = siblingHash(options, DEPLOYMENT_CONTRACT_FILES.runtime);
  const missingHash = siblingHash(options, DEPLOYMENT_CONTRACT_FILES.missingFields);
  const references = [
    externalReference("evidence", DEPLOYMENT_CONTRACT_FILES.cyclonedx, bundleReferenceComment("Artifact evidence profile supporting the observed formulation.", Boolean(evidenceHash)), evidenceHash),
    externalReference("configuration", DEPLOYMENT_CONTRACT_FILES.runtime, bundleReferenceComment("Runtime requirements derived from the observed artifact.", Boolean(runtimeHash)), runtimeHash),
    externalReference("evidence", DEPLOYMENT_CONTRACT_FILES.missingFields, bundleReferenceComment("Declared-lineage gaps constraining formulation comparison.", Boolean(missingHash)), missingHash),
  ];
  const { component } = cycloneDxSubject(analysis, options, references);
  const formulas = [{
    "bom-ref": "deepbom:formula:observed-artifact-export",
    properties: properties([
      ["deepbom:formulation:kind", "observed"],
      ["deepbom:formulation:evidenceClass", observed.converter_family_evidence_class],
      ["deepbom:formulation:converterFamily", observed.converter_family],
      ["deepbom:formulation:converterFamilyEvidenceClass", observed.converter_family_evidence_class],
      ["deepbom:formulation:converterVersion", observed.converter_version],
      ["deepbom:formulation:exportFramework", observed.export_framework],
      ["deepbom:formulation:exportFrameworkEvidenceClass", observed.export_framework_evidence_class],
      ["deepbom:formulation:exportFrameworkVersion", observed.export_framework_version],
      ["deepbom:formulation:conversionMetadataStatus", observed.conversion_metadata_status],
      ["deepbom:formulation:quantizationClassification", observed.quantization_classification],
      ["deepbom:formulation:observedPayload", JSON.stringify(observed)],
    ]),
  }];
  if (declared) {
    formulas.push({
      "bom-ref": "deepbom:formula:declared-release-input",
      properties: properties([
        ["deepbom:formulation:kind", "declared"],
        ["deepbom:formulation:evidenceClass", "DECLARED_UNVERIFIED"],
        ["deepbom:formulation:declaredPayload", JSON.stringify(declared)],
      ]),
    });
  }
  return cycloneDxEnvelope(component, generatedAt, options, formulas, properties([
    ["deepbom:formulation:documentSchema", "deepbom.observed_formulation_profile.v1.1"],
    ["deepbom:formulation:declaredInputSchema", "deepbom.declared_formulation_input.v1"],
    ["deepbom:formulation:declaredInputAcceptedForm", "A direct declared-formulation object or a CycloneDX 1.7 BOM containing formulation properties deepbom:formulation:kind=declared and deepbom:formulation:declaredPayload."],
    ["deepbom:formulation:comparisonStatus", comparison.status],
    ["deepbom:formulation:comparableFieldCount", comparison.comparable_field_count],
    ["deepbom:formulation:mismatchCount", comparison.mismatch_count],
    ["deepbom:formulation:mismatches", comparison.mismatches ? JSON.stringify(comparison.mismatches) : null],
    ["deepbom:formulation:interpretationBoundary", "Observed formulation facts do not identify an exact converter binary when the artifact omits that identity. A mismatch is emitted only against supplied comparable declarations."],
  ]), {}, "observed-formulation");
}

function compileDefinition(runtimeEvidence, name) {
  return runtimeEvidence?.selector_context?.build?.compile_definitions
    ?.find((item) => text(item?.name) === name)?.value ?? null;
}

function buildConfigurationBinding(runtimeEvidence, requirement) {
  return bindTfliteBuildRequirement(runtimeEvidence, requirement);
}

function tfliteDelegateRequirementBinding(runtimeEvidence, requirement) {
  return bindTfliteDelegateRequirement(runtimeEvidence, requirement);
}

function conditionallyDelegatableRatioAndPercent(value) {
  const ratio = finite(value);
  return ratio == null ? {} : {
    affected_conditionally_delegatable_mac_ratio: ratio,
    affected_conditionally_delegatable_mac_percent: ratio * 100,
  };
}

function runtimeBuildRequirements(analysis, runtimeEvidence) {
  const risks = analysis?.delegation_repair?.runtime_build_risks || [];
  const xnnpack = risks.length ? risks.map((risk) => ({
      id: risk.id,
      backend: "XNNPACK",
      required_configuration: risk.required_build_configuration,
      binding: buildConfigurationBinding(runtimeEvidence, risk.required_build_configuration),
      conditional_impact: {
        baseline_conditionally_delegatable_ops: Number(risk.baseline_conditionally_delegatable_op_count || 0),
        affected_conditionally_delegatable_ops: Number(risk.affected_conditionally_delegatable_op_count || 0),
        ...conditionallyDelegatableRatioAndPercent(risk.affected_conditionally_delegatable_mac_ratio),
        absent_condition_remaining_conditionally_delegatable_ops: Number(risk.absent_condition_remaining_conditionally_delegatable_op_count || 0),
        absent_condition_remaining_predicted_delegate_segments: Number(risk.absent_condition_remaining_predicted_delegate_segment_count || 0),
      },
      evidence_class: risk.evidence_class || "CONDITIONAL_SOURCE_BACKED_CONFIGURATION_SCENARIO",
      interpretation_boundary: risk.interpretation_boundary || null,
    })) : (() => {
      const requirements = new Map();
      for (const op of analysis?.ops || []) {
        const requirement = text(op?.xnnpack_build_requirement);
        if (!requirement || Number(op?.xnnpack_chain_id) < 0) continue;
        if (!requirements.has(requirement)) requirements.set(requirement, []);
        requirements.get(requirement).push(Number(op.index));
      }
      return [...requirements].map(([requiredConfiguration, opIndices], index) => ({
        id: `xnnpack_required_build_configuration_${index}`,
        backend: "XNNPACK",
        required_configuration: requiredConfiguration,
        binding: buildConfigurationBinding(runtimeEvidence, requiredConfiguration),
        conditional_impact: {
          affected_conditionally_delegatable_ops: opIndices.length,
          affected_op_indices: opIndices,
          remaining_coverage_not_recomputed: true,
        },
        evidence_class: "SOURCE_BACKED_REQUIREMENT_WITHOUT_COUNTERFACTUAL",
        interpretation_boundary: "The requirement is emitted by predicted-delegated op rules; conditional assignment impact was not recomputed.",
      }));
    })();
  const alternate = (analysis?.tflite_delegate_compatibility_evidence?.build_requirements || []).map((requirement) => ({
    id: requirement.id,
    backend: requirement.profile === "tflite_gpu" ? "TFLite GPU delegate" : "TFLite NNAPI delegate",
    required_configuration: requirement.required_configuration,
    binding: tfliteDelegateRequirementBinding(runtimeEvidence, requirement),
    conditional_impact: {
      affected_source_candidate_ops: Number(requirement.affected_source_candidate_op_count || 0),
      support_or_assignment_not_established: true,
    },
    evidence_class: requirement.evidence_class || "REQUIREMENT",
    interpretation_boundary: analysis.tflite_delegate_compatibility_evidence.interpretation_boundary,
  }));
  return [...xnnpack, ...alternate];
}

function numericalAbiWitness(analysis) {
  const row = (analysis?.channel_vitality?.ops || []).find((item) =>
    Number(item.mode_dependent_constant_output_channel_count || 0) > 0
      && item.mode_dependent_constant_channel_indices?.length,
  );
  if (!row) return null;
  const channelIndex = Number(row.mode_dependent_constant_channel_indices[0]);
  const channel = row.top_channels?.find((item) => Number(item.channel_index) === channelIndex) || null;
  return {
    op_index: Number(row.op_index),
    op_name: row.op_name,
    channel_index: channelIndex,
    default_output_code_interval: channel
      ? [channel.default_minimum_output_code, channel.default_maximum_output_code]
      : null,
    single_rounding_output_code_interval: channel
      ? [channel.single_minimum_output_code, channel.single_maximum_output_code]
      : null,
    default_constant: channel ? channel.default_minimum_output_code === channel.default_maximum_output_code : null,
    single_rounding_constant: channel ? channel.single_minimum_output_code === channel.single_maximum_output_code : null,
    vitality_ledger_sha256: normalizedSha256(row.vitality_ledger_sha256) || null,
  };
}

function numericalAbiRequirement(analysis, runtimeEvidence) {
  const equivalence = analysis?.rounding_equivalence || null;
  const fidelity = analysis?.requantization_fidelity || null;
  if (!equivalence && !fidelity) return null;
  const definition = compileDefinition(runtimeEvidence, "TFLITE_SINGLE_ROUNDING");
  const build = text(runtimeEvidence?.runtime?.build);
  const buildMatch = /(?:^|[\s;,])TFLITE_SINGLE_ROUNDING(?:=|\s+)(0|1|false|true)(?:$|[\s;,])/i.exec(build);
  const raw = definition ?? buildMatch?.[1] ?? null;
  const normalized = raw == null ? null : /^(1|true)$/i.test(text(raw))
    ? "enabled"
    : /^(0|false)$/i.test(text(raw))
      ? "disabled"
      : null;
  const divergentChannels = Number(
    equivalence?.divergent_channel_count
      ?? fidelity?.single_rounding_encoding_divergence_channel_count
      ?? 0,
  );
  return {
    id: "tflite_single_rounding",
    compile_definition: "TFLITE_SINGLE_ROUNDING",
    binding_status: normalized ? "declared" : "pending",
    declared_mode: normalized,
    evidence_class: normalized ? "DECLARED_RUNTIME_BUILD" : "NOT_ASSESSED",
    exact_static_impact: {
      assessed_channels: Number(equivalence?.assessed_channel_count ?? fidelity?.assessed_channel_count ?? 0),
      divergent_channels: divergentChannels,
      divergent_interval_states: text(equivalence?.divergent_state_count_decimal) || null,
      maximum_absolute_output_delta_codes: finite(equivalence?.maximum_absolute_output_delta),
      multiplier_encoding_divergence_channels: Number(fidelity?.single_rounding_encoding_divergence_channel_count || 0),
      build_mode_dependent_constant_channels: Number(analysis?.channel_vitality?.mode_dependent_constant_output_channel_count || 0),
      representative_witness: numericalAbiWitness(analysis),
    },
    release_gate_required: divergentChannels > 0,
    interpretation_boundary: "Static evidence proves build-mode sensitivity over the assessed integer domain; it does not identify the deployed compile flag or observed task-level impact.",
  };
}

function preprocessingRuntimeRequirement(analysis, options) {
  const contract = preprocessingContract(analysis);
  const declared = options.productionPreprocessingContract
    || options.releaseManifest?.production_preprocessing_contract
    || analysis?.release_manifest?.production_preprocessing_contract
    || null;
  const declaredId = text(declared?.contract_id) || null;
  const implementationSha256 = normalizedSha256(declared?.implementation_sha256) || null;
  const exactIds = contract.exact_contract_ids;
  const match = declaredId ? exactIds.includes(declaredId) : null;
  return {
    status: contract.status,
    exact_contract_ids: exactIds,
    exact_candidate_count: contract.exact_candidate_count,
    portfolio_ledger_sha256: contract.portfolio_ledger_sha256,
    production_binding: declared
      ? {
          status: match === true && implementationSha256
            ? "bound_exact_contract"
            : match === false
              ? "contradiction_declared_contract_not_exact"
              : "partial_implementation_hash_missing",
          contract_id: declaredId,
          implementation_sha256: implementationSha256,
          exact_contract_match: match,
        }
      : {
          status: exactIds.length ? "pending_production_preprocessing_binding" : "not_assessed_no_exact_static_contract",
          contract_id: null,
          implementation_sha256: null,
          exact_contract_match: null,
        },
    interpretation_boundary: "Exact candidates are finite-domain artifact realizability proofs. They do not identify which preprocessing implementation the production application executes.",
  };
}

function interfaceRuntimeRequirement(analysis, options, identity) {
  const ledger = options.interfaceLedger || buildInterfaceQuantizationContractLedger(analysis);
  const expected = ledger.parameters.map((row) => ({
    direction: row.direction,
    ordinal: row.ordinal,
    tensor_name: row.tensor_name,
    dtype: row.dtype,
    shape: row.shape,
    shape_signature: row.shape_signature,
    affine_mapping_status: row.quantization.affine_mapping_status,
    granularity: row.quantization.granularity,
    parameterization: row.quantization.parameterization,
    scales: row.quantization.scales,
    zero_points: row.quantization.zero_points,
    axis: row.quantization.axis,
    block_size: row.quantization.block_size,
    interface_contract_sha256: row.interface_contract_sha256,
  }));
  const declared = options.productionInterfaceContract
    || options.releaseManifest?.production_interface_contract
    || analysis?.release_manifest?.production_interface_contract
    || null;
  const comparison = compareInterfaceContracts(ledger, declared, identity.sha256);
  const status = ledger.invalid_or_incomplete_parameter_count
    ? "contradiction_invalid_artifact_interface_contract"
    : comparison.status;
  return {
    schema: ledger.schema,
    status: ledger.invalid_or_incomplete_parameter_count ? "invalid_or_incomplete" : expected.length ? "binding_required" : "not_applicable",
    ledger_sha256: ledger.ledger_sha256,
    external_parameter_count: expected.length,
    complete_affine_parameter_count: ledger.quantized_parameter_count,
    unquantized_parameter_count: ledger.unquantized_parameter_count,
    invalid_or_incomplete_parameter_count: ledger.invalid_or_incomplete_parameter_count,
    parameters: expected,
    production_binding: {
      status,
      artifact_sha256: comparison.declared_artifact_sha256,
      implementation_sha256: comparison.implementation_sha256,
      declared_parameter_count: comparison.declared_parameter_count,
      missing_parameter_count: comparison.mismatches.filter((row) => row.declared === "missing").length,
      mismatched_parameter_count: comparison.mismatch_count,
      gate_result: comparison.gate_result,
      mismatch_details: comparison.mismatches,
      declaration_validation: comparison.declaration_validation,
    },
    comparison,
    interpretation_boundary: "The binding compares a hash-identified application encoder/decoder declaration with every artifact-derived external interface contract, including interfaces with no affine mapping. Passing proves declared contract identity, not preprocessing correctness or model suitability.",
  };
}

function semanticVersionParts(value) {
  const match = /^v?(\d+)\.(\d+)(?:\.(\d+))?/.exec(text(value));
  return match ? match.slice(1).map((part) => Number(part || 0)) : null;
}

function versionAtLeast(observed, minimum) {
  const left = semanticVersionParts(observed);
  const right = semanticVersionParts(minimum);
  if (!left || !right) return null;
  for (let index = 0; index < 3; index += 1) {
    if (left[index] !== right[index]) return left[index] > right[index];
  }
  return true;
}

function runtimeFloorComplete(floor) {
  return new Set([
    "complete",
    "complete_for_observed_builtin_op_versions",
    "assessed_onnx_and_model_local_domains",
    "assessed_onnx_model_local_and_source_backed_contrib_domains",
  ]).has(text(floor?.status).toLowerCase());
}

function gateItem(code, severity, requirement, status, evidenceClass) {
  return { code, severity, requirement, status, evidence_class: evidenceClass };
}

export function buildRuntimeRequirementManifest(analysis, options = {}) {
  const generatedAt = options.generatedAt || new Date().toISOString();
  const identity = artifactIdentity(analysis, options);
  const runtimeEvidence = options.runtimeAssignmentEvidence || null;
  const floor = runtimeFloor(analysis);
  const buildRequirements = runtimeBuildRequirements(analysis, runtimeEvidence);
  const numericalAbi = numericalAbiRequirement(analysis, runtimeEvidence);
  const preprocessing = preprocessingRuntimeRequirement(analysis, options);
  const interfaceContract = interfaceRuntimeRequirement(analysis, options, identity);
  const conformance = assessLiteRtInt8Conformance(analysis);
  const observedRuntimeVersion = text(runtimeEvidence?.runtime?.version) || null;
  const floorSatisfied = floor.minimum_version && observedRuntimeVersion
    ? versionAtLeast(observedRuntimeVersion, floor.minimum_version)
    : null;
  const floorCoverageComplete = runtimeFloorComplete(floor);
  const artifactItems = [];
  const runtimeItems = [];
  if (!floorCoverageComplete) {
    artifactItems.push(gateItem(
      "runtime_floor_incomplete",
      "high",
      `${floor.runtime} runtime-floor map coverage must be complete for observed artifact domains`,
      "block",
      floor.evidence_class,
    ));
  }
  if (conformance.status === "nonconformant") {
    artifactItems.push(gateItem(
      "current_litert_int8_profile_nonconformance",
      "high",
      `${conformance.profile}: ${conformance.violation_codes.join(", ")}`,
      options.ciPolicy?.currentLiteRtInt8Profile === "block" ? "block" : "review",
      conformance.evidence_class,
    ));
  }
  if (interfaceContract.invalid_or_incomplete_parameter_count) {
    artifactItems.push(gateItem(
      "invalid_interface_affine_contract",
      "high",
      "Serialized interface scale, zero-point, and axis cardinality must form a complete affine contract",
      "block",
      "DERIVED",
    ));
  }
  const importedArtifactSha = normalizedSha256(runtimeEvidence?.artifact_sha256);
  if (runtimeEvidence && identity.sha256 && !importedArtifactSha) {
    runtimeItems.push(gateItem(
      "runtime_evidence_artifact_hash_unbound",
      "critical",
      `Runtime evidence must bind artifact SHA-256 ${identity.sha256}`,
      "pending",
      "NOT_ASSESSED",
    ));
  }
  if (runtimeEvidence && identity.sha256 && importedArtifactSha && importedArtifactSha !== identity.sha256) {
    runtimeItems.push(gateItem(
      "runtime_evidence_artifact_hash_mismatch",
      "critical",
      `Runtime evidence artifact SHA-256 must equal ${identity.sha256}`,
      "block",
      runtimeEvidence.evidence_class || "OBSERVED_RUNTIME",
    ));
  }
  const importedProfileSha = normalizedSha256(runtimeEvidence?.target_profile_sha256);
  const expectedProfileSha = normalizedSha256(analysis?.target_profile?.profile_sha256);
  if (runtimeEvidence && expectedProfileSha && !importedProfileSha) {
    runtimeItems.push(gateItem(
      "runtime_evidence_target_profile_hash_unbound",
      "high",
      `Runtime evidence must bind target-profile SHA-256 ${expectedProfileSha}`,
      "pending",
      "NOT_ASSESSED",
    ));
  }
  if (runtimeEvidence && expectedProfileSha && importedProfileSha && importedProfileSha !== expectedProfileSha) {
    runtimeItems.push(gateItem(
      "runtime_evidence_target_profile_hash_mismatch",
      "high",
      `Runtime evidence target-profile SHA-256 must equal ${expectedProfileSha}`,
      "block",
      runtimeEvidence.evidence_class || "OBSERVED_RUNTIME",
    ));
  }
  if (runtimeEvidence && !normalizedSha256(runtimeEvidence?.runtime?.binary_sha256)) {
    runtimeItems.push(gateItem(
      "runtime_binary_hash_unbound",
      "high",
      "Imported runtime evidence must bind the executed runtime binary SHA-256",
      "pending",
      "NOT_ASSESSED",
    ));
  }
  if (floor.minimum_version) {
    runtimeItems.push(gateItem(
      floorSatisfied === false ? "runtime_floor_not_satisfied" : "runtime_floor_unbound",
      "medium",
      `${floor.runtime} >= ${floor.minimum_version}`,
      floorSatisfied === false ? "block" : floorSatisfied === true ? "pass" : "pending",
      floorSatisfied == null ? "NOT_ASSESSED" : "DECLARED_RUNTIME_VERSION",
    ));
  }
  for (const requirement of buildRequirements) {
    runtimeItems.push(gateItem(
      "required_build_configuration",
      requirement.conditional_impact?.absent_condition_remaining_conditionally_delegatable_ops === 0 ? "critical" : "high",
      requirement.required_configuration,
      requirement.binding.status === "declared_present"
        ? "pass"
        : requirement.binding.status.startsWith("contradiction")
          ? "block"
          : "pending",
      requirement.binding.evidence_class,
    ));
  }
  if (numericalAbi?.release_gate_required) {
    runtimeItems.push(gateItem(
      "numerical_abi_mode",
      Number(numericalAbi.exact_static_impact.build_mode_dependent_constant_channels || 0) > 0 ? "high" : "medium",
      numericalAbi.compile_definition,
      numericalAbi.binding_status === "declared"
        ? "pass"
        : numericalAbi.binding_status.startsWith("contradiction")
          ? "block"
          : "pending",
      numericalAbi.evidence_class,
    ));
  }
  if (preprocessing.exact_contract_ids.length) {
    runtimeItems.push(gateItem(
      "production_preprocessing_binding",
      "high",
      `Bind production preprocessing implementation SHA-256 to one exact contract: ${preprocessing.exact_contract_ids.join(", ")}`,
      preprocessing.production_binding.status === "bound_exact_contract"
        ? "pass"
        : preprocessing.production_binding.status.startsWith("contradiction")
          ? "block"
          : "pending",
      preprocessing.production_binding.status === "bound_exact_contract" ? "DECLARED_RELEASE_BINDING" : "NOT_ASSESSED",
    ));
  }
  if (interfaceContract.external_parameter_count) {
    const binding = interfaceContract.production_binding;
    runtimeItems.push(gateItem(
      "production_interface_contract_binding",
      "high",
      `Bind the production tensor encoder/decoder SHA-256 to ${interfaceContract.external_parameter_count} artifact external interface contract(s)`,
      binding.status === "bound_exact_contract" ? "pass"
        : binding.status.startsWith("contradiction") ? "block" : "pending",
      binding.status === "bound_exact_contract" ? "DECLARED_RELEASE_BINDING" : "NOT_ASSESSED",
    ));
  }
  const artifactResult = artifactItems.some((item) => item.status === "block")
    ? "block"
    : artifactItems.some((item) => item.status === "review")
      ? "review"
      : "pass";
  const runtimeResult = runtimeItems.some((item) => item.status === "block")
    ? "block"
    : runtimeItems.some((item) => item.status === "pending")
      ? "pending"
      : "pass";
  const combinedResult = artifactResult === "block" || runtimeResult === "block"
    ? "block"
    : runtimeResult === "pending"
      ? "pending"
      : artifactResult === "review"
        ? "review"
        : "pass";
  const format = text(analysis?.format).toLowerCase();
  const target = {
    id: text(analysis?.target_profile?.id) || null,
    label: text(analysis?.target_profile?.label) || null,
    profile_sha256: normalizedSha256(analysis?.target_profile?.profile_sha256) || null,
  };
  return {
    schema: "deepbom.runtime_requirement_manifest.v1.3",
    document_type: "runtime-requirement-manifest",
    generated_at: generatedAt,
    subject: identity,
    target_independent_requirements: {
      necessary_runtime_floor: {
        ...floor,
        coverage_complete: floorCoverageComplete,
      },
      current_litert_int8_profile_conformance: conformance,
      backend_build_requirements: buildRequirements.map(({ conditional_impact, ...requirement }) => requirement),
      numerical_abi_requirements: numericalAbi ? [{
        id: numericalAbi.id,
        compile_definition: numericalAbi.compile_definition,
        binding_status: numericalAbi.binding_status,
        declared_mode: numericalAbi.declared_mode,
        evidence_class: numericalAbi.evidence_class,
        release_gate_required: numericalAbi.release_gate_required,
        interpretation_boundary: numericalAbi.interpretation_boundary,
      }] : [],
      preprocessing_contract: {
        status: preprocessing.status,
        exact_contract_ids: preprocessing.exact_contract_ids,
        portfolio_ledger_sha256: preprocessing.portfolio_ledger_sha256,
        production_binding: preprocessing.production_binding,
      },
      interface_quantization_contract: interfaceContract,
    },
    target_specific_impact: {
      target_profile: target,
      backend_build_requirement_impacts: buildRequirements.map((requirement) => ({
        id: requirement.id,
        conditional_impact: requirement.conditional_impact,
      })),
      numerical_abi_impact: numericalAbi?.exact_static_impact || null,
    },
    necessary_runtime_floor: {
      ...floor,
      coverage_complete: floorCoverageComplete,
      observed_runtime_version: observedRuntimeVersion,
      observed_floor_satisfied: floorSatisfied,
    },
    backend_build_requirements: buildRequirements,
    numerical_abi_requirements: numericalAbi ? [numericalAbi] : [],
    preprocessing_contract_requirement: preprocessing,
    interface_contract_requirement: interfaceContract,
    onnx_execution_provider_source_profiles: format === "onnx"
      ? {
          status: "assessed",
          profiles: (analysis?.ort_compatibility_evidence?.execution_providers || []).map((provider) => ({
            execution_provider: provider.execution_provider,
            source_scope: provider.source_scope,
            assessed_ops: Number(provider.assessed_op_count || 0),
            artifact_precheck_candidates: Number(provider.source_candidate_after_artifact_precheck_count || 0),
            evidence_boundary: "Source registration and artifact-condition precheck; not GetCapability assignment or observed placement.",
          })),
        }
      : {
          status: "not_applicable_non_onnx_artifact",
          profiles: [],
        },
    tflite_delegate_source_profiles: format === "tflite" && analysis?.tflite_delegate_compatibility_evidence
      ? {
          status: analysis.tflite_delegate_compatibility_evidence.assessment_status,
          schema: analysis.tflite_delegate_compatibility_evidence.schema,
          rulepack_sha256: analysis.tflite_delegate_compatibility_evidence.rulepack_sha256,
          tensorflow_source_commit: analysis.tflite_delegate_compatibility_evidence.tensorflow_source_commit,
          profiles: (analysis.tflite_delegate_compatibility_evidence.profiles || []).map((profile) => ({
            id: profile.id,
            assessed_ops: Number(profile.assessed_graph_op_count || 0),
            source_candidates: Number(profile.source_candidate_after_artifact_precheck_count || 0),
            definite_exclusions: Number(profile.definite_exclusion_count || 0),
            selected_build_status: profile.selected_build_status,
            runtime_assignment_status: profile.runtime_assignment_status,
          })),
          interpretation_boundary: analysis.tflite_delegate_compatibility_evidence.interpretation_boundary,
        }
      : {
          status: format === "tflite" ? "not_assessed" : "not_applicable_non_tflite_artifact",
          profiles: [],
        },
    runtime_binding: runtimeEvidence ? {
      status: "imported",
      evidence_class: runtimeEvidence.evidence_class || "OBSERVED_RUNTIME",
      schema: runtimeEvidence.schema || null,
      runtime: runtimeEvidence.runtime || null,
      artifact_sha256: runtimeEvidence.artifact_sha256 || null,
      target_profile_sha256: runtimeEvidence.target_profile_sha256 || null,
      source: runtimeEvidence.source || null,
    } : {
      status: "not_imported",
      evidence_class: "NOT_ASSESSED",
    },
    artifact_static_gate: {
      policy: "block_on_incomplete_deterministic_runtime_floor",
      result: artifactResult,
      items: artifactItems,
    },
    runtime_binding_gate: {
      policy: "pending_until_runtime_and_application_bindings_are_imported; block_only_on_observed_contradiction",
      result: runtimeResult,
      items: runtimeItems,
    },
    ci_gate: {
      policy: "two_stage_artifact_and_runtime_binding_gate",
      result: combinedResult,
      blocker_count: [...artifactItems, ...runtimeItems].filter((item) => item.status === "block").length,
      pending_count: runtimeItems.filter((item) => item.status === "pending").length,
      blockers: [...artifactItems, ...runtimeItems].filter((item) => item.status === "block"),
      pending_requirements: runtimeItems.filter((item) => item.status === "pending"),
      review_count: artifactItems.filter((item) => item.status === "review").length,
      review_items: artifactItems.filter((item) => item.status === "review"),
      evaluation_boundary: "Review is a deterministic artifact finding under the default compatibility policy. Pending means required runtime or application evidence has not been imported. Block means a strict artifact policy failed or imported evidence contradicts a requirement.",
    },
    related_documents: [{
      role: "subject_evidence_bom",
      path: DEPLOYMENT_CONTRACT_FILES.cyclonedx,
      component_bom_ref: identity.component_bom_ref,
      integrity_binding: "The evidence BOM hashes this manifest; the complete ZIP member set is bound by deepbom_contract_pack_manifest.json.",
    }],
    provenance: analyzerProvenance(options),
  };
}

function provenanceField({
  id,
  label,
  category,
  status,
  evidenceClass,
  value = null,
  sourcePointer = null,
  searchedPointers = [],
  reason,
  impact,
  remediation,
}) {
  return {
    id,
    label,
    category,
    status,
    evidence_class: evidenceClass || (status === "present"
      ? "DECLARED"
      : status === "partial"
        ? "DERIVED_PARTIAL"
        : status === "not_applicable"
          ? "NOT_APPLICABLE"
          : "NOT_ASSESSABLE"),
    value,
    source_pointer: sourcePointer,
    searched_pointers: [...new Set(searchedPointers)],
    reason,
    impact,
    remediation,
  };
}

function fieldCounts(fields) {
  return Object.fromEntries(["present", "partial", "missing", "not_applicable"].map((status) => [
    status,
    fields.filter((field) => field.status === status).length,
  ]));
}

function releaseManifestFields(analysis, options, observed) {
  const metadata = analysis?.metadata_presence || {};
  const release = options.releaseManifest || analysis?.release_manifest || {};
  const quantization = quantizationSummary(analysis);
  const sourceCheckpoint = firstValue({ analysis, release, metadata }, [
    "release.source_checkpoint_sha256", "analysis.provenance.source_checkpoint_sha256", "metadata.source_checkpoint_sha256",
  ]);
  const declaredFrameworkName = text(firstValue({ release }, ["release.export_framework.name", "release.export_framework_name"]));
  const declaredFrameworkVersion = text(firstValue({ release }, ["release.export_framework.version", "release.export_framework_version"]));
  const observedFramework = observed.export_framework;
  const observedFrameworkVersion = observed.export_framework_version;
  const exactConverterVersion = firstValue({ release, metadata }, ["release.converter_version", "metadata.converter_tensorflow_version"]);
  const exportConfiguration = firstValue({ release, analysis }, ["release.export_configuration", "analysis.export_configuration"]);
  const quantizationConfiguration = firstValue({ release, analysis }, ["release.quantization_configuration", "analysis.quantization_configuration"]);
  const representativeDataset = firstValue({ release, analysis }, ["release.representative_dataset_id", "analysis.representative_dataset_id"]);
  const pipelineId = firstValue({ release, analysis }, ["release.build_pipeline_id", "analysis.build_pipeline_id"]);
  const declaredSoftwareReleaseId = firstValue({ release, analysis }, ["release.software_release_id", "analysis.software_release_id"]);
  const metadataModelVersion = text(metadata.metadata_model_version);
  const requirementId = firstValue({ release, analysis }, ["release.model_requirement_id", "analysis.model_requirement_id"]);
  const noPredecessor = release.no_predecessor === true || /^(initial|initial_release|none)$/i.test(text(release.previous_release_status));
  const previousArtifact = firstValue({ release, analysis }, ["release.previous_artifact_sha256", "analysis.previous_artifact_sha256"]);
  const quantized = quantization.quantized_tensor_count > 0;
  const calibrationDatasetApplicable = quantized && quantization.classification !== "block_or_tensor_encoded_weights";
  const frameworkStatus = declaredFrameworkName && declaredFrameworkVersion
    ? "present"
    : declaredFrameworkName || declaredFrameworkVersion || observedFramework
      ? "partial"
      : "missing";
  const frameworkValue = declaredFrameworkName || declaredFrameworkVersion
    ? { name: declaredFrameworkName || null, version: declaredFrameworkVersion || null }
    : observedFramework
      ? { inferred_family: observedFramework, version: observedFrameworkVersion || null }
      : null;
  const converterStatus = exactConverterVersion
    ? "present"
    : observed.converter_family !== "unresolved"
      ? "partial"
      : "missing";
  const releasePointers = ["/release_manifest", "/metadata_presence", "/ops", "/tensors"];
  return [
    provenanceField({
      id: "source_checkpoint_sha256",
      label: "Source checkpoint SHA-256",
      category: "release_manifest",
      status: sourceCheckpoint ? "present" : "missing",
      value: sourceCheckpoint,
      sourcePointer: sourceCheckpoint ? "/release_manifest/source_checkpoint_sha256" : null,
      searchedPointers: ["/release_manifest/source_checkpoint_sha256", "/provenance/source_checkpoint_sha256", "/metadata_presence/source_checkpoint_sha256"],
      reason: sourceCheckpoint ? "Immutable source checkpoint identity is bound." : "A deployment artifact does not normally retain the source checkpoint digest.",
      impact: "Without it, the released runtime model cannot be unambiguously traced back to the training checkpoint.",
      remediation: "Record the immutable source checkpoint SHA-256 in the release manifest.",
    }),
    provenanceField({
      id: "export_framework",
      label: "Export framework and version",
      category: "release_manifest",
      status: frameworkStatus,
      evidenceClass: frameworkStatus === "present" ? "DECLARED" : frameworkStatus === "partial" ? observed.export_framework_evidence_class : "NOT_ASSESSABLE",
      value: frameworkValue,
      sourcePointer: frameworkStatus === "present" ? "/release_manifest/export_framework" : observedFramework ? "/observed_formulation/export_framework" : null,
      searchedPointers: releasePointers,
      reason: frameworkStatus === "present" ? "The release context declares framework identity and version." : observedFramework ? "Only a framework family is inferred from artifact evidence; no complete declaration is embedded." : "Framework identity and version are not declared.",
      impact: "Without an immutable framework identity, converter behavior and operator lowering cannot be reproduced exactly.",
      remediation: "Bind export framework name, package version, source commit, and binary or container digest.",
    }),
    provenanceField({
      id: "converter_version",
      label: "Converter version",
      category: "release_manifest",
      status: converterStatus,
      evidenceClass: exactConverterVersion ? "DECLARED_OR_OBSERVED_CONVERSION_METADATA" : converterStatus === "partial" ? observed.converter_family_evidence_class : "NOT_ASSESSABLE",
      value: exactConverterVersion || (converterStatus === "partial" ? {
        inferred_family: observed.converter_family,
        structural_fingerprint: observed.structural_fingerprint,
      } : null),
      sourcePointer: exactConverterVersion ? "/metadata_presence/converter_tensorflow_version" : converterStatus === "partial" ? "/observed_formulation" : null,
      searchedPointers: releasePointers,
      reason: exactConverterVersion ? "A converter version is supplied or embedded." : converterStatus === "partial" ? "Artifact evidence identifies a converter family but not an exact converter binary or version." : "Schema and op versions identify compatibility, not the converter binary.",
      impact: "Without the exact converter build, graph rewrites and numerical contracts cannot be reproduced bit-for-bit.",
      remediation: "Record converter package version, source commit, and binary or container digest.",
    }),
    provenanceField({
      id: "export_configuration",
      label: "Export configuration",
      category: "release_manifest",
      status: exportConfiguration ? "present" : "missing",
      value: exportConfiguration,
      sourcePointer: exportConfiguration ? "/release_manifest/export_configuration" : null,
      searchedPointers: ["/release_manifest/export_configuration", "/export_configuration"],
      reason: exportConfiguration ? "Export options are bound by supplied release context." : "Graph structure does not uniquely recover exporter options.",
      impact: "Options such as keepdims, fused activations, signatures, and optimization passes can change graph structure and delegate partitioning.",
      remediation: "Persist canonical exporter options and their SHA-256.",
    }),
    provenanceField({
      id: "quantization_calibration_configuration",
      label: "Quantization and calibration configuration",
      category: "release_manifest",
      status: !quantized ? "not_applicable" : quantizationConfiguration ? "present" : "partial",
      value: quantizationConfiguration || (quantized ? { artifact_quantization_classification: quantization.classification } : null),
      sourcePointer: quantizationConfiguration ? "/release_manifest/quantization_configuration" : quantized ? "/quantization_status" : null,
      searchedPointers: ["/release_manifest/quantization_configuration", "/quantization_configuration", "/quantization_status", "/tensors"],
      reason: !quantized
        ? "No quantized tensors were observed."
        : quantizationConfiguration
          ? "Release context binds the quantization recipe."
          : quantization.classification === "block_or_tensor_encoded_weights"
            ? "The GGUF tensor directory exposes block encodings, but not the quantizer build, source weights, or quantization command."
            : "The artifact exposes resulting scales and zero-points, but not the complete recipe or calibration run.",
      impact: "Without the recipe and representative-data binding, quantization drift and calibration regressions cannot be reproduced.",
      remediation: quantized ? "Record quantizer version, scheme selection, calibration options, and calibration-data binding." : "No quantization binding is required for this float artifact.",
    }),
    provenanceField({
      id: "representative_dataset_id",
      label: "Representative dataset ID",
      category: "release_manifest",
      status: !calibrationDatasetApplicable ? "not_applicable" : representativeDataset ? "present" : "missing",
      value: representativeDataset,
      sourcePointer: representativeDataset ? "/release_manifest/representative_dataset_id" : null,
      searchedPointers: ["/release_manifest/representative_dataset_id", "/representative_dataset_id"],
      reason: !calibrationDatasetApplicable
        ? quantized
          ? "Block-encoded weight storage is observed, but the artifact does not establish a calibration-dependent representative-dataset contract."
          : "No calibration-dependent quantized path was observed."
        : representativeDataset
          ? "Representative dataset identity is supplied."
          : "Representative calibration data cannot be reconstructed from scales alone.",
      impact: "Without it, calibration ranges and resulting integer contracts cannot be regenerated or audited against dataset changes.",
      remediation: calibrationDatasetApplicable ? "Record dataset/version ID, selection query, preprocessing revision, and immutable digest." : "Not applicable.",
    }),
    provenanceField({
      id: "build_pipeline_id",
      label: "Build pipeline ID",
      category: "release_manifest",
      status: pipelineId ? "present" : "missing",
      value: pipelineId,
      sourcePointer: pipelineId ? "/release_manifest/build_pipeline_id" : null,
      searchedPointers: ["/release_manifest/build_pipeline_id", "/build_pipeline_id"],
      reason: pipelineId ? "Build pipeline identity is supplied." : "The artifact does not bind the CI/CD execution that produced it.",
      impact: "Without a pipeline run identity, toolchain, environment, and policy evidence cannot be reconstructed.",
      remediation: "Record immutable pipeline or workflow ID and run ID.",
    }),
    provenanceField({
      id: "software_release_id",
      label: "Software release ID",
      category: "release_manifest",
      status: declaredSoftwareReleaseId ? "present" : metadataModelVersion ? "partial" : "missing",
      value: declaredSoftwareReleaseId || metadataModelVersion || null,
      sourcePointer: declaredSoftwareReleaseId ? "/release_manifest/software_release_id" : metadataModelVersion ? "/metadata_presence/metadata_model_version" : null,
      searchedPointers: ["/release_manifest/software_release_id", "/software_release_id", "/metadata_presence/metadata_model_version"],
      reason: declaredSoftwareReleaseId ? "Software release identity is supplied." : metadataModelVersion ? "A model metadata version is present but is not necessarily the enclosing software release." : "No release identity is bound.",
      impact: "Without it, a model artifact cannot be tied to the application or firmware release that shipped it.",
      remediation: "Bind the model artifact to the shipping software release ID.",
    }),
    provenanceField({
      id: "model_requirement_id",
      label: "Model requirement ID",
      category: "release_manifest",
      status: requirementId ? "present" : "missing",
      value: requirementId,
      sourcePointer: requirementId ? "/release_manifest/model_requirement_id" : null,
      searchedPointers: ["/release_manifest/model_requirement_id", "/model_requirement_id"],
      reason: requirementId ? "Requirement identity is supplied." : "The artifact does not identify the requirement or acceptance criterion it satisfies.",
      impact: "Without it, audit findings cannot be evaluated against an approved model requirement or acceptance criterion.",
      remediation: "Record the immutable requirement or specification ID and revision.",
    }),
    provenanceField({
      id: "previous_release_artifact_sha256",
      label: "Previous release artifact SHA-256",
      category: "release_manifest",
      status: previousArtifact ? "present" : noPredecessor ? "not_applicable" : "missing",
      value: previousArtifact || (noPredecessor ? { no_predecessor: true } : null),
      sourcePointer: previousArtifact ? "/release_manifest/previous_artifact_sha256" : noPredecessor ? "/release_manifest/no_predecessor" : null,
      searchedPointers: ["/release_manifest/previous_artifact_sha256", "/previous_artifact_sha256", "/release_manifest/no_predecessor", "/release_manifest/previous_release_status"],
      reason: previousArtifact ? "Prior released artifact identity is supplied." : noPredecessor ? "The release explicitly declares that no predecessor exists." : "The artifact does not carry a predecessor link or no-predecessor declaration.",
      impact: "Without an explicit predecessor state, release deltas and change-control lineage cannot be established.",
      remediation: "Record the immediately previous released artifact SHA-256 or an explicit no-predecessor declaration.",
    }),
  ];
}

function artifactEmbeddedFields(analysis) {
  const metadata = analysis?.metadata_presence || {};
  const format = text(analysis?.format || "tflite").toLowerCase();
  const runtime = analysis?.runtime_compat || {};
  const tflite = format === "tflite";
  const preprocessing = preprocessingContract(analysis);
  return [
    provenanceField({
      id: "serving_signature",
      label: tflite ? "TFLite SignatureDef" : "Model serving signature",
      category: "artifact_embedded",
      status: metadata.has_signature_defs || Number(metadata.signature_count || 0) > 0 ? "present" : "missing",
      value: metadata.signature_keys?.length ? metadata.signature_keys : null,
      sourcePointer: "/metadata_presence/signature_keys",
      searchedPointers: ["/metadata_presence/has_signature_defs", "/metadata_presence/signature_count", "/metadata_presence/signature_keys"],
      reason: metadata.has_signature_defs ? "A serving signature is embedded." : "No typed serving signature is embedded.",
      impact: "Without a signature, input/output names alone do not bind a stable callable interface.",
      remediation: "Embed and version the serving signature used by the application.",
    }),
    provenanceField({
      id: "typed_model_metadata",
      label: tflite ? "TFLITE_METADATA" : "Typed model metadata",
      category: "artifact_embedded",
      status: metadata.has_model_metadata ? "present" : "missing",
      value: metadata.has_model_metadata ? {
        schema_identifier: metadata.metadata_schema_identifier || null,
        model_metadata_entry_count: Number(metadata.model_metadata_entry_count || 0),
      } : null,
      sourcePointer: "/metadata_presence",
      searchedPointers: ["/metadata_presence/has_model_metadata", "/metadata_presence/metadata_entries", "/metadata_presence/metadata_schema_identifier"],
      reason: metadata.has_model_metadata ? "Typed model metadata is embedded." : "No typed model metadata block is embedded.",
      impact: "Without typed metadata, tooling cannot reliably recover model identity, process units, associated files, or semantic contracts.",
      remediation: "Embed the format-native typed metadata block and bind its schema version.",
    }),
    provenanceField({
      id: "declared_min_runtime_version",
      label: "Declared minimum runtime version",
      category: "artifact_embedded",
      status: text(runtime.min_runtime_version) ? "present" : "missing",
      value: text(runtime.min_runtime_version) || null,
      sourcePointer: text(runtime.min_runtime_version) ? "/runtime_compat/min_runtime_version" : null,
      searchedPointers: ["/runtime_compat/min_runtime_version", "/metadata_presence/metadata_entries"],
      reason: text(runtime.min_runtime_version) ? "The artifact declares a runtime minimum." : "A runtime floor may be derived from op versions, but no minimum is embedded by the producer.",
      impact: "Without a declaration, deployment systems cannot compare the producer-stated compatibility floor against analyzer-derived requirements.",
      remediation: "Embed the producer-declared minimum runtime version and retain the derived floor separately.",
    }),
    provenanceField({
      id: "model_version",
      label: "Model version",
      category: "artifact_embedded",
      status: text(metadata.metadata_model_version || metadata.model_version) ? "present" : "missing",
      value: text(metadata.metadata_model_version || metadata.model_version) || null,
      sourcePointer: "/metadata_presence/metadata_model_version",
      searchedPointers: ["/metadata_presence/metadata_model_version", "/metadata_presence/model_version"],
      reason: text(metadata.metadata_model_version || metadata.model_version) ? "A model version is embedded." : "No model version is embedded.",
      impact: "Without it, identical model names cannot be ordered or linked to release history without relying only on hashes.",
      remediation: "Embed a stable model semantic or release version while retaining the artifact SHA-256 as identity.",
    }),
    provenanceField({
      id: "model_author",
      label: "Model author",
      category: "artifact_embedded",
      status: text(metadata.metadata_author) ? "present" : "missing",
      value: text(metadata.metadata_author) || null,
      sourcePointer: "/metadata_presence/metadata_author",
      searchedPointers: ["/metadata_presence/metadata_author"],
      reason: text(metadata.metadata_author) ? "Model author is embedded." : "No model author is embedded.",
      impact: "Without an author or owning organization, responsibility and maintenance escalation are not machine-readable.",
      remediation: "Embed model author or owner identity and an immutable organization reference.",
    }),
    provenanceField({
      id: "model_license",
      label: "Model license",
      category: "artifact_embedded",
      status: text(metadata.metadata_license) ? "present" : "missing",
      value: text(metadata.metadata_license) || null,
      sourcePointer: "/metadata_presence/metadata_license",
      searchedPointers: ["/metadata_presence/metadata_license"],
      reason: text(metadata.metadata_license) ? "Model license is embedded." : "No model license is embedded.",
      impact: "Without a license declaration, redistribution and deployment permissions cannot be evaluated from the artifact.",
      remediation: "Embed an SPDX license identifier or a cryptographically bound license reference.",
    }),
    provenanceField({
      id: "output_label_map",
      label: "Output label or semantic map",
      category: "artifact_embedded",
      status: metadata.output_semantics_documented ? "present" : Number(metadata.output_label_file_count || 0) > 0 ? "partial" : "missing",
      value: {
        declared_label_files: Number(metadata.output_label_file_count || 0),
        verified_output0_label_files: Number(metadata.verified_output0_label_file_count || 0),
      },
      sourcePointer: "/metadata_presence/output_associated_files",
      searchedPointers: ["/metadata_presence/output_semantics_documented", "/metadata_presence/output_label_file_count", "/metadata_presence/verified_output0_label_file_count", "/metadata_presence/output_associated_files"],
      reason: metadata.output_semantics_documented ? "A payload- and cardinality-verified output mapping is embedded." : "No machine-verified output-0 label or semantic mapping is embedded.",
      impact: "Without it, class indices or output tensors cannot be mapped to application meaning reproducibly.",
      remediation: "Embed and hash the output label map or typed postprocessing contract.",
    }),
    provenanceField({
      id: "preprocessing_contract",
      label: "Preprocessing contract",
      category: "artifact_embedded",
      status: metadata.documented_preprocessing ? "present" : preprocessing.exact_contract_ids.length ? "partial" : "missing",
      evidenceClass: metadata.documented_preprocessing ? "OBSERVED" : preprocessing.exact_contract_ids.length ? "DERIVED_CANDIDATE_SET" : "NOT_ASSESSABLE",
      value: metadata.documented_preprocessing ? {
        status: metadata.preprocessing_contract_status,
      } : preprocessing.exact_contract_ids.length ? {
        exact_artifact_realization_candidates: preprocessing.exact_contract_ids,
        production_selection_unbound: true,
      } : null,
      sourcePointer: metadata.documented_preprocessing ? "/metadata_presence/input_process_units" : preprocessing.exact_contract_ids.length ? "/preprocessing_realizability" : null,
      searchedPointers: ["/metadata_presence/documented_preprocessing", "/metadata_presence/preprocessing_contract_status", "/metadata_presence/input_process_units", "/preprocessing_realizability"],
      reason: metadata.documented_preprocessing ? "The artifact embeds recognized process units." : preprocessing.exact_contract_ids.length ? "Static analysis derives exact realizability candidates, but the artifact does not select or document the production implementation." : "No preprocessing contract is embedded or deterministically reconstructed.",
      impact: "Without it, a numerically valid model can silently receive application inputs under an incompatible normalization or channel-order contract.",
      remediation: "Embed the preprocessing contract and bind the production implementation SHA-256.",
    }),
  ];
}

export function buildMissingProvenanceFieldSpecification(analysis, options = {}) {
  const generatedAt = options.generatedAt || new Date().toISOString();
  const identity = artifactIdentity(analysis, options);
  const observed = observedFormulation(analysis);
  const releaseFields = releaseManifestFields(analysis, options, observed);
  const artifactFields = artifactEmbeddedFields(analysis);
  const fields = [...releaseFields, ...artifactFields];
  const statusCounts = fieldCounts(fields);
  const releaseCounts = fieldCounts(releaseFields);
  const artifactCounts = fieldCounts(artifactFields);
  return {
    schema: "deepbom.missing_provenance_field_specification.v1.1",
    document_type: "missing-provenance-field-specification",
    generated_at: generatedAt,
    subject: identity,
    purpose: "Machine-readable inventory of release-lineage and artifact-embedded fields required for reproducibility, integration, and change control.",
    status_counts: statusCounts,
    release_manifest_status_counts: releaseCounts,
    artifact_embedded_status_counts: artifactCounts,
    release_manifest_fields: releaseFields,
    artifact_embedded_fields: artifactFields,
    fields,
    ci_gate: {
      policy: "require_present_unless_not_applicable",
      result: statusCounts.missing || statusCounts.partial ? "incomplete" : "complete",
      missing_field_count: statusCounts.missing,
      partial_field_count: statusCounts.partial,
      noncompliant_field_count: statusCounts.missing + statusCounts.partial,
      release_manifest_result: releaseCounts.missing || releaseCounts.partial ? "incomplete" : "complete",
      artifact_embedded_result: artifactCounts.missing || artifactCounts.partial ? "incomplete" : "complete",
    },
    related_documents: [{
      role: "subject_evidence_bom",
      path: DEPLOYMENT_CONTRACT_FILES.cyclonedx,
      component_bom_ref: identity.component_bom_ref,
      integrity_binding: "The evidence BOM hashes this specification; the complete ZIP member set is bound by deepbom_contract_pack_manifest.json.",
    }],
    provenance: analyzerProvenance(options),
    interpretation_boundary: "Missing means not found in the artifact or supplied release context, not proof that the organization lacks the information. Partial preserves artifact-derived evidence without promoting it to a producer declaration.",
  };
}

function jsonMemberSha256(document) {
  return sha256TextHex(JSON.stringify(document, null, 2));
}

export function buildInterfaceContractLedgerDocument(analysis, options = {}) {
  const identity = artifactIdentity(analysis, options);
  return {
    ...(options.interfaceLedger || buildInterfaceQuantizationContractLedger(analysis)),
    generated_at: options.generatedAt || new Date().toISOString(),
    subject: identity,
    provenance: analyzerProvenance(options),
  };
}

export function buildDeploymentContractDocuments(analysis, options = {}) {
  if (!analysis || typeof analysis !== "object") throw new Error("Analyzed model evidence is required.");
  const generatedAt = options.generatedAt || new Date().toISOString();
  const shared = { ...options, generatedAt };
  const artifactEnvelope = buildArtifactEvidenceEnvelope(analysis, {
    ...shared,
    provenance: analyzerProvenance(shared),
  });
  const irIdentity = artifactIdentity(analysis, shared);
  const artifactIrContext = resolveArtifactIrContext(analysis, {
    filename: irIdentity.name,
    format: irIdentity.format,
    sha256: irIdentity.sha256,
    size: irIdentity.byte_length ?? 0,
    artifact_set_sha256: analysis?.artifact_set?.artifact_set_sha256 || null,
  }, {
    artifactIrContext: options.artifactIrContext || null,
    artifactIr: options.artifactIr || null,
    runtimeEvidence: options.runtimeEvidence || options.runtimeAssignmentEvidence || null,
  });
  if (!artifactIrContext) throw new Error("Canonical Artifact Evidence IR could not be resolved for the deployment contract.");
  const artifactIr = artifactIrContext.artifact_ir;
  const canonicalInterfaceLedger = artifactEnvelope.interfaces || buildInterfaceQuantizationContractLedger(analysis);
  const contractShared = { ...shared, interfaceLedger: canonicalInterfaceLedger };
  const interfaceContracts = buildInterfaceContractLedgerDocument(analysis, contractShared);
  const cycloneDx20Preview = buildCycloneDx20ParameterContractPreview(analysis, {
    ...contractShared,
  });
  const runtime = buildRuntimeRequirementManifest(analysis, contractShared);
  const missing = buildMissingProvenanceFieldSpecification(analysis, shared);
  const firstHashes = {
    [DEPLOYMENT_CONTRACT_FILES.artifactEnvelope]: jsonMemberSha256(artifactEnvelope),
    [DEPLOYMENT_CONTRACT_FILES.artifactIr]: jsonMemberSha256(artifactIr),
    [DEPLOYMENT_CONTRACT_FILES.interfaceContracts]: jsonMemberSha256(interfaceContracts),
    [DEPLOYMENT_CONTRACT_FILES.cyclonedx20Preview]: jsonMemberSha256(cycloneDx20Preview),
    [DEPLOYMENT_CONTRACT_FILES.runtime]: jsonMemberSha256(runtime),
    [DEPLOYMENT_CONTRACT_FILES.missingFields]: jsonMemberSha256(missing),
  };
  const evidence = buildCycloneDxEvidenceDocument(analysis, {
    ...shared,
    artifactEvidenceEnvelope: artifactEnvelope,
    artifactIr,
    externalDocumentHashes: firstHashes,
  });
  const evidenceHash = jsonMemberSha256(evidence);
  const formulation = buildObservedFormulationDocument(analysis, {
    ...shared,
    externalDocumentHashes: {
      ...firstHashes,
      [DEPLOYMENT_CONTRACT_FILES.cyclonedx]: evidenceHash,
    },
  });
  const documents = {
    cyclonedx_evidence: evidence,
    cyclonedx_2_0_parameter_contract_preview: cycloneDx20Preview,
    artifact_evidence_envelope: artifactEnvelope,
    artifact_ir: artifactIr,
    interface_contract_ledger: interfaceContracts,
    observed_formulation: formulation,
    runtime_requirement_manifest: runtime,
    missing_provenance_field_specification: missing,
  };
  const memberDigests = {
    [DEPLOYMENT_CONTRACT_FILES.cyclonedx]: evidenceHash,
    [DEPLOYMENT_CONTRACT_FILES.formulation]: jsonMemberSha256(formulation),
    ...firstHashes,
  };
  return {
    schema: "deepbom.deployment_contract_export_set.v1.4",
    generated_at: generatedAt,
    subject: artifactIdentity(analysis, shared),
    files: { ...DEPLOYMENT_CONTRACT_FILES },
    integrity: {
      hash_method: "SHA-256 over UTF-8 JSON.stringify(document, null, 2) bytes",
      member_sha256: memberDigests,
      reference_graph: "acyclic sibling hashes plus deepbom_contract_pack_manifest.json for the complete member digest set",
    },
    documents,
  };
}
