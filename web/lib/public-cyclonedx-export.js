import { artifactIrOperators, artifactIrValues } from "./artifact-ir-selectors.js";
import {
  ANALYZER_SEMANTIC_VERSION,
  ANALYZER_VERSION,
  RULEPACK_VERSION,
} from "./app-config.js";
import {
  ANALYZER_BUILD_COMMIT,
  ANALYZER_BUILD_SOURCE_STATE,
  ANALYZER_BUNDLE_CONTENT_SHA256,
} from "./build-metadata.js";
import { buildInterfaceQuantizationContractLedger } from "./quantization-contract-summary.js";
import { canonicalJson, normalizeJsonContractValue } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import {
  analyzerContentVersion,
  analyzerBomRef,
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
import { tensorRtCycloneDxPropertyEntries } from "./tensorrt-cyclonedx-properties.js";
import { validateArtifactEvidenceIr } from "./artifact-ir.js";

const CYCLONEDX_17_SCHEMA = "http://cyclonedx.org/schema/bom-1.7.schema.json";
const AUTHOR = Object.freeze({
  name: "Jun-Hwan Kwon",
  orcid: "0000-0002-6464-3895",
});
const SHA256_PATTERN = /^[a-f0-9]{64}$/;

export function buildPublicCycloneDxDocuments(analysis, options = {}) {
  const generatedAt = options.generatedAt || new Date().toISOString();
  const interfaceLedger = buildInterfaceQuantizationContractLedger(analysis);
  return {
    generated_at: generatedAt,
    documents: {
      cyclonedx_evidence: buildPublicCycloneDx17ArtifactContract(analysis, {
        ...options,
        generatedAt,
        interfaceLedger,
      }),
    },
  };
}

export function buildPublicCycloneDx17ArtifactContract(analysis = {}, options = {}) {
  const generatedAt = options.generatedAt || new Date().toISOString();
  const ledger = options.interfaceLedger || buildInterfaceQuantizationContractLedger(analysis);
  const sha256 = normalizeSha256(options.hash || analysis.model_sha256);
  const artifactIr = options.artifactIr ? validateArtifactEvidenceIr(options.artifactIr) : null;
  if (artifactIr && sha256 && artifactIr.artifact.sha256 !== sha256) {
    throw new Error("Public CycloneDX Artifact IR is not bound to the exported artifact SHA-256.");
  }
  const name = cleanText(analysis.filename) || "model";
  const bomRef = artifactBomRef(sha256, name);
  const contentVersion = artifactContentVersion(
    sha256,
    analysis?.metadata_presence?.metadata_model_version || analysis?.metadata_presence?.model_version,
  );
  const graphTotals = {
    operator_count: nonNegativeInteger(analysis.operator_count ?? artifactIrOperators(analysis)?.length),
    tensor_count: nonNegativeInteger(analysis.tensor_count ?? artifactIrValues(analysis)?.length),
    mac_count: nonNegativeNumber(analysis.total_macs),
  };
  const quantization = analysis.quantization_status || {};
  const dtypeInventory = tensorDtypeInventory(analysis);
  const format = cleanText(analysis.format).toLowerCase() || "unknown";
  const llm = analysis.on_device_llm || null;
  const byteIntegrity = analysis.artifact_byte_integrity?.schema
    ? normalizeJsonContractValue(analysis.artifact_byte_integrity)
    : null;
  const byteIntegritySummary = byteIntegrity ? {
    schema: byteIntegrity.schema,
    status: byteIntegrity.status,
    evidence_class: byteIntegrity.evidence_class,
    file_size: byteIntegrity.file_size,
    conservation_status: byteIntegrity.conservation_status,
    classified_bytes: byteIntegrity.classified_bytes,
    flatbuffer_referenced_end: byteIntegrity.flatbuffer_referenced_end,
    terminal_zero_alignment_bytes: byteIntegrity.terminal_zero_alignment_bytes,
    metadata_archive_status: byteIntegrity.metadata_archive_status,
    metadata_archive_start: byteIntegrity.metadata_archive_start,
    metadata_archive_end: byteIntegrity.metadata_archive_end,
    metadata_archive_bytes: byteIntegrity.metadata_archive_bytes,
    unowned_trailing_bytes: byteIntegrity.unowned_trailing_bytes,
    partial_buffer_overlap_count: byteIntegrity.partial_buffer_overlap_count,
    flatbuffer_archive_overlap_bytes: byteIntegrity.flatbuffer_archive_overlap_bytes,
    issue_count: byteIntegrity.issue_count,
  } : null;
  const byteIntegritySha256 = byteIntegrity
    ? sha256TextHex(canonicalJson(byteIntegrity))
    : null;
  const llmEvidence = ["tflite", "onnx", "gguf", "safetensors"].includes(format) && llm?.schema === "deepbom.on_device_llm_contract.v2";
  const serializedLlmContainer = ["gguf", "safetensors"].includes(format) && llmEvidence;
  const cliScenario = analysis?.llm_token_budget_scenario || analysis?.cli_context_scenario;
  const quantizationProperties = serializedLlmContainer ? [
    ["deepbom:model:storageEncodingClassification", cleanText(quantization.classification) || "storage_only"],
    ["deepbom:model:storageEncodingBasis", format === "gguf"
      ? "Source-pinned GGML tensor block encodings and exact serialized ranges; no execution graph, affine activation contract, or Q/DQ placement is inferred."
      : "SafeTensors dtype, shape, and exact serialized ranges; no execution graph, affine activation contract, or Q/DQ placement is inferred."],
  ] : [
    ["deepbom:model:quantizationClassification", cleanText(quantization.classification || analysis.quantization_classification) || "not_assessed"],
    ["deepbom:model:fullIntegerQuantized", booleanOrNull(quantization.full_integer)],
    ["deepbom:model:quantizationClassificationBasis", "External I/O dtype, complete internal tensor dtype inventory, every MAC-bearing compute op's activation dtype path, and serialized Q/DQ inventory."],
    ["deepbom:model:quantizedTensorCount", nonNegativeInteger(analysis.quantized_tensors)],
    ["deepbom:model:quantizedComputeOperatorCount", nonNegativeInteger(quantization.quantized_compute_ops)],
    ["deepbom:model:computeOperatorCount", nonNegativeInteger(quantization.compute_ops)],
    ["deepbom:model:quantizedComputeMacRatio", finiteOrNull(quantization.quantized_compute_mac_percent)],
    ["deepbom:model:serializedQuantizeOperatorCount", nonNegativeInteger(quantization.quantize_ops)],
    ["deepbom:model:serializedDequantizeOperatorCount", nonNegativeInteger(quantization.dequantize_ops)],
  ];
  const llmProperties = llmEvidence ? [
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
  ] : [];
  const tensorRtProperties = tensorRtCycloneDxPropertyEntries(analysis);
  const component = {
    type: "machine-learning-model",
    name,
    ...artifactComponentMetadata(analysis),
    ...(contentVersion.version ? { version: contentVersion.version } : {}),
    "bom-ref": bomRef,
    ...(sha256 ? { hashes: [{ alg: "SHA-256", content: sha256 }] } : {}),
    modelCard: {
      "bom-ref": `${bomRef}-model-card`,
      modelParameters: {
        inputs: ledger.parameters.filter((row) => row.direction === "input").map(modelCardParameter),
        outputs: ledger.parameters.filter((row) => row.direction === "output").map(modelCardParameter),
      },
    },
    properties: compactProperties([
      ["deepbom:contract:schema", "deepbom.public_cyclonedx_artifact_contract.v1.1"],
      ["deepbom:model:format", format],
      ["deepbom:model:versionBasis", contentVersion.basis],
      ["deepbom:model:fileSizeBytes", nonNegativeNumber(options.fileSizeBytes ?? analysis.file_size_bytes)],
      ["deepbom:model:hashBasis", cleanText(analysis.artifact_bundle?.hash_basis) || "artifact_file_bytes_sha256"],
      ["deepbom:model:graphTotals", JSON.stringify(graphTotals)],
      ["deepbom:model:artifactIrSchema", artifactIr?.schema],
      ["deepbom:model:artifactIrSha256", artifactIr?.artifact_ir_sha256],
      ["deepbom:model:serializedContractStatus", analysis?.onnx_contract_conflict?.status],
      ["deepbom:model:contractConflictCapsuleSha256", analysis?.onnx_contract_conflict?.capsule_sha256],
      ["deepbom:model:contractConflictRootCount", analysis?.onnx_contract_conflict?.summary?.unconditional_root_conflict_count],
      ["deepbom:model:contractConflictConditionalVariantCount", analysis?.onnx_contract_conflict?.summary?.condition_bound_invalid_variant_count],
      ["deepbom:model:contractConflictBlockedMacRows", analysis?.onnx_contract_conflict?.summary?.blocked_mac_row_count],
      ...quantizationProperties,
      ...safeTensorsQuantizationPropertyEntries(analysis),
      ["deepbom:model:tensorDtypeInventory", JSON.stringify(dtypeInventory)],
      ["deepbom:model:artifactByteIntegritySchema", byteIntegrity?.schema],
      ["deepbom:model:artifactByteIntegrityLedgerSha256", byteIntegritySha256],
      ["deepbom:model:artifactByteIntegritySummary", byteIntegritySummary ? JSON.stringify(byteIntegritySummary) : null],
      ...llmProperties,
      ...tensorRtProperties,
      ["deepbom:model:interfaceContractSchema", ledger.schema],
      ["deepbom:model:interfaceContractLedgerSha256", ledger.ledger_sha256],
      ["deepbom:model:interfaceContractLedger", JSON.stringify(ledger)],
      ["deepbom:model:completeAffineInterfaceCount", ledger.quantized_parameter_count],
      ["deepbom:model:unquantizedInterfaceCount", ledger.unquantized_parameter_count],
      ["deepbom:model:invalidOrIncompleteInterfaceCount", ledger.invalid_or_incomplete_parameter_count],
      ["deepbom:evidenceBoundary", serializedLlmContainer
        ? "Standalone artifact identity, exact tensor storage, declared architecture fields, hash-bound selected repository sidecars, conditional state/compute scenarios, and lower-bound-only memory feasibility where complete. Runtime-private memory, complete allocation or assignment, prompt behavior, task accuracy, clinical validity or utility, safety/effectiveness, and release readiness are not established."
        : llmEvidence
          ? "Standalone artifact identity, graph totals, internal tensor dtype and quantization evidence, serialized transformer-operator evidence, bounded transformer-like motif candidates, external state-name candidates, and serialized external I/O contracts. Graph motifs do not establish an LLM architecture, KV layout, tokenizer, generation behavior, runtime assignment, task accuracy, clinical performance, or release readiness."
        : "Standalone artifact identity, graph totals, complete internal tensor dtype inventory, graph-level 8-bit activation/MAC coverage, serialized Q/DQ inventory, and serialized external I/O contracts. Runtime assignment, source-data preprocessing, task accuracy, clinical performance, and release readiness are not established."],
    ]),
  };
  return {
    $schema: CYCLONEDX_17_SCHEMA,
    bomFormat: "CycloneDX",
    specVersion: "1.7",
    serialNumber: cycloneDxSerialNumber({ artifactSha256: sha256, generatedAt, profile: "public-cyclonedx-1.7-artifact-contract" }),
    version: 1,
    metadata: {
      timestamp: generatedAt,
      lifecycles: [{ phase: "post-build" }],
      tools: {
        components: [{
          type: "application",
          name: "DEEPBOM",
          version: analyzerContentVersion(ANALYZER_SEMANTIC_VERSION, ANALYZER_BUILD_COMMIT, ANALYZER_BUNDLE_CONTENT_SHA256),
          "bom-ref": analyzerBomRef("DEEPBOM", ANALYZER_SEMANTIC_VERSION, ANALYZER_BUILD_COMMIT),
          externalReferences: [interfaceCorpusValidationExternalReference()],
          properties: compactProperties([
            ["deepbom:analyzer:buildVersion", ANALYZER_VERSION],
            ["deepbom:analyzer:semanticVersion", ANALYZER_SEMANTIC_VERSION],
            ["deepbom:analyzer:buildCommit", ANALYZER_BUILD_COMMIT],
            ["deepbom:analyzer:buildSourceState", ANALYZER_BUILD_SOURCE_STATE],
            ["deepbom:analyzer:bundleContentSha256", ANALYZER_BUNDLE_CONTENT_SHA256],
            ["deepbom:analyzer:rulepackVersion", RULEPACK_VERSION],
            ...interfaceCorpusValidationProperties(),
          ]),
        }],
      },
      authors: [{
        "bom-ref": `https://orcid.org/${AUTHOR.orcid}`,
        name: AUTHOR.name,
      }],
      component,
    },
    properties: compactProperties([
      ["deepbom:profile", "public-standalone-artifact-contract"],
      ["deepbom:documentAuthor:orcid", `https://orcid.org/${AUTHOR.orcid}`],
      ["deepbom:artifactIrSha256", artifactIr?.artifact_ir_sha256],
      ["deepbom:privacy", "Generated locally in the browser; artifact bytes are not uploaded for this export."],
    ]),
  };
}

function modelCardParameter(parameter) {
  const shape = parameter.shape?.length ? `[${parameter.shape.join(",")}]` : "shape=unbound";
  const quantization = parameter.quantization || {};
  const quant = quantization.status === "not_quantized"
    ? "unquantized"
    : `${String(quantization.granularity || "unknown").replaceAll("_", "-")}; scale_count=${quantization.scale_count || 0}; zero_point_count=${quantization.zero_point_count || 0}${quantization.scales?.length === 1 ? `; scale=${quantization.scales[0]}` : ""}${quantization.zero_points?.length === 1 ? `; zero_point=${quantization.zero_points[0]}` : ""}`;
  return { format: `${parameter.dtype || "UNKNOWN"} ${shape}; ${quant}` };
}

function tensorDtypeInventory(analysis) {
  const declared = Array.isArray(analysis?.tensor_types) ? analysis.tensor_types : [];
  if (declared.length) return Object.fromEntries(declared.map((row) => [String(row.name || "UNKNOWN"), nonNegativeInteger(row.count) ?? 0]));
  const counts = {};
  for (const tensor of artifactIrValues(analysis) || []) {
    const dtype = cleanText(tensor?.dtype).toUpperCase() || "UNKNOWN";
    counts[dtype] = (counts[dtype] || 0) + 1;
  }
  return counts;
}

function compactProperties(entries) {
  return entries.flatMap(([name, value]) => value === null || value === undefined || value === ""
    ? []
    : [{ name, value: typeof value === "string" ? value : String(value) }]);
}

function normalizeSha256(value) {
  const digest = cleanText(value).toLowerCase();
  return SHA256_PATTERN.test(digest) ? digest : "";
}

function cleanText(value) {
  return String(value ?? "").trim();
}

function nonNegativeInteger(value) {
  const number = Number(value);
  return Number.isSafeInteger(number) && number >= 0 ? number : null;
}

function nonNegativeNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : null;
}

function finiteOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function booleanOrNull(value) {
  return typeof value === "boolean" ? value : null;
}
