import { sha256TextHex } from "./sha256-sync.js";
import { canonicalJson } from "./report-utils.js";
import { buildStateStorageScenarios } from "./llm-specialized-projection.js";
import { assessOnDeviceLlmRuntimeManifest, buildLlmStateScenarioMatrix, validateOnDeviceLlmRuntimeContract } from "./on-device-llm-runtime-manifest.js";
import { buildLlmMemoryFeasibility, validateLlmMemoryFeasibility } from "./llm-memory-feasibility.js";
import { buildLlmLayerStorageLedger } from "./llm-layer-storage.js";
import { buildLlmStaticMemoryPlacement, validateLlmStaticMemoryPlacement } from "./llm-static-memory-placement.js";
import { buildTensorRtLlmContract, validateTensorRtLlmContract } from "./tensorrt-llm-contract.js";

export const ON_DEVICE_LLM_CONTRACT_SCHEMA = "deepbom.on_device_llm_contract.v2";

const GENERATION_FIELDS = Object.freeze([
  "do_sample", "temperature", "top_p", "top_k", "typical_p", "min_p",
  "max_new_tokens", "min_new_tokens", "repetition_penalty", "num_beams",
  "bos_token_id", "eos_token_id", "pad_token_id",
]);
const MEDICAL_DECLARATION_FIELDS = Object.freeze([
  "intended_use", "patient_population", "clinical_workflow", "human_oversight",
  "task_and_acceptance_metrics", "evaluation_dataset_and_lineage",
  "prompt_and_output_constraints", "privacy_and_phi_handling", "postmarket_monitoring_plan",
]);

function text(value) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function positiveInteger(value) {
  return Number.isSafeInteger(value) && value > 0 ? value : null;
}

function exact(value) {
  return { value: value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null, decimal: String(value) };
}

function exactFrom(value) {
  if (value && typeof value === "object" && /^\d+$/.test(String(value.decimal || ""))) return BigInt(value.decimal);
  if (Number.isSafeInteger(value) && value >= 0) return BigInt(value);
  if (typeof value === "string" && /^\d+$/.test(value)) return BigInt(value);
  return null;
}

function serializedEncodingContract(analysis, storage) {
  const encodingInventory = Array.isArray(storage?.encodings) ? storage.encodings.map((row) => ({ ...row })) : [];
  const assignmentSignatureRows = (analysis?.tensors || []).map((row) => ({
    index: Number.isSafeInteger(Number(row.index)) ? Number(row.index) : null,
    name: String(row.name || ""),
    dtype: String(row.dtype || "UNKNOWN"),
    shape: Array.isArray(row.shape) ? row.shape.map((value) => Number.isSafeInteger(Number(value)) ? Number(value) : null) : [],
    byte_length: Number.isSafeInteger(Number(row.byte_length)) ? Number(row.byte_length)
      : Number.isSafeInteger(Number(row.initializer_available_bytes)) ? Number(row.initializer_available_bytes)
        : Number.isSafeInteger(Number(row.initializer_bytes)) ? Number(row.initializer_bytes)
          : Number.isSafeInteger(Number(row.buffer_data_length)) ? Number(row.buffer_data_length) : null,
  })).sort((left, right) => Number(left.index ?? 0) - Number(right.index ?? 0) || String(left.name).localeCompare(String(right.name)));
  return {
    encoding_signature_schema: "deepbom.llm_serialized_encoding_signature.v1",
    encoding_inventory: encodingInventory,
    encoding_inventory_sha256: sha256TextHex(canonicalJson(encodingInventory)),
    encoding_inventory_signature_basis: "SHA-256 over UTF-8 RFC8785-JCS canonical JSON of the complete emitted encoding_inventory array. Rows are ordered by exact serialized bytes descending, then dtype.",
    tensor_encoding_assignment_sha256: sha256TextHex(canonicalJson(assignmentSignatureRows)),
    tensor_encoding_assignment_signature_basis: "SHA-256 over UTF-8 RFC8785-JCS canonical JSON of tensor rows sorted by numeric index then name and projected to index, name, dtype, shape, and byte_length.",
    recipe_interpretation: "Content-addressed serialized encoding inventory and tensor-to-encoding assignment only. Quantizer command, converter build, importance matrix, source checkpoint, and recipe authenticity are not inferred.",
  };
}

function kvStorageScenarios(projection) {
  const perTokenElements = exactFrom(projection?.elements_per_token_per_batch);
  const contextElements = exactFrom(projection?.elements_at_context_batch_one);
  if (perTokenElements == null || contextElements == null) return [];
  return [8, 16, 32].map((storageBits) => ({
    storage_bits: storageBits,
    bytes_per_token_per_batch: exact(perTokenElements * BigInt(storageBits / 8)),
    bytes_at_declared_context_batch_one: exact(contextElements * BigInt(storageBits / 8)),
    evidence_class: "DERIVED_CONDITIONAL_SCENARIO",
  }));
}

function boundedGenerationConfig(config) {
  if (!config || typeof config !== "object" || Array.isArray(config)) return {};
  return Object.fromEntries(GENERATION_FIELDS
    .filter((key) => Object.hasOwn(config, key))
    .map((key) => [key, config[key]]));
}

function tokenIds(value) {
  const values = Array.isArray(value) ? value : value == null ? [] : [value];
  return values.filter((item) => Number.isSafeInteger(item));
}

function deploymentDeclarationContract(sidecar) {
  const document = sidecar?.document;
  if (!document) return {
    status: "not_selected",
    evidence_class: "NOT_ASSESSABLE",
    source: null,
    sha256: null,
    byte_length: null,
    coverage: {
      declared: 0,
      required: MEDICAL_DECLARATION_FIELDS.length,
      declared_fields: [],
      missing: [...MEDICAL_DECLARATION_FIELDS],
    },
    issues: [],
  };
  const issues = [];
  if (!document || typeof document !== "object" || Array.isArray(document)) issues.push({ code: "LLM_DEPLOYMENT_DECLARATION_NOT_OBJECT" });
  if (document?.schema !== "deepbom.on_device_llm_deployment_declaration.v1") issues.push({ code: "LLM_DEPLOYMENT_DECLARATION_SCHEMA_INVALID", observed: document?.schema ?? null });
  for (const key of MEDICAL_DECLARATION_FIELDS) {
    const value = document?.[key];
    if (value == null) continue;
    if (["task_and_acceptance_metrics", "evaluation_dataset_and_lineage"].includes(key)) {
      if (!Array.isArray(value) || !value.length || value.some((row) => !row || typeof row !== "object" || Array.isArray(row))) {
        issues.push({ code: "LLM_DEPLOYMENT_DECLARATION_ARRAY_INVALID", field: key });
      }
    } else if (typeof value !== "string" || !value.trim()) issues.push({ code: "LLM_DEPLOYMENT_DECLARATION_TEXT_INVALID", field: key });
  }
  const declared = MEDICAL_DECLARATION_FIELDS.filter((key) => {
    const value = document?.[key];
    return Array.isArray(value) ? value.length > 0 : typeof value === "string" && Boolean(value.trim());
  });
  const missing = MEDICAL_DECLARATION_FIELDS.filter((key) => !declared.includes(key));
  return {
    status: issues.length ? "invalid" : missing.length ? "declared_unverified_partial" : "declared_unverified_complete",
    evidence_class: "DECLARED_UNVERIFIED",
    source: sidecar?.path || null,
    sha256: sidecar?.sha256 || null,
    byte_length: sidecar?.byte_length ?? null,
    coverage: { declared: declared.length, required: MEDICAL_DECLARATION_FIELDS.length, declared_fields: declared, missing },
    issues,
  };
}

function hfSidecarContract(sidecars, architecture) {
  const tokenizerConfig = sidecars?.tokenizer_config?.document || null;
  const generationConfig = sidecars?.generation_config?.document || null;
  const config = sidecars?.architecture_config?.document || null;
  const issues = [];
  const vocabSize = positiveInteger(architecture?.vocabulary_size);
  for (const [source, document] of [["architecture_config", config], ["generation_config", generationConfig]]) {
    for (const key of ["bos_token_id", "eos_token_id", "pad_token_id"]) {
      for (const value of tokenIds(document?.[key])) {
        if (vocabSize != null && (value < 0 || value >= vocabSize)) issues.push({
          code: "LLM_SPECIAL_TOKEN_ID_OUT_OF_RANGE", source, field: key, value, vocabulary_size: vocabSize,
        });
      }
    }
  }
  for (const key of ["bos_token_id", "eos_token_id", "pad_token_id"]) {
    const declared = tokenIds(config?.[key]);
    const generated = tokenIds(generationConfig?.[key]);
    if (declared.length && generated.length && JSON.stringify(declared) !== JSON.stringify(generated)) issues.push({
      code: "LLM_GENERATION_TOKEN_ID_CONFLICT", field: key, architecture_config: declared, generation_config: generated,
    });
  }
  const embeddedTemplate = text(tokenizerConfig?.chat_template);
  const fileTemplate = sidecars?.chat_template || null;
  const embeddedTemplateSha = embeddedTemplate ? sha256TextHex(embeddedTemplate) : null;
  if (embeddedTemplateSha && fileTemplate?.sha256 && embeddedTemplateSha !== fileTemplate.sha256) issues.push({
    code: "LLM_CHAT_TEMPLATE_AMBIGUOUS", tokenizer_config_sha256: embeddedTemplateSha, chat_template_file_sha256: fileTemplate.sha256,
  });
  const identities = Object.values(sidecars || {}).filter((row) => row?.path && row?.sha256).map((row) => ({
    role: row.role,
    path: row.path,
    byte_length: row.byte_length,
    sha256: row.sha256,
  })).sort((left, right) => left.path.localeCompare(right.path));
  const tokenizerFiles = identities.filter((row) => ["tokenizer_config", "tokenizer_definition", "special_token_map", "chat_template"].includes(row.role));
  return {
    status: issues.length ? "invalid" : tokenizerFiles.length ? "partially_bound" : "not_selected",
    evidence_class: tokenizerFiles.length ? "OBSERVED/DECLARED_UNVERIFIED" : "NOT_ASSESSABLE",
    definition_files: tokenizerFiles,
    tokenizer_config: tokenizerConfig ? {
      model_max_length: positiveInteger(tokenizerConfig.model_max_length),
      padding_side: text(tokenizerConfig.padding_side),
      truncation_side: text(tokenizerConfig.truncation_side),
    } : null,
    chat_template: embeddedTemplateSha ? {
      status: fileTemplate?.sha256 === embeddedTemplateSha ? "identical_dual_declaration" : "declared_in_tokenizer_config",
      sha256: embeddedTemplateSha,
      source: "tokenizer_config.json",
    } : fileTemplate ? { status: "declared_in_file", sha256: fileTemplate.sha256, source: fileTemplate.path } : { status: "not_selected" },
    generation: generationConfig ? {
      status: "declared_unverified",
      source: sidecars.generation_config.path,
      sha256: sidecars.generation_config.sha256,
      values: boundedGenerationConfig(generationConfig),
    } : { status: "not_selected", values: {} },
    issues,
  };
}

function ggufContract(analysis) {
  const semantic = analysis?.gguf?.semantic_contract || {};
  const fields = semantic.architecture_fields || {};
  return {
    source_status: semantic.status || "not_declared",
    source_evidence_class: semantic.evidence_class || "NOT_ASSESSABLE",
    architecture: {
      family: text(semantic.architecture),
      kind: "dense_transformer_decoder",
      architecture_names: [],
      context_length: positiveInteger(semantic.context_length),
      vocabulary_size: positiveInteger(semantic.tokenizer?.vocabulary_count),
      hidden_size: positiveInteger(semantic.embedding_length),
      intermediate_size: positiveInteger(semantic.feed_forward_length),
      layer_count: positiveInteger(semantic.block_count),
      attention_head_count: positiveInteger(semantic.attention_head_count),
      kv_head_count: positiveInteger(semantic.attention_head_count_kv),
      head_width: positiveInteger(semantic.attention_key_length || semantic.derived_attention_head_width),
      gqa_query_heads_per_kv_head: positiveInteger(semantic.derived_gqa_query_heads_per_kv_head),
      position_encoding: semantic.position_encoding || {
        status: Object.keys(fields).some((key) => key.startsWith("rope.")) ? "declared" : "not_declared",
      },
    },
    tokenizer: {
      ...semantic.tokenizer,
      evidence_class: semantic.tokenizer?.status === "assessed" ? "OBSERVED" : "NOT_ASSESSABLE",
      definition_files: [],
      chat_template: semantic.tokenizer?.chat_template || { status: "not_declared" },
    },
    generation: { status: "not_embedded_as_runtime_policy", values: {} },
    kv_state_projection: semantic.kv_state_projection || null,
    compute_projection: semantic.compute_projection || null,
    issues: semantic.issues || [],
  };
}

function safeTensorsContract(analysis, sidecars) {
  const source = analysis?.safetensors?.hf_architecture_contract || {};
  const fields = source?.fields || {};
  const architecture = {
    family: text(source.model_type),
    kind: text(source.architecture_kind) || "dense_transformer_decoder",
    architecture_names: Array.isArray(source.architectures) ? source.architectures : [],
    context_length: positiveInteger(fields.max_position_embeddings),
    vocabulary_size: positiveInteger(fields.vocab_size),
    hidden_size: positiveInteger(fields.hidden_size),
    intermediate_size: positiveInteger(fields.intermediate_size),
    layer_count: positiveInteger(fields.num_hidden_layers),
    attention_head_count: positiveInteger(fields.num_attention_heads),
    kv_head_count: positiveInteger(fields.num_key_value_heads),
    head_width: positiveInteger(fields.head_dim),
    gqa_query_heads_per_kv_head: positiveInteger(source.gqa_query_heads_per_kv_head),
    position_encoding: source.position_encoding || { status: "not_projected_from_registered_config" },
    moe: source.moe_projection || null,
    ssm: source.recurrent_state_projection ? {
      state_size: positiveInteger(fields.state_size),
      convolution_kernel: positiveInteger(fields.conv_kernel),
      expansion_factor: Number.isFinite(fields.expand) && fields.expand > 0 ? fields.expand : null,
      time_step_rank: positiveInteger(fields.time_step_rank),
      recurrent_layer_count: positiveInteger(source.recurrent_state_projection?.layer_count),
    } : null,
  };
  const sidecar = hfSidecarContract(sidecars, architecture);
  const deploymentDeclaration = deploymentDeclarationContract(sidecars?.deployment_declaration);
  return {
    source_status: source.status || "not_assessed_config_not_selected",
    source_evidence_class: source.evidence_class || "NOT_ASSESSABLE",
    architecture,
    tokenizer: sidecar,
    generation: sidecar.generation,
    kv_state_projection: source.kv_state_projection || null,
    recurrent_state_projection: source.recurrent_state_projection || null,
    compute_projection: source.compute_projection || null,
    deployment_declaration: deploymentDeclaration,
    issues: [...(source.issues || []), ...sidecar.issues, ...deploymentDeclaration.issues],
  };
}

const EXPLICIT_TRANSFORMER_OPS = new Set([
  "ATTENTION", "MULTIHEADATTENTION", "GROUPQUERYATTENTION", "PAGEDATTENTION",
  "ROTARYEMBEDDING", "EMBEDLAYERNORMALIZATION", "SKIPLAYERNORMALIZATION",
  "MATMULNBITS", "GATEDRELATIVEPOSITIONBIAS", "BIASGELU", "FASTGELU",
]);

function normalizedOpName(value) {
  return String(value || "").replace(/[^A-Za-z0-9]/g, "").toUpperCase();
}

function exactLogicalTensorBytes(tensor) {
  const bits = {
    BOOL: 8, INT4: 4, UINT4: 4, INT8: 8, UINT8: 8, FLOAT16: 16, BFLOAT16: 16,
    INT16: 16, UINT16: 16, FLOAT32: 32, INT32: 32, UINT32: 32, FLOAT64: 64,
    INT64: 64, UINT64: 64,
  }[String(tensor?.dtype || "").toUpperCase()];
  const shape = Array.isArray(tensor?.shape_signature) && tensor.shape_signature.length
    ? tensor.shape_signature : tensor?.shape;
  if (!bits || !Array.isArray(shape) || !shape.length || shape.some((value) => !Number.isSafeInteger(Number(value)) || Number(value) < 0)) return null;
  const elements = shape.reduce((product, value) => product * BigInt(value), 1n);
  return exact((elements * BigInt(bits) + 7n) / 8n);
}

function graphTensorStorage(analysis) {
  const format = String(analysis?.format || "").toLowerCase();
  const structureBinding = format === "onnx" ? analysis?.onnx_external_data_structure_binding : null;
  if (structureBinding?.range_conservation_status === "complete") {
    const elementCount = exactFrom(structureBinding.declared_element_count);
    const byteLength = exactFrom(structureBinding.declared_payload_bytes);
    const encodings = (structureBinding.encoding_inventory || []).map((row) => {
      const rowElements = exactFrom(row.element_count);
      const rowBytes = exactFrom(row.declared_payload_bytes);
      return {
        dtype: String(row.dtype || "UNKNOWN"),
        tensor_count: Number(row.tensor_count || 0),
        element_count: rowElements != null && rowElements <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(rowElements) : null,
        element_count_decimal: rowElements == null ? null : String(rowElements),
        byte_length: rowBytes != null && rowBytes <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(rowBytes) : null,
        byte_length_decimal: rowBytes == null ? null : String(rowBytes),
        effective_bits_per_element: row.effective_bits_per_element ?? null,
      };
    });
    return {
      status: "assessed_serialized_constant_structure_payload_values_not_assessed",
      element_count: elementCount <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(elementCount) : null,
      element_count_decimal: String(elementCount),
      byte_length: byteLength <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(byteLength) : null,
      byte_length_decimal: String(byteLength),
      unique_byte_length: structureBinding.unique_payload_bytes?.number ?? null,
      unique_byte_length_decimal: structureBinding.unique_payload_bytes?.decimal ?? null,
      effective_bits_per_element: elementCount > 0n ? Number(byteLength * 8_000_000n / elementCount) / 1_000_000 : null,
      numerical_payload_decode: structureBinding.numerical_payload_decode,
      encodings,
      format,
    };
  }
  const tensors = Array.isArray(analysis?.tensors) ? analysis.tensors : [];
  const constants = tensors.filter((tensor) => tensor?.constant_buffer === true
    || Number(tensor?.initializer_available_bytes || tensor?.initializer_bytes || tensor?.buffer_data_length || 0) > 0);
  const size = analysis?.size_breakdown || {};
  const byteLength = Number.isSafeInteger(Number(size.available_initializer_bytes))
    ? Number(size.available_initializer_bytes)
    : Number.isSafeInteger(Number(size.constant_bytes)) ? Number(size.constant_bytes) : null;
  const elementCount = Number.isSafeInteger(Number(size.available_initializer_scalar_elements))
    ? Number(size.available_initializer_scalar_elements)
    : Number.isSafeInteger(Number(size.stored_scalar_elements)) ? Number(size.stored_scalar_elements) : null;
  const byDtype = new Map();
  for (const tensor of constants) {
    const dtype = String(tensor.dtype || "UNKNOWN");
    const bytes = Number(tensor.initializer_available_bytes || tensor.initializer_bytes || tensor.buffer_data_length || 0);
    const elements = Number(tensor.initializer_stored_elements || 0);
    const row = byDtype.get(dtype) || { dtype, tensor_count: 0, element_count: 0, byte_length: 0 };
    row.tensor_count += 1;
    if (Number.isSafeInteger(elements) && elements >= 0) row.element_count += elements;
    if (Number.isSafeInteger(bytes) && bytes >= 0) row.byte_length += bytes;
    byDtype.set(dtype, row);
  }
  const encodings = [...byDtype.values()].sort((left, right) => right.byte_length - left.byte_length || left.dtype.localeCompare(right.dtype)).map((row) => ({
    dtype: row.dtype,
    tensor_count: row.tensor_count,
    element_count: row.element_count,
    element_count_decimal: String(row.element_count),
    byte_length: row.byte_length,
    byte_length_decimal: String(row.byte_length),
    effective_bits_per_element: row.element_count ? row.byte_length * 8 / row.element_count : null,
  }));
  return {
    status: byteLength == null ? "not_assessed_constant_storage" : "assessed_serialized_constant_storage",
    element_count: elementCount,
    element_count_decimal: elementCount == null ? null : String(elementCount),
    byte_length: byteLength,
    byte_length_decimal: byteLength == null ? null : String(byteLength),
    effective_bits_per_element: byteLength != null && elementCount ? byteLength * 8 / elementCount : null,
    encodings,
    format,
  };
}

function graphLlmEvidence(analysis) {
  const ops = Array.isArray(analysis?.ops) ? analysis.ops : [];
  const tensors = Array.isArray(analysis?.tensors) ? analysis.tensors : [];
  const rows = ops.map((op, position) => ({
    op_index: Number.isSafeInteger(Number(op?.index)) ? Number(op.index) : position,
    name: String(op?.name || "UNKNOWN"),
    domain: text(op?.domain),
    version: Number.isSafeInteger(Number(op?.version)) ? Number(op.version) : null,
    normalized_name: normalizedOpName(op?.name),
  }));
  const count = (names) => rows.filter((row) => names.has(row.normalized_name)).length;
  const explicit = rows.filter((row) => EXPLICIT_TRANSFORMER_OPS.has(row.normalized_name));
  const primitive = {
    matrix_multiply: count(new Set(["MATMUL", "BATCHMATMUL", "FULLYCONNECTED", "GEMM"])),
    softmax: count(new Set(["SOFTMAX", "LOGSOFTMAX"])),
    normalization: count(new Set(["LAYERNORMALIZATION", "RMSNORMALIZATION", "L2NORMALIZATION"])),
    embedding_gather: count(new Set(["GATHER", "GATHERND", "EMBEDLAYERNORMALIZATION"])),
    elementwise_activation: count(new Set(["GELU", "FASTGELU", "SILU", "SWISH", "BIASGELU"])),
  };
  const transformerMotif = primitive.matrix_multiply >= 2 && primitive.softmax >= 1 && primitive.normalization >= 1;
  const interfaceRows = [...(analysis?.inputs || []), ...(analysis?.outputs || [])];
  const stateCandidates = interfaceRows.filter((tensor) => /(?:^|[._/])(past|present|cache|key|value|kv)(?:[._/]|$)/i.test(String(tensor?.name || ""))).map((tensor) => ({
    tensor_index: Number.isSafeInteger(Number(tensor?.index)) ? Number(tensor.index) : null,
    name: String(tensor?.name || ""),
    dtype: String(tensor?.dtype || "UNKNOWN"),
    shape: Array.isArray(tensor?.shape_signature) && tensor.shape_signature.length ? tensor.shape_signature : Array.isArray(tensor?.shape) ? tensor.shape : [],
    logical_bytes_if_static: exactLogicalTensorBytes(tensor),
    classification: "serialized_name_candidate_not_semantic_proof",
  }));
  return {
    schema: "deepbom.serialized_llm_graph_evidence.v1",
    status: explicit.length ? "explicit_transformer_operator_evidence_present"
      : transformerMotif ? "transformer_like_motif_candidate" : "no_serialized_llm_specific_evidence",
    evidence_class: explicit.length ? "OBSERVED_SERIALIZED_OPERATOR_SEMANTICS"
      : transformerMotif ? "OBSERVED_GRAPH_WITH_HEURISTIC_CLASSIFICATION" : "OBSERVED_NOT_DETECTED",
    graph_op_count: rows.length,
    graph_signature_sha256: sha256TextHex(canonicalJson(rows)),
    explicit_operator_count: explicit.length,
    explicit_operators: explicit,
    primitive_counts: primitive,
    transformer_motif_candidate: transformerMotif,
    external_state_candidate_count: stateCandidates.length,
    external_state_candidates: stateCandidates,
    interpretation_boundary: "An explicit Attention-family operator establishes only its serialized operator semantics, not that the model is a language model. A decomposed MatMul/Softmax/normalization motif is heuristic and may represent vision or another attention workload. Interface names identify candidates only. Architecture family, layer count, KV layout, tokenizer, generation policy, task, and accuracy remain unbound without identity-bound metadata.",
  };
}

function graphContainerContract(analysis) {
  const graph = graphLlmEvidence(analysis);
  return {
    source_status: graph.status,
    source_evidence_class: graph.evidence_class,
    architecture: {
      family: null,
      kind: "not_established_from_serialized_graph_alone",
      architecture_names: [],
      context_length: null,
      vocabulary_size: null,
      hidden_size: null,
      intermediate_size: null,
      layer_count: null,
      attention_head_count: null,
      kv_head_count: null,
      head_width: null,
      gqa_query_heads_per_kv_head: null,
      position_encoding: { status: "not_established" },
    },
    tokenizer: { status: "not_embedded_or_identity_bound", evidence_class: "NOT_ASSESSABLE", definition_files: [], chat_template: { status: "not_selected" } },
    generation: { status: "not_embedded_or_identity_bound", values: {} },
    kv_state_projection: null,
    recurrent_state_projection: null,
    compute_projection: null,
    graph,
    storage: graphTensorStorage(analysis),
    issues: [],
  };
}

export function buildOnDeviceLlmContract(analysis = {}, { sidecars = {} } = {}) {
  const format = String(analysis?.format || "").toLowerCase();
  const normalized = format === "gguf" ? ggufContract(analysis)
    : format === "safetensors" ? safeTensorsContract(analysis, sidecars)
      : ["onnx", "tflite"].includes(format) ? graphContainerContract(analysis) : null;
  if (!normalized) return {
    schema: ON_DEVICE_LLM_CONTRACT_SCHEMA,
    status: "not_applicable_format",
    evidence_class: "NOT_APPLICABLE",
    format,
  };
  const storage = normalized.storage || analysis?.tensor_storage_summary || {};
  const encodingContract = serializedEncodingContract(analysis, storage);
  const layerStorage = buildLlmLayerStorageLedger(analysis, normalized.architecture, storage);
  const architectureAssessed = ["gguf", "safetensors"].includes(format)
    && /assessed/.test(normalized.source_status) && !/not_assessed/.test(normalized.source_status);
  const layerStorageInvalid = String(layerStorage.status || "").startsWith("invalid");
  const invalid = normalized.source_status === "invalid" || normalized.source_status === "invalid_config" || normalized.issues.length > 0 || layerStorageInvalid;
  const deploymentDeclaration = normalized.deployment_declaration || deploymentDeclarationContract(null);
  const contract = {
    schema: ON_DEVICE_LLM_CONTRACT_SCHEMA,
    status: invalid ? "invalid" : architectureAssessed ? "assessed_static_artifact_contract" : "partial_architecture_contract",
    evidence_class: architectureAssessed ? "OBSERVED/SOURCE_BACKED/DERIVED" : normalized.source_evidence_class,
    format,
    artifact_role: format === "gguf" ? text(analysis?.metadata?.["general.type"]) || "model"
      : format === "safetensors" ? "model_tensor_repository" : "serialized_executable_graph",
    architecture: normalized.architecture,
    storage: {
      status: storage.status || "not_assessed",
      serialized_parameter_count: storage.element_count ?? null,
      serialized_parameter_count_decimal: storage.element_count_decimal ?? null,
      serialized_tensor_bytes: storage.byte_length ?? null,
      serialized_tensor_bytes_decimal: storage.byte_length_decimal ?? null,
      effective_bits_per_parameter: storage.effective_bits_per_element ?? null,
      ...encodingContract,
      layer_storage: layerStorage,
      basis: "Reused from deepbom.tensor_storage_summary.v1; no independent recount.",
    },
    tokenizer: normalized.tokenizer,
    generation: normalized.generation,
    state: {
      kv_projection: normalized.kv_state_projection,
      recurrent_projection: normalized.recurrent_state_projection,
      storage_scenarios: kvStorageScenarios(normalized.kv_state_projection),
      recurrent_storage_scenarios: buildStateStorageScenarios(normalized.recurrent_state_projection),
      scenario_matrix: [],
      scenario_boundary: normalized.kv_state_projection && normalized.recurrent_state_projection
        ? "Hybrid byte rows sum transformer KV elements for the selected context with context-independent SSM recurrent elements, then apply batch and storage width. They are not runtime allocation, residency, paging, or device-memory measurements."
        : normalized.recurrent_state_projection
          ? "Recurrent-state byte rows are conditional on batch and storage width, remain context-independent under the pinned Mamba cache shape, and are not runtime allocation, residency, paging, or device-memory measurements."
        : "KV byte rows are conditional on context, batch, and cache storage width. They are not runtime allocation, residency, paging, or device-memory measurements.",
    },
    compute: {
      projection: normalized.compute_projection,
      boundary: "Architecture equations are static scenarios, not a serialized execution graph, lowered kernel count, latency estimate, or throughput measurement.",
    },
    serialized_graph: normalized.graph || null,
    runtime_contract: null,
    tensorrt_llm: null,
    memory_feasibility: null,
    static_memory_placement: null,
    medical_ai_claim_boundary: {
      status: deploymentDeclaration.status === "not_selected" ? "not_established_by_model_artifact" : deploymentDeclaration.status,
      declaration: deploymentDeclaration,
      required_external_evidence: deploymentDeclaration.coverage?.missing || [...MEDICAL_DECLARATION_FIELDS],
      established: ["artifact_identity", "serialized_tensor_storage", "declared_architecture_fields", "conditional_state_and_compute_scenarios_where_complete"],
      not_established: ["task_accuracy", "clinical_validity", "clinical_utility", "safety_effectiveness", "runtime_assignment", "release_readiness"],
    },
    issue_count: normalized.issues.length + (layerStorageInvalid ? 1 : 0),
    issues: [...normalized.issues, ...(layerStorageInvalid ? [{ code: "LLM_LAYER_STORAGE_CONTRACT_INVALID", status: layerStorage.status }] : [])],
  };
  contract.state.scenario_matrix = buildLlmStateScenarioMatrix(contract);
  contract.runtime_contract = assessOnDeviceLlmRuntimeManifest(sidecars?.runtime_manifest, analysis, contract);
  contract.tensorrt_llm = buildTensorRtLlmContract(analysis, contract, {
    engineConfig: sidecars?.tensorrt_llm_engine_config || null,
    binding: sidecars?.tensorrt_llm_binding || null,
  });
  contract.runtime_contract.gguf_backend_prerequisites = format === "gguf" ? analysis?.gguf?.backend_compatibility || null : null;
  contract.memory_feasibility = buildLlmMemoryFeasibility(contract);
  contract.static_memory_placement = buildLlmStaticMemoryPlacement(contract, analysis, sidecars?.static_memory_profile);
  if (contract.runtime_contract.status === "invalid") {
    contract.status = "invalid";
    contract.issues = [...contract.issues, ...contract.runtime_contract.issues];
    contract.issue_count = contract.issues.length;
  }
  return contract;
}

export function validateOnDeviceLlmContract(analysis = {}) {
  const contract = analysis?.on_device_llm || {};
  const format = String(analysis?.format || "").toLowerCase();
  const errors = [];
  if (contract.schema !== ON_DEVICE_LLM_CONTRACT_SCHEMA) errors.push("schema_mismatch");
  if (!["gguf", "safetensors", "onnx", "tflite"].includes(format) || contract.format !== format) errors.push("format_mismatch");
  const storage = ["onnx", "tflite"].includes(format) ? graphTensorStorage(analysis) : analysis?.tensor_storage_summary || {};
  if (String(contract.storage?.serialized_parameter_count_decimal ?? "") !== String(storage.element_count_decimal ?? "")
    || String(contract.storage?.serialized_tensor_bytes_decimal ?? "") !== String(storage.byte_length_decimal ?? "")
    || String(contract.storage?.effective_bits_per_parameter ?? "") !== String(storage.effective_bits_per_element ?? "")) errors.push("storage_reuse_mismatch");
  const expectedEncodingContract = serializedEncodingContract(analysis, storage);
  for (const key of ["encoding_signature_schema", "encoding_inventory_sha256", "encoding_inventory_signature_basis",
    "tensor_encoding_assignment_sha256", "tensor_encoding_assignment_signature_basis", "recipe_interpretation"]) {
    if (contract.storage?.[key] !== expectedEncodingContract[key]) errors.push(`encoding_contract_mismatch:${key}`);
  }
  if (JSON.stringify(contract.storage?.encoding_inventory || []) !== JSON.stringify(expectedEncodingContract.encoding_inventory)) errors.push("encoding_contract_mismatch:encoding_inventory");
  const source = format === "gguf" ? analysis?.gguf?.semantic_contract || {}
    : format === "safetensors" ? analysis?.safetensors?.hf_architecture_contract || {} : null;
  const fields = source?.fields || {};
  const expectedArchitecture = format === "gguf" ? {
    family: source.architecture ?? null,
    context_length: source.context_length ?? null,
    vocabulary_size: source.tokenizer?.vocabulary_count ?? null,
    hidden_size: source.embedding_length ?? null,
    intermediate_size: source.feed_forward_length ?? null,
    layer_count: source.block_count ?? null,
    attention_head_count: source.attention_head_count ?? null,
    kv_head_count: source.attention_head_count_kv ?? null,
  } : format === "safetensors" ? {
    family: source.model_type ?? null,
    context_length: fields.max_position_embeddings ?? null,
    vocabulary_size: fields.vocab_size ?? null,
    hidden_size: fields.hidden_size ?? null,
    intermediate_size: fields.intermediate_size ?? null,
    layer_count: fields.num_hidden_layers ?? null,
    attention_head_count: fields.num_attention_heads ?? null,
    kv_head_count: fields.num_key_value_heads ?? null,
  } : {
    family: null,
    context_length: null,
    vocabulary_size: null,
    hidden_size: null,
    intermediate_size: null,
    layer_count: null,
    attention_head_count: null,
    kv_head_count: null,
  };
  for (const [key, value] of Object.entries(expectedArchitecture)) if ((contract.architecture?.[key] ?? null) !== value) errors.push(`architecture_mismatch:${key}`);
  const expectedLayerStorage = buildLlmLayerStorageLedger(analysis, contract.architecture, storage);
  if (canonicalJson(contract.storage?.layer_storage) !== canonicalJson(expectedLayerStorage)) errors.push("layer_storage_recomputation_mismatch");
  if (["onnx", "tflite"].includes(format)) {
    const expectedGraph = graphLlmEvidence(analysis);
    if (canonicalJson(contract.serialized_graph) !== canonicalJson(expectedGraph)) errors.push("serialized_llm_graph_mismatch");
    if (contract.state?.kv_projection || contract.compute?.projection) errors.push("graph_only_architecture_projection_overclaim");
  }
  const projection = contract.state?.kv_projection;
  const perToken = exactFrom(projection?.elements_per_token_per_batch);
  const atContext = exactFrom(projection?.elements_at_context_batch_one);
  const scenarios = contract.state?.storage_scenarios || [];
  if (projection) {
    if (perToken == null || atContext == null || scenarios.length !== 3) errors.push("kv_scenario_cardinality_invalid");
    if (perToken != null && atContext != null) {
      for (const [index, bits] of [8, 16, 32].entries()) {
        const row = scenarios[index];
        if (row?.storage_bits !== bits
          || exactFrom(row?.bytes_per_token_per_batch) !== perToken * BigInt(bits / 8)
          || exactFrom(row?.bytes_at_declared_context_batch_one) !== atContext * BigInt(bits / 8)) errors.push(`kv_scenario_mismatch:${bits}`);
      }
    }
  } else if (scenarios.length) errors.push("kv_scenario_without_projection");
  const recurrent = contract.state?.recurrent_projection;
  const recurrentElements = exactFrom(recurrent?.recurrent_state_elements_all_layers_per_batch);
  const recurrentScenarios = contract.state?.recurrent_storage_scenarios || [];
  if (recurrent) {
    if (recurrentElements == null || recurrentScenarios.length !== 9) errors.push("recurrent_scenario_cardinality_invalid");
    if (recurrentElements != null) for (const row of recurrentScenarios) {
      if (exactFrom(row?.logical_bytes) !== recurrentElements * BigInt(row.batch_size) * BigInt(row.storage_bits / 8)) errors.push(`recurrent_scenario_mismatch:${row.batch_size}:${row.storage_bits}`);
    }
  } else if (recurrentScenarios.length) errors.push("recurrent_scenario_without_projection");
  if (projection && recurrent && !(contract.state?.scenario_matrix || []).every((row) => row.state_kind === "hybrid_kv_ssm")) {
    errors.push("hybrid_kv_ssm_scenario_kind_invalid");
  }
  const bundleFiles = new Map((analysis?.artifact_bundle?.files || []).map((row) => [row.path, row]));
  const boundFiles = [
    ...(contract.tokenizer?.definition_files || []),
    ...(contract.generation?.source ? [{ path: contract.generation.source, sha256: contract.generation.sha256, role: "generation_config" }] : []),
    ...(contract.medical_ai_claim_boundary?.declaration?.source ? [{
      path: contract.medical_ai_claim_boundary.declaration.source,
      sha256: contract.medical_ai_claim_boundary.declaration.sha256,
      role: "deployment_declaration",
    }] : []),
    ...(contract.runtime_contract?.source ? [{ path: contract.runtime_contract.source, sha256: contract.runtime_contract.source_sha256, role: "llm_runtime_manifest" }] : []),
    ...(contract.static_memory_placement?.source ? [{ path: contract.static_memory_placement.source, sha256: contract.static_memory_placement.source_sha256, role: "llm_static_memory_profile" }] : []),
  ];
  for (const row of boundFiles) {
    const bundled = bundleFiles.get(row.path);
    if (!bundled || bundled.sha256 !== row.sha256 || bundled.role !== row.role || bundled.required !== true) errors.push(`sidecar_binding_mismatch:${row.path}`);
  }
  const medical = contract.medical_ai_claim_boundary || {};
  const coverage = medical.declaration?.coverage || {};
  const missing = coverage.missing || [];
  if (new Set(missing).size !== missing.length || missing.some((key) => !MEDICAL_DECLARATION_FIELDS.includes(key))
    || Number(coverage.declared || 0) + missing.length !== MEDICAL_DECLARATION_FIELDS.length
    || Number(coverage.required || 0) !== MEDICAL_DECLARATION_FIELDS.length) errors.push("medical_declaration_coverage_invalid");
  if (contract.issue_count !== (contract.issues || []).length) errors.push("issue_count_mismatch");
  if ((contract.issues || []).length || contract.status === "invalid" || medical.declaration?.status === "invalid") errors.push("contract_contains_invalid_evidence");
  errors.push(...validateOnDeviceLlmRuntimeContract(contract.runtime_contract, analysis, contract));
  errors.push(...validateTensorRtLlmContract(analysis, contract));
  errors.push(...validateLlmMemoryFeasibility(contract));
  errors.push(...validateLlmStaticMemoryPlacement(contract.static_memory_placement, contract, analysis));
  return { valid: errors.length === 0, errors };
}
