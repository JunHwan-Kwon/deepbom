import { inspectArtifactBundle, readArtifactBundle } from "../web/lib/artifact-bundle.js";
import { parseCoreMlModel } from "../web/lib/coreml-metadata-adapter.js";
import { buildDeploymentContractDocuments } from "../web/lib/report-export-contracts.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { buildHfSafeTensorsContract } from "../web/lib/hf-safetensors-contract.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { buildEngineeringEvidenceDocument } from "../web/lib/report-evidence.js";
import { buildPublicCycloneDx17ArtifactContract } from "../web/lib/public-cyclonedx-export.js";
import { registerSafeTensorsSerializedConformance } from "../web/lib/report-conformance-serialized-containers.js";
import { assertCycloneDx17 } from "./cyclonedx-17-schema.mjs";

function expect(value, message) { if (!value) throw new Error(message); }
async function expectReject(promise, pattern, message) {
  try { await promise; } catch (error) {
    if (String(error?.message || error).includes(pattern)) return;
    throw new Error(`${message}: unexpected error ${error?.message || error}`);
  }
  throw new Error(`${message}: expected rejection`);
}

function concat(...chunks) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const result = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.length; }
  return result;
}

function varint(value) {
  let remaining = BigInt(value);
  const bytes = [];
  do {
    let byte = Number(remaining & 0x7fn);
    remaining >>= 7n;
    if (remaining) byte |= 0x80;
    bytes.push(byte);
  } while (remaining);
  return Uint8Array.from(bytes);
}
const key = (field, wire) => varint(field * 8 + wire);
const scalar = (field, value) => concat(key(field, 0), varint(value));
const bytesField = (field, bytes) => concat(key(field, 2), varint(bytes.length), bytes);
const textField = (field, value) => bytesField(field, new TextEncoder().encode(value));

function feature(name, shape, dtype) {
  const packedShape = concat(...shape.map(varint));
  const arrayType = concat(bytesField(1, packedShape), scalar(2, dtype));
  const featureType = bytesField(5, arrayType);
  return concat(textField(1, name), bytesField(3, featureType));
}

function milTensorType(shape, dtype = 11) {
  const dimensions = shape.map((size) => bytesField(3, bytesField(1, scalar(1, size))));
  return bytesField(1, concat(scalar(1, dtype), scalar(2, shape.length), ...dimensions));
}

function milNamedValue(name, shape, dtype = 11) {
  return concat(textField(1, name), bytesField(2, milTensorType(shape, dtype)));
}

const metadata = concat(textField(1, "fixture"), textField(2, "1.0"), textField(3, "test author"), textField(4, "BSD-3-Clause"));
const description = concat(
  bytesField(1, feature("image", [1, 3, 4, 4], 65568)),
  bytesField(10, feature("image", [1, 3, 4, 4], 65568)),
  bytesField(100, metadata),
);
const milBlock = textField(2, "image");
const milFunction = concat(bytesField(1, milNamedValue("image", [1, 3, 4, 4])), textField(2, "CoreML5"), bytesField(3, concat(textField(1, "CoreML5"), bytesField(2, milBlock))));
const milProgram = concat(scalar(1, 1), bytesField(2, concat(textField(1, "main"), bytesField(2, milFunction))));
const coreMlBytes = concat(scalar(1, 6), bytesField(2, description), bytesField(502, milProgram));
const coreMl = parseCoreMlModel(coreMlBytes, "model.mlmodel");
expect(coreMl.coreml.model_type === "mlProgram", "Core ML model type");
expect(coreMl.inputs[0].shape.join("x") === "1x3x4x4" && coreMl.outputs[0].dtype === "FLOAT32", "Core ML interface contract");
expect(coreMl.metadata_presence.metadata_author === "test author", "Core ML metadata");

function file(path, bytes, type = "application/octet-stream") {
  const value = new File([bytes], path.split("/").at(-1), { type });
  Object.defineProperty(value, "webkitRelativePath", { value: path });
  return value;
}

const manifest = {
  fileFormatVersion: "1.0.0",
  rootModelIdentifier: "model-id",
  itemInfoEntries: {
    "model-id": { path: "com.apple.CoreML/model.mlmodel", name: "model.mlmodel", author: "com.apple.CoreML", description: "Core ML model specification" },
    "weights-id": { path: "com.apple.CoreML/weights", name: "weights", author: "com.apple.CoreML", description: "Core ML model weights" },
  },
};
const coreFiles = [
  file("Fixture.mlpackage/Manifest.json", JSON.stringify(manifest), "application/json"),
  file("Fixture.mlpackage/Data/com.apple.CoreML/model.mlmodel", coreMlBytes),
  file("Fixture.mlpackage/Data/com.apple.CoreML/weights/weight.bin", Uint8Array.from([9, 8, 7, 6])),
  file("Fixture.mlpackage/Info.plist", "fixture package metadata", "text/plain"),
];
const corePlan = await inspectArtifactBundle(coreFiles);
expect(corePlan.rootFile.name === "model.mlmodel", "Core ML root manifest binding");
const coreBundle = await readArtifactBundle(coreFiles);
expect(coreBundle.analysis.artifact_bundle.files.length === 4, "Core ML package inventory includes selected unmanifested files");
expect(coreBundle.analysis.artifact_bundle.files.some((item) => item.role === "weights"), "Core ML weights role");
expect(coreBundle.analysis.artifact_bundle.files.some((item) => item.role === "unreferenced_package_file" && item.required === false), "Core ML unmanifested file remains hash-bound without becoming a required package item");
expect(/^[a-f0-9]{64}$/.test(coreBundle.analysis.model_sha256), "Core ML canonical bundle digest");
const coreDocuments = buildDeploymentContractDocuments(coreBundle.analysis, { generatedAt: "2026-08-03T00:00:00.000Z" });
assertCycloneDx17(coreDocuments.documents.cyclonedx_evidence, "Core ML package evidence BOM");
expect(coreDocuments.documents.cyclonedx_evidence.dependencies[0].dependsOn.length === 4, "Core ML subject dependency completeness");
const coreMlBom = buildMlBomDocument(coreBundle.analysis, { timestamp: "2026-08-03T00:00:00.000Z" });
assertCycloneDx17(coreMlBom, "Core ML canonical ML-BOM");
expect(!coreMlBom.components.some((item) => item.name === "training-dataset" || item.name === "browser-local-inference-runtime"), "unobserved components stay absent");

function safeTensorFile(path, tensorName, value) {
  const header = new TextEncoder().encode(JSON.stringify({ [tensorName]: { dtype: "U8", shape: [1], data_offsets: [0, 1] } }));
  const prefix = new Uint8Array(8);
  new DataView(prefix.buffer).setBigUint64(0, BigInt(header.length), true);
  return file(path, concat(prefix, header, Uint8Array.of(value)));
}

function safeTensorPayloadFile(path, descriptors, payload) {
  const header = new TextEncoder().encode(JSON.stringify(descriptors));
  const prefix = new Uint8Array(8);
  new DataView(prefix.buffer).setBigUint64(0, BigInt(header.length), true);
  return file(path, concat(prefix, header, payload));
}

const shardIndex = {
  metadata: { total_size: 2 },
  weight_map: { "a.weight": "model-00001-of-00002.safetensors", "b.weight": "model-00002-of-00002.safetensors" },
};
const shardFiles = [
  file("repo/model.safetensors.index.json", JSON.stringify(shardIndex), "application/json"),
  safeTensorFile("repo/model-00001-of-00002.safetensors", "a.weight", 1),
  safeTensorFile("repo/model-00002-of-00002.safetensors", "b.weight", 2),
  file("repo/config.json", JSON.stringify({ model_type: "fixture" }), "application/json"),
];
const shardBundle = await readArtifactBundle(shardFiles);
expect(shardBundle.analysis.tensor_count === 2, "SafeTensors shard tensor conservation");
expect(shardBundle.analysis.tensors.map((item) => item.index).join(",") === "0,1", "SafeTensors shard aggregate tensor indices");
expect(shardBundle.analysis.safetensors.tensor_count === 2, "SafeTensors shard aggregate metadata tensor count");
expect(shardBundle.analysis.safetensors.payload_byte_length === 2, "SafeTensors shard aggregate payload bytes");
expect(shardBundle.analysis.safetensors.index_binding_status === "complete_bidirectional", "SafeTensors index binding");
expect(shardBundle.analysis.safetensors.hf_config_status === "parsed_and_bundle_bound", "SafeTensors selected config binding");
expect(shardBundle.analysis.safetensors.hf_architecture_contract.status === "not_assessed_unregistered_architecture", "Unknown HF architecture remains explicit rather than inferred");
expect(shardBundle.analysis.tensor_numerical_integrity.decoder_source?.pytorch_commit, "SafeTensors shard summary retains low-precision decoder provenance");
const shardDocuments = buildDeploymentContractDocuments(shardBundle.analysis, { generatedAt: "2026-08-03T00:00:00.000Z" });
assertCycloneDx17(shardDocuments.documents.cyclonedx_evidence, "SafeTensors shard evidence BOM");
expect(shardBundle.analysis.artifact_bundle.files.some((item) => item.role === "architecture_config" && item.required === true), "SafeTensors config used by analysis is a required hash-bound evidence file");
expect(shardDocuments.documents.cyclonedx_evidence.dependencies[0].dependsOn.length === 4, "SafeTensors shard dependencies");

const tinyLlamaConfig = {
  model_type: "llama",
  architectures: ["LlamaForCausalLM"],
  vocab_size: 8,
  hidden_size: 4,
  intermediate_size: 8,
  num_hidden_layers: 1,
  num_attention_heads: 2,
  num_key_value_heads: 1,
  head_dim: 2,
  max_position_embeddings: 16,
  torch_dtype: "float16",
  tie_word_embeddings: true,
  bos_token_id: 1,
  eos_token_id: 2,
  pad_token_id: 0,
};
const tinyChatTemplate = "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}";
const tinyDeploymentDeclaration = {
  schema: "deepbom.on_device_llm_deployment_declaration.v1",
  intended_use: "Research-only local text classification fixture.",
  patient_population: "Synthetic test records only; no patient population claim.",
  clinical_workflow: "Not integrated into clinical care.",
  human_oversight: "Every output requires researcher review.",
  task_and_acceptance_metrics: [{ name: "fixture_exact_match", acceptance_threshold: 1, dataset_id: "tiny-fixture-v1" }],
  evaluation_dataset_and_lineage: [{ id: "tiny-fixture-v1", version: "1" }],
  prompt_and_output_constraints: "Single-turn fixture prompt; output is not medical advice.",
  privacy_and_phi_handling: "Synthetic content only; PHI is prohibited.",
  postmarket_monitoring_plan: "Not applicable to this non-clinical fixture.",
};
const singleSafeFiles = [
  safeTensorPayloadFile("tiny/model.safetensors", {
    "model.embed_tokens.weight": { dtype: "U8", shape: [8, 4], data_offsets: [0, 32] },
  }, Uint8Array.from({ length: 32 }, (_, index) => index)),
  file("tiny/config.json", JSON.stringify(tinyLlamaConfig), "application/json"),
  file("tiny/tokenizer_config.json", JSON.stringify({ model_max_length: 16, padding_side: "right", chat_template: tinyChatTemplate }), "application/json"),
  file("tiny/generation_config.json", JSON.stringify({ do_sample: false, max_new_tokens: 8, bos_token_id: 1, eos_token_id: 2, pad_token_id: 0 }), "application/json"),
  file("tiny/chat_template.jinja", tinyChatTemplate, "text/plain"),
  file("tiny/tokenizer.json", JSON.stringify({ version: "1.0", model: { type: "WordLevel", vocab: {} } }), "application/json"),
  file("tiny/deepbom.deployment.json", JSON.stringify(tinyDeploymentDeclaration), "application/json"),
];
const singlePlan = await inspectArtifactBundle(singleSafeFiles);
expect(singlePlan.kind === "safetensors_single_repository" && singlePlan.config.model_type === "llama", "Single SafeTensors repository and config resolution");
const singleBundle = await readArtifactBundle(singleSafeFiles);
const singleContract = singleBundle.analysis.safetensors.hf_architecture_contract;
expect(singleBundle.analysis.artifact_bundle.kind === "safetensors_single_file_repository", "Single SafeTensors repository bundle identity");
expect(/^[a-f0-9]{64}$/.test(singleBundle.analysis.artifact_bundle.model_source_sha256)
  && singleBundle.analysis.artifact_bundle.model_source_file_count === 2
  && singleBundle.analysis.artifact_bundle.model_source_roles.join(",") === "shard_index,architecture_config,quantization_config,tensor_shard"
  && singleBundle.analysis.artifact_bundle.model_source_hash_basis.includes("quantization_config"),
"SafeTensors model-source identity is independently reproducible from build-input roles");
expect(singleBundle.analysis.safetensors.index_binding_status === "not_required_single_file", "Single SafeTensors index applicability");
expect(singleContract.status === "partial" && singleContract.model_type === "llama", "Registered architecture with partial canonical tensor coverage");
expect(singleContract.kv_cache_elements_per_token_per_batch_decimal === "4", "Exact KV-cache element formula");
expect(singleContract.kv_cache_bytes_per_token_per_batch_if_cache_dtype_matches_declared_decimal === "8", "Conditional KV-cache byte formula");
expect(singleContract.compute_projection.dense_projection_macs_per_layer_per_token.decimal === "144", "Exact canonical decoder dense projection MACs");
expect(singleContract.compute_projection.prefill_transformer_core_macs_at_declared_context.decimal === "3392", "Exact canonical decoder prefill core MACs");
expect(singleContract.compute_projection.decode_with_one_logit_position_macs.decimal === "304", "Exact canonical decoder one-step MACs with logits");
expect(singleContract.tensor_contract.canonical_expected_parameter_count_decimal === "188"
  && singleContract.tensor_contract.canonical_observed_parameter_count_decimal === "32"
  && singleContract.tensor_contract.checkpoint_parameter_count_decimal === "32", "Canonical/checkpoint parameter conservation");
expect(singleBundle.analysis.tensor_storage_summary.element_count_decimal === "32" && singleBundle.analysis.tensor_numerical_integrity.byte_conservation_status === "complete", "Single SafeTensors storage and payload conservation");

const placementBaseFiles = [
  safeTensorPayloadFile("placement/model.safetensors", {
    "model.embed_tokens.weight": { dtype: "U8", shape: [8, 4], data_offsets: [0, 32] },
    "model.layers.0.self_attn.q_proj.weight": { dtype: "U8", shape: [4, 4], data_offsets: [32, 48] },
  }, Uint8Array.from({ length: 48 }, (_, index) => index)),
  file("placement/config.json", JSON.stringify(tinyLlamaConfig), "application/json"),
];
const placementBase = await readArtifactBundle(placementBaseFiles);
const placementSourceSha = placementBase.analysis.artifact_bundle.model_source_sha256;
const placementProfile = {
  schema: "deepbom.llm_static_memory_profile.v1",
  artifact: { format: "safetensors", sha256: placementSourceSha },
  capacities: { cpu_bytes: "1000", accelerator_bytes: "1000" },
  reserves: { cpu_bytes: "0", accelerator_bytes: "0" },
  policy: { layer_order: "highest_index_first", non_layer_pool: "cpu", state_pool: "accelerator", context_length: 16, batch_size: 1, state_storage_bits: 16 },
};
const placementBundle = await readArtifactBundle([
  ...placementBaseFiles,
  file("placement/deepbom.memory-profile.json", JSON.stringify(placementProfile), "application/json"),
]);
expect(placementBundle.analysis.artifact_bundle.model_source_sha256 === placementSourceSha
  && placementBundle.analysis.model_sha256 !== placementBase.analysis.model_sha256,
"Static memory profile binds the stable model-source digest without creating a self-referential bundle hash");
expect(placementBundle.analysis.on_device_llm.static_memory_placement.status === "assessed_lower_bound_candidates"
  && placementBundle.analysis.on_device_llm.static_memory_placement.maximum_accelerator_layer_count_not_disproven === 1
  && placementBundle.analysis.on_device_llm.static_memory_placement.fit_claim === "not_emitted",
"SafeTensors package evaluates all exact conditional CPU/accelerator lower-bound candidates without claiming fit");
expect(placementBundle.analysis.artifact_bundle.files.some((row) => row.role === "llm_static_memory_profile" && row.required === true),
"Static memory profile is a required hash-bound package dependency");
const llmContract = singleBundle.analysis.on_device_llm;
expect(llmContract.schema === "deepbom.on_device_llm_contract.v2" && llmContract.architecture.family === "llama", "Unified on-device LLM architecture contract");
expect(llmContract.state.storage_scenarios.map((row) => row.bytes_per_token_per_batch.decimal).join(",") === "4,8,16", "Conditional 8/16/32-bit KV-cache storage scenarios");
expect(llmContract.tokenizer.chat_template.status === "identical_dual_declaration" && llmContract.tokenizer.definition_files.length === 3, "Hash-bound tokenizer and identical chat-template declarations");
expect(llmContract.generation.values.max_new_tokens === 8 && llmContract.runtime_contract.status === "not_artifact_bound"
  && llmContract.medical_ai_claim_boundary.status === "declared_unverified_complete"
  && llmContract.medical_ai_claim_boundary.declaration.coverage.declared === 9
  && llmContract.medical_ai_claim_boundary.not_established.includes("clinical_validity"), "Generation declaration and runtime/medical claim boundary");
const singleReport = buildEngineeringReport(singleBundle.analysis);
expect(singleReport.includes("On-device LLM Evidence Contract") && singleReport.includes("4 elements/token/batch")
  && singleReport.includes("Declared-context prefill core MACs") && singleReport.includes("3392")
  && singleReport.includes("Memory Lower-bound Scenarios") && singleReport.includes("Encoding inventory / assignment SHA-256"),
"Engineering Report renders checkable HF architecture, storage, KV, memory-lower-bound, and compute evidence");
const singlePublicBom = buildPublicCycloneDx17ArtifactContract(singleBundle.analysis, { generatedAt: "2026-08-13T00:00:00.000Z" });
assertCycloneDx17(singlePublicBom, "SafeTensors public LLM evidence BOM");
const singlePublicProperties = new Map(singlePublicBom.metadata.component.properties.map((row) => [row.name, row.value]));
expect(singlePublicProperties.get("deepbom:model:llmArchitecture") === "llama"
  && singlePublicProperties.get("deepbom:model:llmKvElementsPerTokenPerBatch") === "4"
  && singlePublicProperties.get("deepbom:model:llmMemoryFeasibilityStatus") === "assessed_static_lower_bound_scenarios"
  && singlePublicProperties.get("deepbom:model:llmStaticMemoryCapacityScope") === "single_aggregate_primary_memory_budget"
  && /simultaneously resident/.test(singlePublicProperties.get("deepbom:model:llmStaticMemoryResidencyAssumption"))
  && singlePublicProperties.get("deepbom:model:llmMemoryFitClaim") === "not_emitted"
  && /^[a-f0-9]{64}$/.test(singlePublicProperties.get("deepbom:model:llmTensorEncodingAssignmentSha256"))
  && singlePublicProperties.get("deepbom:model:medicalAiClaimBoundary") === "declared_unverified_complete"
  && singlePublicProperties.get("deepbom:model:medicalAiDeclarationCoverage") === "9/9"
  && !singlePublicProperties.has("deepbom:model:fullIntegerQuantized"), "CycloneDX 1.7 exports concise LLM evidence without graph-quantization overclaim");

const tensorRtLlmConfig = {
  version: "1.2.0",
  pretrained_config: {
    architecture: "LlamaForCausalLM",
    dtype: "float16",
    hidden_size: 4,
    intermediate_size: 8,
    num_hidden_layers: 1,
    num_attention_heads: 2,
    num_key_value_heads: 1,
    head_size: 2,
    vocab_size: 8,
    max_position_embeddings: 16,
    mapping: { world_size: 1, tp_size: 1, pp_size: 1, cp_size: 1 },
    quantization: { quant_algo: null, kv_cache_quant_algo: null, group_size: null, has_zero_point: null, exclude_modules: [] },
  },
  build_config: {
    max_input_len: 8,
    max_seq_len: 16,
    max_batch_size: 1,
    max_beam_width: 1,
    max_num_tokens: 16,
    opt_num_tokens: 8,
    kv_cache_type: "PAGED",
    strongly_typed: true,
    weight_streaming: false,
    plugin_config: {},
  },
};
const tensorRtUnboundFiles = [
  ...singleSafeFiles,
  file("tiny/tensorrt_llm_engine_config.json", JSON.stringify(tensorRtLlmConfig), "application/json"),
];
const tensorRtUnboundBundle = await readArtifactBundle(tensorRtUnboundFiles);
const tensorRtConfigRecord = tensorRtUnboundBundle.analysis.artifact_bundle.files.find((row) => row.role === "tensorrt_llm_engine_config");
expect(tensorRtUnboundBundle.analysis.on_device_llm.tensorrt_llm.status === "candidate_configuration_unbound"
  && tensorRtConfigRecord?.required, "TensorRT-LLM engine config is hash-bound but remains an unbound candidate without a binding manifest");
const tensorRtBinding = {
  schema: "deepbom.tensorrt_llm_artifact_binding.v1",
  source_artifact_sha256: tensorRtUnboundBundle.analysis.artifact_bundle.model_source_sha256,
  engine_config_sha256: tensorRtConfigRecord.sha256,
  engine_files: [],
};
const tensorRtBoundBundle = await readArtifactBundle([
  ...tensorRtUnboundFiles,
  file("tiny/deepbom.tensorrt-llm.json", JSON.stringify(tensorRtBinding), "application/json"),
]);
expect(tensorRtBoundBundle.analysis.on_device_llm.tensorrt_llm.status === "artifact_bound_configuration"
  && tensorRtBoundBundle.analysis.on_device_llm.tensorrt_llm.artifact_binding.source_artifact_sha256 === tensorRtUnboundBundle.analysis.artifact_bundle.model_source_sha256,
"TensorRT-LLM binding resolves against the non-circular model-source digest");
expect(tensorRtBoundBundle.analysis.artifact_bundle.model_source_sha256 === tensorRtUnboundBundle.analysis.artifact_bundle.model_source_sha256
  && tensorRtBoundBundle.analysis.artifact_bundle.bundle_sha256 !== tensorRtUnboundBundle.analysis.artifact_bundle.bundle_sha256,
"Adding the binding manifest changes the complete bundle digest without changing its model-source subject digest");
const singleEvidence = buildEngineeringEvidenceDocument(singleBundle.analysis, {
  reportContext: { identity: { filename: singleBundle.analysis.filename, format: "safetensors", sha256: singleBundle.analysis.model_sha256 } },
  rawEvidenceContext: { identity: { filename: singleBundle.analysis.filename, format: "safetensors", sha256: singleBundle.analysis.model_sha256 } },
});
expect(singleEvidence.evidence.conformance_report.release_export_allowed
  && singleEvidence.evidence.conformance_report.checks.some((row) => row.id === "CF-SAFETENSORS-HF-001" && row.status === "pass"), "SafeTensors architecture projection release conformance");
const tamperedProjection = structuredClone(singleBundle.analysis);
tamperedProjection.safetensors.hf_architecture_contract.compute_projection.decode_transformer_core_macs_at_declared_context.decimal = "273";
await expectReject(Promise.resolve().then(() => buildEngineeringEvidenceDocument(tamperedProjection, {
  reportContext: { identity: { filename: tamperedProjection.filename, format: "safetensors", sha256: tamperedProjection.model_sha256 } },
  rawEvidenceContext: { identity: { filename: tamperedProjection.filename, format: "safetensors", sha256: tamperedProjection.model_sha256 } },
})), "CF-SAFETENSORS-HF-001", "SafeTensors compute projection tampering fails release conformance");

const canonicalTensors = [];
const addTensor = (name, shape, payloadSha = null) => canonicalTensors.push({ name, shape, numerical_integrity: payloadSha ? { payload_sha256: payloadSha } : null });
addTensor("model.embed_tokens.weight", [8, 4], "a".repeat(64));
addTensor("model.norm.weight", [4]);
for (const [suffix, shape] of [
  ["self_attn.q_proj.weight", [4, 4]], ["self_attn.k_proj.weight", [2, 4]], ["self_attn.v_proj.weight", [2, 4]],
  ["self_attn.o_proj.weight", [4, 4]], ["mlp.gate_proj.weight", [8, 4]], ["mlp.up_proj.weight", [8, 4]],
  ["mlp.down_proj.weight", [4, 8]], ["input_layernorm.weight", [4]], ["post_attention_layernorm.weight", [4]],
]) addTensor(`model.layers.0.${suffix}`, shape);
const completeContract = buildHfSafeTensorsContract(tinyLlamaConfig, canonicalTensors);
expect(completeContract.status === "assessed" && completeContract.tensor_contract.canonical_tensor_missing_count === 0, "Complete canonical HF tensor contract");
expect(completeContract.tensor_contract.canonical_expected_parameter_count_decimal === "188"
  && completeContract.tensor_contract.canonical_observed_parameter_count_decimal === "188", "Complete canonical HF parameter count");
const qwen2BiasTensors = canonicalTensors.concat([
  { name: "model.layers.0.self_attn.q_proj.bias", shape: [4] },
  { name: "model.layers.0.self_attn.k_proj.bias", shape: [2] },
  { name: "model.layers.0.self_attn.v_proj.bias", shape: [2] },
]);
const qwen2BiasContract = buildHfSafeTensorsContract({ ...tinyLlamaConfig, model_type: "qwen2" }, qwen2BiasTensors);
expect(qwen2BiasContract.status === "assessed" && qwen2BiasContract.attention_bias_scope === "qkv"
  && qwen2BiasContract.tensor_contract.attention_bias_scope === "qkv"
  && qwen2BiasContract.tensor_contract.canonical_tensor_missing_count === 0
  && qwen2BiasContract.tensor_contract.canonical_expected_parameter_count_decimal === "196"
  && qwen2BiasContract.tensor_contract.canonical_observed_parameter_count_decimal === "196"
  && !qwen2BiasContract.tensor_contract.expected_rows.some((row) => row.tensor_name.endsWith("self_attn.o_proj.bias")),
"Qwen2 fixed Q/K/V-only projection biases conserve the exact canonical parameter contract without inventing an output-projection bias");
const qwen2MissingBiasContract = buildHfSafeTensorsContract({ ...tinyLlamaConfig, model_type: "qwen2" }, qwen2BiasTensors.slice(0, -1));
expect(qwen2MissingBiasContract.status === "partial" && qwen2MissingBiasContract.tensor_contract.canonical_tensor_missing_count === 1
  && qwen2MissingBiasContract.tensor_contract.canonical_tensor_missing_names[0] === "model.layers.0.self_attn.v_proj.bias",
"A missing fixed Qwen2 projection bias remains explicit and cannot pass as an assessed canonical contract");
const llamaBiasTensors = canonicalTensors.concat([
  { name: "model.layers.0.self_attn.q_proj.bias", shape: [4] },
  { name: "model.layers.0.self_attn.k_proj.bias", shape: [2] },
  { name: "model.layers.0.self_attn.v_proj.bias", shape: [2] },
  { name: "model.layers.0.self_attn.o_proj.bias", shape: [4] },
  { name: "model.layers.0.mlp.gate_proj.bias", shape: [8] },
  { name: "model.layers.0.mlp.up_proj.bias", shape: [8] },
  { name: "model.layers.0.mlp.down_proj.bias", shape: [4] },
]);
const llamaBiasContract = buildHfSafeTensorsContract({ ...tinyLlamaConfig, attention_bias: true, mlp_bias: true }, llamaBiasTensors);
expect(llamaBiasContract.status === "assessed" && llamaBiasContract.attention_bias_scope === "all"
  && llamaBiasContract.fields.attention_bias === true && llamaBiasContract.fields.mlp_bias === true
  && llamaBiasContract.tensor_contract.canonical_expected_parameter_count_decimal === "220"
  && llamaBiasContract.tensor_contract.canonical_observed_parameter_count_decimal === "220",
"Llama config-controlled attention and MLP biases conserve the exact canonical parameter contract");
const qwen2ConformanceChecks = [];
registerSafeTensorsSerializedConformance({
  staticAnalysis: { safetensors: { hf_architecture_contract: qwen2BiasContract }, total_macs: null },
  tensors: qwen2BiasTensors,
  ops: [],
  reportText: `${qwen2BiasContract.source.source_commit} ${qwen2BiasContract.source.configuration_sources.qwen2.sha256} ${qwen2BiasContract.source.modeling_sources.qwen2.sha256} does not serialize an execution-operator DAG Declared-context prefill core MACs Canonical / checkpoint parameters`,
  check: (id, condition) => qwen2ConformanceChecks.push({ id, condition }),
});
expect(qwen2ConformanceChecks.some((row) => row.id === "CF-SAFETENSORS-HF-001" && row.condition),
  "Qwen2 Q/K/V-only bias scope independently reconstructs in release conformance");
const tamperedQwen2BiasContract = structuredClone(qwen2BiasContract);
tamperedQwen2BiasContract.attention_bias_scope = "all";
const tamperedQwen2Checks = [];
registerSafeTensorsSerializedConformance({
  staticAnalysis: { safetensors: { hf_architecture_contract: tamperedQwen2BiasContract }, total_macs: null },
  tensors: qwen2BiasTensors,
  ops: [],
  reportText: `${qwen2BiasContract.source.source_commit} ${qwen2BiasContract.source.configuration_sources.qwen2.sha256} ${qwen2BiasContract.source.modeling_sources.qwen2.sha256} does not serialize an execution-operator DAG Declared-context prefill core MACs Canonical / checkpoint parameters`,
  check: (id, condition) => tamperedQwen2Checks.push({ id, condition }),
});
expect(tamperedQwen2Checks.some((row) => row.id === "CF-SAFETENSORS-HF-001" && !row.condition),
  "SafeTensors release conformance rejects a tampered attention-bias scope");
const qwen3Contract = buildHfSafeTensorsContract({ model_type: "qwen3" }, []);
expect(qwen3Contract.status === "partial" && qwen3Contract.fields.head_dim === 128
  && qwen3Contract.fields.num_key_value_heads === 32 && qwen3Contract.source.configuration_sources.qwen3
  && qwen3Contract.source.modeling_sources.qwen3,
"Pinned Qwen3 defaults and source identity are registered without inventing an execution graph");
const tinyPhi3Config = { ...tinyLlamaConfig, model_type: "phi3" };
const phi3Tensors = [
  { name: "model.embed_tokens.weight", shape: [8, 4] }, { name: "model.norm.weight", shape: [4] },
  { name: "model.layers.0.self_attn.qkv_proj.weight", shape: [8, 4] },
  { name: "model.layers.0.self_attn.o_proj.weight", shape: [4, 4] },
  { name: "model.layers.0.mlp.gate_up_proj.weight", shape: [16, 4] },
  { name: "model.layers.0.mlp.down_proj.weight", shape: [4, 8] },
  { name: "model.layers.0.input_layernorm.weight", shape: [4] },
  { name: "model.layers.0.post_attention_layernorm.weight", shape: [4] },
];
const phi3Contract = buildHfSafeTensorsContract(tinyPhi3Config, phi3Tensors);
expect(phi3Contract.status === "assessed" && phi3Contract.tensor_layout_id === "fused_qkv_fused_gate_up_mlp"
  && phi3Contract.tensor_contract.canonical_expected_parameter_count_decimal === "188"
  && phi3Contract.tensor_contract.canonical_observed_parameter_count_decimal === "188",
"Phi-3 fused QKV/gate-up tensor shapes conserve the canonical dense-decoder parameter contract");
let phi3Offset = 0;
const phi3Descriptors = Object.fromEntries(phi3Tensors.map((tensor) => {
  const count = tensor.shape.reduce((product, value) => product * value, 1);
  const start = phi3Offset;
  phi3Offset += count;
  return [tensor.name, { dtype: "U8", shape: tensor.shape, data_offsets: [start, phi3Offset] }];
}));
const phi3Bundle = await readArtifactBundle([
  safeTensorPayloadFile("phi3/model.safetensors", phi3Descriptors, new Uint8Array(phi3Offset)),
  file("phi3/config.json", JSON.stringify(tinyPhi3Config), "application/json"),
]);
const phi3BundleContract = phi3Bundle.analysis.safetensors.hf_architecture_contract;
const phi3Report = buildEngineeringReport(phi3Bundle.analysis);
const phi3Evidence = buildEngineeringEvidenceDocument(phi3Bundle.analysis, {
  reportContext: { identity: { filename: phi3Bundle.analysis.filename, format: "safetensors", sha256: phi3Bundle.analysis.model_sha256 } },
  rawEvidenceContext: { identity: { filename: phi3Bundle.analysis.filename, format: "safetensors", sha256: phi3Bundle.analysis.model_sha256 } },
});
expect(phi3BundleContract.status === "assessed" && phi3Bundle.analysis.tensor_numerical_integrity.byte_conservation_status === "complete"
  && phi3Report.includes(phi3BundleContract.source.configuration_sources.phi3.sha256)
  && phi3Report.includes(phi3BundleContract.source.modeling_sources.phi3.sha256)
  && phi3Evidence.evidence.conformance_report.release_export_allowed
  && phi3Evidence.evidence.conformance_report.checks.some((row) => row.id === "CF-SAFETENSORS-HF-001" && row.status === "pass"),
"Phi-3 package binds payload, config, modeling source, report, and release conformance end to end");
const tamperedPhi3Source = structuredClone(phi3Bundle.analysis);
tamperedPhi3Source.safetensors.hf_architecture_contract.source.modeling_sources.phi3.sha256 = "b".repeat(64);
await expectReject(Promise.resolve().then(() => buildEngineeringEvidenceDocument(tamperedPhi3Source, {
  reportContext: { identity: { filename: tamperedPhi3Source.filename, format: "safetensors", sha256: tamperedPhi3Source.model_sha256 } },
  rawEvidenceContext: { identity: { filename: tamperedPhi3Source.filename, format: "safetensors", sha256: tamperedPhi3Source.model_sha256 } },
})), "CF-SAFETENSORS-HF-001", "SafeTensors modeling-source tampering fails release conformance");
const tinyOlmo2Config = { ...tinyLlamaConfig, model_type: "olmo2" };
const olmo2Tensors = [
  { name: "model.embed_tokens.weight", shape: [8, 4] }, { name: "model.norm.weight", shape: [4] },
  { name: "model.layers.0.self_attn.q_proj.weight", shape: [4, 4] },
  { name: "model.layers.0.self_attn.k_proj.weight", shape: [2, 4] },
  { name: "model.layers.0.self_attn.v_proj.weight", shape: [2, 4] },
  { name: "model.layers.0.self_attn.o_proj.weight", shape: [4, 4] },
  { name: "model.layers.0.mlp.gate_proj.weight", shape: [8, 4] },
  { name: "model.layers.0.mlp.up_proj.weight", shape: [8, 4] },
  { name: "model.layers.0.mlp.down_proj.weight", shape: [4, 8] },
  { name: "model.layers.0.post_attention_layernorm.weight", shape: [4] },
  { name: "model.layers.0.post_feedforward_layernorm.weight", shape: [4] },
  { name: "model.layers.0.self_attn.q_norm.weight", shape: [4] },
  { name: "model.layers.0.self_attn.k_norm.weight", shape: [2] },
];
const olmo2Contract = buildHfSafeTensorsContract(tinyOlmo2Config, olmo2Tensors);
expect(olmo2Contract.status === "assessed" && olmo2Contract.tensor_layout_id === "split_qkv_full_width_qk_norm_post_norms"
  && olmo2Contract.tensor_contract.canonical_expected_parameter_count_decimal === "194"
  && olmo2Contract.tensor_contract.canonical_observed_parameter_count_decimal === "194",
"OLMo2 full-width Q/K norms and post-norm tensor shapes conserve the canonical parameter contract");
const standardFamilyConfig = { ...tinyLlamaConfig, tie_word_embeddings: true };
for (const modelType of ["ministral", "smollm3", "olmo"]) {
  const contract = buildHfSafeTensorsContract({ ...standardFamilyConfig, model_type: modelType }, canonicalTensors);
  expect(contract.status === "assessed" && contract.mlp_projection_matrix_count === 3
    && contract.source.configuration_sources[modelType] && contract.source.modeling_sources[modelType],
  `${modelType} source-pinned split-QKV gated-decoder contract`);
}
const parallelTensors = canonicalTensors.filter((tensor) => !tensor.name.endsWith("post_attention_layernorm.weight"));
const cohere2Contract = buildHfSafeTensorsContract({ ...standardFamilyConfig, model_type: "cohere2" }, parallelTensors);
expect(cohere2Contract.status === "assessed" && cohere2Contract.tensor_layout_id === "split_qkv_parallel_residual_split_gated_mlp",
  "Cohere2 parallel-residual normalization layout is assessed without inventing a post-attention norm");
const cohereTensors = [
  ...parallelTensors,
  { name: "model.layers.0.self_attn.q_norm.weight", shape: [2, 2] },
  { name: "model.layers.0.self_attn.k_norm.weight", shape: [1, 2] },
  { name: "model.layers.0.self_attn.q_proj.bias", shape: [4] },
  { name: "model.layers.0.self_attn.k_proj.bias", shape: [2] },
  { name: "model.layers.0.self_attn.v_proj.bias", shape: [2] },
  { name: "model.layers.0.self_attn.o_proj.bias", shape: [4] },
];
const cohereContract = buildHfSafeTensorsContract({ ...standardFamilyConfig, model_type: "cohere", use_qk_norm: true, attention_bias: true }, cohereTensors);
expect(cohereContract.status === "assessed" && cohereContract.fields.use_qk_norm && cohereContract.fields.attention_bias
  && cohereContract.tensor_contract.canonical_tensor_missing_count === 0,
"Cohere conditional per-head Q/K normalization and attention-bias tensors are bound to declared config flags");
const nemotronTensors = canonicalTensors
  .filter((tensor) => !tensor.name.endsWith("mlp.gate_proj.weight"))
  .concat([
    { name: "model.layers.0.self_attn.q_proj.bias", shape: [4] },
    { name: "model.layers.0.self_attn.k_proj.bias", shape: [2] },
    { name: "model.layers.0.self_attn.v_proj.bias", shape: [2] },
    { name: "model.layers.0.self_attn.o_proj.bias", shape: [4] },
    { name: "model.layers.0.mlp.up_proj.bias", shape: [8] },
    { name: "model.layers.0.mlp.down_proj.bias", shape: [4] },
  ]);
const nemotronContract = buildHfSafeTensorsContract({ ...standardFamilyConfig, model_type: "nemotron", attention_bias: true, mlp_bias: true }, nemotronTensors);
expect(nemotronContract.status === "assessed" && nemotronContract.mlp_projection_matrix_count === 2
  && nemotronContract.compute_projection.schema === "deepbom.canonical_decoder_compute_projection.v1"
  && nemotronContract.compute_projection.dense_projection_macs_per_layer_per_token.decimal === "112"
  && nemotronContract.tensor_contract.canonical_expected_parameter_count_decimal === "180",
"Nemotron two-matrix MLP, conditional biases, exact parameter ledger, and 112-MAC dense scenario");
const genericChecks = [];
registerSafeTensorsSerializedConformance({
  staticAnalysis: { safetensors: { hf_architecture_contract: nemotronContract }, total_macs: null },
  tensors: nemotronTensors,
  ops: [],
  reportText: `${nemotronContract.source.source_commit} ${nemotronContract.source.configuration_sources.nemotron.sha256} ${nemotronContract.source.modeling_sources.nemotron.sha256} does not serialize an execution-operator DAG Declared-context prefill core MACs Canonical / checkpoint parameters`,
  check: (id, condition) => genericChecks.push({ id, condition }),
});
expect(genericChecks.some((row) => row.id === "CF-SAFETENSORS-HF-001" && row.condition),
  "Non-gated SafeTensors compute projection independently reconstructs in release conformance");
const exaoneTensors = canonicalTensors
  .filter((tensor) => !tensor.name.endsWith("input_layernorm.weight") && !tensor.name.endsWith("post_attention_layernorm.weight"))
  .concat([
    { name: "model.layers.0.post_attention_layernorm.weight", shape: [4] },
    { name: "model.layers.0.post_feedforward_layernorm.weight", shape: [4] },
    { name: "model.layers.0.self_attn.q_norm.weight", shape: [2] },
    { name: "model.layers.0.self_attn.k_norm.weight", shape: [2] },
  ]);
const exaoneContract = buildHfSafeTensorsContract({ ...standardFamilyConfig, model_type: "exaone4" }, exaoneTensors);
expect(exaoneContract.status === "assessed" && exaoneContract.tensor_contract.canonical_expected_parameter_count_decimal === "192",
  "EXAONE 4 post-norm and head-width Q/K norm shapes conserve the exact canonical parameter ledger");
const graniteContract = buildHfSafeTensorsContract({ model_type: "granite" }, []);
expect(graniteContract.status === "partial" && graniteContract.source.configuration_sources.granite
  && graniteContract.source.modeling_sources.granite, "Granite split-QKV gated-decoder sources are registered");
const malformedCanonical = canonicalTensors.map((tensor) => tensor.name.endsWith("q_proj.weight") ? { ...tensor, shape: [5, 4] } : tensor);
const invalidContract = buildHfSafeTensorsContract(tinyLlamaConfig, malformedCanonical);
expect(invalidContract.status === "invalid" && invalidContract.tensor_contract.canonical_tensor_shape_mismatch_count === 1, "Canonical tensor shape mismatch fails the architecture contract");
const excessiveLayerContract = buildHfSafeTensorsContract({ ...tinyLlamaConfig, num_hidden_layers: 4097 }, []);
expect(excessiveLayerContract.status === "invalid_config"
  && excessiveLayerContract.issues.some((row) => row.code === "HF_CONFIG_LAYER_COUNT_EXCEEDS_ANALYSIS_LIMIT"), "Hostile SafeTensors layer counts fail before canonical-row allocation");
const unsafeProjectionWidthContract = buildHfSafeTensorsContract({
  ...tinyLlamaConfig,
  hidden_size: Number.MAX_SAFE_INTEGER,
  num_attention_heads: Number.MAX_SAFE_INTEGER,
  num_key_value_heads: 1,
  head_dim: 2,
}, []);
expect(unsafeProjectionWidthContract.status === "invalid_config"
  && unsafeProjectionWidthContract.issues.some((row) => row.code === "HF_CONFIG_DERIVED_PROJECTION_WIDTH_UNSAFE"), "Unsafe derived decoder widths fail before shape construction");
const unsafeFusedWidthContract = buildHfSafeTensorsContract({
  ...tinyPhi3Config,
  intermediate_size: Number.MAX_SAFE_INTEGER,
}, []);
expect(unsafeFusedWidthContract.status === "invalid_config"
  && unsafeFusedWidthContract.issues.some((row) => row.code === "HF_CONFIG_DERIVED_PROJECTION_WIDTH_UNSAFE"),
"Unsafe fused gate/up width fails before canonical shape construction");

const missingShard = shardFiles.slice(0, 2);
await expectReject(inspectArtifactBundle(missingShard), "missing file", "missing shard fails closed");
const traversalManifest = structuredClone(manifest);
traversalManifest.itemInfoEntries["weights-id"].path = "../outside.bin";
await expectReject(inspectArtifactBundle([
  file("Fixture.mlpackage/Manifest.json", JSON.stringify(traversalManifest)),
  coreFiles[1], coreFiles[2],
]), "traversal", "Core ML traversal fails closed");

console.log("Artifact bundle checks passed (Core ML package, SafeTensors single/sharded repositories, HF architecture/KV contracts, hash dependencies, and fail-closed manifests).");
