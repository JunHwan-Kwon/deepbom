import assert from "node:assert/strict";

import { buildHfSafeTensorsContract } from "../web/lib/hf-safetensors-contract.js";
import { buildSafeTensorsQuantizationContract } from "../web/lib/safetensors-quantization-contract.js";
import { buildOnDeviceLlmContract, validateOnDeviceLlmContract } from "../web/lib/on-device-llm-contract.js";
import { assessOnDeviceLlmRuntimeManifest, buildLlmStateScenarioMatrix } from "../web/lib/on-device-llm-runtime-manifest.js";
import { buildLlmMemoryFeasibility, compareLlmMemoryCapacity } from "../web/lib/llm-memory-feasibility.js";
import { buildLlmStaticMemoryPlacement } from "../web/lib/llm-static-memory-placement.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";
import { registerSafeTensorsSerializedConformance } from "../web/lib/report-conformance-serialized-containers.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { buildPublicCycloneDx17ArtifactContract } from "../web/lib/public-cyclonedx-export.js";
import { buildArtifactEvidenceEnvelope } from "../web/lib/artifact-evidence-envelope.js";
import { buildCycloneDxEvidenceDocument } from "../web/lib/report-export-contracts.js";

const tensor = (name, shape) => ({ name, shape, numerical_integrity: { payload_sha256: "a".repeat(64) } });
const storageSummary = (tensors) => {
  const elements = tensors.reduce((sum, row) => sum + row.shape.reduce((product, value) => product * value, 1), 0);
  const bytes = elements * 2;
  return {
    status: "assessed", element_count: elements, element_count_decimal: String(elements), byte_length: bytes,
    byte_length_decimal: String(bytes), effective_bits_per_element: 16,
    encodings: [{ dtype: "F16", tensor_count: tensors.length, element_count: elements, element_count_decimal: String(elements), byte_length: bytes, byte_length_decimal: String(bytes), effective_bits_per_element: 16 }],
  };
};
const mixtralConfig = {
  model_type: "mixtral", vocab_size: 16, hidden_size: 8, intermediate_size: 12,
  num_hidden_layers: 1, num_attention_heads: 2, num_key_value_heads: 1,
  head_dim: 4, max_position_embeddings: 16, num_local_experts: 4,
  num_experts_per_tok: 2, tie_word_embeddings: false,
};
const mixtralTensors = [
  tensor("model.embed_tokens.weight", [16, 8]), tensor("model.norm.weight", [8]), tensor("lm_head.weight", [16, 8]),
  tensor("model.layers.0.input_layernorm.weight", [8]), tensor("model.layers.0.post_attention_layernorm.weight", [8]),
  tensor("model.layers.0.self_attn.q_proj.weight", [8, 8]), tensor("model.layers.0.self_attn.k_proj.weight", [4, 8]),
  tensor("model.layers.0.self_attn.v_proj.weight", [4, 8]), tensor("model.layers.0.self_attn.o_proj.weight", [8, 8]),
  tensor("model.layers.0.block_sparse_moe.gate.weight", [4, 8]),
];
for (let expert = 0; expert < 4; expert += 1) mixtralTensors.push(
  tensor(`model.layers.0.block_sparse_moe.experts.${expert}.w1.weight`, [12, 8]),
  tensor(`model.layers.0.block_sparse_moe.experts.${expert}.w2.weight`, [8, 12]),
  tensor(`model.layers.0.block_sparse_moe.experts.${expert}.w3.weight`, [12, 8]),
);
const mixtral = buildHfSafeTensorsContract(mixtralConfig, mixtralTensors);
assert.equal(mixtral.status, "assessed");
assert.equal(mixtral.architecture_kind, "sparse_moe_decoder");
assert.equal(mixtral.tensor_contract.canonical_tensor_shape_mismatch_count, 0);
assert.equal(mixtral.compute_projection.total_expert_matrix_parameters_all_layers.decimal, "1152");
assert.equal(mixtral.compute_projection.active_expert_matrix_macs_per_layer_per_token.decimal, "576");
assert.equal(mixtral.compute_projection.active_projection_macs_per_layer_per_token.decimal, "800");

const mambaConfig = {
  model_type: "mamba", vocab_size: 16, hidden_size: 8, state_size: 3,
  num_hidden_layers: 1, expand: 2, conv_kernel: 4, time_step_rank: "auto",
  use_bias: false, use_conv_bias: true,
};
const mambaTensors = [
  tensor("backbone.embeddings.weight", [16, 8]), tensor("backbone.norm_f.weight", [8]),
  tensor("backbone.layers.0.norm.weight", [8]), tensor("backbone.layers.0.mixer.in_proj.weight", [32, 8]),
  tensor("backbone.layers.0.mixer.conv1d.weight", [16, 1, 4]), tensor("backbone.layers.0.mixer.conv1d.bias", [16]),
  tensor("backbone.layers.0.mixer.x_proj.weight", [7, 16]), tensor("backbone.layers.0.mixer.dt_proj.weight", [16, 1]),
  tensor("backbone.layers.0.mixer.dt_proj.bias", [16]), tensor("backbone.layers.0.mixer.A_log", [16, 3]),
  tensor("backbone.layers.0.mixer.D", [16]), tensor("backbone.layers.0.mixer.out_proj.weight", [8, 16]),
];
const mamba = buildHfSafeTensorsContract(mambaConfig, mambaTensors);
assert.equal(mamba.status, "assessed");
assert.equal(mamba.architecture_kind, "ssm_recurrent");
assert.equal(mamba.fields.time_step_rank, 1);
assert.equal(mamba.recurrent_state_projection.recurrent_state_elements_all_layers_per_batch.decimal, "112");
assert.equal(mamba.compute_projection.accounted_macs_per_layer_per_token.decimal, "576");

const jambaConfig = {
  model_type: "jamba", vocab_size: 16, hidden_size: 8, intermediate_size: 12,
  num_hidden_layers: 4, num_attention_heads: 2, num_key_value_heads: 1,
  max_position_embeddings: 16, num_experts_per_tok: 1, num_experts: 2,
  expert_layer_period: 2, expert_layer_offset: 1, attn_layer_period: 2, attn_layer_offset: 0,
  mamba_d_state: 3, mamba_d_conv: 4, mamba_expand: 2, mamba_dt_rank: "auto",
  mamba_conv_bias: true, mamba_proj_bias: false, tie_word_embeddings: false,
};
const jambaTensors = [
  tensor("model.embed_tokens.weight", [16, 8]), tensor("model.final_layernorm.weight", [8]), tensor("lm_head.weight", [16, 8]),
];
for (let layer = 0; layer < 4; layer += 1) {
  const prefix = `model.layers.${layer}`;
  jambaTensors.push(tensor(`${prefix}.input_layernorm.weight`, [8]), tensor(`${prefix}.pre_ff_layernorm.weight`, [8]));
  if (layer % 2 === 0) jambaTensors.push(
    tensor(`${prefix}.self_attn.q_proj.weight`, [8, 8]), tensor(`${prefix}.self_attn.k_proj.weight`, [4, 8]),
    tensor(`${prefix}.self_attn.v_proj.weight`, [4, 8]), tensor(`${prefix}.self_attn.o_proj.weight`, [8, 8]),
    tensor(`${prefix}.feed_forward.gate_proj.weight`, [12, 8]), tensor(`${prefix}.feed_forward.up_proj.weight`, [12, 8]),
    tensor(`${prefix}.feed_forward.down_proj.weight`, [8, 12]),
  );
  else {
    jambaTensors.push(
      tensor(`${prefix}.mamba.in_proj.weight`, [32, 8]), tensor(`${prefix}.mamba.conv1d.weight`, [16, 1, 4]),
      tensor(`${prefix}.mamba.conv1d.bias`, [16]), tensor(`${prefix}.mamba.x_proj.weight`, [7, 16]),
      tensor(`${prefix}.mamba.dt_proj.weight`, [16, 1]), tensor(`${prefix}.mamba.dt_proj.bias`, [16]),
      tensor(`${prefix}.mamba.A_log`, [16, 3]), tensor(`${prefix}.mamba.D`, [16]),
      tensor(`${prefix}.mamba.out_proj.weight`, [8, 16]), tensor(`${prefix}.mamba.dt_layernorm.weight`, [1]),
      tensor(`${prefix}.mamba.b_layernorm.weight`, [3]), tensor(`${prefix}.mamba.c_layernorm.weight`, [3]),
      tensor(`${prefix}.feed_forward.router.weight`, [2, 8]),
    );
    for (let expert = 0; expert < 2; expert += 1) jambaTensors.push(
      tensor(`${prefix}.feed_forward.experts.${expert}.gate_proj.weight`, [12, 8]),
      tensor(`${prefix}.feed_forward.experts.${expert}.up_proj.weight`, [12, 8]),
      tensor(`${prefix}.feed_forward.experts.${expert}.down_proj.weight`, [8, 12]),
    );
  }
}
const jamba = buildHfSafeTensorsContract(jambaConfig, jambaTensors);
assert.equal(jamba.status, "assessed");
assert.equal(jamba.architecture_kind, "hybrid_attention_ssm_moe");
assert.deepEqual(jamba.layer_schedule.attention_layer_indices, [0, 2]);
assert.deepEqual(jamba.layer_schedule.mamba_layer_indices, [1, 3]);
assert.deepEqual(jamba.layer_schedule.expert_feed_forward_layer_indices, [1, 3]);
assert.equal(jamba.tensor_contract.canonical_tensor_shape_mismatch_count, 0);
assert.equal(jamba.kv_state_projection.elements_per_token_per_batch.decimal, "16");
assert.equal(jamba.recurrent_state_projection.recurrent_state_elements_all_layers_per_batch.decimal, "224");
assert.equal(jamba.compute_projection.schema, "deepbom.hybrid_jamba_compute_projection.v1");
assert.equal(jamba.compute_projection.attention_layer_count, 2);
assert.equal(jamba.compute_projection.mamba_layer_count, 2);

const analysis = {
  format: "safetensors", model_sha256: "1".repeat(64),
  tensors: mixtralTensors.map((row, index) => ({ ...row, index, dtype: "F16", byte_length: row.shape.reduce((product, value) => product * value, 1) * 2 })),
  tensor_storage_summary: storageSummary(mixtralTensors),
  safetensors: { hf_architecture_contract: mixtral }, artifact_bundle: { files: [] },
};
const llm = buildOnDeviceLlmContract(analysis);
analysis.on_device_llm = llm;
assert.equal(llm.schema, "deepbom.on_device_llm_contract.v2");
assert.equal(llm.architecture.kind, "sparse_moe_decoder");
assert.equal(llm.architecture.moe.expert_count, 4);
assert.equal(llm.state.scenario_matrix.length, 9);
assert.equal(llm.storage.encoding_inventory.length, 1);
assert.match(llm.storage.encoding_inventory_sha256, /^[a-f0-9]{64}$/);
assert.match(llm.storage.tensor_encoding_assignment_sha256, /^[a-f0-9]{64}$/);
assert.equal(llm.storage.encoding_inventory_sha256, sha256TextHex(canonicalJson(llm.storage.encoding_inventory)));
const independentAssignments = analysis.tensors.map((row) => ({
  index: row.index, name: row.name, dtype: row.dtype, shape: row.shape, byte_length: row.byte_length,
})).sort((left, right) => left.index - right.index || left.name.localeCompare(right.name));
assert.equal(llm.storage.tensor_encoding_assignment_sha256, sha256TextHex(canonicalJson(independentAssignments)));
assert.equal(llm.storage.layer_storage.status, "assessed_exact_serialized_layer_storage");
assert.equal(llm.storage.layer_storage.observed_layer_count, 1);
assert.equal(llm.storage.layer_storage.layer_tensor_count, 19);
assert.equal(llm.storage.layer_storage.layer_bytes.decimal, "2784");
assert.equal(llm.storage.layer_storage.non_layer_bytes.decimal, "528");
assert.equal(llm.storage.layer_storage.conservation.status, "pass");
const llmReport = buildEngineeringReport(analysis, { generatedAt: "2026-08-25T00:00:00.000Z" });
assert(llmReport.includes("### Serialized Layer Storage Ledger") && llmReport.includes("2784 B layer + 528 B non-layer"));
const llmBom = buildPublicCycloneDx17ArtifactContract(analysis, { generatedAt: "2026-08-25T00:00:00.000Z" });
const llmProperties = new Map(llmBom.metadata.component.properties.map((row) => [row.name, row.value]));
assert.equal(llmProperties.get("deepbom:model:llmLayerStorageStatus"), "assessed_exact_serialized_layer_storage");
assert.equal(llmProperties.get("deepbom:model:llmLayerSerializedBytes"), "2784");
const evidenceBom = buildCycloneDxEvidenceDocument(analysis, { generatedAt: "2026-08-25T00:00:00.000Z" });
const evidenceProperties = new Map(evidenceBom.metadata.component.properties.map((row) => [row.name, row.value]));
assert.equal(evidenceProperties.get("deepbom:model:llmLayerStorageStatus"), "assessed_exact_serialized_layer_storage");
assert.equal(evidenceProperties.get("deepbom:model:llmNonLayerSerializedBytes"), "528");
const artifactEnvelope = buildArtifactEvidenceEnvelope(analysis, { generatedAt: "2026-08-25T00:00:00.000Z" });
assert(artifactEnvelope.capabilities.assessed.includes("llm_layer_storage"));
assert.equal(artifactEnvelope.capabilities.conservation.valid, true);
assert.equal(llm.memory_feasibility.status, "assessed_static_lower_bound_scenarios");
assert.equal(llm.memory_feasibility.minimum_static_lower_bound_bytes.decimal, "3440");
assert.equal(llm.memory_feasibility.maximum_static_lower_bound_bytes.decimal, "5360");
assert.equal(llm.memory_feasibility.static_scenarios[0].first_capacity_not_exceeded, "512 MiB");
const constrainedMemory = buildLlmMemoryFeasibility(llm, { capacityTiers: [{ label: "2 KiB", bytes: "2048" }, { label: "4 KiB", bytes: "4096" }] });
assert.equal(constrainedMemory.static_scenarios[0].first_capacity_not_exceeded, "4 KiB");
assert.equal(constrainedMemory.static_scenarios[0].lower_bound_exceeded_capacity_count, 1);
assert.equal(compareLlmMemoryCapacity("3440", "2048").status, "lower_bound_exceeds_capacity");
const equalityCapacity = compareLlmMemoryCapacity("3440", "3440");
assert.equal(equalityCapacity.status, "lower_bound_at_or_below_capacity_fit_unresolved");
assert.equal(equalityCapacity.headroom_after_lower_bound_bytes.decimal, "0");
assert.deepEqual(validateOnDeviceLlmContract(analysis), { valid: true, errors: [] });
const staticMemoryProfile = {
  schema: "deepbom.llm_static_memory_profile.v1",
  artifact: { format: "safetensors", sha256: analysis.model_sha256 },
  capacities: { cpu_bytes: "2500", accelerator_bytes: "4000" },
  reserves: { cpu_bytes: "100", accelerator_bytes: "100" },
  policy: {
    layer_order: "highest_index_first", non_layer_pool: "cpu", state_pool: "accelerator",
    context_length: 16, batch_size: 1, state_storage_bits: 16,
  },
};
const staticMemorySidecar = { document: staticMemoryProfile, path: "deepbom.memory-profile.json", sha256: "9".repeat(64) };
const placement = buildLlmStaticMemoryPlacement(llm, analysis, staticMemorySidecar);
assert.equal(placement.status, "assessed_lower_bound_candidates");
assert.equal(placement.candidate_count, 2);
assert.equal(placement.lower_bound_not_exceeding_candidate_count, 1);
assert.equal(placement.minimum_accelerator_layer_count_not_disproven, 1);
assert.equal(placement.maximum_accelerator_layer_count_not_disproven, 1);
assert.equal(placement.candidates[0].cpu_capacity_assessment.status, "accounted_lower_bound_exceeds_effective_capacity");
assert.equal(placement.candidates[1].cpu_accounted_lower_bound_bytes.decimal, "528");
assert.equal(placement.candidates[1].accelerator_accounted_lower_bound_bytes.decimal, "3040");
assert.equal(placement.conservation.status, "pass");
assert.equal(placement.fit_claim, "not_emitted");
assert.throws(() => buildLlmStaticMemoryPlacement(llm, analysis, { ...staticMemorySidecar, document: { ...staticMemoryProfile, artifact: { ...staticMemoryProfile.artifact, sha256: "0".repeat(64) } } }), /not bound to the active artifact/);
assert.throws(() => buildLlmStaticMemoryPlacement(llm, analysis, { ...staticMemorySidecar, document: { ...staticMemoryProfile, reserves: { ...staticMemoryProfile.reserves, cpu_bytes: "2500" } } }), /reserve must be smaller/);
analysis.artifact_bundle.files.push({ path: "deepbom.memory-profile.json", role: "llm_static_memory_profile", required: true, sha256: "9".repeat(64) });
const placedLlm = buildOnDeviceLlmContract(analysis, { sidecars: { static_memory_profile: staticMemorySidecar } });
analysis.on_device_llm = placedLlm;
assert.deepEqual(validateOnDeviceLlmContract(analysis), { valid: true, errors: [] });
const placedReport = buildEngineeringReport(analysis, { generatedAt: "2026-08-25T00:00:00.000Z" });
assert(placedReport.includes("### Conditional CPU / Accelerator Memory Placement") && placedReport.includes("3040"));
const placedBom = buildPublicCycloneDx17ArtifactContract(analysis, { generatedAt: "2026-08-25T00:00:00.000Z" });
const placedProperties = new Map(placedBom.metadata.component.properties.map((row) => [row.name, row.value]));
assert.equal(placedProperties.get("deepbom:model:llmStaticPoolMaximumAcceleratorLayersNotDisproven"), "1");
assert.equal(placedProperties.get("deepbom:model:llmStaticPoolFitClaim"), "not_emitted");
const placedEvidenceBom = buildCycloneDxEvidenceDocument(analysis, { generatedAt: "2026-08-25T00:00:00.000Z" });
const placedEvidenceProperties = new Map(placedEvidenceBom.metadata.component.properties.map((row) => [row.name, row.value]));
assert.equal(placedEvidenceProperties.get("deepbom:model:llmStaticPoolNotDisprovenCandidateCount"), "1");
const placedEnvelope = buildArtifactEvidenceEnvelope(analysis, { generatedAt: "2026-08-25T00:00:00.000Z" });
assert(placedEnvelope.capabilities.assessed.includes("llm_static_memory_placement"));
analysis.on_device_llm = llm;
const tamperedEncoding = structuredClone(llm);
tamperedEncoding.storage.tensor_encoding_assignment_sha256 = "0".repeat(64);
analysis.on_device_llm = tamperedEncoding;
assert(validateOnDeviceLlmContract(analysis).errors.includes("encoding_contract_mismatch:tensor_encoding_assignment_sha256"));
analysis.on_device_llm = llm;

const manifest = {
  schema: "deepbom.on_device_llm_runtime_manifest.v1",
  artifact: { format: "safetensors", sha256: analysis.model_sha256 },
  runtime: { evidence_class: "OBSERVED_RUNTIME", engine: "test-runtime", version: "1.0", binary_sha256: "2".repeat(64), build_configuration_sha256: "3".repeat(64) },
  deployment: { evidence_class: "DECLARED", context_length: 16, batch_size: 2, state_storage_bits: 16 },
  weight_residency: { evidence_class: "OBSERVED_RUNTIME", artifact_serialized_tensor_bytes: "3312", runtime_weight_bytes: "2400", cpu_bytes: "800", accelerator_bytes: "1600", unresident_bytes: "0" },
  layer_placement: { evidence_class: "OBSERVED_RUNTIME", layer_count: 1, assignments: [{ layer_index: 0, location: "accelerator" }] },
  state_cache: { evidence_class: "OBSERVED_RUNTIME", kind: "transformer_kv", logical_bytes: "512", allocated_bytes: "1024", resident_bytes: "512", paging_enabled: true, page_size_bytes: "256", resident_page_count: "2", total_page_count: "4" },
  capture: { capture_id: "capture-1", collected_at: "2026-08-15T00:00:00.000Z", device_identity_sha256: "4".repeat(64) },
};
const runtime = assessOnDeviceLlmRuntimeManifest({ document: manifest, path: "deepbom.runtime.json", sha256: "5".repeat(64) }, analysis, llm);
assert.equal(runtime.status, "artifact_bound_observed_runtime");
assert.equal(runtime.issue_count, 0);
assert.equal(runtime.layer_placement.accelerator_layer_count, 1);
assert.equal(runtime.state_cache.logical_bytes.decimal, "512");
const invalidRuntime = assessOnDeviceLlmRuntimeManifest({ document: { ...manifest, weight_residency: { ...manifest.weight_residency, accelerator_bytes: "1599" } } }, analysis, llm);
assert.equal(invalidRuntime.status, "invalid");
assert(invalidRuntime.issues.some((row) => row.code === "LLM_RUNTIME_WEIGHT_RESIDENCY_CONSERVATION_FAILED"));
analysis.artifact_bundle.files.push({ path: "deepbom.runtime.json", role: "llm_runtime_manifest", required: true, sha256: "5".repeat(64) });
const boundLlm = buildOnDeviceLlmContract(analysis, { sidecars: { runtime_manifest: { document: manifest, path: "deepbom.runtime.json", sha256: "5".repeat(64) } } });
analysis.on_device_llm = boundLlm;
assert.equal(boundLlm.runtime_contract.status, "artifact_bound_observed_runtime");
assert.equal(boundLlm.memory_feasibility.runtime_primary_residency.primary_resident_lower_bound_bytes.decimal, "2912");
assert.equal(boundLlm.memory_feasibility.runtime_primary_residency.primary_allocated_lower_bound_bytes.decimal, "3424");
assert.equal(boundLlm.memory_feasibility.runtime_primary_residency.primary_allocated_accounted_bytes, null);
assert.equal(boundLlm.memory_feasibility.runtime_primary_residency.allocation_accounting_status, "working_memory_categories_unbound");
assert.deepEqual(validateOnDeviceLlmContract(analysis), { valid: true, errors: [] });
const tamperedLlm = structuredClone(boundLlm);
tamperedLlm.runtime_contract.state_cache.logical_bytes.decimal = "511";
analysis.on_device_llm = tamperedLlm;
assert(validateOnDeviceLlmContract(analysis).errors.includes("runtime_state_conservation_invalid"));
const tamperedMemory = structuredClone(boundLlm);
tamperedMemory.memory_feasibility.minimum_static_lower_bound_bytes.decimal = "2127";
analysis.on_device_llm = tamperedMemory;
assert(validateOnDeviceLlmContract(analysis).errors.includes("llm_memory_feasibility_recomputation_mismatch"));
analysis.on_device_llm = boundLlm;

const manifestV2 = {
  ...manifest,
  schema: "deepbom.on_device_llm_runtime_manifest.v2",
  working_memory: {
    evidence_class: "OBSERVED_RUNTIME",
    coverage: "complete_observed_runtime_categories",
    graph_workspace_bytes: "128",
    scratch_bytes: "64",
    packing_and_replica_bytes: "256",
    allocator_overhead_bytes: "32",
    backend_private_bytes: "16",
    other_runtime_bytes: "4",
    accounted_nonweight_runtime_bytes: "500",
  },
};
const runtimeV2 = assessOnDeviceLlmRuntimeManifest({ document: manifestV2, path: "deepbom.runtime.json", sha256: "6".repeat(64) }, analysis, llm);
assert.equal(runtimeV2.status, "artifact_bound_observed_runtime");
assert.equal(runtimeV2.working_memory.category_count, 6);
assert.equal(runtimeV2.working_memory.accounted_nonweight_runtime_bytes.decimal, "500");
const invalidWorkingMemory = assessOnDeviceLlmRuntimeManifest({
  document: { ...manifestV2, working_memory: { ...manifestV2.working_memory, scratch_bytes: "65" } },
}, analysis, llm);
assert(invalidWorkingMemory.issues.some((row) => row.code === "LLM_RUNTIME_WORKING_MEMORY_CONSERVATION_FAILED"));
analysis.artifact_bundle.files[analysis.artifact_bundle.files.findIndex((row) => row.role === "llm_runtime_manifest")].sha256 = "6".repeat(64);
const accountedLlm = buildOnDeviceLlmContract(analysis, { sidecars: { runtime_manifest: { document: manifestV2, path: "deepbom.runtime.json", sha256: "6".repeat(64) } } });
analysis.on_device_llm = accountedLlm;
assert.equal(accountedLlm.memory_feasibility.runtime_primary_residency.working_memory_accounted_bytes.decimal, "500");
assert.equal(accountedLlm.memory_feasibility.runtime_primary_residency.primary_allocated_accounted_bytes.decimal, "3924");
assert.equal(accountedLlm.memory_feasibility.runtime_primary_residency.allocation_accounting_status, "complete_observed_runtime_categories");
assert.deepEqual(validateOnDeviceLlmContract(analysis), { valid: true, errors: [] });
analysis.artifact_bundle.files[analysis.artifact_bundle.files.findIndex((row) => row.role === "llm_runtime_manifest")].sha256 = "5".repeat(64);
analysis.on_device_llm = boundLlm;

const mambaAnalysis = {
  format: "safetensors", model_sha256: "7".repeat(64),
  tensors: mambaTensors.map((row, index) => ({ ...row, index, dtype: "F16", byte_length: row.shape.reduce((product, value) => product * value, 1) * 2 })),
  tensor_storage_summary: storageSummary(mambaTensors),
  safetensors: { hf_architecture_contract: mamba }, artifact_bundle: { files: [] },
};
const mambaLlm = buildOnDeviceLlmContract(mambaAnalysis);
mambaAnalysis.on_device_llm = mambaLlm;
assert.equal(mambaLlm.state.kv_projection, null);
assert.equal(mambaLlm.state.recurrent_projection.recurrent_state_elements_all_layers_per_batch.decimal, "112");
assert.equal(mambaLlm.storage.layer_storage.status, "assessed_exact_serialized_layer_storage");
assert.equal(mambaLlm.storage.layer_storage.layer_bytes.decimal, "1360");
assert.equal(mambaLlm.storage.layer_storage.non_layer_bytes.decimal, "272");
assert.equal(buildLlmStateScenarioMatrix(mambaLlm).length, 9);
assert.deepEqual(validateOnDeviceLlmContract(mambaAnalysis), { valid: true, errors: [] });

const jambaAnalysis = {
  format: "safetensors", model_sha256: "b".repeat(64),
  tensors: jambaTensors.map((row, index) => ({ ...row, index, dtype: "F16", byte_length: row.shape.reduce((product, value) => product * value, 1) * 2 })),
  tensor_storage_summary: storageSummary(jambaTensors),
  safetensors: { hf_architecture_contract: jamba }, artifact_bundle: { files: [] },
};
const jambaLlm = buildOnDeviceLlmContract(jambaAnalysis);
jambaAnalysis.on_device_llm = jambaLlm;
assert.equal(jambaLlm.architecture.kind, "hybrid_attention_ssm_moe");
assert.equal(jambaLlm.architecture.ssm.recurrent_layer_count, 2);
assert.equal(jambaLlm.state.scenario_matrix[0].state_kind, "hybrid_kv_ssm");
assert.equal(jambaLlm.state.scenario_matrix[0].logical_bytes.decimal, "480");
const jambaDeclaredScenario = buildLlmStateScenarioMatrix(jambaLlm, { contexts: [16], batches: [1], storageBits: [16] });
assert.equal(jambaDeclaredScenario[0].logical_bytes.decimal, "960");
assert.deepEqual(validateOnDeviceLlmContract(jambaAnalysis), { valid: true, errors: [] });
const jambaRuntimeManifest = {
  ...manifest,
  artifact: { format: "safetensors", sha256: jambaAnalysis.model_sha256 },
  deployment: { evidence_class: "DECLARED", context_length: 16, batch_size: 1, state_storage_bits: 16 },
  weight_residency: {
    evidence_class: "OBSERVED_RUNTIME", artifact_serialized_tensor_bytes: jambaAnalysis.tensor_storage_summary.byte_length_decimal,
    runtime_weight_bytes: "1", cpu_bytes: "1", accelerator_bytes: "0", unresident_bytes: "0",
  },
  layer_placement: {
    evidence_class: "OBSERVED_RUNTIME", layer_count: 4,
    assignments: [0, 1, 2, 3].map((layer_index) => ({ layer_index, location: "cpu" })),
  },
  state_cache: {
    evidence_class: "OBSERVED_RUNTIME", kind: "hybrid_kv_ssm", logical_bytes: "960", allocated_bytes: "960",
    resident_bytes: "960", paging_enabled: false,
  },
};
const jambaRuntime = assessOnDeviceLlmRuntimeManifest({ document: jambaRuntimeManifest }, jambaAnalysis, jambaLlm);
assert.equal(jambaRuntime.status, "artifact_bound_observed_runtime");
assert.equal(jambaRuntime.issue_count, 0);
const wrongJambaRuntime = assessOnDeviceLlmRuntimeManifest({ document: { ...jambaRuntimeManifest, state_cache: { ...jambaRuntimeManifest.state_cache, logical_bytes: "959" } } }, jambaAnalysis, jambaLlm);
assert(wrongJambaRuntime.issues.some((row) => row.code === "LLM_RUNTIME_STATE_LOGICAL_BYTES_MISMATCH"));

const onnxGraphAnalysis = {
  format: "onnx",
  model_sha256: "8".repeat(64),
  ops: [
    { index: 0, name: "Gather", domain: "", version: 13 },
    { index: 1, name: "MatMul", domain: "", version: 13 },
    { index: 2, name: "GroupQueryAttention", domain: "com.microsoft", version: 1 },
    { index: 3, name: "LayerNormalization", domain: "", version: 17 },
    { index: 4, name: "Softmax", domain: "", version: 13 },
    { index: 5, name: "MatMul", domain: "", version: 13 },
  ],
  tensors: [
    { index: 0, name: "input_ids", dtype: "INT64", shape: [1, 16], role: "input" },
    { index: 1, name: "past_key_values.0.key", dtype: "FLOAT16", shape: [1, 4, 16, 8], role: "input" },
    { index: 2, name: "weight", dtype: "FLOAT16", shape: [32, 32], role: "initializer", constant_buffer: true, initializer_available_bytes: 2048, initializer_stored_elements: 1024 },
    { index: 3, name: "present.0.key", dtype: "FLOAT16", shape: [1, 4, 17, 8], role: "output" },
  ],
  inputs: [
    { index: 0, name: "input_ids", dtype: "INT64", shape: [1, 16] },
    { index: 1, name: "past_key_values.0.key", dtype: "FLOAT16", shape: [1, 4, 16, 8] },
  ],
  outputs: [{ index: 3, name: "present.0.key", dtype: "FLOAT16", shape: [1, 4, 17, 8] }],
  size_breakdown: { available_initializer_bytes: 2048, available_initializer_scalar_elements: 1024 },
  artifact_bundle: { files: [] },
};
onnxGraphAnalysis.on_device_llm = buildOnDeviceLlmContract(onnxGraphAnalysis);
assert.equal(onnxGraphAnalysis.on_device_llm.status, "partial_architecture_contract");
assert.equal(onnxGraphAnalysis.on_device_llm.serialized_graph.explicit_operator_count, 1);
assert.equal(onnxGraphAnalysis.on_device_llm.serialized_graph.transformer_motif_candidate, true);
assert.equal(onnxGraphAnalysis.on_device_llm.serialized_graph.external_state_candidate_count, 2);
assert.equal(onnxGraphAnalysis.on_device_llm.serialized_graph.external_state_candidates[0].logical_bytes_if_static.decimal, "1024");
assert.equal(onnxGraphAnalysis.on_device_llm.state.kv_projection, null, "Graph motifs must not invent KV architecture dimensions.");
assert.equal(onnxGraphAnalysis.on_device_llm.storage.serialized_tensor_bytes_decimal, "2048");
assert.deepEqual(validateOnDeviceLlmContract(onnxGraphAnalysis), { valid: true, errors: [] });
const onnxStructureOnly = structuredClone(onnxGraphAnalysis);
onnxStructureOnly.size_breakdown = { available_initializer_bytes: 0, available_initializer_scalar_elements: 0 };
onnxStructureOnly.onnx_external_data_structure_binding = {
  range_conservation_status: "complete",
  numerical_payload_decode: "not_assessed_scan_policy_structure",
  declared_payload_bytes: { decimal: "2048", number: 2048 },
  unique_payload_bytes: { decimal: "2048", number: 2048 },
  declared_element_count: { decimal: "1024", number: 1024 },
  encoding_inventory: [{
    dtype: "FLOAT16", tensor_count: 1,
    element_count: { decimal: "1024", number: 1024 },
    declared_payload_bytes: { decimal: "2048", number: 2048 },
    effective_bits_per_element: 16,
  }],
};
onnxStructureOnly.on_device_llm = buildOnDeviceLlmContract(onnxStructureOnly);
assert.equal(onnxStructureOnly.on_device_llm.storage.status, "assessed_serialized_constant_structure_payload_values_not_assessed");
assert.equal(onnxStructureOnly.on_device_llm.storage.serialized_parameter_count_decimal, "1024");
assert.equal(onnxStructureOnly.on_device_llm.storage.serialized_tensor_bytes_decimal, "2048");
assert.equal(onnxStructureOnly.on_device_llm.storage.effective_bits_per_parameter, 16);
assert.deepEqual(validateOnDeviceLlmContract(onnxStructureOnly), { valid: true, errors: [] });
const onnxLlmReport = buildEngineeringReport(onnxGraphAnalysis, { generatedAt: "2026-08-19T00:00:00.000Z" });
assert(onnxLlmReport.includes("## On-device LLM Evidence Contract")
  && onnxLlmReport.includes("### Serialized Transformer Graph Evidence")
  && !onnxLlmReport.includes("## Core ML Serialized Graph And Numerical Evidence"),
"ONNX LLM graph evidence is reported without cross-format Core ML content");
const onnxNonLlmAnalysis = structuredClone(onnxGraphAnalysis);
onnxNonLlmAnalysis.ops = [{ index: 0, name: "Conv", domain: "", version: 13, inputs: [0], outputs: [2], macs: 64, estimated_bytes: 128 }];
onnxNonLlmAnalysis.inputs = [{ index: 0, name: "image", dtype: "FLOAT32", shape: [1, 3, 4, 4] }];
onnxNonLlmAnalysis.outputs = [{ index: 2, name: "scores", dtype: "FLOAT32", shape: [1, 2] }];
onnxNonLlmAnalysis.on_device_llm = buildOnDeviceLlmContract(onnxNonLlmAnalysis);
const onnxNonLlmReport = buildEngineeringReport(onnxNonLlmAnalysis, { generatedAt: "2026-08-19T00:00:00.000Z" });
assert(onnxNonLlmReport.includes("## On-device LLM Evidence Contract")
  && onnxNonLlmReport.includes("Applicability scan complete for ONNX")
  && !onnxNonLlmReport.includes("### Serialized Transformer Graph Evidence"),
"A non-LLM ONNX graph reports its completed negative applicability scan without rendering the detailed LLM evidence tables");
const graphTamper = structuredClone(onnxGraphAnalysis.on_device_llm);
graphTamper.serialized_graph.explicit_operator_count = 2;
onnxGraphAnalysis.on_device_llm = graphTamper;
assert(validateOnDeviceLlmContract(onnxGraphAnalysis).errors.includes("serialized_llm_graph_mismatch"));

for (const [name, contract, sourceTensors, reportLabels] of [
  ["mixtral", mixtral, mixtralTensors, "Total expert matrix parameters Active expert matrix MACs"],
  ["mamba", mamba, mambaTensors, "Selective-scan arithmetic Recurrent-state cardinality"],
  ["jamba", jamba, jambaTensors, "Hybrid decode core Selective-scan arithmetic KV-state cardinality Recurrent-state cardinality"],
]) {
  let offset = 0;
  const ranged = sourceTensors.map((row) => {
    const byteLength = row.shape.reduce((product, value) => product * value, 1) * 2;
    const result = { ...row, byte_length: byteLength, data_offset: offset, data_end: offset + byteLength };
    offset += byteLength;
    return result;
  });
  const packedQuantization = buildSafeTensorsQuantizationContract(
    name === "mixtral" ? mixtralConfig : name === "jamba" ? jambaConfig : mambaConfig,
    ranged,
  );
  const report = `${"6".repeat(40)} ${contract.source.source_commit} ${contract.source.configuration_sources[name].sha256} ${contract.source.modeling_sources[name].sha256} does not serialize an execution-operator DAG ${reportLabels} Canonical / checkpoint parameters SafeTensors Packed-weight Quantization Contract NOT ASSESSED`;
  const failures = [];
  registerSafeTensorsSerializedConformance({
    staticAnalysis: { safetensors: { payload_coverage_status: "complete_without_gaps_or_overlaps", payload_byte_length: offset, sharded: false, reference_implementation: { commit: "6".repeat(40), tensor_rs_sha256: "7".repeat(64) }, hf_architecture_contract: contract, quantization_contract: packedQuantization }, ops: [], total_macs: null },
    tensors: ranged, ops: [], reportText: report,
    check: (id, passed, message) => { if (!passed) failures.push(`${id}:${message}`); },
  });
  assert.deepEqual(failures, [], `${name} serialized conformance`);
}

console.log("On-device LLM v2 MoE, SSM, scenario-matrix, and runtime-manifest contracts passed.");
