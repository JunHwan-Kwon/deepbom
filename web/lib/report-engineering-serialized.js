import { formatBytes, formatExactInteger, formatNumber } from "./format.js";
import { code, markdownTable } from "./report-utils.js";
import { coreMlFloorLabel } from "./coreml-deployment-contract.js";
import { safeTensorsQuantizationMarkdown } from "./report-engineering-safetensors-quantization.js";

export function onDeviceLlmEvidenceMarkdown(analysis) {
  const llm = analysis?.on_device_llm || {};
  if (llm.schema !== "deepbom.on_device_llm_contract.v2") return "";
  if (!hasReportableOnDeviceLlmEvidence(analysis, llm)) {
    const format = String(analysis?.format || "artifact").toUpperCase();
    const assessedOps = Number(llm.serialized_graph?.assessed_operator_count ?? analysis?.ops?.length ?? 0);
    return [
      "## On-device LLM Evidence Contract",
      `Applicability scan complete for ${format}: ${formatNumber(assessedOps)} serialized operator(s) assessed; no explicit Attention-family operator, bounded transformer motif, external state-name candidate, or bound architecture contract was detected.`,
      "> No LLM architecture, KV-state contract, decoder execution graph, runtime placement, or task capability is inferred from this negative static signal.",
    ].join("\n\n");
  }
  const architecture = llm.architecture || {};
  const storage = llm.storage || {};
  const tokenizer = llm.tokenizer || {};
  const generation = llm.generation || {};
  const state = llm.state || {};
  const compute = llm.compute?.projection || {};
  const runtime = llm.runtime_contract || {};
  const memory = llm.memory_feasibility || {};
  const placement = llm.static_memory_placement || {};
  const medical = llm.medical_ai_claim_boundary || {};
  const serializedGraph = llm.serialized_graph || null;
  const tensorRtLlm = llm.tensorrt_llm || {};
  const sidecarRows = Array.isArray(tokenizer.definition_files) ? tokenizer.definition_files : [];
  const scenarioRows = Array.isArray(state.scenario_matrix) ? state.scenario_matrix : [];
  const encodingRows = Array.isArray(storage.encoding_inventory) ? storage.encoding_inventory : [];
  const layerStorage = storage.layer_storage || {};
  const layerRows = Array.isArray(layerStorage.layers) ? layerStorage.layers : [];
  const memoryByScenario = new Map((memory.static_scenarios || []).map((row) => [
    `${row.state_kind}:${row.context_length ?? "none"}:${row.batch_size}:${row.storage_bits}`,
    row,
  ]));
  const computeRows = architecture.kind === "sparse_moe_decoder" ? [
    ["Total expert matrix parameters", compute.total_expert_matrix_parameters_all_layers?.decimal || "not derived"],
    ["Active expert matrix MACs / layer / token", compute.active_expert_matrix_macs_per_layer_per_token?.decimal || "not derived"],
    ["Active projection MACs, all layers / token", compute.active_projection_macs_all_layers_per_token?.decimal || "not derived"],
    ["Active decode core MACs at declared context", compute.decode_active_core_macs_at_declared_context?.decimal || "not derived"],
  ] : architecture.kind === "hybrid_attention_ssm_moe" ? [
    ["Attention / Mamba layers", `${compute.attention_layer_count ?? "?"} / ${compute.mamba_layer_count ?? "?"}`],
    ["Accounted projection MACs, all layers / token", compute.accounted_projection_macs_all_layers_per_token?.decimal || "not derived"],
    ["Active MoE MACs, all expert layers / token", compute.active_moe_macs_all_expert_layers_per_token?.decimal || "not derived"],
    ["Hybrid decode core MACs at declared context", compute.decode_accounted_core_macs_at_declared_context?.decimal || "not derived"],
    ["Selective-scan arithmetic", "excluded rather than estimated"],
  ] : architecture.kind === "ssm_recurrent" ? [
    ["Accounted projections + depthwise convolution / layer / token", compute.accounted_macs_per_layer_per_token?.decimal || "not derived"],
    ["Accounted MACs, all layers / token", compute.accounted_macs_all_layers_per_token?.decimal || "not derived"],
    ["Selective-scan arithmetic", "excluded rather than estimated"],
  ] : [
    ["Dense projections, all layers / token", compute.dense_projection_macs_all_layers_per_token?.decimal || "not derived"],
    ["Declared-context prefill core MACs", compute.prefill_transformer_core_macs_at_declared_context?.decimal || "not derived"],
    ["Declared-context decode core MACs", compute.decode_transformer_core_macs_at_declared_context?.decimal || "not derived"],
    ["Decode plus one logits position", compute.decode_with_one_logit_position_macs?.decimal || "not derived"],
  ];
  return [
    "## On-device LLM Evidence Contract",
    "",
    `Schema ${code(llm.schema)}; status **${llm.status || "not assessed"}**; evidence **${llm.evidence_class || "NOT_ASSESSABLE"}**.`,
    markdownTable(["Contract", "Artifact-derived value"], [
      ["Format / role", `${llm.format || "not bound"} / ${llm.artifact_role || "not declared"}`],
      ["Architecture", `${architecture.family || "not declared"}${architecture.architecture_names?.length ? ` (${architecture.architecture_names.join(", ")})` : ""}`],
      ["Architecture kind", architecture.kind || "not classified"],
      ["Context / vocabulary", `${architecture.context_length ?? "not declared"} / ${architecture.vocabulary_size ?? "not declared"}`],
      ["Hidden / intermediate / layers", [architecture.hidden_size, architecture.intermediate_size, architecture.layer_count].map((value) => value ?? "not derived").join(" / ")],
      ["Attention / KV heads / head width / GQA", [architecture.attention_head_count, architecture.kv_head_count, architecture.head_width, architecture.gqa_query_heads_per_kv_head].map((value) => value ?? "not derived").join(" / ")],
      ["MoE total / active experts", architecture.moe ? `${architecture.moe.expert_count} / ${architecture.moe.active_expert_count_per_token}` : "not applicable"],
      ["SSM state / convolution / time-step rank", architecture.ssm ? `${architecture.ssm.state_size} / ${architecture.ssm.convolution_kernel} / ${architecture.ssm.time_step_rank}` : "not applicable"],
      ["Position encoding", architecture.position_encoding?.status || "not projected"],
      ["Serialized parameters", `${storage.serialized_parameter_count_decimal || "not derived"}; ${storage.serialized_tensor_bytes_decimal || "not derived"} B; ${storage.effective_bits_per_parameter || "not derived"} effective bits/parameter`],
      ["Serialized layer storage", `${layerStorage.status || "not assessed"}; ${layerStorage.observed_layer_count ?? 0}/${layerStorage.expected_layer_count ?? "?"} layer namespaces; ${layerStorage.layer_bytes?.decimal || "?"} B layer + ${layerStorage.non_layer_bytes?.decimal || "?"} B non-layer`],
      ["Encoding signature schema", storage.encoding_signature_schema || "not emitted"],
      ["Encoding inventory / assignment SHA-256", `${storage.encoding_inventory_sha256 || "not derived"} / ${storage.tensor_encoding_assignment_sha256 || "not derived"}`],
      ["Tokenizer / chat template", `${tokenizer.status || "not assessed"} / ${tokenizer.chat_template?.status || "not declared"}`],
      ["Generation policy", generation.status || "not selected"],
      ["KV-state cardinality", state.kv_projection ? `${state.kv_projection.elements_per_token_per_batch?.decimal || "?"} elements/token/batch; ${state.kv_projection.elements_at_context_batch_one?.decimal || "?"} elements at declared context, batch 1` : "not derived"],
      ["Recurrent-state cardinality", state.recurrent_projection ? `${state.recurrent_projection.recurrent_state_elements_all_layers_per_batch?.decimal || "?"} elements/batch; context-independent after initialization` : "not applicable"],
      ["Compute scenario", compute.status || "not assessed"],
      ["Runtime binding", runtime.status || "not artifact-bound"],
      ["Memory feasibility", `${memory.status || "not assessed"}; fit claim ${memory.fit_claim || "not emitted"}`],
      ["Static memory capacity scope", `${memory.capacity_scope || "not assessed"}; ${memory.residency_assumption || "residency assumption not emitted"}`],
      ["Conditional CPU / accelerator placement", placement.status === "not_bound" ? "not bound" : `${placement.status || "not assessed"}; ${placement.lower_bound_not_exceeding_candidate_count ?? 0}/${placement.candidate_count ?? 0} candidates not disproven by accounted lower bounds`],
      ["Medical AI claim boundary", medical.status || "not established by model artifact"],
      ["Deployment declaration coverage", medical.declaration?.coverage ? `${medical.declaration.coverage.declared}/${medical.declaration.coverage.required}; ${medical.declaration.evidence_class}` : "0/9; NOT_ASSESSABLE"],
    ]),
    serializedGraph ? "### Serialized Transformer Graph Evidence" : "",
    serializedGraph ? markdownTable(["Field", "Observed value"], [
      ["Assessment", `${serializedGraph.status}; ${serializedGraph.evidence_class}`],
      ["Graph operators / explicit transformer operators", `${formatNumber(serializedGraph.graph_op_count)} / ${formatNumber(serializedGraph.explicit_operator_count)}`],
      ["MatMul / Softmax / normalization / Gather", `${formatNumber(serializedGraph.primitive_counts?.matrix_multiply || 0)} / ${formatNumber(serializedGraph.primitive_counts?.softmax || 0)} / ${formatNumber(serializedGraph.primitive_counts?.normalization || 0)} / ${formatNumber(serializedGraph.primitive_counts?.embedding_gather || 0)}`],
      ["Transformer-like motif", serializedGraph.transformer_motif_candidate ? "heuristic candidate; not architecture proof" : "not detected"],
      ["External state-name candidates", formatNumber(serializedGraph.external_state_candidate_count || 0)],
      ["Graph signature SHA-256", code(serializedGraph.graph_signature_sha256)],
    ]) : "",
    serializedGraph?.explicit_operators?.length ? markdownTable(["Op index", "Serialized operator", "Domain", "Version"], serializedGraph.explicit_operators.map((row) => [
      row.op_index, code(row.name), row.domain || "default", row.version ?? "not declared",
    ])) : "",
    serializedGraph ? `> ${serializedGraph.interpretation_boundary}` : "",
    tensorRtLlm.status && tensorRtLlm.status !== "not_selected" ? "### TensorRT-LLM Static Deployment Contract" : "",
    tensorRtLlm.status && tensorRtLlm.status !== "not_selected" ? markdownTable(["Field", "Observed or derived value"], [
      ["Status / evidence", `${tensorRtLlm.status} / ${tensorRtLlm.evidence_class}`],
      ["Engine config SHA-256", code(tensorRtLlm.engine_config?.sha256 || "not bound")],
      ["Model-source binding", `${tensorRtLlm.artifact_binding?.status || "not selected"}; ${tensorRtLlm.artifact_binding?.source_artifact_sha256 || "model-source digest not bound"}`],
      ["World / TP / PP / CP", `${tensorRtLlm.parallelism?.world_size ?? "?"} / ${tensorRtLlm.parallelism?.tensor_parallel_size ?? "?"} / ${tensorRtLlm.parallelism?.pipeline_parallel_size ?? "?"} / ${tensorRtLlm.parallelism?.context_parallel_size ?? "?"}`],
      ["Layers per PP rank", tensorRtLlm.parallelism?.layer_partition_per_pipeline_rank?.join(" / ") || "not derived"],
      ["Per-rank weight bytes", tensorRtLlm.parallelism?.weight_bytes_per_rank_reason || "not assessed"],
      ["Max input / sequence / batch", `${tensorRtLlm.build_limits?.max_input_length ?? "?"} / ${tensorRtLlm.build_limits?.max_sequence_length ?? "?"} / ${tensorRtLlm.build_limits?.maximum_batch_size ?? "?"}`],
      ["Weight / KV quantization", `${tensorRtLlm.quantization?.weight_activation_algorithm ?? "not declared"} / ${tensorRtLlm.quantization?.kv_cache_algorithm ?? "not declared"}`],
      ["Conditional logical KV state", tensorRtLlm.kv_cache_scenario?.logical_bytes?.decimal ? `${tensorRtLlm.kv_cache_scenario.logical_bytes.decimal} B` : "not derived"],
      ["Contract SHA-256", code(tensorRtLlm.contract_sha256 || "not emitted")],
    ]) : "",
    tensorRtLlm.status && tensorRtLlm.status !== "not_selected" ? `> ${tensorRtLlm.interpretation_boundary}` : "",
    encodingRows.length ? "### Serialized Precision And Encoding Inventory" : "",
    encodingRows.length ? markdownTable(["Encoding", "Tensors", "Elements", "Serialized bytes", "Effective bits / element"], encodingRows.map((row) => [
      code(row.dtype), formatNumber(row.tensor_count), row.element_count_decimal, row.byte_length_decimal, row.effective_bits_per_element || "not assessable",
    ])) : "",
    encodingRows.length ? `> ${storage.recipe_interpretation}` : "",
    layerRows.length ? "### Serialized Layer Storage Ledger" : "",
    layerRows.length ? markdownTable(["Layer", "Tensors", "Exact serialized bytes"], layerRows.map((row) => [
      row.layer_index, formatNumber(row.tensor_count), row.serialized_bytes?.decimal || "not derived",
    ])) : "",
    layerRows.length ? `Layer/non-layer conservation: **${layerStorage.conservation?.status || "not assessed"}**; observed layer namespaces ${formatNumber(layerStorage.observed_layer_count || 0)}/${formatNumber(layerStorage.expected_layer_count || 0)}; non-layer tensors ${formatNumber(layerStorage.non_layer_tensor_count || 0)} and ${layerStorage.non_layer_bytes?.decimal || "?"} B.` : "",
    layerRows.length ? `> ${layerStorage.boundary}` : "",
    placement.status && placement.status !== "not_bound" ? "### Conditional CPU / Accelerator Memory Placement" : "",
    placement.status && placement.status !== "not_bound" ? markdownTable(["Accelerator layers", "CPU layer bytes", "Accelerator layer bytes", "CPU lower bound", "Accelerator lower bound", "Assessment"], (placement.candidates || []).map((row) => [
      row.accelerator_layer_count,
      row.cpu_layer_serialized_bytes?.decimal || "not derived",
      row.accelerator_layer_serialized_bytes?.decimal || "not derived",
      row.cpu_accounted_lower_bound_bytes?.decimal || "not derived",
      row.accelerator_accounted_lower_bound_bytes?.decimal || "not derived",
      row.status,
    ])) : "",
    placement.status && placement.status !== "not_bound" ? `Profile SHA-256 ${code(placement.normalized_profile_sha256)}; candidate range not disproven by accounted lower bounds: ${placement.minimum_accelerator_layer_count_not_disproven ?? "none"}-${placement.maximum_accelerator_layer_count_not_disproven ?? "none"}.` : "",
    placement.status && placement.status !== "not_bound" ? `> ${placement.boundary}` : "",
    scenarioRows.length ? `### Conditional ${state.kv_projection && state.recurrent_projection ? "Hybrid KV + SSM State" : state.recurrent_projection ? "Recurrent-state" : "KV-cache"} And Memory Lower-bound Scenarios` : "",
    scenarioRows.length ? markdownTable(["State", "Context", "Batch", "Storage", "Logical state bytes", "Conditional resident-set lower bound", "First tier not exceeded", "Lower bound exceeds tiers", "Evidence"], scenarioRows.map((row) => {
      const feasibility = memoryByScenario.get(`${row.state_kind}:${row.context_length ?? "none"}:${row.batch_size}:${row.storage_bits}`) || {};
      return [
        row.state_kind, row.context_length ?? "context-independent", row.batch_size, `${row.storage_bits} bit`, row.logical_bytes?.decimal || "not derived",
        feasibility.static_lower_bound_bytes?.decimal || "not derived", feasibility.first_capacity_not_exceeded || "not assessed",
        feasibility.lower_bound_exceeded_capacity_count ?? "not assessed", row.evidence_class,
      ];
    })) : "",
    scenarioRows.length ? `> ${state.scenario_boundary || "KV byte rows are conditional scenarios, not runtime memory measurements."}` : "",
    memory.boundary ? `> ${memory.boundary}` : "",
    sidecarRows.length ? "### Hash-bound Tokenizer And Generation Sidecars" : "",
    sidecarRows.length ? markdownTable(["Role", "Path", "Bytes", "SHA-256"], sidecarRows.map((row) => [row.role, code(row.path), formatNumber(row.byte_length), code(row.sha256)])) : "",
    generation.values && Object.keys(generation.values).length ? `Declared generation values: ${code(JSON.stringify(generation.values))}. These are repository declarations, not observed runtime settings.` : "",
    compute.status ? markdownTable(["Architecture scenario", "Exact value"], computeRows) : "",
    compute.status ? `> ${llm.compute?.boundary || compute.boundary || "Architecture equations do not establish runtime work."}` : "",
    "### Runtime And Medical Deployment Evidence Boundary",
    "",
    `Runtime binding contract: ${(runtime.required_bindings || []).map((value) => code(value)).join(", ") || "not enumerated"}; status ${code(runtime.status || "not artifact-bound")}.`,
    runtime.source ? `Runtime manifest: ${code(runtime.source)}; SHA-256 ${code(runtime.source_sha256)}; ${runtime.evidence_class}.` : "",
    runtime.weight_residency ? `Exclusive runtime weight residency: CPU ${runtime.weight_residency.cpu_bytes?.decimal || "?"} B / accelerator ${runtime.weight_residency.accelerator_bytes?.decimal || "?"} B / unresident ${runtime.weight_residency.unresident_bytes?.decimal || "?"} B; runtime total ${runtime.weight_residency.runtime_weight_bytes?.decimal || "?"} B.` : "",
    runtime.layer_placement ? `Layer placement: CPU ${runtime.layer_placement.cpu_layer_count} / accelerator ${runtime.layer_placement.accelerator_layer_count} / unresident ${runtime.layer_placement.unresident_layer_count}; evidence ${runtime.layer_placement.evidence_class}.` : "",
    runtime.state_cache ? `Runtime state cache: ${runtime.state_cache.kind}; logical ${runtime.state_cache.logical_bytes?.decimal || "?"} B / resident ${runtime.state_cache.resident_bytes?.decimal || "?"} B / allocated ${runtime.state_cache.allocated_bytes?.decimal || "?"} B; paging ${runtime.state_cache.paging_enabled ? "enabled" : "disabled"}; evidence ${runtime.state_cache.evidence_class}.` : "",
    runtime.working_memory ? `Runtime working-memory accounting: workspace ${runtime.working_memory.graph_workspace_bytes?.decimal || "?"} B / scratch ${runtime.working_memory.scratch_bytes?.decimal || "?"} B / packing and replicas ${runtime.working_memory.packing_and_replica_bytes?.decimal || "?"} B / allocator overhead ${runtime.working_memory.allocator_overhead_bytes?.decimal || "?"} B / backend-private ${runtime.working_memory.backend_private_bytes?.decimal || "?"} B / other ${runtime.working_memory.other_runtime_bytes?.decimal || "?"} B; conserved subtotal ${runtime.working_memory.accounted_nonweight_runtime_bytes?.decimal || "?"} B; ${runtime.working_memory.coverage}.` : "Runtime working-memory categories: not bound; they are not treated as zero.",
    memory.runtime_primary_residency ? `Runtime primary-memory accounting: resident lower bound ${memory.runtime_primary_residency.primary_resident_lower_bound_bytes?.decimal || "?"} B / allocated lower bound ${memory.runtime_primary_residency.primary_allocated_lower_bound_bytes?.decimal || "?"} B${memory.runtime_primary_residency.primary_allocated_accounted_bytes ? ` / accounted allocation ${memory.runtime_primary_residency.primary_allocated_accounted_bytes.decimal} B` : " / accounted allocation NOT ASSESSABLE"}; first reference tiers not exceeded ${memory.runtime_primary_residency.resident_capacity_assessment?.first_capacity_not_exceeded || "?"} / ${memory.runtime_primary_residency.allocated_capacity_assessment?.first_capacity_not_exceeded || "?"}. No per-pool or whole-device fit claim is emitted.` : "",
    `Medical declaration fields still required: ${(medical.required_external_evidence || []).map((value) => code(value)).join(", ") || "none missing from the declaration; validation and release evidence remain external"}.`,
    medical.declaration?.sha256 ? `Hash-bound deployment declaration: ${code(medical.declaration.source)}; SHA-256 ${code(medical.declaration.sha256)}. Its contents remain ${medical.declaration.evidence_class}; presence is not validation.` : "",
    `Not established by this artifact: ${(medical.not_established || []).map((value) => code(value)).join(", ") || "task accuracy, clinical validity, runtime assignment, and release readiness"}.`,
    llm.issue_count ? markdownTable(["LLM contract issue", "Evidence"], (llm.issues || []).map((row) => [row.code || "LLM_CONTRACT_ISSUE", code(JSON.stringify(row))])) : "",
  ].filter(Boolean).join("\n");
}

function hasReportableOnDeviceLlmEvidence(analysis, llm) {
  const format = String(analysis?.format || "").toLowerCase();
  if (format === "gguf") return true;
  if (llm.architecture?.family || (llm.architecture?.kind && !String(llm.architecture.kind).startsWith("not_"))) return true;
  const graph = llm.serialized_graph || {};
  if (Number(graph.explicit_operator_count || 0) > 0 || Number(graph.external_state_candidate_count || 0) > 0
    || graph.transformer_motif_candidate === true) return true;
  if (llm.tensorrt_llm?.status && llm.tensorrt_llm.status !== "not_selected") return true;
  if (llm.runtime_contract?.source || llm.medical_ai_claim_boundary?.declaration?.sha256) return true;
  return Boolean(llm.tokenizer?.definition_files?.length || (llm.generation?.values && Object.keys(llm.generation.values).length));
}

export function serializedFormatEvidenceMarkdown(analysis) {
  const format = String(analysis?.format || "").toLowerCase();
  if (!["gguf", "safetensors", "coreml", "executorch"].includes(format)) return "";
  const storage = analysis.tensor_storage_summary || {};
  const encodingRows = Array.isArray(storage.encodings) ? storage.encodings : [];
  const duplicateRows = Array.isArray(storage.duplicate_groups) ? storage.duplicate_groups : [];
  const storageLedger = storage.schema ? [
    "## Serialized Tensor Storage Ledger (OBSERVED + DERIVED)",
    "",
    markdownTable(["Field", "Value"], [
      ["Stored scalar elements", `${storage.element_count_decimal || "not assessed"}${storage.element_count == null && storage.element_count_decimal ? " (decimal exact; exceeds binary64 safe integer range)" : ""}`],
      ["Declared tensor bytes", `${storage.byte_length_decimal || "not assessed"} B`],
      ["Effective stored bits / element", storage.effective_bits_per_element || "not assessed"],
      ["Encoding families", formatNumber(storage.encoding_count || 0)],
      ["Content-addressed duplicate candidates", `${formatNumber(storage.content_addressed_duplicate_group_count || 0)} group(s), ${formatNumber(storage.content_addressed_duplicate_bytes_after_first || 0)} B after first copies`],
    ]),
    encodingRows.length ? "### Encoding Distribution" : "",
    encodingRows.length ? markdownTable(["Encoding", "Tensors", "Elements", "Bytes", "Effective bits / element"], encodingRows.map((row) => [
      code(row.dtype), formatNumber(row.tensor_count), row.element_count_decimal, row.byte_length_decimal, row.effective_bits_per_element || "not assessed",
    ])) : "",
    duplicateRows.length ? "### Content-addressed Duplicate Candidates" : "",
    duplicateRows.length ? markdownTable(["Dtype / shape", "Tensors", "Bytes after first", "Payload SHA-256", "Names"], duplicateRows.slice(0, 32).map((row) => [
      `${row.dtype} / ${(row.shape || []).join("x") || "scalar"}`, formatNumber(row.tensor_count), formatNumber(row.duplicate_bytes_after_first), row.payload_sha256, (row.tensor_names || []).map(code).join(" / "),
    ])) : "",
    duplicateRows.length > 32 ? `> ${formatNumber(duplicateRows.length - 32)} additional duplicate candidate group(s) remain in structured evidence.` : "",
    "> Duplicate candidates require identical dtype, shape, byte length, and payload SHA-256. This does not prove runtime aliasing or semantic weight tying.",
  ].filter(Boolean).join("\n") : "";
  if (format === "executorch") {
    const program = analysis.executorch_program || null;
    const flatTensor = analysis.executorch_flat_tensor || null;
    const pte = analysis.executorch_container === "pte";
    const segments = pte ? program?.segments || [] : flatTensor?.segments || [];
    const tensors = Array.isArray(analysis.tensors) ? analysis.tensors : [];
    const external = program?.external_tensor_data || {};
    return [
      "## ExecuTorch Serialized Contract (OBSERVED + SOURCE_PINNED + DERIVED)",
      "",
      `### ${pte ? "ET12 Program Evidence" : "FT01 Tensor Data Evidence"}`,
      "",
      markdownTable(["Field", "Value"], [
        ["Wire identifier / schema version", `${pte ? program?.identifier || "not decoded" : flatTensor?.identifier || "not decoded"} / ${analysis.version ?? "not decoded"}`],
        ["Pinned schema", `${program?.source?.repository || flatTensor?.source?.repository || "pytorch/executorch"}@${program?.source?.commit || flatTensor?.source?.commit || "not bound"}`],
        ["Execution plans / instructions", pte ? `${formatNumber(analysis.subgraphs || 0)} / ${formatNumber(analysis.operator_count || 0)}` : "not applicable to FT01"],
        ["Kernel / delegate calls", pte ? `${formatNumber(program?.kernel_instruction_count || 0)} / ${formatNumber(program?.delegate_instruction_count || 0)}` : "not applicable"],
        ["Tensor contracts", `${formatNumber(tensors.length)} total / ${formatNumber(tensors.filter((tensor) => tensor.buffer_data_length_decimal != null).length)} logical payloads statically bounded`],
        ["Appended segments", `${formatNumber(segments.length)} / ${analysis.size_breakdown?.appended_segment_bytes_decimal || "0"} B / ${analysis.size_breakdown?.status || "status not emitted"}`],
        ["Stored tensor integrity", `${analysis.weight_integrity?.status || "not assessed"}; ${formatNumber(analysis.weight_integrity?.assessed_tensors || 0)} tensor(s). ${analysis.weight_integrity?.detail || "No interpretation boundary emitted."}`],
        ["Nominal MAC contract", `${analysis.mac_assessment?.status || "not assessed"}; complete=${String(analysis.mac_assessment?.complete === true)}. ${analysis.mac_assessment?.detail || "No nominal MAC method is bound."}`],
        ["Planned non-constant memory", pte ? `${analysis.tensor_liveness?.status || "status not emitted"}; ${analysis.tensor_liveness?.planned_non_const_memory_decimal || "0"} B across ${(analysis.tensor_liveness?.per_device || []).map((row) => `${row.device}:${row.bytes_decimal} B`).join(" / ") || "no materialized buffers"}` : `${analysis.tensor_liveness?.status || "not applicable"}`],
        ["Peak instruction-level AOT address occupancy", pte ? analysis.tensor_liveness?.peak_planned_live_allocation_decimal == null
          ? `${analysis.tensor_liveness?.peak_planned_live_allocation_status || "not assessed"}; no value promoted`
          : `${analysis.tensor_liveness.peak_planned_live_allocation_decimal} B; ${analysis.tensor_liveness.peak_planned_live_allocation_status}. Planned address union only, not runtime RSS.` : "not applicable to FT01"],
        ["External PTD resolution", pte ? `${external.status || "not applicable"}; ${formatNumber(external.verified_contract_count || 0)}/${formatNumber(external.required_name_count || 0)} exact name/dtype/shape/logical-byte/layout contracts; ${external.verified_logical_bytes_decimal || "0"} B verified` : "this artifact is FT01 data"],
        ["Runtime compatibility boundary", `${analysis.runtime_compat?.runtime_version_basis || "No runtime-version basis emitted."} ${analysis.runtime_compat?.detail || ""}`.trim()],
        ["Claim boundary", pte ? program?.graph_boundary || "Argument direction, delegate-internal graphs, and runtime execution are not inferred." : "FT01 does not serialize an execution graph."],
      ]),
      pte && program?.plans?.length ? "### Execution Plans" : "",
      pte && program?.plans?.length ? markdownTable(["Plan", "Values / tensors", "Registry / delegates", "Chains / instructions", "Planned memory"], program.plans.map((plan) => [
        code(plan.name), `${formatNumber(plan.value_count)} / ${formatNumber(plan.tensor_count)}`, `${formatNumber(plan.operator_registry_count)} / ${formatNumber(plan.delegate_count)}`, `${formatNumber(plan.chain_count)} / ${formatNumber(plan.instruction_count)} (${formatNumber(plan.kernel_instruction_count)} kernel + ${formatNumber(plan.delegate_instruction_count)} delegate + ${formatNumber(plan.non_compute_instruction_count)} move/control/free)`, `${plan.non_const_memory_bytes_decimal} B`,
      ])) : "",
      pte && external.bindings?.length ? "### External PTD Tensor Bindings" : "",
      pte && external.bindings?.length ? markdownTable(["Tensor", "PTD", "Status", "Dtype / shape", "Logical / serialized span bytes", "Reasons"], external.bindings.map((row) => [
        code(row.name), code(row.filename), row.status, `${row.dtype || "?"} / ${(row.shape || []).join("x") || "scalar"}`, `${row.logical_bytes_decimal || "?"} / ${row.serialized_span_bytes_decimal || "?"}`, (row.reasons || []).join(" / ") || "none",
      ])) : "",
      pte && program?.delegates?.length ? "### Serialized Backend Delegates" : "",
      pte && program?.delegates?.length ? markdownTable(["Plan", "Backend", "Processed data", "Compile specs"], program.delegates.map((row) => [
        formatNumber(row.plan_index), code(row.backend_id), `${row.processed_location}:${row.processed_index}`, (row.compile_specs || []).map((spec) => `${code(spec.key)} (${formatNumber(spec.value_bytes)} B)`).join(" / ") || "none",
      ])) : "",
      "> Serialized DelegateCall rows are AOT artifact evidence. Portable KernelCall direction is source-bound only when its argument vector matches the pinned registry; unmatched/custom KernelCall and DelegateCall internals remain unbound. Planned address occupancy does not establish runtime RSS, backend availability, successful delegate initialization, executed assignment, physical transfer, or latency.",
    ].filter(Boolean).join("\n");
  }
  if (format === "gguf") {
    const gguf = analysis.gguf || {};
    const status = analysis.quantization_status || {};
    const source = gguf.type_traits_source || {};
    const backend = gguf.backend_compatibility || {};
    return [
      storageLedger,
      onDeviceLlmEvidenceMarkdown(analysis),
      storageLedger ? "" : "",
      "## GGUF Tensor Encoding Evidence (OBSERVED + SOURCE_PINNED)",
      "",
      markdownTable(["Field", "Value"], [
        ["Serialized tensors", formatNumber(analysis.tensor_count || 0)],
        ["Block-encoded / scalar / unsupported", `${formatNumber(status.block_quantized_tensor_count || 0)} / ${formatNumber(status.scalar_encoded_tensor_count || 0)} / ${formatNumber(status.unsupported_encoding_tensor_count || 0)}`],
        ["Declared tensor payload", `${formatBytes(gguf.declared_tensor_byte_length || 0)} (${formatNumber(gguf.declared_tensor_byte_length || 0)} B)`],
        ["Payload range conservation", gguf.payload_coverage_status || "not assessed"],
        ["Serialized endian", gguf.endianness || "not bound"],
        ["Quantization version", gguf.quantization_version == null ? "not declared" : code(gguf.quantization_version)],
        ["GGML type-trait source", source.source_commit ? `${source.repository || "ggml-org/llama.cpp"}@${source.source_commit}; ${source.type_traits_source || "ggml/src/ggml.c"} SHA-256 ${code(source.type_traits_source_sha256 || "not embedded")}` : "not bound"],
        ["GGML block-layout source", source.source_commit ? `${source.repository || "ggml-org/llama.cpp"}@${source.source_commit}; ${source.block_layout_source || "ggml/src/ggml-common.h"} SHA-256 ${code(source.block_layout_source_sha256 || "not embedded")}` : "not bound"],
        ["Execution graph boundary", "GGUF does not serialize an execution-operator DAG; delegation, Q/DQ placement, activation precision, and runtime assignment are not inferred"],
      ]),
      backend.schema ? "### llama.cpp Backend Prerequisite Contract" : "",
      backend.schema ? markdownTable(["Field", "Value"], [
        ["Status", `${backend.status || "not assessed"}; compatibility ${backend.compatibility_conclusion || "not concluded"}`],
        ["Architecture registry", `${backend.architecture || "not declared"}; match ${backend.architecture_registry_match ? "yes" : "no"} against ${formatNumber(backend.pinned_architecture_count || 0)} pinned names`],
        ["Storage precheck", `${formatNumber(backend.tensor_count || 0)} tensor(s), ${formatNumber(backend.encoding_count || 0)} encoding(s), ${formatNumber(backend.invalid_or_unknown_storage_tensor_count || 0)} invalid or unknown`],
        ["Selected backend / build", `${backend.selected_backend_profile_id || "not selected"} / ${backend.selected_runtime_build_manifest_status || "not bound"}`],
        ["Execution graph", backend.execution_graph_status || "not assessed"],
        ["Pinned source", backend.source?.source_commit ? `${backend.source.repository}@${backend.source.source_commit}; ${(Object.values(backend.source.files || {})).map((row) => `${row.path} SHA-256 ${row.sha256}`).join("; ")}` : "not bound"],
        ["Claim boundary", backend.boundary || "Static prerequisite evidence does not establish backend execution."],
      ]) : "",
      backend.profiles?.length ? markdownTable(["Backend", "Build option / default", "Registration", "Artifact precheck", "Runtime status"], backend.profiles.map((row) => [
        row.label,
        `${row.cmake_option} / ${row.declared_default}`,
        `${row.compiled_registration_macro} -> ${row.registration_function}`,
        row.assessment_status,
        `${row.build_option_binding}; ${row.operator_support_evidence}`,
      ])) : "",
    ].join("\n");
  }
  if (format === "safetensors") {
    const safe = analysis.safetensors || {};
    const source = safe.reference_implementation || {};
    const hf = safe.hf_architecture_contract || {};
    const hfFields = hf.fields || {};
    const tensorContract = hf.tensor_contract || {};
    return [
      storageLedger,
      onDeviceLlmEvidenceMarkdown(analysis),
      safeTensorsQuantizationMarkdown(analysis),
      "## SafeTensors Storage Evidence (OBSERVED)",
      "",
      markdownTable(["Field", "Value"], [
        ["Serialized tensors", formatNumber(analysis.tensor_count || 0)],
        ["Header bytes", formatNumber(safe.header_byte_length || 0)],
        ["Tensor payload bytes", formatNumber(safe.payload_byte_length || 0)],
        ["Payload range conservation", safe.payload_coverage_status || "not assessed"],
        ["Duplicate-key validation", safe.duplicate_key_validation || "not assessed"],
        ...(safe.sharded ? [
          ["Sharded repository", `${formatNumber(safe.shard_count || 0)} shard(s); ${formatNumber(safe.index_tensor_count || 0)} index tensor binding(s); ${safe.index_binding_status || "not assessed"}`],
          ["Bundle identity", analysis.artifact_bundle?.bundle_sha256 ? `SHA-256 ${code(analysis.artifact_bundle.bundle_sha256)} over canonical path/size/digest/role manifest` : "not bound"],
        ] : []),
        ["Model-source identity", analysis.artifact_bundle?.model_source_sha256
          ? `SHA-256 ${code(analysis.artifact_bundle.model_source_sha256)} over ${formatNumber(analysis.artifact_bundle.model_source_file_count || 0)} shard-index, architecture-config, quantization-config, and tensor-shard file(s); deployment/binding sidecars excluded by role`
          : "not emitted for a standalone file analysis"],
        ["Reference implementation", source.commit ? `${source.repository || "huggingface/safetensors"}@${source.commit}; ${source.tensor_source || "safetensors/src/tensor.rs"} SHA-256 ${code(source.tensor_rs_sha256 || "not embedded")}` : "not bound"],
        ["Packed low-precision binding", source.pytorch_commit ? `${source.pytorch_repository || "pytorch/pytorch"}@${source.pytorch_commit}; SafeTensors torch binding SHA-256 ${code(source.torch_binding_source_sha256 || "not embedded")}; ${formatNumber((source.pytorch_sources || []).length)} pinned dtype implementation source(s); OCP MX ${source.ocp_mx_version || "not bound"}` : "not bound"],
        ["F6 numerical boundary", source.subbyte_boundary || "not declared"],
        ["Execution quantization boundary", "SafeTensors standardizes tensor names, dtype, shape, and byte ranges; it does not serialize an execution graph or activation Q/DQ placement. A selected repository may separately bind a source-pinned packed-weight contract."],
      ]),
      hf.schema ? "### Hugging Face Canonical Tensor Binding" : "",
      hf.schema ? markdownTable(["Field", "Value"], [
        ["Config binding", `${safe.hf_config_status || "not assessed"}${safe.hf_config_path ? `; ${code(safe.hf_config_path)}` : ""}`],
        ["Status / model type", `${hf.status || "not assessed"} / ${hf.model_type || "not declared"}`],
        ["Canonical module layout", hf.architecture_kind === "sparse_moe_decoder"
          ? `${hf.tensor_layout_id || "not registered"}; ${hfFields.num_local_experts || "?"} serialized experts, top-${hfFields.num_experts_per_tok || "?"} active per token; router and expert tensors source-bound`
          : hf.architecture_kind === "hybrid_attention_ssm_moe"
            ? `${hf.tensor_layout_id || "not registered"}; ${hfFields.attention_layer_count || "?"} attention + ${hfFields.mamba_layer_count || "?"} Mamba layers; ${hfFields.expert_feed_forward_layer_count || "?"} expert FFN layers; hybrid KV/SSM state source-bound`
          : hf.architecture_kind === "ssm_recurrent"
            ? `${hf.tensor_layout_id || "not registered"}; recurrent SSM state ${hfFields.state_size || "?"}, convolution kernel ${hfFields.conv_kernel || "?"}, time-step rank ${hfFields.time_step_rank || "?"}`
            : `${hf.tensor_layout_id || "not registered"}; ${hf.mlp_projection_matrix_count || "not derived"} MLP projection matrix/matrices per layer; attention bias scope ${hf.attention_bias_scope || "not derived"}; MLP bias ${hfFields.mlp_bias === true ? "all MLP projections" : "absent"}`],
        ["Canonical tensor contract", `${tensorContract.status || "not assessed"}; shapes ${formatNumber(tensorContract.canonical_tensor_shape_match_count || 0)}/${formatNumber(tensorContract.canonical_tensor_check_count || 0)} matched; ${formatNumber(tensorContract.canonical_tensor_missing_count || 0)} required tensor(s) missing`],
        ["Canonical / checkpoint parameters", `${tensorContract.canonical_expected_parameter_count_decimal || "not derived"} expected canonical parameter(s); ${tensorContract.canonical_observed_parameter_count_decimal || "not derived"} observed across canonical names; ${tensorContract.checkpoint_parameter_count_decimal || "not derived"} in the complete checkpoint`],
        ["Tied-weight evidence", tensorContract.tied_weight_binding_status || "not assessed"],
        ["Pinned source", hf.source?.source_commit ? `${hf.source.repository}@${hf.source.source_commit} (${hf.source.release_tag || "untagged"}); config ${hf.source.configuration_sources?.[hf.model_type]?.path || "not registered"} SHA-256 ${code(hf.source.configuration_sources?.[hf.model_type]?.sha256 || "not bound")}; state-dict modules ${hf.source.modeling_sources?.[hf.model_type]?.path || "not registered"} SHA-256 ${code(hf.source.modeling_sources?.[hf.model_type]?.sha256 || "not bound")}` : "not bound"],
        ["Claim boundary", hf.compute_projection?.boundary || hf.cache_dtype_boundary || hf.boundary || hf.reason || "No architecture contract was derived"],
      ]) : "",
      hf.issue_count ? markdownTable(["Architecture issue", "Severity", "Evidence"], (hf.issues || []).map((row) => [row.code, row.severity || "review", code(JSON.stringify(row))])) : "",
    ].join("\n");
  }
  if (format !== "coreml") return "";
  const coreml = analysis.coreml || {};
  const status = analysis.quantization_status || {};
  const source = coreml.source_basis || {};
  const deploymentFloor = coreml.deployment_floor || {};
  const ops = Array.isArray(analysis.ops) ? analysis.ops : [];
  const integrity = analysis.weight_integrity || {};
  const macs = analysis.mac_assessment || {};
  const size = analysis.size_breakdown || {};
  const live = analysis.tensor_liveness || {};
  const preprocessing = Array.isArray(coreml.neural_network?.preprocessing) ? coreml.neural_network.preprocessing : [];
  const classical = coreml.classical_model || null;
  const pipeline = coreml.pipeline || null;
  const milScopes = coreml.mil_scope_intrinsic_cost || null;
  const milCompression = coreml.mil_compression_contract || null;
  const flexibleScenarios = analysis.flexible_input_scenarios || coreml.flexible_input_scenarios || null;
  const graphUnit = coreml.model_type === "mlProgram" ? "MIL SSA operation(s)"
    : pipeline ? "pipeline operation(s)"
      : classical ? "classical model operation(s)" : "legacy NeuralNetwork layer(s)";
  const layerLimit = 200;
  const layerRows = ops.slice(0, layerLimit).map((op) => {
    const weights = Array.isArray(op.coreml_weights) ? op.coreml_weights
      : Array.isArray(op.coreml_classical_model?.parameters) ? op.coreml_classical_model.parameters : [];
    const storage = weights.length
      ? weights.map((weight) => `${weight.role}:${weight.storage}`).join(", ")
      : "none decoded";
    const bytes = weights.reduce((sum, weight) => sum + Number(weight.byte_length || 0), 0);
    return [
      `#${formatNumber(op.index)}`,
      `${op.name}${op.coreml_layer_name ? ` / ${code(op.coreml_layer_name)}` : ""}`,
      `${(op.inputs || []).length} / ${(op.outputs || []).length}`,
      `${(op.output_shapes || []).length ? (op.output_shapes || []).map((shape) => code(`[${shape.join(", ")}]`)).join(" / ") : "not derived"}${op.shape_status ? `; ${op.shape_status}` : ""}`,
      op.macs_decimal != null || op.macs != null
        ? `${formatExactInteger(op.macs_decimal, op.macs)} MACs${op.macs == null ? " (exact decimal; numeric mirror withheld)" : ""}`
        : op.macs_status || "not assessed",
      `${formatNumber(weights.length)} / ${formatNumber(bytes)} B`,
      storage,
      op.quantization_state || "none",
      op.coreml_weight_scan_status || "not assessed",
    ];
  });
  const classicalRows = classical ? [
    ["Classical payload", classical.kind],
    ...(classical.kind.startsWith("glm") ? [
      ["Coefficient matrix", `${formatNumber(classical.coefficient_row_count)} row(s) x ${formatNumber(classical.coefficient_width)} coefficient(s); ${formatNumber(classical.offset_count)} offset(s)`],
      ["Class contract", classical.class_labels ? `${formatNumber(classical.class_labels.values.length)} ${classical.class_labels.kind} labels; encoding ${classical.class_encoding}` : "regression"],
    ] : []),
    ...(classical.kind.startsWith("supportVector") ? [
      ["Kernel / support vectors", `${classical.kernel.kind}; ${formatNumber(classical.support_vectors.count)} ${classical.support_vectors.kind} vector(s), width ${formatNumber(classical.support_vectors.width)}`],
      ["Coefficient / rho contract", `${formatNumber(classical.coefficient_row_count)} row(s), ${formatNumber(classical.coefficient_count)} coefficient(s), ${formatNumber(classical.rho_count)} rho value(s)`],
    ] : []),
    ...(classical.kind.startsWith("treeEnsemble") ? [
      ["Tree structure", `${formatNumber(classical.tree_count)} tree(s); ${formatNumber(classical.branch_node_count)} branch / ${formatNumber(classical.leaf_node_count)} leaf node(s); maximum depth ${formatNumber(classical.maximum_depth)}`],
      ["Prediction contract", `${formatNumber(classical.prediction_dimension_count)} dimension(s); ${formatNumber(classical.base_prediction_count)} serialized base value(s)`],
    ] : []),
    ["Source validation", classical.source_validation],
  ] : [];
  const pipelineRows = pipeline?.model_summaries || [];
  const milScopeRows = (milScopes?.scope_rows || []).map((row) => [
    row.scope_class || "scope", code(row.scope || "unnamed"), row.status || "not assessed", formatNumber(row.operator_count || 0),
    `${formatNumber(row.assessed_mac_operator_count || 0)}/${formatNumber(row.mac_compute_operator_count || 0)}`,
    row.complete_nominal_macs_decimal == null ? `${row.assessed_nominal_macs_decimal || "0"} assessed subtotal` : `${row.complete_nominal_macs_decimal} complete`,
    row.complete_output_payload_bytes_decimal == null ? `${row.assessed_output_payload_bytes_decimal || "0"} B assessed subtotal` : `${row.complete_output_payload_bytes_decimal} B complete`,
    row.scope_local_liveness?.peak_bytes == null ? row.scope_local_liveness?.status || "not assessed" : `${formatBytes(row.scope_local_liveness.peak_bytes)}; ${row.scope_local_liveness.status}`,
  ]);
  const compressionRows = (milCompression?.transforms || []).map((row) => [
    `#${formatNumber(row.op_index)}`,
    code(row.op_type || row.transform || "unknown"),
    row.representation || "not classified",
    row.status || "not assessed",
    row.logical_output_elements == null ? "not assessed" : formatNumber(row.logical_output_elements),
    row.scale_elements != null
      ? `${formatNumber(row.scale_elements)} scale(s); block ${code(`[${(row.block_shape || []).join(", ")}]`)}`
      : row.palette_count != null
        ? `${formatNumber(row.palette_count)} palette value(s); ${formatNumber(row.index_bits)}-bit index; vector ${formatNumber(row.vector_size)}${row.vector_axis == null ? "" : ` on axis ${formatNumber(row.vector_axis)}`}`
        : row.stored_nonzero_elements != null
          ? `${formatNumber(row.stored_nonzero_elements)} serialized nonzero value(s); mask population ${row.mask_population_status || "not assessed"}`
          : "source rule not implemented",
    row.source_file && row.source_sha256 ? `${row.source_file}; SHA-256 ${code(row.source_sha256)}` : "not source-bound",
  ]);
  const flexibleScenarioRows = (flexibleScenarios?.scenarios || []).map((row) => [
    `#${formatNumber(row.scenario_index)}`,
    row.scenario_kind || "case",
    (row.input_shapes || []).map((item) => `${code(item.name)} ${code(`[${(item.shape || []).join(", ")}]`)} (${item.case_kind})`).join("; "),
    row.total_macs_decimal == null ? row.status : `${row.total_macs_decimal} MACs`,
    row.input_logical_payload_bytes == null ? "not assessed" : formatBytes(row.input_logical_payload_bytes),
    row.output_logical_payload_bytes == null ? "not assessed" : formatBytes(row.output_logical_payload_bytes),
    row.peak_live_logical_payload_bytes == null ? row.peak_live_status || "not assessed" : `${formatBytes(row.peak_live_logical_payload_bytes)}; ${row.peak_live_status}`,
    (row.residuals || []).join("; ") || "none",
  ]);
  const classicalSources = Array.isArray(source.classical_model_sources) ? source.classical_model_sources : [];
  return [
    "## Core ML Serialized Graph And Numerical Evidence (OBSERVED + SOURCE_PINNED)",
    "",
    markdownTable(["Field", "Value"], [
      ["Model representation", `${coreml.model_type || "not decoded"}; Core ML specification ${coreml.specification_version ?? "unknown"}`],
      ["Declared Core ML load floor", deploymentFloor.declared_load_floor ? `${coreMlFloorLabel(deploymentFloor.declared_load_floor)}; ${deploymentFloor.status}; ${deploymentFloor.evidence_class}` : `${deploymentFloor.status || "not assessed"}; pinned table does not map this specification`],
      ["Observed-feature floor", deploymentFloor.observed_feature_minimum_specification_version == null ? "not assessed" : `specification ${deploymentFloor.observed_feature_minimum_specification_version}; ${deploymentFloor.observed_feature_floor ? coreMlFloorLabel(deploymentFloor.observed_feature_floor) : "OS mapping unavailable"}; declared specification remains the loader floor`],
      ["Deployment-floor method", deploymentFloor.method || "not emitted"],
      ["Predicted feature binding", coreml.description?.predicted_feature_name ? code(coreml.description.predicted_feature_name) : "not declared"],
      ["Predicted probabilities binding", coreml.description?.predicted_probabilities_name ? code(coreml.description.predicted_probabilities_name) : "not declared"],
      ["Serialized execution graph", ops.length ? `${formatNumber(ops.length)} ${graphUnit} / ${formatNumber(analysis.tensor_count || 0)} typed tensor/blob value(s)` : "not decoded for this model representation"],
      ["Serialized numerical storage", status.summary || "not assessed"],
      ["Legacy WeightParams bytes", status.weight_parameter_bytes == null ? "not applicable to this representation" : `${formatBytes(status.weight_parameter_bytes)} (${formatNumber(status.weight_parameter_bytes)} B); quantized ${formatNumber(status.quantized_weight_parameter_bytes || 0)} B / FP32 ${formatNumber(status.fp32_weight_parameter_bytes || 0)} B / FP16 ${formatNumber(status.fp16_weight_parameter_bytes || 0)} B`],
      ["Legacy quantization granularity", status.quantized_weight_parameter_count == null ? "not applicable to this representation" : `${formatNumber(status.per_axis_quantized_weight_parameter_count || 0)} per-axis linear / ${formatNumber(Math.max(0, Number(status.quantized_weight_parameter_count || 0) - Number(status.per_axis_quantized_weight_parameter_count || 0)))} single-scale or LUT among ${formatNumber(status.quantized_weight_parameter_count || 0)} quantized WeightParams`],
      ["Legacy WeightParams field coverage", status.layer_count == null ? "not applicable to this representation" : `${formatNumber(status.scanned_layer_count || 0)}/${formatNumber(status.layer_count)} layers; ${status.assessment_status || "not assessed"}`],
      ["Numerical parameter coverage", integrity.parameter_count == null ? "not assessed" : `${formatNumber(integrity.assessed_parameter_count || 0)}/${formatNumber(integrity.parameter_count)} parameters; ${formatNumber(integrity.assessed_payload_bytes || 0)}/${integrity.payload_bytes == null ? "unbound" : formatNumber(integrity.payload_bytes)} B; ${integrity.status || "not assessed"}`],
      ["Source-backed MAC coverage", macs.compute_ops == null ? "not assessed" : `${formatNumber(macs.assessed_compute_ops || 0)}/${formatNumber(macs.compute_ops)} MAC-bearing layers; ${macs.complete_macs_decimal == null ? `${macs.assessed_macs_decimal || "0"} assessed subtotal; complete model total not established` : `${macs.complete_macs_decimal} MACs complete`}; ${macs.safe_number_mirror_status || "mirror status not emitted"}`],
      ...(milScopes ? [["MIL intrinsic-scope coverage", `${formatNumber(milScopes.scope_count || 0)} scope(s), ${formatNumber(milScopes.nested_scope_count || 0)} nested; ${milScopes.status}; global total ${milScopes.global_execution_total_status}; scope-local liveness ${formatNumber(milScopes.scope_liveness_assessed_count || 0)} assessed / ${formatNumber(milScopes.scope_liveness_partial_count || 0)} partial / ${formatNumber(milScopes.scope_liveness_unassessed_count || 0)} unassessed`]] : []),
      ...(milCompression ? [["MIL serialized compression contracts", `${formatNumber(milCompression.exact_contract_count || 0)}/${formatNumber(milCompression.transform_count || 0)} exact; ${formatNumber(milCompression.partial_contract_count || 0)} explicit residual; ${milCompression.status}`]] : []),
      ...(flexibleScenarios ? [["Flexible input scenarios", `${formatNumber(flexibleScenarios.assessed_scenario_count || 0)}/${formatNumber(flexibleScenarios.scenario_count || 0)} evaluated exactly; ${flexibleScenarios.status}; unbounded range ${flexibleScenarios.has_unbounded_range ? "present" : "absent"}`]] : []),
      ["Constant/file byte conservation", size.constant_bytes == null ? `${size.status || "partial"}; external package constant bytes are not fully bound` : `${formatBytes(size.constant_bytes)} exact constant payload + ${formatBytes(size.structure_overhead_bytes)} protobuf/package remainder = ${formatBytes(size.file_size)} selected artifact bytes`],
      ["Peak live logical activation payload", live.peak_bytes == null ? `${live.status || "not assessed"}; ${formatNumber(live.unassessed_tensor_count || 0)} tensor(s) unresolved` : `${formatBytes(live.peak_bytes)} at #${formatNumber(live.peak_at_op || 0)} ${live.peak_at_op_name || ""}; ${live.status || "assessed"}`],
      ["Serialized preprocessing", preprocessing.length
        ? preprocessing.map((item) => item.kind === "image_scaler"
          ? `${code(item.feature_name)} image_scaler (${formatNumber(item.serialized_field_count || 0)} explicitly serialized scalar field(s))`
          : `${code(item.feature_name)} mean_image (${formatNumber(item.value_count || 0)} FP32 values, ${formatNumber(item.byte_length || 0)} B, range ${item.value_min} to ${item.value_max})`).join("; ")
        : "none serialized; preprocessing outside the artifact is not inferred"],
      ["Pinned NeuralNetwork.proto", coreml.neural_network ? source.neural_network_proto_sha256 ? `${source.repository || "apple/coremltools"}@${source.source_commit || "unbound"}; SHA-256 ${code(source.neural_network_proto_sha256)}` : "not bound" : "not applicable to this representation"],
      ["Pinned OS availability / MLComputePlan source", source.coremltools_init_sha256 && source.deployment_compatibility_sha256 && source.compute_plan_sha256 ? `${source.coremltools_init} SHA-256 ${code(source.coremltools_init_sha256)}; ${source.deployment_compatibility} SHA-256 ${code(source.deployment_compatibility_sha256)}; ${source.compute_plan} SHA-256 ${code(source.compute_plan_sha256)}` : "not bound"],
      ["Pinned legacy quantization implementation", source.quantization_implementation_sha256 ? `${source.quantization_implementation}; ${source.repository || "apple/coremltools"}@${source.source_commit || "unbound"}; SHA-256 ${code(source.quantization_implementation_sha256)}` : "not bound"],
      ...(coreml.model_type === "mlProgram" ? [["Pinned MIL.proto", source.mil_proto_sha256 ? `${source.repository || "apple/coremltools"}@${source.source_commit || "unbound"}; SHA-256 ${code(source.mil_proto_sha256)}` : "not bound"]] : []),
      ...(coreml.model_type === "mlProgram" ? [["Pinned MIL conv/linear/matmul definitions", source.mil_conv_definition_sha256 && source.mil_linear_definition_sha256 ? `${source.mil_conv_definition} SHA-256 ${code(source.mil_conv_definition_sha256)}; ${source.mil_linear_definition} SHA-256 ${code(source.mil_linear_definition_sha256)}` : "not bound"]] : []),
      ...(coreml.model_type === "mlProgram" ? [["Pinned MIL compression definitions", source.mil_compression_ios18_definition_sha256 && source.mil_constexpr_ios16_definition_sha256 ? `${source.mil_compression_ios18_definition} SHA-256 ${code(source.mil_compression_ios18_definition_sha256)}; ${source.mil_constexpr_ios16_definition} SHA-256 ${code(source.mil_constexpr_ios16_definition_sha256)}` : "not bound"]] : []),
      ...((classical || pipeline) ? [["Pinned classical/pipeline sources", classicalSources.length ? `${formatNumber(classicalSources.length)} protobuf/validator file(s); ${source.repository || "apple/coremltools"}@${source.source_commit || "unbound"}; every content SHA-256 retained in structured evidence` : "not bound"]] : []),
      ["Execution boundary", "Serialized layer/WeightParams and source-backed shape/MAC evidence only; CPU/GPU/ANE assignment, activation execution precision, fused runtime lowering, and latency are not inferred without bound runtime evidence"],
    ]),
    ...(classicalRows.length ? ["", "### Classical Model Contract", markdownTable(["Field", "Value"], classicalRows)] : []),
    ...(pipelineRows.length ? ["", "### Pipeline Stage Contract", markdownTable(["Stage", "Name", "Model type", "Graph status", "Operations"], pipelineRows.map((row) => [
      `#${formatNumber(row.index)}`, code(row.name), row.model_type, row.graph_status, row.op_count == null ? "not decoded" : formatNumber(row.op_count),
    ]))] : []),
    ...(milScopeRows.length ? ["", "### MIL One-Invocation Scope Ledger", markdownTable(["Class", "Scope", "Status", "Ops", "MAC coverage", "Nominal MACs", "Logical output payload", "Scope-local liveness"], milScopeRows), `> ${milScopes.method}`, `> ${milScopes.interpretation_boundary}`] : []),
    ...(compressionRows.length ? ["", "### Core ML Serialized Compression Contracts", markdownTable(["Operation", "Transform", "Representation", "Assessment", "Logical output elements", "Serialized contract", "Pinned rule"], compressionRows), `> ${milCompression.boundary}`] : []),
    ...(flexibleScenarioRows.length ? ["", "### Core ML Flexible Input Scenarios", markdownTable(["Case", "Kind", "Input shapes", "Nominal MACs", "Input payload", "Output payload", "Peak live payload", "Residual"], flexibleScenarioRows), `> ${flexibleScenarios.range_interpretation}`] : []),
    ...(layerRows.length ? [
      "",
      markdownTable(["Operation", "Type / name", "Inputs / outputs", "Output shape(s) / contract", "MAC assessment", "Parameters / bytes", "Stored encodings", "Quant state", "Numerical scan"], layerRows),
      ...(ops.length > layerLimit ? ["", `Layer table truncated explicitly: ${formatNumber(layerLimit)}/${formatNumber(ops.length)} rows shown; complete rows remain in engineering_evidence.json.`] : []),
    ] : []),
  ].join("\n");
}

export function coreMlStaticResourceMarkdown(analysis) {
  if (String(analysis?.format || "").toLowerCase() !== "coreml") return "";
  const size = analysis.size_breakdown || {};
  const live = analysis.tensor_liveness || {};
  return [
    "## Core ML Static Resource Accounting (DERIVED)",
    "",
    markdownTable(["Metric", "Value", "Boundary"], [
      ["Selected model/package bytes", size.file_size == null ? "not assessed" : `${formatBytes(size.file_size)} (${formatNumber(size.file_size)} B)`, "OBSERVED selected byte total"],
      ["Decoded constant payload", size.constant_bytes == null ? "not fully bound" : `${formatBytes(size.constant_bytes)} (${formatNumber(size.constant_bytes)} B)`, `${formatNumber(size.constant_tensor_count || 0)} decoded parameter/blob record(s); ${size.status || "not assessed"}`],
      ["Container and non-constant remainder", size.structure_overhead_bytes == null ? "not assessed" : `${formatBytes(size.structure_overhead_bytes)} (${formatNumber(size.structure_overhead_bytes)} B)`, "Protobuf/package structure and metadata are not separately attributable"],
      ["Static live activation payload", live.peak_bytes == null ? "not assessed" : `${formatBytes(live.peak_bytes)} (${formatNumber(live.peak_bytes)} B)`, `${live.peak_bytes_status || live.status || "not assessed"}; ${formatNumber(live.assessed_tensor_count || 0)} assessed / ${formatNumber(live.unassessed_tensor_count || 0)} unresolved dense tensor(s)`],
      ["Peak program point", live.peak_at_op == null ? "not assessed" : `#${formatNumber(live.peak_at_op)} ${live.peak_at_op_name || ""}`, "Producer-to-last-consumer graph sweep; not a Core ML runtime allocation trace"],
    ]),
    "",
    `> ${live.method || "Runtime allocator alignment, backend-private buffers, scratch, fusion, and device placement require measured runtime evidence."}`,
  ].join("\n");
}
