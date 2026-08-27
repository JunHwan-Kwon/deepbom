import { renderLlmDeviceFeasibilityView } from "./llm-device-feasibility-view.js";

function node(doc, tag, className, text = null) {
  const element = doc.createElement(tag);
  if (className) element.className = className;
  if (text != null) element.textContent = text;
  return element;
}

function exactText(value) {
  const decimal = value?.decimal;
  if (typeof decimal !== "string" || !/^\d+$/.test(decimal)) return "not derived";
  try { return BigInt(decimal).toLocaleString("en-US"); } catch { return decimal; }
}

function byteText(value) {
  const decimal = value?.decimal;
  if (typeof decimal !== "string" || !/^\d+$/.test(decimal)) return "not derived";
  const bytes = BigInt(decimal);
  const units = [["TiB", 1024n ** 4n], ["GiB", 1024n ** 3n], ["MiB", 1024n ** 2n], ["KiB", 1024n]];
  const unit = units.find(([, size]) => bytes >= size);
  if (!unit) return `${bytes.toLocaleString("en-US")} B`;
  const scaled = bytes * 100n / unit[1];
  return `${bytes.toLocaleString("en-US")} B (${scaled / 100n}.${String(scaled % 100n).padStart(2, "0")} ${unit[0]})`;
}

function valueText(value) {
  return value === null || value === undefined || value === "" ? "not declared" : String(value);
}

function metricBand(doc, title, rows, tone = "neutral") {
  const section = node(doc, "section", `llm-contract-band tone-${tone}`);
  section.append(node(doc, "h4", "", title));
  const ledger = node(doc, "dl", "llm-contract-ledger");
  for (const [label, value] of rows) {
    ledger.append(node(doc, "dt", "", label), node(doc, "dd", "", valueText(value)));
  }
  section.append(ledger);
  return section;
}

function table(doc, headers, rows, className = "") {
  const scroll = node(doc, "div", "llm-table-scroll");
  const element = node(doc, "table", `llm-contract-table ${className}`.trim());
  const head = node(doc, "thead");
  const headRow = node(doc, "tr");
  for (const header of headers) headRow.append(node(doc, "th", "", header));
  head.append(headRow);
  const body = node(doc, "tbody");
  for (const row of rows) {
    const tr = node(doc, "tr");
    for (const value of row) tr.append(node(doc, "td", "", valueText(value)));
    body.append(tr);
  }
  element.append(head, body);
  scroll.append(element);
  return scroll;
}

export function renderOnDeviceLlmView(container, analysis = {}) {
  if (!container) return;
  const doc = container.ownerDocument;
  container.replaceChildren();
  const contract = analysis?.on_device_llm;
  if (!contract || contract.schema !== "deepbom.on_device_llm_contract.v2") {
    container.append(node(doc, "p", "llm-empty", "No on-device LLM contract is available for this artifact."));
    return;
  }

  const architecture = contract.architecture || {};
  const tokenizer = contract.tokenizer || {};
  const state = contract.state || {};
  const compute = contract.compute?.projection || {};
  const runtime = contract.runtime_contract || {};
  const memory = contract.memory_feasibility || {};
  const placement = contract.static_memory_placement || {};
  const medical = contract.medical_ai_claim_boundary || {};
  const tensorRtLlm = contract.tensorrt_llm || {};
  const layerStorage = contract.storage?.layer_storage || {};
  const head = node(doc, "div", "llm-contract-head");
  const title = node(doc, "div");
  title.append(node(doc, "span", "llm-eyebrow", "ON-DEVICE LLM EVIDENCE"), node(doc, "h3", "", contract.serialized_graph
    ? "Serialized transformer evidence and deployment boundary" : "Architecture, state, and deployment contract"));
  const status = node(doc, "div", `llm-status ${contract.status === "invalid" ? "risk" : contract.status?.startsWith("assessed") ? "assessed" : "partial"}`);
  status.append(node(doc, "strong", "", contract.status || "not assessed"), node(doc, "span", "", contract.evidence_class || "NOT_ASSESSABLE"));
  head.append(title, status);
  container.append(head, renderLlmDeviceFeasibilityView(doc, memory, runtime));

  const grid = node(doc, "div", "llm-contract-grid");
  grid.append(
    metricBand(doc, "Serialized architecture", [
      ["Family", architecture.family],
      ["Architecture kind", architecture.kind],
      ["Context / vocabulary", `${valueText(architecture.context_length)} / ${valueText(architecture.vocabulary_size)}`],
      ["Hidden / FFN / layers", `${valueText(architecture.hidden_size)} / ${valueText(architecture.intermediate_size)} / ${valueText(architecture.layer_count)}`],
      ["Attention / KV / width", `${valueText(architecture.attention_head_count)} / ${valueText(architecture.kv_head_count)} / ${valueText(architecture.head_width)}`],
      ["GQA ratio", architecture.gqa_query_heads_per_kv_head],
      ["MoE total / active experts", architecture.moe ? `${valueText(architecture.moe.expert_count)} / ${valueText(architecture.moe.active_expert_count_per_token)}` : "not applicable"],
      ["SSM state / conv / rank", architecture.ssm ? `${valueText(architecture.ssm.state_size)} / ${valueText(architecture.ssm.convolution_kernel)} / ${valueText(architecture.ssm.time_step_rank)}` : "not applicable"],
    ], "assessed"),
    metricBand(doc, "Tensor storage", [
      ["Parameters", contract.storage?.serialized_parameter_count_decimal],
      ["Tensor bytes", contract.storage?.serialized_tensor_bytes_decimal],
      ["Effective bits / parameter", contract.storage?.effective_bits_per_parameter],
      ["Encoding families", contract.storage?.encoding_inventory?.length || 0],
      ["Encoding signature", contract.storage?.encoding_signature_schema],
      ["Encoding inventory SHA-256", contract.storage?.encoding_inventory_sha256],
      ["Tensor assignment SHA-256", contract.storage?.tensor_encoding_assignment_sha256],
      ["Layer storage", `${valueText(layerStorage.status)}; ${valueText(layerStorage.observed_layer_count)}/${valueText(layerStorage.expected_layer_count)}`],
      ["Layer / non-layer bytes", `${byteText(layerStorage.layer_bytes)} / ${byteText(layerStorage.non_layer_bytes)}`],
      ["Artifact role", contract.artifact_role],
    ], "assessed"),
    metricBand(doc, "Tokenizer and generation", [
      ["Tokenizer", tokenizer.status],
      ["Vocabulary", tokenizer.vocabulary_count ?? architecture.vocabulary_size],
      ["Chat template", tokenizer.chat_template?.status],
      ["Bound files", tokenizer.definition_files?.length || 0],
      ["Generation policy", contract.generation?.status],
    ], tokenizer.status === "invalid" ? "risk" : tokenizer.status === "assessed" ? "assessed" : "partial"),
    metricBand(doc, "Runtime binding", [
      ["Status", runtime.status],
      ["Engine / build", runtime.runtime ? `${valueText(runtime.runtime.engine)} / ${valueText(runtime.runtime.version)}` : "external evidence required"],
      ["Weight CPU / accelerator", runtime.weight_residency ? `${exactText(runtime.weight_residency.cpu_bytes)} / ${exactText(runtime.weight_residency.accelerator_bytes)}` : "external evidence required"],
      ["Layer CPU / accelerator", runtime.layer_placement ? `${valueText(runtime.layer_placement.cpu_layer_count)} / ${valueText(runtime.layer_placement.accelerator_layer_count)}` : "external evidence required"],
      ["State resident / allocated", runtime.state_cache ? `${exactText(runtime.state_cache.resident_bytes)} / ${exactText(runtime.state_cache.allocated_bytes)}` : "external evidence required"],
      ["Medical declaration", medical.status],
      ["Declaration coverage", medical.declaration?.coverage ? `${medical.declaration.coverage.declared}/${medical.declaration.coverage.required}` : "0/9"],
    ], "partial"),
    metricBand(doc, "Memory feasibility boundary", [
      ["Status", memory.status],
      ["Serialized weight floor", byteText(memory.serialized_weight_floor_bytes)],
      ["Conditional resident-set range", `${byteText(memory.minimum_static_lower_bound_bytes)} to ${byteText(memory.maximum_static_lower_bound_bytes)}`],
      ["Conditional scenarios", memory.static_scenarios?.length || 0],
      ["Capacity scope", memory.capacity_scope],
      ["Runtime primary residency", memory.runtime_primary_residency?.status || "not artifact-bound"],
      ["Fit claim", memory.fit_claim === "not_emitted" ? "not emitted" : memory.fit_claim],
    ], memory.status?.startsWith("assessed") ? "assessed" : "partial"),
  );
  container.append(grid);

  const graph = contract.serialized_graph;
  if (graph) {
    container.append(node(doc, "h4", "llm-section-title", "Serialized graph evidence"));
    container.append(metricBand(doc, "Transformer and state signals", [
      ["Assessment", graph.status],
      ["Explicit transformer operators", `${graph.explicit_operator_count}/${graph.graph_op_count}`],
      ["MatMul / Softmax / Norm / Gather", `${graph.primitive_counts?.matrix_multiply || 0} / ${graph.primitive_counts?.softmax || 0} / ${graph.primitive_counts?.normalization || 0} / ${graph.primitive_counts?.embedding_gather || 0}`],
      ["Transformer-like motif", graph.transformer_motif_candidate ? "candidate; not architecture proof" : "not detected"],
      ["External state-name candidates", graph.external_state_candidate_count],
      ["Graph signature SHA-256", graph.graph_signature_sha256],
    ], graph.explicit_operator_count ? "assessed" : graph.transformer_motif_candidate ? "partial" : "neutral"));
    if (graph.explicit_operators?.length) container.append(table(doc, ["Op", "Index", "Domain", "Version"], graph.explicit_operators.map((row) => [
      row.name, row.op_index, row.domain || "default", row.version,
    ])));
    if (graph.external_state_candidates?.length) container.append(table(doc, ["Serialized interface", "Dtype", "Shape", "Static logical bytes", "Evidence"], graph.external_state_candidates.map((row) => [
      row.name, row.dtype, row.shape?.join("x") || "unresolved", byteText(row.logical_bytes_if_static), row.classification,
    ])));
    container.append(node(doc, "p", "llm-boundary-note", graph.interpretation_boundary));
  }

  if (tensorRtLlm.status && tensorRtLlm.status !== "not_selected") {
    container.append(node(doc, "h4", "llm-section-title", "TensorRT-LLM static deployment contract"));
    container.append(metricBand(doc, "Build, parallelism, and trust boundary", [
      ["Status", `${tensorRtLlm.status} / ${tensorRtLlm.evidence_class}`],
      ["Engine config", `${tensorRtLlm.engine_config?.path || "not bound"} / ${tensorRtLlm.engine_config?.sha256 || "digest unavailable"}`],
      ["Model-source binding", tensorRtLlm.artifact_binding?.source_artifact_sha256 || "not bound"],
      ["World / TP / PP / CP", `${valueText(tensorRtLlm.parallelism?.world_size)} / ${valueText(tensorRtLlm.parallelism?.tensor_parallel_size)} / ${valueText(tensorRtLlm.parallelism?.pipeline_parallel_size)} / ${valueText(tensorRtLlm.parallelism?.context_parallel_size)}`],
      ["Layers per PP rank", tensorRtLlm.parallelism?.layer_partition_per_pipeline_rank?.join(" / ") || "not derived"],
      ["Max input / sequence / batch", `${valueText(tensorRtLlm.build_limits?.max_input_length)} / ${valueText(tensorRtLlm.build_limits?.max_sequence_length)} / ${valueText(tensorRtLlm.build_limits?.maximum_batch_size)}`],
      ["Weight / KV quantization", `${valueText(tensorRtLlm.quantization?.weight_activation_algorithm)} / ${valueText(tensorRtLlm.quantization?.kv_cache_algorithm)}`],
      ["Weight streaming", tensorRtLlm.build_limits?.weight_streaming],
      ["Conditional logical KV state", byteText(tensorRtLlm.kv_cache_scenario?.logical_bytes)],
      ["Per-rank weight bytes", "not inferred without per-rank engine/build evidence"],
    ], tensorRtLlm.status === "artifact_bound_configuration" ? "assessed" : tensorRtLlm.status === "invalid" ? "risk" : "partial"));
    container.append(node(doc, "p", "llm-boundary-note", tensorRtLlm.interpretation_boundary));
  }

  const encodings = contract.storage?.encoding_inventory || [];
  if (encodings.length) {
    container.append(node(doc, "h4", "llm-section-title", "Serialized precision and encoding inventory"));
    container.append(table(doc, ["Encoding", "Tensors", "Elements", "Serialized bytes", "Effective bits / element"], encodings.map((row) => [
      row.dtype,
      row.tensor_count,
      row.element_count_decimal,
      row.byte_length_decimal,
      row.effective_bits_per_element ?? "not assessable",
    ]), "llm-encoding-table"));
    container.append(node(doc, "p", "llm-boundary-note", contract.storage?.recipe_interpretation));
  }

  if (layerStorage.layers?.length) {
    container.append(node(doc, "h4", "llm-section-title", "Serialized layer storage ledger"));
    container.append(table(doc, ["Layer", "Tensors", "Exact serialized bytes"], layerStorage.layers.map((row) => [
      row.layer_index, row.tensor_count, byteText(row.serialized_bytes),
    ]), "llm-layer-storage-table"));
    container.append(node(doc, "p", "llm-boundary-note", `${layerStorage.conservation?.status === "pass" ? "Layer and non-layer bytes conserve exactly against the serialized tensor ledger. " : "Serialized layer conservation failed. "}${layerStorage.boundary || "Runtime packing and residency remain external."}`));
  }

  if (placement.status && placement.status !== "not_bound") {
    container.append(node(doc, "h4", "llm-section-title", "Conditional CPU / accelerator memory placement"));
    container.append(metricBand(doc, "Static placement profile", [
      ["Assessment", placement.status],
      ["Candidate policies evaluated", valueText(placement.candidate_count)],
      ["Lower-bound candidates not disproven", valueText(placement.lower_bound_not_exceeding_candidate_count)],
      ["Accelerator layers not disproven", placement.minimum_accelerator_layer_count_not_disproven == null
        ? "none" : `${placement.minimum_accelerator_layer_count_not_disproven}-${placement.maximum_accelerator_layer_count_not_disproven}`],
      ["Logical state", `${valueText(placement.logical_state?.kind)}; ${byteText(placement.logical_state?.bytes)}`],
    ], placement.lower_bound_not_exceeding_candidate_count ? "assessed" : "risk"));
    container.append(table(doc, ["Accelerator layers", "CPU layer bytes", "Accelerator layer bytes", "CPU lower bound", "Accelerator lower bound", "Assessment"], (placement.candidates || []).map((row) => [
      row.accelerator_layer_count,
      byteText(row.cpu_layer_serialized_bytes),
      byteText(row.accelerator_layer_serialized_bytes),
      byteText(row.cpu_accounted_lower_bound_bytes),
      byteText(row.accelerator_accounted_lower_bound_bytes),
      row.status,
    ]), "llm-memory-placement-table"));
    container.append(node(doc, "p", "llm-boundary-note", placement.boundary));
  }

  const scenarios = state.scenario_matrix || [];
  if (scenarios.length) {
    const memoryByScenario = new Map((memory.static_scenarios || []).map((row) => [
      `${row.state_kind}:${row.context_length ?? "none"}:${row.batch_size}:${row.storage_bits}`,
      row,
    ]));
    container.append(node(doc, "h4", "llm-section-title", state.kv_projection && state.recurrent_projection
      ? "Hybrid KV + SSM state and memory lower-bound scenarios"
      : state.recurrent_projection ? "Recurrent-state and memory lower-bound scenarios" : "KV-cache and memory lower-bound scenarios"));
    container.append(table(doc, ["State", "Context", "Batch", "Storage", "Logical state", "Conditional resident-set lower bound", "First tier not exceeded", "Lower bound exceeds tiers"], scenarios.map((row) => {
      const feasibility = memoryByScenario.get(`${row.state_kind}:${row.context_length ?? "none"}:${row.batch_size}:${row.storage_bits}`) || {};
      return [
      row.state_kind === "hybrid_kv_ssm" ? "Hybrid KV + SSM" : row.state_kind === "ssm_recurrent" ? "SSM recurrent" : "Transformer KV",
      row.context_length == null ? "context-independent" : row.context_length,
      row.batch_size,
      `${row.storage_bits}-bit`,
      byteText(row.logical_bytes),
      byteText(feasibility.static_lower_bound_bytes),
      feasibility.first_capacity_not_exceeded,
      feasibility.lower_bound_exceeded_capacity_count,
      ];
    }), "llm-memory-table"));
    container.append(node(doc, "p", "llm-boundary-note", `Residency assumption: ${memory.residency_assumption}`));
    container.append(node(doc, "p", "llm-boundary-note", state.scenario_boundary));
    container.append(node(doc, "p", "llm-boundary-note", memory.boundary));
  } else if (memory.static_scenarios?.length) {
    const row = memory.static_scenarios[0];
    container.append(node(doc, "h4", "llm-section-title", "Serialized-weight memory floor"));
    container.append(table(doc, ["Conditional weight floor", "First tier not exceeded", "Lower bound exceeds tiers", "Assessment"], [[
      byteText(row.static_lower_bound_bytes),
      row.first_capacity_not_exceeded,
      row.lower_bound_exceeded_capacity_count,
      "State contract unbound; no fit claim",
    ]]));
    container.append(node(doc, "p", "llm-boundary-note", memory.boundary));
  }

  if (memory.runtime_primary_residency) {
    const row = memory.runtime_primary_residency;
    container.append(node(doc, "h4", "llm-section-title", "Artifact-bound primary residency"));
    container.append(table(doc, ["Evidence", "Context / batch / state", "Resident lower bound", "Allocated lower bound", "Accounted working memory", "Accounted allocation", "Assessment"], [[
      row.evidence_class,
      `${valueText(row.context_length)} / ${valueText(row.batch_size)} / ${valueText(row.state_storage_bits)}-bit`,
      byteText(row.primary_resident_lower_bound_bytes),
      byteText(row.primary_allocated_lower_bound_bytes),
      row.working_memory_accounted_bytes ? byteText(row.working_memory_accounted_bytes) : "not bound",
      row.primary_allocated_accounted_bytes ? byteText(row.primary_allocated_accounted_bytes) : "not assessable",
      row.allocation_accounting_status,
    ]]));
    container.append(node(doc, "p", "llm-boundary-note", row.boundary));
  }

  if (compute.status) {
    container.append(node(doc, "h4", "llm-section-title", "Source-backed architecture scenario"));
    const computeRows = architecture.kind === "sparse_moe_decoder" ? [
      ["Total expert matrix parameters", exactText(compute.total_expert_matrix_parameters_all_layers)],
      ["Active expert matrix MACs / layer / token", exactText(compute.active_expert_matrix_macs_per_layer_per_token)],
      ["Active projections, all layers / token", exactText(compute.active_projection_macs_all_layers_per_token)],
      ["Active decode core at declared context", exactText(compute.decode_active_core_macs_at_declared_context)],
    ] : architecture.kind === "hybrid_attention_ssm_moe" ? [
      ["Attention / Mamba layers", `${valueText(compute.attention_layer_count)} / ${valueText(compute.mamba_layer_count)}`],
      ["Accounted projections, all layers / token", exactText(compute.accounted_projection_macs_all_layers_per_token)],
      ["Active MoE, all expert layers / token", exactText(compute.active_moe_macs_all_expert_layers_per_token)],
      ["Hybrid decode core at declared context", exactText(compute.decode_accounted_core_macs_at_declared_context)],
      ["Selective-scan arithmetic", "excluded, not estimated"],
    ] : architecture.kind === "ssm_recurrent" ? [
      ["Accounted projections + depthwise conv / layer / token", exactText(compute.accounted_macs_per_layer_per_token)],
      ["Accounted MACs, all layers / token", exactText(compute.accounted_macs_all_layers_per_token)],
      ["Selective-scan arithmetic", "excluded, not estimated"],
    ] : [
      ["Dense projections, all layers / token", exactText(compute.dense_projection_macs_all_layers_per_token)],
      ["Prefill core at declared context", exactText(compute.prefill_transformer_core_macs_at_declared_context)],
      ["Decode core at declared context", exactText(compute.decode_transformer_core_macs_at_declared_context)],
      ["Decode plus one logits position", exactText(compute.decode_with_one_logit_position_macs)],
    ];
    container.append(table(doc, ["Scenario", "Exact value"], computeRows));
    container.append(node(doc, "p", "llm-boundary-note", contract.compute?.boundary));
  }

  const boundaries = node(doc, "div", "llm-boundary-grid");
  const runtimeBound = String(runtime.status || "").startsWith("artifact_bound_");
  boundaries.append(
    metricBand(doc, runtimeBound ? "Runtime manifest coverage" : "Runtime evidence still required", (runtime.required_bindings || []).map((value) => [value.replaceAll("_", " "), runtimeBound ? "bound and validated" : "unbound"]), runtimeBound ? "assessed" : "partial"),
    metricBand(doc, "Medical AI evidence still required", (medical.required_external_evidence || []).length
      ? medical.required_external_evidence.map((value) => [value.replaceAll("_", " "), "external"])
      : [["Declaration fields", "complete but unverified"], ["Clinical validation", "still external"]], "risk"),
  );
  container.append(boundaries);
  const notEstablished = (medical.not_established || []).join(" / ");
  container.append(node(doc, "p", "llm-claim-boundary", `Not established by this artifact: ${notEstablished || "task accuracy, clinical validity, runtime assignment, or release readiness"}.`));
}
