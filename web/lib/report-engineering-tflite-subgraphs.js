import { formatNumber, padOp } from "./format.js";
import { code, markdownTable } from "./report-utils.js";

export function tfliteSubgraphInventoryMarkdown(analysis) {
  if (String(analysis?.format || "").toLowerCase() === "onnx") return "";
  const inventory = analysis?.tflite_subgraph_inventory || {};
  const rows = Array.isArray(inventory.rows) ? inventory.rows : [];
  const references = Array.isArray(inventory.references) ? inventory.references : [];
  const contracts = Array.isArray(inventory.control_flow_contracts) ? inventory.control_flow_contracts : [];
  const sources = Array.isArray(inventory.control_flow_sources) ? inventory.control_flow_sources : [];
  const nominalMacSources = Array.isArray(inventory.nominal_mac_sources) ? inventory.nominal_mac_sources : [];
  const renderedRows = rows.slice(0, 64);
  const renderedReferences = references.slice(0, 128);
  const renderedContracts = contracts.slice(0, 128);
  const intrinsicValue = (value, assessed) => value == null
    ? `PARTIAL; assessed subtotal ${formatNumber(assessed || 0)} B`
    : `${formatNumber(value)} B`;
  const renderedIntrinsicRows = renderedRows.map((row) => {
    const cost = row.intrinsic_cost || {};
    const exactMirror = cost.assessed_nominal_macs == null ? "decimal-only" : formatNumber(cost.assessed_nominal_macs);
    const modeledMirror = cost.modeled_scenario_macs == null ? "decimal-only" : formatNumber(cost.modeled_scenario_macs);
    const completeMirror = cost.complete_nominal_macs == null ? "not losslessly emitted" : formatNumber(cost.complete_nominal_macs);
    const macs = cost.complete_nominal_macs_decimal != null
      ? `${cost.complete_nominal_macs_decimal} nominal MACs (${completeMirror}); exact ${cost.assessed_nominal_macs_decimal || "0"} (${exactMirror}); modeled ${cost.modeled_scenario_macs_decimal || "0"} (${modeledMirror})`
      : Number(cost.mac_compute_operator_count || 0) === 0
        ? `not applicable; exact ${cost.assessed_nominal_macs_decimal || "0"} (${exactMirror}); modeled ${cost.modeled_scenario_macs_decimal || "0"} (${modeledMirror})`
        : `PARTIAL; exact ${cost.assessed_nominal_macs_decimal || "0"} (${exactMirror}) + modeled ${cost.modeled_scenario_macs_decimal || "0"} (${modeledMirror}); ${formatNumber(cost.unassessed_mac_operator_count || 0)} residual`;
    return [
      `S${row.subgraph_index}`,
      row.invocation_semantics || "not emitted",
      `${cost.status || "not emitted"}; ${cost.evidence_class || "not emitted"}; ${cost.invocation_basis || "not emitted"}`,
      `${formatNumber(cost.assessed_nominal_mac_operator_count || 0)}/${formatNumber(cost.mac_compute_operator_count || 0)} exact; ${formatNumber(cost.modeled_scenario_mac_operator_count || 0)} modeled; ${formatNumber(cost.unassessed_mac_operator_count || 0)} residual`,
      macs,
      `${intrinsicValue(cost.logical_tensor_payload_bytes, cost.assessed_logical_tensor_payload_bytes)}; ${formatNumber(cost.assessed_tensor_payload_count || 0)} assessed / ${formatNumber(cost.unassessed_tensor_payload_count || 0)} residual tensor(s)`,
      `${intrinsicValue(cost.logical_operator_io_payload_bytes, cost.assessed_logical_operator_io_payload_bytes)}; ${formatNumber(cost.assessed_operator_io_tensor_slot_count || 0)} assessed / ${formatNumber(cost.unassessed_operator_io_tensor_slot_count || 0)} residual slot(s)`,
      `${intrinsicValue(cost.graph_input_payload_bytes, 0)} / ${intrinsicValue(cost.graph_output_payload_bytes, 0)}`,
      `${formatNumber(cost.physical_unique_constant_bytes || 0)} physical B in ${formatNumber(cost.physical_unique_constant_buffer_count || 0)} buffer(s) / ${formatNumber(cost.logical_constant_reference_bytes || 0)} logical-reference B`,
    ];
  });
  const residuals = rows.flatMap((row) => (row.operator_intrinsics || [])
    .filter((operator) => ["modeled_scenario", "not_assessed"].includes(operator.mac_assessment_status)
      || operator.logical_io_payload_status !== "assessed")
    .map((operator) => ({ subgraph: row.subgraph_index, ...operator }))).slice(0, 128);
  const deep = analysis?.tflite_subgraph_deep_analysis || {};
  const deepRows = Array.isArray(deep.rows) ? deep.rows : [];
  const renderedDeepRows = deepRows.slice(0, 64);
  const assessedValue = (contract, field, suffix = "") => contract?.[field] == null
    ? `${contract?.status || "not assessed"}`
    : `${formatNumber(contract[field])}${suffix}`;
  return [
    "## TFLite Subgraph And Control-flow Inventory (OBSERVED/DERIVED)",
    markdownTable(["Field", "Value"], [
      ["Schema / status / evidence", `${inventory.schema || "not emitted"} / ${inventory.status || "not emitted"} / ${inventory.evidence_class || "not emitted"}`],
      ["Parsed subgraphs", `${formatNumber(inventory.parsed_subgraph_count || 0)}/${formatNumber(inventory.subgraph_count || 0)}`],
      ["Primary subgraph", `S${formatNumber(inventory.primary_subgraph_index || 0)}`],
      ["Primary / nested serialized operators", `${formatNumber(inventory.primary_operator_count || 0)} / ${formatNumber(inventory.nested_operator_count || 0)}; ${formatNumber(inventory.serialized_operator_count || 0)} total`],
      ["Primary / nested serialized tensors", `${formatNumber(inventory.primary_tensor_count || 0)} / ${formatNumber(inventory.nested_tensor_count || 0)}; ${formatNumber(inventory.serialized_tensor_count || 0)} total`],
      ["Control-flow/computation references", formatNumber(inventory.control_flow_reference_count || 0)],
      ["Pinned IF/WHILE/CALL_ONCE contracts", `${formatNumber(inventory.assessed_control_flow_contract_count || 0)} assessed + ${formatNumber(inventory.partial_control_flow_contract_count || 0)} partial = ${formatNumber(inventory.control_flow_contract_count || 0)}`],
      ["Signature entrypoints", formatNumber(inventory.signature_entrypoint_count || 0)],
      ["Reachable subgraphs", `${formatNumber(inventory.reachable_subgraph_count || 0)}/${formatNumber(inventory.subgraph_count || 0)}; unreachable ${(inventory.unreachable_subgraph_indices || []).map((index) => `S${index}`).join(", ") || "none"}`],
      ["Per-invocation intrinsic cost", `${rows[0]?.intrinsic_cost?.schema || "not emitted"}; ${formatNumber(rows.filter((row) => String(row?.intrinsic_cost?.status || "").startsWith("assessed")).length)} assessed + ${formatNumber(rows.filter((row) => row?.intrinsic_cost?.status === "partial").length)} partial; never summed across control flow`],
      ["Pinned schema source", `${code(inventory.schema_source || "not emitted")} / ${code(inventory.schema_source_sha256 || "not emitted")}`],
      ["Pinned control-flow Prepare sources", sources.map((source) => `${source.role}: ${code(source.path)} SHA-256 ${code(source.sha256)}`).join(" / ") || "not emitted"],
      ["Pinned nominal-MAC tensor contracts", nominalMacSources.map((source) => `${source.role}: ${code(source.path)} SHA-256 ${code(source.sha256)}`).join(" / ") || "not emitted"],
      ["Method", inventory.method || "not emitted"],
    ]),
    "### Serialized Subgraph Ledger",
    markdownTable(["Subgraph", "Name", "Reachability", "Inputs / outputs", "Ops", "Tensors", "Constants", "Quantized / per-axis / sparse", "Outgoing / incoming refs", "Operator histogram"], renderedRows.map((row) => [
      `S${row.subgraph_index}`, code(row.name || "(unnamed)"), row.reachable_from_entrypoint ? "reachable" : "unreachable",
      `${(row.input_tensor_indices || []).join(", ") || "none"} / ${(row.output_tensor_indices || []).join(", ") || "none"}`,
      formatNumber(row.operator_count || 0), formatNumber(row.tensor_count || 0), formatNumber(row.constant_tensor_count || 0),
      `${formatNumber(row.quantized_tensor_count || 0)} / ${formatNumber(row.per_axis_tensor_count || 0)} / ${formatNumber(row.sparse_tensor_count || 0)}`,
      `${formatNumber(row.control_flow_reference_count || 0)} / ${formatNumber(row.incoming_reference_count || 0)}`,
      (row.operator_histogram || []).map((item) => `${item.name}:${item.count}`).join(" / ") || "none",
    ])),
    rows.length > renderedRows.length ? `> ${formatNumber(rows.length - renderedRows.length)} additional subgraph row(s) remain in machine-readable evidence.` : "",
    "### Per-invocation Intrinsic Cost Ledger",
    markdownTable(["Subgraph", "Invocation semantics", "Status", "MAC coverage", "Nominal MAC value", "Tensor payload", "Operator I/O payload", "Graph input / output", "Constants"], renderedIntrinsicRows),
    residuals.length ? "### Intrinsic Cost Residual Ledger" : "",
    residuals.length ? markdownTable(["Subgraph", "Operator", "MAC status / class", "MAC value", "I/O payload", "Reason"], residuals.map((operator) => [
      `S${operator.subgraph}`, `#${padOp(operator.operator_index)} ${operator.name}`,
      `${operator.mac_assessment_status} / ${operator.mac_formula_class}`,
      operator.nominal_macs_decimal == null ? "not assessed" : `${operator.nominal_macs_decimal} MACs`,
      operator.logical_io_payload_bytes == null ? `PARTIAL; ${formatNumber(operator.assessed_logical_io_payload_bytes || 0)} B assessed` : `${formatNumber(operator.logical_io_payload_bytes)} B`,
      operator.mac_assessment_reason,
    ])) : "All registered MAC-family operators and logical operator I/O payloads are complete in their individual invocation scope.",
    "### Intrinsic Cost Method And Boundary",
    markdownTable(["Subgraph", "Method", "Interpretation boundary"], renderedRows.map((row) => [
      `S${row.subgraph_index}`, row.intrinsic_cost?.method || "not emitted", row.intrinsic_cost?.interpretation_boundary || "not emitted",
    ])),
    renderedReferences.length ? "### Control-flow And Computation References" : "",
    renderedReferences.length ? markdownTable(["Source", "Operator", "Role", "Target"], renderedReferences.map((reference) => [
      `S${reference.source_subgraph_index}`, `#${padOp(reference.source_op_index)} ${reference.source_op_name}`,
      reference.role, `S${reference.target_subgraph_index} ${code(reference.target_subgraph_name || "(unnamed)")}`,
    ])) : "No schema-defined subgraph reference is serialized.",
    references.length > renderedReferences.length ? `> ${formatNumber(references.length - renderedReferences.length)} additional reference row(s) remain in machine-readable evidence.` : "",
    renderedContracts.length ? "### Pinned Control-flow Prepare Contracts" : "",
    renderedContracts.length ? markdownTable(["Source", "Operator", "Status", "Inputs / outputs", "Targets", "Condition", "Method"], renderedContracts.map((contract) => [
      `S${contract.source_subgraph_index}`, `#${padOp(contract.source_op_index)} ${contract.source_op_name}`, contract.status,
      `${formatNumber(contract.source_input_count)} / ${formatNumber(contract.source_output_count)}`,
      (contract.target_subgraph_indices || []).map((index) => `S${index}`).join(", ") || "none",
      contract.condition_contract_status || "not applicable", contract.method,
    ])) : "No IF, WHILE, or CALL_ONCE Prepare-time interface contract is serialized.",
    contracts.length > renderedContracts.length ? `> ${formatNumber(contracts.length - renderedContracts.length)} additional control-flow contract row(s) remain in machine-readable evidence.` : "",
    `> ${inventory.execution_count_boundary || "Nested serialized operators are not execution counts."}`,
    deepRows.length ? "## TFLite Per-subgraph Deep Analysis (OBSERVED/DERIVED/PREDICTED_SOURCE_PINNED)" : "",
    deepRows.length ? markdownTable(["Field", "Value"], [
      ["Schema / status / evidence", `${deep.schema || "not emitted"} / ${deep.status || "not emitted"} / ${deep.evidence_class || "not emitted"}`],
      ["Assessed subgraphs", `${formatNumber(deep.assessed_subgraph_count || 0)}/${formatNumber(deep.subgraph_count || 0)}`],
      ["Primary scope", `S${formatNumber(deep.primary_subgraph_index || 0)}; top-level proof ledgers are referenced without duplication`],
      ["Method", deep.method || "not emitted"],
      ["Execution boundary", deep.execution_count_boundary || "not emitted"],
    ]) : "",
    deepRows.length ? "### Independent Scope Evidence" : "",
    deepRows.length ? markdownTable([
      "Scope", "Status", "Ops / tensors", "Intrinsic MACs", "Quantized / per-axis tensors",
      "Predicted XNNPACK", "Predicted boundaries", "Liveness / arena", "Weight / numerical evidence",
    ], renderedDeepRows.map((row) => {
      const delegated = row.delegate || {};
      const boundaries = row.predicted_partition_boundaries || {};
      const liveness = row.tensor_liveness || {};
      const arena = row.tensor_arena_plan || {};
      const weight = row.weight_integrity || {};
      return [
        `S${row.subgraph_index} ${code(row.name || "(unnamed)")}; ${row.reachable_from_entrypoint ? "reachable" : "unreachable"}; ${row.invocation_semantics || "invocation semantics not emitted"}`,
        `${row.status || "not emitted"}; ${row.evidence_class || "not emitted"}`,
        `${formatNumber(row.operator_count || 0)} / ${formatNumber(row.tensor_count || 0)}`,
        formatNumber(row.total_macs || 0),
        `${formatNumber(row.quantized_tensor_count || 0)} / ${formatNumber(row.per_axis_tensor_count || 0)}; ${row.quantization_status?.status || row.quantization_status?.classification || "classified in raw evidence"}`,
        `${formatNumber(delegated.predicted_delegated_operator_count || 0)} delegated + ${formatNumber(delegated.predicted_fallback_operator_count || 0)} fallback = ${formatNumber(delegated.assessed_operator_count || 0)}; ${formatNumber((delegated.predicted_delegated_mac_ratio || 0) * 100)}% MAC; ${formatNumber(delegated.chain_count || 0)} chain(s)`,
        `${formatNumber(boundaries.edge_count || 0)} boundary edge(s); ${boundaries.summed_edge_payload_bytes == null ? `${boundaries.payload_coverage_status || "payload not assessed"}` : `${formatNumber(boundaries.summed_edge_payload_bytes)} B`}`,
        `${assessedValue(liveness, "peak_bytes", " B")} / ${assessedValue(arena, "combined_arena_bytes", " B")}`,
        `${weight.status || "not emitted"}; ${row.advanced_numerical_storage || "not emitted"}${row.advanced_numerical_evidence_pointers?.length ? ` (${formatNumber(row.advanced_numerical_evidence_pointers.length)} top-level pointers)` : ""}`,
      ];
    })) : "",
    deepRows.length > renderedDeepRows.length ? `> ${formatNumber(deepRows.length - renderedDeepRows.length)} additional deep-scope row(s) remain in machine-readable evidence.` : "",
    deepRows.length ? `> ${deep.execution_count_boundary || "Per-subgraph rows are not cross-scope execution totals."}` : "",
  ].filter(Boolean).join("\n\n");
}
