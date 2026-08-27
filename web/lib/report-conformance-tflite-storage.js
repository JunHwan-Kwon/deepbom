export function registerTfliteStorageConformance({ check, staticAnalysis, engineeringReportText, mlBomDocument }) {
  const sparse = staticAnalysis?.tflite_sparse_storage_contract || {};
  const sparseRows = Array.isArray(sparse.rows) ? sparse.rows : [];
  const sparseSerializedRows = sparseRows.filter((row) => row?.encoding?.stored_value_bytes != null);
  const sparseDecodedRows = sparseRows.filter((row) => row?.encoding?.status === "assessed");
  const sparsePartialRows = sparseRows.filter((row) => String(row?.encoding?.status || "").startsWith("partial_"));
  const sparseSum = (field) => sparseRows.reduce((sum, row) => sum + Number(row?.encoding?.[field] || 0), 0);
  const sparseStatus = sparseRows.length === 0 ? "not_applicable" : sparsePartialRows.length ? "partial" : "assessed";
  check("CF-TFLITE-SPARSE-001", sparse.schema === "deepbom.tflite_sparse_storage_contract.v1"
    && sparse.evidence_class === "DERIVED" && sparse.status === sparseStatus
    && Number(sparse.sparse_tensor_count || 0) === sparseRows.length
    && Number(sparse.serialized_value_tensor_count || 0) === sparseSerializedRows.length
    && Number(sparse.fully_decoded_tensor_count || 0) === sparseDecodedRows.length
    && Number(sparse.partial_tensor_count || 0) === sparsePartialRows.length,
  "TFLite sparse storage status and tensor-count ledger do not reconstruct.", ["/evidence/static_analysis/tflite_sparse_storage_contract"]);
  check("CF-TFLITE-SPARSE-002", sparseRows.every((row) => {
    const encoding = row?.encoding || {};
    const logical = Number(encoding.logical_element_count || 0);
    const stored = Number(encoding.stored_element_count || 0);
    const implicit = Number(encoding.implicit_zero_element_count || 0);
    return Number.isInteger(Number(row.subgraph_index)) && Number(row.subgraph_index) >= 0
      && logical === stored + implicit
      && (encoding.expected_stored_value_bytes == null || Number(encoding.expected_stored_value_bytes) === stored * Number(encoding.storage_width_bytes))
      && (encoding.stored_value_bytes == null || Number(encoding.stored_value_bytes) === Number(encoding.expected_stored_value_bytes))
      && /^[0-9a-f]{64}$/.test(String(encoding.canonical_metadata_sha256 || ""));
  })
    && Number(sparse.logical_element_count || 0) === sparseSum("logical_element_count")
    && Number(sparse.stored_element_count || 0) === sparseSum("stored_element_count")
    && Number(sparse.implicit_zero_element_count || 0) === sparseSum("implicit_zero_element_count")
    && Number(sparse.logical_element_count || 0) === Number(sparse.stored_element_count || 0) + Number(sparse.implicit_zero_element_count || 0)
    && Number(sparse.serialized_value_bytes || 0) === sparseSum("stored_value_bytes"),
  "TFLite sparse logical/stored/implicit element or serialized-byte conservation failed.", ["/evidence/static_analysis/tflite_sparse_storage_contract/rows"]);
  check("CF-TFLITE-SPARSE-003", sparse.source_commit === "87bbf65b8d23d3f06912b1b2183587e1884bc45c"
    && sparse.schema_source_sha256 === "3bfa613428459de18db5d70d8581e7b6afd127c4522bb18ff59c8e589c3b75a1"
    && sparse.converter_source_sha256 === "2c032a4202d549a39c09978b4951d2200014b4429b061235b70f66da148a418c"
    && engineeringReportText.includes("## TFLite Sparse Storage Contract")
    && engineeringReportText.includes(sparse.schema) && engineeringReportText.includes(sparse.schema_source_sha256)
    && engineeringReportText.includes(sparse.converter_source_sha256),
  "TFLite sparse source identity or Engineering Report binding is incomplete.", ["/evidence/static_analysis/tflite_sparse_storage_contract", "/engineering_report.md"]);
  const properties = mlBomDocument?.metadata?.component?.properties || [];
  const propertyValue = (name) => properties.find((item) => item.name === name)?.value;
  check("CF-TFLITE-SPARSE-004", propertyValue("deepbom:model:tfliteSparseStorageSchema") === sparse.schema
    && propertyValue("deepbom:model:tfliteSparseTensorCount") === String(sparse.sparse_tensor_count || 0)
    && propertyValue("deepbom:model:tfliteSparseLogicalElements") === String(sparse.logical_element_count || 0)
    && propertyValue("deepbom:model:tfliteSparseStoredElements") === String(sparse.stored_element_count || 0)
    && propertyValue("deepbom:model:tfliteSparseImplicitZeroElements") === String(sparse.implicit_zero_element_count || 0)
    && propertyValue("deepbom:model:tfliteSparseSerializedValueBytes") === String(sparse.serialized_value_bytes || 0),
  "CycloneDX model component does not preserve the TFLite sparse storage summary.", ["/evidence/static_analysis/tflite_sparse_storage_contract", "/evidence/mlbom_cyclonedx/metadata/component/properties"]);

  const inventory = staticAnalysis?.tflite_subgraph_inventory || {};
  const rows = Array.isArray(inventory.rows) ? inventory.rows : [];
  const references = Array.isArray(inventory.references) ? inventory.references : [];
  const entrypoints = Array.isArray(inventory.signature_entrypoints) ? inventory.signature_entrypoints : [];
  const contracts = Array.isArray(inventory.control_flow_contracts) ? inventory.control_flow_contracts : [];
  const controlSources = Array.isArray(inventory.control_flow_sources) ? inventory.control_flow_sources : [];
  const nominalMacSources = Array.isArray(inventory.nominal_mac_sources) ? inventory.nominal_mac_sources : [];
  const serializedOperators = rows.reduce((sum, row) => sum + Number(row.operator_count || 0), 0);
  const serializedTensors = rows.reduce((sum, row) => sum + Number(row.tensor_count || 0), 0);
  check("CF-TFLITE-SUBGRAPH-001", inventory.schema === "deepbom.tflite_subgraph_inventory.v1.3"
    && inventory.status === "assessed" && inventory.evidence_class === "OBSERVED/SOURCE_PINNED/DERIVED"
    && Number(inventory.subgraph_count || 0) === rows.length && Number(inventory.parsed_subgraph_count || 0) === rows.length
    && Number(inventory.primary_subgraph_index || 0) === 0
    && Number(inventory.serialized_operator_count || 0) === serializedOperators
    && Number(inventory.serialized_tensor_count || 0) === serializedTensors
    && Number(inventory.primary_operator_count || 0) === Number(rows[0]?.operator_count || 0)
    && Number(inventory.primary_tensor_count || 0) === Number(rows[0]?.tensor_count || 0)
    && Number(inventory.nested_operator_count || 0) === serializedOperators - Number(rows[0]?.operator_count || 0)
    && Number(inventory.nested_tensor_count || 0) === serializedTensors - Number(rows[0]?.tensor_count || 0),
  "TFLite all-subgraph operator/tensor inventory does not conserve.", ["/evidence/static_analysis/tflite_subgraph_inventory/rows"]);
  const validLocalTensor = (tensor, row) => Number.isInteger(tensor) && (tensor === -1 || tensor >= 0 && tensor < Number(row.tensor_count || 0));
  check("CF-TFLITE-SUBGRAPH-002", rows.every((row, index) => Number(row.subgraph_index) === index
    && (row.input_tensor_indices || []).every((tensor) => validLocalTensor(tensor, row))
    && (row.output_tensor_indices || []).every((tensor) => validLocalTensor(tensor, row))
    && (row.operator_histogram || []).reduce((sum, item) => sum + Number(item.count || 0), 0) === Number(row.operator_count || 0)
    && Number(row.control_flow_reference_count || 0) === references.filter((reference) => Number(reference.source_subgraph_index) === index).length
    && Number(row.incoming_reference_count || 0) === references.filter((reference) => Number(reference.target_subgraph_index) === index).length)
    && references.every((reference) => {
      const source = rows[Number(reference.source_subgraph_index)];
      const target = rows[Number(reference.target_subgraph_index)];
      return Boolean(source && target) && Number(reference.source_op_index) >= 0
        && Number(reference.source_op_index) < Number(source.operator_count || 0)
        && (source.operator_histogram || []).some((item) => item.name === reference.source_op_name)
        && target.name === reference.target_subgraph_name;
    }) && Number(inventory.control_flow_reference_count || 0) === references.length
    && Number(inventory.signature_entrypoint_count || 0) === entrypoints.length,
  "TFLite subgraph local tensor, operator histogram, incoming/outgoing reference, or SignatureDef ledger is inconsistent.", ["/evidence/static_analysis/tflite_subgraph_inventory"]);
  const reachable = new Set([0, ...entrypoints.map((entry) => Number(entry.subgraph_index))]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const reference of references) {
      if (reachable.has(Number(reference.source_subgraph_index)) && !reachable.has(Number(reference.target_subgraph_index))) {
        reachable.add(Number(reference.target_subgraph_index));
        changed = true;
      }
    }
  }
  const unreachable = rows.map((_, index) => index).filter((index) => !reachable.has(index));
  check("CF-TFLITE-SUBGRAPH-003", Number(inventory.reachable_subgraph_count || 0) === reachable.size
    && JSON.stringify(inventory.unreachable_subgraph_indices || []) === JSON.stringify(unreachable)
    && rows.every((row) => Boolean(row.reachable_from_entrypoint) === reachable.has(Number(row.subgraph_index))),
  "TFLite subgraph entrypoint reachability does not reconstruct from control-flow and SignatureDef references.", ["/evidence/static_analysis/tflite_subgraph_inventory/references", "/evidence/static_analysis/tflite_subgraph_inventory/signature_entrypoints"]);
  const expectedNominalMacSources = new Map([
    ["conv_2d_ohwi_nhwc_contract", "66a2fef9a8e7fe81b7bdd9d18bd099cc589546ac29cca7665711de890fba9281"],
    ["depthwise_conv_2d_1hwo_nhwc_contract", "343f85c01e6adf2b21dbcd7e610ae04acf78f4ba1fea912e2fb02e33c92f6629"],
    ["fully_connected_output_input_contract", "a2667242af7d0d933d31408a0393974718e82da221248db9cb25aac2a8d3c585"],
    ["conv_3d_dhwio_ndhwc_contract", "7dfd75d047b7d22f76c365d48ecb1facad4656897ed3d58a661afcb0ad503b36"],
  ]);
  check("CF-TFLITE-SUBGRAPH-004", inventory.source_commit === "87bbf65b8d23d3f06912b1b2183587e1884bc45c"
    && inventory.schema_source_sha256 === "3bfa613428459de18db5d70d8581e7b6afd127c4522bb18ff59c8e589c3b75a1"
    && nominalMacSources.length === expectedNominalMacSources.size
    && nominalMacSources.every((source) => expectedNominalMacSources.get(source.role) === source.sha256
      && typeof source.path === "string" && source.path.startsWith("tensorflow/lite/kernels/")
      && engineeringReportText.includes(source.path) && engineeringReportText.includes(source.sha256))
    && engineeringReportText.includes("## TFLite Subgraph And Control-flow Inventory")
    && engineeringReportText.includes(inventory.schema) && engineeringReportText.includes(inventory.schema_source_sha256)
    && references.slice(0, 128).every((reference) => engineeringReportText.includes(reference.source_op_name)
      && engineeringReportText.includes(reference.role) && engineeringReportText.includes(reference.target_subgraph_name)),
  "TFLite all-subgraph source identity, nominal-MAC source contract, or Engineering Report reference ledger is incomplete.", ["/evidence/static_analysis/tflite_subgraph_inventory", "/engineering_report.md"]);
  const expectedControlSources = new Map([
    ["if_prepare_contract", "e23c06a2a3984ae704153c667f891be2bbd31d03be523a85cc976ea6ca75a428"],
    ["while_prepare_contract", "6ac2e230d01317af3c39bb2a730c8d1aa632cc707f9043b3d8bb6e2123389005"],
    ["call_once_prepare_contract", "21bf79b3ba4c47bf63090b5cf3b7734591f79ddcecc8b4364b18997ca159a44d"],
    ["control_flow_tensor_propagation", "acce5ae7657930db1eac7a7548c8e74ff91314fa2ad03681d3968125f6665a46"],
  ]);
  const assessedContracts = contracts.filter((row) => row.status === "assessed");
  const partialContracts = contracts.filter((row) => row.status === "partial");
  check("CF-TFLITE-SUBGRAPH-005", Number(inventory.control_flow_contract_count || 0) === contracts.length
    && Number(inventory.assessed_control_flow_contract_count || 0) === assessedContracts.length
    && Number(inventory.partial_control_flow_contract_count || 0) === partialContracts.length
    && assessedContracts.length + partialContracts.length === contracts.length
    && contracts.every((contract) => {
      const source = rows[Number(contract.source_subgraph_index)];
      const sourceReferences = references.filter((reference) => Number(reference.source_subgraph_index) === Number(contract.source_subgraph_index)
        && Number(reference.source_op_index) === Number(contract.source_op_index));
      return Boolean(source) && ["IF", "WHILE", "CALL_ONCE"].includes(contract.source_op_name)
        && (source.operator_histogram || []).some((item) => item.name === contract.source_op_name)
        && (contract.target_subgraph_indices || []).every((target) => rows[Number(target)])
        && JSON.stringify(contract.target_subgraph_indices || []) === JSON.stringify(sourceReferences.map((reference) => Number(reference.target_subgraph_index)))
        && typeof contract.method === "string" && contract.method.length > 0;
    })
    && controlSources.length === expectedControlSources.size
    && controlSources.every((source) => expectedControlSources.get(source.role) === source.sha256
      && typeof source.path === "string" && source.path.startsWith("tensorflow/lite/kernels/")
      && engineeringReportText.includes(source.path) && engineeringReportText.includes(source.sha256))
    && contracts.slice(0, 128).every((contract) => engineeringReportText.includes(contract.method)),
  "TFLite IF/WHILE/CALL_ONCE Prepare-time contract or pinned source ledger is inconsistent.", ["/evidence/static_analysis/tflite_subgraph_inventory/control_flow_contracts", "/evidence/static_analysis/tflite_subgraph_inventory/control_flow_sources", "/engineering_report.md"]);
  const unsignedDecimal = (value) => /^(0|[1-9][0-9]*)$/.test(String(value ?? "")) ? BigInt(value) : null;
  const safeNumberMatchesDecimal = (numberValue, decimalValue) => {
    const parsed = unsignedDecimal(decimalValue);
    if (parsed == null) return false;
    return parsed <= BigInt(Number.MAX_SAFE_INTEGER)
      ? Number(numberValue) === Number(parsed)
      : numberValue == null;
  };
  const payloadFromTensorIndices = (indices, tensors) => {
    let assessedBytes = 0;
    let assessedSlots = 0;
    let unassessedSlots = 0;
    for (const indexValue of indices || []) {
      const index = Number(indexValue);
      if (index < 0) continue;
      const tensor = tensors[index];
      if (!tensor) return null;
      if (tensor.logical_payload_bytes == null) unassessedSlots += 1;
      else {
        assessedBytes += Number(tensor.logical_payload_bytes);
        assessedSlots += 1;
      }
    }
    return {
      assessedBytes,
      assessedSlots,
      unassessedSlots,
      totalBytes: unassessedSlots === 0 ? assessedBytes : null,
    };
  };
  check("CF-TFLITE-SUBGRAPH-007", rows.every((row) => {
    const cost = row.intrinsic_cost || {};
    const tensors = Array.isArray(row.tensor_intrinsics) ? row.tensor_intrinsics : [];
    const operators = Array.isArray(row.operator_intrinsics) ? row.operator_intrinsics : [];
    if (cost.schema !== "deepbom.tflite_subgraph_intrinsic_cost.v1"
      || cost.evidence_class !== "OBSERVED/DERIVED"
      || cost.invocation_basis !== "one_invocation_of_this_serialized_subgraph"
      || typeof cost.method !== "string" || !cost.method.length
      || typeof cost.interpretation_boundary !== "string" || !cost.interpretation_boundary.length
      || typeof row.invocation_semantics !== "string" || !row.invocation_semantics.length
      || tensors.length !== Number(row.tensor_count || 0)
      || operators.length !== Number(row.operator_count || 0)) return false;
    if (!tensors.every((tensor, index) => Number(tensor.tensor_index) === index
      && Array.isArray(tensor.shape) && Array.isArray(tensor.shape_signature)
      && typeof tensor.dtype === "string"
      && ["assessed_static", "assessed_serialized_batch1", "not_assessed"].includes(tensor.payload_status)
      && (tensor.logical_payload_bytes == null) === (tensor.payload_status === "not_assessed")
      && typeof tensor.payload_reason === "string" && tensor.payload_reason.length > 0
      && Number.isSafeInteger(Number(tensor.buffer_data_offset)) && Number(tensor.buffer_data_offset) >= 0
      && Number.isSafeInteger(Number(tensor.buffer_data_length)) && Number(tensor.buffer_data_length) >= 0)) return false;
    const tensorPayload = payloadFromTensorIndices(tensors.map((_, index) => index), tensors);
    const graphInputPayload = payloadFromTensorIndices(row.input_tensor_indices, tensors);
    const graphOutputPayload = payloadFromTensorIndices(row.output_tensor_indices, tensors);
    if (!tensorPayload || !graphInputPayload || !graphOutputPayload) return false;
    let operatorIoBytes = 0;
    let operatorAssessedSlots = 0;
    let operatorUnassessedSlots = 0;
    for (const [index, operator] of operators.entries()) {
      if (Number(operator.operator_index) !== index
        || !(row.operator_histogram || []).some((item) => item.name === operator.name)
        || !(operator.inputs || []).every((tensor) => validLocalTensor(tensor, row))
        || !(operator.outputs || []).every((tensor) => validLocalTensor(tensor, row))) return false;
      const payload = payloadFromTensorIndices([...(operator.inputs || []), ...(operator.outputs || [])], tensors);
      if (!payload
        || Number(operator.assessed_logical_io_payload_bytes || 0) !== payload.assessedBytes
        || Number(operator.assessed_io_tensor_slot_count || 0) !== payload.assessedSlots
        || Number(operator.unassessed_io_tensor_slot_count || 0) !== payload.unassessedSlots
        || Number(operator.present_io_tensor_slot_count || 0) !== payload.assessedSlots + payload.unassessedSlots
        || (operator.logical_io_payload_bytes == null ? null : Number(operator.logical_io_payload_bytes)) !== payload.totalBytes
        || operator.logical_io_payload_status !== (payload.unassessedSlots === 0 ? "assessed" : payload.assessedSlots ? "partial" : "not_assessed")) return false;
      operatorIoBytes += payload.assessedBytes;
      operatorAssessedSlots += payload.assessedSlots;
      operatorUnassessedSlots += payload.unassessedSlots;
      if (operator.mac_assessment_status === "not_applicable") {
        if (operator.mac_formula_class !== "not_applicable" || operator.nominal_macs != null || operator.nominal_macs_decimal != null) return false;
      } else if (["assessed_nominal", "modeled_scenario"].includes(operator.mac_assessment_status)) {
        if (!safeNumberMatchesDecimal(operator.nominal_macs, operator.nominal_macs_decimal)) return false;
      } else if (operator.mac_assessment_status !== "not_assessed"
        || operator.mac_formula_class === "not_applicable"
        || operator.nominal_macs != null || operator.nominal_macs_decimal != null) return false;
    }
    const exact = operators.filter((operator) => operator.mac_assessment_status === "assessed_nominal");
    const modeled = operators.filter((operator) => operator.mac_assessment_status === "modeled_scenario");
    const unassessed = operators.filter((operator) => operator.mac_assessment_status === "not_assessed");
    const compute = operators.filter((operator) => operator.mac_formula_class !== "not_applicable");
    const exactSum = exact.reduce((sum, operator) => sum + unsignedDecimal(operator.nominal_macs_decimal), 0n);
    const modeledSum = modeled.reduce((sum, operator) => sum + unsignedDecimal(operator.nominal_macs_decimal), 0n);
    const complete = compute.length > 0 && modeled.length === 0 && unassessed.length === 0;
    const expectedStatus = compute.length === 0 && tensorPayload.unassessedSlots === 0 && operatorUnassessedSlots === 0
      ? "assessed_no_mac_compute"
      : modeled.length === 0 && unassessed.length === 0 && tensorPayload.unassessedSlots === 0 && operatorUnassessedSlots === 0
        ? "assessed" : "partial";
    const constantTensors = tensors.filter((tensor) => tensor.constant_buffer);
    const constantLogicalBytes = constantTensors.reduce((sum, tensor) => sum + Number(tensor.buffer_data_length || 0), 0);
    const uniqueConstants = new Map(constantTensors
      .filter((tensor) => Number(tensor.buffer_data_length || 0) > 0)
      .map((tensor) => [`${tensor.buffer_data_offset}:${tensor.buffer_data_length}`, Number(tensor.buffer_data_length)]));
    const uniqueConstantBytes = [...uniqueConstants.values()].reduce((sum, value) => sum + value, 0);
    return cost.status === expectedStatus
      && Number(cost.mac_compute_operator_count || 0) === compute.length
      && Number(cost.assessed_nominal_mac_operator_count || 0) === exact.length
      && Number(cost.modeled_scenario_mac_operator_count || 0) === modeled.length
      && Number(cost.unassessed_mac_operator_count || 0) === unassessed.length
      && safeNumberMatchesDecimal(cost.assessed_nominal_macs, exactSum.toString())
      && String(cost.assessed_nominal_macs_decimal) === exactSum.toString()
      && safeNumberMatchesDecimal(cost.modeled_scenario_macs, modeledSum.toString())
      && String(cost.modeled_scenario_macs_decimal) === modeledSum.toString()
      && (complete ? String(cost.complete_nominal_macs_decimal) === exactSum.toString()
        && safeNumberMatchesDecimal(cost.complete_nominal_macs, exactSum.toString())
        : cost.complete_nominal_macs == null && cost.complete_nominal_macs_decimal == null)
      && Number(cost.assessed_logical_tensor_payload_bytes || 0) === tensorPayload.assessedBytes
      && Number(cost.assessed_tensor_payload_count || 0) === tensorPayload.assessedSlots
      && Number(cost.unassessed_tensor_payload_count || 0) === tensorPayload.unassessedSlots
      && (cost.logical_tensor_payload_bytes == null ? null : Number(cost.logical_tensor_payload_bytes)) === tensorPayload.totalBytes
      && Number(cost.assessed_logical_operator_io_payload_bytes || 0) === operatorIoBytes
      && Number(cost.assessed_operator_io_tensor_slot_count || 0) === operatorAssessedSlots
      && Number(cost.unassessed_operator_io_tensor_slot_count || 0) === operatorUnassessedSlots
      && (cost.logical_operator_io_payload_bytes == null ? null : Number(cost.logical_operator_io_payload_bytes)) === (operatorUnassessedSlots === 0 ? operatorIoBytes : null)
      && (cost.graph_input_payload_bytes == null ? null : Number(cost.graph_input_payload_bytes)) === graphInputPayload.totalBytes
      && (cost.graph_output_payload_bytes == null ? null : Number(cost.graph_output_payload_bytes)) === graphOutputPayload.totalBytes
      && Number(cost.logical_constant_reference_bytes || 0) === constantLogicalBytes
      && Number(cost.physical_unique_constant_bytes || 0) === uniqueConstantBytes
      && Number(cost.physical_unique_constant_buffer_count || 0) === uniqueConstants.size;
  })
    && engineeringReportText.includes("### Per-invocation Intrinsic Cost Ledger")
    && engineeringReportText.includes(rows[0]?.intrinsic_cost?.schema || "missing-intrinsic-schema")
    && rows.slice(0, 64).every((row) => engineeringReportText.includes(row.invocation_semantics || "missing-invocation-semantics")),
  "TFLite per-subgraph intrinsic MAC, payload, constant, or report ledger does not independently reconstruct.", ["/evidence/static_analysis/tflite_subgraph_inventory/rows", "/engineering_report.md"]);
  check("CF-TFLITE-SUBGRAPH-006", propertyValue("deepbom:model:tfliteSubgraphInventorySchema") === inventory.schema
    && propertyValue("deepbom:model:tfliteSubgraphCount") === String(inventory.subgraph_count || 0)
    && propertyValue("deepbom:model:tfliteSerializedOperatorCount") === String(inventory.serialized_operator_count || 0)
    && propertyValue("deepbom:model:tflitePrimaryOperatorCount") === String(inventory.primary_operator_count || 0)
    && propertyValue("deepbom:model:tfliteControlFlowReferenceCount") === String(inventory.control_flow_reference_count || 0)
    && propertyValue("deepbom:model:tfliteControlFlowContractCount") === String(inventory.control_flow_contract_count || 0)
    && propertyValue("deepbom:model:tfliteControlFlowContractPartialCount") === String(inventory.partial_control_flow_contract_count || 0)
    && propertyValue("deepbom:model:tfliteSubgraphIntrinsicCostSchema") === rows[0]?.intrinsic_cost?.schema
    && propertyValue("deepbom:model:tfliteSubgraphIntrinsicAssessedCount") === String(rows.filter((row) => String(row?.intrinsic_cost?.status || "").startsWith("assessed")).length)
    && propertyValue("deepbom:model:tfliteSubgraphIntrinsicPartialCount") === String(rows.filter((row) => row?.intrinsic_cost?.status === "partial").length),
  "CycloneDX model component does not preserve the TFLite all-subgraph summary.", ["/evidence/static_analysis/tflite_subgraph_inventory", "/evidence/mlbom_cyclonedx/metadata/component/properties"]);

  const deep = staticAnalysis?.tflite_subgraph_deep_analysis || {};
  const deepRows = Array.isArray(deep.rows) ? deep.rows : [];
  const close = (left, right) => Number.isFinite(Number(left)) && Number.isFinite(Number(right))
    && Math.abs(Number(left) - Number(right)) <= Math.max(1e-9, Math.abs(Number(right)) * 1e-12);
  check("CF-TFLITE-DEEP-001", deep.schema === "deepbom.tflite_subgraph_deep_analysis.v1"
    && deep.status === "assessed_all_serialized_subgraphs"
    && deep.evidence_class === "OBSERVED/DERIVED/PREDICTED_SOURCE_PINNED"
    && Number(deep.subgraph_count || 0) === rows.length
    && Number(deep.assessed_subgraph_count || 0) === rows.length
    && Number(deep.primary_subgraph_index || 0) === 0
    && deepRows.length === rows.length
    && deepRows.every((row, index) => Number(row.subgraph_index) === index
      && row.name === rows[index]?.name
      && Number(row.operator_count || 0) === Number(rows[index]?.operator_count || 0)
      && Number(row.tensor_count || 0) === Number(rows[index]?.tensor_count || 0)
      && Boolean(row.reachable_from_entrypoint) === Boolean(rows[index]?.reachable_from_entrypoint)
      && row.invocation_semantics === rows[index]?.invocation_semantics),
  "TFLite deep-scope identity, count, reachability, or invocation ledger differs from the serialized subgraph inventory.", ["/evidence/static_analysis/tflite_subgraph_deep_analysis", "/evidence/static_analysis/tflite_subgraph_inventory/rows"]);
  check("CF-TFLITE-DEEP-002", deepRows.every((row, subgraphIndex) => {
    const opRows = Array.isArray(row.operator_evidence) ? row.operator_evidence : [];
    const delegate = row.delegate || {};
    const inventoryCost = rows[subgraphIndex]?.intrinsic_cost || {};
    const opMacs = opRows.reduce((sum, op) => sum + Number(op.nominal_macs || 0), 0);
    const completeIntrinsic = inventoryCost.complete_nominal_macs;
    return opRows.length === Number(row.operator_count || 0)
      && opRows.every((op, index) => Number(op.operator_index) === index
        && typeof op.name === "string" && Array.isArray(op.inputs) && Array.isArray(op.outputs)
        && Number.isFinite(Number(op.nominal_macs)) && Number(op.nominal_macs) >= 0)
      && close(opMacs, row.total_macs)
      && Number(delegate.assessed_operator_count || 0) === opRows.length
      && Number(delegate.predicted_delegated_operator_count || 0) + Number(delegate.predicted_fallback_operator_count || 0) === opRows.length
      && close(Number(delegate.predicted_delegated_macs || 0) + Number(delegate.predicted_fallback_macs || 0), row.total_macs)
      && (Number(row.total_macs || 0) === 0
        ? Number(delegate.predicted_delegated_mac_ratio || 0) === 0
        : close(delegate.predicted_delegated_mac_ratio, Number(delegate.predicted_delegated_macs || 0) / Number(row.total_macs)))
      && (subgraphIndex === 0 || completeIntrinsic == null || close(row.total_macs, completeIntrinsic))
      && typeof row.tensor_liveness?.status === "string"
      && typeof row.tensor_arena_plan?.status === "string"
      && typeof row.predicted_partition_boundaries?.status === "string"
      && typeof row.weight_integrity?.status === "string";
  }),
  "TFLite per-scope op MAC, delegate denominator/MAC conservation, intrinsic MAC, or memory/integrity evidence does not reconstruct.", ["/evidence/static_analysis/tflite_subgraph_deep_analysis/rows", "/evidence/static_analysis/tflite_subgraph_inventory/rows"]);
  const primaryDeep = deepRows[0] || {};
  const nestedDeep = deepRows.slice(1);
  check("CF-TFLITE-DEEP-003", primaryDeep.advanced_numerical_storage === "referenced_top_level_without_duplication"
    && primaryDeep.advanced_numerical_evidence == null
    && Array.isArray(primaryDeep.advanced_numerical_evidence_pointers)
    && primaryDeep.advanced_numerical_evidence_pointers.length === 18
    && primaryDeep.advanced_numerical_evidence_pointers.every((pointer) => typeof pointer === "string" && pointer.startsWith("/"))
    && nestedDeep.every((row) => row.advanced_numerical_storage === "embedded_in_scope_row"
      && Array.isArray(row.advanced_numerical_evidence_pointers) && row.advanced_numerical_evidence_pointers.length === 0
      && row.advanced_numerical_evidence && typeof row.advanced_numerical_evidence === "object")
    && engineeringReportText.includes("## TFLite Per-subgraph Deep Analysis")
    && engineeringReportText.includes(deep.schema)
    && deepRows.slice(0, 64).every((row) => engineeringReportText.includes(`S${row.subgraph_index}`)
      && engineeringReportText.includes(row.invocation_semantics)),
  "TFLite deep numerical evidence is duplicated, absent from nested scopes, or missing from the Engineering Report.", ["/evidence/static_analysis/tflite_subgraph_deep_analysis", "/engineering_report.md"]);
  const deepDelegated = deepRows.reduce((sum, row) => sum + Number(row?.delegate?.predicted_delegated_operator_count || 0), 0);
  const deepFallback = deepRows.reduce((sum, row) => sum + Number(row?.delegate?.predicted_fallback_operator_count || 0), 0);
  check("CF-TFLITE-DEEP-004", propertyValue("deepbom:model:tfliteSubgraphDeepAnalysisSchema") === deep.schema
    && propertyValue("deepbom:model:tfliteSubgraphDeepAssessedCount") === String(deep.assessed_subgraph_count || 0)
    && propertyValue("deepbom:model:tfliteSubgraphDeepPredictedDelegateCount") === String(deepDelegated)
    && propertyValue("deepbom:model:tfliteSubgraphDeepPredictedFallbackCount") === String(deepFallback),
  "CycloneDX model component does not preserve the TFLite deep-scope summary.", ["/evidence/static_analysis/tflite_subgraph_deep_analysis", "/evidence/mlbom_cyclonedx/metadata/component/properties"]);
}
