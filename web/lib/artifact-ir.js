import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const ARTIFACT_IR_SCHEMA = "deepbom.artifact_ir.v2";
export const ARTIFACT_IR_METHOD_VERSION = "2.1.0";

const SHA256 = /^[a-f0-9]{64}$/;
const GRAPH_FORMATS = new Set(["tflite", "onnx", "coreml", "mlmodel", "executorch", "pte", "ptd"]);
const INLINE_PARAMETER_LIMIT = 32;

export function buildArtifactEvidenceIr(analysis, artifact = {}, { runtimeEvidence = null } = {}) {
  if (!analysis || typeof analysis !== "object") throw new Error("Artifact IR requires analyzed artifact evidence.");
  const format = normalizeFormat(analysis.format || artifact.format);
  const identity = artifactIdentity(analysis, artifact, format);
  const tensors = list(analysis.tensors);
  const graph = GRAPH_FORMATS.has(format)
    ? buildSerializedGraph(analysis, format, tensors)
    : notSerializedGraph(format);
  const storage = buildStorageTopology(analysis, format, tensors);
  const architecture = buildArchitectureProjection(analysis, format, tensors);
  const quantization = buildQuantizationContracts(analysis, format, graph, storage, tensors);
  const overlays = buildOverlays(analysis, format, graph, architecture, runtimeEvidence, identity.sha256);
  const body = {
    schema: ARTIFACT_IR_SCHEMA,
    method_version: ARTIFACT_IR_METHOD_VERSION,
    hash_contract: {
      algorithm: "SHA-256",
      canonicalization: "RFC8785-JCS",
      source_encoding: "UTF-8",
      excluded_pointers: ["/artifact_ir_sha256"],
    },
    artifact: identity,
    evidence_layers: {
      native_evidence: "artifact_serialized_or_hash_bound_sidecar_facts",
      canonical_ir: "format_neutral_identity_and_relationship_projection",
      static_projection: "separate_overlay_not_graph_fact",
      runtime_evidence: overlays.runtime.length ? "imported_identity_bound_overlay" : "not_imported",
    },
    graph,
    storage_topology: storage,
    architecture_projection: architecture,
    quantization_contracts: quantization,
    overlays,
    completeness: completeness(graph, storage, architecture, quantization, overlays),
    interpretation_boundary: "The canonical graph, storage, architecture, and quantization ledgers preserve distinct evidence scopes. Static backend eligibility and imported runtime observations are overlays and never rewrite serialized artifact facts. A missing executable graph is represented as not_serialized; no execution edge is synthesized from tensor names or architecture order.",
  };
  validateArtifactIrBody(body);
  return deepFreeze({ ...body, artifact_ir_sha256: sha256TextHex(canonicalJson(body)) });
}

export function validateArtifactEvidenceIr(document) {
  const body = clone(document);
  const digest = String(body.artifact_ir_sha256 || "").toLowerCase();
  delete body.artifact_ir_sha256;
  validateArtifactIrBody(body);
  if (!SHA256.test(digest) || digest !== sha256TextHex(canonicalJson(body))) {
    throw new Error("Artifact IR SHA-256 is invalid.");
  }
  return deepFreeze({ ...body, artifact_ir_sha256: digest });
}

function buildSerializedGraph(analysis, format, tensors) {
  const ops = canonicalOps(analysis.ops);
  const scope = primaryScope(analysis, format);
  const valueByIndex = new Map(tensors.map((tensor, position) => [tensorIndex(tensor, position), tensor]));
  const producer = new Map();
  const consumers = new Map();
  const operators = ops.map((op) => {
    const inputPorts = tensorPorts(op.inputs, scope.id);
    const outputPorts = tensorPorts(op.outputs, scope.id);
    const operatorRef = operatorId(scope.id, op.index);
    outputPorts.forEach(({ native_index: index, port }) => {
      if (!producer.has(index)) producer.set(index, { operator_ref: operatorRef, port });
    });
    inputPorts.forEach(({ native_index: index, port }) => {
      if (!consumers.has(index)) consumers.set(index, []);
      consumers.get(index).push({ operator_ref: operatorRef, port });
    });
    return {
      id: operatorRef,
      legacy_graph_node_id: `op:${op.index}`,
      scope_ref: scope.id,
      native_index: op.index,
      op_type: String(op.name || `OP_${op.index}`),
      name: optionalText(op.graph_node_name || op.coreml_layer_name || op.name),
      domain: String(op.domain || format),
      version: optionalInteger(op.version ?? op.op_version),
      inputs: inputPorts.map(({ port, value_ref }) => ({ port, value_ref })),
      outputs: outputPorts.map(({ port, value_ref }) => ({ port, value_ref })),
      metrics: {
        macs: exactInteger(op.macs_decimal ?? op.macs),
        mac_assessment_status: String(op.macs_status || (op.macs_decimal != null || op.macs != null ? "assessed" : "not_assessed")),
        logical_io_bytes: exactInteger(op.estimated_bytes),
      },
      quantization_summary: {
        state: String(op.quantization_state || (op.quantized_compute_path ? "8bit_compute" : "none")),
        risk: String(op.quant_risk || "none"),
      },
      topology: {
        depth: nonNegativeInteger(op.topo_depth),
        role: optionalText(op.topo_role),
        stage: optionalText(op.stage_key ?? op.stage_index),
      },
      native_source: nativeOperatorLocator(format, scope, op),
      completeness: "serialized_operator_with_available_canonical_contract",
    };
  });
  const graphInputs = new Set(integerArray(analysis.input_tensor_indices));
  const graphOutputs = new Set(integerArray(analysis.output_tensor_indices));
  const values = [...valueByIndex.entries()].sort(([left], [right]) => left - right).map(([index, tensor]) => ({
    id: valueId(scope.id, index),
    scope_ref: scope.id,
    native_index: index,
    name: String(tensor.name || `tensor_${index}`),
    value_kind: String(tensor.value_kind || "tensor"),
    dtype: String(tensor.dtype || "UNKNOWN").toUpperCase(),
    shape: dimensions(tensor.shape),
    shape_signature: Array.isArray(tensor.shape_signature) ? dimensions(tensor.shape_signature) : null,
    shape_contract_status: String(tensor.contract_status || tensor.buffer_data_status || "not_declared"),
    producer: producer.get(index) || null,
    consumers: uniqueConsumers(consumers.get(index) || []),
    roles: [
      ...(graphInputs.has(index) || tensor.role === "input" ? ["graph_input"] : []),
      ...(graphOutputs.has(index) || tensor.role === "output" ? ["graph_output"] : []),
      ...(tensor.constant_buffer || positiveStorageBytes(tensor, format) > 0 ? ["serialized_constant_or_storage"] : []),
    ],
    storage_refs: positiveStorageBytes(tensor, format) > 0 ? [storageId(index)] : [],
    logical_byte_length: logicalTensorBytes(tensor),
    native_source: nativeValueLocator(format, scope, tensor, index),
  }));
  const additional = materializeAdditionalScopes(analysis, format, scope.id);
  const scopes = [scope, ...additional.scopes];
  const allOperators = [...operators, ...additional.operators];
  const allValues = [...values, ...additional.values];
  const macAssessment = graphMacAssessment(analysis, operators);
  const serializedScopeAssessedMacs = allOperators.reduce((sum, row) => sum + BigInt(row.metrics?.macs?.decimal || "0"), 0n);
  return {
    status: "serialized",
    executable_graph_status: "serialized_artifact_graph",
    primary_scope_ref: scope.id,
    scopes,
    scope_relationships: additional.scope_relationships,
    operators: allOperators,
    values: allValues,
    inputs: values.filter((value) => value.roles.includes("graph_input")).map((value) => value.id),
    outputs: values.filter((value) => value.roles.includes("graph_output")).map((value) => value.id),
    totals: {
      scope_count: scopes.length,
      materialized_scope_count: scopes.filter((row) => row.materialization_status === "materialized").length,
      scope_relationship_count: additional.scope_relationships.length,
      operator_count: allOperators.length,
      value_count: allValues.length,
      relationship_count: allValues.reduce((sum, value) => sum + value.consumers.length, 0),
      macs: macAssessment.total,
      assessed_macs: macAssessment.assessed,
      serialized_scope_assessed_macs: exact(serializedScopeAssessedMacs),
      mac_assessment: macAssessment.assessment,
    },
    completeness: additional.scopes.length
      ? additional.scopes.every((row) => row.materialization_status === "materialized")
        ? "all_serialized_scopes_materialized"
        : "primary_scope_materialized_nested_scope_inventory_preserved"
      : "serialized_scope_materialized",
    interpretation_boundary: "Operators, values, ports, and producer-consumer relationships come from the serialized artifact. Primary-scope assessed_macs is the model-entry nominal total. serialized_scope_assessed_macs is only the sum of independently serialized scope rows and is not an execution total for conditional, repeated, or function scopes.",
  };
}

function notSerializedGraph(format) {
  return {
    status: "not_serialized",
    executable_graph_status: "not_serialized_by_format",
    primary_scope_ref: null,
    scopes: [], scope_relationships: [], operators: [], values: [], inputs: [], outputs: [],
    totals: { scope_count: 0, materialized_scope_count: 0, scope_relationship_count: 0, operator_count: 0, value_count: 0, relationship_count: 0, macs: null, assessed_macs: null, serialized_scope_assessed_macs: null, mac_assessment: null },
    completeness: "complete_not_applicable",
    interpretation_boundary: `${format || "This container"} does not serialize an executable operator graph. Runtime frameworks may construct one externally; DEEPBOM does not infer it from tensor names or architecture order.`,
  };
}

function buildStorageTopology(analysis, format, tensors) {
  const tensorObjects = tensors.map((tensor, position) => storageObject(format, tensor, tensorIndex(tensor, position)))
    .filter((row) => BigInt(row.serialized_byte_length?.decimal || "0") > 0n);
  const nestedTfliteObjects = format === "tflite" ? list(analysis?.tflite_subgraph_inventory?.rows)
    .filter((row) => Number(row.subgraph_index) !== 0)
    .flatMap((row) => list(row.tensor_intrinsics).map((tensor, position) => {
      const index = nonNegativeInteger(tensor.tensor_index) ?? position;
      const byteLength = exactInteger(tensor.buffer_data_length);
      const start = nonNegativeInteger(tensor.buffer_data_offset);
      return {
        id: scopedStorageId(`scope:tflite:subgraph:${row.subgraph_index}`, index),
        native_index: index,
        scope_ref: `scope:tflite:subgraph:${row.subgraph_index}`,
        name: String(tensor.name || `tensor_${index}`),
        dtype: String(tensor.dtype || "UNKNOWN").toUpperCase(),
        shape: dimensions(tensor.shape),
        serialized_byte_length: byteLength,
        byte_range: start != null && byteLength ? { status: "exact", offset_basis: "artifact_absolute", start, end_exclusive: safeExactSum(start, byteLength.decimal) }
          : { status: byteLength ? "length_only" : "not_assessed", offset_basis: null, start: null, end_exclusive: null },
        payload_sha256: null,
        encoding: { family: "scalar_or_declared_tensor_encoding", name: String(tensor.dtype || "UNKNOWN").toUpperCase(), block_elements: null, block_bytes: null },
        native_source: { format: "tflite", path: `SubGraph[${row.subgraph_index}].tensors[${index}].buffer` },
      };
    }).filter((row) => BigInt(row.serialized_byte_length?.decimal || "0") > 0n)) : [];
  const parameterObjects = (["coreml", "mlmodel"].includes(format) ? list(analysis?.weight_integrity?.parameters) : []).map((parameter, index) => ({
    id: `storage:parameter:${index}`,
    source_parameter_index: index,
    native_index: index,
    name: String(parameter.layer_name ? `${parameter.layer_name}/${parameter.role || index}` : parameter.name || `parameter_${index}`),
    dtype: String(parameter.storage || parameter.dtype || "UNKNOWN").toUpperCase(),
    shape: dimensions(parameter.shape),
    serialized_byte_length: exactInteger(parameter.byte_length),
    byte_range: { status: "length_only", offset_basis: null, start: null, end_exclusive: null },
    payload_sha256: normalizeSha256(parameter?.numerical_integrity?.payload_sha256 || parameter.payload_sha256),
    encoding: { family: parameter.quantization ? "coreml_declared_quantized_weight" : "scalar_or_declared_parameter_encoding", name: String(parameter.storage || parameter.dtype || "UNKNOWN"), block_elements: null, block_bytes: null },
    native_source: { format, path: `weight_integrity.parameters[${index}]` },
  })).filter((row) => BigInt(row.serialized_byte_length?.decimal || "0") > 0n);
  const objects = [...tensorObjects, ...nestedTfliteObjects, ...parameterObjects];
  const bytes = objects.reduce((sum, row) => sum + BigInt(row.serialized_byte_length?.decimal || "0"), 0n);
  return {
    status: objects.length ? "assessed_serialized_objects" : "not_applicable_no_serialized_tensor_payload_ledger",
    objects,
    totals: {
      object_count: objects.length,
      serialized_object_bytes_sum: exact(bytes),
      exact_range_count: objects.filter((row) => row.byte_range?.status === "exact").length,
      payload_digest_count: objects.filter((row) => SHA256.test(String(row.payload_sha256 || ""))).length,
    },
    interpretation_boundary: "Storage objects preserve serialized payload identity and ranges where the parser exposes them. Summed object bytes are not file size, runtime allocation, residency, repacking, or transfer volume.",
  };
}

function storageObject(format, tensor, index) {
  const byteLength = exactInteger(positiveStorageBytes(tensor, format));
  const absoluteOffset = nonNegativeInteger(tensor?.numerical_integrity?.byte_offset_absolute ?? tensor.buffer_data_offset);
  const relativeStart = nonNegativeInteger(tensor.data_offset);
  const start = absoluteOffset ?? relativeStart;
  const end = start != null && byteLength ? safeExactSum(start, byteLength.decimal) : null;
  return {
    id: storageId(index),
    native_index: index,
    name: String(tensor.name || `tensor_${index}`),
    dtype: String(tensor.dtype || "UNKNOWN").toUpperCase(),
    shape: dimensions(tensor.shape),
    serialized_byte_length: byteLength,
    byte_range: start != null && end != null
      ? { status: "exact", offset_basis: absoluteOffset != null ? "artifact_absolute" : "format_payload_relative", start, end_exclusive: end }
      : { status: byteLength ? "length_only" : "not_assessed", offset_basis: null, start: null, end_exclusive: null },
    payload_sha256: normalizeSha256(tensor?.numerical_integrity?.payload_sha256 || tensor.external_sidecar_sha256),
    encoding: storageEncoding(format, tensor),
    native_source: nativeStorageLocator(format, tensor, index),
  };
}

function buildArchitectureProjection(analysis, format, tensors) {
  const layerStorage = analysis?.on_device_llm?.storage?.layer_storage;
  const layers = list(layerStorage?.layers);
  if (layers.length) {
    const nodes = layers.map((layer) => ({
      id: architectureLayerId(layer.layer_index),
      kind: "decoder_layer_storage_group",
      native_index: Number(layer.layer_index),
      label: `Decoder layer ${layer.layer_index}`,
      tensor_count: nonNegativeInteger(layer.tensor_count),
      serialized_bytes: exactInteger(layer.serialized_bytes?.decimal ?? layer.serialized_bytes?.value ?? layer.serialized_bytes),
      grouping_basis: clone(layerStorage.namespace || null),
    }));
    return {
      status: "derived_from_serialized_tensor_namespace",
      executable_graph_status: "not_claimed",
      kind: "llm_layer_storage",
      nodes,
      relationships: [],
      totals: { node_count: nodes.length, relationship_count: 0 },
      interpretation_boundary: "Layer rows are storage groups derived from serialized tensor namespaces. Their numeric order is an architecture coordinate, not a runtime edge, execution schedule, lowering, or placement claim.",
    };
  }
  if (!GRAPH_FORMATS.has(format) && tensors.length) {
    const groups = namespaceGroups(tensors, format);
    return {
      status: "derived_from_serialized_tensor_namespace",
      executable_graph_status: "not_claimed",
      kind: "tensor_storage_namespace",
      nodes: groups,
      relationships: [],
      totals: { node_count: groups.length, relationship_count: 0 },
      interpretation_boundary: "Namespace groups organize tensor storage for review. No edge, call order, execution dependency, or runtime graph is inferred.",
    };
  }
  return {
    status: "not_applicable_serialized_graph_available",
    executable_graph_status: "represented_in_graph_ledger",
    kind: null, nodes: [], relationships: [], totals: { node_count: 0, relationship_count: 0 },
    interpretation_boundary: "No separate architecture projection is required when the artifact graph is serialized.",
  };
}

function buildQuantizationContracts(analysis, format, graph, storage, tensors) {
  const graphValues = new Map(graph.values.map((value) => [value.native_index, value.id]));
  const storageObjects = new Map(storage.objects.map((value) => [value.native_index, value.id]));
  const records = [];
  for (const [position, tensor] of tensors.entries()) {
    const index = tensorIndex(tensor, position);
    const scales = finiteNumberArray(tensor.scale_sample?.length ? tensor.scale_sample : tensor.interface_scale_values);
    const zeroPoints = integerArray(tensor.zero_point_sample?.length ? tensor.zero_point_sample : tensor.interface_zero_point_values);
    const encoded = format === "gguf" && /^Q\d|^IQ\d|^TQ\d|^MXFP|^BF16$/i.test(String(tensor.dtype || ""));
    if (!scales.length && !zeroPoints.length && !encoded) continue;
    const parameterization = encoded
      ? { kind: "per_block", axes: [], block_size: positiveInteger(tensor.block_elements) }
      : quantizationParameterization(tensor, scales);
    const subjectRef = graphValues.get(index) || storageObjects.get(index);
    if (!subjectRef) continue;
    records.push({
      id: `quantization:${subjectRef}`,
      subject_ref: subjectRef,
      mapping: encoded ? {
        family: "format_defined_block_encoding",
        scheme: String(tensor.dtype || "unknown").toUpperCase(),
        zero_point_constraint: "format_defined",
      } : {
        family: "affine",
        scheme: "affine_unspecified_symmetry",
        zero_point_constraint: zeroPointConstraint(zeroPoints),
      },
      parameterization,
      storage: {
        data_type: String(tensor.dtype || "UNKNOWN").toUpperCase(),
        ...codeDomain(tensor.dtype),
        block_bytes: positiveInteger(tensor.block_bytes),
      },
      parameters: encoded ? {
        status: "encoded_in_format_defined_blocks",
        scale: null,
        zero_point: null,
      } : {
        status: scales.length && zeroPoints.length ? "complete_affine_vectors" : "partial_affine_vectors",
        scale: parameterVector(scales),
        zero_point: parameterVector(zeroPoints),
      },
      source: {
        native_locator: nativeQuantizationLocator(format, tensor, index),
        evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
      },
      completeness: encoded || (scales.length && zeroPoints.length) ? "complete_for_serialized_contract" : "partial_serialized_contract",
    });
  }
  for (const [index, parameter] of (["coreml", "mlmodel"].includes(format) ? list(analysis?.weight_integrity?.parameters) : []).entries()) {
    const quantization = parameter?.quantization;
    if (!quantization) continue;
    const subjectRef = `storage:parameter:${index}`;
    if (!storage.objects.some((row) => row.id === subjectRef)) continue;
    const perAxis = Number(quantization.scale_count || 0) > 1;
    records.push({
      id: `quantization:${subjectRef}`,
      subject_ref: subjectRef,
      mapping: {
        family: quantization.scheme === "lookup_table" ? "lookup_table" : "affine_scale_and_additive_bias",
        scheme: String(quantization.scheme || "unknown"),
        zero_point_constraint: "not_represented_as_zero_point",
      },
      parameterization: {
        kind: perAxis ? "per_axis" : "per_tensor",
        axes: Number.isSafeInteger(Number(quantization.axis)) ? [Number(quantization.axis)] : [],
        axis_status: perAxis && !Number.isSafeInteger(Number(quantization.axis)) ? "not_exposed_by_serialized_weight_contract" : "not_applicable_or_explicit",
        block_size: null,
      },
      storage: {
        data_type: parameter.storage === "int8_dynamic" ? "INT8" : `UINT${quantization.number_of_bits || ""}`,
        code_min: parameter.storage === "int8_dynamic" ? -128 : 0,
        code_max: parameter.storage === "int8_dynamic" ? 127 : Number.isSafeInteger(Number(quantization.number_of_bits)) ? 2 ** Number(quantization.number_of_bits) - 1 : null,
        code_domain_status: "coreml_serialized_weight_encoding",
        block_bytes: null,
      },
      parameters: {
        status: "complete_serialized_parameter_digests",
        scale: digestDescriptor(quantization.scale_count, quantization.scale_payload_sha256),
        additive_bias: digestDescriptor(quantization.bias_count, quantization.bias_payload_sha256),
        lookup_table: digestDescriptor(quantization.lookup_table_count, quantization.lookup_table_payload_sha256),
      },
      source: { native_locator: { format: "coreml", path: `weight_integrity.parameters[${index}].quantization` }, evidence_class: "OBSERVED_SERIALIZED_ARTIFACT" },
      completeness: "complete_for_serialized_contract",
    });
  }
  const safeContract = analysis?.safetensors?.quantization_contract;
  const storageByName = new Map(storage.objects.map((row) => [row.name, row.id]));
  for (const [index, module] of list(safeContract?.modules).entries()) {
    const related = Object.values(module?.tensors || {}).map((row) => storageByName.get(String(row?.tensor_name || ""))).filter(Boolean);
    const subjectRef = related[0];
    if (!subjectRef) continue;
    records.push({
      id: `quantization:safetensors:module:${index}`,
      subject_ref: subjectRef,
      related_storage_refs: [...new Set(related)],
      mapping: { family: "packed_integer_weight", scheme: String(safeContract.method || "unknown"), zero_point_constraint: module.symmetric === true ? "declared_symmetric" : module.symmetric === false ? "declared_asymmetric" : "not_declared" },
      parameterization: { kind: "per_group", axes: Number.isSafeInteger(Number(module.logical_weight_axis)) ? [Number(module.logical_weight_axis)] : [], block_size: positiveInteger(module.group_size), group_count: nonNegativeInteger(module.group_count) },
      storage: { data_type: `PACKED_UINT${module.bits || safeContract.bits || ""}`, code_min: 0, code_max: Number.isSafeInteger(Number(module.bits || safeContract.bits)) ? 2 ** Number(module.bits || safeContract.bits) - 1 : null, code_domain_status: "source_pinned_packed_layout", block_bytes: null },
      parameters: { status: String(module.quantization_payload_integrity?.status || module.status || "not_assessed"), scale: { count: decimalCount(module.scale_element_count), sha256: null, inline_values: null, inline_status: "referenced_storage_object" }, zero_point: { count: decimalCount(module.zero_point_code_capacity), sha256: null, inline_values: null, inline_status: module.zero_point_storage_transform || "not_assessed" } },
      source: { native_locator: { format: "safetensors", path: `safetensors.quantization_contract.modules[${index}]` }, evidence_class: String(safeContract.evidence_class || "OBSERVED/DERIVED_FROM_PINNED_FORMAT_SOURCE") },
      completeness: module.status === "pass" ? "complete_for_serialized_contract" : "partial_serialized_contract",
    });
  }
  return {
    status: records.length ? "assessed" : "not_applicable_no_serialized_quantization_contract",
    records,
    totals: {
      record_count: records.length,
      affine_record_count: records.filter((row) => row.mapping.family === "affine").length,
      block_encoding_record_count: records.filter((row) => row.mapping.family === "format_defined_block_encoding").length,
      complete_record_count: records.filter((row) => row.completeness === "complete_for_serialized_contract").length,
      partial_record_count: records.filter((row) => row.completeness !== "complete_for_serialized_contract").length,
    },
    interpretation_boundary: "Each record is bound to one canonical value or storage object. Per-tensor, per-axis, and per-block are partition descriptions, not an ordered precision scale. A zero zero-point is preserved as observed data and is not by itself promoted to a symmetric-quantization claim.",
  };
}

function buildOverlays(analysis, format, graph, architecture, runtimeEvidence, artifactSha256) {
  const staticRows = [];
  let staticSummary = null;
  const primaryOperators = graph.operators.filter((row) => row.scope_ref === graph.primary_scope_ref);
  if (graph.status === "serialized" && format === "tflite") {
    for (const operator of primaryOperators) {
      const op = list(analysis.ops).find((row) => Number(row?.index) === operator.native_index);
      const eligible = op?.xnnpack_supported === true;
      staticRows.push({
        subject_ref: operator.id,
        state: eligible ? "CONDITIONALLY_DELEGATABLE" : op?.xnnpack_supported === false ? "PREDICTED_FALLBACK_OR_BREAK" : "NOT_ASSESSABLE",
        backend: eligible ? "xnnpack" : "cpu",
        evidence_state: eligible ? "ARTIFACT_ELIGIBLE" : op?.xnnpack_supported === false ? "SOURCE_REGISTERED" : "NOT_ASSESSABLE",
        evidence_class: eligible ? "PREDICTED_SOURCE_AND_ARTIFACT_ELIGIBILITY" : op?.xnnpack_supported === false ? "PREDICTED" : "NOT_ASSESSABLE",
        reason_codes: compactStrings([op?.xnnpack_reason || op?.xnnpack_break_class]),
        unresolved_predicates: compactStrings([op?.xnnpack_build_requirement]),
      });
    }
    staticSummary = { profile_id: "xnnpack", backend: "XNNPACK", original_op_engine_selection_claim: false };
  } else if (graph.status === "serialized" && format === "onnx") {
    const projection = analysis?.tensorrt_static_preflight?.projection;
    if (projection && list(projection.rows).length === primaryOperators.length) {
      const byIndex = new Map(list(projection.rows).map((row) => [Number(row.op_index), row]));
      for (const operator of primaryOperators) {
        const row = byIndex.get(operator.native_index);
        if (!row) throw new Error(`Artifact IR TensorRT overlay is missing operator ${operator.native_index}.`);
        staticRows.push({
          subject_ref: operator.id,
          state: String(row.state || "UNRESOLVED"),
          backend: String(projection.profile_id || "tensorrt"),
          evidence_state: row.state === "CONDITIONALLY_ELIGIBLE" ? "ARTIFACT_ELIGIBLE" : row.state === "DEFINITE_EXCLUSION" ? "BUILD_INCLUDED" : "NOT_ASSESSABLE",
          evidence_class: String(projection.evidence_class || analysis?.tensorrt_static_preflight?.evidence_class || "NOT_ASSESSABLE"),
          reason_codes: compactStrings(row.reason_codes),
          unresolved_predicates: compactStrings(row.unresolved_predicates),
        });
      }
      const engine = analysis?.tensorrt_static_preflight?.engine_inspector_evidence;
      staticSummary = {
        profile_id: String(projection.profile_id || "tensorrt"),
        backend: String(projection.label || projection.profile_id || "TensorRT"),
        projection_schema: String(projection.schema || ""),
        evidence_class: String(projection.evidence_class || "NOT_ASSESSABLE"),
        state_counts: clone(projection.state_counts || {}),
        parser_observation_status: analysis?.tensorrt_static_preflight?.parser_observation?.coverage_status || "not_observed",
        engine_inspector_status: engine?.status || "not_observed",
        engine_sha256: normalizeSha256(engine?.engine?.sha256),
        engine_source_mapping_status: engine?.source_mapping_status || "not_exposed",
        original_op_engine_selection_claim: false,
      };
    }
  }
  const llmScenario = selectLlmPlacementScenario(analysis?.accelerator_profile_binding);
  if (llmScenario) {
    const selected = new Set(integerArray(llmScenario.serialized_layer_offload?.selected_layer_indices));
    for (const node of architecture.nodes) staticRows.push({
      subject_ref: node.id,
      state: selected.has(node.native_index) ? "CONDITIONAL_SERIALIZED_LAYER_RESIDENCY_CANDIDATE" : "CONDITIONAL_OTHER_POOL_RESIDENCY_CANDIDATE",
      backend: selected.has(node.native_index) ? "nvidia_accelerator" : "cpu_or_other_pool",
      evidence_state: "CONFIGURATION_BOUND_STATIC_LOWER_BOUND",
      evidence_class: "DERIVED_CONDITIONAL_STATIC_LOWER_BOUND",
      reason_codes: [], unresolved_predicates: [],
    });
    staticSummary = {
      profile_id: "llm_static_memory_placement",
      backend: "declared accelerator residency scenario",
      source: llmScenario.scenario_source,
      context_length: llmScenario.context_length,
      batch_size: llmScenario.batch_size,
      storage_bits: llmScenario.storage_bits,
      fit_claim: llmScenario.fit_claim,
      original_op_engine_selection_claim: false,
    };
  }
  const staticOverlay = staticRows.length ? [{
    id: "overlay:static-placement:0",
    kind: "static_placement",
    evidence_class: staticSummary?.evidence_class || staticRows[0]?.evidence_class || "DERIVED",
    summary: staticSummary,
    rows: staticRows,
    interpretation_boundary: "Static eligibility or residency candidates do not establish selected-build acceptance, executed assignment, physical transfer, kernel choice, or latency.",
  }] : [];
  const runtime = normalizeRuntimeOverlay(runtimeEvidence, graph, architecture, artifactSha256);
  return { static: staticOverlay, runtime };
}

function normalizeRuntimeOverlay(runtimeEvidence, graph, architecture, artifactSha256) {
  if (!runtimeEvidence || typeof runtimeEvidence !== "object") return [];
  const valid = new Set([...graph.operators.map((row) => row.id), ...architecture.nodes.map((row) => row.id)]);
  const primaryOperatorByIndex = new Map(graph.operators
    .filter((row) => row.scope_ref === graph.primary_scope_ref)
    .map((row) => [row.native_index, row.id]));
  const declaredRuntimeNodes = list(runtimeEvidence.runtime_nodes);
  const candidateRows = declaredRuntimeNodes.length
    ? declaredRuntimeNodes
    : list(runtimeEvidence.rows || runtimeEvidence.assignments);
  const inputRows = declaredRuntimeNodes.length
    ? candidateRows
    : candidateRows.filter((row) => {
      const hasRuntimeIdentity = runtimeNodeIdentity(row) != null;
      const hasCanonicalBinding = runtimeSourceBinding(row, primaryOperatorByIndex).subjectRefs.length > 0;
      return hasRuntimeIdentity && hasCanonicalBinding;
    });
  if (!inputRows.length) return [];
  const runtimeArtifactSha256 = normalizeSha256(runtimeEvidence.artifact_sha256 || runtimeEvidence.artifact?.sha256);
  if (!runtimeArtifactSha256 || runtimeArtifactSha256 !== artifactSha256) throw new Error("Runtime overlay is not bound to the active artifact SHA-256.");
  const groupedRuntimeNodes = new Map();
  inputRows.forEach((row, index) => {
    const runtimeNodeRef = runtimeNodeIdentity(row);
    if (!runtimeNodeRef) throw new Error(`Runtime overlay row ${index} does not declare a runtime node identity.`);
    const binding = runtimeSourceBinding(row, primaryOperatorByIndex);
    const sourceSubjectRefs = binding.subjectRefs;
    if (sourceSubjectRefs.some((subjectRef) => !valid.has(subjectRef))) throw new Error("Runtime overlay references an unknown canonical subject.");
    const backend = optionalText(row.backend || row.provider || row.delegate);
    const evidenceClass = String(row.evidence_class || "MEASURED_IMPORTED");
    const existing = groupedRuntimeNodes.get(runtimeNodeRef);
    if (existing && existing.backend !== backend) throw new Error(`Runtime overlay node ${runtimeNodeRef} declares conflicting backends.`);
    if (existing && existing.evidence_class !== evidenceClass) throw new Error(`Runtime overlay node ${runtimeNodeRef} declares conflicting evidence classes.`);
    const current = existing || {
      runtime_node_ref: runtimeNodeRef,
      backend,
      source_subject_refs: new Set(),
      mapping_bases: new Set(),
      runtime_node_kind: optionalText(row.runtime_node_kind || row.kind),
      evidence_class: evidenceClass,
    };
    sourceSubjectRefs.forEach((subjectRef) => current.source_subject_refs.add(subjectRef));
    if (binding.mappingBasis) current.mapping_bases.add(binding.mappingBasis);
    groupedRuntimeNodes.set(runtimeNodeRef, current);
  });
  const runtimeNodes = [...groupedRuntimeNodes.values()].map((row) => {
    const sourceSubjectRefs = [...row.source_subject_refs].sort();
    const mappingBasis = sourceSubjectRefs.length === 0
      ? "explicit_unmapped_runtime_node"
      : row.mapping_bases.size === 1
        ? [...row.mapping_bases][0]
        : "explicit_subject_ref_and_native_index_import";
    return {
      runtime_node_ref: row.runtime_node_ref,
      backend: row.backend,
      source_subject_refs: sourceSubjectRefs,
      mapping_cardinality: sourceSubjectRefs.length > 1 ? "fused_one_to_many" : sourceSubjectRefs.length === 1 ? "one_to_one" : "unmapped",
      mapping_basis: mappingBasis,
      runtime_node_kind: row.runtime_node_kind,
      evidence_class: row.evidence_class,
    };
  }).sort((left, right) => left.runtime_node_ref.localeCompare(right.runtime_node_ref));
  if (!runtimeNodes.length) return [];
  const rows = runtimeNodes.flatMap((node) => node.source_subject_refs.map((subjectRef) => ({
    subject_ref: subjectRef,
    runtime_node_ref: node.runtime_node_ref,
    backend: node.backend,
    evidence_class: node.evidence_class,
    mapping_cardinality: node.mapping_cardinality,
  })));
  const mappedSubjects = new Set(rows.map((row) => row.subject_ref));
  const summary = {
    runtime_node_count: runtimeNodes.length,
    mapped_runtime_node_count: runtimeNodes.filter((row) => row.source_subject_refs.length).length,
    unmapped_runtime_node_count: runtimeNodes.filter((row) => !row.source_subject_refs.length).length,
    one_to_one_runtime_node_count: runtimeNodes.filter((row) => row.mapping_cardinality === "one_to_one").length,
    fused_runtime_node_count: runtimeNodes.filter((row) => row.mapping_cardinality === "fused_one_to_many").length,
    source_subject_reference_count: rows.length,
    distinct_source_subject_count: mappedSubjects.size,
    canonical_subject_count: valid.size,
    canonical_subject_coverage_status: mappedSubjects.size === valid.size ? "complete_explicit_mapping" : mappedSubjects.size ? "partial_explicit_mapping" : "not_mapped",
    name_similarity_mapping_used: false,
  };
  return [{
    id: "overlay:runtime:0",
    kind: "runtime_assignment",
    evidence_class: "IMPORTED_IDENTITY_BOUND_RUNTIME_EVIDENCE",
    summary,
    runtime_nodes: runtimeNodes,
    rows,
    interpretation_boundary: "Imported runtime nodes preserve artifact-bound explicit source_subject_refs or primary-scope native op indices, including one-to-one and fused one-to-many mappings. Generated or unresolved runtime nodes remain explicitly unmapped; DEEPBOM never reconciles them by name similarity.",
  }];
}

function runtimeNodeIdentity(row) {
  const explicit = optionalText(row.runtime_node_ref || row.node_ref || row.runtime_node_name || row.node_name || row.id);
  if (explicit) return explicit;
  const nativeIndex = nonNegativeInteger(row.runtime_node_index);
  return nativeIndex == null ? null : `runtime-node:${optionalText(row.backend || row.provider || row.delegate) || "unknown"}:${nativeIndex}`;
}

function runtimeSourceBinding(row, primaryOperatorByIndex) {
  const explicit = compactStrings(row.source_subject_refs?.length ? row.source_subject_refs : row.subject_ref);
  const rawIndices = row.source_op_indices?.length ? row.source_op_indices : row.op_indices?.length ? row.op_indices : row.op_index;
  const indices = integerArray(Array.isArray(rawIndices) ? rawIndices : rawIndices == null ? [] : [rawIndices])
    .filter((value, position, all) => value >= 0 && all.indexOf(value) === position);
  const indexed = indices.map((index) => {
    const subjectRef = primaryOperatorByIndex.get(index);
    if (!subjectRef) throw new Error(`Runtime overlay references unknown primary-scope operator index ${index}.`);
    return subjectRef;
  });
  const subjectRefs = [...new Set([...explicit, ...indexed])];
  return {
    subjectRefs,
    mappingBasis: explicit.length && indexed.length
      ? "explicit_subject_ref_and_native_index_import"
      : explicit.length
        ? "explicit_subject_ref_import"
        : indexed.length
          ? "explicit_primary_scope_native_index_import"
          : null,
  };
}

function validateArtifactIrBody(value) {
  if (value?.schema !== ARTIFACT_IR_SCHEMA || value?.method_version !== ARTIFACT_IR_METHOD_VERSION) throw new Error("Artifact IR schema identity is invalid.");
  if (value?.hash_contract?.algorithm !== "SHA-256"
    || value?.hash_contract?.canonicalization !== "RFC8785-JCS"
    || value?.hash_contract?.source_encoding !== "UTF-8"
    || JSON.stringify(value?.hash_contract?.excluded_pointers) !== JSON.stringify(["/artifact_ir_sha256"])) {
    throw new Error("Artifact IR hash contract is invalid.");
  }
  if (!SHA256.test(String(value.artifact?.sha256 || "")) || !text(value.artifact?.filename, 1000) || !text(value.artifact?.format, 40)) throw new Error("Artifact IR artifact identity is invalid.");
  const graph = value.graph;
  if (!graph || !Array.isArray(graph.scopes) || !Array.isArray(graph.scope_relationships) || !Array.isArray(graph.operators) || !Array.isArray(graph.values)) throw new Error("Artifact IR graph ledger is invalid.");
  if (graph.status === "not_serialized" && (graph.scopes.length || graph.operators.length || graph.values.length || graph.totals.relationship_count)) {
    throw new Error("Artifact IR fabricated an executable graph for a graphless format.");
  }
  if (graph.totals.scope_count !== graph.scopes.length || graph.totals.materialized_scope_count !== graph.scopes.filter((row) => row.materialization_status === "materialized").length
    || graph.totals.scope_relationship_count !== graph.scope_relationships.length || graph.totals.operator_count !== graph.operators.length || graph.totals.value_count !== graph.values.length) throw new Error("Artifact IR graph count conservation failed.");
  const scopeIds = uniqueIds(graph.scopes, "graph scope");
  const operatorIds = uniqueIds(graph.operators, "operator");
  const valueIds = uniqueIds(graph.values, "value");
  const operatorById = new Map(graph.operators.map((row) => [row.id, row]));
  const valueById = new Map(graph.values.map((row) => [row.id, row]));
  if (graph.status === "serialized" && !scopeIds.has(graph.primary_scope_ref)) throw new Error("Artifact IR primary graph scope reference is invalid.");
  for (const scope of graph.scopes) {
    if (scope.parent_scope_ref != null && (!scopeIds.has(scope.parent_scope_ref) || scope.parent_scope_ref === scope.id)) {
      throw new Error("Artifact IR graph scope parent reference is invalid.");
    }
    if (scope.materialization_status === "materialized") {
      const scopeOperatorCount = graph.operators.filter((row) => row.scope_ref === scope.id).length;
      const scopeValueCount = graph.values.filter((row) => row.scope_ref === scope.id).length;
      if (scope.declared_operator_count != null && scope.declared_operator_count !== scopeOperatorCount) throw new Error("Artifact IR materialized scope operator count is inconsistent.");
      if (scope.declared_value_count != null && scope.declared_value_count !== scopeValueCount) throw new Error("Artifact IR materialized scope value count is inconsistent.");
    }
  }
  for (const operator of graph.operators) {
    if (!scopeIds.has(operator.scope_ref)) throw new Error("Artifact IR operator scope reference is invalid.");
    for (const port of [...list(operator.inputs), ...list(operator.outputs)]) {
      const graphValue = valueById.get(port.value_ref);
      if (!graphValue) throw new Error("Artifact IR operator port references an unknown value.");
      if (graphValue.scope_ref !== operator.scope_ref) throw new Error("Artifact IR operator port crosses graph scopes.");
    }
  }
  for (const relationship of graph.scope_relationships) {
    if (!scopeIds.has(relationship.source_scope_ref) || !scopeIds.has(relationship.target_scope_ref)
      || (relationship.source_operator_ref && !operatorIds.has(relationship.source_operator_ref))) {
      throw new Error("Artifact IR scope relationship reference is invalid.");
    }
    if (relationship.source_operator_ref && operatorById.get(relationship.source_operator_ref)?.scope_ref !== relationship.source_scope_ref) {
      throw new Error("Artifact IR scope relationship source operator belongs to a different scope.");
    }
  }
  let relationshipCount = 0;
  for (const value of graph.values) {
    if (!scopeIds.has(value.scope_ref)) throw new Error("Artifact IR value scope reference is invalid.");
    if (value.producer && !operatorIds.has(value.producer.operator_ref)) throw new Error("Artifact IR value producer reference is invalid.");
    for (const consumer of list(value.consumers)) if (!operatorIds.has(consumer.operator_ref)) throw new Error("Artifact IR value consumer reference is invalid.");
    if (value.producer && operatorById.get(value.producer.operator_ref)?.scope_ref !== value.scope_ref) throw new Error("Artifact IR value producer crosses graph scopes.");
    for (const consumer of list(value.consumers)) {
      if (operatorById.get(consumer.operator_ref)?.scope_ref !== value.scope_ref) throw new Error("Artifact IR value consumer crosses graph scopes.");
    }
    relationshipCount += list(value.consumers).length;
  }
  if (relationshipCount !== graph.totals.relationship_count) throw new Error("Artifact IR relationship conservation failed.");
  for (const inputRef of list(graph.inputs)) {
    const graphValue = valueById.get(inputRef);
    if (!graphValue || graphValue.scope_ref !== graph.primary_scope_ref || !list(graphValue.roles).includes("graph_input")) throw new Error("Artifact IR graph input reference is invalid.");
  }
  for (const outputRef of list(graph.outputs)) {
    const graphValue = valueById.get(outputRef);
    if (!graphValue || graphValue.scope_ref !== graph.primary_scope_ref || !list(graphValue.roles).includes("graph_output")) throw new Error("Artifact IR graph output reference is invalid.");
  }
  const primaryOperators = graph.operators.filter((row) => row.scope_ref === graph.primary_scope_ref);
  const assessedMacs = primaryOperators.reduce((sum, row) => sum + BigInt(row.metrics?.macs?.decimal || "0"), 0n);
  const serializedScopeMacs = graph.operators.reduce((sum, row) => sum + BigInt(row.metrics?.macs?.decimal || "0"), 0n);
  if (graph.status === "serialized" && (String(graph.totals.assessed_macs?.decimal || "0") !== assessedMacs.toString()
    || String(graph.totals.serialized_scope_assessed_macs?.decimal || "0") !== serializedScopeMacs.toString())) throw new Error("Artifact IR MAC conservation failed.");
  const storage = value.storage_topology;
  if (!storage || !Array.isArray(storage.objects) || storage.totals.object_count !== storage.objects.length) throw new Error("Artifact IR storage count conservation failed.");
  const storageIds = uniqueIds(storage.objects, "storage object");
  const storageBytes = storage.objects.reduce((sum, row) => sum + BigInt(row.serialized_byte_length?.decimal || "0"), 0n);
  if (String(storage.totals.serialized_object_bytes_sum?.decimal || "0") !== storageBytes.toString()) throw new Error("Artifact IR storage byte conservation failed.");
  if (storage.totals.exact_range_count !== storage.objects.filter((row) => row.byte_range?.status === "exact").length
    || storage.totals.payload_digest_count !== storage.objects.filter((row) => SHA256.test(String(row.payload_sha256 || ""))).length) {
    throw new Error("Artifact IR storage evidence count conservation failed.");
  }
  for (const graphValue of graph.values) {
    for (const storageRef of list(graphValue.storage_refs)) {
      if (!storageIds.has(storageRef)) throw new Error("Artifact IR value references an unknown storage object.");
    }
  }
  const architecture = value.architecture_projection;
  if (!architecture || !Array.isArray(architecture.nodes) || !Array.isArray(architecture.relationships)) throw new Error("Artifact IR architecture projection is invalid.");
  const architectureIds = uniqueIds(architecture.nodes, "architecture node");
  if (architecture.totals.node_count !== architecture.nodes.length || architecture.totals.relationship_count !== architecture.relationships.length) throw new Error("Artifact IR architecture count conservation failed.");
  if (graph.status === "not_serialized" && architecture.relationships.length) throw new Error("Artifact IR graphless architecture projection contains fabricated relationships.");
  for (const node of architecture.nodes) {
    for (const storageRef of list(node.storage_object_refs)) if (!storageIds.has(storageRef)) throw new Error("Artifact IR architecture node references an unknown storage object.");
  }
  const quantization = value.quantization_contracts;
  if (!quantization || !Array.isArray(quantization.records) || !quantization.totals) throw new Error("Artifact IR quantization ledger is invalid.");
  const quantIds = uniqueIds(quantization.records, "quantization record");
  if (quantIds.size !== quantization.totals.record_count) throw new Error("Artifact IR quantization count conservation failed.");
  if (quantization.totals.affine_record_count !== quantization.records.filter((row) => row.mapping?.family === "affine").length
    || quantization.totals.block_encoding_record_count !== quantization.records.filter((row) => row.mapping?.family === "format_defined_block_encoding").length
    || quantization.totals.complete_record_count !== quantization.records.filter((row) => row.completeness === "complete_for_serialized_contract").length
    || quantization.totals.partial_record_count !== quantization.records.filter((row) => row.completeness !== "complete_for_serialized_contract").length) {
    throw new Error("Artifact IR quantization classification count conservation failed.");
  }
  const subjects = new Set([...valueIds, ...storageIds]);
  for (const row of quantization.records) {
    if (!subjects.has(row.subject_ref)) throw new Error("Artifact IR quantization subject reference is invalid.");
    for (const storageRef of list(row.related_storage_refs)) if (!storageIds.has(storageRef)) throw new Error("Artifact IR quantization record references an unknown related storage object.");
  }
  if (!value.overlays || !Array.isArray(value.overlays.static) || !Array.isArray(value.overlays.runtime)) throw new Error("Artifact IR overlay ledger is invalid.");
  const overlaySubjects = new Set([...operatorIds, ...architectureIds]);
  const allOverlays = [...value.overlays.static, ...value.overlays.runtime];
  uniqueIds(allOverlays, "overlay");
  for (const overlay of allOverlays) {
    for (const row of list(overlay.rows)) if (!overlaySubjects.has(row.subject_ref)) throw new Error("Artifact IR overlay subject reference is invalid.");
    if (overlay.kind === "runtime_assignment") validateRuntimeReconciliation(overlay, overlaySubjects);
  }
  if (value.completeness?.static_overlay_count !== value.overlays.static.length
    || value.completeness?.runtime_overlay_count !== value.overlays.runtime.length
    || value.completeness?.unknown_is_zero !== false) {
    throw new Error("Artifact IR completeness summary is inconsistent.");
  }
  if (!text(value.interpretation_boundary, 2400)) throw new Error("Artifact IR interpretation boundary is missing.");
}

function primaryScope(analysis, format) {
  const base = { native_index: 0, materialization_status: "materialized", parent_scope_ref: null, invocation_semantics: "model_entrypoint" };
  if (format === "tflite") return { ...base, id: "scope:tflite:subgraph:0", kind: "tflite_subgraph", name: String(analysis?.tflite_subgraph_inventory?.rows?.[0]?.name || "subgraph_0") };
  if (format === "onnx") return { ...base, id: "scope:onnx:main_graph", kind: "onnx_graph", name: String(analysis.graph_name || "main_graph") };
  if (format === "coreml" || format === "mlmodel") return { ...base, id: "scope:coreml:primary", kind: "coreml_serialized_program", name: String(analysis.coreml?.selected_function || "primary") };
  return { ...base, id: `scope:${format}:primary`, kind: `${format}_serialized_program`, name: "primary" };
}

function materializeAdditionalScopes(analysis, format, primaryId) {
  if (format === "tflite") return materializeTfliteScopes(analysis, primaryId);
  if (format === "onnx") return materializeOnnxScopes(analysis, primaryId);
  return { scopes: [], operators: [], values: [], scope_relationships: [] };
}

function materializeTfliteScopes(analysis, primaryId) {
  const rows = list(analysis?.tflite_subgraph_inventory?.rows).filter((row) => Number(row.subgraph_index) !== 0);
  const scopes = [];
  const operators = [];
  const values = [];
  for (const row of rows) {
    const nativeIndex = nonNegativeInteger(row.subgraph_index);
    if (nativeIndex == null) continue;
    const scopeId = `scope:tflite:subgraph:${nativeIndex}`;
    const tensorRows = list(row.tensor_intrinsics);
    const operatorRows = list(row.operator_intrinsics);
    const materialized = tensorRows.length === nonNegativeInteger(row.tensor_count)
      && operatorRows.length === nonNegativeInteger(row.operator_count);
    scopes.push({
      id: scopeId,
      kind: "tflite_subgraph",
      native_index: nativeIndex,
      name: String(row.name || `subgraph_${nativeIndex}`),
      materialization_status: materialized ? "materialized" : "inventory_only",
      parent_scope_ref: null,
      invocation_semantics: String(row.invocation_semantics || "serialized_subgraph_execution_count_not_bound"),
      declared_operator_count: nonNegativeInteger(row.operator_count),
      declared_value_count: nonNegativeInteger(row.tensor_count),
    });
    if (!materialized) continue;
    const producer = new Map();
    const consumers = new Map();
    for (const [position, operator] of operatorRows.entries()) {
      const index = nonNegativeInteger(operator.operator_index) ?? position;
      const operatorRef = operatorId(scopeId, index);
      const inputPorts = tensorPorts(operator.inputs, scopeId);
      const outputPorts = tensorPorts(operator.outputs, scopeId);
      for (const { native_index: tensorIndexValue, port } of outputPorts) {
        if (!producer.has(tensorIndexValue)) producer.set(tensorIndexValue, { operator_ref: operatorRef, port });
      }
      for (const { native_index: tensorIndexValue, port } of inputPorts) {
        if (!consumers.has(tensorIndexValue)) consumers.set(tensorIndexValue, []);
        consumers.get(tensorIndexValue).push({ operator_ref: operatorRef, port });
      }
      operators.push({
        id: operatorRef,
        legacy_graph_node_id: `scope:${nativeIndex}:op:${index}`,
        scope_ref: scopeId,
        native_index: index,
        op_type: String(operator.name || `OP_${index}`),
        name: optionalText(operator.name),
        domain: "tflite",
        version: optionalInteger(operator.version),
        inputs: inputPorts.map(({ port, value_ref }) => ({ port, value_ref })),
        outputs: outputPorts.map(({ port, value_ref }) => ({ port, value_ref })),
        metrics: {
          macs: exactInteger(operator.nominal_macs_decimal ?? operator.nominal_macs),
          mac_assessment_status: String(operator.mac_assessment_status || "not_assessed"),
          logical_io_bytes: exactInteger(operator.logical_io_payload_bytes),
        },
        quantization_summary: { state: "not_projected_from_scope_intrinsics", risk: "not_assessed" },
        topology: { depth: null, role: null, stage: null },
        native_source: { format: "tflite", path: `SubGraph[${nativeIndex}].operators[${index}]`, byte_range: null },
        completeness: "serialized_nested_operator_intrinsic_contract",
      });
    }
    const graphInputs = new Set(integerArray(row.input_tensor_indices).filter((index) => index >= 0));
    const graphOutputs = new Set(integerArray(row.output_tensor_indices).filter((index) => index >= 0));
    for (const [position, tensor] of tensorRows.entries()) {
      const index = nonNegativeInteger(tensor.tensor_index) ?? position;
      const storageBytes = nonNegativeInteger(tensor.buffer_data_length) || 0;
      values.push({
        id: valueId(scopeId, index),
        scope_ref: scopeId,
        native_index: index,
        name: String(tensor.name || `tensor_${index}`),
        value_kind: "tensor",
        dtype: String(tensor.dtype || "UNKNOWN").toUpperCase(),
        shape: dimensions(tensor.shape),
        shape_signature: Array.isArray(tensor.shape_signature) && tensor.shape_signature.length ? dimensions(tensor.shape_signature) : null,
        shape_contract_status: String(tensor.payload_status || "serialized_shape"),
        producer: producer.get(index) || null,
        consumers: uniqueConsumers(consumers.get(index) || []),
        roles: [
          ...(graphInputs.has(index) ? ["graph_input"] : []),
          ...(graphOutputs.has(index) ? ["graph_output"] : []),
          ...(tensor.constant_buffer || storageBytes > 0 ? ["serialized_constant_or_storage"] : []),
        ],
        storage_refs: storageBytes > 0 ? [scopedStorageId(scopeId, index)] : [],
        logical_byte_length: exactInteger(tensor.logical_payload_bytes),
        native_source: { format: "tflite", path: `SubGraph[${nativeIndex}].tensors[${index}]` },
      });
    }
  }
  const scopeIds = new Set(scopes.map((row) => row.id));
  const scopeRelationships = list(analysis?.tflite_subgraph_inventory?.references).map((row, index) => {
    const sourceScopeRef = `scope:tflite:subgraph:${row.source_subgraph_index}`;
    const targetScopeRef = `scope:tflite:subgraph:${row.target_subgraph_index}`;
    return {
      id: `scope-relationship:tflite:${index}`,
      relation: "serialized_subgraph_reference",
      source_scope_ref: sourceScopeRef,
      source_operator_ref: operatorId(sourceScopeRef, row.source_op_index),
      target_scope_ref: targetScopeRef,
      role: String(row.role || "subgraph_reference"),
    };
  }).filter((row) => (row.source_scope_ref === primaryId || scopeIds.has(row.source_scope_ref)) && scopeIds.has(row.target_scope_ref));
  return { scopes, operators, values, scope_relationships: scopeRelationships };
}

function materializeOnnxScopes(analysis, primaryId) {
  const ledger = list(analysis?.onnx_domain_analysis?.scope_ledger);
  const nested = ledger.filter((row) => String(row.scope || "main_graph") !== "main_graph")
    .sort((left, right) => String(left.scope).localeCompare(String(right.scope)));
  const scopeIdByName = new Map([["main_graph", primaryId]]);
  nested.forEach((row, index) => scopeIdByName.set(String(row.scope), `scope:onnx:nested:${index}`));
  const scopes = nested.map((row, index) => ({
    id: scopeIdByName.get(String(row.scope)),
    kind: String(row.scope_class || "nested_graph"),
    native_index: index,
    name: String(row.name || row.scope),
    materialization_status: Array.isArray(row.nodes) && Array.isArray(row.values) ? "materialized" : "inventory_only",
    parent_scope_ref: scopeIdByName.get(String(row.parent_scope || "main_graph")) || primaryId,
    invocation_semantics: String(row.invocation_semantics || (row.scope_class === "local_function_body" ? "model_local_function_definition" : "serialized_nested_graph")),
    declared_operator_count: nonNegativeInteger(row.node_count),
    declared_value_count: nonNegativeInteger(row.value_count),
  }));
  const operators = [];
  const values = [];
  const scopeRelationships = [];
  for (const [scopePosition, row] of nested.entries()) {
    const scopeId = scopeIdByName.get(String(row.scope));
    if (!Array.isArray(row.nodes) || !Array.isArray(row.values)) continue;
    const producer = new Map();
    const consumers = new Map();
    for (const [position, node] of row.nodes.entries()) {
      const index = nonNegativeInteger(node.node_index) ?? position;
      const operatorRef = operatorId(scopeId, index);
      const inputPorts = namedValuePorts(node.inputs, scopeId, row.values);
      const outputPorts = namedValuePorts(node.outputs, scopeId, row.values);
      for (const { native_index: valueIndex, port } of outputPorts) if (!producer.has(valueIndex)) producer.set(valueIndex, { operator_ref: operatorRef, port });
      for (const { native_index: valueIndex, port } of inputPorts) {
        if (!consumers.has(valueIndex)) consumers.set(valueIndex, []);
        consumers.get(valueIndex).push({ operator_ref: operatorRef, port });
      }
      operators.push({
        id: operatorRef,
        legacy_graph_node_id: `scope:${scopePosition + 1}:op:${index}`,
        scope_ref: scopeId,
        native_index: index,
        op_type: String(node.op_name || `OP_${index}`),
        name: optionalText(node.node_name || node.op_name),
        domain: String(node.domain || "ai.onnx"),
        version: optionalInteger(node.imported_opset),
        inputs: inputPorts.map(({ port, value_ref }) => ({ port, value_ref })),
        outputs: outputPorts.map(({ port, value_ref }) => ({ port, value_ref })),
        metrics: { macs: null, mac_assessment_status: "not_assessed_scope_local_cost_not_projected", logical_io_bytes: null },
        quantization_summary: { state: "not_projected_from_nested_scope", risk: "not_assessed" },
        topology: { depth: null, role: null, stage: null },
        native_source: { format: "onnx", path: String(node.native_path || `${row.scope}.node[${index}]`), byte_range: null },
        completeness: "serialized_nested_operator_and_port_identity",
      });
    }
    for (const [position, value] of row.values.entries()) {
      const index = nonNegativeInteger(value.value_index) ?? position;
      values.push({
        id: valueId(scopeId, index),
        scope_ref: scopeId,
        native_index: index,
        name: String(value.name || `value_${index}`),
        value_kind: String(value.value_kind || "tensor"),
        dtype: String(value.dtype || "UNKNOWN").toUpperCase(),
        shape: dimensions(value.shape),
        shape_signature: null,
        shape_contract_status: String(value.contract_status || "serialized_name_only"),
        producer: producer.get(index) || null,
        consumers: uniqueConsumers(consumers.get(index) || []),
        roles: compactStrings(value.roles),
        storage_refs: [],
        logical_byte_length: null,
        native_source: { format: "onnx", path: String(value.native_path || `${row.scope}.value[${JSON.stringify(String(value.name || ""))}]`) },
      });
    }
    const parent = scopeIdByName.get(String(row.parent_scope || "main_graph")) || primaryId;
    const ownerNodeIndex = nonNegativeInteger(row.owner_node_index);
    scopeRelationships.push({
      id: `scope-relationship:onnx:${scopePosition}`,
      relation: row.scope_class === "local_function_body" ? "serialized_local_function_definition" : "serialized_nested_graph_ownership",
      source_scope_ref: parent,
      source_operator_ref: ownerNodeIndex == null ? null : operatorId(parent, ownerNodeIndex),
      target_scope_ref: scopeId,
      role: String(row.owner_role || row.scope_class || "nested_scope"),
    });
  }
  return { scopes, operators, values, scope_relationships: scopeRelationships };
}

function graphMacAssessment(analysis, operators) {
  const ledger = analysis?.mac_assessment;
  const compute = nonNegativeInteger(ledger?.compute_ops);
  const assessedCount = nonNegativeInteger(ledger?.assessed_compute_ops);
  const unassessedCount = nonNegativeInteger(ledger?.not_assessed_compute_ops);
  const assessed = exactInteger(ledger?.total_assessed_macs_decimal ?? ledger?.total_assessed_macs);
  if (compute != null && assessedCount != null && unassessedCount != null && assessedCount + unassessedCount === compute && assessed) {
    return {
      total: unassessedCount === 0 ? assessed : null,
      assessed,
      assessment: { status: unassessedCount === 0 ? "complete" : assessedCount ? "partial" : "not_assessed", compute_op_count: compute, assessed_compute_op_count: assessedCount, unassessed_compute_op_count: unassessedCount, scope: String(ledger.metric_scope || "nominal tensor-contraction MACs") },
    };
  }
  const sum = operators.reduce((total, row) => total + BigInt(row.metrics?.macs?.decimal || "0"), 0n);
  return { total: exact(sum), assessed: exact(sum), assessment: { status: "complete_without_separate_compute_denominator", compute_op_count: null, assessed_compute_op_count: null, unassessed_compute_op_count: null, scope: "sum_of_operator_macs_present_in_artifact_ir" } };
}

function validateRuntimeReconciliation(overlay, validSubjects) {
  const runtimeIds = new Set();
  let flattenedReferenceCount = 0;
  for (const node of list(overlay.runtime_nodes)) {
    if (!text(node.runtime_node_ref, 1000) || runtimeIds.has(node.runtime_node_ref)) throw new Error("Artifact IR runtime node identity is invalid or duplicated.");
    runtimeIds.add(node.runtime_node_ref);
    const references = compactStrings(node.source_subject_refs);
    if (references.some((subjectRef) => !validSubjects.has(subjectRef))) throw new Error("Artifact IR runtime node references an unknown canonical subject.");
    if ((references.length === 0 && node.mapping_cardinality !== "unmapped")
      || (references.length === 1 && node.mapping_cardinality !== "one_to_one")
      || (references.length > 1 && node.mapping_cardinality !== "fused_one_to_many")) {
      throw new Error("Artifact IR runtime mapping cardinality is inconsistent.");
    }
    flattenedReferenceCount += references.length;
  }
  if (overlay.summary?.runtime_node_count !== runtimeIds.size
    || overlay.summary?.source_subject_reference_count !== flattenedReferenceCount
    || overlay.rows.length !== flattenedReferenceCount) {
    throw new Error("Artifact IR runtime reconciliation count conservation failed.");
  }
  const distinctSubjects = new Set(overlay.rows.map((row) => row.subject_ref));
  if (overlay.summary?.distinct_source_subject_count !== distinctSubjects.size
    || overlay.summary?.mapped_runtime_node_count !== list(overlay.runtime_nodes).filter((row) => list(row.source_subject_refs).length > 0).length
    || overlay.summary?.unmapped_runtime_node_count !== list(overlay.runtime_nodes).filter((row) => list(row.source_subject_refs).length === 0).length
    || overlay.summary?.name_similarity_mapping_used !== false) {
    throw new Error("Artifact IR runtime reconciliation summary is inconsistent.");
  }
  for (const row of overlay.rows) if (!runtimeIds.has(row.runtime_node_ref)) throw new Error("Artifact IR runtime assignment references an unknown runtime node.");
}

function completeness(graph, storage, architecture, quantization, overlays) {
  return {
    graph: graph.completeness,
    storage: storage.status,
    architecture: architecture.status,
    quantization: quantization.status,
    static_overlay_count: overlays.static.length,
    runtime_overlay_count: overlays.runtime.length,
    unknown_is_zero: false,
  };
}

function artifactIdentity(analysis, artifact, format) {
  const sha256 = normalizeSha256(artifact.sha256 || analysis.model_sha256 || analysis.artifact_sha256);
  if (!sha256) throw new Error("Artifact IR requires an artifact SHA-256.");
  const size = nonNegativeInteger(artifact.size ?? artifact.byte_length ?? analysis.file_size_bytes ?? analysis.file_size);
  return {
    filename: String(artifact.filename || analysis.filename || `model.${format}`),
    format,
    sha256,
    byte_length: size == null ? null : exact(BigInt(size)),
    artifact_set_sha256: normalizeSha256(artifact.artifact_set_sha256 || analysis?.artifact_set?.artifact_set_sha256),
  };
}

function nativeOperatorLocator(format, scope, op) {
  if (format === "tflite") return { format, path: `SubGraph[${scope.native_index}].operators[${op.index}]`, byte_range: null };
  if (format === "onnx") return { format, path: `ModelProto.graph.node[${op.index}]`, byte_range: null };
  if (format === "coreml" || format === "mlmodel") return { format: "coreml", path: `Model.primary.operations[${op.index}]`, byte_range: null };
  if (["executorch", "pte", "ptd"].includes(format)) return { format: "executorch", path: `Program.execution_plan[0].chains.instructions[${op.index}]`, byte_range: null };
  return { format, path: `operators[${op.index}]`, byte_range: null };
}

function nativeValueLocator(format, scope, tensor, index) {
  if (format === "tflite") return { format, path: `SubGraph[${scope.native_index}].tensors[${index}]` };
  if (format === "onnx") return { format, path: `ModelProto.graph.value[name=${JSON.stringify(String(tensor.name || ""))}]` };
  return { format, path: `values[${index}]` };
}

function nativeStorageLocator(format, tensor, index) {
  if (format === "gguf") return { format, path: `tensor_infos[${index}]`, payload_offset_basis: "tensor_data_section" };
  if (format === "safetensors") return { format, path: `header[${JSON.stringify(String(tensor.name || ""))}]`, payload_offset_basis: "data_section" };
  if (format === "tflite") return { format, path: `SubGraph[0].tensors[${index}].buffer` };
  if (format === "onnx") return { format, path: `ModelProto.graph.initializer[name=${JSON.stringify(String(tensor.name || ""))}]` };
  return { format, path: `serialized_parameters[${index}]` };
}

function nativeQuantizationLocator(format, tensor, index) {
  if (format === "tflite") return { format, path: `SubGraph[0].tensors[${index}].quantization` };
  if (format === "onnx") return { format, path: `graph.value[${JSON.stringify(String(tensor.name || ""))}].quantization_bindings` };
  if (format === "gguf") return { format, path: `tensor_infos[${index}].ggml_type` };
  return { format, path: `tensors[${index}].quantization` };
}

function storageEncoding(format, tensor) {
  if (format === "gguf") return { family: "ggml_block_encoding", name: String(tensor.dtype || "UNKNOWN"), block_elements: positiveInteger(tensor.block_elements), block_bytes: positiveInteger(tensor.block_bytes) };
  return { family: "scalar_or_declared_tensor_encoding", name: String(tensor.dtype || "UNKNOWN").toUpperCase(), block_elements: null, block_bytes: null };
}

function quantizationParameterization(tensor, scales) {
  const declared = String(tensor.quantization_parameterization || tensor.scale_mode || "").toLowerCase().replaceAll("-", "_");
  const blockSize = positiveInteger(tensor.quantization_block_size);
  if (blockSize || declared.includes("block")) return { kind: "per_block", axes: optionalAxis(tensor), block_size: blockSize };
  if (scales.length > 1 || declared.includes("axis") || declared.includes("channel")) return { kind: "per_axis", axes: optionalAxis(tensor), block_size: null };
  return { kind: "per_tensor", axes: [], block_size: null };
}

function optionalAxis(tensor) {
  const value = optionalInteger(tensor.quantized_dimension);
  return value == null ? [] : [value];
}

function parameterVector(values) {
  const normalized = list(values);
  return {
    count: normalized.length,
    sha256: sha256TextHex(canonicalJson(normalized)),
    inline_values: normalized.length <= INLINE_PARAMETER_LIMIT ? normalized : null,
    inline_status: normalized.length <= INLINE_PARAMETER_LIMIT ? "complete" : "digest_only_large_vector",
  };
}

function digestDescriptor(count, digest) {
  const normalizedCount = nonNegativeInteger(count) || 0;
  return {
    count: normalizedCount,
    sha256: normalizeSha256(digest),
    inline_values: null,
    inline_status: normalizedCount ? "digest_only_native_parameter_vector" : "not_applicable",
  };
}

function decimalCount(value) {
  if (typeof value === "string" && /^\d+$/.test(value)) return value;
  const number = nonNegativeInteger(value);
  return number == null ? null : String(number);
}

function zeroPointConstraint(values) {
  if (!values.length) return "not_encoded";
  return values.every((value) => value === 0) ? "observed_all_zero_not_symmetry_proof" : "observed_contains_nonzero";
}

function codeDomain(dtype) {
  const key = String(dtype || "").toUpperCase();
  const domains = { INT8: [-128, 127], UINT8: [0, 255], INT16: [-32768, 32767], UINT16: [0, 65535], INT4: [-8, 7], UINT4: [0, 15] };
  const row = domains[key];
  return row ? { code_min: row[0], code_max: row[1], code_domain_status: "declared_storage_dtype" } : { code_min: null, code_max: null, code_domain_status: "format_defined_or_not_integer" };
}

function namespaceGroups(tensors, format) {
  const groups = new Map();
  for (const [position, tensor] of tensors.entries()) {
    const label = String(tensor.name || "unnamed").split(".").slice(0, 2).join(".") || "unnamed";
    if (!groups.has(label)) groups.set(label, { count: 0, bytes: 0n, members: [] });
    const group = groups.get(label);
    group.count += 1;
    group.bytes += BigInt(positiveStorageBytes(tensor, format));
    group.members.push(storageId(tensorIndex(tensor, position)));
  }
  return [...groups.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([label, group], index) => ({
    id: `architecture:namespace:${index}`, kind: "storage_namespace", native_index: index, label,
    tensor_count: group.count, serialized_bytes: exact(group.bytes), storage_object_refs: group.members,
  }));
}

function selectLlmPlacementScenario(binding) {
  const rows = list(binding?.llm_accelerator_residency?.scenarios);
  const selected = rows.filter((row) => row.scenario_source === "cli_declared");
  return selected.length === 1 && selected[0]?.serialized_layer_offload ? selected[0] : null;
}

function logicalTensorBytes(tensor) {
  return exactInteger(tensor.logical_payload_bytes ?? tensor.byte_length ?? tensor.buffer_data_length ?? tensor.initializer_available_bytes);
}

function positiveStorageBytes(tensor, format = "") {
  const candidates = [
    ...(["gguf", "safetensors"].includes(format) ? [tensor.byte_length] : []),
    tensor.buffer_data_length,
    tensor.initializer_available_bytes,
    tensor.initializer_bytes,
    tensor.serialized_payload_bytes,
  ];
  for (const value of candidates) {
    const number = nonNegativeInteger(value);
    if (number && number > 0) return number;
  }
  return 0;
}

function safeExactSum(start, decimalLength) {
  const result = BigInt(start) + BigInt(decimalLength);
  return result <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(result) : result.toString();
}

function uniqueConsumers(rows) {
  const values = new Map();
  for (const row of rows) values.set(`${row.operator_ref}:${row.port}`, row);
  return [...values.values()].sort((left, right) => left.operator_ref.localeCompare(right.operator_ref) || left.port - right.port);
}

function tensorPorts(values, scopeId) {
  return list(values).map((value, port) => ({ native_index: Number(value), port }))
    .filter((row) => Number.isSafeInteger(row.native_index) && row.native_index >= 0)
    .map((row) => ({ ...row, value_ref: valueId(scopeId, row.native_index) }));
}

function namedValuePorts(names, scopeId, values) {
  const indexByName = new Map(list(values).map((row, index) => [String(row.name || ""), nonNegativeInteger(row.value_index) ?? index]));
  return list(names).map((name, port) => ({ native_index: indexByName.get(String(name || "")), port }))
    .filter((row) => row.native_index != null)
    .map((row) => ({ ...row, value_ref: valueId(scopeId, row.native_index) }));
}

function uniqueIds(rows, label) {
  const ids = new Set();
  for (const row of list(rows)) {
    if (!text(row?.id, 600) || ids.has(row.id)) throw new Error(`Artifact IR ${label} identity is invalid or duplicated.`);
    ids.add(row.id);
  }
  return ids;
}

function canonicalOps(value) {
  return list(value).map((op, position) => ({ ...op, index: Number.isSafeInteger(Number(op?.index)) ? Number(op.index) : position }))
    .sort((left, right) => left.index - right.index);
}

function operatorId(scopeId, index) { return `operator:${scopeId}:${index}`; }
function valueId(scopeId, index) { return `value:${scopeId}:${index}`; }
function storageId(index) { return `storage:tensor:${index}`; }
function scopedStorageId(scopeId, index) { return `storage:${scopeId}:tensor:${index}`; }
function architectureLayerId(index) { return `architecture:layer:${index}`; }
function tensorIndex(tensor, fallback) { return Number.isSafeInteger(Number(tensor?.index)) ? Number(tensor.index) : fallback; }
function normalizeFormat(value) { return String(value || "unknown").trim().toLowerCase().replace(".mlmodel", "coreml"); }
function normalizeSha256(value) { const normalized = String(value || "").trim().toLowerCase(); return SHA256.test(normalized) ? normalized : null; }
function list(value) { return Array.isArray(value) ? value : []; }
function clone(value) { return value == null ? value : JSON.parse(JSON.stringify(value)); }
function dimensions(value) { return list(value).map((item) => Number.isSafeInteger(Number(item)) ? Number(item) : String(item)); }
function integerArray(value) { return list(value).map(Number).filter(Number.isSafeInteger); }
function finiteNumberArray(value) { return list(value).map(Number).filter(Number.isFinite); }
function compactStrings(value) { return list(Array.isArray(value) ? value : [value]).map((item) => String(item || "").trim()).filter(Boolean); }
function optionalText(value) { const normalized = String(value ?? "").trim(); return normalized || null; }
function text(value, maximum) { const normalized = String(value ?? "").trim(); return normalized.length > 0 && normalized.length <= maximum; }
function nonNegativeInteger(value) { const number = Number(value); return Number.isSafeInteger(number) && number >= 0 ? number : null; }
function positiveInteger(value) { const number = Number(value); return Number.isSafeInteger(number) && number > 0 ? number : null; }
function optionalInteger(value) { if (value === null || value === undefined || value === "") return null; const number = Number(value); return Number.isSafeInteger(number) ? number : null; }
function exactInteger(value) {
  if (typeof value === "string" && /^\d+$/.test(value)) return exact(BigInt(value));
  if (typeof value === "bigint" && value >= 0n) return exact(value);
  if (Number.isSafeInteger(Number(value)) && Number(value) >= 0) return exact(BigInt(Number(value)));
  return null;
}
function exact(value) { return { decimal: value.toString(), number: value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null }; }
function deepFreeze(value) { if (!value || typeof value !== "object" || Object.isFrozen(value)) return value; Object.freeze(value); for (const child of Object.values(value)) deepFreeze(child); return value; }
