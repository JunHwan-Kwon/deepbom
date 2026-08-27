const TFLITE_RUNTIME_INFO_ADAPTER_SCHEMA = "deepbom.tflite_runtime_info_adapter.v1";
const RUNTIME_ASSIGNMENT_SCHEMA = "deepbom.runtime_assignment.v1.9";
const MAX_PROTO_BYTES = 32 * 1024 * 1024;
const MAX_REPEATED_ITEMS = 1_000_000;

export const TFLITE_RUNTIME_INFO_SOURCE = Object.freeze({
  runtime_version_basis: "TensorFlow Lite at tensorflow/tensorflow@87bbf65b8d23d3f06912b1b2183587e1884bc45c",
  source_commit: "tensorflow/tensorflow@87bbf65b8d23d3f06912b1b2183587e1884bc45c",
  source_file: "tensorflow/lite/profiling/model_runtime_info.cc",
  source_ref: "https://github.com/tensorflow/tensorflow/blob/87bbf65b8d23d3f06912b1b2183587e1884bc45c/tensorflow/lite/profiling/model_runtime_info.cc",
  source_sha256: "9a6edf838fe149c54efe0700bcdc2faf58dd5343f1370538e91bf5ed8a0e11b6",
  proto_file: "tensorflow/lite/profiling/proto/model_runtime_info.proto",
  proto_ref: "https://github.com/tensorflow/tensorflow/blob/87bbf65b8d23d3f06912b1b2183587e1884bc45c/tensorflow/lite/profiling/proto/model_runtime_info.proto",
  proto_sha256: "7829c3163339ce7dea01091a5154a06cb302f35573ad51d3af06a2a09b95a8fb",
  delegation_metadata_file: "tensorflow/lite/optional_debug_tools.cc",
  delegation_metadata_ref: "https://github.com/tensorflow/tensorflow/blob/87bbf65b8d23d3f06912b1b2183587e1884bc45c/tensorflow/lite/optional_debug_tools.cc",
  delegation_metadata_sha256: "607010c8f7aba721bd5d96f45cb08c55226a800e55946da708368d09d4545260",
  export_driver_file: "tensorflow/lite/tools/benchmark/benchmark_tflite_model.cc",
  export_driver_ref: "https://github.com/tensorflow/tensorflow/blob/87bbf65b8d23d3f06912b1b2183587e1884bc45c/tensorflow/lite/tools/benchmark/benchmark_tflite_model.cc",
  export_driver_sha256: "e26a3e4e26300442b47d27e1b515ccbc85b34425625d60f637faa28973f7a8f7",
  export_flag: "benchmark_model --export_model_runtime_info=true --model_runtime_info_output_file=<path>",
});

export function parseTfliteRuntimeInfoSource(input, analysis) {
  if (String(analysis?.format || "tflite").toLowerCase() !== "tflite") {
    throw new Error("TFLite ModelRuntimeDetails evidence requires an active TFLite artifact.");
  }
  const bytes = toProtoBytes(input, "TFLite ModelRuntimeDetails evidence");
  if (!bytes.length) throw new Error("TFLite ModelRuntimeDetails protobuf is empty.");
  if (bytes.length > MAX_PROTO_BYTES) throw new Error("TFLite ModelRuntimeDetails protobuf must be 32 MiB or smaller.");
  const model = parseModel(new ProtoReader(bytes, "ModelRuntimeDetails"));
  const profile = bindPrimarySubgraph(model, analysis);
  return {
    kind: "tflite_model_runtime_info",
    schema: TFLITE_RUNTIME_INFO_ADAPTER_SCHEMA,
    source_byte_length: bytes.length,
    model_name: model.model_name,
    providers: profile.providers,
    assignments: profile.assignments,
    partitions: profile.partitions,
    topology_binding: profile.topology_binding,
    original_node_count: profile.original_node_count,
    delegate_node_count: profile.delegate_node_count,
    execution_plan_node_count: profile.execution_plan_node_count,
    delegated_op_count: profile.delegated_op_count,
    nondelegated_op_count: profile.nondelegated_op_count,
    source_basis: TFLITE_RUNTIME_INFO_SOURCE,
  };
}

export function previewTfliteRuntimeInfoMapping(profile, analysis) {
  if (profile?.kind !== "tflite_model_runtime_info") throw new Error("Expected parsed TFLite ModelRuntimeDetails evidence.");
  const graphOps = analysis?.ops || [];
  if (profile.assignments.length !== graphOps.length || profile.topology_binding?.matched_original_op_count !== graphOps.length) {
    throw new Error("TFLite runtime plan no longer matches the active graph.");
  }
  return {
    assignment_count: profile.assignments.length,
    graph_op_count: graphOps.length,
    delegate_partition_count: profile.partitions.length,
    delegated_op_count: profile.delegated_op_count,
    execution_plan_node_count: profile.execution_plan_node_count,
    topology_match_count: profile.topology_binding.matched_original_op_count,
    providers: profile.providers,
  };
}

export function buildTfliteRuntimeAssignmentDocument(profile, analysis, metadata = {}) {
  const preview = previewTfliteRuntimeInfoMapping(profile, analysis);
  const profileSha256 = requiredSha(metadata.profileSha256, "profileSha256");
  const runtimeVersion = requiredText(metadata.runtimeVersion, "runtimeVersion", 80);
  const runtimeBuild = requiredText(metadata.runtimeBuild, "runtimeBuild", 500);
  const collectedAt = requiredText(metadata.collectedAt, "collectedAt", 64);
  const captureId = requiredText(metadata.captureId, "captureId", 160);
  const backend = profile.providers.join(" + ");
  return {
    schema: RUNTIME_ASSIGNMENT_SCHEMA,
    artifact_sha256: String(analysis?.model_sha256 || "").toLowerCase(),
    target_profile_id: String(analysis?.target_profile?.id || ""),
    target_profile_sha256: String(analysis?.target_profile?.profile_sha256 || "").toLowerCase(),
    runtime: {
      name: "TensorFlow Lite / LiteRT",
      version: runtimeVersion,
      backend,
      build: runtimeBuild,
      binary_sha256: metadata.binarySha256 || null,
      graph_optimization_level: null,
      execution_mode: null,
    },
    source: {
      kind: "tflite_model_runtime_info_proto_adapter",
      collected_at: collectedAt,
      capture_id: captureId,
      capture_binding_semantics: "DECLARED_BENCHMARK_INVOCATION_IDENTIFIER",
      assignment_semantics: "original_graph_op_assignment",
      partition_semantics: "partition_id_identifies_runtime_partition_when_present",
      duration_semantics: "not_collected",
      duration_statistic: null,
      profile_sha256: profileSha256,
      adapter: {
        schema: TFLITE_RUNTIME_INFO_ADAPTER_SCHEMA,
        ...TFLITE_RUNTIME_INFO_SOURCE,
        source_byte_length: profile.source_byte_length,
        source_model_name: profile.model_name,
        subgraph_id: 0,
        original_node_count: profile.original_node_count,
        delegate_node_count: profile.delegate_node_count,
        execution_plan_node_count: profile.execution_plan_node_count,
        delegated_op_count: profile.delegated_op_count,
        nondelegated_op_count: profile.nondelegated_op_count,
        mapping_method_counts: countBy(profile.assignments, (item) => item.mapping_method),
        mapping_coverage_ratio: preview.assignment_count / Math.max(1, preview.graph_op_count),
        topology_binding: profile.topology_binding,
        partitions: profile.partitions,
        artifact_binding: "active_artifact_exact_original_op_topology",
        source_artifact_sha256_embedded: false,
        runtime_identity_semantics: "DECLARED",
        assignment_evidence_class: "OBSERVED_RUNTIME",
        interpretation_boundary: "The exported proto directly observes the TFLite execution plan, delegate node names, and exact replaced original node IDs. The proto does not embed artifact SHA-256, runtime version/build, collection time, capture ID, timings, tensor-copy materialization, or executed microkernel symbols; DeepBOM binds it to the active artifact only after exact original-op name and input/output tensor-ID comparison, while capture identity remains declared.",
      },
    },
    assignments: profile.assignments,
  };
}

function bindPrimarySubgraph(model, analysis) {
  const primary = model.subgraphs.find((subgraph) => subgraph.id === 0);
  if (!primary) throw new Error("TFLite ModelRuntimeDetails has no primary subgraph with id 0.");
  if (primary.subgraph_type !== 1) throw new Error("TFLite ModelRuntimeDetails primary subgraph is not TFLITE_SUBGRAPH.");
  const ops = [...(analysis?.ops || [])].sort((left, right) => Number(left.index) - Number(right.index));
  if (!ops.length) throw new Error("The active TFLite graph has no parsed ops to bind.");
  if (ops.some((op, index) => Number(op.index) !== index)) throw new Error("The active TFLite graph op indices are not contiguous from zero.");
  const nodeById = uniqueById(primary.nodes, "runtime node");
  const executionIds = uniqueIntegers(primary.execution_plan, "execution-plan node id");
  for (const id of executionIds) if (!nodeById.has(id)) throw new Error(`TFLite runtime execution plan references unknown node ${id}.`);

  const originalNodes = [];
  for (const op of ops) {
    const node = nodeById.get(Number(op.index));
    if (!node || node.delegate_details) throw new Error(`TFLite runtime plan is missing original op #${op.index}.`);
    if (node.name !== String(op.name || "")) throw new Error(`TFLite runtime op name mismatch at #${op.index}: expected ${op.name}, observed ${node.name || "<empty>"}.`);
    if (!sameIntegerArray(node.inputs, op.inputs || [])) throw new Error(`TFLite runtime input tensor IDs do not match original op #${op.index}.`);
    if (!sameIntegerArray(node.outputs, op.outputs || [])) throw new Error(`TFLite runtime output tensor IDs do not match original op #${op.index}.`);
    originalNodes.push(node);
  }

  const delegateNodes = primary.nodes.filter((node) => node.delegate_details != null);
  for (const node of primary.nodes) {
    if (node.id < ops.length && !originalNodes.includes(node)) throw new Error(`TFLite runtime node id ${node.id} collides with an original op id.`);
    if (node.id >= ops.length && !node.delegate_details) throw new Error(`TFLite runtime node ${node.id} is neither an original op nor a delegate node.`);
  }
  const executionSet = new Set(executionIds);
  const replacedOwner = new Map();
  const partitions = delegateNodes.map((node) => {
    if (node.delegated_to_node_id != null) throw new Error(`Delegate node ${node.id} cannot itself be delegated.`);
    const details = node.delegate_details;
    if (!details.delegate_name) throw new Error(`Delegate node ${node.id} has no delegate_name.`);
    if (!details.replaced_ids.length) throw new Error(`Delegate node ${node.id} replaces no original nodes.`);
    if (!executionSet.has(node.id)) throw new Error(`Delegate node ${node.id} is absent from the execution plan.`);
    const replacedIds = uniqueIntegers(details.replaced_ids, `delegate node ${node.id} replaced op id`).sort((a, b) => a - b);
    for (const originalId of replacedIds) {
      if (originalId < 0 || originalId >= ops.length) throw new Error(`Delegate node ${node.id} replaces unknown original op ${originalId}.`);
      if (replacedOwner.has(originalId)) throw new Error(`Original op ${originalId} is listed by multiple delegate nodes.`);
      replacedOwner.set(originalId, node.id);
      const original = nodeById.get(originalId);
      if (original?.delegated_to_node_id !== node.id) throw new Error(`Original op ${originalId} and delegate node ${node.id} do not have a symmetric delegation mapping.`);
    }
    return {
      partition_id: `subgraph:0/delegate_node:${node.id}`,
      delegate_node_id: node.id,
      delegate_name: details.delegate_name,
      runtime_node_name: node.name,
      replaced_op_ids: replacedIds,
    };
  }).sort((left, right) => left.delegate_node_id - right.delegate_node_id);

  const assignments = originalNodes.map((node) => {
    const delegated = node.delegated_to_node_id != null;
    if (delegated) {
      if (executionSet.has(node.id)) throw new Error(`Delegated original op ${node.id} must not remain in the execution plan.`);
      const owner = replacedOwner.get(node.id);
      if (owner !== node.delegated_to_node_id) throw new Error(`Delegated original op ${node.id} has no symmetric delegate-node owner.`);
      const partition = partitions.find((item) => item.delegate_node_id === owner);
      return {
        op_index: node.id,
        op_name: node.name,
        provider: partition.delegate_name,
        delegated: true,
        partition_id: partition.partition_id,
        kernel: null,
        duration_us: null,
        duration_sum_us: null,
        sample_count: null,
        mapping_method: "runtime_info_original_node_id_and_symmetric_delegate_map",
        runtime_node_index: partition.delegate_node_id,
        runtime_node_name: partition.runtime_node_name || partition.delegate_name,
        graph_node_name: null,
      };
    }
    if (!executionSet.has(node.id)) throw new Error(`Non-delegated original op ${node.id} is absent from the execution plan.`);
    return {
      op_index: node.id,
      op_name: node.name,
      provider: "TFLite non-delegated kernel",
      delegated: false,
      partition_id: null,
      kernel: null,
      duration_us: null,
      duration_sum_us: null,
      sample_count: null,
      mapping_method: "runtime_info_original_node_id_execution_plan",
      runtime_node_index: node.id,
      runtime_node_name: node.name,
      graph_node_name: null,
    };
  });
  if (executionIds.length !== delegateNodes.length + assignments.filter((item) => !item.delegated).length) {
    throw new Error("TFLite runtime execution plan contains nodes outside the validated delegate/original assignment set.");
  }
  const providers = [...new Set(assignments.map((item) => item.provider))].sort();
  return {
    providers,
    assignments,
    partitions,
    original_node_count: originalNodes.length,
    delegate_node_count: delegateNodes.length,
    execution_plan_node_count: executionIds.length,
    delegated_op_count: assignments.filter((item) => item.delegated).length,
    nondelegated_op_count: assignments.filter((item) => !item.delegated).length,
    topology_binding: {
      method: "exact_original_op_id_name_and_input_output_tensor_ids",
      matched_original_op_count: originalNodes.length,
      graph_op_count: ops.length,
      input_output_tensor_id_arrays_matched: true,
      source_artifact_sha256_embedded: false,
    },
  };
}

function parseModel(reader) {
  const result = { model_name: null, subgraphs: [] };
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) result.model_name = reader.stringField(wire, singular, field, "model_name");
    else if (field === 2) result.subgraphs.push(parseSubgraph(reader.message(wire, "RuntimeSubgraph")));
    else reader.skip(wire);
    guardLength(result.subgraphs, "subgraphs");
  }
  if (!result.subgraphs.length) throw new Error("TFLite ModelRuntimeDetails contains no subgraphs.");
  uniqueById(result.subgraphs, "runtime subgraph");
  return result;
}

function parseSubgraph(reader) {
  const result = { id: null, nodes: [], execution_plan: [], subgraph_type: null, name: null };
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) result.id = reader.intField(wire, singular, field, "subgraph_id");
    else if (field === 2) reader.skip(wire);
    else if (field === 3) result.nodes.push(parseNode(reader.message(wire, "Node")));
    else if (field === 4) result.execution_plan.push(...reader.repeatedInt(wire, "execution_plan"));
    else if (field === 5) result.subgraph_type = reader.intField(wire, singular, field, "subgraph_type");
    else if (field === 6) result.name = reader.stringField(wire, singular, field, "subgraph_name");
    else reader.skip(wire);
    guardLength(result.nodes, "nodes");
    guardLength(result.execution_plan, "execution_plan");
  }
  if (result.id == null) throw new Error("TFLite RuntimeSubgraph is missing subgraph_id.");
  if (result.subgraph_type == null) throw new Error(`TFLite RuntimeSubgraph ${result.id} is missing subgraph_type.`);
  uniqueById(result.nodes, `subgraph ${result.id} node`);
  return result;
}

function parseNode(reader) {
  const result = { id: null, name: null, type: null, inputs: [], outputs: [], delegated_to_node_id: null, delegate_details: null };
  const singular = new Set();
  let oneof = null;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) result.id = reader.intField(wire, singular, field, "node.id");
    else if (field === 2) result.name = reader.stringField(wire, singular, field, "node.name");
    else if (field === 3) result.type = reader.stringField(wire, singular, field, "node.type");
    else if (field === 4) result.inputs.push(...reader.repeatedInt(wire, "node.inputs", true));
    else if (field === 5) result.outputs.push(...reader.repeatedInt(wire, "node.outputs", true));
    else if (field === 6 || field === 7 || field === 10) reader.skip(wire);
    else if (field === 8) {
      if (oneof != null) throw new Error("TFLite runtime node contains duplicate or conflicting node_info fields.");
      oneof = field;
      result.delegate_details = parseDelegateDetails(reader.message(wire, "DelegateNodeDetails"));
    } else if (field === 9) {
      if (oneof != null) throw new Error("TFLite runtime node contains duplicate or conflicting node_info fields.");
      oneof = field;
      result.delegated_to_node_id = reader.intField(wire, singular, field, "node.delegated_to_node_id");
    } else reader.skip(wire);
    guardLength(result.inputs, "node.inputs");
    guardLength(result.outputs, "node.outputs");
  }
  if (result.id == null || result.name == null || result.type == null) throw new Error("TFLite runtime node is missing id, name, or type.");
  return result;
}

function parseDelegateDetails(reader) {
  const result = { delegate_name: null, replaced_ids: [] };
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) result.delegate_name = reader.stringField(wire, singular, field, "delegate_name");
    else if (field === 2) result.replaced_ids.push(...reader.repeatedInt(wire, "tflite_node_ids_replaced"));
    else reader.skip(wire);
    guardLength(result.replaced_ids, "tflite_node_ids_replaced");
  }
  return result;
}

export class ProtoReader {
  constructor(bytes, label) {
    this.bytes = bytes;
    this.position = 0;
    this.label = label;
  }

  get done() { return this.position === this.bytes.length; }

  key() {
    const key = this.varint();
    const field = Math.floor(key / 8);
    const wire = key % 8;
    if (field <= 0) throw new Error(`${this.label} contains an invalid protobuf field number.`);
    return { field, wire };
  }

  rawVarint() {
    let value = 0n;
    for (let shift = 0n; shift < 70n; shift += 7n) {
      if (this.position >= this.bytes.length) throw new Error(`${this.label} contains a truncated protobuf varint.`);
      const byte = this.bytes[this.position++];
      value |= BigInt(byte & 0x7f) << shift;
      if ((byte & 0x80) === 0) {
        if (value > 0xffffffffffffffffn) throw new Error(`${this.label} protobuf integer exceeds the uint64 range.`);
        return value;
      }
    }
    throw new Error(`${this.label} contains an overlong protobuf varint.`);
  }

  varint() {
    const value = this.rawVarint();
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${this.label} protobuf integer exceeds the safe integer range.`);
    return Number(value);
  }

  intField(wire, singular, field, name) {
    this.requireWire(wire, 0, name);
    if (singular.has(field)) throw new Error(`${this.label} contains duplicate singular field ${name}.`);
    singular.add(field);
    const value = this.varint();
    if (!Number.isSafeInteger(value) || value < 0 || value > 0x7fffffff) throw new Error(`${this.label} field ${name} must be a non-negative int32.`);
    return value;
  }

  int64Field(wire, singular, field, name) {
    this.requireWire(wire, 0, name);
    if (singular.has(field)) throw new Error(`${this.label} contains duplicate singular field ${name}.`);
    singular.add(field);
    const value = BigInt.asIntN(64, this.rawVarint());
    if (value < BigInt(Number.MIN_SAFE_INTEGER) || value > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error(`${this.label} field ${name} exceeds the safe integer range.`);
    }
    return Number(value);
  }

  floatField(wire, singular, field, name) {
    this.requireWire(wire, 5, name);
    if (singular.has(field)) throw new Error(`${this.label} contains duplicate singular field ${name}.`);
    singular.add(field);
    const end = this.position + 4;
    if (end > this.bytes.length) throw new Error(`${this.label} field ${name} is truncated.`);
    const value = new DataView(this.bytes.buffer, this.bytes.byteOffset + this.position, 4).getFloat32(0, true);
    this.position = end;
    if (!Number.isFinite(value)) throw new Error(`${this.label} field ${name} must be finite.`);
    return value;
  }

  stringField(wire, singular, field, name) {
    if (singular.has(field)) throw new Error(`${this.label} contains duplicate singular field ${name}.`);
    singular.add(field);
    const bytes = this.bytesField(wire, name);
    try {
      return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    } catch {
      throw new Error(`${this.label} field ${name} is not valid UTF-8.`);
    }
  }

  repeatedInt(wire, name, allowNegative = false) {
    if (wire === 0) {
      return [this.int32Value(name, allowNegative)];
    }
    const packed = this.message(wire, `${this.label}.${name}`);
    const values = [];
    while (!packed.done) {
      values.push(packed.int32Value(name, allowNegative));
      guardLength(values, name);
    }
    return values;
  }

  int32Value(name, allowNegative) {
    const raw = this.rawVarint();
    if (raw <= 0x7fffffffn) return Number(raw);
    if (allowNegative && raw >= 0xffffffff80000000n) return Number(BigInt.asIntN(32, raw));
    throw new Error(`${this.label} field ${name} contains a negative or invalid int32.`);
  }

  message(wire, name) { return new ProtoReader(this.bytesField(wire, name), name); }

  bytesField(wire, name) {
    this.requireWire(wire, 2, name);
    const length = this.varint();
    const end = this.position + length;
    if (!Number.isSafeInteger(end) || end > this.bytes.length) throw new Error(`${this.label} field ${name} exceeds its protobuf message boundary.`);
    const value = this.bytes.subarray(this.position, end);
    this.position = end;
    return value;
  }

  skip(wire) {
    if (wire === 0) this.rawVarint();
    else if (wire === 1) this.advance(8);
    else if (wire === 2) this.advance(this.varint());
    else if (wire === 5) this.advance(4);
    else throw new Error(`${this.label} contains unsupported protobuf wire type ${wire}.`);
  }

  advance(length) {
    const end = this.position + length;
    if (!Number.isSafeInteger(end) || end > this.bytes.length) throw new Error(`${this.label} contains a truncated protobuf field.`);
    this.position = end;
  }

  requireWire(actual, expected, name) {
    if (actual !== expected) throw new Error(`${this.label} field ${name} has wire type ${actual}; expected ${expected}.`);
  }
}

export function toProtoBytes(input, evidenceLabel = "TFLite protobuf evidence") {
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (ArrayBuffer.isView(input)) return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  throw new Error(`${evidenceLabel} must be binary protobuf bytes.`);
}

function uniqueById(items, label) {
  const result = new Map();
  for (const item of items) {
    if (item.id == null) throw new Error(`${label} is missing an id.`);
    if (result.has(item.id)) throw new Error(`${label} contains duplicate id ${item.id}.`);
    result.set(item.id, item);
  }
  return result;
}

function uniqueIntegers(values, label) {
  const seen = new Set();
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${label} must be a non-negative integer.`);
    if (seen.has(value)) throw new Error(`${label} contains duplicate id ${value}.`);
    seen.add(value);
  }
  return [...seen];
}

function sameIntegerArray(left, right) {
  return left.length === right.length && left.every((value, index) => value === Number(right[index]));
}

function guardLength(items, label) {
  if (items.length > MAX_REPEATED_ITEMS) throw new Error(`TFLite ModelRuntimeDetails ${label} exceeds ${MAX_REPEATED_ITEMS.toLocaleString("en-US")} entries.`);
}

function countBy(items, keyFor) {
  return items.reduce((counts, item) => {
    const key = keyFor(item);
    counts[key] = (counts[key] || 0) + 1;
    return counts;
  }, {});
}

function requiredText(value, field, maxLength) {
  const text = String(value || "").trim();
  if (!text) throw new Error(`TFLite runtime metadata ${field} is required.`);
  if (text.length > maxLength) throw new Error(`TFLite runtime metadata ${field} exceeds ${maxLength} characters.`);
  return text;
}

function requiredSha(value, field) {
  const text = String(value || "").trim().toLowerCase();
  if (!/^[a-f0-9]{64}$/.test(text)) throw new Error(`TFLite runtime metadata ${field} must be SHA-256.`);
  return text;
}
