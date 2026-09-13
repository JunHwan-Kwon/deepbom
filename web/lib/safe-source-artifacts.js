import { sha256BytesHex } from "./sha256-sync.js";

const MAX_PROTOBUF_BYTES = 1024 * 1024 * 1024;
const MAX_PROTOBUF_FIELDS = 2_000_000;
const MAX_ZIP_ENTRIES = 100_000;
const MAX_ZIP_TEXT_BYTES = 16 * 1024 * 1024;
const MAX_ZIP_CENTRAL_DIRECTORY_BYTES = 64 * 1024 * 1024;
const MAX_ZIP_TEXT_COMPRESSION_RATIO = 200;
const UTF8 = new TextDecoder("utf-8", { fatal: true });

export function analyzeTensorFlowGraphDef(bytes, filename = "graph.pb") {
  requireBytes(bytes, MAX_PROTOBUF_BYTES, "TensorFlow GraphDef");
  const graph = parseGraphDef(bytes, "GraphDef");
  return graphDefAnalysis(graph, filename, "graphdef");
}

export function analyzeTensorFlowProtobuf(bytes, filename = "graph.pb") {
  requireBytes(bytes, MAX_PROTOBUF_BYTES, "TensorFlow protobuf");
  if (/^saved_model\.pb$/i.test(String(filename).split(/[\\/]/).pop() || "")) {
    const graphs = parseSavedModelGraphDefs(bytes);
    const primary = graphDefAnalysis(graphs[0], filename, "savedmodel", {
      source: "SavedModel.meta_graphs[0].graph_def",
      artifact_role: "model_package",
    });
    primary.savedmodel = {
      schema: "deepbom.tensorflow_savedmodel_safe_static_contract.v1",
      meta_graph_count: graphs.length,
      materialized_meta_graph_indices: [0],
      signature_defs: graphs[0].signatures,
      additional_meta_graphs: graphs.slice(1).map((graph, index) => ({ index: index + 1, node_count: graph.nodes.length, function_count: graph.functions.length, signature_count: graph.signatures.length })),
      package_closure_status: "protobuf_only_variables_assets_and_fingerprints_not_bound",
      evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
      unsafe_execution_performed: false,
      interpretation_boundary: "The first serialized MetaGraph GraphDef and SignatureDef tensor contracts are materialized. Additional MetaGraphs are inventoried but not merged into one execution graph, and this standalone protobuf does not bind the SavedModel directory, variables, assets, fingerprints, kernels, or runtime behavior.",
    };
    primary.format_extensions.savedmodel = primary.savedmodel;
    return primary;
  }
  return analyzeTensorFlowGraphDef(bytes, filename);
}

export function analyzeHdf5Envelope(bytes, filename = "model.h5") {
  requireBytes(bytes, Number.MAX_SAFE_INTEGER, "HDF5");
  const signatureOffset = findHdf5Signature(bytes);
  if (signatureOffset < 0) throw new Error("HDF5 signature was not found at an allowed user-block boundary.");
  const cursor = signatureOffset + 8;
  const version = bytes[cursor];
  if (![0, 1, 2, 3].includes(version)) throw new Error(`Unsupported HDF5 superblock version ${version}.`);
  const header = version <= 1 ? parseHdf5V01(bytes, cursor, version) : parseHdf5V23(bytes, cursor, version);
  return baseEnvelopeAnalysis(filename, "hdf5", {
    schema: "deepbom.hdf5_safe_envelope.v1",
    status: "superblock_assessed_object_graph_not_materialized",
    evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
    signature_offset: signatureOffset,
    ...header,
    unsafe_deserialization_performed: false,
    object_construction_performed: false,
    profile_candidates: profileCandidates(filename),
    interpretation_boundary: "The HDF5 superblock and container identity are parsed without loading Keras or Python objects. Group, dataset, link, filter, tensor, architecture, and executable-graph coverage are not claimed in this release.",
  }, "training_checkpoint_or_model_container");
}

export async function readHdf5EnvelopeFile(file, filename = file?.name || "model.h5") {
  requireFile(file, "HDF5");
  const offsets = [0];
  for (let offset = 512; offset < file.size && offset <= 1024 * 1024 * 1024; offset *= 2) offsets.push(offset);
  for (const offset of offsets) {
    const length = Math.min(512, file.size - offset);
    if (length < 8) continue;
    const bytes = new Uint8Array(await file.slice(offset, offset + length).arrayBuffer());
    if (findHdf5Signature(bytes) !== 0) continue;
    const analysis = analyzeHdf5Envelope(bytes, filename);
    analysis.file_size = file.size;
    analysis.file_size_bytes = file.size;
    analysis.safe_source_artifact.signature_offset = offset;
    analysis.safe_source_artifact.file_size_bytes = file.size;
    analysis.safe_source_artifact.header_range = { start: offset, end_exclusive: offset + bytes.length };
    return { analysis, retainedBytes: bytes };
  }
  throw new Error("HDF5 signature was not found at an allowed user-block boundary within the bounded 1 GiB search range.");
}

export async function analyzeZipModelEnvelope(bytes, filename = "model.pt2", forcedFormat = "") {
  requireBytes(bytes, Number.MAX_SAFE_INTEGER, "ZIP model container");
  const archive = parseZipDirectory(bytes);
  const lower = String(filename).toLowerCase();
  const format = forcedFormat || (lower.endsWith(".pt2") ? "pt2" : lower.endsWith(".keras") ? "keras" : "pytorch_checkpoint");
  const selected = await inspectKnownZipMembers(archive, format, (entry) => extractZipEntry(bytes, entry));
  const role = format === "pt2" ? "exported_program" : format === "keras" ? "model_package" : "training_checkpoint";
  const analysis = baseEnvelopeAnalysis(filename, format, {
    schema: "deepbom.safe_zip_model_envelope.v1",
    status: selected.status,
    evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
    archive: zipArchiveContract(archive),
    ...selected,
    unsafe_deserialization_performed: false,
    native_code_executed: false,
    interpretation_boundary: format === "pytorch_checkpoint"
      ? "The archive and bounded pickle opcode/global inventory are inspected without torch.load, pickle.load, Python imports, object construction, or native code execution. A checkpoint does not establish an executable model graph."
      : "The archive and bounded declarative members are inspected without framework object construction or native code execution. Opaque or unsafe members remain hash-addressed and are not interpreted as an executable runtime graph.",
  }, role);
  if (format === "pt2") materializePt2DeclarativeGraph(analysis, selected);
  if (format === "keras") materializeKerasDeclarativeGraph(analysis, selected);
  return analysis;
}

export async function readZipModelEnvelopeFile(file, filename = file?.name || "model.pt2", forcedFormat = "") {
  requireFile(file, "ZIP model container");
  const tailLength = Math.min(file.size, 65_557);
  const tailStart = file.size - tailLength;
  const tail = new Uint8Array(await file.slice(tailStart, file.size).arrayBuffer());
  const eocd = findEocd(tail);
  const eocdView = new DataView(tail.buffer, tail.byteOffset, tail.byteLength);
  const entryCount = eocdView.getUint16(eocd + 10, true);
  const centralDirectorySize = eocdView.getUint32(eocd + 12, true);
  const centralDirectoryOffset = eocdView.getUint32(eocd + 16, true);
  validateZipDirectoryBounds(entryCount, centralDirectorySize, centralDirectoryOffset, file.size, tailStart + eocd);
  const central = new Uint8Array(await file.slice(centralDirectoryOffset, centralDirectoryOffset + centralDirectorySize).arrayBuffer());
  const archive = parseZipDirectoryRecords(central, entryCount, centralDirectoryOffset, centralDirectorySize);
  const lower = String(filename).toLowerCase();
  const format = forcedFormat || (lower.endsWith(".pt2") ? "pt2" : lower.endsWith(".keras") ? "keras" : "pytorch_checkpoint");
  const selected = await inspectKnownZipMembers(archive, format, (entry) => extractZipFileEntry(file, entry));
  const role = format === "pt2" ? "exported_program" : format === "keras" ? "model_package" : "training_checkpoint";
  const analysis = baseEnvelopeAnalysis(filename, format, {
    schema: "deepbom.safe_zip_model_envelope.v1",
    status: selected.status,
    evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
    archive: zipArchiveContract(archive),
    ...selected,
    unsafe_deserialization_performed: false,
    native_code_executed: false,
    file_access: { mode: "bounded_ranges", full_artifact_loaded: false, central_directory_bytes: centralDirectorySize },
    interpretation_boundary: format === "pytorch_checkpoint"
      ? "The archive and bounded pickle opcode/global inventory are inspected without torch.load, pickle.load, Python imports, object construction, or native code execution. A checkpoint does not establish an executable model graph."
      : "The archive and bounded declarative members are inspected without framework object construction or native code execution. Opaque or unsafe members remain hash-addressed and are not interpreted as an executable runtime graph.",
  }, role);
  if (format === "pt2") materializePt2DeclarativeGraph(analysis, selected);
  if (format === "keras") materializeKerasDeclarativeGraph(analysis, selected);
  analysis.file_size = file.size;
  analysis.file_size_bytes = file.size;
  return { analysis, retainedBytes: tail };
}

export async function readSafeSourceArtifactFile(file, format, filename = file?.name || "model") {
  requireFile(file, "Safe source artifact");
  const normalized = String(format || "").toLowerCase();
  if (normalized === "hdf5") return readHdf5EnvelopeFile(file, filename);
  if (["keras", "pt2"].includes(normalized)) return readZipModelEnvelopeFile(file, filename, normalized);
  if (normalized === "pytorch_checkpoint") {
    const prefix = new Uint8Array(await file.slice(0, Math.min(file.size, 4)).arrayBuffer());
    if (prefix.length === 4 && prefix[0] === 0x50 && prefix[1] === 0x4b && prefix[2] === 0x03 && prefix[3] === 0x04) {
      return readZipModelEnvelopeFile(file, filename, normalized);
    }
    if (file.size > MAX_PROTOBUF_BYTES) throw new Error(`Legacy PyTorch checkpoint exceeds the ${MAX_PROTOBUF_BYTES}-byte bounded scanner limit.`);
    const bytes = new Uint8Array(await file.arrayBuffer());
    const analysis = analyzeLegacyPyTorchEnvelope(bytes, filename);
    analysis.file_size = file.size;
    analysis.file_size_bytes = file.size;
    return { analysis, retainedBytes: bytes };
  }
  if (normalized === "tensorflow_protobuf") {
    if (file.size > MAX_PROTOBUF_BYTES) throw new Error(`TensorFlow protobuf exceeds the ${MAX_PROTOBUF_BYTES}-byte bounded parser limit.`);
    const bytes = new Uint8Array(await file.arrayBuffer());
    const analysis = analyzeTensorFlowProtobuf(bytes, filename);
    analysis.file_size = file.size;
    analysis.file_size_bytes = file.size;
    return { analysis, retainedBytes: bytes };
  }
  throw new Error(`No safe source artifact reader is registered for ${normalized || "unknown"}.`);
}

export async function analyzeLegacyPyTorchEnvelope(bytes, filename = "model.pt") {
  requireBytes(bytes, Number.MAX_SAFE_INTEGER, "PyTorch checkpoint");
  const pickle = inspectPickle(bytes);
  return baseEnvelopeAnalysis(filename, "pytorch_checkpoint", {
    schema: "deepbom.pytorch_safe_envelope.v1",
    status: "legacy_pickle_opcode_inventory_assessed_payload_not_deserialized",
    evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
    container: "legacy_pickle_stream_or_unrecognized_checkpoint_envelope",
    pickle,
    unsafe_deserialization_performed: false,
    native_code_executed: false,
    interpretation_boundary: "Only bounded pickle opcodes and literal GLOBAL references are inventoried. STACK_GLOBAL targets, object semantics, tensors, architecture, and executable graph remain not assessable without unsafe object construction.",
  }, "training_checkpoint");
}

export function parseSavedModelGraphDefs(bytes) {
  const fields = protobufFields(bytes, "SavedModel", 0);
  const metaGraphs = fields.filter((field) => field.number === 2 && field.wire === 2);
  if (!metaGraphs.length) throw new Error("SavedModel contains no MetaGraphDef records.");
  return metaGraphs.map((field, index) => {
    const meta = protobufFields(field.bytes, `SavedModel.meta_graphs[${index}]`, 1);
    const graph = meta.find((item) => item.number === 2 && item.wire === 2);
    if (!graph) return null;
    const parsed = parseGraphDef(graph.bytes, `SavedModel.meta_graphs[${index}].graph_def`);
    parsed.signatures = parseSignatureDefs(meta, `SavedModel.meta_graphs[${index}]`);
    return parsed;
  }).filter(Boolean);
}

export function graphDefAnalysis(graph, filename, format = "graphdef", supplemental = {}) {
  const nodeIndex = new Map(graph.nodes.map((node, index) => [node.name, index]));
  const nodeByName = new Map(graph.nodes.map((node) => [node.name, node]));
  const endpoints = new Map();
  const endpoint = (name, port = 0) => {
    const key = `${name}:${port}`;
    if (!endpoints.has(key)) endpoints.set(key, endpoints.size);
    return endpoints.get(key);
  };
  for (const node of graph.nodes) for (const input of node.inputs) if (!input.control) endpoint(input.node, input.port);
  for (const signature of graph.signatures || []) for (const item of [...signature.inputs, ...signature.outputs]) {
    const parsed = parseTensorFlowInput(item.tensor_name);
    if (!parsed.control) endpoint(parsed.node, parsed.port);
  }
  const signatureTensorInfo = new Map((graph.signatures || []).flatMap((signature) => [...signature.inputs, ...signature.outputs]).map((item) => [item.tensor_name, item]));
  const consumers = new Set(graph.nodes.flatMap((node) => node.inputs.filter((input) => !input.control).map((input) => `${input.node}:${input.port}`)));
  const ops = graph.nodes.map((node, index) => ({
    index,
    name: node.op,
    graph_node_name: node.name,
    domain: "tensorflow.graphdef",
    version: null,
    inputs: node.inputs.filter((input) => !input.control).map((input) => endpoint(input.node, input.port)),
    outputs: [...endpoints.keys()].filter((key) => key.startsWith(`${node.name}:`)).map((key) => endpoints.get(key)).sort((a, b) => a - b),
    macs: null,
    macs_status: "not_assessed_operator_semantics_and_shapes_not_materialized",
  }));
  const tensors = [...endpoints.entries()].sort(([, left], [, right]) => left - right).map(([name, index]) => {
    const sourceName = name.replace(/:\d+$/, "");
    const contract = graphDefOutputContract(nodeByName.get(sourceName), signatureTensorInfo.get(name));
    return {
      index, name, dtype: contract.dtype, shape: contract.shape, value_kind: "graph_value", constant_buffer: contract.constant,
      serialized_payload_bytes: contract.serialized_payload_bytes,
      numerical_integrity: contract.payload_sha256 ? { payload_sha256: contract.payload_sha256 } : null,
      contract_status: contract.status,
    };
  });
  const controlDependencies = graph.nodes.flatMap((node, targetIndex) => node.inputs.filter((input) => input.control).map((input) => ({
    source_node_name: input.node,
    source_op_index: nodeIndex.has(input.node) ? nodeIndex.get(input.node) : null,
    target_node_name: node.name,
    target_op_index: targetIndex,
    status: nodeIndex.has(input.node) ? "resolved" : "unresolved",
  })));
  return {
    format,
    filename,
    file_size: supplemental.file_size ?? null,
    file_size_bytes: supplemental.file_size ?? null,
    ops,
    tensors,
    operator_count: ops.length,
    tensor_count: tensors.length,
    input_tensor_indices: uniqueNumbers((graph.signatures || []).flatMap((signature) => signature.inputs).map((item) => tensorIndexForName(item.tensor_name, endpoints))),
    output_tensor_indices: uniqueNumbers((graph.signatures || []).flatMap((signature) => signature.outputs).map((item) => tensorIndexForName(item.tensor_name, endpoints))),
    total_macs: null,
    total_macs_decimal: null,
    mac_confidence: "partial",
    mac_assessment: {
      status: "not_assessed",
      metric_scope: "serialized GraphDef node ledger; compute classification and shapes not materialized",
      compute_ops: ops.length,
      assessed_compute_ops: 0,
      not_assessed_compute_ops: ops.length,
      total_assessed_macs: 0,
      total_assessed_macs_decimal: "0",
    },
    graphdef: {
      schema: "deepbom.tensorflow_graphdef_static_contract.v1",
      status: "serialized_nodes_dependencies_and_bounded_endpoint_contracts_materialized",
      node_count: graph.nodes.length,
      function_count: graph.functions.length,
      control_dependency_count: controlDependencies.length,
      unresolved_input_count: graph.nodes.flatMap((node) => node.inputs).filter((input) => !nodeIndex.has(input.node)).length,
      versions: graph.versions,
      functions: graph.functions,
      node_attribute_summaries: graph.nodes.map((node, index) => ({ index, name: node.name, op: node.op, attributes: node.attributes })),
      control_dependencies: controlDependencies,
      source: supplemental.source || "GraphDef protobuf",
      unsafe_execution_performed: false,
      graph_input_output_status: graph.signatures?.length ? "explicit_signature_def_endpoints" : "not_explicitly_serialized_by_graphdef",
      interpretation_boundary: "Node names, op identities, referenced data inputs, control inputs, and function inventory are serialized facts. GraphDef does not explicitly identify model inputs or outputs, and unreferenced operation outputs are not synthesized. Tensor shapes, dtypes, function-call binding, execution schedule, kernels, placement, and runtime behavior are not inferred when not materialized.",
    },
    format_extensions: { [format]: { node_count: graph.nodes.length, function_count: graph.functions.length } },
    findings: [],
    recommendations: [],
    artifact_role: supplemental.artifact_role || "exported_program",
  };
}

function parseSignatureDefs(metaFields, label) {
  const entries = metaFields.filter((field) => field.number === 5 && field.wire === 2);
  if (entries.length > 100_000) throw new Error(`${label} exceeds 100,000 SignatureDef map entries.`);
  const seen = new Set();
  return entries.map((field, index) => {
    const fields = protobufFields(field.bytes, `${label}.signature_def[${index}]`, 2);
    const key = requiredString(fields, 1, `${label}.signature_def[${index}].key`);
    if (seen.has(key)) throw new Error(`${label} repeats SignatureDef key ${JSON.stringify(key)}.`);
    seen.add(key);
    const value = fields.find((item) => item.number === 2 && item.wire === 2);
    if (!value) throw new Error(`${label}.signature_def[${index}].value is missing.`);
    const signature = protobufFields(value.bytes, `${label}.signature_def[${index}].value`, 3);
    return {
      key,
      method_name: optionalString(signature, 3),
      inputs: parseTensorInfoMap(signature, 1, `${label}.signature_def[${index}].inputs`),
      outputs: parseTensorInfoMap(signature, 2, `${label}.signature_def[${index}].outputs`),
    };
  });
}

function parseTensorInfoMap(signatureFields, fieldNumber, label) {
  const entries = signatureFields.filter((field) => field.number === fieldNumber && field.wire === 2);
  const seen = new Set();
  return entries.map((field, index) => {
    const fields = protobufFields(field.bytes, `${label}[${index}]`, 4);
    const key = requiredString(fields, 1, `${label}[${index}].key`);
    if (seen.has(key)) throw new Error(`${label} repeats key ${JSON.stringify(key)}.`);
    seen.add(key);
    const value = fields.find((item) => item.number === 2 && item.wire === 2);
    if (!value) throw new Error(`${label}[${index}].value is missing.`);
    const info = protobufFields(value.bytes, `${label}[${index}].value`, 5);
    const tensorName = optionalString(info, 1);
    if (!tensorName) throw new Error(`${label}[${index}] does not expose a dense TensorInfo name.`);
    const dtype = info.find((item) => item.number === 2 && item.wire === 0);
    const shape = info.find((item) => item.number === 3 && item.wire === 2);
    return {
      key,
      tensor_name: tensorName,
      dtype: dtype ? tensorFlowDtype(safeNumber(dtype.value, `${label}[${index}].dtype`)) : "UNKNOWN",
      shape: shape ? parseTensorShape(shape.bytes, `${label}[${index}].tensor_shape`) : [],
      evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
    };
  });
}

function tensorIndexForName(name, endpoints) {
  const parsed = parseTensorFlowInput(name);
  return endpoints.get(`${parsed.node}:${parsed.port}`) ?? null;
}

function uniqueNumbers(values) {
  return [...new Set(values.filter((value) => Number.isSafeInteger(value) && value >= 0))].sort((left, right) => left - right);
}

function parseGraphDef(bytes, label) {
  const fields = protobufFields(bytes, label, 0);
  const nodes = fields.filter((field) => field.number === 1 && field.wire === 2).map((field, index) => parseNodeDef(field.bytes, `${label}.node[${index}]`));
  const library = fields.find((field) => field.number === 2 && field.wire === 2);
  const versions = fields.find((field) => field.number === 4 && field.wire === 2);
  return { nodes, functions: library ? parseFunctionLibrary(library.bytes, `${label}.library`) : [], versions: versions ? parseVersionDef(versions.bytes) : null };
}

function parseNodeDef(bytes, label) {
  const fields = protobufFields(bytes, label, 1);
  const name = requiredString(fields, 1, `${label}.name`);
  const op = requiredString(fields, 2, `${label}.op`);
  const inputs = fields.filter((field) => field.number === 3 && field.wire === 2).map((field, index) => parseTensorFlowInput(decode(field.bytes, `${label}.input[${index}]`)));
  const attributeFields = fields.filter((field) => field.number === 5 && field.wire === 2);
  if (attributeFields.length > 100_000) throw new Error(`${label} exceeds 100,000 attributes.`);
  const attributes = {};
  for (const [index, field] of attributeFields.entries()) {
    const attribute = parseNodeAttribute(field.bytes, `${label}.attr[${index}]`);
    if (Object.hasOwn(attributes, attribute.key)) throw new Error(`${label} repeats attribute ${JSON.stringify(attribute.key)}.`);
    attributes[attribute.key] = attribute.value;
  }
  return { name, op, inputs, device: optionalString(fields, 4), attribute_entry_count: attributeFields.length, attributes };
}

function parseNodeAttribute(bytes, label) {
  const fields = protobufFields(bytes, label, 2);
  const key = requiredString(fields, 1, `${label}.key`);
  const value = fields.find((field) => field.number === 2 && field.wire === 2);
  if (!value) throw new Error(`${label}.value is missing.`);
  return { key, value: parseAttrValue(value.bytes, `${label}.value`) };
}

function parseAttrValue(bytes, label) {
  const fields = protobufFields(bytes, label, 3);
  const type = fields.find((field) => field.number === 6 && field.wire === 0);
  const shape = fields.find((field) => field.number === 7 && field.wire === 2);
  const tensor = fields.find((field) => field.number === 8 && field.wire === 2);
  const list = fields.find((field) => field.number === 1 && field.wire === 2);
  return {
    type: type ? tensorFlowDtype(safeNumber(type.value, `${label}.type`)) : null,
    shape: shape ? parseTensorShape(shape.bytes, `${label}.shape`) : null,
    tensor: tensor ? parseTensorProto(tensor.bytes, `${label}.tensor`) : null,
    list_shapes: list ? protobufFields(list.bytes, `${label}.list`, 4).filter((field) => field.number === 7 && field.wire === 2).map((field, index) => parseTensorShape(field.bytes, `${label}.list.shape[${index}]`)) : [],
    scalar_kinds_present: fields.filter((field) => [2,3,4,5,9,10,11,12,13,14].includes(field.number)).map((field) => field.number),
  };
}

function parseTensorProto(bytes, label) {
  const fields = protobufFields(bytes, label, 4);
  const dtypeField = fields.find((field) => field.number === 1 && field.wire === 0);
  const shapeField = fields.find((field) => field.number === 2 && field.wire === 2);
  const content = fields.find((field) => field.number === 4 && field.wire === 2)?.bytes || null;
  return {
    dtype: dtypeField ? tensorFlowDtype(safeNumber(dtypeField.value, `${label}.dtype`)) : "UNKNOWN",
    shape: shapeField ? parseTensorShape(shapeField.bytes, `${label}.tensor_shape`) : [],
    tensor_content_byte_length: content?.length || 0,
    tensor_content_sha256: content ? sha256BytesHex(content) : null,
    typed_value_field_count: fields.filter((field) => field.number >= 5 && field.number <= 19).length,
  };
}

function parseTensorShape(bytes, label) {
  const fields = protobufFields(bytes, label, 4);
  const unknownRank = fields.find((field) => field.number === 3 && field.wire === 0);
  if (unknownRank?.value === 1n) return null;
  return fields.filter((field) => field.number === 2 && field.wire === 2).map((field, index) => {
    const dim = protobufFields(field.bytes, `${label}.dim[${index}]`, 5);
    const size = dim.find((item) => item.number === 1 && item.wire === 0);
    if (!size || size.value > BigInt(Number.MAX_SAFE_INTEGER)) return -1;
    return Number(size.value);
  });
}

function graphDefOutputContract(node, signatureInfo = null) {
  if (!node) return { dtype: "UNKNOWN", shape: [], constant: false, serialized_payload_bytes: 0, payload_sha256: null, status: "referenced_source_node_unresolved" };
  const tensor = node.attributes?.value?.tensor || null;
  const declaredShape = node.attributes?.shape?.shape ?? node.attributes?._output_shapes?.list_shapes?.[0] ?? tensor?.shape ?? null;
  const dtype = tensor?.dtype || node.attributes?.dtype?.type || node.attributes?.T?.type || signatureInfo?.dtype || "UNKNOWN";
  const resolvedShape = Array.isArray(declaredShape) ? declaredShape : Array.isArray(signatureInfo?.shape) ? signatureInfo.shape : [];
  return {
    dtype,
    shape: resolvedShape,
    constant: node.op === "Const" && Boolean(tensor),
    serialized_payload_bytes: tensor?.tensor_content_byte_length || 0,
    payload_sha256: tensor?.tensor_content_sha256 || null,
    status: tensor ? "serialized_tensor_proto_contract" : declaredShape || node.attributes?.dtype?.type || node.attributes?.T?.type
      ? "serialized_node_attribute_contract" : signatureInfo ? "serialized_signature_def_contract" : "serialized_endpoint_name_shape_and_dtype_not_materialized",
  };
}

function tensorFlowDtype(value) {
  return ({ 1:"FLOAT32",2:"FLOAT64",3:"INT32",4:"UINT8",5:"INT16",6:"INT8",7:"STRING",8:"COMPLEX64",9:"INT64",10:"BOOL",14:"BFLOAT16",17:"UINT16",19:"FLOAT16",22:"UINT32",23:"UINT64" })[value] || `TF_DATATYPE_${value}`;
}

function parseTensorFlowInput(value) {
  const control = value.startsWith("^");
  const body = control ? value.slice(1) : value;
  const match = /^(.*?)(?::(\d+))?$/.exec(body);
  if (!match || !match[1]) throw new Error(`Invalid TensorFlow NodeDef input reference ${JSON.stringify(value)}.`);
  const port = match[2] == null ? 0 : Number(match[2]);
  if (!Number.isSafeInteger(port) || port < 0) throw new Error(`Invalid TensorFlow output port in ${JSON.stringify(value)}.`);
  return { serialized: value, node: match[1], port, control };
}

function parseFunctionLibrary(bytes, label) {
  return protobufFields(bytes, label, 1).filter((field) => field.number === 1 && field.wire === 2).map((field, index) => {
    const fields = protobufFields(field.bytes, `${label}.function[${index}]`, 2);
    const signature = fields.find((item) => item.number === 1 && item.wire === 2);
    const functionName = signature ? optionalString(protobufFields(signature.bytes, `${label}.function[${index}].signature`, 3), 1) : null;
    return {
      index,
      name: functionName || `function_${index}`,
      node_count: fields.filter((item) => item.number === 3 && item.wire === 2).length,
      return_entry_count: fields.filter((item) => item.number === 4 && item.wire === 2).length,
      control_return_entry_count: fields.filter((item) => item.number === 6 && item.wire === 2).length,
    };
  });
}

function parseVersionDef(bytes) {
  const fields = protobufFields(bytes, "VersionDef", 1);
  return {
    producer: varintNumber(fields.find((field) => field.number === 1)),
    min_consumer: varintNumber(fields.find((field) => field.number === 2)),
    bad_consumers: fields.filter((field) => field.number === 3).map(varintNumber).filter((value) => value != null),
  };
}

function protobufFields(bytes, label, depth) {
  if (depth > 8) throw new Error(`${label} exceeds the bounded protobuf nesting depth.`);
  const fields = [];
  let offset = 0;
  while (offset < bytes.length) {
    if (fields.length >= MAX_PROTOBUF_FIELDS) throw new Error(`${label} exceeds ${MAX_PROTOBUF_FIELDS} protobuf fields.`);
    const tag = readVarint(bytes, offset, label); offset = tag.next;
    const number = Number(tag.value >> 3n);
    const wire = Number(tag.value & 7n);
    if (!Number.isSafeInteger(number) || number <= 0) throw new Error(`${label} contains an invalid protobuf field number.`);
    if (wire === 0) { const value = readVarint(bytes, offset, label); offset = value.next; fields.push({ number, wire, value: value.value }); }
    else if (wire === 1) { requireRange(bytes, offset, 8, label); fields.push({ number, wire, bytes: bytes.subarray(offset, offset + 8) }); offset += 8; }
    else if (wire === 2) { const length = readVarint(bytes, offset, label); offset = length.next; const size = safeNumber(length.value, `${label} length`); requireRange(bytes, offset, size, label); fields.push({ number, wire, bytes: bytes.subarray(offset, offset + size) }); offset += size; }
    else if (wire === 5) { requireRange(bytes, offset, 4, label); fields.push({ number, wire, bytes: bytes.subarray(offset, offset + 4) }); offset += 4; }
    else throw new Error(`${label} contains unsupported protobuf wire type ${wire}.`);
  }
  return fields;
}

function parseZipDirectory(bytes) {
  const eocd = findEocd(bytes);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const entryCount = view.getUint16(eocd + 10, true);
  const centralDirectorySize = view.getUint32(eocd + 12, true);
  const centralDirectoryOffset = view.getUint32(eocd + 16, true);
  validateZipDirectoryBounds(entryCount, centralDirectorySize, centralDirectoryOffset, bytes.length, eocd);
  const central = bytes.subarray(centralDirectoryOffset, centralDirectoryOffset + centralDirectorySize);
  return parseZipDirectoryRecords(central, entryCount, centralDirectoryOffset, centralDirectorySize);
}

function parseZipDirectoryRecords(bytes, entryCount, centralDirectoryOffset, centralDirectorySize) {
  const entries = [];
  const names = new Set();
  let offset = 0;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < entryCount; index += 1) {
    requireRange(bytes, offset, 46, "ZIP central record");
    if (view.getUint32(offset, true) !== 0x02014b50) throw new Error("ZIP central-directory signature is invalid.");
    const flags = view.getUint16(offset + 8, true);
    const compressionMethod = view.getUint16(offset + 10, true);
    const crc32 = view.getUint32(offset + 16, true);
    const compressedSize = view.getUint32(offset + 20, true);
    const uncompressedSize = view.getUint32(offset + 24, true);
    const nameLength = view.getUint16(offset + 28, true);
    const extraLength = view.getUint16(offset + 30, true);
    const commentLength = view.getUint16(offset + 32, true);
    const localHeaderOffset = view.getUint32(offset + 42, true);
    requireRange(bytes, offset + 46, nameLength + extraLength + commentLength, "ZIP central record payload");
    const memberPath = safeArchivePath(decode(bytes.subarray(offset + 46, offset + 46 + nameLength), `ZIP entry ${index}`));
    const folded = memberPath.toLowerCase();
    if (names.has(folded)) throw new Error(`ZIP archive repeats or case-collides path ${memberPath}.`);
    names.add(folded);
    if (flags & 1) throw new Error(`Encrypted ZIP entry is not accepted: ${memberPath}.`);
    entries.push({ path: memberPath, flags, compressionMethod, crc32, compressedSize, uncompressedSize, localHeaderOffset, encrypted: false });
    offset += 46 + nameLength + extraLength + commentLength;
  }
  if (offset !== centralDirectorySize) throw new Error("ZIP central-directory size does not conserve parsed records.");
  return { entries, centralDirectoryOffset, centralDirectorySize };
}

function validateZipDirectoryBounds(entryCount, centralDirectorySize, centralDirectoryOffset, fileSize, eocdOffset) {
  if (entryCount === 0xffff || centralDirectorySize === 0xffffffff || centralDirectoryOffset === 0xffffffff) throw new Error("ZIP64 model archives are not supported by the bounded envelope parser.");
  if (entryCount > MAX_ZIP_ENTRIES) throw new Error(`ZIP archive exceeds ${MAX_ZIP_ENTRIES} entries.`);
  if (centralDirectorySize > MAX_ZIP_CENTRAL_DIRECTORY_BYTES) throw new Error(`ZIP central directory exceeds ${MAX_ZIP_CENTRAL_DIRECTORY_BYTES} bytes.`);
  if (centralDirectoryOffset + centralDirectorySize !== eocdOffset || centralDirectoryOffset + centralDirectorySize > fileSize) throw new Error("ZIP central-directory bounds do not terminate at the EOCD record.");
}

async function inspectKnownZipMembers(archive, format, extract) {
  const basename = (wanted) => archive.entries.filter((entry) => entry.path.toLowerCase().endsWith(`/${wanted}`) || entry.path.toLowerCase() === wanted);
  if (format === "pytorch_checkpoint") {
    const pickles = archive.entries.filter((entry) => /(^|\/)data\.pkl$|\.pkl$/i.test(entry.path));
    const inventories = [];
    for (const entry of pickles.slice(0, 16)) inventories.push({ path: entry.path, pickle: inspectPickle(await extract(entry)) });
    return {
      status: "archive_and_pickle_opcode_inventory_assessed_payload_not_deserialized",
      pickle_members: inventories,
      storage_member_count: archive.entries.filter((entry) => /(^|\/)data\/\d+$/i.test(entry.path)).length,
      native_binary_members: archive.entries.filter((entry) => /\.(so|dll|dylib|cubin|ptx)$/i.test(entry.path)).map((entry) => entry.path),
    };
  }
  if (format === "pt2") {
    const formatMarkers = basename("archive_format");
    const versionMarkers = basename("archive_version");
    if (formatMarkers.length > 1 || versionMarkers.length > 1) throw new Error("PT2 archive repeats its format or version marker.");
    const archiveFormat = formatMarkers[0] ? decode(await extract(formatMarkers[0]), formatMarkers[0].path).trim() : null;
    const archiveVersion = versionMarkers[0] ? decode(await extract(versionMarkers[0]), versionMarkers[0].path).trim() : null;
    const jsonEntries = archive.entries.filter((entry) => /(^|\/)models\/[^/]+\.json$/i.test(entry.path)
      || /(^|\/)data\/(weights|constants)\/[^/]+_config\.json$/i.test(entry.path)
      || /(^|\/)extra\/module_info\.json$/i.test(entry.path));
    const documents = [];
    for (const entry of jsonEntries.slice(0, 16)) documents.push({ path: entry.path, document: parseJson(await extract(entry), entry.path) });
    return {
      status: archiveFormat !== "pt2" ? "archive_assessed_pt2_format_marker_missing_or_invalid"
        : documents.some((row) => /(^|\/)models\/[^/]+\.json$/i.test(row.path)) ? "pt2_archive_and_bounded_declarative_members_assessed"
          : "pt2_archive_assessed_declarative_model_definition_not_found",
      archive_format: archiveFormat,
      archive_version: archiveVersion,
      declarative_documents: documents,
      opaque_native_members: archive.entries.filter((entry) => /\.(so|dll|dylib|cubin|ptx)$/i.test(entry.path)).map((entry) => entry.path),
      embedded_pickle_members: archive.entries.filter((entry) => /\.(pt|pth|pkl|pickle)$/i.test(entry.path)).map((entry) => entry.path),
      declarative_member_limit: 16,
      declarative_members_omitted: Math.max(0, jsonEntries.length - 16),
    };
  }
  const configs = [...basename("config.json"), ...basename("metadata.json")];
  const documents = [];
  for (const entry of configs.slice(0, 8)) documents.push({ path: entry.path, document: parseJson(await extract(entry), entry.path) });
  return {
    status: documents.length ? "keras_archive_and_declarative_config_assessed_objects_not_constructed" : "keras_archive_assessed_config_not_found",
    declarative_documents: documents,
    layer_inventory: kerasLayerInventory(documents.find((row) => /config\.json$/i.test(row.path))?.document),
  };
}

async function extractZipEntry(bytes, entry) {
  validateExtractedZipEntry(entry);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const offset = entry.localHeaderOffset;
  requireRange(bytes, offset, 30, `ZIP local header ${entry.path}`);
  if (view.getUint32(offset, true) !== 0x04034b50) throw new Error(`ZIP local header signature is invalid for ${entry.path}.`);
  const nameLength = view.getUint16(offset + 26, true);
  const extraLength = view.getUint16(offset + 28, true);
  const start = offset + 30 + nameLength + extraLength;
  requireRange(bytes, start, entry.compressedSize, `ZIP member ${entry.path}`);
  const compressed = bytes.subarray(start, start + entry.compressedSize);
  return decompressAndVerifyZipEntry(compressed, entry);
}

async function extractZipFileEntry(file, entry) {
  validateExtractedZipEntry(entry);
  const local = new Uint8Array(await file.slice(entry.localHeaderOffset, entry.localHeaderOffset + 30).arrayBuffer());
  requireRange(local, 0, 30, `ZIP local header ${entry.path}`);
  const view = new DataView(local.buffer, local.byteOffset, local.byteLength);
  if (view.getUint32(0, true) !== 0x04034b50) throw new Error(`ZIP local header signature is invalid for ${entry.path}.`);
  const flags = view.getUint16(6, true);
  const method = view.getUint16(8, true);
  const nameLength = view.getUint16(26, true);
  const extraLength = view.getUint16(28, true);
  if ((flags & 1) || method !== entry.compressionMethod) throw new Error(`ZIP local header conflicts with the central directory for ${entry.path}.`);
  const headerPayload = new Uint8Array(await file.slice(entry.localHeaderOffset + 30, entry.localHeaderOffset + 30 + nameLength + extraLength).arrayBuffer());
  const localName = safeArchivePath(decode(headerPayload.subarray(0, nameLength), `ZIP local entry ${entry.path}`));
  if (localName !== entry.path) throw new Error(`ZIP local and central paths disagree for ${entry.path}.`);
  const start = entry.localHeaderOffset + 30 + nameLength + extraLength;
  if (start + entry.compressedSize > file.size) throw new Error(`ZIP member ${entry.path} exceeds the source byte range.`);
  const compressed = new Uint8Array(await file.slice(start, start + entry.compressedSize).arrayBuffer());
  return decompressAndVerifyZipEntry(compressed, entry);
}

function validateExtractedZipEntry(entry) {
  if (entry.uncompressedSize > MAX_ZIP_TEXT_BYTES || entry.compressedSize > MAX_ZIP_TEXT_BYTES) throw new Error(`ZIP declarative member ${entry.path} exceeds the ${MAX_ZIP_TEXT_BYTES}-byte extraction limit.`);
  if (entry.compressedSize > 0 && entry.uncompressedSize / entry.compressedSize > MAX_ZIP_TEXT_COMPRESSION_RATIO) throw new Error(`ZIP declarative member ${entry.path} exceeds the ${MAX_ZIP_TEXT_COMPRESSION_RATIO}:1 compression-ratio limit.`);
}

async function decompressAndVerifyZipEntry(compressed, entry) {
  validateExtractedZipEntry(entry);
  let output;
  if (entry.compressionMethod === 0) output = new Uint8Array(compressed);
  else if (entry.compressionMethod === 8) {
    if (typeof DecompressionStream !== "function") throw new Error(`Deflate decompression is unavailable for ${entry.path}.`);
    output = new Uint8Array(await new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream("deflate-raw"))).arrayBuffer());
  } else throw new Error(`ZIP member ${entry.path} uses unsupported compression method ${entry.compressionMethod}.`);
  if (output.length !== entry.uncompressedSize) throw new Error(`ZIP member ${entry.path} uncompressed size mismatch.`);
  if (crc32(output) !== entry.crc32) throw new Error(`ZIP member ${entry.path} CRC-32 mismatch.`);
  return output;
}

function zipArchiveContract(archive) {
  return {
    entry_count: archive.entries.length,
    central_directory_offset: archive.centralDirectoryOffset,
    central_directory_size: archive.centralDirectorySize,
    total_claimed_uncompressed_bytes: archive.entries.reduce((sum, entry) => sum + BigInt(entry.uncompressedSize), 0n).toString(),
    entries: archive.entries.map((entry) => ({
      path: entry.path, compression_method: entry.compressionMethod, crc32: entry.crc32.toString(16).padStart(8, "0"),
      compressed_bytes: entry.compressedSize, uncompressed_bytes: entry.uncompressedSize, local_header_offset: entry.localHeaderOffset, encrypted: entry.encrypted,
    })),
  };
}

function inspectPickle(bytes) {
  const globals = [];
  const opcodeCounts = new Map();
  let stackGlobalCount = 0;
  let offset = 0;
  let stopped = false;
  while (offset < bytes.length && !stopped) {
    const opcode = bytes[offset];
    opcodeCounts.set(`0x${opcode.toString(16).padStart(2, "0")}`, (opcodeCounts.get(`0x${opcode.toString(16).padStart(2, "0")}`) || 0) + 1);
    offset += 1;
    if (opcode === 0x63) {
      const moduleEnd = bytes.indexOf(0x0a, offset);
      const nameEnd = moduleEnd < 0 ? -1 : bytes.indexOf(0x0a, moduleEnd + 1);
      if (moduleEnd < 0 || nameEnd < 0) throw new Error("Pickle GLOBAL opcode is truncated.");
      globals.push({ module: decode(bytes.subarray(offset, moduleEnd), "pickle GLOBAL module"), name: decode(bytes.subarray(moduleEnd + 1, nameEnd), "pickle GLOBAL name") });
      offset = nameEnd + 1;
    } else if (opcode === 0x93) stackGlobalCount += 1;
    else if (opcode === 0x2e) stopped = true;
    else offset = skipPickleArgument(bytes, offset, opcode);
  }
  return {
    byte_length: bytes.length,
    sha256: sha256BytesHex(bytes),
    literal_global_references: globals.slice(0, 10_000),
    literal_global_count: globals.length,
    stack_global_count: stackGlobalCount,
    opcode_histogram: Object.fromEntries([...opcodeCounts].sort(([left], [right]) => left.localeCompare(right))),
    stop_opcode_observed: stopped,
    parsed_byte_length: offset,
    completeness: !stopped ? "partial_stop_opcode_not_reached" : stackGlobalCount ? "partial_stack_global_targets_not_resolved" : "literal_global_inventory_complete_for_opcode_stream",
  };
}

function materializePt2DeclarativeGraph(analysis, selected) {
  const modelDocuments = (selected.declarative_documents || []).filter((row) => /(^|\/)models\/[^/]+\.json$/i.test(row.path));
  const primary = modelDocuments[0]?.document;
  const graph = primary?.graph_module?.graph;
  if (!graph || !Array.isArray(graph.nodes) || !Array.isArray(graph.inputs) || !Array.isArray(graph.outputs)) {
    analysis.pt2 = {
      schema: "deepbom.pt2_declarative_graph.v1",
      graph_materialized: false,
      model_definition_count: modelDocuments.length,
      status: modelDocuments.length ? "model_definition_present_graph_schema_not_materialized" : "model_definition_not_present",
      interpretation_boundary: "No executable graph is claimed unless a bounded models/*.json document exposes the pinned ExportedProgram graph_module.graph fields.",
    };
    analysis.format_extensions.pt2 = analysis.pt2;
    return;
  }
  if (graph.nodes.length > 1_000_000) throw new Error("PT2 declarative graph exceeds 1,000,000 nodes.");
  const inputNames = uniqueNames(graph.inputs.flatMap(pt2ArgumentNames));
  const outputNames = uniqueNames(graph.outputs.flatMap(pt2ArgumentNames));
  const nodes = graph.nodes.map((node, index) => {
    if (!node || typeof node !== "object" || typeof node.target !== "string" || !node.target) throw new Error(`PT2 graph node ${index} has no serialized target.`);
    return {
      index,
      name: node.target,
      graph_node_name: typeof node.name === "string" && node.name ? node.name : `node_${index}`,
      domain: "pytorch.export",
      version: null,
      input_names: uniqueNames((Array.isArray(node.inputs) ? node.inputs : []).flatMap((item) => pt2ArgumentNames(item?.arg))),
      output_names: uniqueNames((Array.isArray(node.outputs) ? node.outputs : []).flatMap(pt2ArgumentNames)),
      macs: null,
      macs_status: "not_assessed_operator_semantics_and_shapes_not_executed",
    };
  });
  const valueNames = uniqueNames([
    ...inputNames,
    ...outputNames,
    ...nodes.flatMap((row) => [...row.input_names, ...row.output_names]),
    ...Object.keys(graph.tensor_values || {}),
  ]);
  const valueIndex = new Map(valueNames.map((name, index) => [name, index]));
  analysis.ops = nodes.map(({ input_names, output_names, ...node }) => ({
    ...node,
    inputs: input_names.map((name) => valueIndex.get(name)),
    outputs: output_names.map((name) => valueIndex.get(name)),
  }));
  analysis.tensors = valueNames.map((name, index) => {
    const meta = graph.tensor_values?.[name];
    const shape = pt2StaticShape(meta?.sizes);
    return {
      index,
      name,
      dtype: pt2Dtype(meta?.dtype),
      shape: shape || [],
      shape_signature: shape || [],
      value_kind: meta ? "tensor" : "serialized_graph_argument",
      constant_buffer: false,
      contract_status: meta ? (shape ? "serialized_tensor_meta_static_shape" : "serialized_tensor_meta_symbolic_or_unresolved_shape") : "serialized_argument_name_only",
    };
  });
  analysis.operator_count = analysis.ops.length;
  analysis.tensor_count = analysis.tensors.length;
  analysis.input_tensor_indices = inputNames.map((name) => valueIndex.get(name));
  analysis.output_tensor_indices = outputNames.map((name) => valueIndex.get(name));
  analysis.mac_assessment = {
    status: "not_assessed",
    metric_scope: "serialized PT2 ExportedProgram JSON node ledger; no operator execution or framework object construction",
    compute_ops: analysis.ops.length,
    assessed_compute_ops: 0,
    not_assessed_compute_ops: analysis.ops.length,
    total_assessed_macs: 0,
    total_assessed_macs_decimal: "0",
  };
  analysis.pt2 = {
    schema: "deepbom.pt2_declarative_graph.v1",
    graph_materialized: true,
    primary_model_definition_path: modelDocuments[0].path,
    model_definition_count: modelDocuments.length,
    additional_model_definitions: modelDocuments.slice(1).map((row) => row.path),
    operation_count: analysis.ops.length,
    value_count: analysis.tensors.length,
    nested_graph_argument_count: countKey(primary, "as_graph"),
    schema_version: primary.schema_version || null,
    opset_version: primary.opset_version || null,
    torch_version: primary.torch_version || null,
    unsafe_deserialization_performed: false,
    interpretation_boundary: "Operations and named value dependencies come only from the first bounded models/*.json ExportedProgram document. Additional model definitions and higher-order nested graph bodies are inventoried but not merged. Tensor payloads, .pt sample inputs, native binaries, kernels, runtime order, and execution behavior are not loaded or inferred.",
  };
  analysis.format_extensions.pt2 = analysis.pt2;
}

function materializeKerasDeclarativeGraph(analysis, selected) {
  const configRow = (selected.declarative_documents || []).find((row) => /(^|\/)config\.json$/i.test(row.path));
  const document = configRow?.document;
  const layers = document?.config?.layers;
  const rootClass = String(document?.class_name || "");
  if (!Array.isArray(layers) || !["Sequential", "Functional", "Model"].includes(rootClass) || layers.length > 1_000_000) {
    analysis.keras = {
      schema: "deepbom.keras_declarative_layer_graph.v1",
      graph_materialized: false,
      status: Array.isArray(layers) ? "unsupported_or_unbounded_model_config" : "model_layer_config_not_present",
      interpretation_boundary: "A Keras layer graph is materialized only for a bounded Sequential, Functional, or Model config.json. Keras objects and custom layer code are never constructed or imported.",
    };
    analysis.format_extensions.keras = analysis.keras;
    return;
  }
  const nameToIndex = new Map();
  const records = layers.map((layer, index) => {
    const name = String(layer?.config?.name || layer?.name || `layer_${index}`);
    if (!name || nameToIndex.has(name)) throw new Error(`Keras config contains a missing or duplicate layer name ${JSON.stringify(name)}.`);
    nameToIndex.set(name, index);
    return { layer, index, name, className: String(layer?.class_name || "unknown") };
  });
  const explicitInputs = records.map((row) => uniqueNames(kerasHistoryNames(row.layer?.inbound_nodes))).map((names) => names.filter((name) => nameToIndex.has(name)));
  const sequential = rootClass === "Sequential";
  const valueNames = records.map((row) => `${row.name}:output:0`);
  analysis.tensors = records.map((row, index) => {
    const shape = kerasStaticShape(row.layer?.build_config?.input_shape ?? row.layer?.config?.batch_shape ?? row.layer?.config?.batch_input_shape);
    return {
      index,
      name: valueNames[index],
      dtype: String(row.layer?.config?.dtype || "UNKNOWN").toUpperCase(),
      shape: shape || [],
      shape_signature: shape || [],
      value_kind: "keras_declared_layer_output",
      constant_buffer: false,
      contract_status: shape ? "declared_config_shape" : "serialized_layer_output_name_shape_not_assessable",
    };
  });
  analysis.ops = records.map((row, index) => {
    const predecessors = explicitInputs[index].length ? explicitInputs[index]
      : sequential && index > 0 ? [records[index - 1].name] : [];
    return {
      index,
      name: row.className,
      graph_node_name: row.name,
      domain: "keras.config",
      version: null,
      inputs: predecessors.map((name) => nameToIndex.get(name)),
      outputs: [index],
      macs: null,
      macs_status: "not_assessed_layer_class_does_not_establish_lowered_operator_semantics",
    };
  });
  const declaredInputs = uniqueNames(kerasEndpointNames(document?.config?.input_layers));
  const declaredOutputs = uniqueNames(kerasEndpointNames(document?.config?.output_layers));
  const inputIndices = declaredInputs.map((name) => nameToIndex.get(name)).filter(Number.isSafeInteger);
  const outputIndices = declaredOutputs.map((name) => nameToIndex.get(name)).filter(Number.isSafeInteger);
  analysis.operator_count = analysis.ops.length;
  analysis.tensor_count = analysis.tensors.length;
  analysis.input_tensor_indices = inputIndices.length ? inputIndices : records.filter((row, index) => row.className === "InputLayer" || (!explicitInputs[index].length && (!sequential || index === 0))).map((row) => row.index);
  analysis.output_tensor_indices = outputIndices.length ? outputIndices : (records.length ? [records.at(-1).index] : []);
  analysis.mac_assessment = {
    status: "not_assessed",
    metric_scope: "serialized Keras config layer graph; no layer construction, lowering, kernel selection, or execution",
    compute_ops: analysis.ops.length,
    assessed_compute_ops: 0,
    not_assessed_compute_ops: analysis.ops.length,
    total_assessed_macs: 0,
    total_assessed_macs_decimal: "0",
  };
  analysis.keras = {
    schema: "deepbom.keras_declarative_layer_graph.v1",
    graph_materialized: true,
    root_class_name: rootClass,
    primary_config_path: configRow.path,
    layer_count: records.length,
    explicit_inbound_reference_count: explicitInputs.reduce((sum, rows) => sum + rows.length, 0),
    sequential_order_used: sequential,
    unsafe_deserialization_performed: false,
    custom_layer_execution_claim: false,
    interpretation_boundary: "Layer identities and connections are projected only from serialized Keras config order or explicit keras_history references. Layer classes are not constructed, imported, lowered, or treated as equivalent to backend operators. Weights, concrete function traces, kernels, runtime order, performance, and model quality remain separate evidence.",
  };
  analysis.format_extensions.keras = analysis.keras;
}

function pt2ArgumentNames(value) {
  if (!value || typeof value !== "object") return [];
  if (Array.isArray(value)) return value.flatMap(pt2ArgumentNames);
  const direct = [value.as_tensor?.name, value.as_token?.name, value.as_sym_int?.as_name, value.as_sym_bool?.as_name, value.as_sym_float?.as_name].filter((item) => typeof item === "string" && item);
  const repeated = [value.as_tensors, value.as_optional_tensors, value.as_nested_tensors].flatMap((item) => Array.isArray(item) ? item.flatMap(pt2ArgumentNames) : []);
  return [...direct, ...repeated, ...pt2ArgumentNames(value.as_optional_tensor)];
}

function uniqueNames(rows) { return [...new Set(rows.filter((row) => typeof row === "string" && row.length && row.length <= 100_000))]; }
function pt2StaticShape(sizes) {
  if (!Array.isArray(sizes)) return null;
  const result = [];
  for (const size of sizes) {
    if (!Number.isSafeInteger(size?.as_int) || size.as_int < 0) return null;
    result.push(size.as_int);
  }
  return result;
}
function pt2Dtype(value) {
  if (typeof value === "string" && value) return value.toUpperCase();
  if (Number.isSafeInteger(value)) return `PT2_SCALAR_TYPE_${value}`;
  return "UNKNOWN";
}
function countKey(value, key) {
  if (!value || typeof value !== "object") return 0;
  if (Array.isArray(value)) return value.reduce((sum, row) => sum + countKey(row, key), 0);
  return Object.entries(value).reduce((sum, [name, row]) => sum + (name === key ? 1 : 0) + countKey(row, key), 0);
}
function kerasHistoryNames(value) {
  if (!value || typeof value !== "object") return [];
  if (Array.isArray(value)) return value.flatMap(kerasHistoryNames);
  const history = value.keras_history;
  const current = Array.isArray(history) && typeof history[0] === "string" ? [history[0]] : [];
  return [...current, ...Object.values(value).flatMap(kerasHistoryNames)];
}
function kerasEndpointNames(value) {
  if (!Array.isArray(value)) return [];
  return value.flatMap((row) => Array.isArray(row) && typeof row[0] === "string" ? [row[0]] : kerasEndpointNames(row));
}
function kerasStaticShape(value) {
  if (!Array.isArray(value)) return null;
  const shape = [];
  for (const item of value) {
    if (item == null) shape.push(-1);
    else if (Number.isSafeInteger(item) && item >= 0) shape.push(item);
    else return null;
  }
  return shape;
}

function skipPickleArgument(bytes, offset, opcode) {
  const noArgument = new Set([0x28,0x29,0x30,0x31,0x32,0x4e,0x51,0x52,0x5d,0x61,0x62,0x64,0x65,0x6c,0x6f,0x73,0x74,0x75,0x7d,0x81,0x85,0x86,0x87,0x88,0x89,0x8f,0x90,0x91,0x92,0x93,0x94,0x97,0x98]);
  if (noArgument.has(opcode)) return offset;
  if ([0x4a,0x72,0x84].includes(opcode)) return checkedAdvance(bytes, offset, 4, "pickle opcode argument");
  if ([0x4b,0x68,0x71,0x82,0x80].includes(opcode)) return checkedAdvance(bytes, offset, 1, "pickle opcode argument");
  if ([0x4d,0x83].includes(opcode)) return checkedAdvance(bytes, offset, 2, "pickle opcode argument");
  if ([0x47,0x95].includes(opcode)) return checkedAdvance(bytes, offset, 8, "pickle opcode argument");
  if ([0x46,0x49,0x4c,0x50,0x53,0x56,0x67,0x70].includes(opcode)) return skipPickleLine(bytes, offset, "pickle line argument");
  if (opcode === 0x69) return skipPickleLine(bytes, skipPickleLine(bytes, offset, "pickle INST module"), "pickle INST name");
  if ([0x54,0x58,0x42].includes(opcode)) return skipPickleSized(bytes, offset, 4, "pickle 32-bit byte string");
  if ([0x55,0x43,0x8c,0x8a].includes(opcode)) return skipPickleSized(bytes, offset, 1, "pickle 8-bit byte string");
  if ([0x8d,0x8e,0x96].includes(opcode)) return skipPickleSized(bytes, offset, 8, "pickle 64-bit byte string");
  if (opcode === 0x8b) return skipPickleSized(bytes, offset, 4, "pickle LONG4");
  throw new Error(`Unsupported pickle opcode 0x${opcode.toString(16).padStart(2, "0")} at byte ${offset - 1}.`);
}

function skipPickleLine(bytes, offset, label) {
  const end = bytes.indexOf(0x0a, offset);
  if (end < 0) throw new Error(`${label} is truncated.`);
  return end + 1;
}

function skipPickleSized(bytes, offset, width, label) {
  requireRange(bytes, offset, width, label);
  let length = 0n;
  for (let index = 0; index < width; index += 1) length |= BigInt(bytes[offset + index]) << BigInt(index * 8);
  const size = safeNumber(length, `${label} length`);
  return checkedAdvance(bytes, offset + width, size, label);
}

function checkedAdvance(bytes, offset, length, label) {
  requireRange(bytes, offset, length, label);
  return offset + length;
}

function baseEnvelopeAnalysis(filename, format, contract, artifactRole) {
  return {
    format, filename, file_size: null, file_size_bytes: null, ops: [], tensors: [], operator_count: 0, tensor_count: 0,
    input_tensor_indices: [], output_tensor_indices: [], total_macs: null, total_macs_decimal: null, mac_confidence: "not_applicable",
    mac_assessment: { status: "not_applicable_no_executable_graph_materialized", compute_ops: 0, assessed_compute_ops: 0, not_assessed_compute_ops: 0, total_assessed_macs: 0, total_assessed_macs_decimal: "0" },
    safe_source_artifact: contract,
    artifact_role: artifactRole,
    format_extensions: { [format]: contract },
    findings: [], recommendations: [],
  };
}

function parseHdf5V23(bytes, cursor, version) {
  requireRange(bytes, cursor, 4, "HDF5 superblock v2/v3");
  const offsetSize = bytes[cursor + 1], lengthSize = bytes[cursor + 2], flags = bytes[cursor + 3];
  requireAddressSizes(offsetSize, lengthSize);
  let offset = cursor + 4;
  const baseAddress = readLeInteger(bytes, offset, offsetSize); offset += offsetSize;
  const superblockExtensionAddress = readLeInteger(bytes, offset, offsetSize); offset += offsetSize;
  const endOfFileAddress = readLeInteger(bytes, offset, offsetSize); offset += offsetSize;
  const rootObjectHeaderAddress = readLeInteger(bytes, offset, offsetSize); offset += offsetSize;
  requireRange(bytes, offset, 4, "HDF5 superblock checksum");
  return { superblock_version: version, offset_size: offsetSize, length_size: lengthSize, file_consistency_flags: flags, base_address: baseAddress, superblock_extension_address: superblockExtensionAddress, end_of_file_address: endOfFileAddress, root_object_header_address: rootObjectHeaderAddress, checksum_status: "present_not_validated" };
}

function parseHdf5V01(bytes, cursor, version) {
  requireRange(bytes, cursor, 20, "HDF5 superblock v0/v1");
  const offsetSize = bytes[cursor + 5], lengthSize = bytes[cursor + 6];
  requireAddressSizes(offsetSize, lengthSize);
  let offset = cursor + 16 + (version === 1 ? 4 : 0);
  const baseAddress = readLeInteger(bytes, offset, offsetSize); offset += offsetSize;
  const freeSpaceAddress = readLeInteger(bytes, offset, offsetSize); offset += offsetSize;
  const endOfFileAddress = readLeInteger(bytes, offset, offsetSize); offset += offsetSize;
  const driverInfoAddress = readLeInteger(bytes, offset, offsetSize); offset += offsetSize;
  requireRange(bytes, offset, offsetSize * 2, "HDF5 root group symbol table entry");
  const rootObjectHeaderAddress = readLeInteger(bytes, offset + offsetSize, offsetSize);
  return { superblock_version: version, offset_size: offsetSize, length_size: lengthSize, base_address: baseAddress, free_space_address: freeSpaceAddress, end_of_file_address: endOfFileAddress, driver_info_address: driverInfoAddress, root_object_header_address: rootObjectHeaderAddress, checksum_status: "not_present_in_superblock_version" };
}

function findHdf5Signature(bytes) {
  const signature = [0x89,0x48,0x44,0x46,0x0d,0x0a,0x1a,0x0a];
  for (let offset = 0; offset + 8 <= bytes.length; offset = offset === 0 ? 512 : offset * 2) {
    if (signature.every((value, index) => bytes[offset + index] === value)) return offset;
    if (offset > 1024 * 1024 * 1024) break;
  }
  return -1;
}

function profileCandidates(filename) {
  const lower = String(filename).toLowerCase();
  return [
    { id: "generic_hdf5", applicability: "applicable", status: "superblock_only" },
    { id: "keras_weights_h5", applicability: /weights.*\.h(?:df)?5$/.test(lower) ? "candidate_from_filename" : "unknown", status: "not_assessed_without_object_graph" },
    { id: "legacy_keras_model_h5", applicability: /\.h(?:df)?5$/.test(lower) ? "possible" : "unknown", status: "not_assessed_without_root_attributes" },
  ];
}

function kerasLayerInventory(config) {
  const layers = config?.config?.layers;
  if (!Array.isArray(layers)) return { status: "not_materialized", layers: [] };
  return {
    status: "declared_config_inventory",
    layers: layers.slice(0, 100_000).map((layer, index) => ({ index, name: String(layer?.config?.name || layer?.name || `layer_${index}`), class_name: String(layer?.class_name || "unknown") })),
    execution_order_claim: false,
  };
}

function parseJson(bytes, label) {
  try { return JSON.parse(decode(bytes, label)); }
  catch (error) { throw new Error(`${label} is not valid JSON: ${error?.message || error}`); }
}
function findEocd(bytes) {
  const minimum = Math.max(0, bytes.length - 65_557);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let offset = bytes.length - 22; offset >= minimum; offset -= 1) if (bytes[offset] === 0x50 && bytes[offset + 1] === 0x4b && bytes[offset + 2] === 0x05 && bytes[offset + 3] === 0x06) {
    const commentLength = view.getUint16(offset + 20, true);
    if (offset + 22 + commentLength !== bytes.length) continue;
    if (view.getUint16(offset + 4, true) !== 0 || view.getUint16(offset + 6, true) !== 0 || view.getUint16(offset + 8, true) !== view.getUint16(offset + 10, true)) throw new Error("Multi-disk ZIP archives are not supported.");
    return offset;
  }
  throw new Error("ZIP end-of-central-directory record was not found.");
}
function safeArchivePath(value) {
  const normalized = String(value).replaceAll("\\", "/");
  if (!normalized || normalized.startsWith("/") || /^[A-Za-z]:/.test(normalized) || normalized.includes("\0") || normalized.split("/").some((part) => !part || part === "." || part === "..")) throw new Error(`Unsafe ZIP member path ${JSON.stringify(value)}.`);
  return normalized;
}
function requiredString(fields, number, label) { const field = fields.find((item) => item.number === number && item.wire === 2); if (!field) throw new Error(`${label} is missing.`); const value = decode(field.bytes, label); if (!value || value.length > 100_000) throw new Error(`${label} is invalid.`); return value; }
function optionalString(fields, number) { const field = fields.find((item) => item.number === number && item.wire === 2); return field ? decode(field.bytes, `field ${number}`) : null; }
function readVarint(bytes, offset, label) { let value = 0n; for (let shift = 0n; shift <= 63n; shift += 7n) { if (offset >= bytes.length) throw new Error(`${label} contains a truncated varint.`); const byte = bytes[offset++]; value |= BigInt(byte & 0x7f) << shift; if (!(byte & 0x80)) return { value, next: offset }; } throw new Error(`${label} contains an overlong varint.`); }
function varintNumber(field) { return field?.wire === 0 ? safeNumber(field.value, "protobuf integer") : null; }
function safeNumber(value, label) { if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${label} exceeds the safe integer range.`); return Number(value); }
function decode(bytes, label) { try { return UTF8.decode(bytes); } catch { throw new Error(`${label} is not valid UTF-8.`); } }
function requireBytes(bytes, maximum, label) { if (!(bytes instanceof Uint8Array)) throw new Error(`${label} requires Uint8Array bytes.`); if (bytes.length > maximum) throw new Error(`${label} exceeds the ${maximum}-byte parser limit.`); }
function requireFile(file, label) { if (!file || typeof file.slice !== "function" || !Number.isSafeInteger(Number(file.size)) || Number(file.size) < 0) throw new Error(`${label} requires a bounded Blob-compatible file.`); }
function requireRange(bytes, offset, length, label) { if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || offset < 0 || length < 0 || offset + length > bytes.length) throw new Error(`${label} exceeds the source byte range.`); }
function requireAddressSizes(offsetSize, lengthSize) { if (![2,4,8,16,32].includes(offsetSize) || ![2,4,8,16,32].includes(lengthSize)) throw new Error(`Unsupported HDF5 address sizes offset=${offsetSize}, length=${lengthSize}.`); }
function readLeInteger(bytes, offset, size) { requireRange(bytes, offset, size, "HDF5 integer"); let value = 0n; for (let index = 0; index < size; index += 1) value |= BigInt(bytes[offset + index]) << BigInt(index * 8); const undefinedValue = (1n << BigInt(size * 8)) - 1n; return value === undefinedValue ? null : value.toString(); }
function crc32(bytes) { let value = 0xffffffff; for (const byte of bytes) { value ^= byte; for (let bit = 0; bit < 8; bit += 1) value = (value >>> 1) ^ ((value & 1) ? 0xedb88320 : 0); } return (value ^ 0xffffffff) >>> 0; }
