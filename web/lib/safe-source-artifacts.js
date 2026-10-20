import { sha256BytesHex } from "./sha256-sync.js";
import { protobufWireFields, protobufVarint as readVarint } from "./portable-protobuf.js";

const MAX_PROTOBUF_BYTES = 1024 * 1024 * 1024;
const MAX_ZIP_ENTRIES = 100_000;
const MAX_ZIP_TEXT_BYTES = 16 * 1024 * 1024;
const MAX_ZIP_CENTRAL_DIRECTORY_BYTES = 64 * 1024 * 1024;
const MAX_ZIP_TEXT_COMPRESSION_RATIO = 200;
const UTF8 = new TextDecoder("utf-8", { fatal: true });
const tensorProtoPayloads = new WeakMap();

export function analyzeTensorFlowGraphDef(bytes, filename = "graph.pb") {
  requireBytes(bytes, MAX_PROTOBUF_BYTES, "TensorFlow GraphDef");
  const graph = parseGraphDef(bytes, "GraphDef");
  return graphDefAnalysis(graph, filename, "graphdef");
}

export function analyzeTensorFlowProtobuf(bytes, filename = "graph.pb", {metaGraphIndex = 0} = {}) {
  requireBytes(bytes, MAX_PROTOBUF_BYTES, "TensorFlow protobuf");
  if (/^saved_model\.pb$/i.test(String(filename).split(/[\\/]/).pop() || "")) {
    const graphs = parseSavedModelGraphDefs(bytes);
    if (!Number.isSafeInteger(metaGraphIndex) || metaGraphIndex < 0 || !graphs[metaGraphIndex]) throw new Error("Selected SavedModel MetaGraph does not exist");
    const selected = graphs[metaGraphIndex];
    const primary = graphDefAnalysis(selected, filename, "savedmodel", {
      source: `SavedModel.meta_graphs[${metaGraphIndex}].graph_def`,
      artifact_role: "model_package",
    });
    primary.savedmodel = {
      schema: "deepbom.tensorflow_savedmodel_safe_static_contract.v1",
      meta_graph_count: graphs.length,
      selected_meta_graph_index: metaGraphIndex,
      materialized_meta_graph_indices: [metaGraphIndex],
      signature_defs: selected.signatures,
      meta_graphs: graphs.map((graph,index) => ({index,tags:graph.tags,signatures:graph.signatures,functions:graph.functions,
        resource_variables:graph.nodes.filter(n=>n.op === "VarHandleOp").map(n=>({name:n.name,attributes:n.attributes})),
        object_graph:graph.object_graph, node_count:graph.nodes.length})),
      additional_meta_graphs: graphs.map((graph,index) => ({index,node_count:graph.nodes.length,function_count:graph.functions.length,signature_count:graph.signatures.length})).filter(row=>row.index!==metaGraphIndex),
      package_closure_status: "protobuf_only_variables_assets_and_fingerprints_not_bound",
      evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
      unsafe_execution_performed: false,
      interpretation_boundary: "The explicitly selected MetaGraph GraphDef and SignatureDef tensor contracts are materialized. Additional MetaGraphs are inventoried but not merged into one execution graph, and this standalone protobuf does not bind the SavedModel directory, variables, assets, fingerprints, kernels, or runtime behavior.",
    };
    primary.format_extensions.savedmodel = primary.savedmodel;
    return primary;
  }
  if (metaGraphIndex !== 0) throw new Error("MetaGraph selection applies only to SavedModel");
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
    await attachHdf5Objects(analysis, file);
    return { analysis, retainedBytes: bytes };
  }
  throw new Error("HDF5 signature was not found at an allowed user-block boundary within the bounded 1 GiB search range.");
}

export async function analyzeZipModelEnvelope(bytes, filename = "model.pt2", forcedFormat = "") {
  requireBytes(bytes, Number.MAX_SAFE_INTEGER, "ZIP model container");
  const archive = await parseZipDirectory(bytes);
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
  const { archive, tail } = await readZipDirectory(file.size, async (start, end) =>
    new Uint8Array(await file.slice(start, end).arrayBuffer()));
  const centralDirectorySize = archive.centralDirectorySize;
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
  if (format === "keras") {
    const weights = await readKerasWeightMember(file);
    if (weights) await attachHdf5Objects(analysis, weights.bytes, weights.path);
  }
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
    if (!graph) throw new Error(`SavedModel MetaGraph ${index} has no graph_def`);
    const parsed = parseGraphDef(graph.bytes, `SavedModel.meta_graphs[${index}].graph_def`);
    parsed.signatures = parseSignatureDefs(meta, `SavedModel.meta_graphs[${index}]`);
    const info = meta.find(f=>f.number===1&&f.wire===2);
    parsed.tags = info ? protobufFields(info.bytes,"MetaInfoDef",2).filter(f=>f.number===4&&f.wire===2).map(f=>decode(f.bytes,"MetaGraph tag")) : [];
    const object = meta.find(f=>f.number===7&&f.wire===2);
    parsed.object_graph = object ? parseSavedObjectGraph(object.bytes) : null;
    return parsed;
  });
}

export function graphDefAnalysis(graph, filename, format = "graphdef", supplemental = {}) {
  const nodeIndex = new Map(graph.nodes.map((node, index) => [node.name, index]));
  if (nodeIndex.size !== graph.nodes.length) throw new Error("GraphDef contains duplicate node names");
  const nodeByName = new Map(graph.nodes.map((node) => [node.name, node]));
  const endpoints = new Map();
  const endpoint = (name, port = 0) => {
    const key = `${name}:${port}`;
    if (!endpoints.has(key)) endpoints.set(key, endpoints.size);
    return endpoints.get(key);
  };
  for (const node of graph.nodes) for (const input of node.inputs) if (!input.control) endpoint(input.node, input.port);
  for (const node of graph.nodes) if (node.op === "Const" && node.attributes?.value?.tensor) endpoint(node.name, 0);
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
    outputs: (() => {
      const rows = [...endpoints.keys()].filter(key=>key.startsWith(`${node.name}:`)).map(key=>({index:endpoints.get(key),port:Number(key.slice(node.name.length+1))}));
      const ports = rows.length ? Array(rows.reduce((max,row)=>Math.max(max,row.port),0)+1).fill(-1) : [];
      for (const row of rows) ports[row.port]=row.index;
      return ports;
    })(),
    macs: null,
    macs_status: "not_assessed_operator_semantics_and_shapes_not_materialized",
  }));
  const tensors = [...endpoints.entries()].sort(([, left], [, right]) => left - right).map(([name, index]) => {
    const sourceName = name.replace(/:\d+$/, "");
    const contract = graphDefOutputContract(nodeByName.get(sourceName), signatureTensorInfo.get(name), Number(name.slice(name.lastIndexOf(":") + 1)));
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
    list_types: list ? protobufFields(list.bytes, `${label}.list`, 4).filter(f=>f.number===6).flatMap(f=>{
      if(f.wire===0)return [tensorFlowDtype(safeNumber(f.value,"DataType"))];
      if(f.wire!==2)throw new Error("Invalid AttrValue list type wire");
      const values=[];let offset=0;while(offset<f.bytes.length){const n=readVarint(f.bytes,offset,"DataType list");offset=n.next;values.push(tensorFlowDtype(safeNumber(n.value,"DataType")));}return values;
    }) : [],
    list_shapes: list ? protobufFields(list.bytes, `${label}.list`, 4).filter((field) => field.number === 7 && field.wire === 2).map((field, index) => parseTensorShape(field.bytes, `${label}.list.shape[${index}]`)) : [],
    function_ref: (() => { const fn=fields.find(f=>f.number===10&&f.wire===2); return fn ? optionalString(protobufFields(fn.bytes,`${label}.func`,4),1) : null; })(),
    scalar_kinds_present: fields.filter((field) => [2,3,4,5,9,10,11,12,13,14].includes(field.number)).map((field) => field.number),
  };
}

function parseTensorProto(bytes, label) {
  const fields = protobufFields(bytes, label, 4);
  const dtypeField = fields.find((field) => field.number === 1 && field.wire === 0);
  const shapeField = fields.find((field) => field.number === 2 && field.wire === 2);
  const content = fields.find((field) => field.number === 4 && field.wire === 2)?.bytes || null;
  const result = {
    dtype: dtypeField ? tensorFlowDtype(safeNumber(dtypeField.value, `${label}.dtype`)) : "UNKNOWN",
    shape: shapeField ? parseTensorShape(shapeField.bytes, `${label}.tensor_shape`) : [],
    tensor_content_byte_length: content?.length || 0,
    tensor_content_sha256: content ? sha256BytesHex(content) : null,
    typed_value_field_count: fields.filter((field) => field.number >= 5 && field.number <= 19).length,
  };
  tensorProtoPayloads.set(result, {fields, content});
  return result;
}

export function parseTensorShape(bytes, label) {
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

function graphDefOutputContract(node, signatureInfo = null, port = 0) {
  if (!node) return { dtype: "UNKNOWN", shape: [], constant: false, serialized_payload_bytes: 0, payload_sha256: null, status: "referenced_source_node_unresolved" };
  const tensor = port === 0 ? node.attributes?.value?.tensor || null : null;
  const declaredShape = node.attributes?._output_shapes?.list_shapes?.[port] ?? (port === 0 ? node.attributes?.shape?.shape ?? tensor?.shape : null) ?? null;
  const homogeneous = new Set(["Identity","IdentityN","Add","AddV2","Sub","Mul","RealDiv","Relu","Relu6","Sigmoid","Tanh","Conv2D","DepthwiseConv2dNative","MatMul","BatchMatMul","BatchMatMulV2","BiasAdd","Reshape","Transpose","ConcatV2","Split","SplitV"]);
  const typedOutput = ["Placeholder","PlaceholderWithDefault","ReadVariableOp"].includes(node.op) ? node.attributes?.dtype?.type
    : node.op === "VarHandleOp" ? "RESOURCE"
    : ["PartitionedCall","StatefulPartitionedCall"].includes(node.op) ? node.attributes?.Tout?.list_types?.[port]
    : node.op === "IdentityN" ? node.attributes?.T?.list_types?.[port]
    : homogeneous.has(node.op) ? node.attributes?.T?.type : null;
  const dtype = tensor?.dtype || signatureInfo?.dtype || typedOutput || "UNKNOWN";
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

export function tensorFlowDtype(value) {
  return ({ 1:"FLOAT32",2:"FLOAT64",3:"INT32",4:"UINT8",5:"INT16",6:"INT8",7:"STRING",8:"COMPLEX64",9:"INT64",10:"BOOL",14:"BFLOAT16",17:"UINT16",19:"FLOAT16",22:"UINT32",23:"UINT64" })[value] || `TF_DATATYPE_${value}`;
}

// TensorProto decoding reuses the shared scalar decoder for tensor_content.
// Typed repeated fields follow TensorFlow MakeNdarray's last-value padding,
// without constructing a Graph, Session, checkpoint object, or custom op.
export function tensorFlowNumericSources(bytes, {savedModel = false, metaGraphIndex = 0} = {}) {
  const graph = savedModel ? parseSavedModelGraphDefs(bytes)[metaGraphIndex] : parseGraphDef(bytes, "GraphDef");
  if (!graph) throw new Error("Selected SavedModel MetaGraph does not exist");
  return graph.nodes.filter(node => node.op === "Const" && node.attributes?.value?.tensor).map(node => {
    const tensor = node.attributes.value.tensor;
    return {name:`${node.name}:0`, dtype:tensor.dtype, shape:tensor.shape,
      visit:async (visitor, {maxValues = 100_000_000, signal} = {}) => {
        const {fields, content} = tensorProtoPayloads.get(tensor);
        const unavailable = reason => ({status:"not_assessed", reason});
        if (!Array.isArray(tensor.shape) || tensor.shape.some(d => !Number.isSafeInteger(d) || d < 0)) return unavailable("shape_not_exact");
        const count = tensor.shape.reduce((n, d) => n * BigInt(d), 1n);
        if (count > BigInt(maxValues)) return unavailable("value_budget_exceeded");
        if (fields.find(f => f.number === 3)?.value > 0n) return unavailable("tensor_proto_version_not_qualified");
        for (const number of [1,2,3,4]) if (fields.filter(f => f.number === number).length > 1) throw new Error("TensorProto repeats a singular field");
        const typedFields = fields.filter(f => f.number >= 5 && f.number <= 18);
        if (content?.length) {
          if (typedFields.length) throw new Error("TensorProto contains conflicting numeric representations");
          const {visitDeclaredNumericBytes} = await import("../onnx.js");
          signal?.throwIfAborted();
          const result = visitDeclaredNumericBytes(content, tensor.dtype, tensor.shape, visitor);
          return result.ok ? {status:"assessed", representation: /^COMPLEX/.test(tensor.dtype) ? "complex_magnitude" : "stored_scalar", payload_sha256:sha256BytesHex(content), invalid_encoding_count:0} : unavailable(result.reason);
        }
        const field = {FLOAT32:5,FLOAT64:6,INT32:7,INT16:7,UINT16:7,INT8:7,UINT8:7,INT64:10,BOOL:11,FLOAT16:13,BFLOAT16:13,UINT32:16,UINT64:17}[tensor.dtype];
        if (!field) return unavailable("tensor_proto_typed_encoding_not_qualified");
        if (typedFields.some(f => f.number !== field)) throw new Error("TensorProto typed payload contradicts dtype");
        const values = [];
        const append = value => { if (values.length >= maxValues || BigInt(values.length) >= count) throw new Error("TensorProto typed value count exceeds its declared shape or budget"); values.push(value); };
        for (const item of typedFields) {
          if (field === 5 || field === 6) {
            const width = field === 5 ? 4 : 8;
            if (![2, field === 5 ? 5 : 1].includes(item.wire) || item.bytes.length % width) throw new Error("TensorProto float field is truncated or has invalid wire type");
            const view = new DataView(item.bytes.buffer, item.bytes.byteOffset, item.bytes.byteLength);
            for (let offset = 0; offset < item.bytes.length; offset += width) append(width === 4 ? view.getFloat32(offset, true) : view.getFloat64(offset, true));
          } else if (item.wire === 0) append(item.value);
          else if (item.wire === 2) {
            for (let offset = 0; offset < item.bytes.length;) {const v = readVarint(item.bytes, offset, "TensorProto packed integer"); append(v.value); offset = v.next;}
          } else throw new Error("TensorProto integer field has invalid wire type");
          if (values.length > maxValues || BigInt(values.length) > count) throw new Error("TensorProto typed value count exceeds its declared shape or budget");
        }
        const {float16ToNumber, bfloat16ToNumber} = await import("./scalar-numeric.js");
        const convert = value => {
          if (field === 5 || field === 6) return value;
          if (field === 13) return tensor.dtype === "FLOAT16" ? float16ToNumber(Number(BigInt.asUintN(16, value))) : bfloat16ToNumber(Number(BigInt.asUintN(16, value)));
          if (tensor.dtype === "BOOL") return value === 0n ? 0 : 1;
          const width = Number(tensor.dtype.match(/\d+$/)[0]);
          const v = tensor.dtype.startsWith("UINT") ? BigInt.asUintN(width, value) : BigInt.asIntN(width, value);
          return width === 64 ? v : Number(v);
        };
        const fallback = values.length ? values.at(-1) : field === 5 || field === 6 ? 0 : 0n;
        for (let i = 0; i < Number(count); i++) { if (i % 65536 === 0) signal?.throwIfAborted(); visitor(convert(i < values.length ? values[i] : fallback)); }
        return {status:"assessed", representation:"stored_scalar", payload_sha256:null, invalid_encoding_count:0};
      }};
  });
}

function parseTensorFlowInput(value) {
  const control = value.startsWith("^");
  const body = control ? value.slice(1) : value;
  const match = /^(.*?)(?::(\d+))?$/.exec(body);
  if (!match || !match[1]) throw new Error(`Invalid TensorFlow NodeDef input reference ${JSON.stringify(value)}.`);
  const port = match[2] == null ? 0 : Number(match[2]);
  if (!Number.isSafeInteger(port) || port < 0 || port > 100_000) throw new Error(`Invalid TensorFlow output port in ${JSON.stringify(value)}.`);
  return { serialized: value, node: match[1], port, control };
}

function parseFunctionLibrary(bytes, label) {
  const seen = new Set();
  return protobufFields(bytes, label, 1).filter(field=>field.number===1&&field.wire===2).map((field,index) => {
    const fields = protobufFields(field.bytes, `${label}.function[${index}]`,2);
    const signature = fields.find(f=>f.number===1&&f.wire===2);
    const sig = signature ? protobufFields(signature.bytes,`${label}.signature`,3) : [];
    const name = optionalString(sig,1);
    if (!name || seen.has(name)) throw new Error("Missing or repeated FunctionDef name"); seen.add(name);
    const args = number => sig.filter(f=>f.number===number&&f.wire===2).map(f=>{const a=protobufFields(f.bytes,"ArgDef",4);return {name:optionalString(a,1),dtype:tensorFlowDtype(varintNumber(a.find(f=>f.number===3))||0),type_attr:optionalString(a,4),number_attr:optionalString(a,5),type_list_attr:optionalString(a,6)};});
    const map = number => fields.filter(f=>f.number===number&&f.wire===2).map(f=>{const a=protobufFields(f.bytes,"FunctionDef.map",3);return {key:requiredString(a,1,"map key"),value:requiredString(a,2,"map value")};});
    const nodes = fields.filter(f=>f.number===3&&f.wire===2).map((f,i)=>parseNodeDef(f.bytes,`${label}.node[${i}]`));
    return {index,name,node_count:nodes.length,return_entry_count:map(4).length,control_return_entry_count:map(6).length,
      inputs:args(2),outputs:args(3),nodes,returns:map(4),control_returns:map(6),
      status:"serialized_function_definition_not_inlined_or_counted_as_execution"};
  });
}

function parseSavedObjectGraph(bytes) {
  const fields=protobufFields(bytes,"SavedObjectGraph",1);
  const nodes=fields.filter(f=>f.number===1&&f.wire===2).map((field,index)=>{
    const row=protobufFields(field.bytes,"SavedObject",2), variable=row.find(f=>f.number===7&&f.wire===2);
    const refs=number=>row.filter(f=>f.number===number&&f.wire===2).map(f=>{const ref=protobufFields(f.bytes,"ObjectReference",3);return {node_id:varintNumber(ref.find(f=>f.number===1))??0,local_name:optionalString(ref,2)};});
    const vf=variable?protobufFields(variable.bytes,"SavedVariable",3):[];
    const shape=vf.find(f=>f.number===2&&f.wire===2);
    return {index,children:refs(1),dependencies:refs(15),variable:variable?{
      name:optionalString(vf,6),dtype:tensorFlowDtype(varintNumber(vf.find(f=>f.number===1))||0),shape:shape?parseTensorShape(shape.bytes,"SavedVariable.shape"):null,
      trainable:varintNumber(vf.find(f=>f.number===3))===1}:null};
  });
  for(const row of nodes) for(const ref of [...row.children,...row.dependencies]) if(ref.node_id<0||ref.node_id>=nodes.length) throw new Error("SavedObjectGraph reference outside its node inventory");
  return {nodes,status:"serialized_object_relationships_not_checkpoint_key_mapping"};
}

function parseVersionDef(bytes) {
  const fields = protobufFields(bytes, "VersionDef", 1);
  return {
    producer: varintNumber(fields.find((field) => field.number === 1)),
    min_consumer: varintNumber(fields.find((field) => field.number === 2)),
    bad_consumers: fields.filter((field) => field.number === 3).map(varintNumber).filter((value) => value != null),
  };
}

export function protobufFields(bytes, label, depth) {
  if (depth > 8) throw new Error(`${label} exceeds the bounded protobuf nesting depth.`);
  return protobufWireFields(bytes, label).map(({ no: number, wire, value, offset, length }) => wire === 0
    ? { number, wire, value: BigInt(value) }
    : { number, wire, bytes: bytes.subarray(offset, offset + length) });
}

async function parseZipDirectory(bytes) {
  return (await readZipDirectory(bytes.length, async (start, end) => bytes.subarray(start, end))).archive;
}

// PKWARE APPNOTE 6.3.10, sections 4.3.14–16 and 4.5.3. Full and range readers
// share the same bounds and exact 64-bit decoding; addresses must fit JS indexing.
async function readZipDirectory(fileSize, read) {
  const range = async (start, end) => {
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start || end > fileSize) throw new Error("ZIP record exceeds the source byte range.");
    const bytes = await read(start, end);
    if (bytes.length !== end - start) throw new Error("ZIP record read was truncated.");
    return bytes;
  };
  const tailStart = Math.max(0, fileSize - 65_577);
  const tail = await range(tailStart, fileSize);
  const localEocd = findEocd(tail), eocd = tailStart + localEocd;
  const v = new DataView(tail.buffer, tail.byteOffset, tail.byteLength);
  let entryCount = v.getUint16(localEocd + 10, true);
  let centralDirectorySize = v.getUint32(localEocd + 12, true);
  let centralDirectoryOffset = v.getUint32(localEocd + 16, true);
  const legacy = [entryCount, centralDirectorySize, centralDirectoryOffset];
  let directoryEnd = eocd;
  const locator = eocd >= 20 ? await range(eocd - 20, eocd) : new Uint8Array();
  const lv = new DataView(locator.buffer, locator.byteOffset, locator.byteLength);
  const zip64 = locator.length === 20 && lv.getUint32(0, true) === 0x07064b50;
  if (zip64) {
    if (lv.getUint32(4, true) !== 0 || lv.getUint32(16, true) !== 1) throw new Error("Multi-disk ZIP64 archives are not supported.");
    directoryEnd = safeNumber(lv.getBigUint64(8, true), "ZIP64 directory record offset");
    const record = await range(directoryEnd, directoryEnd + 56);
    const rv = new DataView(record.buffer, record.byteOffset, record.byteLength);
    const recordSize = safeNumber(rv.getBigUint64(4, true), "ZIP64 record size");
    if (rv.getUint32(0, true) !== 0x06064b50 || recordSize < 44 || recordSize > 1024 * 1024 || directoryEnd + 12 + recordSize !== eocd - 20) throw new Error("ZIP64 directory record bounds are invalid.");
    if (rv.getUint32(16, true) !== 0 || rv.getUint32(20, true) !== 0 || rv.getBigUint64(24, true) !== rv.getBigUint64(32, true)) throw new Error("Multi-disk ZIP64 archives are not supported.");
    entryCount = safeNumber(rv.getBigUint64(32, true), "ZIP64 entry count");
    centralDirectorySize = safeNumber(rv.getBigUint64(40, true), "ZIP64 directory size");
    centralDirectoryOffset = safeNumber(rv.getBigUint64(48, true), "ZIP64 directory offset");
    [entryCount, centralDirectorySize, centralDirectoryOffset].forEach((value, i) => {
      if (legacy[i] !== [0xffff, 0xffffffff, 0xffffffff][i] && legacy[i] !== value) throw new Error("ZIP64 and legacy directory fields disagree.");
    });
  } else if (entryCount === 0xffff || centralDirectorySize === 0xffffffff || centralDirectoryOffset === 0xffffffff) throw new Error("ZIP64 directory locator is missing.");
  validateZipDirectoryBounds(entryCount, centralDirectorySize, centralDirectoryOffset, fileSize, directoryEnd);
  const central = await range(centralDirectoryOffset, centralDirectoryOffset + centralDirectorySize);
  const archive = parseZipDirectoryRecords(central, entryCount, centralDirectoryOffset, centralDirectorySize);
  return { archive, tail };
}

function zip64Extra(extra, fields) {
  let pos = 0, data = null;
  const view = new DataView(extra.buffer, extra.byteOffset, extra.byteLength);
  while (pos < extra.length) {
    requireRange(extra, pos, 4, "ZIP extra header");
    const kind = view.getUint16(pos, true), size = view.getUint16(pos + 2, true);
    requireRange(extra, pos + 4, size, "ZIP extra field");
    if (kind === 1) {
      if (data) throw new Error("ZIP entry repeats ZIP64 extra field.");
      data = extra.subarray(pos + 4, pos + 4 + size);
    }
    pos += 4 + size;
  }
  let cursor = 0;
  return fields.map(([value, sentinel, width, label]) => {
    if (value !== sentinel) return value;
    if (!data) throw new Error("ZIP entry requires a ZIP64 extra field.");
    requireRange(data, cursor, width, label);
    const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
    const result = width === 8 ? safeNumber(dv.getBigUint64(cursor, true), label) : dv.getUint32(cursor, true);
    cursor += width;
    return result;
  });
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
    let compressedSize = view.getUint32(offset + 20, true);
    let uncompressedSize = view.getUint32(offset + 24, true);
    const nameLength = view.getUint16(offset + 28, true);
    const extraLength = view.getUint16(offset + 30, true);
    const commentLength = view.getUint16(offset + 32, true);
    let localHeaderOffset = view.getUint32(offset + 42, true);
    let disk = view.getUint16(offset + 34, true);
    requireRange(bytes, offset + 46, nameLength + extraLength + commentLength, "ZIP central record payload");
    [uncompressedSize, compressedSize, localHeaderOffset, disk] = zip64Extra(bytes.subarray(offset + 46 + nameLength, offset + 46 + nameLength + extraLength), [
      [uncompressedSize, 0xffffffff, 8, "ZIP64 uncompressed size"],
      [compressedSize, 0xffffffff, 8, "ZIP64 compressed size"],
      [localHeaderOffset, 0xffffffff, 8, "ZIP64 local offset"],
      [disk, 0xffff, 4, "ZIP64 disk"],
    ]);
    if (disk !== 0) throw new Error("Multi-disk ZIP entries are not supported.");
    if (!Number.isSafeInteger(localHeaderOffset + 30 + compressedSize) || localHeaderOffset + 30 + compressedSize > centralDirectoryOffset) throw new Error("ZIP member overlaps or exceeds the central directory.");
    const memberPath = safeArchivePath(decode(bytes.subarray(offset + 46, offset + 46 + nameLength), `ZIP entry ${index}`));
    const folded = memberPath.toLowerCase();
    if (names.has(folded)) throw new Error(`ZIP archive repeats or case-collides path ${memberPath}.`);
    names.add(folded);
    if (flags & 1) throw new Error(`Encrypted ZIP entry is not accepted: ${memberPath}.`);
    entries.push({ path: memberPath, flags, compressionMethod, crc32, compressedSize, uncompressedSize, localHeaderOffset, dataEnd: centralDirectoryOffset, encrypted: false });
    offset += 46 + nameLength + extraLength + commentLength;
  }
  if (offset !== centralDirectorySize) throw new Error("ZIP central-directory size does not conserve parsed records.");
  return { entries, centralDirectoryOffset, centralDirectorySize };
}

function validateZipDirectoryBounds(entryCount, centralDirectorySize, centralDirectoryOffset, fileSize, eocdOffset) {
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
  return extractZipFileEntry({ size: bytes.length, slice: (start, end) => ({ arrayBuffer: async () => bytes.slice(start, end).buffer }) }, entry);
}

async function extractZipFileEntry(file, entry, maximum = MAX_ZIP_TEXT_BYTES) {
  validateExtractedZipEntry(entry, maximum);
  const local = new Uint8Array(await file.slice(entry.localHeaderOffset, entry.localHeaderOffset + 30).arrayBuffer());
  requireRange(local, 0, 30, `ZIP local header ${entry.path}`);
  const view = new DataView(local.buffer, local.byteOffset, local.byteLength);
  if (view.getUint32(0, true) !== 0x04034b50) throw new Error(`ZIP local header signature is invalid for ${entry.path}.`);
  const flags = view.getUint16(6, true), method = view.getUint16(8, true);
  const nameLength = view.getUint16(26, true), extraLength = view.getUint16(28, true);
  if (flags !== entry.flags || (flags & 1) || method !== entry.compressionMethod) throw new Error(`ZIP local header conflicts with the central directory for ${entry.path}.`);
  const start = entry.localHeaderOffset + 30 + nameLength + extraLength;
  if (!Number.isSafeInteger(start + entry.compressedSize) || start + entry.compressedSize > Math.min(file.size, entry.dataEnd)) throw new Error(`ZIP member ${entry.path} exceeds the source byte range.`);
  const headerPayload = new Uint8Array(await file.slice(entry.localHeaderOffset + 30, start).arrayBuffer());
  requireRange(headerPayload, 0, nameLength + extraLength, "ZIP local extra fields");
  const localName = safeArchivePath(decode(headerPayload.subarray(0, nameLength), `ZIP local entry ${entry.path}`));
  if (localName !== entry.path) throw new Error(`ZIP local and central paths disagree for ${entry.path}.`);
  if (!(flags & 8)) {
    const [size, compressed] = zip64Extra(headerPayload.subarray(nameLength), [
      [view.getUint32(22, true), 0xffffffff, 8, "ZIP64 local uncompressed size"],
      [view.getUint32(18, true), 0xffffffff, 8, "ZIP64 local compressed size"],
    ]);
    if (size !== entry.uncompressedSize || compressed !== entry.compressedSize || view.getUint32(14, true) !== entry.crc32) throw new Error(`ZIP local sizes/CRC disagree for ${entry.path}.`);
  }
  const compressed = new Uint8Array(await file.slice(start, start + entry.compressedSize).arrayBuffer());
  if (compressed.length !== entry.compressedSize) throw new Error(`ZIP member ${entry.path} is truncated.`);
  return decompressAndVerifyZipEntry(compressed, entry, maximum);
}

function validateExtractedZipEntry(entry, maximum = MAX_ZIP_TEXT_BYTES) {
  if (entry.uncompressedSize > maximum || entry.compressedSize > maximum) throw new Error(`ZIP declarative member ${entry.path} exceeds the ${maximum}-byte extraction limit.`);
  if (entry.compressedSize > 0 && entry.uncompressedSize / entry.compressedSize > MAX_ZIP_TEXT_COMPRESSION_RATIO) throw new Error(`ZIP declarative member ${entry.path} exceeds the ${MAX_ZIP_TEXT_COMPRESSION_RATIO}:1 compression-ratio limit.`);
}

async function decompressAndVerifyZipEntry(compressed, entry, maximum = MAX_ZIP_TEXT_BYTES) {
  validateExtractedZipEntry(entry, maximum);
  let output;
  if (entry.compressionMethod === 0) output = new Uint8Array(compressed);
  else if (entry.compressionMethod === 8) {
    if (typeof DecompressionStream !== "function") throw new Error(`Deflate decompression is unavailable for ${entry.path}.`);
    const reader = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("deflate-raw")).getReader();
    const chunks = []; let length = 0;
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        length += value.length;
        if (length > entry.uncompressedSize || length > maximum) throw new Error(`ZIP member ${entry.path} exceeds its decompression budget.`);
        chunks.push(value);
      }
    } finally { await reader.cancel(); reader.releaseLock(); }
    output = new Uint8Array(length); let offset = 0;
    for (const chunk of chunks) { output.set(chunk, offset); offset += chunk.length; }
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
  const memo = new Set();
  while (offset < bytes.length && !stopped) {
    const opcode = bytes[offset];
    opcodeCounts.set(`0x${opcode.toString(16).padStart(2, "0")}`, (opcodeCounts.get(`0x${opcode.toString(16).padStart(2, "0")}`) || 0) + 1);
    offset += 1;
    if ([0x67, 0x68, 0x6a, 0x70, 0x71, 0x72, 0x94].includes(opcode)) {
      let index;
      if (opcode === 0x94) index = memo.size;
      else if (opcode === 0x67 || opcode === 0x70) {
        const end = skipPickleLine(bytes, offset, "pickle memo index");
        const text = decode(bytes.subarray(offset, end - 1), "pickle memo index");
        if (!/^[0-9]+$/.test(text)) throw new Error("Invalid pickle memo index.");
        index = Number(text); offset = end;
      } else {
        const width = [0x6a, 0x72].includes(opcode) ? 4 : 1;
        requireRange(bytes, offset, width, "pickle memo index");
        index = width === 4 ? new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(offset, true) : bytes[offset];
        offset += width;
      }
      if (!Number.isSafeInteger(index) || index > 1_000_000) throw new Error("Pickle memo index exceeds the bounded inventory budget.");
      if ([0x67, 0x68, 0x6a].includes(opcode)) {
        if (!memo.has(index)) throw new Error("Pickle references an undefined memo index.");
      } else memo.add(index);
      continue;
    }
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
  analysis.mac_assessment = {
    status: "not_assessed", metric_scope: "serialized Keras config calls; no execution or lowering",
    compute_ops: 0, assessed_compute_ops: 0, not_assessed_compute_ops: 0,
    total_assessed_macs: 0, total_assessed_macs_decimal: "0",
  };
  analysis.keras = {
    schema: "deepbom.keras_declarative_layer_graph.v1", graph_materialized: true,
    root_class_name: rootClass, primary_config_path: configRow.path,
    layer_count: layers.length, sequential_order_used: rootClass === "Sequential",
    unsafe_deserialization_performed: false, custom_layer_execution_claim: false,
  };
  materializeKerasCalls(analysis, document, configRow.path);
  analysis.format_extensions.keras = analysis.keras;
}

function kerasReferences(value) {
  if (!value || typeof value !== "object") return [];
  if (Array.isArray(value)) {
    if (typeof value[0] === "string" && Number.isSafeInteger(value[1]) && Number.isSafeInteger(value[2])) return [{layer:value[0], call:value[1], port:value[2]}];
    return value.flatMap(kerasReferences);
  }
  if (Array.isArray(value.keras_history)) {
    const refs = kerasReferences(value.keras_history);
    return refs.map(ref => ({...ref, shape:kerasStaticShape(value.shape), dtype:typeof value.dtype === "string" ? value.dtype.toUpperCase() : null}));
  }
  return Object.values(value).flatMap(kerasReferences);
}

function materializeKerasCalls(analysis, document, configPath) {
  const modules = [], bodies = [];
  const walk = (model, prefix = "", depth = 0) => {
    if (depth > 64 || modules.length > 100_000) throw new Error("Keras module inventory exceeds its bounded depth or count");
    const names = new Set();
    for (const layer of model.config?.layers || []) {
      const name = layer.config?.name || layer.name;
      if (typeof name !== "string" || !name || names.has(name)) throw new Error("Keras module name is missing or duplicated within its parent");
      names.add(name);
      const path = prefix ? `${prefix}/${name}` : name;
      modules.push({path, parent:prefix || null, class_name:layer.class_name, config:layer.config || {}, evidence_basis:"serialized_config_containment"});
      if (Array.isArray(layer.config?.layers)) {
        bodies.push({module_path:path, input_refs:kerasReferences(layer.config.input_layers), output_refs:kerasReferences(layer.config.output_layers),
          calls:layer.config.layers.flatMap(child => (child.inbound_nodes || []).map((call, index) => ({module:child.config?.name || child.name, invocation:index, input_refs:kerasReferences(call)}))),
          status:"declarative_body_not_inlined_into_parent_calls"});
        walk(layer, path, depth + 1);
      }
    }
  };
  walk(document);
  const layers = document.config.layers, sequential = document.class_name === "Sequential", calls = [], byCall = new Map(), values = new Map();
  const key = ref => JSON.stringify([ref.layer, ref.call]);
  const valueKey = ref => JSON.stringify([ref.layer, ref.call, ref.port]);
  for (const [index, layer] of layers.entries()) {
    const name = layer.config?.name || layer.name;
    const inbound = layer.inbound_nodes?.length ? layer.inbound_nodes : [null];
    for (const [invocation, input] of inbound.entries()) {
      const refs = input ? kerasReferences(input) : sequential && index > 0 ? [{layer:layers[index - 1].config?.name || layers[index - 1].name, call:0, port:0}] : [];
      const call = {layer:name, invocation, definition:layer, refs, index:calls.length};
      calls.push(call); byCall.set(key({layer:name,call:invocation}), call);
      if (calls.length > 100_000) throw new Error("Keras invocation count exceeds 100,000");
    }
  }
  const ensure = ref => {
    if (!byCall.has(key(ref)) || ref.call < 0 || ref.port < 0 || ref.port > 1024) throw new Error("Keras history references an unresolved invocation or unbounded port");
    const id = valueKey(ref);
    const existing = values.get(id);
    if (existing) {
      if (ref.shape && existing.shape && JSON.stringify(existing.shape) !== JSON.stringify(ref.shape)) throw new Error("Keras tensor histories disagree on shape");
      if (ref.dtype && existing.dtype && existing.dtype !== ref.dtype) throw new Error("Keras tensor histories disagree on dtype");
      existing.shape ||= ref.shape; existing.dtype ||= ref.dtype;
    } else values.set(id, {...ref, index:values.size});
    return values.get(id).index;
  };
  for (const call of calls) {
    for (const ref of call.refs) ensure(ref);
    if (sequential || call.definition.class_name === "InputLayer") ensure({layer:call.layer,call:call.invocation,port:0,
      ...(call.definition.class_name === "InputLayer" ? {shape:kerasStaticShape(call.definition.config?.batch_shape ?? call.definition.config?.batch_input_shape), dtype:typeof call.definition.config?.dtype === "string" ? call.definition.config.dtype.toUpperCase() : null} : {})});
  }
  const inputs = kerasReferences(document.config.input_layers), outputs = kerasReferences(document.config.output_layers);
  inputs.forEach(ensure); outputs.forEach(ensure);
  analysis.tensors = [...values.values()].map(value => ({index:value.index, name:`${value.layer}@${value.call}:output:${value.port}`,
    shape:value.shape || [], shape_signature:value.shape || [], shape_declared:Array.isArray(value.shape), dtype:value.dtype || "UNKNOWN",
    value_kind:"keras_declared_layer_output", constant_buffer:false,
    contract_status:value.shape ? "serialized_keras_tensor_contract" : "serialized_layer_output_name_shape_not_assessable"}));
  const valuesByCall = new Map();
  for (const value of values.values()) { const id=key(value); if(!valuesByCall.has(id))valuesByCall.set(id,[]); valuesByCall.get(id).push(value); }
  analysis.ops = calls.map(call => {
    const ports = valuesByCall.get(key({layer:call.layer,call:call.invocation})) || [];
    const maxPort = ports.reduce((n, value) => Math.max(n, value.port), -1);
    const outputRefs = Array.from({length:maxPort + 1}, (_, port) => values.get(valueKey({layer:call.layer,call:call.invocation,port}))?.index ?? -1);
    return {index:call.index, name:call.definition.class_name, graph_node_name:`${call.layer}@${call.invocation}`, domain:"keras.config", version:null,
      inputs:call.refs.map(ref => values.get(valueKey(ref)).index), outputs:outputRefs, macs:null,
      source_module_path:call.layer, source_call_index:call.invocation,
      source_annotation:{scope_names:[call.layer],class_names:[call.definition.class_name],declared_call_index:call.invocation,basis:"serialized_keras_config"},
      macs_status:"not_assessed_layer_class_does_not_establish_lowered_operator_semantics"};
  });
  analysis.input_tensor_indices = inputs.length ? inputs.map(ref => values.get(valueKey(ref)).index) : calls.filter(c => c.definition.class_name === "InputLayer" || sequential && c.index === 0).map(c => values.get(valueKey({layer:c.layer,call:c.invocation,port:0}))?.index).filter(Number.isSafeInteger);
  analysis.output_tensor_indices = outputs.length ? outputs.map(ref => values.get(valueKey(ref)).index) : sequential && calls.length ? [values.get(valueKey({layer:calls.at(-1).layer,call:0,port:0})).index] : [];
  analysis.operator_count = calls.length; analysis.tensor_count = values.size;
  analysis.mac_assessment.compute_ops = analysis.mac_assessment.not_assessed_compute_ops = calls.length;
  Object.assign(analysis.keras, {invocation_count:calls.length, module_inventory:modules, nested_declarative_bodies:bodies,
    explicit_inbound_reference_count:calls.reduce((n, call) => n + call.refs.length, 0), primary_config_path:configPath,
    interpretation_boundary:"Module containment, call indices, tensor ports and declared tensor contracts are preserved separately. Nested model bodies are inventoried, not inlined or executed. A build input shape is not an output shape. Functional operations absent from config, lowering, kernel behavior and numerical quality remain unobserved."});
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

function varintNumber(field) { return field?.wire === 0 ? safeNumber(field.value, "protobuf integer") : null; }
function safeNumber(value, label) { if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${label} exceeds the safe integer range.`); return Number(value); }
function decode(bytes, label) { try { return UTF8.decode(bytes); } catch { throw new Error(`${label} is not valid UTF-8.`); } }
function requireBytes(bytes, maximum, label) { if (!(bytes instanceof Uint8Array)) throw new Error(`${label} requires Uint8Array bytes.`); if (bytes.length > maximum) throw new Error(`${label} exceeds the ${maximum}-byte parser limit.`); }
function requireFile(file, label) { if (!file || typeof file.slice !== "function" || !Number.isSafeInteger(Number(file.size)) || Number(file.size) < 0) throw new Error(`${label} requires a bounded Blob-compatible file.`); }
function requireRange(bytes, offset, length, label) { if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || offset < 0 || length < 0 || offset + length > bytes.length) throw new Error(`${label} exceeds the source byte range.`); }
function requireAddressSizes(offsetSize, lengthSize) { if (![2,4,8,16,32].includes(offsetSize) || ![2,4,8,16,32].includes(lengthSize)) throw new Error(`Unsupported HDF5 address sizes offset=${offsetSize}, length=${lengthSize}.`); }
function readLeInteger(bytes, offset, size) { requireRange(bytes, offset, size, "HDF5 integer"); let value = 0n; for (let index = 0; index < size; index += 1) value |= BigInt(bytes[offset + index]) << BigInt(index * 8); const undefinedValue = (1n << BigInt(size * 8)) - 1n; return value === undefinedValue ? null : value.toString(); }
function crc32(bytes) { let value = 0xffffffff; for (const byte of bytes) { value ^= byte; for (let bit = 0; bit < 8; bit += 1) value = (value >>> 1) ^ ((value & 1) ? 0xedb88320 : 0); } return (value ^ 0xffffffff) >>> 0; }


export async function readKerasWeightMember(file) {
  requireFile(file, "Keras weight container");
  const {archive} = await readZipDirectory(file.size, async (start,end) => new Uint8Array(await file.slice(start,end).arrayBuffer()));
  const entries = archive.entries.filter(entry => entry.path === "model.weights.h5");
  if (!entries.length) return null;
  if (entries.length !== 1) throw new Error("Keras archive repeats model.weights.h5");
  return {path:entries[0].path, bytes:await extractZipFileEntry(file,entries[0],128*1024*1024)};
}

async function attachHdf5Objects(analysis, source, memberPath = null) {
  try {
    const {inspectHdf5Objects, applyHdf5Inventory} = await import("./hdf5-source.js");
    applyHdf5Inventory(analysis,await inspectHdf5Objects(source),{memberPath});
  } catch (error) {
    analysis.format_extensions ||= {};
    analysis.format_extensions.hdf5_objects = {status:"not_assessed", reason:String(error.message), member_path:memberPath};
  }
}
