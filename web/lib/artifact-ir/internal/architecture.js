import { GRAPH_FORMATS } from "./constants.js";
import { clone, exact, exactInteger, list, nonNegativeInteger, positiveStorageBytes, storageId, tensorIndex } from "./shared.js";

export function buildArchitectureProjection(analysis, format, tensors) {
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

function architectureLayerId(index) { return `architecture:layer:${index}`; }
