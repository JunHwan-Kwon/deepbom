import { buildValueType, valueElementCount, validateValueType } from "./ir-value-type.js";
import { canonicalJson } from "./report-utils.js";

export function buildLogicalInventory(analysis, graph, storage, format) {
  const objects = new Map(storage.objects.filter(row => !row.scope_ref && row.id.startsWith("storage:tensor:")).map(row => [row.native_index, row.id]));
  const values = graph.status === "serialized" ? graph.values.map(row => ({ id: row.id, name: row.name, dtype: row.dtype, shape: [...row.shape],
    type_contract: row.type_contract, graph_value_ref: row.id, storage_refs: [...row.storage_refs], element_count: valueElementCount(row.type_contract) }))
    : (analysis.tensors || []).map((row, position) => {
      const index = Number.isSafeInteger(row.index) && row.index >= 0 ? row.index : position;
      const type = buildValueType(row, format, "inventory:primary");
      return { id: `inventory:tensor:${index}`, name: String(row.name || `tensor_${index}`), dtype: String(row.dtype || "UNKNOWN").toUpperCase(), shape: Array.isArray(row.shape) ? row.shape.map(d => d == null ? "?" : typeof d === "bigint" ? d.toString() : d) : [],
        type_contract: type, graph_value_ref: null, storage_refs: objects.has(index) ? [objects.get(index)] : [], element_count: valueElementCount(type) };
    });
  return { schema: "deepbom.logical_inventory.v1", values, count: values.length,
    interpretation_boundary: "Logical values include empty tensors. Inventory membership does not establish an executable graph or serialized payload storage." };
}

export function validateLogicalInventory(inventory, graphValues, objects) {
  if (inventory?.schema !== "deepbom.logical_inventory.v1" || !Array.isArray(inventory.values) || inventory.count !== inventory.values.length) throw new Error("IR logical inventory count is invalid.");
  const seen = new Set(), graph = new Map(graphValues.map(row => [row.id, row])), storage = new Set(objects.map(row => row.id));
  for (const row of inventory.values) {
    if (!row.id || seen.has(row.id)) throw new Error("IR logical inventory has duplicate identities.");
    seen.add(row.id); validateValueType(row.type_contract);
    if (canonicalJson(row.element_count) !== canonicalJson(valueElementCount(row.type_contract))) throw new Error("IR logical element count contradicts the type.");
    if (!Array.isArray(row.storage_refs) || row.storage_refs.some(ref => !storage.has(ref))) throw new Error("IR logical storage binding is invalid.");
    if (row.graph_value_ref !== null) {
      const source = graph.get(row.graph_value_ref);
      if (!source || canonicalJson(row.type_contract) !== canonicalJson(source.type_contract) || canonicalJson(row.storage_refs) !== canonicalJson(source.storage_refs)) throw new Error("IR logical inventory contradicts its graph value.");
    }
  }
  if (graphValues.some(row => !seen.has(row.id))) throw new Error("IR logical inventory omitted a graph value.");
}
