import { portableContract } from "./portable-contracts.js";
import { canonicalJsonInput } from "./report-utils.js";
const irContract = (operation, input) => portableContract("ir_semantics", canonicalJsonInput({ operation, ...input }));
import { buildValueType, valueElementCount } from "./ir-value-type.js";


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
  irContract("logical_inventory", {inventory,graph_values:graphValues,objects});
}
