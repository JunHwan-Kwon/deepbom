import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const CYCLONEDX_COMPONENT_INVENTORY_SCHEMA = "deepbom.cyclonedx_component_inventory.v1";

export function buildCycloneDxComponentInventory(document) {
  if (document?.bomFormat !== "CycloneDX" || document?.specVersion !== "1.7") {
    throw new Error("Component inventory requires a CycloneDX 1.7 JSON document.");
  }
  const root = objectOrNull(document?.metadata?.component);
  const members = Array.isArray(document.components) ? document.components : [];
  const seen = new Map();
  const components = [];
  if (root) appendComponent(components, seen, root, "metadata.component", true);
  for (let index = 0; index < members.length; index += 1) {
    appendComponent(components, seen, members[index], `components[${index}]`, false);
  }
  const body = {
    schema: CYCLONEDX_COMPONENT_INVENTORY_SCHEMA,
    cyclonedx_spec_version: "1.7",
    root_component_present: Boolean(root),
    inventory_component_count: members.length,
    visible_component_count: components.length,
    components,
    consumer_contract: "metadata.component is the BOM root; components[] contains additional inventory members. Consumers must inspect both locations and must not require the root to be duplicated in components[].",
    interpretation_boundary: "This projection verifies component discovery and identity only. It does not validate the complete BOM, execute external references, or infer dependencies that are not serialized.",
  };
  return { ...body, inventory_sha256: sha256TextHex(canonicalJson(body)) };
}

export function validateCycloneDxComponentInventory(inventory) {
  if (inventory?.schema !== CYCLONEDX_COMPONENT_INVENTORY_SCHEMA) throw new Error("CycloneDX component inventory schema is invalid.");
  const rows = Array.isArray(inventory.components) ? inventory.components : null;
  if (!rows || inventory.visible_component_count !== rows.length) throw new Error("CycloneDX component inventory count is invalid.");
  if (inventory.root_component_present !== rows.some((row) => row.root === true)) throw new Error("CycloneDX root-component presence is inconsistent.");
  const { inventory_sha256: observed, ...body } = inventory;
  if (observed !== sha256TextHex(canonicalJson(body))) throw new Error("CycloneDX component inventory digest is invalid.");
  return inventory;
}

function appendComponent(output, seen, component, pointer, root) {
  if (!component || typeof component !== "object" || Array.isArray(component)) {
    throw new Error(`CycloneDX component at ${pointer} is not an object.`);
  }
  const bomRef = String(component["bom-ref"] || "");
  if (!bomRef) throw new Error(`CycloneDX component at ${pointer} has no bom-ref.`);
  if (seen.has(bomRef)) throw new Error(`CycloneDX component bom-ref ${bomRef} is duplicated at ${seen.get(bomRef)} and ${pointer}.`);
  seen.set(bomRef, pointer);
  output.push({
    bom_ref: bomRef,
    pointer,
    root,
    type: String(component.type || ""),
    name: String(component.name || ""),
    version: component.version == null ? null : String(component.version),
  });
}

function objectOrNull(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : null;
}
