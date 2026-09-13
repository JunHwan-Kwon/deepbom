// Public DeepBOM vendor-property contract used by both export and reconciliation.
// These names describe DeepBOM output only; they are not standard fields and do
// not imply registration or endorsement by any external standards body.

export const DEEPBOM_PROPERTY_TAXONOMY_SCHEMA = "deepbom.property_taxonomy.v1";

export const DEEPBOM_RECONCILABLE_COMPONENT_PROPERTIES = Object.freeze([
  "deepbom:contract:schema",
  "deepbom:model:artifactIrSchema",
  "deepbom:model:artifactIrSha256",
  "deepbom:model:format",
  "deepbom:model:fileSizeBytes",
  "deepbom:model:hashBasis",
  "deepbom:model:storageEncodingClassification",
  "deepbom:model:quantizationClassification",
  "deepbom:model:llmEncodingInventorySha256",
  "deepbom:model:llmTensorEncodingAssignmentSha256",
  "deepbom:model:llmChatTemplateSha256",
  "deepbom:model:interfaceContractSchema",
  "deepbom:model:interfaceContractLedgerSha256",
  "deepbom:model:completeAffineInterfaceCount",
  "deepbom:model:unquantizedInterfaceCount",
  "deepbom:model:invalidOrIncompleteInterfaceCount",
]);

export const DEEPBOM_STRUCTURED_EVIDENCE_PROPERTIES = Object.freeze([
  "deepbom:model:activationPath",
  "deepbom:model:tensorDtypeInventory",
  "deepbom:finding:highSeverityFindings",
  "deepbom:conformance:violationCodes",
  "deepbom:preprocessing:exactContractIds",
  "deepbom:model:graphTotals",
  "deepbom:model:artifactByteIntegritySummary",
  "deepbom:model:interfaceContractLedger",
]);

export function deepBomPropertyMap(properties) {
  const result = new Map();
  for (const row of properties || []) {
    const name = String(row?.name || "");
    if (!name || result.has(name)) continue;
    result.set(name, String(row?.value ?? ""));
  }
  return result;
}
