export { MODEL_IR_SCHEMA, ARTIFACT_IR_SCHEMA as MODEL_IR_SOURCE_SCHEMA } from "../../evidence-ir.js";
export const MODEL_IR_METHOD_VERSION = "1.1.2";
export const SHA256 = /^[a-f0-9]{64}$/;

export const EVIDENCE_CLASSES = Object.freeze([
  "OBSERVED_SERIALIZED_ARTIFACT",
  "OBSERVED_RUNTIME",
  "DERIVED",
  "PREDICTED",
  "DECLARED_UNVERIFIED",
  "NOT_ASSESSABLE",
]);

export const RELATIONSHIP_KINDS = Object.freeze([
  "data_dependency",
  "control_dependency",
  "call",
  "region_ownership",
  "state_dependency",
  "alias",
  "storage_binding",
  "parameter_binding",
]);
