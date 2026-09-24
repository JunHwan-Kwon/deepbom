export const MODEL_IR_SCHEMA = "deepbom.model_ir.v1";
export const MODEL_IR_METHOD_VERSION = "1.1.0";
export const MODEL_IR_SOURCE_SCHEMA = "deepbom.artifact_ir.v2";
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
