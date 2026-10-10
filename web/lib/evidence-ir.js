// Single owner of the current Evidence IR family identities.
// Schema transitions produce new evidence; stored documents are not rewritten.
export const EVIDENCE_IR_NAME = "DEEPBOM Evidence IR";
export const ARTIFACT_IR_SCHEMA = "deepbom.artifact_ir.v2";
export const MODEL_IR_SCHEMA = "deepbom.model_ir.v1";
export const WEIGHT_IR_SCHEMA = "deepbom.weight_ir.v1";
export const ACTIVATION_IR_SCHEMA = "deepbom.activation_ir.v1";
export const PROVENANCE_IR_SCHEMA = "deepbom.provenance_ir.v1";
export const PROVENANCE_INPUT_SCHEMA = "deepbom.provenance_input.v1";
export const PROVENANCE_SUMMARY_SCHEMA = "deepbom.provenance_summary.v1";
export const PROVENANCE_DIGEST_FIELD = "provenance_ir_sha256";

const schema = name => `https://deepbom.org/schemas/${name}`;
export const EVIDENCE_IR_LAYERS = Object.freeze([
  { id: "artifact", name: "Artifact IR", schema: ARTIFACT_IR_SCHEMA, section: "artifact_ir", digest_field: "artifact_ir_sha256", optional: false, json_schema: schema("deepbom-artifact-ir-v2.schema.json") },
  { id: "model", name: "Model IR", schema: MODEL_IR_SCHEMA, section: "model_ir", digest_field: "model_ir_sha256", optional: false, json_schema: schema("deepbom-model-ir-v1.schema.json") },
  { id: "weight", name: "Weight IR", schema: WEIGHT_IR_SCHEMA, section: "weight_ir", digest_field: "weight_ir_sha256", optional: true, json_schema: schema("deepbom-numerical-ir-v1.schema.json#/$defs/weight_ir") },
  { id: "activation", name: "Activation IR", schema: ACTIVATION_IR_SCHEMA, section: "activation_ir", digest_field: "activation_ir_sha256", optional: true, json_schema: schema("deepbom-numerical-ir-v1.schema.json#/$defs/activation_ir") },
  { id: "provenance", name: "Provenance IR", schema: PROVENANCE_IR_SCHEMA, section: "provenance_ir", digest_field: PROVENANCE_DIGEST_FIELD, optional: true, json_schema: schema("deepbom-provenance-ir-v1.schema.json") },
].map(Object.freeze));
export const PROVENANCE_IR = EVIDENCE_IR_LAYERS.find(layer => layer.id === "provenance");

// Additive family: preserve the historical v1/v2 family documents and identifiers.
export const SNAPSHOT_IR_SCHEMA = "deepbom.snapshot_ir.v1";
export const TYPED_PROVENANCE_IR_SCHEMA = "deepbom.provenance_ir.v2";
export const CURRENT_EVIDENCE_IR_FAMILY = schema("deepbom-evidence-ir-v3.schema.json");
export const ADDITIONAL_EVIDENCE_IR_LAYERS = Object.freeze([
  Object.freeze({id:"snapshot",name:"Snapshot IR",schema:SNAPSHOT_IR_SCHEMA,digest_field:"snapshot_ir_sha256",optional:true,json_schema:schema("deepbom-evidence-workflow-v1.schema.json#/$defs/snapshot")}),
  Object.freeze({id:"typed-provenance",name:"Typed Provenance IR",schema:TYPED_PROVENANCE_IR_SCHEMA,digest_field:"provenance_ir_sha256",optional:true,json_schema:schema("deepbom-evidence-workflow-v1.schema.json#/$defs/provenance")}),
]);
