import { validateNativeDocument, NATIVE_SCHEMAS } from "./native-evidence.js";
import { exactFields, requireEvidence as need, sealDocument, subjectReference, evidenceContract } from "./evidence-identity.js";

import {SNAPSHOT_IR_SCHEMA} from "./evidence-ir.js";
export {SNAPSHOT_IR_SCHEMA};
export const DATASET_RELEASE_SCHEMA = "deepbom.dataset_release_input.v1";
export const CONFIGURATION_INPUT_SCHEMA = "deepbom.configuration_input.v1";
const BOUNDARY = "Fixed state only within the declared scope. Identity is not authenticity, complete data capture, execution equivalence or clinical validation.";

function build(kind, id, scope, contents) {
  const doc = sealDocument({ schema: SNAPSHOT_IR_SCHEMA, method_version: "1.0.0", kind, identity: structuredClone(id), scope, contents: structuredClone(contents), completeness: "declared_scope_only", boundary: BOUNDARY }, "snapshot_ir_sha256");
  return validateSnapshotIr(doc);
}
export function snapshotFromNative(document) {
  validateNativeDocument(document);
  need(document.schema === NATIVE_SCHEMAS.snapshot, "model snapshot adapter requires a native state snapshot");
  return build("model_state", { namespace: "deepbom.native", name: "model-state", version: document.snapshot_sha256 },
    { fixed_aspects: ["adapter_definition", "named_tensor_identities"], excluded_aspects: ["optimizer_state", "rng_state", "external_python_state"], consistency: "adapter_declared_scope" },
    [{ id: "state", role: "native_model_state", ref: subjectReference("native_state", document.snapshot_sha256, { schema: document.schema }) }]);
}
export function snapshotFromManifest(input) {
  exactFields(input, ["schema", "identity", "scope", "contents"], [], "snapshot manifest input");
  need([DATASET_RELEASE_SCHEMA, CONFIGURATION_INPUT_SCHEMA].includes(input.schema), "unsupported manifest schema");
  const kind = input.schema === DATASET_RELEASE_SCHEMA ? "dataset_release" : "configuration";
  return build(kind, input.identity, input.scope, input.contents);
}
export function validateSnapshotIr(doc) {
  evidenceContract("snapshot", doc);
  return doc;
}
