import { validateNativeDocument, NATIVE_SCHEMAS } from "./native-evidence.js";
import { assertDocumentDigest, exactFields, evidenceText, requireEvidence as need, sealDocument, subjectReference, validateSubjectReference, uniqueEvidenceIds, canonicalEvidenceDigest } from "./evidence-identity.js";

import {SNAPSHOT_IR_SCHEMA} from "./evidence-ir.js";
export {SNAPSHOT_IR_SCHEMA};
export const DATASET_RELEASE_SCHEMA = "deepbom.dataset_release_input.v1";
export const CONFIGURATION_INPUT_SCHEMA = "deepbom.configuration_input.v1";
const KINDS = ["model_state", "dataset_release", "configuration"];
const BOUNDARY = "Fixed state only within the declared scope. Identity is not authenticity, complete data capture, execution equivalence or clinical validation.";

function identity(value) {
  exactFields(value, ["namespace", "name", "version"], [], "snapshot identity");
  for (const key of ["namespace", "name", "version"]) evidenceText(value[key], key);
  return { ...value };
}
function validateContents(contents) {
  uniqueEvidenceIds(contents, "snapshot contents");
  need(contents.length > 0, "snapshot needs fixed content references");
  for (const row of contents) {
    exactFields(row, ["id", "role", "ref"], ["path"], "snapshot content");
    evidenceText(row.role, "content role");
    validateSubjectReference(row.ref);
    need(row.ref.subject_ref === null, "snapshot contents must reference whole content documents or files");
    if (row.path !== undefined) {
      evidenceText(row.path, "relative content path");
      need(!row.path.includes("\\") && !row.path.includes("\0") && !row.path.startsWith("/") && !/^[a-zA-Z]:/.test(row.path) && row.path.split("/").every(p => p && p !== "." && p !== ".."), "unsafe manifest content path");
    }
  }
  const paths = contents.filter(r => r.path !== undefined).map(r => r.path);
  need(new Set(paths).size === paths.length, "duplicate manifest content path");
}
function build(kind, id, scope, contents) {
  const doc = sealDocument({ schema: SNAPSHOT_IR_SCHEMA, method_version: "1.0.0", kind, identity: identity(id), scope, contents: structuredClone(contents), completeness: "declared_scope_only", boundary: BOUNDARY }, "snapshot_ir_sha256");
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
  exactFields(doc, ["schema", "method_version", "kind", "identity", "scope", "contents", "completeness", "boundary", "snapshot_ir_sha256"], [], "Snapshot IR");
  need(doc.schema === SNAPSHOT_IR_SCHEMA && doc.method_version === "1.0.0" && KINDS.includes(doc.kind), "unsupported Snapshot IR contract");
  identity(doc.identity);
  exactFields(doc.scope, ["fixed_aspects", "excluded_aspects", "consistency"], [], "snapshot scope");
  for (const key of ["fixed_aspects", "excluded_aspects"]) {
    const rows = doc.scope[key];
    need(Array.isArray(rows) && rows.length <= 256 && new Set(rows).size === rows.length, "invalid snapshot scope list");
    rows.forEach(s => evidenceText(s, key));
  }
  need(doc.scope.fixed_aspects.length > 0 && !doc.scope.fixed_aspects.some(s => doc.scope.excluded_aspects.includes(s)), "snapshot fixed/excluded aspects conflict");
  need(["adapter_declared_scope", "manifest_declared_scope"].includes(doc.scope.consistency), "unsupported capture consistency claim");
  need(doc.completeness === "declared_scope_only" && doc.boundary === BOUNDARY, "snapshot completeness/boundary contradiction");
  validateContents(doc.contents);
  if (doc.kind === "model_state") {
    need(doc.contents.length === 1 && doc.contents[0].id === "state" && doc.contents[0].role === "native_model_state" && doc.contents[0].ref.kind === "native_state", "model-state snapshot source mismatch");
    need(canonicalEvidenceDigest(doc.scope) === canonicalEvidenceDigest({fixed_aspects:["adapter_definition","named_tensor_identities"],excluded_aspects:["optimizer_state","rng_state","external_python_state"],consistency:"adapter_declared_scope"}), "model-state scope exceeds the native adapter contract");
    need(doc.identity.namespace === "deepbom.native" && doc.identity.name === "model-state" && doc.identity.version === doc.contents[0].ref.sha256, "native snapshot identity mismatch");
  }
  if (doc.kind !== "model_state") need(doc.scope.consistency === "manifest_declared_scope", "manifest cannot assert atomic state capture");
  return assertDocumentDigest(doc, "snapshot_ir_sha256");
}
