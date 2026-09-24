import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

const SHA = /^[a-f0-9]{64}$/;
export function inputManifestSha256(inputs) {
  return sha256TextHex(canonicalJson(inputs.map(row => ({ value_ref: row.value_ref, native_locator: row.native_locator,
    dtype: row.dtype, shape: row.shape, values_sha256: row.values_sha256 }))));
}

// Detached signatures are preserved as claims. Verification needs a caller's trusted key policy.
export function validateRuntimeProvenance(value, inputs, run) {
  const keys = ["schema", "environment", "environment_sha256", "input_manifest_sha256", "code_sha256", "trace_sha256", "signature", "trust"];
  if (!value || Object.keys(value).some(key => !keys.includes(key)) || keys.some(key => !Object.hasOwn(value, key))
    || value.schema !== "deepbom.runtime_provenance.v1" || value.trust !== "declared_not_attested") throw new Error("Runtime provenance contract is invalid.");
  canonicalJson(value);
  if (!value.environment || typeof value.environment !== "object" || Array.isArray(value.environment)
    || value.environment_sha256 !== sha256TextHex(canonicalJson(value.environment))) throw new Error("Runtime environment digest is inconsistent.");
  if (value.input_manifest_sha256 !== inputManifestSha256(inputs)) throw new Error("Runtime provenance input digest does not bind the captured inputs.");
  for (const key of ["code_sha256", "trace_sha256"]) if (value[key] !== null && !SHA.test(value[key])) throw new Error(`Runtime provenance ${key} is invalid.`);
  if (run.runtime_evidence && value.trace_sha256 !== run.runtime_evidence.sha256) throw new Error("Runtime provenance trace conflicts with imported evidence.");
  if (value.signature !== null) {
    const signature = value.signature;
    if (Object.keys(signature).sort().join(",") !== ["algorithm", "key_id", "payload_sha256", "value_base64", "verification"].sort().join(",")
      || signature.algorithm !== "Ed25519" || typeof signature.key_id !== "string" || !signature.key_id
      || signature.verification !== "not_verified" || !/^[A-Za-z0-9+/]{86}==$/.test(signature.value_base64)) throw new Error("Runtime provenance detached signature is invalid.");
    const { signature: ignored, ...statement } = value;
    const signed = { artifact_sha256: run.execution.artifact_sha256, configuration_sha256: run.execution.configuration_sha256, collector_sha256: run.collector.sha256, provenance: statement };
    if (signature.payload_sha256 !== sha256TextHex(canonicalJson(signed))) throw new Error("Runtime provenance signature payload does not bind this run.");
  }
  return value;
}
