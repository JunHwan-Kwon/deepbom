import { portableContract } from "./portable-contracts.js";
import { canonicalJsonInput } from "./report-utils.js";
const irContract = (operation, input) => portableContract("ir_semantics", canonicalJsonInput({ operation, ...input }));

export function inputManifestSha256(inputs) { return irContract("input_manifest", {inputs}); }
// Detached signatures remain claims; this validates binding, not signer trust.
export function validateRuntimeProvenance(value, inputs, run) {
  irContract("runtime_provenance", {value,inputs,run}); return value;
}
