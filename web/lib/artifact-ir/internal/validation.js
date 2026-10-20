import { portableContract } from "../../portable-contracts.js";
import { canonicalJson } from "../../report-utils.js";
export function validateArtifactIrBody(value) {
  canonicalJson(value);
  portableContract("semantic_validation",{operation:"artifact",value});
}
