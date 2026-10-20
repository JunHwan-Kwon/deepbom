import { portableContract } from "../portable-contracts.js";
import { canonicalJson } from "../report-utils.js";
export function inspectBomReference(node,observation,modelSha256,refersToCurrentModel,observationIndex) {
  const input={operation:"bom",node,observation:observation ?? null,model_sha256:modelSha256,current:refersToCurrentModel,observation_index:observationIndex};
  canonicalJson(input);
  return portableContract("provenance",input);
}
