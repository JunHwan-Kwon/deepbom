import { portableContract } from "./portable-contracts.js";
export function normalizeArtifactIrRuntimeOverlay(value, graph, architecture, artifactSha256) {
  return portableContract("semantic_validation",{operation:"runtime_overlay",value,graph,architecture,artifact_sha256:artifactSha256});
}
export function validateArtifactIrRuntimeReconciliation(value,validSubjects) {
  portableContract("semantic_validation",{operation:"runtime_reconciliation",value,subjects:[...validSubjects]});
}
