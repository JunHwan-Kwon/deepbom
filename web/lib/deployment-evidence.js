import { portableContract } from "./portable-contracts.js";
import { canonicalJsonInput } from "./report-utils.js";
import { exactFields, requireEvidence as need, sealDocument, assertDocumentDigest, canonicalEvidenceDigest } from "./evidence-identity.js";
export const DEPLOYMENT_INPUT_SCHEMA = "deepbom.deployment_context_input.v1";
export const DEPLOYMENT_RESULT_SCHEMA = "deepbom.deployment_context.v1";
export function buildDeploymentEvidence(input,context){
  // WASM owns input admission and interval selection. The host supplies only
  // explicit context lookups; no URLs or references are fetched implicitly.
  const admitted=canonicalJsonInput(input);
  let selected=portableContract("evidence_workflow", {operation:"deployment",value:admitted,matched:false});
  const references=selected.references;
  for(const ref of references.filter(r=>r.kind==="snapshot")){const d=context.document(ref);if(d)need(d.kind==="configuration","deployment references a non-configuration snapshot");}
  const bindings=references.map(ref=>context.resolve(ref));
  selected=portableContract("evidence_workflow", {operation:"deployment",value:admitted,matched:bindings.every(r=>r.status==="matched")});
  return sealDocument({schema:DEPLOYMENT_RESULT_SCHEMA,method_version:"1.0.0",input:structuredClone(input),bindings,candidate_deployments:selected.candidate_deployments,status:selected.status,same_as_evaluated:selected.same_as_evaluated,boundary:"Consistency of supplied declarations only. Installation does not prove execution; generated/displayed/used are not inferred from each other. File identity does not authenticate clinical use."},"deployment_context_sha256");
}
export function validateDeploymentEvidence(doc,context){
  exactFields(doc,["schema","method_version","input","bindings","candidate_deployments","status","same_as_evaluated","boundary","deployment_context_sha256"],[],"deployment context");
  need(doc.schema===DEPLOYMENT_RESULT_SCHEMA&&doc.method_version==="1.0.0","unsupported deployment evidence");assertDocumentDigest(doc,"deployment_context_sha256");
  need(context,"deployment verification requires its original context");
  need(canonicalEvidenceDigest(buildDeploymentEvidence(doc.input,context))===canonicalEvidenceDigest(doc),"deployment result does not reconstruct");return doc;
}
