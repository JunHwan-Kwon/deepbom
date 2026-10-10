import { exactFields, requireEvidence as need, evidenceText, validateSubjectReference, sameReference, uniqueEvidenceIds, sealDocument, assertDocumentDigest, canonicalEvidenceDigest, evidenceTimestamp } from "./evidence-identity.js";
export const DEPLOYMENT_INPUT_SCHEMA = "deepbom.deployment_context_input.v1";
export const DEPLOYMENT_RESULT_SCHEMA = "deepbom.deployment_context.v1";
const timestamp = evidenceTimestamp;
function configuration(ref) {validateSubjectReference(ref);need(ref.kind==="snapshot","configuration must reference a Snapshot IR");}
export function buildDeploymentEvidence(input,context){
  exactFields(input,["schema","evaluated_configuration","deployments","execution"],[],"deployment context input");
  need(input.schema===DEPLOYMENT_INPUT_SCHEMA,"unsupported deployment context");configuration(input.evaluated_configuration);
  uniqueEvidenceIds(input.deployments,"deployment records");
  const references=[input.evaluated_configuration];
  for(const d of input.deployments){exactFields(d,["id","configuration","site","start","end","source"],[],"deployment record");configuration(d.configuration);validateSubjectReference(d.source);references.push(d.configuration,d.source);evidenceText(d.site,"site");const start=timestamp(d.start);if(d.end!==null)need(timestamp(d.end)>start,"deployment interval must be positive");}
  const e=input.execution;
  if(e!==null){exactFields(e,["event_id","site","at","clock_uncertainty_ms","configuration","output_state","source"],[],"execution record");configuration(e.configuration);validateSubjectReference(e.source);references.push(e.configuration,e.source);evidenceText(e.event_id,"execution event");evidenceText(e.site,"execution site");timestamp(e.at);need(Number.isSafeInteger(e.clock_uncertainty_ms)&&e.clock_uncertainty_ms>=0,"invalid clock uncertainty");need(Number.isSafeInteger(timestamp(e.at)+e.clock_uncertainty_ms)&&Number.isSafeInteger(timestamp(e.at)-e.clock_uncertainty_ms),"clock uncertainty exceeds exact time range");need(["generated","displayed","used"].includes(e.output_state),"unknown output observation");}
  for(const ref of references.filter(r=>r.kind==="snapshot")){const d=context.document(ref);if(d)need(d.kind==="configuration","deployment references a non-configuration snapshot");}
  const bindings=references.map(ref=>context.resolve(ref));
  const candidates=e?input.deployments.filter(d=>d.site===e.site&&timestamp(d.start)<=timestamp(e.at)+e.clock_uncertainty_ms&&(d.end===null||timestamp(d.end)>timestamp(e.at)-e.clock_uncertainty_ms)):[];
  const exact=e&&candidates.length===1&&sameReference(candidates[0].configuration,e.configuration)&&timestamp(candidates[0].start)<=timestamp(e.at)-e.clock_uncertainty_ms&&(candidates[0].end===null||timestamp(candidates[0].end)>timestamp(e.at)+e.clock_uncertainty_ms);
  return sealDocument({schema:DEPLOYMENT_RESULT_SCHEMA,method_version:"1.0.0",input:structuredClone(input),bindings,candidate_deployments:candidates.map(r=>r.id),status:!e?"execution_not_provided":bindings.some(r=>r.status!=="matched")?"unresolved_evidence":!exact?"ambiguous_or_conflicting_deployment":"consistent_declared_execution",same_as_evaluated:e?sameReference(input.evaluated_configuration,e.configuration):null,boundary:"Consistency of supplied declarations only. Installation does not prove execution; generated/displayed/used are not inferred from each other. File identity does not authenticate clinical use."},"deployment_context_sha256");
}
export function validateDeploymentEvidence(doc,context){
  exactFields(doc,["schema","method_version","input","bindings","candidate_deployments","status","same_as_evaluated","boundary","deployment_context_sha256"],[],"deployment context");
  need(doc.schema===DEPLOYMENT_RESULT_SCHEMA&&doc.method_version==="1.0.0","unsupported deployment evidence");assertDocumentDigest(doc,"deployment_context_sha256");
  need(context,"deployment verification requires its original context");
  need(canonicalEvidenceDigest(buildDeploymentEvidence(doc.input,context))===canonicalEvidenceDigest(doc),"deployment result does not reconstruct");return doc;
}
