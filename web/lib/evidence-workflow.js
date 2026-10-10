import {connectOmopSnapshot} from "./evidence-workflow-omop.js";
import { createEvidenceContext } from "./evidence-context.js";
import { snapshotFromNative, snapshotFromManifest, validateSnapshotIr, SNAPSHOT_IR_SCHEMA } from "./snapshot-ir.js";
import { buildEvaluationProtocol, validateEvaluationProtocol, buildEvaluationRecord, validateEvaluationRecord, validateMeasurementAgainstProtocol, EVALUATION_PROTOCOL_SCHEMA, EVALUATION_RECORD_SCHEMA } from "./evaluation-evidence.js";
import { buildProvenanceV2, validateProvenanceV2, PROVENANCE_V2_SCHEMA } from "./provenance-v2.js";
import { buildChangeReview, validateChangeReview, CHANGE_REVIEW_SCHEMA } from "./change-review.js";
import { buildDeploymentEvidence, validateDeploymentEvidence, DEPLOYMENT_RESULT_SCHEMA } from "./deployment-evidence.js";
import { exactFields, requireEvidence as need, sealDocument, canonicalEvidenceDigest, assertDocumentDigest } from "./evidence-identity.js";

export const EVIDENCE_WORKFLOW_SCHEMA = "deepbom.evidence_workflow_request.v1";
export const EVIDENCE_WORKFLOW_RESULT_SCHEMA = "deepbom.evidence_workflow_result.v1";
export const EVIDENCE_OPERATIONS = Object.freeze(["snapshot","omop","protocol","evaluation","provenance","review","deployment","verify"]);
const VALIDATORS={
  [EVALUATION_PROTOCOL_SCHEMA]:[validateEvaluationProtocol,"protocol_sha256","evidence_document"],
  [EVALUATION_RECORD_SCHEMA]:[validateEvaluationRecord,"evaluation_sha256","evidence_document"],
};
export function runEvidenceWorkflow(request,{files=[]}={}){
  return execute(request, {files});
}
function execute(request,{files=[],recordedFiles=[]}={}){
  exactFields(request,["schema","operation","input","documents"],[],"workflow request");
  need(request.schema===EVIDENCE_WORKFLOW_SCHEMA&&EVIDENCE_OPERATIONS.includes(request.operation),"unsupported evidence workflow operation");
  const context=createEvidenceContext(request.documents,{validators:VALIDATORS,files,recordedFiles});
  let document;
  switch(request.operation){
    case "snapshot": document=request.input.schema==="deepbom.model_state_snapshot.v1"?snapshotFromNative(request.input):snapshotFromManifest(request.input);break;
    case "protocol": document=buildEvaluationProtocol(request.input);break;
    case "evaluation": {
      document=buildEvaluationRecord(request.input);
      const protocol=context.document(document.protocol);
      if(protocol)validateMeasurementAgainstProtocol(document,protocol);
      break;
    }
    case "omop": document=connectOmopSnapshot(request.input,context);break;
    case "provenance": document=buildProvenanceV2(request.input,context);break;
    case "review": document=buildChangeReview(request.input,context);break;
    case "deployment": document=buildDeploymentEvidence(request.input,context);break;
    case "verify": {
      const d=request.input;
      if(d.schema===SNAPSHOT_IR_SCHEMA)document=validateSnapshotIr(d);
      else if(d.schema===PROVENANCE_V2_SCHEMA)document=validateProvenanceV2(d,context);
      else if(d.schema===CHANGE_REVIEW_SCHEMA)document=validateChangeReview(d,context);
      else if(d.schema===DEPLOYMENT_RESULT_SCHEMA)document=validateDeploymentEvidence(d,context);
      else if(VALIDATORS[d.schema])document=VALIDATORS[d.schema][0](d);
      else throw new Error("Evidence: unsupported workflow verification contract");
      break;
    }
  }
  const contentChecks=document.schema===SNAPSHOT_IR_SCHEMA?document.contents.map(row=>({id:row.id,...context.resolve(row.ref)})):[];
  const result = sealDocument({schema:EVIDENCE_WORKFLOW_RESULT_SCHEMA,operation:request.operation,request:structuredClone(request),file_observations:[...files.map(f=>f.ref),...recordedFiles].sort((a,b)=>a.sha256<b.sha256?-1:a.sha256>b.sha256?1:0),request_sha256:canonicalEvidenceDigest(request),context_sha256:context.context_sha256,execution_status:"completed",document:structuredClone(document),content_checks:contentChecks,boundary:"Completion means the requested evidence operation ran. Content identity, observation coverage, external assertions and decision status remain separate."},"result_sha256");
  need(new TextEncoder().encode(JSON.stringify(result,null,2)+"\n").length<=16*1024*1024,"Workflow result exceeds the 16 MiB preservation limit; reduce the supplied evidence scope");
  return result;
}
/** Recompute semantics from preserved records; this does not re-read model files. */
export function validateWorkflowResult(result){
  exactFields(result,["schema","operation","request","file_observations","request_sha256","context_sha256","execution_status","document","content_checks","boundary","result_sha256"],[],"workflow result");
  need(result.schema===EVIDENCE_WORKFLOW_RESULT_SCHEMA && result.execution_status==="completed","unsupported workflow result");
  assertDocumentDigest(result,"result_sha256");
  const rebuilt=execute(result.request,{recordedFiles:result.file_observations});
  need(canonicalEvidenceDigest(rebuilt)===canonicalEvidenceDigest(result),"workflow result contradicts its source records");
  return result;
}
export function queryWorkflowResult(result,{offset=0,limit=25}={}){
  validateWorkflowResult(result);need(Number.isSafeInteger(offset)&&offset>=0&&Number.isSafeInteger(limit)&&limit>=1&&limit<=100,"invalid workflow pagination");
  const rows=result.document.checks||result.document.bindings||result.content_checks;
  return {schema:"deepbom.evidence_workflow_query.v1",result_sha256:result.result_sha256,document_schema:result.document.schema,status:result.document.status??"record_created",total:rows.length,returned:rows.slice(offset,offset+limit).length,offset,next_offset:offset+limit<rows.length?offset+limit:null,rows:rows.slice(offset,offset+limit),validation_scope:"Recomputed from preserved records; file observations are recorded claims, not freshly verified bytes.",boundary:result.boundary};
}
