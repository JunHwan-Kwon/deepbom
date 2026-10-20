// Supporting result index, not a second Training IR or a rewritten event log.
import { validateTrainingBundle, validateNativeBundle } from "./native-evidence.js";
import { exactKeys, requireCondition as need, SHA256 } from "./numerical-ir/common.js";
import { canonicalEvidenceDigest } from "./evidence-identity.js";

export function validateTrainingAnalysis(input) {
  exactKeys(input,["bundle","index","objects"],"training analysis view");
  validateTrainingBundle(input.bundle);
  const index=input.index;
  exactKeys(index,["schema","run_id","segment_id","bindings","counts","copy_bytes","copy_seconds","boundary"],"analysis bindings");
  need(index.schema==="deepbom.analysis_bindings.v1","unsupported analysis binding version");
  need(index.run_id===input.bundle.training.run_id && index.segment_id===input.bundle.training.segment_id,"foreign analysis run");
  need(Array.isArray(index.bindings)&&index.bindings.length<=10000,"analysis binding budget exceeded");
  need(Number.isSafeInteger(index.copy_bytes)&&index.copy_bytes>=0&&Number.isFinite(index.copy_seconds)&&index.copy_seconds>=0,"invalid capture cost totals");
  const seen=new Set();
  for(const row of index.bindings) {
    need(row.run_id===index.run_id&&row.segment_id===index.segment_id,"foreign analysis binding");
    need(/^(0|[1-9][0-9]*)$/.test(row.capture_sequence)&&SHA256.test(row.job_sha256),"invalid capture/job identity");
    const identity={run_id:row.run_id,segment_id:row.segment_id,capture_sequence:row.capture_sequence,at:row.at,job_sha256:row.job_sha256};
    need(canonicalEvidenceDigest(identity)===row.binding_sha256&&!seen.has(row.binding_sha256),"analysis binding identity/uniqueness mismatch");seen.add(row.binding_sha256);
    need(["pending","commit_pending","completed","failed","cancelled"].includes(row.status),"invalid analysis status");
    if(row.status==="completed"||row.status==="commit_pending") {
      need(typeof row.late==="boolean"&&row.completed_after_capture===true,"missing analysis completion boundary");
      exactKeys(row.evidence,["snapshot","model_ir","weight_ir"],"analysis result references");
      const result={activation_ir:null};
      for(const [key,ref] of Object.entries(row.evidence)) {
        need(SHA256.test(ref.sha256),"invalid analysis evidence digest");
        const doc=input.objects[ref.sha256];
        need(doc?.schema===ref.schema&&Object.entries(doc).some(([key,value])=>key.endsWith("_sha256")&&value===ref.sha256),"missing analysis evidence");
        result[key]=doc;
      }
      validateNativeBundle(result);
      if (row.status === "completed" && !row.late) {
        const matches = input.bundle.events.filter(event => event.kind === "model_snapshot" && event.at.analysis_binding_sha256 === row.binding_sha256);
        need(matches.length === 1 && matches[0].payload.evidence.snapshot.sha256 === row.evidence.snapshot.sha256,"analysis completion lacks one matching Training event");
      }
    }
  }
  exactKeys(index.counts,["requested","admitted","completed","skipped","failed"],"analysis counts");
  for(const v of Object.values(index.counts))need(Number.isSafeInteger(v)&&v>=0,"invalid analysis count");
  need(index.counts.admitted===index.bindings.length&&index.counts.requested===index.counts.admitted+index.counts.skipped,"analysis admission counts disagree");
  need(index.counts.completed===index.bindings.filter(r=>r.status==="completed").length,"analysis completion counts disagree");
  need(index.counts.failed===index.bindings.filter(r=>r.status==="failed"||r.status==="cancelled").length,"analysis failure counts disagree");
  return input;
}
