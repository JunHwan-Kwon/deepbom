import {reportedDecimal,addExactDecimals,subtractExactDecimals,negateExactDecimal,halveExactDecimal,compareExactDecimals,exactDecimalText,exactDecimalNumber} from "./exact-rational.js";
import {openOptimizationReport,queryOptimizationReport} from "./optimization-report-access.js";
import { EVALUATION_RECORD_SCHEMA, validateEvaluationProtocol, validateMeasurementAgainstProtocol } from "./evaluation-evidence.js";
import { exactFields, requireEvidence as need, validateSubjectReference, sameReference, referenceKey, sealDocument, assertDocumentDigest, canonicalEvidenceDigest } from "./evidence-identity.js";

export const CHANGE_REVIEW_SCHEMA = "deepbom.change_review.v1";
export const CHANGE_REVIEW_INPUT_SCHEMA = "deepbom.change_review_input.v1";
export const CHANGE_REVIEW_RULES = Object.freeze([
  { id:"DB-CHANGE-BINDING", version:"1.0.0", owner:"evidence-context.js", purpose:"Resolve exactly the declared subjects and supplied evidence" },
  { id:"DB-CHANGE-COMPARABILITY", version:"1.0.0", owner:"change-review.js", purpose:"Same protocol, release, cohort, code, environment, unit and paired set when required" },
  { id:"DB-CHANGE-REGRESSION", version:"1.0.0", owner:"change-review.js", purpose:"After-minus-before with declared direction and rounding interval; no derived CI" },
]);
function validateInput(input){
  exactFields(input,["schema","before","after","change_kind","protocol","evaluations","evidence"],[],"change review input");
  need(input.schema===CHANGE_REVIEW_INPUT_SCHEMA,"unsupported review input");
  for(const k of ["before","after","protocol"])validateSubjectReference(input[k]);
  need(!sameReference(input.before,input.after),"before and after must be distinct subjects");
  need(["preserving_transformation","approximate_transformation","model_change","configuration_change"].includes(input.change_kind),"unknown change kind");
  for(const k of ["evaluations","evidence"]){need(Array.isArray(input[k])&&input[k].length<=512,"review reference limit exceeded"); input[k].forEach(validateSubjectReference);need(new Set(input[k].map(referenceKey)).size===input[k].length,"duplicate review evidence reference");}
}
export function buildChangeReview(input, context){
  validateInput(input);
  const protocol=context.document(input.protocol);need(protocol,"review requires supplied protocol");validateEvaluationProtocol(protocol);
  const refs=[input.before,input.after,input.protocol,...input.evaluations,...input.evidence];
  const bindings=refs.map(ref=>context.resolve(ref));
  const records=input.evaluations.map(ref=>{
    need(ref.schema===EVALUATION_RECORD_SCHEMA,"review evaluation reference has wrong type");
    const r=context.document(ref);if(r)validateMeasurementAgainstProtocol(r,protocol);return r;
  }).filter(Boolean);
  need(records.every(r=>sameReference(r.subject,input.before)||sameReference(r.subject,input.after)),"evaluation belongs to a different model");
  const structuralEvidence=[];
  const subjectDigest=(ref)=>{
    if(ref.kind==="snapshot"){
      const snapshot=context.document(ref);
      return snapshot?.kind==="model_state"?snapshot.contents[0].ref.sha256:ref.sha256;
    }
    return ref.sha256;
  };
  for(const ref of input.evidence){
    const d=context.document(ref);if(!d)continue;
    if(d.schema==="deepbom.optimization_report.v1"){
      const summary=queryOptimizationReport(openOptimizationReport(JSON.stringify(d)));
      need(summary.before_sha256===subjectDigest(input.before)&&summary.after_sha256===subjectDigest(input.after),"optimization report is bound to different before/after states");
      structuralEvidence.push({ref,status:"validated_record",summary});
    }else if(d.schema==="deepbom.semantic_artifact_diff.v1"){
      need(d.baseline.sha256===subjectDigest(input.before)&&d.candidate.sha256===subjectDigest(input.after),"artifact diff belongs to different before/after files");
      structuralEvidence.push({ref,status:"validated_record",summary:structuredClone(d)});
    }
  }
  const checks=[];
  for(const population of protocol.required_populations)for(const metric of protocol.metrics){
    const before=records.filter(r=>r.population===population&&sameReference(r.subject,input.before));
    const after=records.filter(r=>r.population===population&&sameReference(r.subject,input.after));
    need(before.length<=1&&after.length<=1,"ambiguous evaluations for model/population");
    const row={id:JSON.stringify([population,metric.id]),population,metric:metric.id,rule_id:"DB-CHANGE-COMPARABILITY",rule_version:"1.0.0",status:"not_assessed",reason:"Missing before/after evaluation",before:null,after:null,delta:null,delta_exact:null,delta_ci:null,unit:metric.unit,analysis_unit:metric.analysis_unit,source_refs:[]};
    const b=before[0],a=after[0],bm=b?.metrics.find(r=>r.id===metric.id),am=a?.metrics.find(r=>r.id===metric.id);
    if(bm&&am){
      row.before=bm.value;row.after=am.value;row.source_refs=[{evaluation_sha256:b.evaluation_sha256,pointer:`/metrics/${b.metrics.indexOf(bm)}`},{evaluation_sha256:a.evaluation_sha256,pointer:`/metrics/${a.metrics.indexOf(am)}`}];
      const common=["dataset","cohort","code","environment","configuration"].every(k=>sameReference(b[k],a[k]));
      const paired=protocol.comparison!=="paired"||(b.pair_set&&a.pair_set&&sameReference(b.pair_set,a.pair_set)&&bm.denominator===am.denominator);
      const evidence=[...bindings,...[b,a].flatMap(r=>[r.subject,r.dataset,r.cohort,r.run,r.code,r.environment,r.configuration,...(r.pair_set?[r.pair_set]:[])].map(ref=>context.resolve(ref)))];
      if(!common||!paired)row.reason="Evaluation conditions or paired sample identity are not comparable";
      else if(evidence.some(r=>r.status!=="matched"))row.reason="Required model, data or execution evidence is unresolved";
      else if(protocol.comparison==="descriptive")row.reason="Protocol permits description only";
      else {
        const delta=subtractExactDecimals(reportedDecimal(am.value),reportedDecimal(bm.value));
        row.delta=exactDecimalNumber(delta);row.delta_exact=exactDecimalText(delta);row.rule_id="DB-CHANGE-REGRESSION";
        const rounding=halveExactDecimal(addExactDecimals(reportedDecimal(bm.resolution??0),reportedDecimal(am.resolution??0)));
        row.rounding_uncertainty=exactDecimalNumber(rounding);row.rounding_uncertainty_exact=exactDecimalText(rounding);
        const regression=metric.higher_is_better?negateExactDecimal(delta):delta;
        if(metric.max_regression===null){row.status="not_assessed";row.reason="Delta calculated; no predeclared acceptance threshold";}
        else if(compareExactDecimals(subtractExactDecimals(regression,rounding),reportedDecimal(metric.max_regression))>0){row.status="fail";row.reason="Declared regression threshold exceeded";}
        else if(compareExactDecimals(addExactDecimals(regression,rounding),reportedDecimal(metric.max_regression))<=0){row.status="pass";row.reason="Within declared regression threshold";}
        else {row.status="not_assessed";row.reason="Reported rounding cannot resolve the threshold";}
      }
    }
    checks.push(row);
  }
  const counts={total:checks.length,pass:checks.filter(r=>r.status==="pass").length,fail:checks.filter(r=>r.status==="fail").length,not_assessed:checks.filter(r=>r.status==="not_assessed").length};
  return sealDocument({schema:CHANGE_REVIEW_SCHEMA,method_version:"1.0.0",input:structuredClone(input),protocol:structuredClone(protocol),bindings,checks,counts,structural_evidence:structuralEvidence,structural_coverage:structuralEvidence.length?"supplied_record":"not_supplied",criterion_scope:"externally_reported_evaluations",transformation_equivalence:"not_established",rules:structuredClone(CHANGE_REVIEW_RULES),status:counts.fail?"criteria_not_met":counts.not_assessed?"incomplete":"criteria_met",decision:"not_made",boundary:"External measurements are not independently recomputed. Criteria satisfaction is not clinical safety, regulatory compliance or deployment approval. No CI is inferred from marginal CIs."},"review_sha256");
}
export function validateChangeReview(doc,context=null){
  exactFields(doc,["schema","method_version","input","protocol","bindings","checks","counts","structural_evidence","structural_coverage","criterion_scope","transformation_equivalence","rules","status","decision","boundary","review_sha256"],[],"change review");
  need(doc.schema===CHANGE_REVIEW_SCHEMA&&doc.method_version==="1.0.0","unsupported review");validateInput(doc.input);validateEvaluationProtocol(doc.protocol);assertDocumentDigest(doc,"review_sha256");
  need(doc.decision==="not_made","technical review cannot assert approval");
  need(context,"review verification requires the original evidence context");
  if(context)need(canonicalEvidenceDigest(buildChangeReview(doc.input,context))===canonicalEvidenceDigest(doc),"review contradicts its source evidence");
  else need(doc.counts.total===doc.checks.length&&doc.counts.pass+doc.counts.fail+doc.counts.not_assessed===doc.counts.total,"review counts mismatch");
  return doc;
}
