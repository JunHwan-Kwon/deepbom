import {exactCountValue} from "./exact-integer.js";
import { exactFields, evidenceText, requireEvidence as need, uniqueEvidenceIds, validateSubjectReference, sealDocument, assertDocumentDigest } from "./evidence-identity.js";

export const EVALUATION_PROTOCOL_SCHEMA = "deepbom.evaluation_protocol.v1";
export const EVALUATION_RECORD_SCHEMA = "deepbom.evaluation_record.v1";
export function exactCount(value, label, positive = false) {
  need(typeof value === "string" && value.length <= 100 && exactCountValue(value) !== null, `${label} must be an exact decimal count`);
  need(!positive || exactCountValue(value) > 0n, `${label} must be positive`);
  return exactCountValue(value);
}
function finite(value, label) { need(typeof value === "number" && Number.isFinite(value), `${label} must be a finite number`); }
function protocolBody(doc) {
  need(doc.schema === EVALUATION_PROTOCOL_SCHEMA, "unsupported evaluation protocol");
  evidenceText(doc.id,"protocol ID");
  need(["paired", "unpaired", "descriptive"].includes(doc.comparison), "unknown comparison mode");
  need(doc.data_policy === "same_release", "only explicit same-release comparisons are supported");
  uniqueEvidenceIds(doc.metrics, "protocol metrics"); need(doc.metrics.length > 0 && doc.metrics.length <= 64, "protocol needs at least one metric");
  for (const m of doc.metrics) {
    exactFields(m,["id","unit","analysis_unit","higher_is_better","max_regression","range","aggregation","missing_policy","repeated_observations"],[],"metric definition");
    for(const k of ["aggregation","missing_policy","repeated_observations"])evidenceText(m[k],k);
    evidenceText(m.unit,"metric unit"); evidenceText(m.analysis_unit,"analysis unit");
    need(typeof m.higher_is_better === "boolean", "metric direction missing");
    if(m.max_regression !== null){finite(m.max_regression,"maximum regression");need(m.max_regression>=0,"negative regression threshold");}
    if(m.range !== null){need(Array.isArray(m.range)&&m.range.length===2,"metric range invalid"); m.range.forEach(x=>finite(x,"metric range"));need(m.range[0]<=m.range[1],"metric range reversed");}
  }
  need(Array.isArray(doc.required_populations)&&doc.required_populations.length>0&&doc.required_populations.length<=64&&new Set(doc.required_populations).size===doc.required_populations.length,"protocol populations invalid");
  doc.required_populations.forEach(x=>evidenceText(x,"population"));
  evidenceText(doc.boundary,"protocol boundary");
}
export function buildEvaluationProtocol(input) {
  exactFields(input,["schema","id","comparison","data_policy","metrics","required_populations","boundary"],[],"protocol input");
  protocolBody(input); return sealDocument(structuredClone(input),"protocol_sha256");
}
export function validateEvaluationProtocol(doc) {
  exactFields(doc,["schema","id","comparison","data_policy","metrics","required_populations","boundary","protocol_sha256"],[],"protocol");
  protocolBody(doc); return assertDocumentDigest(doc,"protocol_sha256");
}
function recordBody(doc) {
  need(doc.schema === EVALUATION_RECORD_SCHEMA,"unsupported evaluation record");
  for(const k of ["subject","protocol","dataset","cohort","run","code","environment","configuration"])validateSubjectReference(doc[k]);
  need(doc.protocol.schema===EVALUATION_PROTOCOL_SCHEMA&&doc.protocol.kind==="evidence_document","evaluation protocol reference mismatch");
  evidenceText(doc.population,"population");
  need(doc.origin === "externally_reported", "import cannot assert local recomputation");
  need(["paired","unpaired","descriptive"].includes(doc.comparison),"unknown evaluation comparison mode");
  if(doc.pair_set!==null)validateSubjectReference(doc.pair_set);
  if(doc.comparison!=="paired")need(doc.pair_set===null,"non-paired record cannot claim a pair set");
  uniqueEvidenceIds(doc.metrics,"evaluation metrics"); need(doc.metrics.length>0&&doc.metrics.length<=64,"evaluation has no metrics");
  for(const m of doc.metrics){
    exactFields(m,["id","value","unit","analysis_unit","denominator","missing_count","resolution","ci"],[],"measurement");
    finite(m.value,"estimate");evidenceText(m.unit,"measurement unit");evidenceText(m.analysis_unit,"measurement analysis unit");
    exactCount(m.denominator,"denominator",true); exactCount(m.missing_count,"missing count");
    if(m.resolution!==null){finite(m.resolution,"measurement resolution");need(m.resolution>0,"measurement resolution must be positive");}
    if(m.ci!==null){
      exactFields(m.ci,["low","high","level","method","resampling_unit","repetitions"],[],"confidence interval");
      for(const k of ["low","high","level"])finite(m.ci[k],`CI ${k}`);
      need(m.ci.low<=m.ci.high&&m.ci.level>0&&m.ci.level<1,"invalid CI bounds/level");
      evidenceText(m.ci.method,"CI method");
      if(m.ci.resampling_unit!==null)evidenceText(m.ci.resampling_unit,"resampling unit");
      if(m.ci.repetitions!==null)exactCount(m.ci.repetitions,"resampling repetitions",true);
    }
  }
  evidenceText(doc.boundary,"evaluation boundary");
}
export function buildEvaluationRecord(input) {
  exactFields(input,["schema","subject","protocol","dataset","cohort","run","code","environment","configuration","population","origin","comparison","pair_set","metrics","boundary"],[],"evaluation input");
  recordBody(input);return sealDocument(structuredClone(input),"evaluation_sha256");
}
export function validateEvaluationRecord(doc) {
  exactFields(doc,["schema","subject","protocol","dataset","cohort","run","code","environment","configuration","population","origin","comparison","pair_set","metrics","boundary","evaluation_sha256"],[],"evaluation record");
  recordBody(doc);return assertDocumentDigest(doc,"evaluation_sha256");
}
export function validateMeasurementAgainstProtocol(record, protocol) {
  validateEvaluationRecord(record);validateEvaluationProtocol(protocol);
  need(record.protocol.sha256===protocol.protocol_sha256,"evaluation refers to another protocol");
  need(record.comparison===protocol.comparison,"comparison mode disagrees with protocol");
  need(protocol.required_populations.includes(record.population),"evaluation population outside protocol");
  for(const m of record.metrics){
    const rule=protocol.metrics.find(r=>r.id===m.id);
    need(rule&&rule.unit===m.unit&&rule.analysis_unit===m.analysis_unit,"metric definition/unit/analysis unit mismatch");
    need(rule.range===null||(m.value>=rule.range[0]&&m.value<=rule.range[1]),"estimate outside protocol range");
  }
  return record;
}
