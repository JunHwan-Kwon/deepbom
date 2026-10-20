// Display/query projection only. Arithmetic remains in the report and IR owners.
import { assertDocumentDigest, subjectReference } from './evidence-identity.js';
import { validateOptimizationReportProjection } from './native-optimization-report.js';
const units={trainable_parameter_elements:'element',parameter_elements:'element',buffer_elements:'element',state_elements:'element',state_bytes:'byte'};

export function optimizationFacts(report) {
  assertDocumentDigest(report,'report_sha256');
  validateOptimizationReportProjection(report);
  const reference=subjectReference('evidence_document',report.report_sha256,{schema:report.schema});
  const rows=[];
  const add=(id,value,unit,pointer,scope,status='assessed',conditions=null)=>rows.push({
    id,value_decimal:value,unit,status:value===null?'not_assessed':status,scope,conditions,
    source:{reference,pointer},method:{name:'native_optimization_report_projection',version:'1.0.0'},
    basis:'static_derived',reason:value===null?'source_did_not_establish_value':null});
  for(const side of ['baseline','candidate']) {
    for(const [name,value] of Object.entries(report.comparison.totals[side])) add(`${side}.${name}`,value,units[name]||(name.includes('byte')?'byte':'element'),`/comparison/totals/${side}/${name}`,'unique native state objects; trainability from framework declaration');
    const cost=report[side].cost;
    add(`${side}.macs`,cost.macs,'MAC',`/${side}/cost/macs`,cost.scope,cost.status,
      {shape_observation_status:report[side].shape_observation_status,inputs:report[side].inputs,
        assessed_invocations:cost.assessed_invocations,observed_invocations:cost.observed_invocations,unmapped_invocations:cost.unmapped_invocations});
  }
  for(const [name,value] of Object.entries(report.static_deltas))add(`delta.${name}`,value,units[name]||(name.includes('byte')?'byte':'element'),`/static_deltas/${name}`,'candidate minus baseline; same metric definition, different model states');
  add('delta.macs',report.cost_delta,'MAC','/cost_delta','candidate minus baseline observed module contraction subtotals; inspect both probe conditions','assessed',
    {before:report.baseline.inputs,after:report.candidate.inputs,before_status:report.baseline.cost.status,after_status:report.candidate.cost.status});
  return {schema:'deepbom.analysis_facts.v1',source:reference,rows,
    validation:structuredClone(report.validation),
    boundary:'These facts retain source units, conditions and scope. Collection completion, structural validity, output equivalence, task quality and measured latency are independent. A subtotal is not complete graph cost.'};
}

export function boundRows(base, rows, {offset=0,limit=25,maxBytes=65536} = {}) {
  if(!Number.isSafeInteger(offset)||offset<0||!Number.isSafeInteger(limit)||limit<1||limit>100||!Number.isSafeInteger(maxBytes)||maxBytes<2048||maxBytes>1024*1024)throw Error('Query offset/limit/byte budget is invalid');
  const result={...base,matched:rows.length,returned:0,offset,next_offset:null,rows:[],budget:{max_bytes:maxBytes,termination:'page_limit'}};
  const encodedBytes=value=>new TextEncoder().encode(JSON.stringify(value)).length;
  for(const row of rows.slice(offset,offset+limit)) {
    result.rows.push(row);result.returned=result.rows.length;result.next_offset=offset+result.returned<rows.length?offset+result.returned:null;
    if(encodedBytes(result)>maxBytes){result.rows.pop();result.returned=result.rows.length;result.next_offset=offset+result.returned<rows.length?offset+result.returned:null;result.budget.termination='byte_budget';break;}
  }
  if(!result.returned&&offset<rows.length)throw Error('One selected row exceeds query budget; request a narrower section or larger maxBytes');
  if(result.next_offset===null)result.budget.termination='exhausted';
  if(encodedBytes(result)>maxBytes)throw Error('Query metadata exceeds byte budget');
  return result;
}
