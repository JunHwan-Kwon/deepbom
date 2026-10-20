import { optimizationFacts, boundRows } from "./analysis-facts.js";
// Shared read-only access contract for browser, Node CLI and local MCP.
import validate from './optimization-report-validator.js';
import { buildNativeOptimizationDiff } from './native-optimization-diff.js';
import { parseStrictJson } from './metadata-model-adapters.js';
import { validateOptimizationReportProjection } from './native-optimization-report.js';

export const REPORT_MAX_BYTES=16*1024*1024;
export function openOptimizationReport(text,expected) {
  if(typeof text!=='string'||new TextEncoder().encode(text).length>REPORT_MAX_BYTES)throw Error('Report exceeds the 16 MiB import limit');
  const report=parseStrictJson(text);
  if(!validate(report))throw Error(`Invalid optimization report: ${validate.errors[0].instancePath} ${validate.errors[0].message}`);
  if(expected!==undefined&&(!/^[a-f0-9]{64}$/i.test(expected)||report.report_sha256!==expected.toLowerCase()))throw Error('Expected report SHA-256 does not match');
  const diff=buildNativeOptimizationDiff(report);
  validateOptimizationReportProjection(report);
  return {report,diff};
}
export function queryOptimizationReport({report,diff},options={}) {
  if(options.version===2)return queryOptimizationReportV2({report,diff},options);
  if(options.version!==undefined&&options.version!==1)throw Error("Unknown query version");
  if(['maxBytes','sort'].some(k=>Object.hasOwn(options,k)))throw Error('Byte budget and sorting require query version 2');
  const {section='summary',offset=0,limit=25,subject,search=''}=options;
  if(!['summary','changes','structure','subject'].includes(section))throw Error('Unknown report section');
  if(!Number.isSafeInteger(offset)||offset<0||!Number.isSafeInteger(limit)||limit<1||limit>100)throw Error('Offset must be non-negative; limit must be 1–100');
  if(typeof search!=='string'||search.length>512)throw Error('Search must be at most 512 characters');
  if(subject!==undefined&&(typeof subject!=='string'||subject.length>4096))throw Error('Invalid subject');
  if(section==='subject'&&!subject)throw Error('The subject section requires an exact subject ID');
  if(section==='summary'&&['subject','search','offset','limit'].some(k=>Object.hasOwn(options,k)))throw Error('Summary does not accept subject, search or pagination; select changes, structure or subject');
  const base={schema:'deepbom.optimization_report_query.v1',report_sha256:report.report_sha256,before_sha256:diff.before_sha256,after_sha256:diff.after_sha256,section,boundary:diff.boundary};
  if(section==='summary')return {...base,context:report.context,counts:diff.counts,changed_entries:diff.changed_entries,totals:report.comparison.totals,static_deltas:report.static_deltas,cost:{before:report.baseline.cost,after:report.candidate.cost,delta:report.cost_delta},rule_count:diff.rules.length};
  const source=section==='changes'?diff.entries.filter(e=>e.tags.length):diff.entries;
  const matched=source.filter(e=>(!subject||e.id===subject)&&(!search||JSON.stringify([e.id,e.tags,e.before?.module?.kind,e.after?.module?.kind]).toLowerCase().includes(search.toLowerCase())));
  if(section==='subject'&&!matched.length)throw Error('Unknown report subject');
  const rows=matched.slice(offset,offset+limit);
  const rules=new Set(rows.flatMap(e=>e.rules));
  return {...base,total:source.length,matched:matched.length,returned:rows.length,offset,next_offset:offset+rows.length<matched.length?offset+rows.length:null,rows,rules:diff.rules.filter(r=>rules.has(r.index))};
}

export function queryOptimizationReportV2({report,diff},options={}) {
  const allowed=['version','section','offset','limit','subject','search','maxBytes','sort'];
  if(Object.keys(options).some(k=>!allowed.includes(k)))throw Error('Unknown report query option');
  const {section='summary',search='',subject,sort='source'}=options;
  if(!['summary','facts','validation','invocations','changes','structure','subject'].includes(section))throw Error('Unknown v2 report section');
  if(typeof search!=='string'||search.length>512||subject!==undefined&&(typeof subject!=='string'||subject.length>4096))throw Error('Invalid report search/subject');
  if(!['source','macs_desc'].includes(sort)||sort==='macs_desc'&&section!=='invocations')throw Error('Sorting is only defined for the complete invocation scope');
  const facts=optimizationFacts(report);
  const base={schema:'deepbom.optimization_report_query.v2',report_sha256:report.report_sha256,before_sha256:diff.before_sha256,after_sha256:diff.after_sha256,
    section,boundary:facts.boundary,selection:{search,subject:subject??null,sort},export_scope:'returned_rows_with_source_references'};
  let rows;
  if(section==='summary')rows=[{id:'summary',counts:diff.counts,changed_entries:diff.changed_entries,rule_count:diff.rules.length,
    totals:report.comparison.totals,static_deltas:report.static_deltas,cost:{before:report.baseline.cost,after:report.candidate.cost,delta:report.cost_delta},
    validation:facts.validation,probe_conditions:{before:report.baseline.inputs,after:report.candidate.inputs}}];
  else if(section==='facts')rows=facts.rows;
  else if(section==='validation')rows=[{id:'validation',...facts.validation,source:{reference:facts.source,pointer:'/validation'}}];
  else if(section==='invocations')rows=['baseline','candidate'].flatMap(side=>report[side].invocations.map((row,index)=>({id:`${side}:${row.subject}:${row.invocation}`,side,...row,source:{reference:facts.source,pointer:`/${side}/invocations/${index}`}})));
  else rows=section==='changes'?diff.entries.filter(row=>row.tags.length):diff.entries;
  if(sort==='macs_desc')rows=[...rows].sort((a,b)=>{
    const x=a.macs?.value_decimal,y=b.macs?.value_decimal;
    if(x===null||x===undefined)return y===null||y===undefined?0:1;
    if(y===null||y===undefined)return -1;return BigInt(x)>BigInt(y)?-1:BigInt(x)<BigInt(y)?1:0;
  });
  const total=rows.length;
  rows=rows.filter(row=>(!subject||row.id===subject||row.subject===subject)&&(!search||JSON.stringify(row).toLowerCase().includes(search.toLowerCase())));
  if(section==='subject'&&(!subject||!rows.length))throw Error('Subject query requires an existing exact subject ID');
  return boundRows({...base,total},rows,options);
}
