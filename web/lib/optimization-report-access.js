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
