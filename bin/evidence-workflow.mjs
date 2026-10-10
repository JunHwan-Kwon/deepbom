import {parseStrictJson} from "../web/lib/metadata-model-adapters.js";
import {projectEvidenceRecord} from "../web/lib/evidence-standard-projection.js";
import {open} from 'node:fs/promises';
import {spawn} from 'node:child_process';
import {runEvidenceWorkflow,validateWorkflowResult,queryWorkflowResult} from '../web/lib/evidence-workflow.js';
import {subjectReference} from '../web/lib/evidence-identity.js';
import {evidenceWorkflowHtml,evidenceWorkflowReport} from '../web/lib/evidence-workflow-report.js';
import {writeOutputAtomically} from './deepbom-automation.mjs';
export const EVIDENCE_HELP=`Run a common evidence workflow without executing model code.
  deepbom evidence-workflow <request-or-result.json> [--file <sha256>:<local-path>]...
    [--format json|html|pdf|cyclonedx|spdx] [--output <new-file>]
    [--offset <n> --limit <1-100>] [--expected-sha256 <result-digest>]
Operations: snapshot, omop, protocol, evaluation, provenance, review, deployment, verify.
Saved results are reconstructed from preserved records, not freshly verified model bytes.
Requests: 16 MiB; each selected file: 64 MiB; total: 256 MiB. No network access.
PDF requires deepbom[report]; DEEPBOM_REPORT_PYTHON selects the Python interpreter.
A completed command does not mean that criteria were met. Read document.status and counts.
`;
export async function readBoundedFile(filename,maximum){
  const h=await open(filename,'r');try{const s=await h.stat();if(!s.isFile()||s.size>maximum)throw Error(`Input must be a regular file of at most ${maximum} bytes`);
    const b=Buffer.alloc(s.size+1);let n=0;while(n<b.length){const r=await h.read(b,n,b.length-n,null);if(!r.bytesRead)break;n+=r.bytesRead;}
    if(n!==s.size)throw Error('Input changed while reading');return b.subarray(0,n);
  }finally{await h.close();}
}
export async function runEvidenceCommand(argv){
  if(argv.length===1&&['-h','--help'].includes(argv[0])){process.stdout.write(EVIDENCE_HELP);return;}
  if(!argv[0]||argv[0].startsWith('-'))throw Error(EVIDENCE_HELP);
  const opts={},selected=[];for(let i=1;i<argv.length;i++){
    const key=argv[i].replace(/^--/,'');if(!argv[i].startsWith('--')||!['file','format','output','offset','limit','expected-sha256'].includes(key)||i+1>=argv.length)throw Error(`Invalid evidence option: ${argv[i]}`);
    const value=argv[++i];if(key==='file')selected.push(value);else {if(Object.hasOwn(opts,key))throw Error(`Duplicate ${key}`);opts[key]=value;}
  }
  const input=parseStrictJson(new TextDecoder('utf-8',{fatal:true}).decode(await readBoundedFile(argv[0],16*1024*1024)));
  if(selected.length>256)throw Error('Too many file selections');
  const files=[];let total=0;
  for(const item of selected){if(!/^[a-f0-9]{64}:.+$/s.test(item))throw Error('Use --file <lowercase-sha256>:<path>');const bytes=await readBoundedFile(item.slice(65),64*1024*1024);total+=bytes.length;if(total>256*1024*1024)throw Error('Selected files exceed 256 MiB');files.push({ref:subjectReference('artifact_file',item.slice(0,64)),bytes});}
  const saved=input.schema==='deepbom.evidence_workflow_result.v1';
  if(saved&&files.length)throw Error('To reverify bytes, use the preserved request with --file, not a saved result');
  const result=saved?validateWorkflowResult(input):runEvidenceWorkflow(input,{files});
  if(opts['expected-sha256']&&result.result_sha256!==opts['expected-sha256'])throw Error('Workflow result SHA-256 mismatch');
  const paging=['offset','limit'].some(k=>Object.hasOwn(opts,k));
  if(paging&&opts.format&&opts.format!=='json')throw Error('Paging is only available for JSON');
  const format=opts.format||'json';if(!['json','html','pdf','cyclonedx','spdx'].includes(format))throw Error('Unknown evidence format');
  let data;
  if(paging){const q={};for(const k of ['offset','limit'])if(opts[k]!==undefined){if(!/^\d+$/.test(opts[k]))throw Error(`Invalid ${k}`);q[k]=Number(opts[k]);}data=JSON.stringify(queryWorkflowResult(result,q),null,2)+'\n';}
  else if(['cyclonedx','spdx'].includes(format))data=JSON.stringify(projectEvidenceRecord(result,format),null,2)+'\n';
  else if(format==='html')data=evidenceWorkflowHtml(result);
  else if(format==='pdf'){if(!opts.output)throw Error('PDF requires --output');data=await pdf(evidenceWorkflowReport(result));}
  else data=JSON.stringify(result,null,2)+'\n';
  if(opts.output)await writeOutputAtomically(opts.output,data,{noClobber:true});else process.stdout.write(data);
}
function pdf(view){return new Promise((resolve,reject)=>{
  const child=spawn(process.env.DEEPBOM_REPORT_PYTHON||'python3',['-m','deepbom.evidence_render'],{stdio:['pipe','pipe','pipe'],windowsHide:true});
  let size=0,err='',failure;const out=[];const timer=setTimeout(()=>{failure=Error('PDF rendering timed out');child.kill('SIGKILL');},120000);
  child.stdout.on('data',b=>{size+=b.length;if(size>32*1024*1024){failure=Error('PDF exceeds output limit');child.kill('SIGKILL');}else out.push(b);});child.stderr.on('data',b=>{err=(err+b.toString()).slice(-8192);});child.stdin.on('error',()=>{});child.on('error',e=>{failure=e;});
  child.on('close',code=>{clearTimeout(timer);const bytes=Buffer.concat(out);if(failure||code!==0||bytes.subarray(0,5).toString()!=='%PDF-')reject(failure||Error(err||'Install deepbom[report] for PDF'));else resolve(bytes);});child.stdin.end(JSON.stringify(view));
});}
