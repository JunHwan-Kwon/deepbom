import {open} from 'node:fs/promises';
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import {openOptimizationReport, queryOptimizationReport, REPORT_MAX_BYTES} from '../web/lib/optimization-report-access.js';
import {optimizationReportHtml} from '../web/lib/optimization-report-html.js';
import {writeOutputAtomically} from './deepbom-automation.mjs';

export const REPORT_HELP=`Read a saved native optimization report; no model code is executed.
  deepbom optimization-report <report.json> [--section summary|changes|structure|subject]
    [--subject <exact-id>] [--search <text>] [--offset <n>] [--limit <1-100>]
    [--expected-sha256 <canonical-report-digest>]
  deepbom optimization-report <report.json> --format json|diff|html|pdf --output <file>
PDF requires the matching Python deepbom[report] package. DEEPBOM_REPORT_PYTHON selects its interpreter.
This npm command reads report.json; the Python saved-package command accepts a candidate directory.
`;
export async function runOptimizationReport(argv) {
  if(argv.length===1&&['--help','-h'].includes(argv[0]))return process.stdout.write(REPORT_HELP);
  const filename=argv[0];if(!filename||filename.startsWith('-'))throw Error(REPORT_HELP);
  const options={},allowed=['section','subject','search','offset','limit','expected-sha256','format','output'];
  let mcpExport=false;
  for(let i=1;i<argv.length;i++){
    if(argv[i]==='--mcp-export'){if(mcpExport)throw Error('Duplicate option');mcpExport=true;continue;}
    const k=argv[i].replace(/^--/,'');
    if(!argv[i].startsWith('--')||!allowed.includes(k)||Object.hasOwn(options,k)||i+1>=argv.length)throw Error(`Invalid report option: ${argv[i]}`);
    options[k]=argv[++i];
  }
  if(mcpExport&&options.output)throw Error('MCP export does not write a local output');
  if(options.format&&['section','subject','search','offset','limit'].some(k=>Object.hasOwn(options,k)))throw Error('Query selectors cannot be combined with export format');
  if(mcpExport&&!options.format)throw Error('MCP export requires format');
  const handle=await open(filename,'r');let text;
  try{const s=await handle.stat();if(!s.isFile()||s.size>REPORT_MAX_BYTES)throw Error('Report must be a regular JSON file of at most 16 MiB');
    // Read only one bounded descriptor; never allocate from a growing file.
    const bytes=Buffer.alloc(s.size+1);let n=0;
    while(n<bytes.length){const r=await handle.read(bytes,n,bytes.length-n,null);if(!r.bytesRead)break;n+=r.bytesRead;}
    if(n!==s.size)throw Error('Report changed while reading');text=new TextDecoder('utf-8',{fatal:true}).decode(bytes.subarray(0,n));
  }finally{await handle.close();}
  const data=openOptimizationReport(text,options['expected-sha256']);
  if(!options.format){
    const query={};for(const k of ['section','subject','search'])if(Object.hasOwn(options,k))query[k]=options[k];
    for(const k of ['offset','limit'])if(Object.hasOwn(options,k)){if(!/^\d+$/.test(options[k]))throw Error(`Invalid ${k}`);query[k]=Number(options[k]);}
    const bytes=JSON.stringify(queryOptimizationReport(data,query),null,2)+'\n';
    if(options.output)await writeOutputAtomically(options.output,bytes,{noClobber:true});else process.stdout.write(bytes);
    return;
  }
  const format=options.format;if(!['json','diff','html','pdf'].includes(format))throw Error('Report format must be json, diff, html or pdf');
  const mime={json:'application/json',diff:'application/json',html:'text/html',pdf:'application/pdf'}[format];
  const bytes=format==='pdf'?await renderPdf(text):Buffer.from(format==='html'?optimizationReportHtml(data):JSON.stringify(format==='diff'?data.diff:data.report,null,2)+'\n');
  if(mcpExport){
    const result={schema:'deepbom.optimization_report_export.v1',report_sha256:data.report.report_sha256,filename:format==='diff'?'report.diff.json':`report.${format}`,mime_type:mime,byte_length:bytes.length,file_sha256:createHash('sha256').update(bytes).digest('hex'),data_base64:bytes.toString('base64')};
    process.stdout.write(JSON.stringify(result)+'\n');
  }else if(options.output){await writeOutputAtomically(options.output,bytes,{noClobber:true});process.stdout.write(JSON.stringify({report_sha256:data.report.report_sha256,output:options.output,mime_type:mime,byte_length:bytes.length})+'\n');}
  else {if(format==='pdf')throw Error('PDF export requires --output');process.stdout.write(bytes);}
}

function renderPdf(text){
  return new Promise((resolve,reject)=>{
    const child=spawn(process.env.DEEPBOM_REPORT_PYTHON||'python3',['-m','deepbom.optimization.render'],{stdio:['pipe','pipe','pipe'],windowsHide:true});
    const out=[];let size=0,err='',failure;
    const stop=()=>child.kill('SIGKILL');
    const cancel=()=>{stop();process.exitCode=1;};
    process.once('SIGTERM',cancel);process.once('SIGINT',cancel);
    const timer=setTimeout(()=>{failure=Error('PDF rendering timed out');stop();},120000);timer.unref();
    child.stdout.on('data',b=>{size+=b.length;if(size>32*1024*1024){failure=Error('PDF exceeds 32 MiB export limit');stop();}else out.push(b);});
    child.stderr.on('data',b=>{if(err.length<8192)err+=b.toString().slice(0,8192-err.length);});
    child.stdin.on('error',()=>{});
    child.on('error',e=>{failure=e;});
    child.on('close',code=>{clearTimeout(timer);process.removeListener('SIGTERM',cancel);process.removeListener('SIGINT',cancel);
      const bytes=Buffer.concat(out);
      if(failure||code!==0||bytes.subarray(0,5).toString()!=='%PDF-')reject(Error(`PDF export requires the matching Python deepbom[report] installation. Set DEEPBOM_REPORT_PYTHON to its interpreter. ${failure?.message||err.trim()||'Renderer failed'}`));else resolve(bytes);
    });child.stdin.end(text);
  });
}
