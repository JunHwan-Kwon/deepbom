import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir,symlink} from 'node:fs/promises';
import {spawn,spawnSync} from 'node:child_process';
import path from 'node:path';
import {createServer} from 'node:http';
import {createHash} from 'node:crypto';
import Ajv from 'ajv/dist/2020.js';
import {chromium} from 'playwright';
import {launchChromium} from './browser-launch.mjs';
import {buildNativeEvidence,nativeDigest} from '../web/lib/native-evidence.js';
import {buildNativeOptimizationReport} from '../web/lib/native-optimization-report.js';
import {openOptimizationReport,queryOptimizationReport} from '../web/lib/optimization-report-access.js';
import {optimizationReportHtml} from '../web/lib/optimization-report-html.js';

const root=process.cwd(),scratch=path.resolve('.local-validation/optimization-report-channels');await mkdir(scratch,{recursive:true});
const input={schema:'deepbom.native_evidence_input.v1',definition:{framework:{name:'pytorch',version:'fixture',backend:'torch'},nodes:Array.from({length:120},(_,i)=>({id:`layer-${i}`,kind:'torch.nn.modules.activation.ReLU',config:{},state_refs:[],mode:'eval'})),relationships:[],inputs:{},scope:'module_tree',limitations:[]},tensors:[],capture:null,options:{max_values:0,advanced:false}};
input.definition.nodes[0].state_refs=['layer-0.weight'];
input.tensors=[{id:'layer-0.weight',name:'layer-0.weight',role:'parameter',trainable:true,aliases:[],dtype:'F32',shape:[2],byte_order:'little',data_base64:Buffer.alloc(8).toString('base64')}];
const before=await buildNativeEvidence(structuredClone(input));input.definition.nodes[1].config.note='<img src=x onerror=alert(1)>';const after=await buildNativeEvidence(input);
const noProbe={status:'not_requested',inputs:[],outputs:[],invocations:[]};
const report=buildNativeOptimizationReport({baseline:before,candidate:after,transformations:[],validation:{},context:{version:1,objective:'structure',target:'fixture',selection:'explicit_rules',user_rationale:null,probes:{baseline:noProbe,candidate:noProbe}}});
const filename=path.join(scratch,'report.json');await writeFile(filename,JSON.stringify(report));
const data=openOptimizationReport(JSON.stringify(report));
const ajv=new Ajv({strict:true});ajv.addSchema(JSON.parse(await readFile('docs/schemas/deepbom-optimization-diff-v1.schema.json')));const validate=ajv.compile(JSON.parse(await readFile('docs/schemas/deepbom-optimization-report-access-v1.schema.json')));
const q=queryOptimizationReport(data,{section:'structure',offset:50,limit:50});assert(validate(q),JSON.stringify(validate.errors));assert.equal(q.rows[0].id,'layer-50');assert.equal(q.next_offset,100);
assert(validate(queryOptimizationReport(data)));
assert.throws(()=>openOptimizationReport(JSON.stringify(report),'0'.repeat(64)),/SHA-256/);
assert.throws(()=>openOptimizationReport('{"schema":1,"schema":2}'));
assert.throws(()=>queryOptimizationReport(data,{section:'summary',subject:'layer-1'}),/Summary/);
for(const modify of [r=>r.static_deltas.state_bytes='1',r=>r.baseline.cost.macs='4',r=>r.baseline.nodes[0].extra=true]){const r=structuredClone(report);modify(r);const {report_sha256,...body}=r;r.report_sha256=nativeDigest(body);assert.throws(()=>openOptimizationReport(JSON.stringify(r)));}
assert(!optimizationReportHtml(data).includes('<img'));
const cli=spawnSync(process.execPath,['bin/deepbom.mjs','optimization-report',filename,'--section','changes'],{encoding:'utf8'});assert.equal(cli.status,0,cli.stderr);assert.deepEqual(JSON.parse(cli.stdout),queryOptimizationReport(data,{section:'changes'}));
for(const file of process.argv.slice(2))openOptimizationReport(await readFile(file,'utf8'));

const invalidTensorReports=[];
for(const [name,modify] of [
  ['missing',r=>{r.comparison.tensors=[];}],
  ['foreign',r=>{r.comparison.tensors[0].id='foreign';}],
  ['inequality',r=>{r.comparison.tensors[0].payload_equal=false;}],
]){
  const r=structuredClone(report);modify(r);
  const {comparison_sha256,...c}=r.comparison;r.comparison.comparison_sha256=nativeDigest(c);
  const {report_sha256,...body}=r;r.report_sha256=nativeDigest(body);
  const file=path.join(scratch,`tensor-${name}.json`);await writeFile(file,JSON.stringify(r));invalidTensorReports.push(file);
  assert.throws(()=>openOptimizationReport(JSON.stringify(r)),/tensor comparison/);
  const rejected=spawnSync(process.execPath,['bin/deepbom.mjs','optimization-report',file],{encoding:'utf8'});
  assert.notEqual(rejected.status,0);assert.match(rejected.stderr,/tensor comparison/);
}

const server=createServer(async(req,res)=>{
  let uri=new URL(req.url,'http://local').pathname;
  if(uri==='/reports/optimization/')uri='/web/reports/optimization/index.html';
  const file=path.resolve(root,'.'+uri);
  if(!file.startsWith(root+path.sep)||!uri.startsWith('/web/')){res.statusCode=404;res.end();return;}
  try{res.setHeader('Content-Type',uri.endsWith('.js')?'text/javascript':uri.endsWith('.css')?'text/css':'text/html');res.end(await readFile(file));}catch{res.statusCode=404;res.end();}
});await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try{
  browser=await launchChromium(chromium);const page=await browser.newPage({acceptDownloads:true});const errors=[];page.on('pageerror',e=>errors.push(e.message));let uploads=0;page.on('request',r=>{if(r.method()==='POST')uploads++;});
  await page.goto(`http://127.0.0.1:${server.address().port}/reports/optimization/`);await page.locator('#report-file').setInputFiles(filename);await page.locator('#report-content').waitFor({state:'visible',timeout:60000});
  assert.match(await page.locator('#report-detail').textContent(),/layer-1/);assert.equal(await page.locator('#report-detail img').count(),0);
  await page.locator('#report-filter').selectOption('structure');await page.locator('#report-next').click();assert.match(await page.locator('#report-page').textContent(),/51–100/);
  await page.locator('#tree-next').click();assert.match(await page.locator('#tree-page').textContent(),/2 \/ 4/);assert.equal(await page.locator('#after-tree button').count(),35);
  await page.locator('#report-search').fill('layer-119');assert.equal(await page.locator('#report-entries button').count(),1);
  const dl=page.waitForEvent('download');await page.locator('#export-diff').click();assert.deepEqual(JSON.parse(await readFile(await (await dl).path(),'utf8')),data.diff);
  const htmlDownload=page.waitForEvent('download');await page.locator('#export-html').click();assert.equal(await readFile(await (await htmlDownload).path(),'utf8'),optimizationReportHtml(data));
  const popup=page.waitForEvent('popup');await page.locator('#export-pdf').click();const printPage=await popup;assert.equal(await printPage.title(),'DEEPBOM Model Optimization Report');assert.match(await printPage.locator('body').textContent(),new RegExp(report.report_sha256));await printPage.close();
  await page.emulateMedia({colorScheme:'dark'});await page.screenshot({path:path.join(scratch,'web-dark.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  const malformed=path.join(scratch,'invalid.json');await writeFile(malformed,'{}');await page.locator('#report-file').setInputFiles(malformed);await page.locator('#report-status[data-error=true]').waitFor();assert(await page.locator('#report-content').isHidden());
  for(const [i,file] of process.argv.slice(2).entries()){await page.locator('#report-file').setInputFiles(file);await page.locator('#report-content').waitFor({state:'visible'});assert.match(await page.locator('#report-detail').textContent(),/spatial/);await page.setViewportSize({width:1440,height:1050});await page.screenshot({path:path.join(scratch,`actual-report-${i}.png`),fullPage:true});}
  for(const file of invalidTensorReports){await page.locator('#report-file').setInputFiles(file);await page.locator('#report-status[data-error=true]').waitFor();assert.match(await page.locator('#report-status').textContent(),/tensor comparison/);assert(await page.locator('#report-content').isHidden());}
  assert.equal(uploads,0);assert.deepEqual(errors,[]);
}finally{await browser?.close();await new Promise(r=>server.close(r));}

const child=spawn(process.execPath,['bin/deepbom.mjs','mcp'],{env:{...process.env,DEEPBOM_MCP_ALLOWED_ROOTS:scratch},stdio:['pipe','pipe','pipe']});let seq=0,buffer='',stderr='';const pending=new Map();child.stdout.setEncoding('utf8');child.stderr.on('data',b=>stderr+=b);child.stdout.on('data',c=>{buffer+=c;let at;while((at=buffer.indexOf('\n'))>=0){const line=buffer.slice(0,at);buffer=buffer.slice(at+1);if(!line)continue;const m=JSON.parse(line);pending.get(m.id)?.(m);}});
const request=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq,t=setTimeout(()=>{pending.delete(id);reject(Error(`MCP timeout: ${stderr}`));},60000);pending.set(id,m=>{clearTimeout(t);pending.delete(id);resolve(m);});child.stdin.write(JSON.stringify({jsonrpc:'2.0',id,method,params})+'\n');});
const call=(name,args)=>request('tools/call',{name,arguments:args});
try{
  await request('initialize',{protocolVersion:'2025-11-25',clientInfo:{name:'report-check',version:'1'},capabilities:{}});
  const result=(await call('deepbom_optimization_report',{path:filename,section:'changes'})).result;assert.deepEqual(result.structuredContent,queryOptimizationReport(data,{section:'changes'}));
  for(const format of ['json','diff','html',...(process.env.DEEPBOM_REPORT_PYTHON?['pdf']:[])]){
    const out=(await call('deepbom_export_optimization_report',{path:filename,expected_sha256:report.report_sha256,format})).result;
    assert(!out.isError,JSON.stringify(out));assert(validate(out.structuredContent),JSON.stringify(validate.errors));const bytes=Buffer.from(out.content[1].resource.blob,'base64');assert.equal(bytes.length,out.structuredContent.byte_length);assert.equal(createHash('sha256').update(bytes).digest('hex'),out.structuredContent.file_sha256);if(format==='pdf')assert.equal(bytes.subarray(0,5).toString(),'%PDF-');
  }
  for(const args of [{path:'/etc/passwd'},{path:filename,expected_sha256:'0'.repeat(64)},{path:filename,limit:0},{path:filename,section:'subject',subject:'nonexistent'},{path:filename,passthrough:'--execute'}])assert((await call('deepbom_optimization_report',args)).result.isError);
  for(const file of invalidTensorReports)assert((await call('deepbom_optimization_report',{path:file})).result.isError);
  const link=path.join(scratch,'escape.json');await symlink('/etc/passwd',link).catch(e=>{if(e.code!=='EEXIST')throw e;});assert((await call('deepbom_optimization_report',{path:link})).result.isError);
  assert((await call('deepbom_export_optimization_report',{path:filename,format:'json'})).result.isError);
}finally{child.stdin.end();child.kill();}
console.log('Report channels passed: schema/common arithmetic, CLI parity, Web import/diff/hierarchy/paging/exports/theme/mobile, no upload, stale digest/malformed report/root escape rejection, MCP structured queries and file resources.');
