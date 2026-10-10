import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,mkdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {createServer} from 'node:http';
import {chromium} from 'playwright';
import {syntheticScenario} from '../examples/evidence-workflow/synthetic.mjs';
import {runEvidenceWorkflow,queryWorkflowResult} from '../web/lib/evidence-workflow.js';
import {evidenceWorkflow} from '../sdk/index.mjs';
const root=process.cwd(),dir=await mkdtemp(path.join(tmpdir(),'deepbom-workflow-'));
const fixture=syntheticScenario(),expected=runEvidenceWorkflow(fixture.request,{files:fixture.files}),fileOptions=[];
await writeFile(path.join(dir,'request.json'),JSON.stringify(fixture.request));
await writeFile(path.join(dir,'result.json'),JSON.stringify(expected));
for(const [i,f] of fixture.files.entries()){const name=path.join(dir,`source-${i}.bin`);await writeFile(name,f.bytes);fileOptions.push({sha256:f.ref.sha256,path:name});}
const sdk=await evidenceWorkflow(fixture.request,{files:fileOptions});assert.deepEqual(sdk,expected);
assert.deepEqual(await evidenceWorkflow(expected),expected);
const mcp=spawn(process.execPath,['bin/deepbom.mjs','mcp'],{cwd:root,env:{...process.env,DEEPBOM_MCP_ALLOWED_ROOTS:dir},stdio:['pipe','pipe','pipe']});
const pending=new Map();let serial=0,buffer='',stderr='';mcp.stderr.on('data',b=>{stderr+=b.toString();});
mcp.stdout.on('data',b=>{buffer+=b.toString();let newline;while((newline=buffer.indexOf('\n'))>=0){const line=buffer.slice(0,newline);buffer=buffer.slice(newline+1);if(!line.trim())continue;const v=JSON.parse(line);pending.get(v.id)?.(v);pending.delete(v.id);}});
const call=(method,params)=>new Promise((resolve,reject)=>{const id=++serial;const timer=setTimeout(()=>{pending.delete(id);reject(Error('MCP timeout '+stderr));},30000);pending.set(id,x=>{clearTimeout(timer);resolve(x);});mcp.stdin.write(JSON.stringify({jsonrpc:'2.0',id,method,params})+'\n');});
try{await call('initialize',{protocolVersion:'2025-11-25',capabilities:{},clientInfo:{name:'workflow-test',version:'1'}});mcp.stdin.write(JSON.stringify({jsonrpc:'2.0',method:'notifications/initialized'})+'\n');
 const response=await call('tools/call',{name:'deepbom_evidence_workflow',arguments:{path:path.join(dir,'request.json'),files:fileOptions,limit:1}});
 assert(!response.result.isError,JSON.stringify(response));
 const data=JSON.parse(response.result.content[0].text);assert.deepEqual(data,queryWorkflowResult(expected,{limit:1}));
 const denied=await call('tools/call',{name:'deepbom_evidence_workflow',arguments:{path:'/etc/passwd'}});assert(denied.result.isError);
}finally{mcp.kill();}
const server=createServer(async(req,res)=>{try{const u=new URL(req.url,'http://local');const relative=u.pathname==='/reports/evidence/'?'web/reports/evidence/index.html':u.pathname.replace(/^\//,'');const target=path.resolve(root,relative);if(!target.startsWith(root+path.sep))throw Error('path');const data=await readFile(target);res.setHeader('Content-Type',target.endsWith('.js')?'text/javascript':target.endsWith('.css')?'text/css':target.endsWith('.html')?'text/html':'application/octet-stream');res.end(data);}catch{res.writeHead(404);res.end();}});await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({headless:true});
try{
 const page=await browser.newPage({viewport:{width:1400,height:950}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(`http://127.0.0.1:${server.address().port}/reports/evidence/`);
 await page.locator('#request-file').setInputFiles(path.join(dir,'request.json'));await page.locator('#run').click();await page.waitForFunction(()=>document.querySelector('#verdict').textContent==='incomplete');
 await page.locator('#evidence-files').setInputFiles(fileOptions.map(f=>f.path));await page.locator('#run').click();await page.waitForFunction(()=>document.querySelector('#verdict').textContent==='criteria_not_met');
 await page.locator('#rows button').first().click();assert.match(await page.locator('#detail').textContent(),/evaluation_sha256/);
 const download=page.waitForEvent('download');await page.locator('#save-json').click();const handle=await download;const bytes=await readFile(await handle.path(),'utf8');assert.deepEqual(JSON.parse(bytes),expected);
 await page.locator('#search').fill('original');assert.equal(await page.locator('#rows button').count(),1);
 await mkdir('.local-validation/snapshot-release-2.3.0',{recursive:true});await page.screenshot({path:'.local-validation/snapshot-release-2.3.0/evidence-browser.png',fullPage:true});
 await page.locator('#evidence-files').setInputFiles([]);await page.locator('#request-file').setInputFiles(path.join(dir,'result.json'));await page.locator('#run').click();await page.waitForFunction(()=>document.querySelector('#verdict').textContent==='criteria_not_met');
 assert.deepEqual(errors,[]);
}finally{await browser.close();await new Promise(r=>server.close(r));}
await writeFile('.local-validation/snapshot-release-2.3.0/channel-scenario.json',JSON.stringify({directory:dir,result_sha256:expected.result_sha256,counts:expected.document.counts,checks:['common core','Node SDK/subprocess CLI','live local MCP','browser Worker','missing files','saved result','JSON download','source detail','search']},null,2));
console.log('Evidence workflow Web/CLI/Node SDK/live MCP parity passed. Scenario: '+dir);
