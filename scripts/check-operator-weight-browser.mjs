import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {mkdir,readFile} from 'node:fs/promises';
import path from 'node:path';
import {chromium} from 'playwright';
import {launchChromium} from './browser-launch.mjs';
import {analyzeOnnxModel} from '../web/onnx.js';
import {getArtifactIrContext} from '../web/lib/artifact-ir-context.js';
import {sha256BytesHex} from '../web/lib/sha256-sync.js';
import {buildWeightIr} from '../web/lib/weight-ir.js';
const root=process.cwd(),out=path.resolve('.local-validation/common-flow-browser');await mkdir(out,{recursive:true});
const bytes=new Uint8Array(await readFile('web/samples/sample_cnn_float.onnx'));
const analysis=analyzeOnnxModel(bytes,'sample_cnn_float.onnx');analysis.model_sha256=sha256BytesHex(bytes);
const model=getArtifactIrContext(analysis,{filename:analysis.filename,format:'onnx',sha256:analysis.model_sha256,size:bytes.length}).model_ir;
let expected;
const server=createServer(async(req,res)=>{try{
  const uri=new URL(req.url,'http://local').pathname;
  const file=path.resolve(root,uri==='/web/'?'web/index.html':'.'+uri);
  if(!file.startsWith(root+path.sep))throw Error('outside root');
  let body=await readFile(file);
  if(uri==='/web/app.js')body=Buffer.concat([body,Buffer.from('\nglobalThis.__flowSelect = index => graphWorkspace.jumpToGraphOp(index); globalThis.__flowContext = () => ({model:currentArtifactIrContext.model_ir,analysis:current});\n')]);
  res.writeHead(200,{'content-type':({'.js':'text/javascript','.css':'text/css','.html':'text/html','.json':'application/json','.wasm':'application/wasm'})[path.extname(file)]||'application/octet-stream','cache-control':'no-store'});res.end(body);
}catch{res.writeHead(404);res.end();}});
await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try{
  browser=await launchChromium(chromium);const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/web/`);
  await page.locator('#dropzone').dispatchEvent('pointerdown');await page.locator('#fileInput').focus();
  await page.waitForFunction(()=>document.querySelector('#status')?.textContent==='Ready',null,{timeout:60000});
  if(await page.locator('#agreementBackdrop').isVisible()){await page.locator('#privacyAgree').check();await page.locator('#acceptAgreement').click();}
  await page.locator('#fileInput').setInputFiles({name:'sample_cnn_float.onnx',mimeType:'application/octet-stream',buffer:Buffer.from(bytes)});
  await page.waitForFunction(()=>!document.querySelector('#runAudit')?.disabled);await page.locator('#runAudit').click();
  await page.waitForFunction(()=>/audit run complete|Audit failed/i.test(document.querySelector('#status')?.textContent||''),null,{timeout:120000});
  assert.match(await page.locator('#status').textContent(),/audit run complete/);
  const actual=await page.evaluate(()=>globalThis.__flowContext());
  expected=await buildWeightIr(actual.model,actual.analysis,bytes);
  await page.evaluate(()=>globalThis.__flowSelect(0));
  await page.waitForFunction(()=>document.querySelectorAll('#opDetail .weight-explorer-evidence').length>0,null,{timeout:120000});
  const card=page.locator('#opDetail .weight-explorer-evidence').first();
  assert.match(await card.textContent(),/Stored elements/);
  assert.match(await card.textContent(),new RegExp(expected.weight_ir_sha256));
  assert.equal(await card.locator(':scope > svg').count(),1);
  await card.locator('[data-bin]').first().focus();await page.keyboard.press('Enter');
  assert.match(await card.locator('[role=status]').textContent(),/stored values/);
  assert(!/Eff\. Rank|\bparams\b|NaN/.test(await card.textContent()));
  await card.locator('summary').filter({hasText:'Distribution detail'}).click();
  assert.equal(await card.locator('[data-distribution-details] tbody tr').count(),7);
  await card.locator('summary').filter({hasText:'Channels, sparsity'}).click();
  await card.locator('summary').filter({hasText:/^Singular spectrum$/}).click();
  await page.waitForFunction(()=>document.querySelector('#opDetail details[data-rendered=true]'));
  assert(await card.locator('details[data-rendered=true] svg').count()>0);
  for(const scheme of ['light','dark']){await page.emulateMedia({colorScheme:scheme});await page.screenshot({path:path.join(out,scheme+'.png'),fullPage:true});}
  await page.setViewportSize({width:390,height:844});
  assert(await card.evaluate(el=>el.scrollWidth<=el.clientWidth+2),'weight card must not overflow');
  await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
  // Rapid selection must not render late evidence for the previous operation.
  await page.evaluate(()=>{globalThis.__flowSelect(2);globalThis.__flowSelect(0);});
  await page.waitForFunction(hash=>document.querySelector('#opDetail .weight-explorer-evidence')?.textContent.includes(hash),expected.weight_ir_sha256);
  assert.deepEqual(errors,[]);
  console.log('Explorer browser passed: actual audit → Model IR binding → worker Weight IR → common charts, matching digest, lazy spectrum, themes/mobile and selection isolation.');
}finally{await browser?.close();await new Promise(r=>server.close(r));}
