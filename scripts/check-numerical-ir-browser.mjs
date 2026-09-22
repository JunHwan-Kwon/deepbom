import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";
import { analyzeOnnxModel } from "../web/onnx.js";
import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { sha256BytesHex, sha256TextHex } from "../web/lib/sha256-sync.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { buildWeightIr } from "../web/lib/weight-ir.js";
import { buildActivationIr } from "../web/lib/activation-ir.js";
import { collectWeightAnalysis } from "../web/lib/weight-analysis.js";
import { linkedOperations, histogramSvg, histogramWindow, zeroLabel } from "../web/lib/weight-visuals.js";
import { statisticsOf } from "../web/lib/numerical-ir/statistics.js";
const root=process.cwd(),bytes=new Uint8Array(await readFile("web/samples/sample_cnn_float.onnx"));
const analysis=analyzeOnnxModel(bytes,"sample.onnx"),sha=sha256BytesHex(bytes);analysis.model_sha256=sha;
const model=getArtifactIrContext(analysis,{filename:"sample.onnx",format:"onnx",sha256:sha,size:bytes.length}).model_ir;
const expected=await buildWeightIr(model,analysis,bytes);
const advancedExpected=(await collectWeightAnalysis(model,analysis,bytes,{weightIr:expected})).weight_analysis;
const outputs=model.program.values.filter(v=>v.roles.includes("graph_output")),inputs=model.program.values.filter(v=>v.roles.includes("graph_input")&&!v.storage_refs.length);
const tensor=v=>({value_ref:v.id,native_locator:v.name||v.id,dtype:v.dtype,shape:v.shape,values:Array(v.shape.reduce((n,d)=>n*d,1)).fill(0)});
const configuration={purpose:"synthetic UI fixture; not a measured execution"};
const capture={schema:"deepbom.activation_capture.v1",source:expected.source,run:{id:"ui-fixture",started_at:"2026-09-22T00:00:00Z",entry_region_ref:model.program.programs[0].entry_region_refs[0],runtime:{name:"test fixture",version:"1",configured_providers:[],device:null},collector:{name:"test fixture",version:"1",sha256:"a".repeat(64)},execution:{artifact_sha256:sha,instrumented_artifact_sha256:null,configuration,configuration_sha256:sha256TextHex(canonicalJson(configuration))},probe:{kind:"synthetic_zeros",description:"Synthetic UI contract test; no execution claim"},runtime_evidence:null},inputs:inputs.map(tensor),requested_value_refs:outputs.map(v=>v.id),captures:outputs.map(tensor),missing:[]};
const expectedActivation=buildActivationIr(model,capture);
assert(linkedOperations(expected.tensors[0],model).length>0);
assert(linkedOperations(expectedActivation.tensors[0],model).length>0);
assert.equal(histogramWindow([{statistics:statisticsOf([NaN,Infinity])}]),null);
assert.equal(zeroLabel(statisticsOf([9007199254740993n])),"Not assessed");
assert.equal(histogramSvg({statistics:statisticsOf([9007199254740993n])},{}),null);
const server=createServer(async(req,res)=>{try{
  const url=new URL(req.url,"http://localhost");
  if(url.pathname==="/"){res.writeHead(200,{"content-type":"text/html"});res.end('<html><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="/web/research-theme.css"><link rel="stylesheet" href="/web/weight-workspace.css"><style>body{margin:20px auto;max-width:1240px;padding:0 12px}</style><main id="host"></main><script type="module">import {installNumericalPanel} from "/web/lib/numerical-ir-panel.js";window.context=await(await fetch("/context")).json();context.source=await(await fetch("/model")).blob();window.panel=installNumericalPanel(document.querySelector("main"),()=>window.context);</script></html>');return;}
  if(url.pathname==="/context"){res.writeHead(200,{"content-type":"application/json"});res.end(JSON.stringify({analysis,model}));return;}
  if(url.pathname==="/model"){res.writeHead(200);res.end(bytes);return;}
  const file=path.resolve(root,"."+url.pathname);if(!file.startsWith(root+path.sep))throw Error("outside root");
  const content=await readFile(file),mime={".css":"text/css",".json":"application/json",".wasm":"application/wasm",".html":"text/html"}[path.extname(file)]||"text/javascript";
  res.writeHead(200,{"content-type":mime});res.end(content);
}catch{res.writeHead(404);res.end();}});
await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
await mkdir(".local-validation/weight-workspace",{recursive:true});let browser;
try{
  browser=await launchChromium(chromium);const page=await browser.newPage({viewport:{width:1440,height:1200}}),errors=[];
  page.on("pageerror",e=>errors.push(e.message));await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.getByText("Structure is ready.",{exact:false}).waitFor();
  assert.equal(await page.locator('[data-inventory] button').count(),6);
  assert.equal(await page.locator('[data-histogram] svg').count(),0,"analysis stays opt-in");
  assert.equal(await page.locator('.weight-connections svg').count(),1,"structure before analysis");
  await page.locator('[data-action="weights"]').click();await page.getByRole("status").filter({hasText:"6/6 payloads assessed"}).waitFor({timeout:60000});
  assert.equal(await page.locator('[data-map] [data-tensor-id]').count(),6);
  assert.equal(await page.locator('[data-histogram] svg').count(),1);
  await page.locator('[data-map] [data-tensor-id="weight:2"]').focus();await page.keyboard.press("Enter");
  assert.equal(await page.locator('[data-inventory] [aria-pressed="true"]').getAttribute('data-tensor-id'),"weight:2");
  await page.locator('[data-bin]').first().focus();await page.keyboard.press("Enter");
  assert.match(await page.locator('[data-bin-readout]').textContent(),/Interval \[/);
  await page.locator('[data-log-count]').check();assert.match(await page.locator('[data-histogram]').textContent(),/log\(1 \+ count\)/);
  const save=async action=>{const promise=page.waitForEvent("download");await page.locator(`[data-action="${action}"]`).click();return await promise;};
  const saved=JSON.parse(await readFile(await(await save("save")).path(),"utf8"));assert.deepEqual(saved.weight_ir,expected);assert.equal(saved.bundle.source.artifact_sha256,sha);assert.deepEqual(saved.weight_analysis,advancedExpected);
  const svg=await readFile(await(await save("svg")).path(),"utf8");assert(svg.includes(model.model_ir_sha256));assert(svg.includes("w2"));
  const png=await readFile(await(await save("png")).path());assert.equal(png.subarray(0,8).toString("hex"),"89504e470d0a1a0a");assert.equal(png.readUInt32BE(16),1520);
  for (const view of ['channels','kernel','similarity','spectrum','sparsity']) {
    await page.locator('[data-inspection]').selectOption(view);
    assert(await page.locator('[data-histogram] svg').count()>0,view+' visual exists');
    assert(await page.locator('.weight-plot-head').isHidden(),'Distribution controls are hidden in advanced views');
    const figure=await readFile(await(await save('svg')).path(),'utf8');assert(figure.includes(advancedExpected.weight_analysis_sha256));
    await page.screenshot({path:'.local-validation/weight-workspace/advanced-'+view+'.png',fullPage:true});
  }
  await page.locator('[data-inspection]').selectOption('quantization');assert.match(await page.locator('[data-histogram]').textContent(),/contract unavailable/);
  await page.locator('[data-baseline-file]').setInputFiles({name:'sample.onnx',mimeType:'application/octet-stream',buffer:Buffer.from(bytes)});
  await page.locator('[data-action="weights"]').click();await page.getByRole('status').filter({hasText:'6/6 payloads assessed'}).waitFor({timeout:60000});
  await page.locator('[data-inspection]').selectOption('comparison');assert(await page.locator('[data-histogram] svg').count()>0);
  const comparison=JSON.parse(await readFile(await(await save('save')).path(),'utf8')).weight_comparison;
  assert.equal(comparison.coverage.compared_count,6);assert(comparison.tensors.every(x=>x.metrics.rmse===0));
  await page.locator('[data-inspection]').selectOption('distribution');
  await page.screenshot({path:".local-validation/weight-workspace/light.png",fullPage:true});
  const lightFill=await page.locator('[data-histogram] svg > rect').getAttribute('fill');await page.evaluate(()=>document.documentElement.dataset.theme="dark");
  await page.waitForFunction(fill=>document.querySelector('[data-histogram] svg > rect')?.getAttribute('fill')!==fill,lightFill);
  await page.screenshot({path:".local-validation/weight-workspace/dark.png",fullPage:true});
  await page.getByRole('searchbox').fill('not-a-real-tensor');assert.equal(await page.locator('[data-map] svg').count(),0);
  await page.getByRole('searchbox').fill('Conv');assert.equal(await page.locator('[data-inventory] button').count(),4,"operation search");await page.getByRole('searchbox').fill('');
  await page.getByLabel('Import activation capture').setInputFiles({name:"capture.json",mimeType:"application/json",buffer:Buffer.from(JSON.stringify(capture))});
  await page.getByRole("status").filter({hasText:"1/1 requested values captured"}).waitFor();
  assert.equal(await page.locator('[data-source="activation"]').getAttribute('aria-pressed'),"true");assert.equal(await page.locator('.weight-connections svg').count(),1);
  const combined=JSON.parse(await readFile(await(await save("save")).path(),"utf8"));assert.deepEqual(combined.activation_ir,expectedActivation);assert.deepEqual(combined.weight_ir,expected);
  await page.getByLabel('Import activation capture').setInputFiles({name:"invalid.json",mimeType:"application/json",buffer:Buffer.from('{"schema":"deepbom.activation_capture.v1"}')});
  await page.getByRole("status").filter({hasText:"missing a field"}).waitFor();assert.equal(await page.locator('[data-histogram] svg').count(),1,"failed import retains evidence");
  await page.locator('[data-source="weight"]').click();await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:".local-validation/weight-workspace/mobile.png",fullPage:true});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),"no viewport overflow");
  await page.evaluate(()=>{window.context=null;window.panel.sync();});assert.equal(await page.locator('[data-histogram] svg').count(),0);assert(await page.locator('[data-action="save"]').isDisabled());
  // Model replacement and cancel must reject late results from older Workers.
  await page.evaluate(async()=>{
    window.context=await(await fetch('/context')).json();context.source=await(await fetch('/model')).blob();panel.sync();
    window.workers=[];window.Worker=class {constructor(){workers.push(this);}postMessage(){}terminate(){this.terminated=true;}};
  });
  await page.locator('[data-action="weights"]').click();await page.locator('[data-action="cancel"]').click();
  await page.evaluate(result=>workers[0].onmessage({data:{result}}),expected);
  assert(await page.locator('[data-action="save"]').isDisabled(),'cancel rejects late worker result');
  await page.locator('[data-action="weights"]').click();
  await page.evaluate(result=>{context=null;panel.sync();workers[1].onmessage({data:{result}});},expected);
  assert.equal(await page.locator('[data-histogram] svg').count(),0,'replacement rejects late worker result');
  // Large inventories remain fully reachable rather than silently stopping at 200.
  await page.evaluate(async()=>{
    context=await(await fetch('/context')).json();context.model.model_ir_sha256='b'.repeat(64);
    const first=context.model.tensors_and_storage.storage_objects[0];
    context.model.tensors_and_storage.storage_objects=Array.from({length:241},(_,i)=>({...first,id:'test:'+i,name:'tensor_'+i}));
    context.model.weight_bindings.bindings=[];panel.sync();
  });
  assert.equal(await page.locator('[data-inventory] button').count(),40);
  assert.match(await page.locator('[data-page-label]').textContent(),/1–40 of 241/);
  await page.locator('[data-page="next"]').click();assert.match(await page.locator('[data-page-label]').textContent(),/41–80 of 241/);
  await page.getByRole('searchbox').fill('tensor_240');assert.equal(await page.locator('[data-inventory] button').count(),1);
  assert.deepEqual(errors,[]);console.log("Weight workspace: Worker/Node parity; opt-in, bindings, keyboard bins, log axis, SVG/PNG/JSON, light/dark/mobile, activation import, invalid capture and artifact reset passed.");
}finally{await browser?.close();await new Promise(resolve=>server.close(resolve));}
