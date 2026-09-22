import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";
import { analyzeOnnxModel } from "../web/onnx.js";
import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { sha256BytesHex } from "../web/lib/sha256-sync.js";
import { buildWeightIr } from "../web/lib/weight-ir.js";
const root=process.cwd(),bytes=new Uint8Array(await readFile("web/samples/sample_cnn_float.onnx"));
const analysis=analyzeOnnxModel(bytes,"sample.onnx"),sha=sha256BytesHex(bytes);analysis.model_sha256=sha;
const model=getArtifactIrContext(analysis,{filename:"sample.onnx",format:"onnx",sha256:sha,size:bytes.length}).model_ir;
const expected=await buildWeightIr(model,analysis,bytes);
const server=createServer(async(req,res)=>{try{
  const url=new URL(req.url,"http://localhost");
  if(url.pathname==="/"){res.writeHead(200,{"content-type":"text/html"});res.end('<main id="host"></main><script type="module">import {installNumericalPanel} from "/web/lib/numerical-ir-panel.js";const context=await(await fetch("/context")).json();context.source=await(await fetch("/model")).blob();installNumericalPanel(document.querySelector("main"),()=>context);</script>');return;}
  if(url.pathname==="/context"){res.writeHead(200,{"content-type":"application/json"});res.end(JSON.stringify({analysis,model}));return;}
  if(url.pathname==="/model"){res.writeHead(200);res.end(bytes);return;}
  const file=path.resolve(root,"."+url.pathname);if(!file.startsWith(root+path.sep))throw new Error("outside root");
  const content=await readFile(file);res.writeHead(200,{"content-type":"text/javascript"});res.end(content);
}catch{res.writeHead(404);res.end();}});
await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));let browser;
try{
  browser=await launchChromium(chromium);const page=await browser.newPage();const errors=[];page.on("pageerror",e=>errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}`);await page.locator("summary").first().click();await page.locator('[data-action="weights"]').click();
  await page.getByRole("status").filter({hasText:"6/6 payloads assessed"}).waitFor({timeout:60_000});
  assert.equal(await page.locator("canvas").count(),6);
  const downloadPromise=page.waitForEvent("download");await page.locator('[data-action="save"]').click();const download=await downloadPromise;
  const saved=JSON.parse(await readFile(await download.path(),"utf8"));assert.deepEqual(saved.weight_ir,expected);assert.equal(saved.bundle.source.artifact_sha256,sha);
  await page.locator('[type="search"]').fill('not-a-real-tensor');assert.equal(await page.locator("canvas").count(),0);
  const invalid={schema:"deepbom.activation_capture.v1"};await page.locator('[type="file"]').setInputFiles({name:"invalid.json",mimeType:"application/json",buffer:Buffer.from(JSON.stringify(invalid))});await page.getByRole("status").filter({hasText:"missing a field"}).waitFor();
  assert.deepEqual(errors,[]);console.log("Numerical browser panel: real Worker statistics equal Node IR; complete JSON download, filtering and invalid activation rejection passed.");
}finally{await browser?.close();await new Promise(resolve=>server.close(resolve));}
