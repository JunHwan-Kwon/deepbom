import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";
import { buildNativeOptimizationDiff } from "../web/lib/native-optimization-diff.js";

const root=process.cwd();
const file=process.argv[2];
assert(file,"Supply a generated report.html from the Python report tests");
const html=await readFile(file,"utf8"),match=html.match(/<script id="model-diff-data" type="application\/json">([\s\S]*?)<\/script>/);
assert(match,"Missing embedded common projection");
const python=JSON.parse(match[1]);
for(const e of python.entries)delete e.patch;
const report=JSON.parse(await readFile(path.join(path.dirname(file),"report.json"),"utf8"));
assert.deepEqual(python,buildNativeOptimizationDiff(report),"Python bridge/common Node projection parity");
const server=createServer(async(req,res)=>{
  if(req.url==="/"){res.setHeader("Content-Type","text/html");res.end("<!doctype html><title>Diff parity</title>");return;}
  const p=path.resolve(root,"."+new URL(req.url,"http://local").pathname);
  if(!p.startsWith(path.join(root,"web")+path.sep)){res.statusCode=404;res.end();return;}
  try{res.setHeader("Content-Type","text/javascript");res.end(await readFile(p));}catch{res.statusCode=404;res.end();}
});
await new Promise(r=>server.listen(0,"127.0.0.1",r));
let browser;
try{
  browser=await launchChromium(chromium);
  const page=await browser.newPage(),errors=[];page.on("pageerror",e=>errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const inBrowser=await page.evaluate(async report=>{
    const {buildNativeOptimizationDiff}=await import("/web/lib/native-optimization-diff.js");return buildNativeOptimizationDiff(report);
  },report);
  assert.deepEqual(inBrowser,python,"Browser/Node/Python field-level projection parity");
  assert.deepEqual(errors,[]);
  console.log("Optimization diff parity: browser common module, Node owner and packaged Python HTML projection are identical.");
}finally{await browser?.close();await new Promise(r=>server.close(r));}
