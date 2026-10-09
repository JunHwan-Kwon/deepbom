import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import Ajv from "ajv/dist/2020.js";
import { buildNativeEvidence, nativeDigest } from "../web/lib/native-evidence.js";
import { buildNativeOptimizationReport } from "../web/lib/native-optimization-report.js";
import { buildNativeOptimizationDiff, optimizationFieldChanges } from "../web/lib/native-optimization-diff.js";

const input = {
  schema: "deepbom.native_evidence_input.v1",
  definition: {
    framework: {name:"pytorch",version:"test",backend:"torch"},
    nodes: [{id:"linear",kind:"torch.nn.modules.linear.Linear",config:{},state_refs:["linear.weight"],mode:"eval"}],
    relationships:[], inputs:{}, scope:"module_tree",limitations:[],
  },
  tensors:[{id:"linear.weight",name:"linear.weight",role:"parameter",trainable:true,aliases:[],dtype:"F32",shape:[2,3],byte_order:"little",data_base64:Buffer.alloc(24).toString("base64")}],
  capture:null,options:{max_values:0,advanced:false},
};
const bundle = await buildNativeEvidence(input);
const tensor = (shape,dtype="float32") => ({shape,dtype});
const probe = {status:"observed",inputs:[tensor([4,3])],outputs:[tensor([4,2])],invocations:[{subject:"linear",invocation:1,inputs:[tensor([4,3])],outputs:[tensor([4,2])]}]};
const request = {baseline:bundle,candidate:bundle,transformations:[],validation:{},context:{version:1,objective:"structure",target:"current-cpu",selection:"explicit_rules",user_rationale:null,probes:{baseline:probe,candidate:probe}}};
const report = buildNativeOptimizationReport(request);
const ajv = new Ajv({strict:true,allErrors:true});
for(const name of ["numerical-ir-v1","native-evidence-v1","optimization-report-v1"])
  ajv.addSchema(JSON.parse(await readFile(`docs/schemas/deepbom-${name}.schema.json`,"utf8")));
const valid = ajv.getSchema("https://deepbom.org/schemas/deepbom-optimization-report-v1.schema.json");
assert(valid(report),JSON.stringify(valid.errors));
assert.equal(report.baseline.cost.macs,"24");
assert.equal(report.candidate.invocations[0].outputs[0].logical_bytes,"32");
assert.equal(report.cost_delta,"0");
assert.equal(report.static_deltas.state_bytes,"0");
const {report_sha256,...body}=report;
assert.equal(nativeDigest(body),report_sha256);
assert.deepEqual(buildNativeOptimizationReport(request),report);
const view=buildNativeOptimizationDiff(report);
const validDiff=ajv.compile(JSON.parse(await readFile("docs/schemas/deepbom-optimization-diff-v1.schema.json","utf8")));
assert(validDiff(view),JSON.stringify(validDiff.errors));
assert.equal(view.changed_entries,0);
assert.equal(view.schema,"deepbom.optimization_diff.v1");
assert.deepEqual(view.counts,{added:0,removed:0,changed:0,unchanged:1});
const reseal=r=>{const {comparison_sha256,...c}=r.comparison;r.comparison.comparison_sha256=nativeDigest(c);const {report_sha256,...body}=r;r.report_sha256=nativeDigest(body);return r;};
const tampered=JSON.parse(JSON.stringify(report));tampered.baseline.nodes[0].config.kernel=5;
assert.throws(()=>buildNativeOptimizationDiff(tampered),/digest mismatch/);
assert.throws(()=>buildNativeOptimizationDiff(reseal(tampered)),/verdicts/);
for (const mutate of [
  r => { r.comparison.tensors = []; },
  r => { r.comparison.tensors[0].id = "foreign"; },
  r => { r.comparison.tensors[0].payload_equal = false; },
  r => { r.comparison.tensors.push(structuredClone(r.comparison.tensors[0])); },
  r => { r.comparison.tensors[0].status = "added"; },
  r => { r.comparison.tensors[0].matching_basis = "guessed"; },
]) {
  const changed = structuredClone(report); mutate(changed);
  assert.throws(() => buildNativeOptimizationDiff(reseal(changed)), /tensor comparison/);
}
const stateOnly=JSON.parse(JSON.stringify(report));stateOnly.candidate.storage[0].payload_sha256="0".repeat(64);stateOnly.comparison.tensors[0].payload_equal=false;
const stateView=buildNativeOptimizationDiff(reseal(stateOnly));
assert.equal(stateView.entries[0].status,"unchanged");assert.deepEqual(stateView.entries[0].tags,["state"]);
assert(stateView.entries[0].fields.some(f=>f.path==="/registered_state"));
const missingState=JSON.parse(JSON.stringify(report));missingState.candidate.nodes[0].state_refs=["foreign"];
missingState.comparison.nodes[0].status="changed";
assert.throws(()=>buildNativeOptimizationDiff(reseal(missingState)),/unbound state/);
const duplicateNode=JSON.parse(JSON.stringify(report));duplicateNode.baseline.nodes.push(duplicateNode.baseline.nodes[0]);
assert.throws(()=>buildNativeOptimizationDiff(reseal(duplicateNode)),/duplicate native node/);
assert.deepEqual(optimizationFieldChanges({x:1},{x:1.0}),[]);
const typed=optimizationFieldChanges({x:0},{x:false});assert.equal(typed.length,1);
const absent=optimizationFieldChanges({}, {"a/b~c":null});
assert.deepEqual(absent,[{path:"/a~1b~0c",before_present:false,after_present:true,before:null,after:null}]);
const removedNull=optimizationFieldChanges({x:null},{});assert.equal(removedNull[0].after_present,false);
const contextOnly=JSON.parse(JSON.stringify(report));contextOnly.candidate.limitations.push("additional limit");
assert.equal(buildNativeOptimizationDiff(reseal(contextOnly)).entries[0].status,"context");
const unowned=JSON.parse(JSON.stringify(report));
unowned.baseline.nodes[0].state_refs=[];unowned.candidate.nodes[0].state_refs=[];
unowned.candidate.storage[0].payload_sha256="0".repeat(64);unowned.comparison.tensors[0].payload_equal=false;
assert(buildNativeOptimizationDiff(reseal(unowned)).entries.some(e=>e.id==="Unowned registered state"&&e.tags.includes("state")));
const opaque=JSON.parse(JSON.stringify(report));opaque.candidate.model_ir_sha256="0".repeat(64);
assert(buildNativeOptimizationDiff(reseal(opaque)).entries[0].fields.some(f=>f.path==="/unprojected_model_ir_identity"));

const no=structuredClone(request);
no.context.probes.baseline=no.context.probes.candidate={status:"not_requested",inputs:[],outputs:[],invocations:[]};
assert.equal(buildNativeOptimizationReport(no).baseline.cost.macs,null);
assert.equal(buildNativeOptimizationReport(no).cost_delta,null);
const twice=structuredClone(request);
twice.context.probes.candidate.invocations.push({...probe.invocations[0],invocation:2});
assert.equal(buildNativeOptimizationReport(twice).candidate.cost.macs,"48");
assert.equal(buildNativeOptimizationReport(twice).comparison.totals.candidate.state_bytes,"24");
const invalid=structuredClone(request);invalid.context.probes.baseline.invocations[0].subject="foreign";
assert.throws(()=>buildNativeOptimizationReport(invalid),/unbound/);
const duplicate=structuredClone(request);duplicate.context.probes.baseline.invocations.push(probe.invocations[0]);
assert.throws(()=>buildNativeOptimizationReport(duplicate),/duplicate/);
const badShape=structuredClone(request);badShape.context.probes.baseline.invocations[0].outputs[0].shape=[4,8];
assert.equal(buildNativeOptimizationReport(badShape).baseline.cost.macs,null);
const scalar=structuredClone(request);scalar.context.probes.baseline.invocations[0].outputs[0]={shape:[],dtype:"float64"};
assert.equal(buildNativeOptimizationReport(scalar).baseline.invocations[0].outputs[0].logical_bytes,"8");
const large=structuredClone(request);large.context.probes.baseline.invocations[0].outputs[0]={shape:[9007199254740991,2],dtype:"float32"};
assert.equal(buildNativeOptimizationReport(large).baseline.invocations[0].outputs[0].logical_bytes,"72057594037927928");
const customInput=structuredClone(input);customInput.definition.nodes[0].kind="custom.Linear";
const custom=await buildNativeEvidence(customInput);
const unqualified=buildNativeOptimizationReport({...request,baseline:custom,candidate:custom});
assert.equal(unqualified.baseline.cost.macs,null);
assert.equal(unqualified.baseline.cost.unmapped_invocations,1);
const ambiguousInput=structuredClone(input);
ambiguousInput.tensors.push({...ambiguousInput.tensors[0],id:"other.weight",name:"other.weight"});
ambiguousInput.definition.nodes[0].state_refs.push("other.weight");
const ambiguous=await buildNativeEvidence(ambiguousInput);
const ambiguousReport=buildNativeOptimizationReport({...request,baseline:ambiguous,candidate:ambiguous});
assert.equal(ambiguousReport.baseline.cost.macs,null);
assert.match(ambiguousReport.baseline.invocations[0].macs.reason,/unambiguous/);
const rankN=structuredClone(request);
rankN.context.probes.candidate.invocations[0].inputs[0].shape=[4,5,3];
rankN.context.probes.candidate.invocations[0].outputs[0].shape=[4,5,2];
assert.equal(buildNativeOptimizationReport(rankN).candidate.cost.macs,"120");
const reluInput=structuredClone(input);reluInput.definition.nodes[0].kind="torch.nn.modules.activation.ReLU";
const relu=await buildNativeEvidence(reluInput);
assert.equal(buildNativeOptimizationReport({...request,baseline:relu,candidate:relu}).candidate.cost.macs,"0");
const contradictory=structuredClone(request);contradictory.context.probes.baseline.status="not_requested";
assert.throws(()=>buildNativeOptimizationReport(contradictory),/unrequested/);
for (const directory of process.argv.slice(2)) {
  const saved=JSON.parse(await readFile(`${directory}/report.json`,"utf8"));
  assert(valid(saved),`${directory}: ${JSON.stringify(valid.errors)}`);
  const projection=buildNativeOptimizationDiff(saved);
  assert(validDiff(projection),JSON.stringify(validDiff.errors));
  assert.equal(projection.before_sha256,saved.baseline.snapshot_sha256);
  assert.equal(projection.after_sha256,saved.candidate.snapshot_sha256);
}
console.log("Optimization report projection: shared exact costs/sizes, deterministic identity, missing/invalid shapes, repeated calls, custom-module boundary and binding rejection passed.");
