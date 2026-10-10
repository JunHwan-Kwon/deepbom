// Integration over real framework outputs from check-native-training.py.
import assert from 'node:assert/strict';
import {readFile,readdir} from 'node:fs/promises';
import path from 'node:path';
import {createEvidenceContext} from '../web/lib/evidence-context.js';
import {subjectReference} from '../web/lib/evidence-identity.js';
import {snapshotFromNative} from '../web/lib/snapshot-ir.js';
import {runEvidenceWorkflow} from '../web/lib/evidence-workflow.js';
import {syntheticScenario} from '../examples/evidence-workflow/synthetic.mjs';
import {buildEvaluationRecord} from '../web/lib/evaluation-evidence.js';
const root=process.argv[2];if(!root)throw Error('Pass the real native test output directory');
const read=async file=>JSON.parse(await readFile(file,'utf8'));
const objects=async dir=>Promise.all((await readdir(dir)).filter(n=>n.endsWith('.json')).map(n=>read(path.join(dir,n))));
const training=await objects(path.join(root,'torch-training','objects'));
const context=createEvidenceContext(training);
for(const ref of context.references)assert.equal(context.resolve(ref).status,'matched');
for(const framework of ['torch','keras']){
 const dir=path.join(root,framework+'-fold'),report=await read(path.join(dir,'report.json'));
 const baseline=(await read(path.join(dir,'report-source.json'))).baseline;
 const docs=[...Object.values(baseline).filter(Boolean),...await objects(path.join(dir,'objects')),report];
 const unique=[...new Map(docs.map(d=>[JSON.stringify(d),d])).values()];
 const states=[report.baseline.snapshot_sha256,report.candidate.snapshot_sha256].map(sha=>unique.find(d=>d.snapshot_sha256===sha&&d.schema==='deepbom.model_state_snapshot.v1'));
 assert(states.every(Boolean));
 const wrapped=states.map(snapshotFromNative),refs=wrapped.map(d=>subjectReference('snapshot',d.snapshot_ir_sha256,{schema:d.schema}));
 const fixture=syntheticScenario();
 const records=fixture.records.map((r,i)=>{const {evaluation_sha256,...body}=r;return buildEvaluationRecord({...body,subject:refs[i%2]});});
 const request={...fixture.request,input:{...fixture.request.input,before:refs[0],after:refs[1],evaluations:records.map(r=>subjectReference('evidence_document',r.evaluation_sha256,{schema:r.schema})),evidence:[subjectReference('evidence_document',report.report_sha256,{schema:report.schema})]},documents:[...unique,...wrapped,...fixture.releases,fixture.protocol,...records]};
 const result=runEvidenceWorkflow(request,{files:fixture.files});
 assert.equal(result.document.structural_coverage,'supplied_record');
 assert.equal(result.document.structural_evidence.length,1);
 assert.deepEqual(result.document.counts,{total:2,pass:1,fail:1,not_assessed:0});
 // These are real model structures with synthetic estimates, never a quality claim.
 console.log(framework+' native snapshot/Model/Weight/report -> common review passed');
}
console.log('Actual Training IR/chunk/Activation/Model/Weight context closure passed');
