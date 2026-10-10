import assert from 'node:assert/strict';
import {syntheticScenario} from '../examples/evidence-workflow/synthetic.mjs';
import {runEvidenceWorkflow,validateWorkflowResult,queryWorkflowResult} from '../web/lib/evidence-workflow.js';
import {sealDocument,subjectReference} from '../web/lib/evidence-identity.js';
import {buildEvaluationRecord,validateEvaluationRecord} from '../web/lib/evaluation-evidence.js';
import {validateSnapshotIr} from '../web/lib/snapshot-ir.js';
import {evidenceWorkflowHtml} from '../web/lib/evidence-workflow-report.js';
const s=syntheticScenario();const result=runEvidenceWorkflow(s.request,{files:s.files});
assert.equal(result.document.status,'criteria_not_met');
assert.deepEqual(result.document.counts,{total:2,pass:1,fail:1,not_assessed:0});
assert.deepEqual(result.document.checks.map(c=>c.delta),[-1,1]);
assert.equal(result.document.checks[0].delta_ci,null);
validateWorkflowResult(result);
assert.equal(queryWorkflowResult(result,{limit:1}).next_offset,1);
assert.equal(queryWorkflowResult(result,{offset:1,limit:1}).rows[0].population,'original');
assert.match(evidenceWorkflowHtml(result),/9007199254740993|synthetic_subject/);
const absent=runEvidenceWorkflow(s.request);assert.equal(absent.document.status,'incomplete');assert.equal(absent.document.counts.pass,0);
assert.throws(()=>runEvidenceWorkflow(s.request,{files:[{...s.files[0],bytes:s.files[1].bytes}]}),/mismatch/);
function reseal(d,key){const {[key]:ignored,...body}=d;return sealDocument(body,key);}
const tampered=structuredClone(result);tampered.document.counts.fail=0;tampered.document.counts.pass=2;tampered.document=reseal(tampered.document,'review_sha256');assert.throws(()=>validateWorkflowResult(reseal(tampered,'result_sha256')),/contradicts/);
function alterRecord(index,change){const fixture=syntheticScenario();const old=fixture.records[index];let modified=structuredClone(old);change(modified);modified=reseal(modified,'evaluation_sha256');fixture.request.documents=fixture.request.documents.map(d=>d===old?modified:d);fixture.request.input.evaluations[index]=subjectReference('evidence_document',modified.evaluation_sha256,{schema:modified.schema});return fixture;}
for(const [label,change,reject] of [
  ['unit',r=>r.metrics[0].unit='other',true],
  ['denominator',r=>r.metrics[0].denominator='101',false],
  ['pair_set',r=>r.pair_set=null,false],
  ['code',r=>r.code=subjectReference('artifact_file','f'.repeat(64)),false],
  ['analysis unit',r=>r.metrics[0].analysis_unit='observation',true],
]){const f=alterRecord(1,change);if(reject)assert.throws(()=>runEvidenceWorkflow(f.request,{files:f.files}),undefined,label);else assert.equal(runEvidenceWorkflow(f.request,{files:f.files}).document.checks[0].status,'not_assessed',label);}
const big=structuredClone(s.records[0]);big.metrics[0].denominator='9007199254740993';validateEvaluationRecord(reseal(big,'evaluation_sha256'));
for(const invalid of [0,100,'1.5','01','-1','NaN']){const d=structuredClone(big);d.metrics[0].denominator=invalid;assert.throws(()=>validateEvaluationRecord(reseal(d,'evaluation_sha256')));}
const round=alterRecord(1,r=>{r.metrics[0].value=2.25;r.metrics[0].resolution=0.1;});assert.equal(runEvidenceWorkflow(round.request,{files:round.files}).document.checks[0].status,'not_assessed');
const snapshot=structuredClone(s.releases[0]);snapshot.scope.fixed_aspects.push('patient_identity');assert.throws(()=>validateSnapshotIr(reseal(snapshot,'snapshot_ir_sha256')),/conflict/);
const duplicate=structuredClone(s.request);duplicate.documents.push(duplicate.documents[0]);assert.throws(()=>runEvidenceWorkflow(duplicate),/duplicate/);
const same=structuredClone(s.request);same.input.after=same.input.before;assert.throws(()=>runEvidenceWorkflow(same),/distinct/);
const missing=structuredClone(s.request);missing.documents=missing.documents.filter(d=>d!==s.records[3]); // independently cloned; remove by hash below
missing.documents=missing.documents.filter(d=>d.evaluation_sha256!==s.records[3].evaluation_sha256);assert.equal(runEvidenceWorkflow(missing,{files:s.files}).document.checks[1].status,'not_assessed');
console.log('Evidence workflow: paired two-population oracle, missing bytes, semantic resealing, exact counts, units, paired identity, rounding, pagination passed.');

// Independent date/interval and OMOP owner integration checks.
const {snapshotFromManifest}=await import('../web/lib/snapshot-ir.js');
const configuration=snapshotFromManifest({schema:'deepbom.configuration_input.v1',identity:{namespace:'synthetic',name:'serving configuration',version:'1'},scope:{fixed_aspects:['model','runtime'],excluded_aspects:['clinical_use'],consistency:'manifest_declared_scope'},contents:[{id:'model',role:'model',ref:s.files[0].ref},{id:'runtime',role:'runtime',ref:s.files[6].ref}]});
const configRef=subjectReference('snapshot',configuration.snapshot_ir_sha256,{schema:configuration.schema});
const deploymentRequest={schema:'deepbom.evidence_workflow_request.v1',operation:'deployment',documents:[configuration],input:{schema:'deepbom.deployment_context_input.v1',evaluated_configuration:configRef,deployments:[{id:'v1',configuration:configRef,site:'synthetic-site',start:'2026-10-01T00:00:00Z',end:'2026-10-02T00:00:00Z',source:s.files[9].ref}],execution:{event_id:'synthetic-event',site:'synthetic-site',at:'2026-10-01T12:00:00Z',clock_uncertainty_ms:1,configuration:configRef,output_state:'generated',source:s.files[9].ref}}};
const deployed=runEvidenceWorkflow(deploymentRequest,{files:s.files});assert.equal(deployed.document.status,'consistent_declared_execution');validateWorkflowResult(deployed);
const end=structuredClone(deploymentRequest);end.input.execution.at='2026-10-02T00:00:00Z';end.input.execution.clock_uncertainty_ms=0;assert.equal(runEvidenceWorkflow(end,{files:s.files}).document.status,'ambiguous_or_conflicting_deployment');
const overlap=structuredClone(deploymentRequest);overlap.input.deployments.push({...overlap.input.deployments[0],id:'overlap'});assert.equal(runEvidenceWorkflow(overlap,{files:s.files}).document.status,'ambiguous_or_conflicting_deployment');
const noExecution=structuredClone(deploymentRequest);noExecution.input.execution=null;assert.equal(runEvidenceWorkflow(noExecution,{files:s.files}).document.status,'execution_not_provided');
for(const date of ['2026-02-30T12:00:00Z','2026-10-01T12:00:00','2026-10-01T24:00:00Z']){const d=structuredClone(deploymentRequest);d.input.execution.at=date;assert.throws(()=>runEvidenceWorkflow(d,{files:s.files}));}
const omop={schema:'deepbom.evidence_workflow_request.v1',operation:'omop',documents:s.releases,input:{schema:'deepbom.omop_snapshot_input.v1',subject:s.files[0].ref,snapshot:subjectReference('snapshot',s.releases[0].snapshot_ir_sha256,{schema:s.releases[0].schema}),omop:{cdm_version:'5.4',instance_id:'synthetic-site',release_id:'1',cdm_source:{cdm_version:'5.4',cdm_source_name:'Synthetic release'}}}};
const provenance=runEvidenceWorkflow(omop,{files:s.files});validateWorkflowResult(provenance);assert.equal(provenance.document.input.relationships[0].role,'associated_with');assert.equal(provenance.document.relationships_status,'declared_unverified');assert(provenance.document.input.nodes[1].attributes.omop_profile.missing_fields.length>0);
const mismatch=structuredClone(omop);mismatch.input.omop.cdm_source.cdm_version='5.5';assert.throws(()=>runEvidenceWorkflow(mismatch),/contradicts/);
const {default:Ajv}=await import('ajv/dist/2020.js');const {readFile}=await import('node:fs/promises');const ajv=new Ajv({strict:true,allErrors:true});
for(const name of ['artifact-ir-v2','model-ir-v1','numerical-ir-v1','provenance-ir-v1','evidence-ir-v1','native-evidence-v1','evidence-ir-v2','evidence-workflow-v1','evidence-ir-v3'])ajv.addSchema(JSON.parse(await readFile(`docs/schemas/deepbom-${name}.schema.json`,'utf8')));
const validate=ajv.getSchema('https://deepbom.org/schemas/deepbom-evidence-workflow-v1.schema.json');
for(const doc of [result,absent,deployed,provenance,s.request,...s.records,s.protocol,...s.releases])assert(validate(doc),JSON.stringify(validate.errors));
assert(ajv.getSchema('https://deepbom.org/schemas/deepbom-evidence-ir-v3.schema.json')(s.releases[0]));
console.log('Deployment intervals, missing execution, OMOP shared metadata profile, additive schema/semantic parity passed.');
const {reportedDecimal,subtractExactDecimals,exactDecimalText,compareExactDecimals,halveExactDecimal}=await import('../web/lib/exact-rational.js');
assert.equal(exactDecimalText(subtractExactDecimals(reportedDecimal(.4),reportedDecimal(.1))),'0.3');
assert.equal(compareExactDecimals(subtractExactDecimals(reportedDecimal(.4),reportedDecimal(.1)),reportedDecimal(.3)),0);
assert.equal(exactDecimalText(subtractExactDecimals(reportedDecimal(-.2),reportedDecimal(.1))),'-0.3');
assert.equal(exactDecimalText(halveExactDecimal(reportedDecimal(5e-324))),'0.'+'0'.repeat(323)+'25');
assert.equal(exactDecimalText(subtractExactDecimals(reportedDecimal(1e308),reportedDecimal(-1e308))),'2'+'0'.repeat(308));
console.log('Reported-decimal arithmetic oracle passed, including exact threshold, subnormal and overflow-sized deltas.');

const {projectEvidenceRecord}=await import('../web/lib/evidence-standard-projection.js');
const {assertCycloneDx17}=await import('./cyclonedx-17-schema.mjs');
const {assertSpdx23}=await import('./spdx-23-schema.mjs');
const {sha256TextHex}=await import('../web/lib/sha256-sync.js');
for(const [format,check] of [['cyclonedx',assertCycloneDx17],['spdx',assertSpdx23]]){
 const projection=projectEvidenceRecord(result,format);check(projection.document);
 assert.equal(projection.source_file.sha256,sha256TextHex(JSON.stringify(result,null,2)+'\n'));
 assert.equal(projection.source_file.sha256,sha256TextHex(projection.source_file.content));
 assert.deepEqual(JSON.parse(projection.source_file.content),result);
 assert(projection.loss_ledger.some(r=>r.status==='not_mapped'));
}
console.log('Evidence projections passed pinned CycloneDX 1.7/SPDX 2.3 validation with byte hashes and explicit losses.');
