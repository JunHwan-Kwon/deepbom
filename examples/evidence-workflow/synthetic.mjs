// Public synthetic evidence only. These measurements are not clinical results.
import {buildEvaluationProtocol,buildEvaluationRecord} from '../../web/lib/evaluation-evidence.js';
import {subjectReference} from '../../web/lib/evidence-identity.js';
import {sha256BytesHex} from '../../web/lib/sha256-sync.js';
import {snapshotFromManifest} from '../../web/lib/snapshot-ir.js';
export function syntheticScenario(){
  const files=['before model state','after model state','release-A extract','release-B extract','cohort definition','evaluation code','runtime environment','pair set A','pair set B','evaluation run'].map(text=>{
    const bytes=new TextEncoder().encode('SYNTHETIC / '+text);return {ref:subjectReference('artifact_file',sha256BytesHex(bytes)),bytes};
  });
  const [before,after,dataA,dataB,cohort,code,environment,pairA,pairB,run]=files.map(f=>f.ref);
  const releases=[dataA,dataB].map((ref,i)=>snapshotFromManifest({schema:'deepbom.dataset_release_input.v1',identity:{namespace:'synthetic',name:'population-'+i,version:'1'},scope:{fixed_aspects:['extract_bytes'],excluded_aspects:['patient_identity'],consistency:'manifest_declared_scope'},contents:[{id:'extract',role:'evaluation_data',ref}]}));
  const protocol=buildEvaluationProtocol({schema:'deepbom.evaluation_protocol.v1',id:'synthetic-regression',comparison:'paired',data_policy:'same_release',metrics:[{id:'absolute_error',unit:'synthetic_units',analysis_unit:'synthetic_subject',higher_is_better:false,max_regression:0.25,range:[0,100],aggregation:'arithmetic_mean_absolute_error',missing_policy:'exclude and count separately',repeated_observations:'one observation per synthetic subject'}],required_populations:['adaptation','original'],boundary:'Synthetic demonstration. No patient measurements; threshold timing is not attested.'});
  const pref=subjectReference('evidence_document',protocol.protocol_sha256,{schema:protocol.schema});
  const records=[];
  for(let p=0;p<2;p++)for(let m=0;m<2;m++)records.push(buildEvaluationRecord({schema:'deepbom.evaluation_record.v1',subject:[before,after][m],protocol:pref,dataset:subjectReference('snapshot',releases[p].snapshot_ir_sha256,{schema:releases[p].schema}),cohort,code,environment,configuration:environment,run,population:protocol.required_populations[p],origin:'externally_reported',comparison:'paired',pair_set:[pairA,pairB][p],metrics:[{id:'absolute_error',value:m===0?2:p===0?1:3,unit:'synthetic_units',analysis_unit:'synthetic_subject',denominator:'100',missing_count:'0',resolution:null,ci:null}],boundary:'Synthetic externally reported estimate; not independently recomputed.'}));
  const request={schema:'deepbom.evidence_workflow_request.v1',operation:'review',input:{schema:'deepbom.change_review_input.v1',before,after,change_kind:'model_change',protocol:pref,evaluations:records.map(r=>subjectReference('evidence_document',r.evaluation_sha256,{schema:r.schema})),evidence:[]},documents:[...releases,protocol,...records]};
  return {request,files,records,protocol,releases};
}
