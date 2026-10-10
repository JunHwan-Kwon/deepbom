import {omopProfile} from './provenance/omop.js';
import {buildProvenanceV2} from './provenance-v2.js';
import {exactFields,validateSubjectReference,requireEvidence as need} from './evidence-identity.js';
export const OMOP_SNAPSHOT_INPUT_SCHEMA='deepbom.omop_snapshot_input.v1';
export function connectOmopSnapshot(input,context){
  exactFields(input,['schema','subject','snapshot','omop'],[],'OMOP snapshot connection');
  need(input.schema===OMOP_SNAPSHOT_INPUT_SCHEMA,'unsupported OMOP connection');
  validateSubjectReference(input.subject);validateSubjectReference(input.snapshot);
  need(input.snapshot.kind==='snapshot','OMOP release requires a Snapshot reference');
  const snapshot=context.document(input.snapshot);if(snapshot)need(snapshot.kind==='dataset_release','OMOP must reference a dataset release');
  const profile=omopProfile({omop:input.omop});
  return buildProvenanceV2({schema:'deepbom.provenance_input.v2',subject:input.subject,nodes:[{id:'model',kind:input.subject.kind==='artifact_file'?'model_artifact':'model_state',ref:input.subject},{id:'release',kind:'data_release',ref:input.snapshot,attributes:{omop_profile:profile}}],relationships:[{id:'model-release',from:'model',to:'release',role:'associated_with'}]},context);
}
