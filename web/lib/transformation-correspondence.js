// Explicit many-to-many correspondence. Never infer lineage from matching names.
import {createEvidenceContext} from './evidence-context.js';
import {exactFields,validateSubjectReference,subjectReference,sealDocument,assertDocumentDigest,canonicalEvidenceDigest,requireEvidence as need} from './evidence-identity.js';

function inventory(document) {
  if(document.schema==='deepbom.model_ir.v2')return document.program.nodes.map(row=>({id:row.id,kind:'module',record:row}));
  if(document.schema==='deepbom.model_ir.v1')return document.program.operations.map(row=>({id:row.id,kind:'operation',record:row}));
  throw Error('Correspondence requires native or serialized Model IR');
}
export function buildTransformationCorrespondence(request) {
  exactFields(request,['schema','source','target','bindings','documents'],[],'correspondence request');
  need(request.schema==='deepbom.correspondence_request.v1','unsupported correspondence request');
  for(const ref of [request.source,request.target]){validateSubjectReference(ref);need(ref.kind==='evidence_document'&&ref.subject_ref===null,'whole Model IR references required');}
  const context=createEvidenceContext(request.documents),source=context.document(request.source),target=context.document(request.target);
  need(source&&target,'both Model IR documents must be supplied');
  const before=inventory(source),after=inventory(target),left=new Map(before.map(row=>[row.id,row])),right=new Map(after.map(row=>[row.id,row]));
  need(Array.isArray(request.bindings)&&request.bindings.length<=10000,'invalid correspondence bindings');
  const ids=new Set(),usedLeft=new Set(),usedRight=new Set();
  const rows=request.bindings.map(binding=>{
    exactFields(binding,['id','source_subjects','target_subjects','status','basis','evidence'],[],'correspondence binding');
    need(typeof binding.id==='string'&&binding.id.length>0&&binding.id.length<=256&&!ids.has(binding.id),'duplicate or invalid binding ID');ids.add(binding.id);
    need(['declared','observed_exporter_metadata','ambiguous'].includes(binding.status),'unsupported correspondence claim');
    need(typeof binding.basis==='string'&&binding.basis.length>0&&binding.basis.length<=4096,'correspondence basis required');
    need(Array.isArray(binding.evidence)&&binding.evidence.length<=64,'bounded evidence references required');
    if(binding.status==='observed_exporter_metadata')need(binding.evidence.length>0,'observed correspondence needs explicit source evidence');
    const refs=(values,all,used,parent)=>{
      need(Array.isArray(values)&&values.length>0&&values.length<=10000&&new Set(values).size===values.length,'nonempty unique subject sets required');
      return values.map(id=>{need(all.has(id),'unknown correspondence subject');need(!used.has(id),'overlapping bindings must be combined into one many-to-many group');used.add(id);return subjectReference('evidence_document',parent.sha256,{schema:parent.schema,subject_ref:id});});
    };
    return {...structuredClone(binding),source_refs:refs(binding.source_subjects,left,usedLeft,request.source),target_refs:refs(binding.target_subjects,right,usedRight,request.target),
      evidence_checks:binding.evidence.map(ref=>context.resolve(ref)),
      interpretation:'identity correspondence claim; no functional, numerical or task equivalence established'};
  });
  return sealDocument({schema:'deepbom.transformation_correspondence.v1',method_version:'1.0.0',request:structuredClone(request),
    source_check:context.resolve(request.source),target_check:context.resolve(request.target),rows,
    unmapped:{source:before.filter(row=>!usedLeft.has(row.id)).map(row=>row.id),target:after.filter(row=>!usedRight.has(row.id)).map(row=>row.id)},
    coverage:{source_total:before.length,target_total:after.length,source_referenced:usedLeft.size,target_referenced:usedRight.size},
    boundary:'Caller-supplied correspondence, preserving native module and exported-operation scope. Evidence availability and correspondence confidence are separate. Unmapped or ambiguous subjects are never treated as equivalent.'},'correspondence_sha256');
}
export function validateTransformationCorrespondence(document) {
  assertDocumentDigest(document,'correspondence_sha256');
  const rebuilt=buildTransformationCorrespondence(document.request);
  need(canonicalEvidenceDigest(rebuilt)===canonicalEvidenceDigest(document),'correspondence does not reconstruct from its bound request');return document;
}
export function buildMappedModelComparison(document) {
  validateTransformationCorrespondence(document);
  const context=createEvidenceContext(document.request.documents);
  const before=new Map(inventory(context.document(document.request.source)).map(row=>[row.id,row]));
  const after=new Map(inventory(context.document(document.request.target)).map(row=>[row.id,row]));
  return sealDocument({schema:'deepbom.mapped_model_comparison.v1',method_version:'1.0.0',correspondence_sha256:document.correspondence_sha256,
    source:document.request.source,target:document.request.target,
    rows:document.rows.map(row=>({id:row.id,status:row.status,source:row.source_subjects.map(id=>before.get(id)),target:row.target_subjects.map(id=>after.get(id)),
      comparison_status:row.status==='ambiguous'?'not_assessed_ambiguous':'aligned_records_only'})),unmapped:document.unmapped,
    boundary:'Opt-in aligned records, including many-to-many lowering. No native-vs-serialized parameter subtraction, semantic equivalence, rename inference or runtime benefit is inferred.'},'comparison_sha256');
}
