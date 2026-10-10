import {validateWorkflowResult} from './evidence-workflow.js';
import {sha256TextHex} from './sha256-sync.js';
import {spdxFilePurposePackage} from './spdx-artifact-export.js';
export function projectEvidenceRecord(result,format){
  validateWorkflowResult(result);
  if(!['cyclonedx','spdx'].includes(format))throw Error('Unsupported evidence projection');
  const content=JSON.stringify(result,null,2)+'\n',sha256=sha256TextHex(content);
  const ledger=[
    {source:'/result_sha256',status:'mapped_identifier',meaning:'DEEPBOM canonical result identity, distinct from serialized file SHA-256'},
    {source:'/',status:'external_document_required',meaning:'The complete evidence-result.json is required to reconstruct all evidence checks'},
    {source:'/document/checks',status:'not_mapped',meaning:'Evaluation checks are not translated into clinical, safety or regulatory verdicts'},
    {source:'/request/documents',status:'not_mapped',meaning:'Snapshot, OMOP, Training, Weight and Activation semantics remain in their source documents'},
  ];
  let document;
  if(format==='cyclonedx')document={$schema:'http://cyclonedx.org/schema/bom-1.7.schema.json',bomFormat:'CycloneDX',specVersion:'1.7',version:1,metadata:{component:{'bom-ref':'evidence-result',type:'file',name:'evidence-result.json',hashes:[{alg:'SHA-256',content:sha256}],properties:[{name:'deepbom:evidence:resultSha256',value:result.result_sha256},{name:'deepbom:evidence:scope',value:'Evidence document inventory only; not a model dependency BOM'}]}}};
  else document={spdxVersion:'SPDX-2.3',dataLicense:'CC0-1.0',SPDXID:'SPDXRef-DOCUMENT',name:'DEEPBOM evidence document inventory',documentNamespace:`https://deepbom.org/spdx/evidence/${sha256}`,creationInfo:{creators:['Tool: DEEPBOM'],created:'1970-01-01T00:00:00Z',comment:'Deterministic serialization epoch; not the evaluation or collection time.'},packages:[spdxFilePurposePackage({id:'SPDXRef-EvidenceResult',name:'evidence-result.json',sha256})],documentDescribes:['SPDXRef-EvidenceResult'],relationships:[],comment:'Evidence document inventory only; no complete model dependency or license claim.'};
  return {schema:'deepbom.evidence_standard_projection.v1',format,document,source_file:{name:'evidence-result.json',encoding:'UTF-8',content,sha256,byte_length:new TextEncoder().encode(content).length,result_sha256:result.result_sha256},loss_ledger:ledger};
}
