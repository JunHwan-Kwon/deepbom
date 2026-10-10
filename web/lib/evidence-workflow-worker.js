import {parseStrictJson} from "./metadata-model-adapters.js";
import {runEvidenceWorkflow,validateWorkflowResult} from './evidence-workflow.js';
import {sha256BytesHex} from './sha256-sync.js';
import {subjectReference} from './evidence-identity.js';
import {evidenceWorkflowReport,evidenceWorkflowHtml} from './evidence-workflow-report.js';
if (typeof self !== "undefined") self.onmessage=async({data})=>{
  try{
    const {request,selected}=data;
    if(!request||request.size>16*1024*1024)throw Error('Request exceeds 16 MiB');
    if(!Array.isArray(selected)||selected.length>256||selected.some(f=>f.size>64*1024*1024)||selected.reduce((n,f)=>n+f.size,0)>256*1024*1024)throw Error('Selected files exceed limits (64 MiB each, 256 MiB total)');
    const input=parseStrictJson(await request.text());
    const files=[];for(const file of selected){const bytes=new Uint8Array(await file.arrayBuffer());files.push({ref:subjectReference('artifact_file',sha256BytesHex(bytes)),bytes});}
    const saved=input.schema==='deepbom.evidence_workflow_result.v1';if(saved&&selected.length)throw Error('Select the original request to reverify files');
    const result=saved?validateWorkflowResult(input):runEvidenceWorkflow(input,{files});
    self.postMessage({result,view:evidenceWorkflowReport(result),html:evidenceWorkflowHtml(result),acquisition:saved?'saved_result':'current_request',selected_files:files.length});
  }catch(e){self.postMessage({error:e.message});}
};
