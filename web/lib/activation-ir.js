import { validateRuntimeProvenance } from "./ir-runtime-provenance.js";
import { validateWeightIr } from "./weight-ir.js";
import { METHOD_VERSION, SHA256, bindSource, checkDigest, exactKeys, requireCondition, seal, shapeCount, sourceContract, parseNumeric } from "./numerical-ir/common.js";
import { statisticsOf, validateStatistics } from "./numerical-ir/statistics.js";
import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const ACTIVATION_IR_SCHEMA="deepbom.activation_ir.v1";
const BOUNDARY="Imported execution evidence is consistency-checked, not remotely attested. Statistics describe only the recorded inputs, run and captured values. Synthetic probes do not establish task accuracy, real workload distributions or causal relationships between layers. Configured providers do not establish per-node placement. Instrumented runs do not establish production latency. Raw inputs and activations are not embedded in this IR.";
function digest(value) { return sha256TextHex(canonicalJson(value)); }
function nonempty(value,label) { requireCondition(typeof value==="string" && value.length>0 && value.length<=4096,`invalid ${label}`); }
function validateRun(run,source,model) {
  exactKeys(run,["id","started_at","entry_region_ref","runtime","collector","execution","probe","runtime_evidence",...(Object.hasOwn(run,"provenance")?["provenance"]:[])],"execution run");
  const entries=model.program.programs.flatMap(p=>p.entry_region_refs);
  requireCondition(entries.length ? entries.includes(run.entry_region_ref) : run.entry_region_ref===null,"run entry region does not match Model IR");
  nonempty(run.id,"run id"); requireCondition(typeof run.started_at==="string" && /^\d{4}-\d{2}-\d{2}T.*Z$/.test(run.started_at) && Number.isFinite(Date.parse(run.started_at)),"invalid run timestamp");
  exactKeys(run.runtime,["name","version","configured_providers","device"],"runtime"); nonempty(run.runtime.name,"runtime name"); nonempty(run.runtime.version,"runtime version");
  requireCondition(Array.isArray(run.runtime.configured_providers) && run.runtime.configured_providers.every(p=>typeof p==="string"&&p.length>0),"invalid configured providers");
  requireCondition(run.runtime.device===null || typeof run.runtime.device==="string","invalid device");
  exactKeys(run.collector,["name","version","sha256"],"collector"); nonempty(run.collector.name,"collector name"); nonempty(run.collector.version,"collector version"); requireCondition(SHA256.test(run.collector.sha256),"missing collector digest");
  exactKeys(run.execution,["artifact_sha256","instrumented_artifact_sha256","configuration_sha256","configuration"],"execution identity");
  requireCondition(run.execution.artifact_sha256===source.artifact_sha256 && SHA256.test(run.execution.configuration_sha256),"execution identity mismatch");
  requireCondition(run.execution.configuration && typeof run.execution.configuration==="object" && !Array.isArray(run.execution.configuration) && digest(run.execution.configuration)===run.execution.configuration_sha256,"execution configuration digest mismatch");
  requireCondition(run.execution.instrumented_artifact_sha256===null || SHA256.test(run.execution.instrumented_artifact_sha256),"invalid instrumented identity");
  exactKeys(run.probe,["kind","description"],"probe"); requireCondition(["representative_data","synthetic_ones","synthetic_zeros","synthetic_identity","synthetic_basis","synthetic_random","custom"].includes(run.probe.kind),"unknown input probe kind"); nonempty(run.probe.description,"probe description");
  if(run.runtime_evidence!==null) { exactKeys(run.runtime_evidence,["schema","sha256"],"runtime evidence link"); nonempty(run.runtime_evidence.schema,"runtime evidence schema"); requireCondition(SHA256.test(run.runtime_evidence.sha256),"invalid runtime evidence digest"); }
}
function compatibleShape(actual,expected,type) {
  if(type?.rank_status==="unknown_rank") return true;
  if(type?.rank_status==="ranked") return actual.length===type.dimensions.length && type.dimensions.every((d,i)=>d.kind!=="constant"||BigInt(d.value.decimal)===BigInt(actual[i]));
  return actual.length===expected.length && expected.every((d,i)=>typeof d==="string"&&/^(0|[1-9][0-9]*)$/.test(d)?BigInt(d)===BigInt(actual[i]):!Number.isSafeInteger(d)||d<0||d===actual[i]);
}
function bindSymbols(rows,model) {
  const bindings=new Map(),values=new Map(model.program.values.map(row=>[row.id,row]));
  for(const row of rows) {
    const type=values.get(row.value_ref)?.type_contract?.root;
    for(const [axis,dim] of (type?.dimensions||[]).entries()) if(dim.kind==="symbol") {
      const key=JSON.stringify([dim.scope_ref,dim.name]),size=row.shape[axis];
      requireCondition(!bindings.has(key)||bindings.get(key)===size,"runtime shapes contradict a shared symbolic dimension");bindings.set(key,size);
    }
  }
}
function validateTensorBinding(row,model,input=false,entry=null) {
  requireCondition(typeof row.native_locator==="string" && row.native_locator.length<=4096,"invalid native locator");
  requireCondition(typeof row.dtype==="string" && row.dtype.length>0,"missing tensor dtype"); shapeCount(row.shape);
  if(row.value_ref===null) { requireCondition(!input || entry===null,"inputs must bind to the selected Model IR entry"); nonempty(row.native_locator,"unmapped capture locator"); return; }
  const value=model.program.values.find(v=>v.id===row.value_ref); requireCondition(value,"capture refers to unknown Model IR value");
  requireCondition(value.dtype===row.dtype || value.dtype==="UNKNOWN","capture dtype contradicts Model IR");
  requireCondition(!value.type_contract || ["tensor","sparse_tensor"].includes(value.type_contract.root.kind),"capture requires a tensor value type");
  requireCondition(compatibleShape(row.shape,value.shape,value.type_contract?.root),"capture shape contradicts Model IR");
  if(input) requireCondition(value.roles.includes("graph_input") && value.region_ref===entry,"input is not a serialized entry graph input");
}
function checkValues(values,dtype) {
  requireCondition(Array.isArray(values),"tensor values must be a flat array");
  for(const raw of values) {
    const value=parseNumeric(raw), integer=/^(U?INT)(8|16|32|64)$/.exec(dtype);
    if(integer) {
      requireCondition(!Object.is(value,-0),"integer activations cannot encode negative zero");
      requireCondition(typeof value==="bigint" || Number.isSafeInteger(value),"integer activation requires exact integer values; use decimal strings for int64");
      const n=BigInt(value), bits=BigInt(integer[2]), unsigned=integer[1]==="UINT", min=unsigned?0n:-(1n<<(bits-1n)), max=(1n<<(bits-(unsigned?0n:1n)))-1n;
      requireCondition(n>=min&&n<=max,"integer activation is outside dtype range");
    } else if(dtype==="BOOL") requireCondition(!Object.is(value,-0)&&(value===0||value===1||value===0n||value===1n),"BOOL activation must be 0 or 1");
    else {
      requireCondition(["FLOAT16","BFLOAT16","FLOAT32","FLOAT64"].includes(dtype) && typeof value==="number","unsupported activation numeric dtype");
      if(dtype==="FLOAT16" && Number.isFinite(value)) {
        const magnitude=Math.abs(value), step=magnitude<2**-14 ? 2**-24 : 2**(Math.floor(Math.log2(magnitude))-10);
        requireCondition(magnitude<=65504 && (magnitude===0 || Number.isInteger(magnitude/step)),"FLOAT16 value is not exactly representable");
      }
      if(dtype==="BFLOAT16" && Number.isFinite(value)) {
        const buffer=new ArrayBuffer(4),view=new DataView(buffer);view.setFloat32(0,value,true);
        requireCondition(Object.is(Math.fround(value),value) && (view.getUint32(0,true)&65535)===0,"BFLOAT16 value is not exactly representable");
      }
      if(dtype==="FLOAT32" && Number.isFinite(value)) requireCondition(Object.is(Math.fround(value),value),"FLOAT32 value is not exactly representable");
    }
  }
}
function eligibleValues(model,entry) { return model.program.values.filter(v=>!v.storage_refs.length&&!(v.roles.includes("graph_input")&&v.region_ref===entry)).map(v=>v.id); }
function coverage(requested,rows,missing,model,entry) {
  const captured=new Set(rows.map(r=>r.value_ref).filter(Boolean)), eligible=eligibleValues(model,entry);
  return { requested_count:requested.length,captured_count:captured.size,missing_count:missing.length,unmapped_capture_count:rows.filter(r=>r.value_ref===null).length,
    eligible_graph_value_count:eligible.length,not_requested_value_refs:eligible.filter(id=>!requested.includes(id)),scope:entry===null?"unmapped_runtime_values":requested.length===eligible.length&&eligible.every(id=>requested.includes(id))?"all_eligible_serialized_values":"explicit_selected_values",
    completeness:missing.length?"partial_requested_capture":"complete_requested_capture" };
}
export function buildActivationIr(modelIr,capture,{maxValues=1_000_000}={}) {
  const {model,source}=sourceContract(modelIr);
  exactKeys(capture,["schema","source","run","inputs","requested_value_refs","captures","missing"],"activation capture");
  requireCondition(capture.schema==="deepbom.activation_capture.v1","unsupported activation capture schema"); bindSource(capture.source,model);validateRun(capture.run,source,model);
  requireCondition(Array.isArray(capture.inputs)&&Array.isArray(capture.captures)&&Array.isArray(capture.missing)&&Array.isArray(capture.requested_value_refs),"invalid capture arrays");
  let count=0;
  const convert=(row,input,index)=>{
    exactKeys(row,["value_ref","native_locator","dtype","shape","values"],"captured tensor");validateTensorBinding(row,model,input,capture.run.entry_region_ref);checkValues(row.values,row.dtype);
    requireCondition(shapeCount(row.shape)===BigInt(row.values.length),"captured tensor shape/value count mismatch");
    count+=row.values.length;requireCondition(Number.isSafeInteger(maxValues)&&maxValues>=0&&count<=maxValues,"activation value budget exceeded");
    return {id:`${input?"input":"activation"}:${index}`,value_ref:row.value_ref,native_locator:row.native_locator,dtype:row.dtype,shape:row.shape,values_sha256:digest({dtype:row.dtype,shape:row.shape,values:row.values}),statistics:statisticsOf(row.values)};
  };
  const inputs=capture.inputs.map((r,i)=>convert(r,true,i)), tensors=capture.captures.map((r,i)=>convert(r,false,i));
  const body={schema:ACTIVATION_IR_SCHEMA,method_version:METHOD_VERSION,source,capture_sha256:digest(capture),run:structuredClone(capture.run),inputs,requested_value_refs:[...capture.requested_value_refs],tensors,missing:structuredClone(capture.missing),coverage:coverage(capture.requested_value_refs,tensors,capture.missing,model,capture.run.entry_region_ref),interpretation_boundary:BOUNDARY};
  return validateActivationIr(seal(body,"activation_ir_sha256"),model);
}
export function validateActivationIr(document,model) {
  exactKeys(document,["schema","method_version","source","capture_sha256","run","inputs","requested_value_refs","tensors","missing","coverage","interpretation_boundary","activation_ir_sha256"],"Activation IR");
  requireCondition(document.schema===ACTIVATION_IR_SCHEMA&&document.method_version===METHOD_VERSION&&document.interpretation_boundary===BOUNDARY,"unsupported Activation IR contract");bindSource(document.source,model);checkDigest(document,"activation_ir_sha256");validateRun(document.run,document.source,model);requireCondition(SHA256.test(document.capture_sha256),"invalid capture digest");
  const requested=document.requested_value_refs,eligible=new Set(eligibleValues(model,document.run.entry_region_ref));
  requireCondition(Array.isArray(requested)&&new Set(requested).size===requested.length&&requested.every(id=>eligible.has(id)),"requested capture values are unknown, static, input or duplicated");
  const accounted=new Set(),inputRefs=new Set();
  for(const [kind,rows] of [["input",document.inputs],["activation",document.tensors]]) {
    requireCondition(Array.isArray(rows),"invalid activation tensors");
    for(const [index,row] of rows.entries()) {
      exactKeys(row,["id","value_ref","native_locator","dtype","shape","values_sha256","statistics"],"activation tensor");
      requireCondition(row.id===`${kind}:${index}`&&SHA256.test(row.values_sha256),"invalid activation identity");validateTensorBinding(row,model,kind==="input",document.run.entry_region_ref);validateStatistics(row.statistics);requireCondition(shapeCount(row.shape)===BigInt(row.statistics.value_count),"activation count mismatch");
      if(kind==="input") { const inputKey=row.value_ref??`unmapped:${row.native_locator}`;requireCondition(!inputRefs.has(inputKey),"duplicate input");inputRefs.add(inputKey); }
      else if(row.value_ref!==null) { requireCondition(requested.includes(row.value_ref)&&!accounted.has(row.value_ref),"unrequested or duplicate capture");accounted.add(row.value_ref); }
    }
  }
  bindSymbols([...document.inputs,...document.tensors],model);
  if(document.run.provenance) validateRuntimeProvenance(document.run.provenance,document.inputs,document.run);
  const required=model.program.values.filter(v=>v.roles.includes("graph_input")&&v.region_ref===document.run.entry_region_ref&&!v.storage_refs.length).map(v=>v.id);
  requireCondition(required.every(id=>inputRefs.has(id)),"run is missing graph input evidence");
  requireCondition(Array.isArray(document.missing),"invalid missing capture ledger");
  for(const row of document.missing) { exactKeys(row,["value_ref","reason"],"missing capture");nonempty(row.reason,"missing capture reason");requireCondition(requested.includes(row.value_ref)&&!accounted.has(row.value_ref),"duplicate or unrequested missing value");accounted.add(row.value_ref); }
  requireCondition(accounted.size===requested.length,"requested value silently omitted");requireCondition(canonicalJson(document.coverage)===canonicalJson(coverage(requested,document.tensors,document.missing,model,document.run.entry_region_ref)),"activation coverage mismatch");return document;
}
export function buildNumericalEvidenceBundle(model,weightIr=null,activationIr=null) {
  const {source}=sourceContract(model);if(weightIr) validateWeightIr(weightIr,model);if(activationIr)validateActivationIr(activationIr,model);
  return seal({schema:"deepbom.numerical_evidence_bundle.v1",source,weight_ir_sha256:weightIr?.weight_ir_sha256||null,activation_ir_sha256:activationIr?.activation_ir_sha256||null},"bundle_sha256");
}
