import { buildWeightIr, validateWeightIr } from "./weight-ir.js";
import { numericSources, hashSource } from "./numerical-ir/weight-sources.js";
import { requireCondition, sourceContract, seal, checkDigest, exactKeys, parseNumeric } from "./numerical-ir/common.js";
import { axisContract, quantizationContracts, tensorQuantization, dequantize } from "./numerical-ir/weight-contracts.js";
import { assessed, notAssessed, matrixView, channelStatistics, similarityAnalysis, spectrumAnalysis, sparsityAnalysis, tileProjection, kernelProjection, compareValues } from "./numerical-ir/weight-math.js";
import { statisticsOf, validateStatistics } from "./numerical-ir/statistics.js";
import { canonicalJson } from "./report-utils.js";

export const WEIGHT_ANALYSIS_SCHEMA="deepbom.weight_analysis.v1";
export const WEIGHT_COMPARISON_SCHEMA="deepbom.weight_comparison.v1";
export const WEIGHT_ANALYSIS_DEFAULTS=Object.freeze({max_values:8_000_000,max_tensor_values:2_000_000,max_channels:4096,max_pairwise_channels:128,max_svd_work:30_000_000,tensor_ids:[],axes:{}});
const BOUNDARY="Complete tensor computations within explicit budgets; no sampled statistics. Floating calculations use rounded binary64. Channel slices follow declared axes; generic slices are not identified as training layers. SVD describes the specified matrix unfolding. Similarity is not functional equivalence. A 2:4 pattern is not backend eligibility or speedup. Endpoint occupancy is not proven clipping. Accuracy and runtime effects require execution evidence.";
export function weightAnalysisOptions(input={}) {
  requireCondition(input&&typeof input==="object"&&!Array.isArray(input),"weight options must be an object");
  requireCondition(Object.keys(input).every(k=>Object.hasOwn(WEIGHT_ANALYSIS_DEFAULTS,k)),"unknown weight analysis option");
  const value={...WEIGHT_ANALYSIS_DEFAULTS,...input};
  for(const [key,limit] of Object.entries({max_values:100_000_000,max_tensor_values:10_000_000,max_channels:16384,max_pairwise_channels:512,max_svd_work:200_000_000}))requireCondition(Number.isSafeInteger(value[key])&&value[key]>=0&&value[key]<=limit,"invalid "+key);
  requireCondition(Array.isArray(value.tensor_ids)&&value.tensor_ids.every(x=>typeof x==="string")&&new Set(value.tensor_ids).size===value.tensor_ids.length,"invalid tensor selection");
  requireCondition(value.axes&&typeof value.axes==="object"&&!Array.isArray(value.axes),"invalid axis options");
  for(const row of Object.values(value.axes))requireCondition(row&&typeof row==="object"&&Object.keys(row).every(k=>["channel_axis","sparsity_axis","kernel_channel"].includes(k))&&Object.values(row).every(x=>Number.isSafeInteger(x)&&x>=0),"invalid tensor axis");
  return structuredClone(value);
}
function coverage(rows) {
  return {inventory_count:rows.length,assessed_count:rows.filter(r=>r.status==="assessed").length,not_assessed_count:rows.filter(r=>r.status!=="assessed").length,decoded_value_count:String(rows.reduce((n,r)=>n+BigInt(r.value_count||"0"),0n)),feature_counts:Object.fromEntries(["channels","similarity","spectrum","sparsity","quantization","projection","kernel"].map(k=>[k,rows.filter(r=>r[k]?.status==="assessed").length]))};
}
function unavailable(row,reason) {return {weight_ref:row.id,storage_ref:row.storage_ref,name:row.name,shape:row.shape,dtype:row.dtype,status:"not_assessed",reason,value_count:"0",representation:row.representation,axes:null,statistics:null,channels:notAssessed(reason),similarity:notAssessed(reason),spectrum:notAssessed(reason),sparsity:notAssessed(reason),quantization:notAssessed(reason),projection:notAssessed(reason),kernel:notAssessed(reason)};}

// decoded is process-local working data and is never included in exported IR.
export async function collectWeightAnalysis(modelInput,analysis,source,{weightIr=null,options={},signal,onProgress}={}) {
  const {model,source:identity}=sourceContract(modelInput),settings=weightAnalysisOptions(options);
  const weight=weightIr?validateWeightIr(weightIr,model):await buildWeightIr(model,analysis,source,{signal,onProgress});
  requireCondition(await hashSource(source,{signal})===identity.artifact_sha256,"advanced weight source digest mismatch");
  const ids=new Set(weight.tensors.map(r=>r.id));for(const id of [...settings.tensor_ids,...Object.keys(settings.axes)])requireCondition(ids.has(id),"unknown weight reference "+id);
  const candidates=await numericSources(source,analysis,model,{captureValues:true,maxCapturedValues:settings.max_values}),contracts=await quantizationContracts(source,analysis);
  const stores=new Map(model.tensors_and_storage.storage_objects.map(s=>[s.id,s])),decoded=new Map(),rows=[];
  let remaining=settings.max_values, remainingChannels=settings.max_channels, remainingSvd=settings.max_svd_work, remainingSimilarityCells=1_000_000;
  for(const row of weight.tensors) {
    signal?.throwIfAborted();
    const n=Number(row.statistics?.value_count),candidate=candidates.find(c=>c.locator===row.native_locator&&c.storage_ref===row.storage_ref);
    let reason=row.status!=="assessed"?row.reason:settings.tensor_ids.length&&!settings.tensor_ids.includes(row.id)?"tensor_not_selected":!n?"empty_tensor":n>settings.max_tensor_values?"tensor_value_budget_exceeded":n>remaining?"total_value_budget_exceeded":row.statistics.unsafe_integer_count!=="0"?"unsafe_integer_arithmetic":!candidate?"decoder_identity_unavailable":row.representation==="complex_magnitude"?"complex_phase_not_preserved":null;
    if(reason){rows.push(unavailable(row,reason));continue;}
    const values=new Float64Array(n);let count=0,invalid=false;
    const result=await candidate.visit(raw=>{requireCondition(count<n,"advanced decoder value count overflow");const x=parseNumeric(raw);values[count++]=Number(x);if(!Number.isFinite(Number(x))||typeof x==="bigint"&&!Number.isSafeInteger(Number(x)))invalid=true;},{maxValues:n,signal});
    if(result.status!=="assessed"||count!==n||invalid){rows.push(unavailable(row,result.reason|| (invalid?"nonfinite_or_unsafe_values":"complete_numeric_values_unavailable")));continue;}
    const axes=axisContract(row,model,settings.axes[row.id]);
    for(const key of ["channel_axis","sparsity_axis"])requireCondition(axes[key]<Math.max(1,row.shape.length),"axis outside tensor rank for "+row.id);
    const quant=dequantize(values,row,tensorQuantization(row,stores.get(row.storage_ref),analysis,contracts),axes.storage_order);
    const matrix=matrixView(quant.values,row.shape,axes.channel_axis,axes.storage_order);
    remaining-=n;
    const channels=matrix.rows<=remainingChannels?channelStatistics(matrix):notAssessed("channel_inventory_exceeds_budget");
    if(channels.status==="assessed")remainingChannels-=matrix.rows;
    const similarity=matrix.rows*matrix.rows<=remainingSimilarityCells?similarityAnalysis(matrix,settings.max_pairwise_channels,0.95):notAssessed("total_similarity_matrix_budget_exceeded");
    if(similarity.status==="assessed")remainingSimilarityCells-=matrix.rows*matrix.rows;
    const spectrum=spectrumAnalysis(matrix,remainingSvd);
    if(spectrum.status==="assessed")remainingSvd-=spectrum.work_count;
    else if(spectrum.reason==="svd_iteration_budget_exceeded")remainingSvd=0;
    const output={weight_ref:row.id,storage_ref:row.storage_ref,name:row.name,shape:row.shape,dtype:row.dtype,status:"assessed",reason:null,value_count:String(n),representation:quant.representation,axes,statistics:statisticsOf(quant.values),channels,similarity,spectrum,sparsity:sparsityAnalysis(quant.values,row.shape,axes.storage_order,axes.sparsity_axis),quantization:quant.report,projection:tileProjection(matrix),kernel:kernelProjection(matrix,row.shape,axes.channel_axis,axes.storage_order,settings.axes[row.id]?.kernel_channel || 0)};
    rows.push(output);decoded.set(row.id,{values:quant.values,matrix,row:output});
    onProgress?.({phase:"weight_analysis",completed:rows.length,total:weight.tensors.length});
  }
  const document=seal({schema:WEIGHT_ANALYSIS_SCHEMA,method_version:"1.0.0",source:identity,weight_ir_sha256:weight.weight_ir_sha256,options:settings,coverage:coverage(rows),tensors:rows,interpretation_boundary:BOUNDARY},"weight_analysis_sha256");
  validateWeightAnalysis(document,weight,model);
  return {weight_ir:weight,weight_analysis:document,decoded};
}
export async function buildWeightEvidence(model,analysis,source,options={}) {
  const {weight_ir,weight_analysis}=await collectWeightAnalysis(model,analysis,source,options);
  return {weight_ir,weight_analysis};
}
export function validateWeightAnalysis(doc,weight,model) {
  validateWeightIr(weight,model);
  exactKeys(doc,["schema","method_version","source","weight_ir_sha256","options","coverage","tensors","interpretation_boundary","weight_analysis_sha256"],"weight analysis");
  requireCondition(doc.schema===WEIGHT_ANALYSIS_SCHEMA&&doc.method_version==="1.0.0"&&doc.interpretation_boundary===BOUNDARY,"unsupported weight analysis");checkDigest(doc,"weight_analysis_sha256");
  requireCondition(doc.weight_ir_sha256===weight.weight_ir_sha256&&canonicalJson(doc.source)===canonicalJson(weight.source),"advanced analysis source mismatch");weightAnalysisOptions(doc.options);
  requireCondition(doc.tensors.length===weight.tensors.length,"advanced inventory omitted a tensor");
  for(const [i,row] of doc.tensors.entries()) {
    exactKeys(row,["weight_ref","storage_ref","name","shape","dtype","status","reason","value_count","representation","axes","statistics","channels","similarity","spectrum","sparsity","quantization","projection","kernel"],"advanced tensor");
    const original=weight.tensors[i];requireCondition(row.dtype===original.dtype&&row.weight_ref===original.id&&row.storage_ref===original.storage_ref&&row.name===original.name&&canonicalJson(row.shape)===canonicalJson(original.shape),"advanced tensor binding mismatch");
    requireCondition(["assessed","not_assessed"].includes(row.status),"invalid advanced status");
    for(const key of ["channels","similarity","spectrum","sparsity","quantization","projection","kernel"]) {
      const feature=row[key];requireCondition(feature&&["assessed","not_assessed"].includes(feature.status),"invalid feature status");
      if(feature.status==="not_assessed"){exactKeys(feature,["status","reason"],"unassessed feature");requireCondition(typeof feature.reason==="string"&&feature.reason.length>0,"missing feature reason");}
      if(row.status==="not_assessed")requireCondition(feature.status==="not_assessed","skipped tensor has assessed feature");
    }
    if(row.status==="not_assessed"){requireCondition(typeof row.reason==="string"&&row.reason.length>0&&row.statistics===null&&row.value_count==="0","invalid skipped tensor");continue;}
    requireCondition(original.status==="assessed"&&row.axes&&["first_axis_fastest","last_axis_fastest"].includes(row.axes.storage_order),"invalid assessed tensor axes");
    for(const key of ["channel_axis","sparsity_axis"])requireCondition(Number.isSafeInteger(row.axes[key])&&row.axes[key]>=0&&row.axes[key]<Math.max(1,row.shape.length),"invalid assessed tensor axis");
    requireCondition(row.value_count===original.statistics.value_count&&row.reason===null,"advanced value count mismatch");validateStatistics(row.statistics);
    requireCondition(row.statistics.value_count===row.value_count,"advanced statistics count mismatch");
    if(row.channels.status==="assessed") {
      requireCondition(row.channels.count===row.channels.channels.length&&row.channels.count*row.channels.values_per_channel===Number(row.value_count),"channel count conservation failed");
      for(const [channelIndex,c] of row.channels.channels.entries()){validateStatistics(c.statistics);requireCondition(c.index===channelIndex&&Number(c.statistics.value_count)===row.channels.values_per_channel,"channel identity/count mismatch");}
    }
    if(row.similarity.status==="assessed") {const s=row.similarity;requireCondition(s.values.length===s.size*s.size&&s.values.every(v=>v===null||Number.isFinite(v)&&v>=-1&&v<=1),"invalid similarity matrix");}
    if(row.spectrum.status==="assessed") {const s=row.spectrum;requireCondition(s.singular_values.length===Math.min(s.rows,s.columns)&&s.normalized_singular_values.every(v=>Number.isFinite(v)&&v>=0)&&s.numerical_rank>=0&&s.numerical_rank<=s.singular_values.length&&s.energy_relative_residual<=1e-10,"invalid SVD result");}
    const sp=row.sparsity,b=sp.block,g=sp.structured_2_4,n=Number(row.value_count);
    requireCondition(sp.status==="assessed"&&Number.isSafeInteger(b.size)&&b.size>0&&b.complete_blocks*b.size+b.trailing_values===n&&b.all_zero_blocks<=b.complete_blocks&&g.groups*4+g.trailing_values===n&&g.compliant_groups<=g.groups,"sparsity group conservation failed");
    requireCondition(g.pattern_satisfied===(g.groups>0&&g.trailing_values===0?g.compliant_groups===g.groups:null),"2:4 conclusion mismatch");
    requireCondition(row.sparsity.value_count===row.value_count&&row.sparsity.zero_count===row.statistics.zero_count,"sparsity count mismatch");
    const validateProjection=(projection,count)=>{requireCondition(projection.status==="assessed"&&projection.cells.length===projection.rows*projection.columns&&projection.source_rows*projection.source_columns===count,"projection shape mismatch");requireCondition(projection.cells.every(c=>Number.isSafeInteger(c.count)&&c.count>0&&Number.isSafeInteger(c.zero_count)&&c.zero_count>=0&&c.zero_count<=c.count&&Number.isFinite(c.mean)&&c.minimum<=c.maximum)&&projection.cells.reduce((n,c)=>n+c.count,0)===count,"projection count conservation failed");};
    validateProjection(row.projection,n);
    if(row.kernel.status==="assessed") {requireCondition(row.kernel.channel_count===(row.shape[row.axes.channel_axis]||1)&&row.kernel.channel_index>=0&&row.kernel.channel_index<row.kernel.channel_count,"kernel binding mismatch");validateProjection(row.kernel.image,n/row.kernel.channel_count);}
    if(row.quantization.status==="assessed") {const q=row.quantization;requireCondition(row.representation==="dequantized_real"&&q.scales.length>0&&q.scales.every(x=>Number.isFinite(x)&&x>0)&&q.zero_points.every(x=>Number.isSafeInteger(x)&&x>=q.encoding_minimum&&x<=q.encoding_maximum)&&q.range_utilization>=0&&q.range_utilization<=1&&q.lower_endpoint_count+q.upper_endpoint_count<=n,"invalid affine quantization report");}
  }
  requireCondition(doc.tensors.reduce((sum,r)=>sum+(r.channels.status==="assessed"?r.channels.count:0),0)<=doc.options.max_channels,"channel budget mismatch");
  requireCondition(doc.tensors.reduce((sum,r)=>sum+(r.spectrum.status==="assessed"?r.spectrum.work_count:0),0)<=doc.options.max_svd_work,"SVD budget mismatch");
  requireCondition(canonicalJson(doc.coverage)===canonicalJson(coverage(doc.tensors))&&Number(doc.coverage.decoded_value_count)<=doc.options.max_values,"advanced coverage mismatch");return doc;
}

function canonicalValues(record, permutation=null) {
  const shape=record.row.shape, rank=shape.length, dims=permutation?permutation.map(i=>shape[i]):shape;
  const axes=permutation || shape.map((_,i)=>i),n=record.values.length,values=new Float64Array(n);
  const native=shape.map((_,i)=>(record.row.axes.storage_order==="first_axis_fastest"?shape.slice(0,i):shape.slice(i+1)).reduce((a,b)=>a*b,1));
  for(let i=0;i<n;i++){let rest=i,offset=0;for(let d=rank-1;d>=0;d--){const c=rest%dims[d];rest=Math.floor(rest/dims[d]);offset+=c*native[axes[d]];}values[i]=record.values[offset];}
  return {values,shape:dims};
}
export function compareWeightAnalyses(baseline,candidate,{mapping=[]}={}) {
  requireCondition(Array.isArray(mapping),"weight mapping must be an array");
  const baseRows=baseline.weight_ir.tensors,candidateRows=candidate.weight_ir.tensors,explicit=new Map(),used=new Set(),pairs=[];
  for(const entry of mapping) {
    requireCondition(entry&&Object.keys(entry).every(k=>["baseline","candidate","candidate_axis_permutation"].includes(k))&&baseRows.some(r=>r.id===entry.baseline)&&candidateRows.some(r=>r.id===entry.candidate)&&!explicit.has(entry.candidate)&&!used.has(entry.baseline),"invalid or duplicate comparison mapping");explicit.set(entry.candidate,entry);used.add(entry.baseline);
  }
  used.clear();
  for(const row of candidateRows) {
    let mappingRow=explicit.get(row.id),base=null,basis="explicit_mapping";
    if(mappingRow)base=baseRows.find(r=>r.id===mappingRow.baseline);
    else {const matches=baseRows.filter(r=>r.name&&r.name===row.name&&canonicalJson(r.shape)===canonicalJson(row.shape));const unique=candidateRows.filter(r=>r.name===row.name&&canonicalJson(r.shape)===canonicalJson(row.shape)).length===1;if(matches.length===1&&unique&&![...explicit.values()].some(e=>e.baseline===matches[0].id))base=matches[0];basis="unique_name_and_shape_not_semantic_attestation";}
    const record={candidate_ref:row.id,baseline_ref:base?.id??null,name:row.name,matching_basis:basis,metrics:null,delta:null,spectrum_change:null};
    if(!base||used.has(base.id)){pairs.push({...record,status:"not_assessed",reason:base?"baseline_already_mapped":"no_unambiguous_tensor_match"});continue;}
    used.add(base.id);
    const left=baseline.decoded.get(base.id),right=candidate.decoded.get(row.id);
    if(!left||!right){pairs.push({...record,status:"not_assessed",reason:"complete_comparable_values_unavailable"});continue;}
    const permutation=mappingRow?.candidate_axis_permutation;
    if(permutation)requireCondition(Array.isArray(permutation)&&permutation.length===row.shape.length&&new Set(permutation).size===permutation.length&&permutation.every(i=>Number.isInteger(i)&&i>=0&&i<row.shape.length),"invalid axis permutation");
    const a=canonicalValues(left),b=canonicalValues(right,permutation);
    if(canonicalJson(a.shape)!==canonicalJson(b.shape)){pairs.push({...record,status:"not_assessed",reason:"aligned_shape_mismatch"});continue;}
    const compatible=left.row.representation===right.row.representation || [left,right].every(r=>r.row.representation==="dequantized_real"||/^(FLOAT|F\d|BF)/.test(r.row.dtype));
    if(!compatible || left.row.representation==="stored_scalar"&&right.row.representation==="stored_scalar"&&left.row.dtype!==right.row.dtype&&![/^(FLOAT|F\d|BF)/.test(left.row.dtype),/^(FLOAT|F\d|BF)/.test(right.row.dtype)].every(Boolean)){pairs.push({...record,status:"not_assessed",reason:"incompatible_numeric_representations"});continue;}
    const compared=compareValues(a.values,b.values),axis=left.row.axes.channel_axis;
    const finiteDelta=compared.differences.every(Number.isFinite);
    const spectrumA=left.row.spectrum,spectrumB=right.row.spectrum;
    const alignedAxis=permutation?permutation.indexOf(right.row.axes.channel_axis):right.row.axes.channel_axis;
    pairs.push({...record,status:"assessed",reason:null,candidate_axis_permutation:permutation||null,value_basis:{baseline:left.row.representation,candidate:right.row.representation,baseline_dtype:left.row.dtype,candidate_dtype:right.row.dtype},metrics:compared.metrics,distributions:{baseline:left.row.statistics,candidate:right.row.statistics},delta:finiteDelta?tileProjection(matrixView(compared.differences,a.shape,axis,"last_axis_fastest")):notAssessed("difference_overflow"),spectrum_change:spectrumA.status==="assessed"&&spectrumB.status==="assessed"&&axis===alignedAxis?assessed({baseline_rank:spectrumA.numerical_rank,candidate_rank:spectrumB.numerical_rank,baseline_singular_values:spectrumA.singular_values,candidate_singular_values:spectrumB.singular_values}):notAssessed("compatible_spectra_unavailable")});
  }
  return seal({schema:WEIGHT_COMPARISON_SCHEMA,method_version:"1.0.0",baseline:{source:baseline.weight_ir.source,weight_ir_sha256:baseline.weight_ir.weight_ir_sha256,weight_analysis_sha256:baseline.weight_analysis.weight_analysis_sha256},candidate:{source:candidate.weight_ir.source,weight_ir_sha256:candidate.weight_ir.weight_ir_sha256,weight_analysis_sha256:candidate.weight_analysis.weight_analysis_sha256},mapping,coverage:{candidate_count:candidateRows.length,compared_count:pairs.filter(p=>p.status==="assessed").length,not_assessed_count:pairs.filter(p=>p.status!=="assessed").length,unmatched_baseline_refs:baseRows.filter(r=>!used.has(r.id)).map(r=>r.id)},tensors:pairs,interpretation_boundary:"Matched values and explicit permutations only; name matching does not establish semantic equivalence. Numerical differences and low-rank structure do not establish task accuracy or measured speed. Raw flat arrays are omitted; exact-cell projections can reveal individual values."},"weight_comparison_sha256");
}
