// One projection for official HTML/PDF/JSON reports. No framework execution.
import { compareNativeEvidence, sealNative, nativeStateTotals } from "./native-evidence.js";
import { canonicalJson } from "./report-utils.js";
import { estimateOnnxMacs } from "../onnx.js";
import { scalarDtypeBytes, shapeElementCount } from "./tensor-size.js";

const need = (ok, reason) => { if (!ok) throw new Error(`Optimization report: ${reason}`); };
const missing = (reason) => ({ status: "not_assessed", value_decimal: null, reason });
const RULES = Object.freeze({
  fold_conv_bn: {
    title: "Fold BatchNorm into convolution",
    reason: "The applied rule requires a direct, unique Conv-BatchNorm path and fixed evaluation statistics. The channel-wise affine transform can be absorbed into the convolution weights and bias.",
    change: "Transform convolution state; replace BatchNorm with Identity. Unaffected state is retained.",
    tradeoff: "Evaluation preconditions apply. Training-mode BatchNorm behavior is not preserved. The runtime may already perform this fusion; no latency improvement is established.",
  },
  inverted: {
    title: "Replace spatial convolution with an inverted block",
    reason: "The applied rule requires a supported dense convolution. Pointwise expansion, channel-wise filtering and pointwise projection separate channel mixing from spatial computation. Automatic discovery selects spatial kernels; explicit rules may select pointwise kernels too.",
    change: "Insert expand Conv, ReLU, depthwise Conv, ReLU and project Conv. Replacement state is newly initialized; unrelated state is retained. No residual connection or extra BatchNorm is added.",
    tradeoff: "The function changes and requires training and task evaluation. Fewer parameters or nominal MACs do not establish lower latency or peak memory. This rule is not a measured candidate ranking.",
  },
});
function logicalBytes(t) {
  if (!t || !Array.isArray(t.shape) || !t.shape.every(d => Number.isSafeInteger(d) && d >= 0)) return null;
  const width = scalarDtypeBytes(String(t.dtype).toUpperCase());
  const count = shapeElementCount(t.shape);
  return width && count ? String(BigInt(count.decimal) * BigInt(width)) : null;
}
function cost(node, call, storage, framework) {
  if(call.inputs.length !== 1 || call.outputs.length !== 1) return missing("Requires one tensor input and one tensor output.");
  const type = node.kind.split(".").at(-1), cfg = node.config;
  const trusted = framework === "pytorch" ? node.kind.startsWith("torch.nn.") : node.kind.startsWith("keras.");
  if (!trusted) return missing("Custom module arithmetic is not inferred from its class name.");
  let x = call.inputs[0].shape, y = call.outputs[0].shape, w, opType, attributes = {};
  const weights = node.state_refs.map(id => storage.get(id)).filter(Boolean);
  if (framework === "pytorch" && ["Conv1d", "Conv2d", "Conv3d", "Linear"].includes(type)) {
    const kernels = weights.filter(t => /(?:^|\.)weight$/.test(t.id));
    if (kernels.length !== 1) return missing("No unambiguous kernel binding.");
    w = kernels[0].shape;
    if (type === "Linear") {
      if (x.length === 2 && y.length === 2) { opType = "Gemm"; attributes = { transB: 1 }; }
      else { opType = "MatMul"; w = w?.length === 2 ? [w[1],w[0]] : null; }
    } else { opType = "Conv"; attributes = { group: cfg.groups }; }
  } else if (framework === "tensorflow" && ["Conv2D", "DepthwiseConv2D", "Dense"].includes(type)) {
    const kernels = weights.filter(t => /(?:^|\/)kernel$/.test(t.id));
    if (kernels.length !== 1) return missing("No unambiguous kernel binding.");
    const kernel = kernels[0].shape;
    if (type === "Dense") {
      w = kernel; opType = x.length === 2 && y.length === 2 ? "Gemm" : "MatMul";
    } else {
      if (cfg.data_format !== "channels_last" || x.length !== 4 || y.length !== 4 || kernel?.length !== 4)
        return missing("Requires qualified channels-last rank-4 convolution.");
      x = [x[0],x[3],x[1],x[2]]; y = [y[0],y[3],y[1],y[2]];
      w = type === "DepthwiseConv2D" ? [kernel[2]*kernel[3],1,kernel[0],kernel[1]] : [kernel[3],kernel[2],kernel[0],kernel[1]];
      if (!w.every(Number.isSafeInteger)) return missing("Kernel shape mapping exceeds the exact dimension range.");
      opType = "Conv"; attributes = {group: type === "DepthwiseConv2D" ? kernel[2] : cfg.groups};
    }
  } else {
    const nonContraction = new Set(["ReLU","ReLU6","GELU","LeakyReLU","LayerNorm","Embedding","Identity","BatchNorm1d","BatchNorm2d","BatchNorm3d","BatchNormalization","Flatten","Dropout","AdaptiveAvgPool2d","GlobalAveragePooling2D","MaxPool2d","AvgPool2d","InputLayer","Softmax","Sigmoid","Tanh"]);
    return nonContraction.has(type) ? {status:"not_applicable",value_decimal:"0",reason:"Excluded from the nominal tensor-contraction MAC metric; not zero execution cost."} : missing("No qualified native-to-common-cost mapping for this module.");
  }
  if (!w) return missing("No unambiguous kernel binding.");
  const tensor = (name,shape) => ({name,shape,dtype:"FLOAT32",shapeDeclared:true,valueKind:"tensor"});
  const result = estimateOnnxMacs({opType,domain:"",inputs:["x","w"],outputs:["y"],attributes:new Map(Object.entries(attributes).map(([k,v])=>[k,{i:v,ints:[],s:""}]))},
    new Map([tensor("x",x),tensor("w",w),tensor("y",y)].map(t=>[t.name,t])));
  return {status:result.status,value_decimal:result.value_decimal,reason:result.reason,mapping:opType};
}
function structure(bundle, probe) {
  const program = bundle.model_ir.program, storage = new Map(bundle.snapshot.tensors.map(t=>[t.id,t]));
  const nodes = new Map(program.nodes.map(n=>[n.id,n]));
  need(probe && ["observed","not_requested"].includes(probe.status) && Array.isArray(probe.invocations), "invalid shape observation");
  const validateTensors = (list) => {
    need(Array.isArray(list), "invalid observed tensor list");
    for(const t of list) need(t && Array.isArray(t.shape) && t.shape.every(d=>d===null || Number.isSafeInteger(d)&&d>=0) && typeof t.dtype==="string", "invalid observed tensor");
  };
  validateTensors(probe.inputs); validateTensors(probe.outputs);
  need(probe.status !== "not_requested" || !probe.invocations.length && !probe.inputs.length && !probe.outputs.length, "unrequested probe contains observations");
  const seen = new Set();
  const calls = probe.invocations.map(c=>{
    need(nodes.has(c.subject) && Number.isSafeInteger(c.invocation) && c.invocation>0 && !seen.has(`${c.subject}\0${c.invocation}`), "unbound or duplicate invocation");
    seen.add(`${c.subject}\0${c.invocation}`);
    validateTensors(c.inputs); validateTensors(c.outputs);
    return {...c, outputs:c.outputs.map(t=>({...t,logical_bytes:logicalBytes(t)})),macs:cost(nodes.get(c.subject),c,storage,program.framework.name)};
  });
  const assessed=calls.filter(c=>c.macs.status==="assessed"), unknown=calls.filter(c=>c.macs.status==="not_assessed");
  const sum=assessed.reduce((s,c)=>s+BigInt(c.macs.value_decimal),0n);
  const allNonContraction = calls.length > 0 && assessed.length === 0 && unknown.length === 0;
  return {snapshot_sha256:bundle.snapshot.snapshot_sha256,model_ir_sha256:bundle.model_ir.model_ir_sha256,
    framework:program.framework,scope:program.scope,limitations:program.limitations,inputs:probe.inputs,outputs:probe.outputs,
    nodes:program.nodes,relationships:program.relationships,storage:bundle.snapshot.tensors,
    shape_observation_status:probe.status,invocations:calls,
    cost:{status:assessed.length?"assessed_subtotal":allNonContraction?"not_applicable":"not_assessed",macs:assessed.length||allNonContraction?String(sum):null,
      assessed_invocations:assessed.length,unmapped_invocations:unknown.length,observed_invocations:calls.length,
      scope:"Nominal contraction MAC subtotal for mapped module invocations on the supplied probe path. Functional operations, unobserved paths, bias, normalization, activations and pooling are outside this subtotal. Not latency or total instructions."}};
}
export function buildNativeOptimizationReport(input) {
  const {baseline,candidate,transformations,context,validation}=input;
  const comparison=compareNativeEvidence(baseline,candidate);
  need(context?.version===1 && ["automatic_rules","explicit_rules","not_recorded"].includes(context.selection), "unsupported report context");
  need(Array.isArray(transformations), "invalid transformation ledger");
  const before=structure(baseline,context.probes.baseline),after=structure(candidate,context.probes.candidate);
  const idsA=new Set(before.nodes.map(n=>n.id)),idsB=new Set(after.nodes.map(n=>n.id));
  const changes=transformations.map((t,index)=>{
    need(t && typeof t.subject==="string" && (idsA.has(t.subject)||idsB.has(t.subject)) && t.request?.kind===t.rule, "unbound transformation");
    const meta=t.rule_version==="1.0.0"?RULES[t.rule]:null;
    const roots=[t.subject,...(typeof t.request.batchnorm==="string"?[t.request.batchnorm]:[])];
    const affected=new Set(roots);
    for(const rels of [before.relationships,after.relationships]) {
      for(let changed=true;changed;) {changed=false;for(const r of rels)if(r.kind==="contains"&&affected.has(r.from)&&!affected.has(r.to)){affected.add(r.to);changed=true;}}
    }
    return {...t,index:index+1,title:meta?.title||"Unregistered transformation description",
      reason:meta?.reason||"Reason not recorded for this rule version; no inferred optimization rationale.",
      description:meta?.change||null,tradeoff:meta?.tradeoff||null,
      reason_basis:"Applied rule semantics, not a measured candidate-selection decision.",
      affected_nodes:comparison.nodes.filter(n=>affected.has(n.id))};
  });
  const deltas=nativeReportDeltas(comparison.totals);
  return sealNative({schema:"deepbom.optimization_report.v1",template:"monochrome.v1",
    baseline:before,candidate:after,context,validation,changes,comparison,
    static_deltas:deltas,cost_delta:before.cost.macs!==null&&after.cost.macs!==null?String(BigInt(after.cost.macs)-BigInt(before.cost.macs)):null,
    boundary:"A deterministic report projection from bound native evidence. Applied-rule reasons are not a hardware optimization ranking. State payload is not peak memory. No measured speed, task quality, delegate or cache benefit is established."},"report_sha256");
}

export function nativeReportDeltas(totals) {
  return Object.fromEntries(Object.keys(totals.baseline).map(k=>[k,String(BigInt(totals.candidate[k])-BigInt(totals.baseline[k]))]));
}

// Imported reports carry observations, not executable models. Check every
// quantity this projection can recompute using the original common owners.
export function validateOptimizationReportProjection(report) {
  const same=(a,b)=>canonicalJson(a)===canonicalJson(b);
  for(const side of ['baseline','candidate']) {
    const s=report[side];
    need(same(nativeStateTotals(s.storage),report.comparison.totals[side]),'stored state totals disagree');
    const projected=structure({snapshot:{snapshot_sha256:s.snapshot_sha256,tensors:s.storage},model_ir:{model_ir_sha256:s.model_ir_sha256,program:s}},report.context.probes[side]);
    need(same(projected,s),'shape, cost or coverage projection disagrees with recorded probe');
  }
  need(same(nativeReportDeltas(report.comparison.totals),report.static_deltas),'static deltas disagree');
  const a=report.baseline.cost.macs,b=report.candidate.cost.macs;
  need(report.cost_delta===(a!==null&&b!==null?String(BigInt(b)-BigInt(a)):null),'MAC delta disagrees');
}
