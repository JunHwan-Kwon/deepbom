import { portableContract } from "./portable-contracts.js";
import { canonicalJson } from "./report-utils.js";
import { canonicalEvidenceDigest } from "./evidence-identity.js";
import { validateWeightIr } from "./weight-ir.js";
import { METHOD_VERSION, seal, sourceContract } from "./numerical-ir/common.js";
import { statisticsOf } from "./numerical-ir/statistics.js";
import { ACTIVATION_IR_SCHEMA } from "./evidence-ir.js";
export { ACTIVATION_IR_SCHEMA };

export function buildActivationIr(modelIr,capture,{maxValues=1_000_000}={}) {
  const {model,source}=sourceContract(modelIr);
  canonicalJson(capture);
  portableContract("semantic_validation", {operation:"activation_capture",value:capture,model,max_values:maxValues});
  const convert=(row,input,index)=>({id:`${input?"input":"activation"}:${index}`,value_ref:row.value_ref,native_locator:row.native_locator,dtype:row.dtype,shape:row.shape,values_sha256:canonicalEvidenceDigest({dtype:row.dtype,shape:row.shape,values:row.values}),statistics:statisticsOf(row.values)});
  const inputs=capture.inputs.map((r,i)=>convert(r,true,i)), tensors=capture.captures.map((r,i)=>convert(r,false,i));
  const coverage=portableContract("semantic_validation",{operation:"activation_coverage",requested:capture.requested_value_refs,tensors,missing:capture.missing,model,entry:capture.run.entry_region_ref});
  const body={schema:ACTIVATION_IR_SCHEMA,method_version:METHOD_VERSION,source,capture_sha256:canonicalEvidenceDigest(capture),run:structuredClone(capture.run),inputs,requested_value_refs:[...capture.requested_value_refs],tensors,missing:structuredClone(capture.missing),coverage,interpretation_boundary:portableContract("semantic_validation",{operation:"activation_boundary"})};
  return validateActivationIr(seal(body,"activation_ir_sha256"),model);
}
export function validateActivationIr(document,model) {
  const validated=sourceContract(model).model;
  canonicalJson(document);
  portableContract("semantic_validation",{operation:"activation",value:document,model:validated});
  return document;
}
export function buildNumericalEvidenceBundle(model,weightIr=null,activationIr=null) {
  const {source}=sourceContract(model);if(weightIr) validateWeightIr(weightIr,model);if(activationIr)validateActivationIr(activationIr,model);
  return seal({schema:"deepbom.numerical_evidence_bundle.v1",source,weight_ir_sha256:weightIr?.weight_ir_sha256||null,activation_ir_sha256:activationIr?.activation_ir_sha256||null},"bundle_sha256");
}
