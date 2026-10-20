import { decodeNativeTensor } from "./native-evidence.js";
import { visitDenseTensorValues } from "./tensor-numerical-integrity.js";
import { numericalWasm, numericalWasmIdentity, initializeNumericalWasm } from "./numerical-wasm.js";
import { ExactMoments } from "./numerical-ir/exact-moments.js";
import { exactKeys, requireCondition as need } from "./numerical-ir/common.js";
import { canonicalEvidenceDigest, sealDocument } from "./evidence-identity.js";
import { validateMetricConfiguration } from "./metric-evidence.js";

export async function evaluateBuiltinMetric(input) {
  exactKeys(input,["metric","samples","options","missing_policy","shape","prediction_dtype","target_dtype"],"metric input");
  await initializeNumericalWasm();
  const wasm=numericalWasm(),methods=JSON.parse(wasm.numerical_metric_methods());
  need(Object.hasOwn(methods,input.metric),"unsupported built-in metric");
  need(Array.isArray(input.samples)&&input.samples.length>0&&input.samples.length<=10000,"metric sample budget exceeded");
  const ids=new Set();for(const sample of input.samples){need(typeof sample.sample_id==="string"&&sample.sample_id.length>0&&sample.sample_id.length<=256&&!ids.has(sample.sample_id),"invalid/duplicate sample identity");ids.add(sample.sample_id);}
  const configuration={schema:"deepbom.metric_configuration.v2",adapter_method:"2.0.0",
    producer:{name:"deepbom-wasm",version:"1",...numericalWasmIdentity(),method:methods[input.metric]},
    metric:input.metric,options:input.options,aggregation:"equal_weight_sample_mean",analysis_unit:"sample",
    missing_policy:input.missing_policy,mask_policy:"none",device:"cpu",prediction_dtype:input.prediction_dtype,target_dtype:input.target_dtype,shape:input.shape};
  validateMetricConfiguration(sealDocument(configuration,"configuration_sha256"));
  need(input.shape[0]===input.samples.length,"sample count differs from shape");
  const samples=[],missing=[],input_refs=[];const mean=new ExactMoments({squares:false});
  try {
    for(const sample of input.samples) {
      exactKeys(sample,["sample_id","prediction","target"],"metric sample");
      const vectors=[];
      for(const row of [sample.prediction,sample.target]) {
        const {raw,count}=decodeNativeTensor(row);
        need(count<=8_388_608n&&JSON.stringify(row.shape)===JSON.stringify([1,...input.shape.slice(1)]),"metric sample shape/budget mismatch");
        need(input.metric==="ssim"? /^F(16|32|64)$/.test(row.dtype):/^([IU](8|16|32|64)|BOOL)$/.test(row.dtype),"metric tensor dtype is incompatible with its input contract");
        const values=new Float64Array(Number(count));let i=0;
        const decoded=await visitDenseTensorValues(raw,row,value=>{need(typeof value!=="bigint"||value>=-9007199254740991n&&value<=9007199254740991n,"unsafe integer metric input");values[i++]=Number(value);},{maxValues:8_388_608});
        need(decoded.status==="assessed"&&i===values.length,"metric tensor decoding incomplete");vectors.push(values);
      }
      input_refs.push({sample_id:sample.sample_id,prediction_sha256:canonicalEvidenceDigest(sample.prediction),target_sha256:canonicalEvidenceDigest(sample.target),digest_scope:"canonical_native_tensor_transport_row"});
      if(vectors.some(v=>v.some(x=>!Number.isFinite(x)))) {
        need(input.missing_policy==="omit_sample","Metric inputs contain nonfinite values");missing.push({sample_id:sample.sample_id,reason:"nonfinite_input"});continue;
      }
      const value=wasm.numerical_quality_metric(input.metric,vectors[0],vectors[1],Uint32Array.from(input.shape.slice(1)),JSON.stringify(input.options));
      if(!Number.isFinite(value)){need(input.missing_policy==="omit_sample","Metric is undefined for an included sample");missing.push({sample_id:sample.sample_id,reason:"undefined_metric"});}
      else {samples.push({sample_id:sample.sample_id,value});mean.add(value);}
    }
    return sealDocument({schema:"deepbom.metric_observation.v1",configuration,configuration_sha256:canonicalEvidenceDigest(configuration),input_refs,
      status:samples.length?"assessed":"not_assessed",reason:samples.length?null:"no_assessed_samples",value:mean.mean(samples.length),
      denominator:String(samples.length),missing_count:String(missing.length),samples,missing,
      boundary:"Task measurement on explicitly supplied predictions and targets; no patient-level aggregation, clinical verdict, confidence interval or model execution is inferred."},"observation_sha256");
  } finally {mean.free();}
}
