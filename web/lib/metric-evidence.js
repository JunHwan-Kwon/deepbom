// Import contracts only: metric arithmetic belongs to the explicitly selected producer.
import {exactFields, assertDocumentDigest, requireEvidence as need} from './evidence-identity.js';
export function validateMetricConfiguration(d) {
  exactFields(d,['schema','adapter_method','producer','metric','options','aggregation','analysis_unit','missing_policy','mask_policy','device','prediction_dtype','target_dtype','shape','configuration_sha256'],[],'metric configuration');
  const builtin=d.schema==='deepbom.metric_configuration.v2'&&d.adapter_method==='2.0.0';
  need(builtin||d.schema==='deepbom.metric_configuration.v1'&&d.adapter_method==='1.0.0','unsupported metric configuration');
  assertDocumentDigest(d,'configuration_sha256');
  if(builtin){
    exactFields(d.producer,['name','version','abi','wasm_sha256','method'],[],'metric producer');
    need(d.producer.name==='deepbom-wasm'&&d.producer.version==='1'&&d.producer.abi==='deepbom.numerical_abi.v1:exact_binary64_v1'&&/^[a-f0-9]{64}$/.test(d.producer.wasm_sha256),'invalid built-in metric producer');
    need(d.producer.method===({ssim:'separable_reflect_binary64_v1',generalized_dice:'per_sample_weighted_counts_binary64_v1'})[d.metric],'unsupported metric method');
  }else{
    exactFields(d.producer,['name','version','torch_version'],[],'metric producer');
    need(d.producer.name==='torchmetrics'&&d.producer.version==='1.9.0'&&typeof d.producer.torch_version==='string','unqualified metric producer');
  }
  need(d.aggregation==='equal_weight_sample_mean'&&d.analysis_unit==='sample'&&d.mask_policy==='none'&&d.device==='cpu','unsupported metric aggregation');
  need(['reject','omit_sample'].includes(d.missing_policy)&&Array.isArray(d.shape)&&d.shape.length<=32&&d.shape.every(x=>Number.isSafeInteger(x)&&x>0),'invalid metric conditions');
  need(typeof d.prediction_dtype==='string'&&typeof d.target_dtype==='string','invalid metric dtypes');
  need(['ssim','generalized_dice'].includes(d.metric),'unsupported metric');
  need(d.options&&typeof d.options==='object'&&!Array.isArray(d.options),'invalid metric options');
  const o=d.options, positive=v=>typeof v==='number'&&Number.isFinite(v)&&v>0;
  if(d.metric==='ssim'){
    exactFields(o,['data_range','gaussian_kernel','sigma','kernel_size','k1','k2'],[],'SSIM options');
    need([4,5].includes(d.shape.length)&&['data_range','sigma','k1','k2'].every(k=>positive(o[k]))&&typeof o.gaussian_kernel==='boolean'&&Number.isSafeInteger(o.kernel_size)&&o.kernel_size>0&&o.kernel_size%2===1,'invalid SSIM conditions');
  }else{
    exactFields(o,['num_classes','include_background','weight_type','input_format','empty_class_policy'],[],'Dice options');
    need(d.shape.length>=3&&Number.isSafeInteger(o.num_classes)&&o.num_classes>1&&typeof o.include_background==='boolean'&&['square','simple','linear'].includes(o.weight_type)&&o.input_format==='index'&&o.empty_class_policy==='torchmetrics_1.9.0','invalid Generalized Dice conditions');
  }
  return d;
}
