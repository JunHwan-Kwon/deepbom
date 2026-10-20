import numpy as np,torch,json
from deepbom.metrics import evaluate

def independent_ssim(p,t,o):
    """Direct multidimensional convolution oracle, not the separable engine."""
    radius=int(3.5*o['sigma']+.5) if o['gaussian_kernel'] else o['kernel_size']//2
    axis=np.arange(-radius,radius+1,dtype=np.float64)
    kernel=np.exp(-axis**2/(2*o['sigma']**2)) if o['gaussian_kernel'] else np.ones(len(axis))
    kernel/=kernel.sum()
    spatial=p.ndim-1
    weights=kernel
    for _ in range(spatial-1):weights=np.multiply.outer(weights,kernel)
    pads=[radius]*spatial
    if spatial==3:pads[-1]=o['kernel_size']//2
    def convolve(a):
        padded=np.pad(a,[(0,0),*[(v,v) for v in pads]],mode='reflect')
        windows=np.lib.stride_tricks.sliding_window_view(padded,(len(kernel),)*spatial,axis=tuple(range(1,a.ndim)))
        return np.tensordot(windows,weights,axes=(tuple(range(windows.ndim-spatial,windows.ndim)),tuple(range(spatial))))
    mp,mt=convolve(p),convolve(t)
    vp,vt=np.maximum(convolve(p*p)-mp*mp,0),np.maximum(convolve(t*t)-mt*mt,0)
    cov=convolve(p*t)-mp*mt
    c1,c2=(o['k1']*o['data_range'])**2,(o['k2']*o['data_range'])**2
    return np.mean((2*mp*mt+c1)*(2*cov+c2)/((mp*mp+mt*mt+c1)*(vp+vt+c2)))

rng=np.random.default_rng(71)
count=0;maximum=0
for shape in [(2,1,14,15),(1,2,9,10,11)]:
  for gaussian in [True,False]:
    options=dict(data_range=1.,gaussian_kernel=gaussian,sigma=.8,kernel_size=5,k1=.01,k2=.03)
    p=torch.from_numpy(rng.random(shape));t=torch.from_numpy(rng.random(shape))
    a=evaluate(p,t,metric='ssim',sample_ids=[str(x) for x in range(shape[0])],options=options)
    independent=float(np.mean([independent_ssim(x.numpy(),y.numpy(),options) for x,y in zip(p,t)]))
    assert abs(a['value']-independent)<1e-12,(a['value'],independent)
    b=evaluate(p,t,metric='ssim',sample_ids=[str(x) for x in range(shape[0])],options=options,backend='torchmetrics')
    delta=abs(a['value']-b['value']);maximum=max(maximum,delta);assert delta<1e-6,(shape,gaussian,a['value'],b['value']);count+=1
for weight in ['square','simple','linear']:
  for background in [True,False]:
    p=torch.from_numpy(rng.integers(0,4,(2,8,9)));t=torch.from_numpy(rng.integers(0,4,(2,8,9)))
    options=dict(num_classes=5,include_background=background,weight_type=weight,input_format='index',empty_class_policy='torchmetrics_1.9.0')
    a=evaluate(p,t,metric='generalized_dice',sample_ids=['a','b'],options=options)
    b=evaluate(p,t,metric='generalized_dice',sample_ids=['a','b'],options=options,backend='torchmetrics')
    delta=abs(a['value']-b['value']);maximum=max(maximum,delta);assert delta<1e-7,(a,b);count+=1
print(json.dumps({'checks':count,'maximum_absolute_difference':maximum,'oracle':'torchmetrics==1.9.0','ssim_tolerance':1e-6,'dice_tolerance':1e-7}))

# The built-in path accepts eager TensorFlow and NumPy without TorchMetrics
# owning its calculation. Constant samples have closed-form expectations.
import tensorflow as tf
options=dict(data_range=1.,gaussian_kernel=False,sigma=1.,kernel_size=3,k1=.01,k2=.03)
p=np.full((1,1,5,5),.25,dtype=np.float64);t=np.full_like(p,.5)
expected=(2*.25*.5+.01**2)/(.25**2+.5**2+.01**2)
for left,right in [(p,t),(torch.from_numpy(p),torch.from_numpy(t)),(tf.constant(p),tf.constant(t))]:
    observation=evaluate(left,right,metric='ssim',sample_ids=['constant'],options=options)
    assert abs(observation['value']-expected)<1e-12
    assert observation['configuration']['schema']=='deepbom.metric_configuration.v2'
p[0,0,0,0]=np.nan
missing=evaluate(p,t,metric='ssim',sample_ids=['missing'],options=options,missing_policy='omit_sample')
assert missing['status']=='not_assessed' and missing['value'] is None and missing['missing_count']=='1'
try:evaluate(p,t,metric='ssim',sample_ids=['missing'],options=options)
except Exception:pass
else:raise AssertionError('Nonfinite input was assessed')
print('PASS independent direct SSIM oracle 1e-12, NumPy/Torch/TF adapters and explicit missing state')

import deepbom
observed=evaluate(np.ones((1,2,2),dtype=np.int32),np.ones((1,2,2),dtype=np.int32),metric="generalized_dice",sample_ids=["roundtrip"],options=dict(num_classes=2,include_background=False,weight_type="square",input_format="index",empty_class_policy="torchmetrics_1.9.0"))
from deepbom.metrics import evaluation_record
metric_name = "generalized_dice"
protocol_input = {"schema":"deepbom.evaluation_protocol.v1","id":"synthetic-dice","comparison":"descriptive","data_policy":"same_release",
    "metrics":[{"id":metric_name,"unit":"1","analysis_unit":"sample","higher_is_better":True,"max_regression":0.0,"range":[0,1],"aggregation":"equal_weight_sample_mean","missing_policy":"reject","repeated_observations":"one row per supplied synthetic sample"}],
    "required_populations":["synthetic"],"boundary":"Synthetic adapter check, no clinical assessment"}
protocol_request = {"schema":"deepbom.evidence_workflow_request.v1","operation":"protocol","input":protocol_input,"documents":[]}
protocol = deepbom.evidence_workflow(protocol_request)["document"]
ref = lambda schema,sha: {"kind":"evidence_document","schema":schema,"sha256":sha,"digest_scope":"canonical_document","subject_ref":None}
file_ref = {"kind":"artifact_file","schema":None,"sha256":"1"*64,"digest_scope":"file_bytes","subject_ref":None}
template = {key:dict(file_ref) for key in ("subject","dataset","cohort","run","code","environment")}
template.update(protocol=ref(protocol["schema"],protocol["protocol_sha256"]),configuration=ref("deepbom.metric_configuration.v2",observed["configuration_sha256"]),population="synthetic",comparison="descriptive",pair_set=None,boundary="Synthetic externally measured estimate")
result=evaluation_record(observed,template,metric_id=metric_name,documents=[protocol])
assert result["document"]["metrics"][0]["value"]==1 and result["document"]["origin"]=="externally_reported"
print("PASS measured metric configuration and result use the common Evaluation workflow")
