"""Optional backend qualification probe. This is not a supported capture adapter."""
import copy, json, time
from importlib.metadata import version
import torch
import torchlens
from deepbom._native.adapters import snapshot_input
from deepbom._native.core import engine

assert version('torchlens') == '2.36.1'
torch.set_num_threads(1)
torch.manual_seed(19)
model = torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.ReLU(),torch.nn.Linear(4,2)).eval()
x = torch.arange(6,dtype=torch.float32).reshape(2,3)
before = snapshot_input(model,max_values=0)
native = engine('snapshot', before)
expected = model(x).detach().clone()
for _ in range(5): model(x)
start = time.perf_counter()
for _ in range(100): model(x)
baseline = (time.perf_counter()-start)/100
observed = copy.deepcopy(model)
with torch.random.fork_rng(), torch.no_grad():
    start=time.perf_counter(); trace=torchlens.trace(observed,x.clone()); elapsed=time.perf_counter()-start
try:
    assert torch.equal(trace[-1].tensor,expected)
    assert snapshot_input(model,max_values=0)==before
    assert snapshot_input(observed,max_values=0)==before
    subjects={n['id'] for n in native['model_ir']['program']['nodes']}
    mapping=[]
    for op in trace.layer_list:
        owner=op.atomic_module_address
        mapping.append({'runtime_op':op.layer_label,'native_subject':owner if owner in subjects else None,
                        'mapping_status':'observed_module_address' if owner in subjects else 'unmapped_functional_or_boundary',
                        'parents':list(op.parents),'shape':list(op.shape)})
    assert len([row for row in mapping if row['native_subject']])==3
    print(json.dumps({'torchlens':version('torchlens'),'torch':torch.__version__,
        'baseline_mean_seconds':baseline,'probe_seconds':elapsed,'cold_probe_ratio':elapsed/baseline,
        'output_equal':True,'state_equal':True,'mapping':mapping,
        'qualification':'synthetic CPU eval probe only; not a production adapter or per-batch performance claim'},indent=2))
finally: trace.cleanup()
