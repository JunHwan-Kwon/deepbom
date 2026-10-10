"""Regression probes for previously observed false claims; synthetic inputs only."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='3', TF_NUM_INTEROP_THREADS='1', TF_NUM_INTRAOP_THREADS='1')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'channels/python/src'))
import tempfile
import types
import torch
import keras
import tensorflow as tf
from deepbom.optimization import optimize, suggest
from deepbom._native.adapters import snapshot
from deepbom.training import Recorder
from deepbom.training.integrations import export, import_logs, read_run

torch.set_num_threads(1)
root=Path(tempfile.mkdtemp(prefix='deepbom-correctness-'))
def rejected(fn,phrase):
    try: fn()
    except (ValueError,RuntimeError) as exc:
        assert phrase.lower() in str(exc).lower(),str(exc)
    else: raise AssertionError('Expected rejection: '+phrase)
m=torch.nn.Sequential(torch.nn.Linear(2,2)).eval()
m.forward=types.MethodType(lambda self,x:self[0](x)+7,m)
rejected(lambda:snapshot(m),'override')
rejected(lambda:optimize(m,rules=[],example_args=(torch.ones(1,2),)),'override')
m=torch.nn.Sequential(torch.nn.Linear(1,1,bias=False))
with torch.no_grad():m[0].weight.fill_(torch.finfo(torch.float32).max)
rejected(lambda:optimize(m,rules=[],example_args=(torch.ones(1,1),),loss_fn=lambda out:((out-out.detach())*(-1e38)).sum()),'nonfinite state')
assert torch.isfinite(m[0].weight).all()
conv=torch.nn.Conv2d(1,1,1);bn=torch.nn.BatchNorm2d(1)
shared=torch.nn.Sequential(conv,conv,bn).eval()
assert suggest(shared)==[]
x=keras.Input((2,2,1));conv=keras.layers.Conv2D(1,1);a=conv(x);b=conv(x);bn=keras.layers.BatchNormalization();model=keras.Model(x,[bn(a),b])
assert suggest(model)==[]
rejected(lambda:optimize(model,rules=[{'kind':'fold_conv_bn','subject':conv.name,'batchnorm':bn.name}],example_args=(tf.ones((1,2,2,1)),)),'Shared Keras')
m=torch.nn.Sequential(torch.nn.Linear(2,1));optimizer=torch.optim.SGD(m.parameters(),lr=.1)
with Recorder(m,optimizer=optimizer,logdir=root/'observations',capture=('gradients','activations','metrics')) as r:
    rejected(lambda:r.before_update(optimizer=optimizer,at={'step':0},variables=[m[0].weight],gradients=[torch.ones(7)]),'shape')
    sha=r.record_snapshot(at={'step':0})
    rejected(lambda:r.record_activations({'0':torch.ones(1,1)},snapshot_sha256=sha,at={'step':0},invocation=1,context={'mode':'eval','input_identity':{'kind':'synthetic'},'state_boundary':'before_forward','at':{'step':99}}),'at')
    rejected(lambda:r.record_activations({'0':torch.ones(1,1)},snapshot_sha256=sha,at={'step':0},invocation=1,context={'mode':None,'input_identity':None,'state_boundary':None}),'context')
    r.record_metrics({'loss':2.0},at={'step':0,'phase':'train'})
    r.record_metrics({'loss':3.0},at={'step':0,'phase':'validation'})
foreign=torch.optim.SGD(torch.nn.Linear(2,1).parameters(),lr=.1)
with Recorder(m,optimizer=foreign,logdir=root/'foreign',capture=('gradients',)) as r:
    rejected(lambda:r.before_update(optimizer=foreign,at={'step':0}),'selected model')
receipt=export(root/'observations',destination='tensorboard',logdir=root/'tb')
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.util.tensor_util import make_ndarray
acc=EventAccumulator(receipt['event_logdir']).Reload()
assert [make_ndarray(e.tensor_proto).item() for e in acc.Tensors('train/loss')]==[2]
assert [make_ndarray(e.tensor_proto).item() for e in acc.Tensors('validation/loss')]==[3]
import_logs('tensorboard',output=root/'imported',logdir=receipt['event_logdir'])
phases={e['at']['phase']:e['payload']['values']['loss'] for e in read_run(root/'imported')['events'] if 'values' in e['payload']}
assert phases=={'train':2,'validation':3},phases
print('P0 correctness regressions passed: execution overrides, post-update finite state, shared calls, gradient shape, activation time/context, optimizer binding, phase-preserving TensorBoard round trip. '+str(root))
