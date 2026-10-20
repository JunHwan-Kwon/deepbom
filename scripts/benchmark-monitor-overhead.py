#!/usr/bin/env python3
"""Repeatable CPU collection-overhead experiment, not a speed guarantee.

Run in a qualified Torch/TensorFlow environment. JSON records actual admitted
work and close time; compare equal completed work before interpreting throughput.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import platform
import sys
from importlib.metadata import version, PackageNotFoundError
import threading
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--frameworks',default='torch,tensorflow')
    parser.add_argument('--widths',default='16,128')
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--every',type=int,default=5)
    args=parser.parse_args()
    if min(args.repetitions,args.steps,args.every)<1 or args.warmup<0:parser.error('Invalid experiment counts')
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    from deepbom._native.core import engine
    def installed_version(name):
        try: return version(name)
        except PackageNotFoundError:
            if name == "tensorflow": return version("tensorflow-cpu")
            raise
    manifest={"schema":"deepbom.monitor_overhead_protocol.v1","settings":vars(args),"seed":912,
              "python":sys.version,"platform":platform.platform(),"cpu_logical_count":os.cpu_count(),
              "framework_versions":{key:installed_version(key) for key in [*args.frameworks.split(','),"numpy","psutil"]},
              "engine":engine("numerical_identity",{}),
              "benchmark_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "acceptance":{"controlled_final_state_and_losses":"exactly_equal","performance":"reported, no universal percent limit","GPU":"not_qualified"},
              "measurement":{"memory":"sampled aggregate child RSS every 10 ms; excludes already exited children","timing":"setup, training and close separately; rotated/reversed order; per-run step quantiles"}}
    (root/'experiment.json').write_text(json.dumps(manifest,indent=2))
    from deepbom.training.session import Monitor
    from deepbom._native.adapters import snapshot_input
    from deepbom._native.core import content_hash
    import numpy as np
    import psutil
    process=psutil.Process()
    rows=[]
    modes=['off','scalar','snapshot_copy','snapshot_worker','activation']
    for framework in args.frameworks.split(','):
      for width in map(int,args.widths.split(',')):
        for repeat in range(args.repetitions):
          order=modes[repeat%len(modes):]+modes[:repeat%len(modes)]
          if repeat%2:order=list(reversed(order))
          for mode in order:
            if framework=='torch':
              import torch
              torch.set_num_threads(1);torch.manual_seed(912)
              model=torch.nn.Sequential(torch.nn.Linear(width,width),torch.nn.Tanh(),torch.nn.Linear(width,4))
              optimizer=torch.optim.SGD(model.parameters(),lr=.001)
              x=torch.linspace(-1,1,8*width).reshape(8,width)
              def step():
                optimizer.zero_grad();y=model(x);loss=y.square().mean();loss.backward();optimizer.step();return float(loss.detach())
              def activation():return model(x).detach()
              subject='2'
            elif framework=='tensorflow':
              import tensorflow as tf
              tf.keras.utils.set_random_seed(912)
              model=tf.keras.Sequential([tf.keras.Input((width,),name="input"),tf.keras.layers.Dense(width,activation='tanh',name='hidden'),tf.keras.layers.Dense(4,name='output')],name='benchmark')
              optimizer=tf.keras.optimizers.SGD(learning_rate=.001)
              x=tf.reshape(tf.linspace(-1.,1.,8*width),(8,width))
              def step():
                with tf.GradientTape() as tape:
                  y=model(x,training=True);loss=tf.reduce_mean(tf.square(y))
                optimizer.apply_gradients(zip(tape.gradient(loss,model.trainable_variables),model.trainable_variables));return float(loss)
              def activation():return model(x,training=True)
              subject='output'
            else:raise ValueError('Unknown framework')
            for _ in range(args.warmup):step()
            directory=root/f'{framework}-{width}-{repeat}-{mode}'
            preset='weights' if mode=='snapshot_worker' else 'activations' if mode=='activation' else 'basic'
            monitor=Monitor(model,optimizer=optimizer if framework=='torch' else None,logdir=directory,preset=preset,
                            every=args.every,max_values=100000,errors='strict') if mode not in ('off','snapshot_copy') else None
            memory={'trainer_rss_peak_bytes':process.memory_info().rss,'worker_rss_peak_bytes':0}
            stop=threading.Event()
            def sample_memory():
              while not stop.wait(.01):
                memory['trainer_rss_peak_bytes']=max(memory['trainer_rss_peak_bytes'],process.memory_info().rss)
                total=0
                for child in process.children(recursive=True):
                  try:total+=child.memory_info().rss
                  except psutil.NoSuchProcess:pass
                memory['worker_rss_peak_bytes']=max(memory['worker_rss_peak_bytes'],total)
            thread=threading.Thread(target=sample_memory,daemon=True);thread.start()
            entered=time.monotonic()
            if monitor:monitor.__enter__()
            setup=time.monotonic()-entered
            times=[];losses=[];copy_seconds=0.;copy_bytes=0;requested=0
            started=time.monotonic()
            try:
              for index in range(args.steps):
                begin=time.monotonic()
                if mode=='activation' and index%args.every==0:
                  sha=monitor.record_snapshot(at={'phase':'train','step':index,'boundary':'before_forward'})
                  monitor.record_activations({subject:activation()},snapshot_sha256=sha,at={'phase':'train','step':index},invocation=f'probe-{index}',
                    context={'mode':'train','input_identity':'fixed-linspace','state_boundary':'before_forward'})
                  requested+=1
                losses.append(step())
                if monitor:
                  monitor.record_metrics({'loss':losses[-1]},at={'phase':'train','step':index})
                  if framework=='tensorflow' and mode=='snapshot_worker' and index%args.every==0:
                    monitor.recorder.observe_snapshot(at={'phase':'train','step':index,'boundary':'after_apply_gradients'})
                if mode=='snapshot_copy' and index%args.every==0:
                  before=time.monotonic();raw=snapshot_input(model,max_values=100000,max_state_bytes=64*1024*1024)
                  encoded=json.dumps(raw,allow_nan=False);copy_bytes+=len(encoded.encode());copy_seconds+=time.monotonic()-before;requested+=1
                times.append(time.monotonic()-begin)
              elapsed=time.monotonic()-started
            finally:
              close_started=time.monotonic()
              if monitor:monitor.__exit__(None,None,None)
              close_elapsed=time.monotonic()-close_started;stop.set();thread.join()
            final=snapshot_input(model,max_values=0)
            state_hash=content_hash(final)
            (root/f"{framework}-{width}-{repeat}-{mode}-state.json").write_text(json.dumps(final))
            receipt=monitor.receipt if monitor else None
            actual=receipt['analysis']['counts'] if receipt and receipt['analysis'] else None
            if actual:requested=actual['requested']
            # Capture-only records encoding work; managed analysis records raw
            # snapshot copy work from the actual admission index. Never report
            # an unmeasured managed copy as zero seconds/bytes.
            copy_measurement=None
            if mode=='snapshot_copy':
              copy_measurement={'basis':'snapshot_copy_and_json_encoding','seconds':copy_seconds,'encoded_bytes':copy_bytes}
            elif actual:
              index=json.loads((directory/'analysis-bindings.json').read_text())
              copy_measurement={'basis':'admitted_immutable_snapshot_capture','seconds':index['copy_seconds'],'raw_state_bytes':index['copy_bytes'],
                                'peak_reserved_bytes':receipt['analysis']['queue']['peak_reserved_bytes']}
            quantile=lambda q:float(np.quantile(times,q))
            row={'framework':framework,'width':width,'repeat':repeat,'mode':mode,'steps':args.steps,'warmup':args.warmup,'every':args.every,
                 'setup_seconds':setup,'training_seconds':elapsed,'steps_per_second':args.steps/elapsed,
                 'step_seconds':{'p50':quantile(.5),'p95':quantile(.95),'p99':quantile(.99)},'close_seconds':close_elapsed,
                 'requested_captures':requested,'analysis_counts':actual,'monitor_receipt':receipt,'copy_measurement':copy_measurement,
                 'memory':memory,'gpu_memory':{'status':'not_assessed','reason':'CPU workload'},'state_sha256':state_hash,'losses':losses,
                 'stored_bytes':sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()) if directory.exists() else 0}
            rows.append(row)
            (root/'measurements.json').write_text(json.dumps({'schema':'deepbom.monitor_overhead_experiment.v1','rows':rows,
              'boundary':'Synthetic CPU workloads. setup/steady/close are separate. Worker RSS is sampled aggregate child RSS, not VRAM. Equal periods do not establish equal completed work.'},indent=2))
            print(framework,width,repeat,mode,round(elapsed,4),actual,flush=True)
    groups={}
    for row in rows:groups.setdefault((row['framework'],row['width'],row['repeat']),[]).append(row)
    for group in groups.values():
      reference=next(r for r in group if r['mode']=='off')
      for row in group:
        assert row['state_sha256']==reference['state_sha256'],('state changed',row['framework'],row['mode'])
        assert row['losses']==reference['losses'],('loss changed',row['framework'],row['mode'])
    (root/'validation.json').write_text(json.dumps({'controlled_cpu_state_and_loss_preserved':True,'runs':len(rows),'frameworks':args.frameworks.split(','),'not_qualified':['GPU','distributed','unbounded models','production default overhead guarantee']},indent=2))

if __name__=='__main__':main()
