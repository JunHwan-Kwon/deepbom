#!/usr/bin/env python3
"""Real process-death, commit-window and identity recovery regression checks."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
from deepbom._ownership import WriterOwnership
from deepbom._native.core import atomic_json, content_hash
from deepbom.jobs import AnalysisQueue
from deepbom.training import recover
from deepbom.training import Recorder
from deepbom.training.integrations import read_run, read_analysis_view

PRODUCER = {"name":"recovery-test","version":"1","code_sha256":"a"*64}


def child(root, script, *, kill=False):
    process = subprocess.Popen([sys.executable,"-c",script,str(root)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        if kill:
            deadline = time.monotonic()+30
            while not (root/"ready").exists():
                if process.poll() is not None: raise AssertionError(process.communicate())
                if time.monotonic()>deadline: raise TimeoutError("Recovery fixture did not start")
                time.sleep(.02)
            os.kill(process.pid,signal.SIGKILL)
        out,err = process.communicate(timeout=45)
        assert process.returncode in (-signal.SIGKILL,17), (out,err,process.returncode)
    finally:
        if process.poll() is None:
            process.kill(); process.wait()


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    path = root/"owner"
    owner = WriterOwnership(path,purpose="test")
    try:
        try: WriterOwnership(path,purpose="test")
        except FileExistsError: pass
        else: raise AssertionError("Concurrent owner accepted")
    finally: owner.close()
    path.write_text("12345")
    try: WriterOwnership(path,purpose="test")
    except FileExistsError: pass
    else: raise AssertionError("Legacy ownership silently stolen")
    path.unlink()

    if hasattr(os, "fork"):
        # A long-lived data-loader child must not keep a dead trainer's lock.
        fork_root = root/"fork"; fork_root.mkdir()
        child(fork_root, '''
import os,sys,time
from pathlib import Path
from deepbom._ownership import WriterOwnership
root=Path(sys.argv[1]);owner=WriterOwnership(root/'owner',purpose='test')
pid=os.fork()
if pid==0:
    os.close(1);os.close(2);time.sleep(60);os._exit(0)
(root/'child.pid').write_text(str(pid));os._exit(17)
''')
        loader_pid = int((fork_root/"child.pid").read_text())
        try:
            with WriterOwnership(fork_root/"owner",purpose="test") as next_owner:
                assert next_owner.recovered is not None
        finally:
            try: os.kill(loader_pid,signal.SIGKILL)
            except ProcessLookupError: pass
        print("PASS forked data-loader does not retain a dead trainer's writer ownership")

    qdir = root/"queue"; qdir.mkdir()
    child(qdir, '''
import sys,time
from pathlib import Path
from unittest.mock import patch
from deepbom.jobs import AnalysisQueue
p={"name":"recovery-test","version":"1","code_sha256":"a"*64}
root=Path(sys.argv[1])
def work(*args):
    (root/'ready').touch();time.sleep(120)
with patch('deepbom.jobs.engine',work):
    queue=AnalysisQueue(root,producer=p)
    queue.submit('snapshot',{'value':7},configuration={})
    time.sleep(120)
''', kill=True)
    with patch("deepbom.jobs.engine",return_value={"value":7}), AnalysisQueue(qdir,producer=PRODUCER) as queue:
        assert queue._ownership.recovered is not None
        jobs = queue.recover(); assert len(jobs)==1
        job = jobs[0]
        assert queue.wait(job)["execution_status"]=="completed"
        assert queue.result(job)=={"value":7}
        # Simulate a crash after immutable result+intent, before status commit.
        queue._receipt(job,"running")
        with patch("deepbom.jobs.engine",side_effect=AssertionError("Committed job reran")):
            assert queue.recover()==[job]
            assert queue.result(job)=={"value":7}
        queue._receipt(job,"running")
        intent = json.loads((qdir/job/"completion.json").read_text())
        intent["result_document_sha256"]="b"*64
        atomic_json(qdir/job/"completion.json",intent,replace=True)
        try: queue.recover()
        except ValueError as error: assert "identity" in str(error)
        else: raise AssertionError("Tampered completion accepted")
    print("PASS real SIGKILL ownership recovery, completion acknowledgement, tamper refusal")

    for phase in ("before_publish","after_publish"):
        run = root/phase
        child(run, '''
import os,sys
from pathlib import Path
from deepbom.training import Recorder
from deepbom._native.core import engine,atomic_json
root=Path(sys.argv[1]);r=Recorder(logdir=root)
r.__enter__();r.record_metrics({'loss':1.25},at={'step':3})
def crash(op,payload):
    result=engine(op,payload)
    if root.name=='after_publish':
        r.store.put(result['chunk']);r.store.put(result['document'])
        atomic_json(root/'latest.json',{'schema':'deepbom.training_head.v1','training_ir_sha256':result['document']['training_ir_sha256']},replace=True)
    os._exit(17)
r._engine=crash;r.flush()
''')
        saved = (run/"pending-training.json").read_bytes()
        intent = json.loads(saved); intent["input_sha256"] = "0"*64
        (run/"pending-training.json").write_text(json.dumps(intent))
        try: recover(run,analyze_pending=False)
        except ValueError as error: assert "identity" in str(error)
        else: raise AssertionError("Altered Training intent accepted")
        (run/"pending-training.json").write_bytes(saved)
        receipt = recover(run,analyze_pending=False)
        bundle = read_run(run)
        assert receipt["training_lifecycle"]=="interrupted"
        assert [e["payload"]["values"] for e in bundle["events"] if e["kind"]=="metrics"]==[{"loss":1.25}]
        assert sum(e["kind"]=="collector_recovered" for e in bundle["events"])==1
        again = recover(run,analyze_pending=False)
        assert again["training_ir_sha256"]==receipt["training_ir_sha256"]
        assert not (run/"pending-training.json").exists()
    print("PASS Training intent replay on both commit windows, exact-once observations and idempotent recovery")

    cli = subprocess.run([sys.executable, "-m", "deepbom", "training", "recover",
                          str(run), "--timeout", "30", "--skip-analysis"],
                         capture_output=True, text=True, check=True)
    assert json.loads(cli.stdout)["training_ir_sha256"] == receipt["training_ir_sha256"]
    invalid = subprocess.run([sys.executable, "-m", "deepbom", "training", "recover",
                              str(run), "--timeout", "nan"], capture_output=True, text=True)
    assert invalid.returncode != 0 and "finite and positive" in invalid.stderr
    print("PASS installed Python CLI recovery and invalid deadline refusal")

    run = root/"unsealed"
    recorder = Recorder(logdir=run)
    recorder.__enter__()
    try:
        try: recover(run)
        except FileExistsError: pass
        else: raise AssertionError("Recovery interrupted an active recorder")
        recorder.record_metrics({"loss":2.5})
        recorder.flush()
        recorder.record_metrics({"loss":1.5})
        def failed_engine(*args): raise RuntimeError("Injected engine failure")
        recorder._engine=failed_engine
        try: recorder.__exit__(None,None,None)
        except RuntimeError: pass
        else: raise AssertionError("Recorder failure was hidden")
        assert (run/"unsealed-events.json").exists()
        recover(run,analyze_pending=False)
        events = read_run(run)["events"]
        assert [e["payload"]["values"]["loss"] for e in events if e["kind"]=="metrics"]==[2.5,1.5]
        assert not (run/"unsealed-events.json").exists()
    finally:
        if not recorder._closed: recorder.__exit__(RuntimeError,None,None)
    print("PASS active writer refused; failed flush journal revalidated without duplicate history")

    for phase in ("pending","commit_pending"):
        run = root/phase
        child(run, '''
import os,sys
from pathlib import Path
import torch
import deepbom
root=Path(sys.argv[1]);torch.set_num_threads(1)
model=torch.nn.Linear(2,1)
m=deepbom.monitor(model,logdir=root,preset='weights',errors='strict')
m.__enter__();m.managed.capture(at={'step':9})
binding=next(iter(m.managed.jobs.values()))
m.managed.queue.wait(binding['job_sha256'])
if root.name=='commit_pending':
    flush=m.recorder.flush
    def crash():
        flush();os._exit(17)
    m.recorder.flush=crash
    m.managed.drain()
os._exit(17)
''')
        receipt = recover(run,timeout=45)
        view = read_analysis_view(run)
        rows = view["index"]["bindings"]
        assert len(rows)==1 and rows[0]["status"]=="completed"
        assert rows[0]["at"]=={"step":9}
        assert rows[0]["late"]==(phase=="pending")
        assert view["index"]["counts"]["completed"]==1
        assert sum(e["kind"]=="model_snapshot" for e in view["bundle"]["events"])==(phase=="commit_pending")
        again = recover(run,timeout=45)
        assert again["training_ir_sha256"]==receipt["training_ir_sha256"]
    print("PASS pending/partially acknowledged analysis bindings, late evidence separated, same step and state")
