from pathlib import Path
import tempfile,time,json,torch
from deepbom.training.session import Monitor
from deepbom.training.integrations import read_run,read_analysis_view
from deepbom._process import run_process,execution_control,RunControl,ProcessCancelled
import os,sys,threading
with tempfile.TemporaryDirectory() as directory:
    model=torch.nn.Linear(3,2); optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    with Monitor(model,optimizer=optimizer,logdir=Path(directory)/'run',preset='weights',every=1,errors='strict') as monitor:
        for i in range(3):
            optimizer.zero_grad();model(torch.ones(2,3)).sum().backward();optimizer.step()
    view=read_analysis_view(Path(directory)/'run')
    assert view['index']['counts']['completed']>0,view
    print('managed counts',view['index']['counts'],monitor.receipt['analysis'])
    with Monitor(model,optimizer=optimizer,logdir=Path(directory)/'late',preset='weights',every=1,errors='strict') as monitor:
        optimizer.zero_grad();model(torch.ones(2,3)).sum().backward();optimizer.step()
        monitor.recorder.finish('ended')
    view=read_analysis_view(Path(directory)/'late')
    assert any(r.get('late') for r in view['index']['bindings']),view
    print('late result preserves terminal',view['bundle']['training']['lifecycle'])
    from deepbom.training.integrations import export
    receipt=export(Path(directory)/'late',destination='tensorboard',logdir=Path(directory)/'tensorboard')
    assert receipt['analysis_bindings_sha256']
    exported=Path(directory)/'tensorboard'/'deepbom-evidence'/view['bundle']['training']['training_ir_sha256']/('analysis-'+receipt['analysis_bindings_sha256'])
    assert json.loads((exported/'index.json').read_text())['analysis_bindings_sha256']==receipt['analysis_bindings_sha256']
    assert len(list(exported.glob('*.json')))>3
    print('Late completion uses the same verified TensorBoard artifact export')

    start=time.monotonic()
    try:run_process([sys.executable,'-c','import time;time.sleep(50)'],environment=os.environ.copy(),timeout=.2,file_limit=1024)
    except TimeoutError:pass
    else:raise AssertionError('hang survived')
    assert time.monotonic()-start<1
    print('process timeout and reap passed')
# Admission precedes independent capture and releases reservations on failure.
from deepbom.jobs import AnalysisQueue
from unittest.mock import patch
producer={'name':'admission-test','version':'1','code_sha256':'a'*64}
with tempfile.TemporaryDirectory() as directory:
    gate=threading.Event();entered=threading.Event()
    def operation(*args):entered.set();gate.wait(3);return {'value':1}
    with patch('deepbom.jobs.engine',operation),AnalysisQueue(directory,producer=producer,max_pending=1,max_request_bytes=1024) as queue:
        job=queue.submit('snapshot',{},configuration={});assert entered.wait(2)
        try:queue.submit_capture('snapshot',lambda:(_ for _ in ()).throw(AssertionError('copied over budget')),configuration={})
        except BufferError:pass
        else:raise AssertionError('full queue admitted capture')
        gate.set();queue.wait(job)
        assert queue.reserved_bytes==0
# Kernel-level resource mechanisms, not just descriptor declarations.
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'large'
    result=run_process([sys.executable,'-c',f'open({str(path)!r},"wb").write(b"x"*1000000)'],environment=os.environ.copy(),timeout=3,file_limit=1024)
    assert result['returncode']!=0 and path.stat().st_size<=1024
    result=run_process([sys.executable,'-c','a=bytearray(512*1024*1024)'],environment=os.environ.copy(),timeout=3,file_limit=1024,memory_bytes=128*1024*1024)
    assert result['returncode']!=0
print('Admission before copy, exact reservation release, OS file/address-space limits passed')

# A cancelled managed job terminates the running process, not just its Future.
with tempfile.TemporaryDirectory() as directory:
    started=threading.Event()
    def hang(*args):
        started.set()
        return run_process([sys.executable,'-c','import time;time.sleep(50)'],environment=os.environ.copy(),timeout=60,file_limit=1024)
    with patch('deepbom.jobs.engine',hang):
        queue=AnalysisQueue(directory,producer=producer,max_pending=1,max_request_bytes=1024,close_timeout=.3)
        job=queue.submit('snapshot',{},configuration={});assert started.wait(2)
        before=time.monotonic()
        # Shutdown must not launch fresh digest engines while holding the lock.
        # The worker already validated this admitted request before `started`.
        with patch.object(queue,'_request',side_effect=AssertionError('shutdown revalidated admitted input')):
            receipt=queue.close()
        assert time.monotonic()-before<1 and receipt['all_jobs_stopped'],receipt
        assert queue.status(job)['execution_status'] in ('failed','cancelled')
# Failed scheduling releases both the capture and transferred reservations.
with tempfile.TemporaryDirectory() as directory:
    with AnalysisQueue(directory,producer=producer,max_request_bytes=1024) as queue:
        with patch.object(queue.executor,'submit',side_effect=RuntimeError('executor unavailable')):
            try:queue.submit('snapshot',{},configuration={})
            except RuntimeError:pass
            else:raise AssertionError('Missing scheduling failure')
        assert queue.reserved_bytes==0 and queue.capturing==0 and not queue.controls
print('PASS close deadline kills a running process; failed scheduling releases reservations')
result=run_process([sys.executable,'-c','import sys;sys.stderr.write("x"*100000+" diagnostic-tail")'],environment=os.environ.copy(),timeout=3,file_limit=1024)
assert len(result['diagnostic'])<=16384 and result['diagnostic'].endswith('diagnostic-tail')
print('PASS actual managed diagnostics keep the bounded final tail')

# Cleanup failures must never replace the startup cause or allow reuse of a
# half-closed monitor. Exercise both cleanup branches without a real child.
from unittest.mock import MagicMock
import warnings
with tempfile.TemporaryDirectory() as directory:
    original=ValueError('unsupported capture environment')
    child=MagicMock()
    child.close.side_effect=RuntimeError('secondary engine cleanup error')
    monitor=Monitor(torch.nn.Linear(3,2),logdir=directory,preset='weights')
    with patch.object(monitor.recorder,'__enter__'), \
         patch.object(monitor.recorder,'__exit__',side_effect=OSError('secondary collector cleanup error')), \
         patch('deepbom._native.session.NativeSession',return_value=child), \
         patch('deepbom.training.managed.ManagedSnapshots',side_effect=original), \
         warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        try:monitor.__enter__()
        except ValueError as actual:assert actual is original
        else:raise AssertionError('Original startup exception was lost')
    assert monitor.closed and not monitor.active and child.close.call_count==1
    assert [d['stage'] for d in monitor.diagnostics]==['startup_collector_cleanup','startup_engine_cleanup']
print('PASS original startup exception survives failures in both cleanup paths')

from deepbom._native.core import atomic_json
with tempfile.TemporaryDirectory() as directory:
    target=Path(directory)/'oversized.json'
    try:atomic_json(target,{'value':'x'*2048},max_bytes=1024)
    except ValueError:pass
    else:raise AssertionError('Oversized transport was accepted')
    assert not target.exists() and not list(Path(directory).glob('.writing-*'))
    with patch('deepbom.training.managed.engine',return_value={'wasm_sha256':'a'*64}):
        recorder=MagicMock();recorder.model=torch.nn.Linear(1,1);recorder.logdir=Path(directory);recorder.max_state_bytes=1024
        from deepbom.training.managed import ManagedSnapshots
        try:ManagedSnapshots(recorder,max_total_bytes=0)
        except ValueError:pass
        else:raise AssertionError('Explicit zero byte budget silently became a default')
print('PASS transport rejected before disk write; invalid explicit budgets are not defaulted')

# Kill the real controlling process; its direct engine must not survive as an
# orphan even though no Python finally/__exit__ can run in that controller.
import subprocess, signal
with tempfile.TemporaryDirectory() as directory:
    marker=Path(directory)/'child-pid'
    temporary=marker.with_suffix('.tmp')
    child_source=(f'import os,time;from pathlib import Path;Path({str(temporary)!r}).write_text(str(os.getpid()));'
                  f'os.replace({str(temporary)!r},{str(marker)!r});time.sleep(50)')
    controller_source=("import os,sys;from deepbom._process import run_process;"
                       f"run_process([sys.executable,'-c',{child_source!r}],environment=os.environ.copy(),timeout=60,file_limit=1024)")
    controller=subprocess.Popen([sys.executable,'-c',controller_source],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    child_pid=None
    def running(pid):
        try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].strip().split()[0] not in ('Z','X')
        except (FileNotFoundError,ProcessLookupError):return False
    try:
        deadline=time.monotonic()+5
        while not marker.exists() and controller.poll() is None and time.monotonic()<deadline:time.sleep(.01)
        assert marker.exists(),'Managed child did not start'
        child_pid=int(marker.read_text());assert running(child_pid)
        controller.kill();controller.wait(timeout=2)
        deadline=time.monotonic()+2
        while running(child_pid) and time.monotonic()<deadline:time.sleep(.01)
        assert not running(child_pid),'Engine survived its controlling process'
    finally:
        if controller.poll() is None:controller.kill();controller.wait(timeout=2)
        controller.stderr.close()
        if child_pid is not None and running(child_pid):os.kill(child_pid,signal.SIGKILL)
print('PASS controller SIGKILL stops its direct engine without Python cleanup')

# Interrupt the drain while a real engine process is running. Closing the
# Future alone is insufficient: the process must stop and the queue be reusable.
from concurrent.futures import wait as real_wait
with tempfile.TemporaryDirectory() as directory:
    marker=Path(directory)/'running'
    interrupted=KeyboardInterrupt('interrupt queue drain')
    def running_engine(*args):
        source=f'import time;from pathlib import Path;Path({str(marker)!r}).write_text("ready");time.sleep(50)'
        return run_process([sys.executable,'-c',source],environment=os.environ.copy(),timeout=60,file_limit=1024)
    with patch('deepbom.jobs.engine',running_engine):
        queue=AnalysisQueue(Path(directory)/'jobs',producer=producer,close_timeout=2)
        job=queue.submit('snapshot',{},configuration={})
        until=time.monotonic()+5
        while not marker.exists() and time.monotonic()<until:time.sleep(.01)
        assert marker.exists()
        calls=[0]
        def interrupt_once(*args,**kwargs):
            calls[0]+=1
            if calls[0]==1:raise interrupted
            return real_wait(*args,**kwargs)
        with patch('deepbom.jobs.wait',interrupt_once):
            try:queue.close()
            except KeyboardInterrupt as actual:assert actual is interrupted
            else:raise AssertionError('Shutdown swallowed Ctrl+C')
        assert queue.close_receipt['all_jobs_stopped'],queue.close_receipt
        assert queue.status(job)['execution_status']=='cancelled'
        assert not queue.lock_path.exists()
        assert queue.close()==queue.close_receipt
    with AnalysisQueue(Path(directory)/'jobs',producer=producer):pass
print('PASS interrupted drain terminates the real engine, releases lock, and preserves Ctrl+C')

with tempfile.TemporaryDirectory() as directory:
    queue=AnalysisQueue(directory,producer=producer)
    original=RuntimeError('original caller failure')
    interrupted=KeyboardInterrupt('secondary shutdown interrupt')
    try:
        with queue:
            with patch.object(queue,'close',side_effect=interrupted):
                assert queue.__exit__(type(original),original,None) is False
            raise original
    except RuntimeError as actual:assert actual is original
    assert queue.close_receipt['all_jobs_stopped']

# Collection failure must not overwrite the actual user exception's lifecycle.
# Interruptions during cleanup must also release the recorder and session.
for cause in (RuntimeError('training failed after observer failure'),KeyboardInterrupt('training interrupted')):
    with tempfile.TemporaryDirectory() as directory, warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        monitor=Monitor(torch.nn.Linear(1,1),logdir=directory)
        try:
            with monitor:
                monitor._call(lambda:(_ for _ in ()).throw(OSError('observer unavailable')))
                raise cause
        except BaseException as actual:assert actual is cause
        else:raise AssertionError('Original training exception was swallowed')
        expected='interrupted' if isinstance(cause,KeyboardInterrupt) else 'failed'
        assert monitor.receipt['training_lifecycle']==expected,monitor.receipt
        assert read_run(directory)['training']['lifecycle']==expected
        assert monitor.receipt['collection_status']=='incomplete'
        assert monitor.closed and not monitor.active and monitor.engine_session.done.is_set()
        assert not (Path(directory)/'.writer.lock').exists()
for original in (None,RuntimeError('original training error')):
    with tempfile.TemporaryDirectory() as directory, warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        monitor=Monitor(torch.nn.Linear(1,1),logdir=directory,preset='weights')
        interrupted=KeyboardInterrupt('interrupted worker close')
        try:
            with monitor:
                with patch.object(monitor.managed,'close',side_effect=interrupted):
                    monitor.__exit__(type(original) if original else None,original,None)
                if original:raise original
        except BaseException as actual:assert actual is (original or interrupted)
        else:raise AssertionError('Expected exception was swallowed')
        assert monitor.closed and not monitor.active
        assert monitor.engine_session.done.is_set()
        assert monitor.managed.queue.close_receipt['all_jobs_stopped']
        assert monitor.receipt['collection_status']=='incomplete'
        assert not (Path(directory)/'.writer.lock').exists()
        assert not (Path(directory)/'analysis-jobs'/'.worker.lock').exists()
        read_run(directory)  # Terminal evidence still passes the common reader.
print('PASS failure/interruption lifecycle, original exception, and all cleanup ownership preserved')

# Explicit loops may have more than one unresolved update token. A snapshot
# belongs to the returned token, not to the latest global attempt counter.
from deepbom.training.collector import Recorder
with tempfile.TemporaryDirectory() as directory:
    model=torch.nn.Linear(1,1);optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    with Recorder(model,optimizer=optimizer,logdir=directory,capture=('weights',)) as recorder:
        first=recorder.before_update(optimizer=optimizer,at={'step':1})
        second=recorder.before_update(optimizer=optimizer,at={'step':2})
        recorder.after_update(second,optimizer=optimizer)
        recorder.after_update(first,optimizer=optimizer)
    events=read_run(directory)['events']
    snapshots=[row['at'] for row in events if row['kind']=='model_snapshot']
    assert [row['optimizer_attempt'] for row in snapshots]==['2','1'],snapshots
    assert [row['optimizer_token'] for row in snapshots]==[second,first]
    assert [row['payload']['token'] for row in events if row['kind']=='update_returned']==[second,first]
print('PASS out-of-order manual update returns preserve exact snapshot attempt/token binding')
