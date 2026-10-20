#!/usr/bin/env python3
"""Queue identity, bounded work, cancellation, recovery and real file intake."""
from pathlib import Path
import tempfile
import threading
import json
from unittest.mock import patch
from deepbom.jobs import AnalysisQueue, _json_primitives
from deepbom.completed_files import CompletedFileWatcher, mark_complete

producer = {"name":"queue-test","version":"1","code_sha256":"a"*64}
for value in [{"한글": [None, True, False, 0, -2.5, "\"\\\n\x00" * 5000]}, [], {}, "😀" * 5000]:
    exact = len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8'))
    budget = [1_000_000, exact]
    _json_primitives(value, budget=budget)
    assert budget[1] == 0
    try: _json_primitives(value, budget=[1_000_000, exact-1])
    except ValueError as error: assert 'pre-copy' in str(error)
    else: raise AssertionError('Pre-copy byte budget failed at the exact serialization boundary')
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp); gate=threading.Event(); began=threading.Event(); seen=[]
    # Accepted wire-size boundaries must also survive storage, execution and
    # restart. Pretty printing used to turn this 460-byte request into 943 bytes.
    payload = {"rows": [{"x": i} for i in range(20)], "label": "한글"}
    request = {"schema":"deepbom.analysis_job_request.v1", "producer":producer,
               "operation":"snapshot", "configuration":{}, "payload":payload}
    exact = len(json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    with patch("deepbom.jobs.engine", return_value={"value":1}), AnalysisQueue(tmp/"exact-budget", producer=producer, max_request_bytes=exact) as queue:
        job = queue.submit("snapshot", payload, configuration={})
        assert queue.wait(job)["execution_status"] == "completed"
        assert (tmp/"exact-budget"/job/"request.json").stat().st_size == exact
    with AnalysisQueue(tmp/"exact-budget", producer=producer, max_request_bytes=exact) as queue:
        assert queue.result(job) == {"value":1}
        assert queue.submit("snapshot", payload, configuration={}) == job
    with patch("deepbom.jobs._MAX_RESULT_BYTES", 20), patch("deepbom.jobs.engine", return_value={"large":"x"*30}), AnalysisQueue(tmp/"oversize-result", producer=producer) as queue:
        job = queue.submit("snapshot", {}, configuration={})
        assert queue.wait(job)["execution_status"] == "failed"
        assert queue.status(job)["retryable"] is False
        assert queue.recover() == []
        assert not (tmp/"oversize-result"/job/"result.json").exists()
    with AnalysisQueue(tmp/'byte-budget', producer=producer, max_request_bytes=512) as queue:
        with patch('deepbom.jobs.json.loads', side_effect=AssertionError('Detached copy started before budget rejection')):
            try: queue.submit('snapshot', {'value':'\x00' * 100_000}, configuration={})
            except ValueError as error: assert 'pre-copy' in str(error)
            else: raise AssertionError('Oversized escaped string was copied')
        assert not queue.futures
    # Capture can be in progress when close expires. Keep ownership until that
    # capture exits, then allow a second close to acknowledge actual cleanup.
    entered=threading.Event(); release=threading.Event(); capture_errors=[]
    queue=AnalysisQueue(tmp/'closing-capture',producer=producer,max_request_bytes=512)
    def capture():
        entered.set();release.wait(5);return {}
    def submit_capture():
        try:queue.submit_capture('snapshot',capture,configuration={})
        except RuntimeError as error:capture_errors.append(error)
    thread=threading.Thread(target=submit_capture);thread.start()
    try:
        assert entered.wait(2)
        receipt=queue.close(deadline=0)
        assert receipt['pending_jobs']==1 and not receipt['all_jobs_stopped']
        assert queue.lock_path.exists()
    finally:
        release.set();thread.join(2)
    assert not thread.is_alive() and len(capture_errors)==1
    assert queue.reserved_bytes==0
    assert queue.close()['all_jobs_stopped'] and not queue.lock_path.exists()
    def fake(op,payload):
        began.set(); gate.wait(10); seen.append(payload); return {"value":payload["value"]}
    with patch("deepbom.jobs.engine",fake), AnalysisQueue(tmp/"jobs",producer=producer,max_pending=2) as queue:
        source={"value":1}; first=queue.submit("snapshot",source,configuration={"mode":"test"}); began.wait(5); source["value"]=2
        second=queue.submit("snapshot",{"value":3},configuration={"mode":"test"})
        try: queue.submit("snapshot",{"value":4},configuration={})
        except BufferError: pass
        else: raise AssertionError("queue exceeded capacity")
        queue.cancel(second); assert queue.wait(second)["execution_status"]=="cancelled"
        gate.set(); assert queue.wait(first)["execution_status"]=="completed"
        assert queue.result(first)=={"value":1} and seen==[{"value":1}]
        assert queue.submit("snapshot",{"value":1},configuration={"mode":"test"})==first
        assert not queue.futures
        (tmp/"jobs"/first/"result.json").write_text('{"value":99}')
        try: queue.result(first)
        except ValueError: pass
        else: raise AssertionError("tampered result accepted")
    with AnalysisQueue(tmp/"recover",producer=producer) as queue:
        with patch("deepbom.jobs.engine",side_effect=TimeoutError("probe timed out")):
            job=queue.submit("snapshot",{"value":3},configuration={}); assert queue.wait(job)["execution_status"]=="failed"
        with patch("deepbom.jobs.engine",return_value={"value":3}):
            assert job in queue.recover(); assert queue.wait(job)["execution_status"]=="completed"
        try: queue.submit("snapshot",{"value":2**53+1},configuration={})
        except ValueError: pass
        else: raise AssertionError("lossy integer identity accepted")
    source=tmp/"source"; source.mkdir(); path=source/"probe.onnx"
    path.write_bytes(Path("web/samples/gpu_partition_probe.onnx").read_bytes())
    mark_complete(path,manifest=source/"probe.deepbom-ready.json",producer=producer)
    with AnalysisQueue(tmp/"watch-jobs",producer=producer) as queue:
        watcher=CompletedFileWatcher(source,queue)
        rows=watcher.scan_once(); assert rows[0]["status"]=="submitted",rows
        job=rows[0]["job_sha256"]; assert queue.wait(job)["execution_status"]=="completed",queue.status(job)
        assert watcher.scan_once()[0]["job_sha256"]==job
        # A changed source with an unchanged completed manifest cannot affect
        # the already staged artifact; a new manifest binds a different hash.
        path.write_bytes(b"partial")
        assert watcher.scan_once()[0]["job_sha256"]==job
    with AnalysisQueue(tmp/"fresh-jobs",producer=producer) as queue:
        rows=CompletedFileWatcher(source,queue).scan_once()
        assert rows[0]["status"]=="not_submitted",rows
print("Analysis queue and completed-file checks passed")
