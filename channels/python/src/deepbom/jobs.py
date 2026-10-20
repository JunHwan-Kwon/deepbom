"""Bounded single-writer jobs. Inputs and completed results are content-addressed."""
from concurrent.futures import CancelledError, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import re
import threading
import time
import math
from ._process import RunControl, ProcessCancelled, execution_control
from ._native.core import atomic_json, content_hash, engine, read_json
from ._ownership import WriterOwnership

_NATIVE_OPERATIONS = {"snapshot", "inspect_native", "tensor_batch", "optimization_report", "validate_run", "correspondence", "mapped_comparison"}
_OPERATIONS = _NATIVE_OPERATIONS | {"evidence_workflow", "audit_file"}
_MAX_RESULT_BYTES = 128 * 1024 * 1024


class AnalysisQueue:
    def __init__(self, directory, *, producer, max_pending=2, max_request_bytes=16*1024*1024,
                 max_total_bytes=None, close_timeout=30, max_storage_bytes=1024*1024*1024):
        if type(max_pending) is not int or not 1 <= max_pending <= 64 or type(max_request_bytes) is not int or not 1 <= max_request_bytes <= 128*1024*1024:
            raise ValueError("Invalid queue bounds")
        if not isinstance(producer, dict) or set(producer) != {"name", "version", "code_sha256"} or not all(isinstance(v,str) and 0 < len(v) <= 256 for v in producer.values()):
            raise ValueError("Provide bounded producer name/version and implementation code_sha256")
        if not re.fullmatch(r"[a-f0-9]{64}", producer["code_sha256"]):
            raise ValueError("Invalid producer code SHA-256")
        self.directory = Path(directory).resolve(); self.directory.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.directory / ".worker.lock"
        self._ownership = WriterOwnership(self.lock_path, purpose="analysis_queue")
        self.producer = dict(producer)
        self.maximum, self.max_request_bytes = max_pending, max_request_bytes
        self.max_total_bytes = max_total_bytes if max_total_bytes is not None else max_pending * max_request_bytes
        if type(self.max_total_bytes) is not int or self.max_total_bytes < max_request_bytes:
            self._ownership.close()
            raise ValueError("Aggregate byte budget must hold at least one request")
        if type(close_timeout) not in (int, float) or not math.isfinite(close_timeout) or close_timeout <= 0 or type(max_storage_bytes) is not int or max_storage_bytes < max_request_bytes:
            self._ownership.close()
            raise ValueError("Invalid shutdown or storage budget")
        self.close_timeout, self.max_storage_bytes = close_timeout, max_storage_bytes
        self.lock, self.closed = threading.RLock(), False
        self.futures, self.cancelled = {}, set()
        self.controls, self.reservations = {}, {}
        self.capturing = self.reserved_bytes = self.peak_reserved_bytes = 0
        self.close_receipt = None
        try:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deepbom-analysis")
        except BaseException:
            self._ownership.close()
            raise

    def _path(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r"[a-f0-9]{64}", job_id):
            raise ValueError("Invalid job identity")
        return self.directory / job_id

    def _read(self, path, maximum=None):
        if path.stat().st_size > (maximum or self.max_request_bytes):
            raise ValueError("Queued document exceeds its byte budget")
        return read_json(path, label="analysis job")

    def _request(self, job_id):
        request = self._read(self._path(job_id) / "request.json")
        if content_hash(request) != job_id or request.get("producer") != self.producer or request.get("operation") not in _OPERATIONS:
            raise ValueError("Queued immutable input identity changed")
        return request

    def _receipt(self, job_id, status, **details):
        body = {"schema":"deepbom.analysis_job.v1", "job_sha256":job_id, "execution_status":status,
                "producer":self.producer, "validation_status":"not_implied_by_execution", **details}
        # Status is the mutable index; request and result documents are immutable.
        # Cancellation and completion are competing writers to the progress
        # index. Serialize them so cancellation_requested cannot overwrite the
        # final failed/cancelled receipt after the process has stopped.
        with self.lock:
            atomic_json(self._path(job_id) / "status.json", body, replace=True)
        return body

    def submit(self, operation, payload, *, configuration):
        return self.submit_capture(operation, lambda: payload, configuration=configuration)

    def submit_capture(self, operation, capture, *, configuration, copy_bytes=0):
        """Reserve queued+running+capture capacity BEFORE invoking a producer.

        copy_bytes accounts for independent raw snapshot bytes in addition to
        the bounded encoded request. This is not an engine RSS guarantee.
        """
        if type(copy_bytes) is not int or copy_bytes < 0 or not callable(capture):
            raise ValueError("Invalid immutable capture reservation")
        reservation = self.max_request_bytes + copy_bytes
        with self.lock:
            if self.closed: raise RuntimeError("Queue is closed")
            if len(self.futures) + self.capturing >= self.maximum or self.reserved_bytes + reservation > self.max_total_bytes:
                raise BufferError("Analysis queue count/aggregate byte budget exhausted before capture")
            # No automatic deletion of completed evidence. Stop admission at quota.
            used = sum(p.stat().st_size for p in self.directory.rglob("*") if p.is_file())
            if used + (len(self.futures)+self.capturing+1)*(self.max_request_bytes+_MAX_RESULT_BYTES+65536) > self.max_storage_bytes:
                raise BufferError("Analysis storage quota exhausted; retain/export evidence or choose a new directory")
            self.capturing += 1
            self.reserved_bytes += reservation
            self.peak_reserved_bytes = max(self.peak_reserved_bytes, self.reserved_bytes)
        try:
            return self._submit_reserved(operation, capture(), configuration=configuration, reservation=reservation)
        finally:
            with self.lock:
                self.capturing -= 1
                self.reserved_bytes -= reservation

    def _submit_reserved(self, operation, payload, *, configuration, reservation):
        if operation not in _OPERATIONS or not isinstance(configuration,dict):
            raise ValueError("Unsupported queued operation or non-object configuration")
        request = {"schema":"deepbom.analysis_job_request.v1", "producer":self.producer,
                   "operation":operation, "configuration":configuration, "payload":payload}
        # Limit while serializing, before assembling the detached JSON copy.
        _json_primitives(request, budget=[1_000_000, self.max_request_bytes])
        chunks, size = [], 0
        for chunk in json.JSONEncoder(allow_nan=False, ensure_ascii=False).iterencode(request):
            size += len(chunk.encode("utf-8"))
            if size > self.max_request_bytes: raise ValueError("Queue request exceeds its copy budget")
            chunks.append(chunk)
        frozen = json.loads("".join(chunks)); job_id = content_hash(frozen)
        with self.lock:
            if self.closed: raise RuntimeError("Queue is closed")
            if job_id in self.futures: return job_id
            path = self._path(job_id)
            if (path / "status.json").exists() and self.status(job_id)["execution_status"] == "completed":
                self.result(job_id)  # Detect corruption before a cache hit.
                return job_id
            path.mkdir(exist_ok=True)
            # Use the same serialization measured before copying. Pretty-print
            # indentation can otherwise make an accepted request unreadable.
            atomic_json(path / "request.json", frozen, compact=True)
            self.cancelled.discard(job_id)
            self.controls[job_id] = RunControl()
            self.reservations[job_id] = reservation
            self.reserved_bytes += reservation  # transfer from capture at return
            try:
                self._receipt(job_id, "queued")
                future = self.executor.submit(self._execute, job_id)
            except BaseException:
                self.controls.pop(job_id, None)
                self.reserved_bytes -= self.reservations.pop(job_id, 0)
                raise
            self.futures[job_id] = future
            future.add_done_callback(lambda done: self._completed(job_id, done))
        return job_id

    def _completed(self, job_id, future):
        with self.lock:
            if self.futures.get(job_id) is future: self.futures.pop(job_id, None)
            self.reserved_bytes -= self.reservations.pop(job_id, 0)
            self.controls.pop(job_id, None)
            self.cancelled.discard(job_id)

    def _execute(self, job_id):
        path = self._path(job_id)
        try:
            with self.lock:
                if job_id in self.cancelled: return self._receipt(job_id, "cancelled")
                self._receipt(job_id, "running")
            with execution_control(self.controls[job_id]):
                request = self._request(job_id)
                if request["operation"] == "evidence_workflow":
                    from .api import evidence_workflow
                    result = evidence_workflow(request["payload"])
                elif request["operation"] == "audit_file":
                    from .api import audit
                    payload = request["payload"]
                    if not isinstance(payload,dict) or set(payload) != {"path", "sha256", "scan"}:
                        raise ValueError("audit_file requires path, sha256 and scan")
                    result = audit(payload["path"], expected_sha256=payload["sha256"], scan=payload["scan"])
                else:
                    result = engine(request["operation"], request["payload"])
                _json_primitives(result, budget=[1_000_000, _MAX_RESULT_BYTES])
                result_digest = content_hash(result)
            with self.lock:
                if job_id in self.cancelled:
                    return self._receipt(job_id, "cancelled", reason="result_discarded_after_running_operation")
                # Never mark a result completed if result() cannot read it.
                atomic_json(path / "result.json", result, compact=True)
                completion = {"job_sha256":job_id, "producer":self.producer,
                              "result_document_sha256":result_digest, "digest_scope":"canonical_complete_result_document"}
                atomic_json(path / "completion.json", completion, compact=True)
                return self._receipt(job_id, "completed", result_document_sha256=completion["result_document_sha256"], digest_scope=completion["digest_scope"])
        except ProcessCancelled:
            return self._receipt(job_id, "cancelled", reason="engine_process_group_cancelled")
        except Exception as error:
            from .api import DeepBomTimeout
            return self._receipt(job_id, "failed", error={"code":getattr(error, "code", type(error).__name__), "stage":"analysis", "message":str(error)[:2000]},
                                 retryable=isinstance(error, (TimeoutError, DeepBomTimeout, BlockingIOError, InterruptedError)))

    def _status_index(self, job_id):
        receipt = self._read(self._path(job_id) / "status.json", 64*1024)
        if receipt.get("job_sha256") != job_id or receipt.get("producer") != self.producer or receipt.get("schema") != "deepbom.analysis_job.v1":
            raise ValueError("Job receipt identity mismatch")
        return receipt

    def poll(self, job_id):
        """Read a bounded progress index, not validated analysis evidence.

        result()/status() still validate immutable input identity. Monitor polls
        frequently without re-decoding the entire snapshot on the trainer thread.
        """
        with self.lock:
            return self._status_index(job_id)

    def status(self, job_id):
        # Validation may start the shared engine. Do not hold the queue lock
        # across hashing: shutdown must be able to signal admitted work.
        self._request(job_id)
        with self.lock:
            return self._status_index(job_id)

    def result(self, job_id):
        receipt = self.status(job_id)
        if receipt["execution_status"] != "completed": raise RuntimeError("Job has not completed")
        result = self._read(self._path(job_id) / "result.json", _MAX_RESULT_BYTES)
        if content_hash(result) != receipt["result_document_sha256"]: raise ValueError("Stored analysis result digest mismatch")
        return result

    def wait(self, job_id, timeout=None):
        with self.lock: future = self.futures.get(job_id)
        if future:
            try: future.result(timeout=timeout)
            except CancelledError: pass
        return self.status(job_id)

    def cancel(self, job_id):
        self._request(job_id)
        with self.lock:
            return self._cancel_admitted(job_id)

    def _cancel_admitted(self, job_id):
        # Called under the queue lock. Shutdown only visits its owned admitted
        # jobs; progress receipts are not a substitute for validated results.
        with self.lock:
            receipt = self._status_index(job_id)
            future = self.futures.get(job_id)
            if future and not future.done():
                self.cancelled.add(job_id)
                self.controls[job_id].cancelled.set()
                if future.cancel(): self._receipt(job_id, "cancelled")
                else: self._receipt(job_id, "cancellation_requested", reason="engine_process_group_termination_requested")
            elif receipt["execution_status"] in ("queued", "running", "cancellation_requested"):
                self._receipt(job_id, "cancelled")
            return self._status_index(job_id)

    def recover(self):
        """Recover complete immutable inputs under exclusive writer ownership.

        A durable completion intent is validated before acknowledging a result.
        A result without that intent is preserved separately and recomputed.
        No partially written input or previously cancelled work is resumed.
        """
        recovered = []
        for path in sorted(self.directory.glob("*/request.json")):
            request = self._request(path.parent.name)
            job = path.parent.name
            with self.lock:
                if job in self.futures: continue
                if not (path.parent / "status.json").exists():
                    self._receipt(job, "queued", reason="recovered_immutable_request")
                receipt = self.status(job)
            state = receipt["execution_status"]
            if state == "completed":
                self.result(job)
                continue
            if state == "cancellation_requested":
                self.cancel(job)
            elif state in ("queued", "running") or state == "failed" and receipt.get("retryable") is True:
                completion_path = path.parent / "completion.json"
                if completion_path.exists():
                    intent = self._read(completion_path, 65536)
                    result = self._read(path.parent / "result.json", _MAX_RESULT_BYTES)
                    expected = {"job_sha256":job, "producer":self.producer,
                                "result_document_sha256":content_hash(result), "digest_scope":"canonical_complete_result_document"}
                    if intent != expected: raise ValueError("Recovered analysis completion identity mismatch")
                    self._receipt(job, "completed", result_document_sha256=intent["result_document_sha256"], digest_scope=intent["digest_scope"])
                    recovered.append(job)
                    continue
                orphan = path.parent / "result.json"
                if orphan.exists():
                    # Never overwrite an unacknowledged result from a failed
                    # attempt, even for a nominally deterministic analyzer.
                    result = self._read(orphan, _MAX_RESULT_BYTES)
                    archived = path.parent / "unacknowledged" / (content_hash(result)+".json")
                    atomic_json(archived, result, compact=True)
                    orphan.unlink()
                try: recovered.append(self.submit(request["operation"], request["payload"], configuration=request["configuration"]))
                except BufferError: break
        return recovered

    def close(self, *, deadline=None):
        started = time.monotonic()
        if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise ValueError("Shutdown deadline must be a finite monotonic timestamp")
        deadline = min(deadline if deadline is not None else math.inf, started+self.close_timeout)
        with self.lock:
            if self.closed and self.close_receipt is not None and self.close_receipt["all_jobs_stopped"]:
                return self.close_receipt
            self.closed = True
            futures = list(self.futures.values())
            # Reserve a portion of this same budget for cancellation/reaping.
            drain_deadline = deadline-min(0.5, self.close_timeout/4)
            for control in self.controls.values(): control.deadline = drain_deadline
        failures = []
        try:
            if futures: wait(futures, timeout=max(0, drain_deadline-time.monotonic()))
        except BaseException as error:
            failures.append(error)
        # A Ctrl+C during drain must still stop admitted work. Signal controls
        # before any fallible status I/O, then attempt every cancellation.
        with self.lock:
            for control in self.controls.values(): control.cancelled.set()
            for job in list(self.futures):
                try: self._cancel_admitted(job)
                except BaseException as error: failures.append(error)
        try:
            if futures: wait(futures, timeout=max(0, deadline-time.monotonic()))
        except BaseException as error:
            failures.append(error)
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except BaseException as error:
            failures.append(error)
        with self.lock:
            pending = sum(not f.done() for f in futures) + self.capturing
        self.close_receipt = {"pending_jobs":pending, "elapsed_seconds":time.monotonic()-started,
                              "all_jobs_stopped":pending == 0, "peak_reserved_bytes":self.peak_reserved_bytes}
        if not pending:
            try: self._ownership.close()
            except BaseException as error: failures.append(error)
        # A surviving controller keeps the lock. Never allow a new writer over it.
        if failures:
            raise next((error for error in failures if not isinstance(error, Exception)), failures[0])
        return self.close_receipt

    def __enter__(self): return self
    def __exit__(self, *exception):
        try: self.close()
        except BaseException:
            if exception[0] is None: raise
        return False


def _json_primitives(value, depth=0, budget=None):
    if budget is None: budget = [1_000_000, 128*1024*1024]
    budget[0] -= 1
    if depth > 128 or budget[0] < 0: raise ValueError("JSON input structure exceeds its budget")
    def charge(size):
        budget[1] -= size
        if budget[1] < 0: raise ValueError("Queue request exceeds its pre-copy JSON byte budget")
    if isinstance(value, dict):
        if any(not isinstance(k,str) for k in value): raise ValueError("JSON keys must be strings")
        charge(2 + max(0, len(value)-1)*2 + len(value)*2)
        for key, item in value.items():
            _json_primitives(key, depth+1, budget)
            _json_primitives(item, depth+1, budget)
    elif isinstance(value, (list,tuple)):
        charge(2 + max(0, len(value)-1)*2)
        for item in value: _json_primitives(item, depth+1, budget)
    elif type(value) is str:
        charge(2)
        # Count the standard serializer's escaping in bounded pieces before
        # it can allocate one very large escaped string or detached request.
        for offset in range(0, len(value), 4096):
            charge(len(json.dumps(value[offset:offset+4096], ensure_ascii=False).encode("utf-8"))-2)
    elif type(value) is int and abs(value) > 2**53-1:
        raise ValueError("Exact large integers must use decimal strings")
    elif value is not None and type(value) not in (str,bool,int,float):
        raise TypeError("Job inputs must be JSON values, never live models or tensors")
    else:
        charge(len(json.dumps(value, allow_nan=False).encode("utf-8")))
