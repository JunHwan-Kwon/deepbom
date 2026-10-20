"""Private single-thread engine session owned by Monitor, with common dispatch.

Session reuse removes per-flush engine startup. It neither changes Training IR
calculation nor runs asynchronously against a live framework model.
"""
import os
from pathlib import Path
import tempfile
import threading
import time
from .._engine import _verified_engine, _strict_json
from .._process import RunControl, execution_control, remaining_seconds, run_process


class NativeSession:
    def __init__(self):
        self.owner = threading.get_ident()
        self.owner_pid = os.getpid()
        self.control = RunControl()
        self.done = threading.Event()
        self.failure = None
        self.sequence = 0
        self.closed = False
        self.temp = tempfile.TemporaryDirectory(prefix="deepbom-session-")
        self.path = Path(self.temp.name)
        try:
            executable, assets = _verified_engine()
            environment = os.environ.copy()
            if assets is not None: environment["DEEPBOM_RUNTIME_ASSET_DIR"] = str(assets)
            def serve():
                try:
                    with execution_control(self.control):
                        result = run_process([str(executable), "evidence-native", "--serve", str(self.path)],
                            environment=environment, timeout=365*86400, file_limit=128*1024*1024,
                            memory_bytes=None)
                        if not self.closed:
                            self.failure = RuntimeError(result["diagnostic"] or "Native session exited unexpectedly")
                except BaseException as error:
                    if not self.closed: self.failure = error
                finally: self.done.set()
            self.thread = threading.Thread(target=serve, name="deepbom-control", daemon=True)
            self.thread.start()
            identity = self._receive("ready.json", 30)
            if identity.get("schema") != "deepbom.native_session.v1" or not identity.get("wasm_sha256"):
                raise RuntimeError("Native session did not complete its WASM handshake")
            self.identity = identity
        except BaseException as error:
            try:
                self.close(time.monotonic()+1)
            except BaseException as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(f"Native session cleanup also failed: {type(cleanup_error).__name__}")
            raise

    def _receive(self, name, timeout):
        deadline = time.monotonic()+remaining_seconds(timeout)
        path = self.path/name
        while not path.exists():
            if self.done.is_set(): raise self.failure or RuntimeError("Native session stopped before reply")
            if time.monotonic() >= deadline:
                self.control.cancelled.set()
                raise TimeoutError("Native session request exceeded its deadline")
            time.sleep(min(.002, max(0, deadline-time.monotonic())))
        if path.stat().st_size > 128*1024*1024: raise ValueError("Native session response exceeds its budget")
        value = _strict_json(path, label="native session")
        path.unlink()
        return value

    def request(self, operation, payload, timeout=300):
        if self.closed or os.getpid() != self.owner_pid or threading.get_ident() != self.owner:
            raise RuntimeError("Native session requires its active owner thread")
        from .core import atomic_json, _json_binary64
        from ..jobs import _json_primitives
        envelope = {"sequence":self.sequence, "request":{"operation":operation,"payload":payload}}
        _json_primitives(envelope)
        atomic_json(self.path/"request.json", envelope, compact=True, max_bytes=128*1024*1024)
        reply = self._receive("result.json", timeout)
        if reply.get("sequence") != self.sequence: raise ValueError("Stale native session reply")
        self.sequence += 1
        if "error" in reply: raise ValueError(reply["error"]["message"])
        return _json_binary64(reply["result"])

    def close(self, deadline):
        if os.getpid() != self.owner_pid:
            raise RuntimeError("Native session cannot be closed by an inherited process")
        self.closed = True
        self.control.cancelled.set()
        if hasattr(self, "thread"):
            self.thread.join(max(0, deadline-time.monotonic()))
        if not hasattr(self, "thread") or self.done.is_set():
            self.temp.cleanup()
        else:
            raise TimeoutError("Native session has not stopped; temporary inputs retained")
