"""One session owner around the existing recorder and optional callbacks."""
from pathlib import Path
import warnings
import math
import time
from .collector import Recorder
from .._native.adapters import framework_of
from .._native.core import atomic_json

PRESETS = {
    "basic": ("metrics",),
    "weights": ("metrics", "weights"),
    "activations": ("metrics", "activations"),
    "gradients": ("metrics", "gradients"),
}


class Monitor:
    def __init__(self, model, *, logdir, optimizer=None, preset="basic",
                 every=100, max_values=100_000, max_state_bytes=64 * 1024 * 1024,
                 errors="record_and_disable", report=False, close_timeout=30,
                 max_pending=2, max_queue_bytes=None):
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {tuple(PRESETS)}")
        if errors not in ("strict", "record_and_disable"):
            raise ValueError("errors must be strict or record_and_disable")
        self.framework = framework_of(model)
        self.recorder = Recorder(model, optimizer=optimizer, logdir=logdir,
                                 observation="auto" if self.framework == "pytorch" and optimizer is not None else "manual",
                                 _observer=self._call,
                                 capture=PRESETS[preset], every=every,
                                 max_values=max_values, max_state_bytes=max_state_bytes)
        self.errors, self.report = errors, report
        if type(close_timeout) not in (int,float) or not math.isfinite(close_timeout) or close_timeout <= 0:
            raise ValueError("close_timeout must be finite and positive")
        self.close_timeout = close_timeout
        self.queue_options = {"max_pending":max_pending,"max_total_bytes":max_queue_bytes,"close_timeout":close_timeout}
        self.managed = None
        self.engine_session = None
        self._engine_token = None
        self.worker_receipt = None
        self.diagnostics, self.handles = [], []
        self.active, self.disabled, self.closed = False, False, False
        self.receipt = None
        self._callback = None

    def __enter__(self):
        if self.active or self.closed:
            raise RuntimeError("Monitor cannot be entered twice")
        self.recorder.__enter__()
        self.active = True
        try:
            import sys
            if sys.platform == "linux":
                from .._native.session import NativeSession
                self.engine_session = NativeSession()
                self.recorder._engine = self.engine_session.request
                from .._native.core import bind_engine
                self._engine_token = bind_engine(self.engine_session.request)
            if "weights" in self.recorder.capture:
                from .managed import ManagedSnapshots
                self.managed = ManagedSnapshots(self.recorder, **self.queue_options)
                self.recorder._snapshot_sink = self.managed.capture
        except BaseException as error:
            # Startup may fail before the managed worker exists. Cleanup is
            # still required, but must preserve the original startup exception.
            try:
                self.recorder.__exit__(type(error),error,error.__traceback__)
            except BaseException as cleanup_error:
                self._diagnose("startup_collector_cleanup", cleanup_error)
            self._unbind_engine()
            if self.engine_session is not None:
                try:
                    self.engine_session.close(time.monotonic()+self.close_timeout)
                except BaseException as cleanup_error:
                    self._diagnose("startup_engine_cleanup", cleanup_error)
            self.active=False
            self.closed=True
            raise
        return self

    def _unbind_engine(self):
        if self._engine_token is not None:
            from .._native.core import reset_engine
            reset_engine(self._engine_token)
            self._engine_token = None

    def _diagnose(self, stage, error):
        if len(self.diagnostics)<32:
            self.diagnostics.append({"stage": stage, "code": type(error).__name__, "message": str(error)[:1000]})
        try:
            warnings.warn(f"DEEPBOM {stage} failed ({type(error).__name__}); see monitor receipt", RuntimeWarning)
        except Warning:
            pass  # A warnings-as-errors policy must not mask the training exception.

    def _call(self, function, *args, **kwargs):
        if not self.active or self.closed:
            raise RuntimeError("Monitor is not active")
        if self.disabled:
            return None
        try:
            if self.managed is not None:
                if getattr(function,"__name__",None)=="finish":
                    from .._process import RunControl, execution_control
                    deadline=time.monotonic()+self.close_timeout
                    with execution_control(RunControl(deadline=deadline)):
                        self.worker_receipt=self.managed.close(deadline-min(5,self.close_timeout/3))
                        return function(*args,**kwargs)
                else:
                    self.managed.drain()
            return function(*args, **kwargs)
        except Exception as error:
            self._diagnose("observation", error)
            self.disabled = True
            if self.errors == "strict":
                raise
            return None

    def record_metrics(self, *args, **kwargs):
        return self._call(self.recorder.record_metrics, *args, **kwargs)

    def record_snapshot(self, *args, **kwargs):
        return self._call(self.recorder.record_snapshot, *args, **kwargs)

    def record_activations(self, *args, **kwargs):
        return self._call(self.recorder.record_activations, *args, **kwargs)

    def before_update(self, *args, **kwargs):
        return self._call(self.recorder.before_update, *args, **kwargs)

    def after_update(self, *args, **kwargs):
        return self._call(self.recorder.after_update, *args, **kwargs)

    def finish(self, *args, **kwargs):
        return self._call(self.recorder.finish, *args, **kwargs)

    def keras_callback(self):
        """Pass to fit inside this context; context owns exception cleanup."""
        if self.framework != "tensorflow":
            raise ValueError("A Keras callback requires a TensorFlow Keras model")
        if self.recorder.capture - {"weights", "metrics"}:
            raise ValueError("fit callbacks observe metrics/weights only; use explicit GradientTape for gradients/activations")
        if self._callback is not None:
            return self._callback
        from .keras_callback import make_callback
        def open_recorder(model):
            if model is not self.recorder.model:
                raise ValueError("Callback model differs from the monitor model")
            if not self.active or self.closed or self.recorder.lifecycle != "running":
                raise RuntimeError("Callback requires a running monitor session")
            return self.recorder
        self._callback = make_callback(open_recorder, observe=self._call)
        return self._callback

    def __exit__(self, typ, value, tb):
        if self.closed:
            return False
        from .._process import RunControl, execution_control
        deadline=time.monotonic()+self.close_timeout
        try:
            with execution_control(RunControl(deadline=deadline)):
                return self._close(typ,value,tb,deadline)
        finally:
            self._unbind_engine()

    def _close(self, typ, value, tb, deadline):
        failures = []
        # Stop capture/hook admission before draining; reserve time for sealing.
        for handle in self.recorder.handles:
            try: handle.remove()
            except BaseException as error:
                failures.append(error)
                self._diagnose("hook_cleanup",error)
        if self.managed is not None:
            try:
                self.worker_receipt=self.managed.close(deadline-min(5,self.close_timeout/3))
            except BaseException as error:
                failures.append(error)
                self._diagnose("worker_close",error)
                try:
                    self.worker_receipt=self.managed.queue.close(deadline=deadline)
                except BaseException as cleanup_error:
                    failures.append(cleanup_error)
                    self._diagnose("worker_cleanup",cleanup_error)
        for handle in self.handles:
            try:
                handle.remove()
            except BaseException as error:
                failures.append(error)
                self._diagnose("hook_cleanup", error)
        try:
            if typ is None and self.disabled and self.recorder.lifecycle == "running":
                self.recorder.finish("interrupted", origin="observation_disabled")
            self.recorder.__exit__(typ, value, tb)
            for error in getattr(self.recorder, "cleanup_errors", ()):
                failures.append(error)
                self._diagnose("collector_close", error)
        except BaseException as error:
            failures.append(error)
            self._diagnose("collector_close", error)
        finally:
            # Even if finish/flush failed, release the writer and collector.
            if not self.recorder._closed:
                try:
                    self.recorder.__exit__(typ or RuntimeError, value, tb)
                except BaseException as error:
                    failures.append(error)
                    self._diagnose("collector_cleanup", error)
        report_path = None
        if self.report and not self.diagnostics:
            try:
                from .viewer import write_report
                report_path = str(write_report(self.recorder.logdir, self.recorder.logdir / "report.html"))
            except BaseException as error:
                failures.append(error)
                self._diagnose("report", error)
        self._unbind_engine()
        if self.engine_session is not None:
            try:
                self.engine_session.close(deadline)
            except BaseException as error:
                failures.append(error)
                self._diagnose("engine_close",error)
                try:
                    self.engine_session.close(deadline)
                except BaseException as cleanup_error:
                    failures.append(cleanup_error)
                    self._diagnose("engine_cleanup",cleanup_error)
        partial = self.managed is not None and self.managed.counts["completed"] != self.managed.counts["requested"]
        self.receipt = {
            "schema": "deepbom.monitor_receipt.v1",
            "run_id": self.recorder.run_id, "logdir": str(self.recorder.logdir),
            "collection_status": "incomplete" if self.diagnostics or partial else "completed",
            "training_lifecycle": self.recorder.lifecycle,
            "capture": sorted(self.recorder.capture), "errors": self.diagnostics,
            "report": report_path, "pending_jobs": self.worker_receipt.get("pending_jobs",0) if self.worker_receipt else len(self.managed.jobs) if self.managed else 0,
            "analysis": {"counts":self.managed.counts,"queue":self.worker_receipt} if self.managed else None,
            "boundary": "Collector completion is not training success. Internal activations and task semantics are never inferred.",
        }
        self.closed = True
        self.active = False
        try:
            atomic_json(self.recorder.logdir / "monitor.json", self.receipt)
        except BaseException as error:
            failures.append(error)
            self.receipt["collection_status"] = "incomplete"
            self._diagnose("receipt", error)
        if failures and typ is None:
            interruption = next((error for error in failures if not isinstance(error, Exception)), None)
            if interruption is not None:
                raise interruption
            if self.errors == "strict":
                raise failures[0]
        return False


def monitor(model, **options):
    return Monitor(model, **options)
