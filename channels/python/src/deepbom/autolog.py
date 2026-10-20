"""Explicit, removable automatic callback attachment for qualified Keras fit."""
import inspect
from importlib.metadata import version
from pathlib import Path
import threading
import uuid
import weakref

_LOCK = threading.RLock()
_REGISTRATIONS = {}


class Autolog:
    def __init__(self, *, framework, logdir, **options):
        if framework != "keras":
            raise ValueError("autolog currently supports keras; use lightning_callback or monitor for other loops")
        if version("wrapt") != "2.5.0":
            raise RuntimeError("autolog requires the qualified wrapt==2.5.0 wrapper API")
        if options.get("preset", "basic") not in ("basic", "weights"):
            raise ValueError("Keras fit autolog collects logged metrics and selected weights only")
        self.framework, self.logdir, self.options = framework, Path(logdir), dict(options)
        self.enabled, self.handle, self.target = True, None, None
        self.running = weakref.WeakSet()
        self.receipts = []
        self.skipped_active_collectors = 0
        import wrapt
        with _LOCK:
            if framework in _REGISTRATIONS:
                raise ValueError("Autolog is already enabled; disable its handle before enabling again")
            _REGISTRATIONS[framework] = self
        try:
            wrapt.register_post_import_hook(self._install, "keras")
        except BaseException:
            self.disable()
            raise

    def _install(self, keras):
        import wrapt
        with _LOCK:
            # A queued post-import hook can fire after disable. It must not patch.
            if not self.enabled or _REGISTRATIONS.get(self.framework) is not self:
                return
            # The import name is stable across tensorflow, tensorflow-cpu and
            # platform-specific distributions. Check the runtime actually used.
            if keras.backend.backend() != "tensorflow":
                self.disable()
                raise RuntimeError("Keras autolog requires the TensorFlow backend")
            import tensorflow as tf
            if not version("keras").startswith("3.11.") or not tf.__version__.startswith("2.20."):
                self.disable()
                raise RuntimeError("Keras autolog requires TensorFlow 2.20.x and Keras 3.11.x")
            self.target = keras.Model
            self.handle = wrapt.wrap_function_wrapper(self.target, "fit", self._fit)

    def _fit(self, wrapped, instance, args, kwargs):
        from .training.collector import _ACTIVE
        from .training.session import Monitor
        with _LOCK:
            skip = not self.enabled or instance in self.running or instance in _ACTIVE
            if skip:
                self.skipped_active_collectors += int(self.enabled)
            else:
                self.running.add(instance)
        if skip:
            return wrapped(*args, **kwargs)
        session = None
        try:
            bound = inspect.signature(wrapped).bind_partial(*args, **kwargs)
            callbacks = bound.arguments.get("callbacks")
            if callbacks is not None and not isinstance(callbacks, (list, tuple)):
                # Preserve Keras' own handling for callback containers we have not
                # qualified; do not mutate an arbitrary user object.
                raise TypeError("Autolog callbacks must be a list, tuple or None")
            session = Monitor(instance, optimizer=getattr(instance, "optimizer", None),
                              logdir=self.logdir / uuid.uuid4().hex, **self.options)
            with session:
                bound.arguments["callbacks"] = [*(callbacks or ()), session.keras_callback()]
                return wrapped(*bound.args, **bound.kwargs)
        finally:
            with _LOCK:
                self.running.discard(instance)
            if session and session.receipt:
                self.receipts.append(session.receipt)
                del self.receipts[:-100]

    def disable(self):
        import wrapt
        with _LOCK:
            self.enabled = False
            if _REGISTRATIONS.get(self.framework) is self:
                del _REGISTRATIONS[self.framework]
            if self.handle is not None:
                # wrapt removes only our handle, including inside another wrapt
                # wrapper. It never overwrites a later third-party replacement.
                wrapt.unwrap_object(self.target, "fit", self.handle, missing_ok=True)
                self.handle = None

    def __enter__(self):
        if not self.enabled:
            raise RuntimeError("Autolog handle is disabled")
        return self

    def __exit__(self, *exception):
        self.disable()
        return False


def autolog(*, framework="keras", logdir, **options):
    return Autolog(framework=framework, logdir=logdir, **options)
