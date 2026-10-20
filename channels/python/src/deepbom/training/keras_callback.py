"""One callback event mapping shared by the explicit and automatic entry points."""

def make_callback(open_recorder, *, observe=None, close=None):
    import keras

    class EvidenceCallback(keras.callbacks.Callback):
        def _call(self, method, *args, **kwargs):
            fn = getattr(self.evidence, method)
            return observe(fn, *args, **kwargs) if observe else fn(*args, **kwargs)

        def on_train_begin(self, logs=None):
            if getattr(self, "started", False):
                raise RuntimeError("Use a new evidence callback/session for each fit invocation")
            self.started = True
            self.evidence = open_recorder(self.model)
            self.epoch, self.fitting = 0, True
            self._call("event", "training_begin", {}, {"origin":"keras_callback"})

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch

        def on_train_batch_end(self, batch, logs=None):
            self._call("record_metrics", logs or {}, at={"phase":"train", "epoch":self.epoch, "batch":batch},
                       definition="keras_callback_logs; framework reductions retained")
            if "weights" in self.evidence.capture and batch % self.evidence.every == 0:
                self._call("observe_snapshot", at={"phase":"train", "epoch":self.epoch, "batch":batch, "boundary":"batch_end"})

        def on_test_batch_end(self, batch, logs=None):
            if not hasattr(self, "evidence"):
                raise RuntimeError("Fit evidence callback has not started; record standalone evaluation explicitly")
            self._call("record_metrics", logs or {}, at={"phase":"validation" if self.fitting else "evaluation", "epoch":self.epoch, "batch":batch},
                       definition="keras_callback_logs; framework reductions retained")

        def on_train_end(self, logs=None):
            self.fitting = False
            try:
                self._call("finish", "ended", origin="keras_on_train_end")
            finally:
                self.close()

        def close(self):
            """Call in finally if fit fails; a monitor context handles this itself."""
            if close and hasattr(self, "evidence") and not self.evidence._closed:
                close(self.evidence)

    return EvidenceCallback()
