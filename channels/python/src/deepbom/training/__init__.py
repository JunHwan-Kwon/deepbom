"""Optional training observations. Never owns backward, optimizer or replay."""

from .collector import watch, Recorder


def callback(*, logdir, **options):
    import keras

    if set(options.get("capture", ("metrics",))) - {"metrics", "weights"}:
        raise ValueError(
            "fit callbacks capture metrics and selected end-of-batch weights; use explicit GradientTape observations for gradients/activations"
        )

    class EvidenceCallback(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.evidence = Recorder(
                self.model,
                optimizer=self.model.optimizer,
                logdir=logdir,
                observation="manual",
                **options
            )
            self.evidence.__enter__()
            self.evidence.event("training_begin", {}, {"origin": "keras_callback"})

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch

        def on_train_batch_end(self, batch, logs=None):
            self.evidence.record_metrics(
                logs or {},
                at={"phase": "train", "epoch": self.epoch, "batch": batch},
                definition="keras_callback_logs; framework reductions retained",
            )
            if "weights" in self.evidence.capture and batch % self.evidence.every == 0:
                self.evidence.record_snapshot(
                    at={
                        "phase": "train",
                        "epoch": self.epoch,
                        "batch": batch,
                        "boundary": "batch_end",
                    }
                )

        def on_test_batch_end(self, batch, logs=None):
            self.evidence.record_metrics(
                logs or {},
                at={"phase": "validation", "batch": batch},
                definition="keras_callback_logs",
            )

        def on_train_end(self, logs=None):
            self.evidence.finish("ended", origin="keras_on_train_end")
            self.evidence.__exit__(None, None, None)

        def close(self):
            """Call from finally if fit aborts before on_train_end."""
            if hasattr(self, "evidence") and not self.evidence._closed:
                self.evidence.__exit__(None, None, None)

    return EvidenceCallback()


def import_logs(*args, **kwargs):
    from .integrations import import_logs as run

    return run(*args, **kwargs)


def export(*args, **kwargs):
    from .integrations import export as run

    return run(*args, **kwargs)


def select_result(directory, *, snapshot_sha256, output):
    """Select a recorded state without implying completed or successful training."""
    from .integrations import read_run
    from .._native.core import engine, atomic_json

    document = engine(
        "select_result",
        {"bundle": read_run(directory), "snapshot_sha256": snapshot_sha256},
    )
    atomic_json(output, document)
    return document
