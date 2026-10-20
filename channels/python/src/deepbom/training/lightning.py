"""Optional Lightning adapter; the existing Monitor/Recorder own all evidence."""
from importlib.metadata import version


def lightning_callback(model, *, logdir, preset="basic", every=100,
                       max_values=100_000, max_state_bytes=64 * 1024 * 1024,
                       errors="record_and_disable", report=False):
    """Observe an explicitly selected model with one automatic optimizer.

    This adapter does not step, backward, unscale, or clip. Gradient observations
    are after Lightning unscales and before clipping. A returned step does not
    assert a weight change; skipped steps without a post-hook remain unfinished.
    Distributed strategies and manual multi-optimizer loops require explicit
    Recorder integration and are not silently reduced to a single-writer run.
    """
    if version("lightning") != "2.6.6":
        raise RuntimeError("Lightning adapter is qualified for lightning==2.6.6")
    from lightning.pytorch.callbacks import Callback
    from .session import Monitor

    class EvidenceCallback(Callback):
        def __init__(self):
            self.session = None
            self.token = None

        def on_fit_start(self, trainer, pl_module):
            if trainer.world_size != 1 or len(trainer.optimizers) != 1 or not pl_module.automatic_optimization:
                raise ValueError("Lightning observation requires one process and one automatic optimizer")
            if str(trainer.precision) not in ("32-true", "bf16-mixed"):
                raise ValueError("Qualified Lightning precision modes are 32-true and bf16-mixed")
            self.session = Monitor(model, logdir=logdir, preset=preset, every=every,
                                   max_values=max_values, max_state_bytes=max_state_bytes,
                                   errors=errors, report=report)
            self.session.__enter__()
            optimizer = trainer.optimizers[0]
            self.session.recorder.optimizer = optimizer
            try:
                def after_step(opt, args, kwargs):
                    if self.token is not None:
                        self.session.after_update(self.token, optimizer=opt,
                            at={"step":int(trainer.global_step), "phase":"train"})
                        self.token = None
                self.session.handles.append(optimizer.register_step_post_hook(after_step))
                self.session._call(self.session.recorder.event, "training_begin", {},
                    {"origin":"lightning_callback", "precision":str(trainer.precision),
                     "accumulate_grad_batches":trainer.accumulate_grad_batches})
            except BaseException as error:
                self.session.__exit__(type(error), error, error.__traceback__)
                raise

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            if not self.session or self.session.closed:
                return
            self.token = self.session.before_update(optimizer=optimizer,
                at={"step":int(trainer.global_step), "epoch":int(trainer.current_epoch), "phase":"train"},
                gradient_stage="lightning_after_unscale_before_clip",
                context={"precision":str(trainer.precision), "accumulate_grad_batches":trainer.accumulate_grad_batches})

        def _metrics(self, trainer, phase, batch, dataloader_idx=None):
            if not self.session or self.session.closed:
                return
            # These are explicitly logged framework metrics, not guessed labels,
            # task outputs or independently recomputed evaluation measures.
            at = {"step":int(trainer.global_step), "epoch":int(trainer.current_epoch), "batch":batch, "phase":phase}
            if dataloader_idx is not None:
                at["dataloader_idx"] = dataloader_idx
            self.session.record_metrics(dict(trainer.logged_metrics),
                at=at,
                definition="lightning.logged_metrics snapshot; original framework reductions and update cadence retained")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self._metrics(trainer, "train", batch_idx)

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            self._metrics(trainer, "sanity_check" if trainer.sanity_checking else "validation", batch_idx, dataloader_idx)

        def on_train_end(self, trainer, pl_module):
            if self.session and not self.session.closed:
                self.session.finish("ended", origin="lightning_on_train_end")

        def on_fit_end(self, trainer, pl_module):
            if self.session:
                self.session.__exit__(None, None, None)

        def on_exception(self, trainer, pl_module, exception):
            if self.session:
                self.session.__exit__(type(exception), exception, exception.__traceback__)

        def teardown(self, trainer, pl_module, stage):
            if self.session and not self.session.closed:
                self.session.__exit__(None, None, None)

    return EvidenceCallback()
