"""Optional training observations. Never owns backward, optimizer or replay."""

from .collector import watch, Recorder


def recover(directory, **options):
    from .recovery import recover as run
    return run(directory, **options)


def lightning_callback(model, **options):
    from .lightning import lightning_callback as create
    return create(model, **options)


def callback(*, logdir, **options):
    from .keras_callback import make_callback
    if set(options.get("capture", ("metrics",))) - {"metrics", "weights"}:
        raise ValueError("fit callbacks capture metrics and selected weights; use explicit GradientTape observations for gradients/activations")
    def open_recorder(model):
        evidence = Recorder(model, optimizer=model.optimizer, logdir=logdir, observation="manual", **options)
        evidence.__enter__()
        return evidence
    return make_callback(open_recorder, close=lambda evidence: evidence.__exit__(None, None, None))


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
