#!/usr/bin/env python3
"""Real CPU framework regression checks. Run with the qualified optional environment."""

import os, sys, hashlib, tempfile, copy, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
PACKAGE_ROOT = os.environ.get(
    "DEEPBOM_TEST_PACKAGE_ROOT", str(ROOT / "channels/python/src")
)
sys.path.insert(0, PACKAGE_ROOT)
import torch, numpy as np, tensorflow as tf, keras
from deepbom._native.core import engine, content_hash, read_json
from deepbom._native.adapters import snapshot
from deepbom.optimization import optimize, load_model, refine, load_evidence
from deepbom.training import watch, Recorder, callback
from deepbom.training.integrations import read_run, export, import_logs
from deepbom.training.viewer import write_report

results = []


def check(name, fn):

    if os.getenv("NATIVE_TEST_GROUP") and os.getenv("NATIVE_TEST_GROUP") not in name:
        return
    fn()
    results.append(name)
    print("PASS", name, flush=True)


def rejected(fn):
    try:
        fn()
    except (ValueError, RuntimeError, TypeError, FileExistsError):
        return
    raise AssertionError("expected rejection")


root = Path(
    os.getenv("NATIVE_TEST_ROOT") or tempfile.mkdtemp(prefix="deepbom-native-tests-")
)
print("ROOT", root, flush=True)
torch.set_num_threads(1)


def cli_rule_files():
    import contextlib
    import io
    from deepbom.optimization.__main__ import main

    folder = root / "cli-rules"
    folder.mkdir(parents=True)
    (folder / "native_cli_fixture.py").write_text(
        "import torch\n"
        "def build_model():\n"
        "    torch.manual_seed(3)\n"
        "    return torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), "
        "torch.nn.BatchNorm2d(4)).eval()\n"
    )
    empty = folder / "empty.json"
    empty.write_text("[]")
    fold = folder / "fold.json"
    fold.write_text('[{"kind":"fold_conv_bn","subject":"0","batchnorm":"1"}]')
    wrong = folder / "object.json"
    wrong.write_text("{}")
    duplicate = folder / "duplicate.json"
    duplicate.write_text('[{"kind":"inverted","kind":"fold_conv_bn"}]')
    sys.path.insert(0, str(folder))
    try:
        for name, rules, status in (
            ("baseline", empty, "no_applicable_transformation"),
            ("folded", fold, "transformed"),
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                assert main([
                    "optimize", "--factory", "native_cli_fixture:build_model",
                    "--rules", str(rules), "--output", str(folder / name),
                ]) == 0
            assert read_json(folder / name / "validation.json")["candidate_status"] == status
            load_model(folder / name)
        for path in (wrong, duplicate):
            rejected(lambda: main([
                "optimize", "--factory", "native_cli_fixture:build_model",
                "--rules", str(path), "--output", str(folder / "rejected"),
            ]))
        assert not (folder / "rejected").exists()
        # Native evidence and engine manifests still require an object by default.
        rejected(lambda: read_json(empty))
    finally:
        sys.path.remove(str(folder))
        sys.modules.pop("native_cli_fixture", None)


check("Native CLI JSON rule arrays, empty baseline and duplicate rejection", cli_rule_files)


def torch_candidates():
    torch.manual_seed(14)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.BatchNorm2d(8), torch.nn.ReLU()
    ).eval()
    x = torch.randn(2, 3, 8, 8)
    original = snapshot(model)["snapshot"]["snapshot_sha256"]
    fold = optimize(
        model, example_args=(x,), output_dir=root / "torch-fold", advanced=True
    )
    torch.testing.assert_close(model(x), fold.model(x), rtol=1e-5, atol=1e-6)
    assert fold.transformations[0]["rule"] == "fold_conv_bn"
    r = refine(fold, rules=[{"kind": "inverted", "subject": "0"}], example_args=(x,))
    assert r.validation["gradient_coverage"]["observed"] == 6
    expected = snapshot(r.model)["snapshot"]["snapshot_sha256"]
    with torch.no_grad():
        next(r.model.parameters()).add_(100)
    r.export(root / "torch-inverted")
    loaded = load_model(root / "torch-inverted")
    assert snapshot(loaded)["snapshot"]["snapshot_sha256"] == expected
    assert snapshot(model)["snapshot"]["snapshot_sha256"] == original
    assert r.parent == fold.snapshot_sha256
    assert int(r.comparison["totals"]["candidate"]["state_elements"]) < int(
        r.comparison["totals"]["baseline"]["state_elements"]
    )
    double = torch.nn.Sequential(torch.nn.Linear(2, 2)).double()
    d = optimize(double, rules=[], output_dir=root / "double")
    assert next(load_model(root / "double").parameters()).dtype == torch.float64
    rejected(
        lambda: optimize(
            model.train(),
            rules=[{"kind": "fold_conv_bn", "subject": "0", "batchnorm": "1"}],
        )
    )
    rejected(lambda: r.export(root / "torch-inverted"))
    tampered = copy.deepcopy(r.evidence)
    tampered["weight_ir"]["source"]["snapshot_sha256"] = "0" * 64
    body = {k: v for k, v in tampered["weight_ir"].items() if k != "weight_ir_sha256"}
    tampered["weight_ir"]["weight_ir_sha256"] = content_hash(body)
    rejected(lambda: engine("validate_bundle", tampered))


check(
    "PyTorch fold/inverted, independent frozen export, dtype, source binding",
    torch_candidates,
)


def keras_candidates():
    keras.utils.set_random_seed(13)
    x = keras.Input((8, 8, 3), name="input")
    y = keras.layers.Conv2D(8, 3, padding="same", name="conv")(x)
    y = keras.layers.BatchNormalization(name="bn")(y)
    model = keras.Model(x, y, name="network")
    sample = tf.ones((2, 8, 8, 3))
    original = snapshot(model)["snapshot"]["snapshot_sha256"]
    fold = optimize(model, example_args=(sample,), output_dir=root / "keras-fold")
    np.testing.assert_allclose(
        model(sample, training=False).numpy(),
        fold.model(sample, training=False).numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    r = optimize(
        model,
        objective="structure",
        example_args=(sample,),
        output_dir=root / "keras-inverted",
        advanced=True,
    )
    assert r.validation["gradient_coverage"]["observed"] == 6
    z = load_model(root / "keras-inverted")
    np.testing.assert_array_equal(z(sample).numpy(), r.model(sample).numpy())
    assert snapshot(model)["snapshot"]["snapshot_sha256"] == original
    # Multiple nested replacements must have stable distinct identities.
    a = keras.Input((8, 8, 3), name="in")
    b = keras.layers.Conv2D(8, 3, padding="same", name="a")(a)
    b = keras.layers.Conv2D(8, 3, padding="same", name="b")(b)
    optimize(
        keras.Model(a, b, name="nested"),
        objective="structure",
        output_dir=root / "keras-nested",
    )


check("Keras fold/inverted, nested state and package round trip", keras_candidates)


def training():
    torch.manual_seed(1)
    base = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.BatchNorm1d(4),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(4, 2),
    )
    a = copy.deepcopy(base)
    b = copy.deepcopy(base)
    data = [torch.randn(3, 4) for _ in range(3)]
    rng = torch.get_rng_state()

    def loop(m, observe):
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        losses = []
        import contextlib

        with (
            watch(
                m,
                optimizer=opt,
                logdir=root / "torch-training",
                capture=("metrics", "gradients", "weights", "activations"),
                chunk_size=3,
            )
            if observe
            else contextlib.nullcontext()
        ) as rec:
            for step, x in enumerate(data):
                opt.zero_grad()
                out = m(x)
                loss = out.square().mean()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
                if rec:
                    rec.record_metrics({"loss": loss}, at={"step": step})
            if rec:
                sha = rec.record_snapshot(at={"step": 3}, advanced=True)
                rec.record_activations(
                    {"3": m(data[0])},
                    snapshot_sha256=sha,
                    at={"step": 3},
                    invocation="extra-forward",
                    context={
                        "mode": "train",
                        "input_identity": "test-fixture-0",
                        "state_boundary": "before_forward",
                    },
                )
                # This extra forward updates BN; compare before the intentional extra call separately.
                rec.finish()
        return losses

    x = loop(a, False)
    torch.set_rng_state(rng)
    y = loop(b, True)
    assert x == y
    # Parameters are exactly equal; BN buffers differ only because of the explicit extra forward.
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    bundle = read_run(root / "torch-training")
    assert bundle["training"]["lifecycle"] == "ended"
    assert len([e for e in bundle["events"] if e["kind"] == "update_returned"]) == 3
    write_report(root / "torch-training", root / "board.html")
    # Metrics-only collector must preserve all state and RNG exactly.
    a = copy.deepcopy(base)
    b = copy.deepcopy(base)

    def one(m, rec=None):
        op = torch.optim.SGD(m.parameters(), lr=0.01)
        op.zero_grad()
        loss = m(data[0]).square().mean()
        loss.backward()
        op.step()
        if rec:
            rec.record_metrics({"loss": loss})

    torch.set_rng_state(rng)
    one(a)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    with Recorder(b, logdir=root / "manual") as rec:
        one(b, rec)
    assert torch.equal(expected_rng, torch.get_rng_state())
    for key, val in a.state_dict().items():
        torch.testing.assert_close(val, b.state_dict()[key], rtol=0, atol=0)
    assert read_run(root / "manual")["training"]["lifecycle"] == "collector_closed"


check(
    "PyTorch collectors, chunk continuity, activations, exact non-interference",
    training,
)


def tensorflow_training():
    keras.utils.set_random_seed(11)
    m = keras.Sequential(
        [keras.Input((4,), name="input"), keras.layers.Dense(2, name="dense")],
        name="tape_model",
    )
    a = keras.models.clone_model(m)
    a.set_weights(m.get_weights())
    x = tf.ones((3, 4))

    def step(model, opt, rec=None):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(model(x) ** 2)
        grads = tape.gradient(loss, model.trainable_variables)
        if rec:
            token = rec.before_update(
                optimizer=opt,
                at={"step": 0},
                loss=loss,
                gradients=grads,
                variables=model.trainable_variables,
            )
        opt.apply_gradients(zip(grads, model.trainable_variables))
        if rec:
            rec.after_update(token, optimizer=opt)
            rec.record_metrics({"loss": loss})

    op = keras.optimizers.SGD(0.01)
    step(a, op)
    op = keras.optimizers.SGD(0.01)
    with watch(
        m,
        optimizer=op,
        observation="manual",
        logdir=root / "tape",
        capture=("metrics", "gradients", "weights"),
    ) as rec:
        step(m, op, rec)
    for x, y in zip(a.get_weights(), m.get_weights()):
        np.testing.assert_array_equal(x, y)
    m.compile(optimizer="sgd", loss="mse")
    cb = callback(logdir=root / "fit", capture=("metrics", "weights"))
    try:
        m.fit(
            np.ones((4, 4), np.float32),
            np.zeros((4, 2), np.float32),
            epochs=1,
            batch_size=2,
            callbacks=[cb],
            verbose=0,
        )
    finally:
        cb.close()
    assert read_run(root / "fit")["training"]["lifecycle"] == "ended"
    rejected(lambda: callback(logdir=root / "unsupported", capture=("gradients",)))


check(
    "GradientTape exact non-interference and real Keras fit callback",
    tensorflow_training,
)


def integrations():
    from mlflow.tracking import MlflowClient

    uri = "sqlite:///" + str(root / "mlflow.db")
    client = MlflowClient(tracking_uri=uri)
    exp = client.create_experiment(
        "native-evidence", artifact_location=str(root / "mlflow-artifacts")
    )
    run = client.create_run(exp)
    client.log_metric(run.info.run_id, "loss", 0.5, step=1)
    r = export(
        root / "torch-training",
        destination="mlflow",
        tracking_uri=uri,
        run_id=run.info.run_id,
    )
    assert r == export(
        root / "torch-training",
        destination="mlflow",
        tracking_uri=uri,
        run_id=run.info.run_id,
    )
    assert client.list_artifacts(run.info.run_id, "deepbom")
    import_logs(
        "mlflow",
        output=root / "import-mlflow",
        tracking_uri=uri,
        run_id=run.info.run_id,
    )
    tb_receipt = export(
        root / "torch-training", destination="tensorboard", logdir=root / "tensorboard"
    )
    import_logs(
        "tensorboard",
        output=root / "import-tb",
        logdir=tb_receipt["event_logdir"],
    )
    assert any(e["kind"] == "metrics" for e in read_run(root / "import-tb")["events"])
    import wandb

    with wandb.init(
        mode="offline", project="deepbom-local-validation", dir=str(root)
    ) as run:
        export(root / "torch-training", destination="wandb", wandb_run=run)


check(
    "MLflow local store, TensorBoard real round trip, W&B offline artifact",
    integrations,
)


def sparse_observations():
    model = torch.nn.Sequential(torch.nn.Embedding(5, 3, sparse=True))
    model.register_parameter("unused", torch.nn.Parameter(torch.ones(1)))
    op = torch.optim.SGD(model.parameters(), lr=0.01)
    with watch(
        model, optimizer=op, logdir=root / "sparse-torch", capture=("gradients",)
    ):
        model(torch.tensor([1, 3])).sum().backward()
        op.step()
    run = read_run(root / "sparse-torch")
    gradients = next(e for e in run["events"] if e["kind"] == "update_attempt")[
        "payload"
    ]["gradients"]
    assert gradients[0]["status"] == "missing_gradient"
    assert gradients[1]["scope"] == "sparse_stored_values_not_dense_gradient"
    model = keras.Sequential(
        [
            keras.Input((2,), dtype="int32", name="input"),
            keras.layers.Embedding(5, 3, name="embedding"),
        ],
        name="sparse",
    )
    op = keras.optimizers.SGD(0.01)
    with watch(
        model,
        optimizer=op,
        observation="manual",
        logdir=root / "sparse-tf",
        capture=("gradients",),
    ) as evidence:
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(model(tf.constant([[1, 3]])))
        gradients = tape.gradient(loss, model.trainable_variables)
        token = evidence.before_update(
            optimizer=op,
            at={"step": 0},
            gradients=gradients,
            variables=model.trainable_variables,
        )
        op.apply_gradients(zip(gradients, model.trainable_variables))
        evidence.after_update(token, optimizer=op)
    run = read_run(root / "sparse-tf")
    g = next(e for e in run["events"] if e["kind"] == "update_attempt")["payload"][
        "gradients"
    ][0]
    assert g["scope"] == "indexed_slices_values_not_dense_gradient"


check(
    "Missing and sparse PyTorch gradients, TensorFlow IndexedSlices",
    sparse_observations,
)


def contract_edges():
    from deepbom.training import select_result

    run = read_run(root / "torch-training")
    state = [e for e in run["events"] if e["kind"] == "model_snapshot"][-1]["payload"][
        "evidence"
    ]["snapshot"]["sha256"]
    selected = select_result(
        root / "torch-training", snapshot_sha256=state, output=root / "selected.json"
    )
    assert selected["training_ir_sha256"] == run["training"]["training_ir_sha256"]
    rejected(
        lambda: select_result(
            root / "torch-training",
            snapshot_sha256="0" * 64,
            output=root / "bad-selected.json",
        )
    )
    frozen = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3))
    frozen[0].weight.requires_grad_(False)
    rejected(lambda: optimize(frozen, objective="structure"))
    x = torch.ones(1, 2, requires_grad=True)
    optimize(torch.nn.Sequential(torch.nn.Linear(2, 2)), rules=[], example_args=(x,))
    assert x.grad is None
    rejected(
        lambda: optimize(
            torch.nn.Sequential(torch.nn.Linear(2, 2)),
            rules=[],
            example_args=(torch.full((1, 2), float("nan")),),
        )
    )
    with Recorder(logdir=root / "exclusive") as writer:
        rejected(lambda: Recorder(logdir=root / "exclusive").__enter__())
        writer.finish()
        rejected(lambda: writer.record_metrics({"late": 1}))
    hooked = torch.nn.Sequential(torch.nn.Linear(2, 2))
    hooked[0].register_forward_hook(lambda m, i, o: o)
    rejected(lambda: optimize(hooked, rules=[]))
    # Exact aliases are retained, not counted twice.
    layer = torch.nn.Linear(2, 2)
    tied = torch.nn.Sequential(layer, layer)
    r = optimize(tied, rules=[], output_dir=root / "tied")
    z = load_model(root / "tied")
    assert z[0] is z[1]
    assert r.evidence["snapshot"]["tensors"][0]["aliases"] == ["1.weight"]
    # Fresh-process factory + weights-only file + generated callable reuse.
    (root / "native_factory.py").write_text(
        "import torch\ndef build_model():\n    return torch.nn.Sequential(torch.nn.Linear(2,2))\n"
    )
    torch.save(
        torch.nn.Sequential(torch.nn.Linear(2, 2)).state_dict(), root / "weights.pt"
    )
    (root / "inputs.json").write_text(
        json.dumps({"args": [{"shape": [1, 2], "dtype": "float32"}], "kwargs": {}})
    )
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([PACKAGE_ROOT, str(root)]),
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "deepbom.optimization",
            "optimize",
            "--factory",
            "native_factory:build_model",
            "--state-file",
            str(root / "weights.pt"),
            "--inputs",
            str(root / "inputs.json"),
            "--output",
            str(root / "cli"),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    code = (
        "import importlib.util; s=importlib.util.spec_from_file_location('selected',r'"
        + str(root / "cli" / "model.py")
        + "');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert m.build_model() is not None"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    (root / "cli" / "state.json").write_text("{}")
    rejected(lambda: load_model(root / "cli"))


check(
    "Native input/output CLI, model alias, finite probes, immutable selection and tampering",
    contract_edges,
)


def compatibility_edges():
    from deepbom._native.core import json_value
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    class Reversed(torch.nn.Sequential):
        def forward(self, value):
            return self[0](self[1](value))

    class Bypass(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(2, 2, 1), torch.nn.BatchNorm2d(2)
            )

        def forward(self, value):
            return self.layers[0](value)

    for custom, conv, bn in [
        (Reversed(torch.nn.Conv2d(2, 2, 1), torch.nn.BatchNorm2d(2)).eval(), "0", "1"),
        (Bypass().eval(), "layers.0", "layers.1"),
    ]:
        candidate = optimize(custom)
        assert candidate.transformations == []
        sample = torch.ones(1, 2, 3, 3)
        torch.testing.assert_close(
            custom(sample), candidate.model(sample), rtol=0, atol=0
        )
        rejected(
            lambda: optimize(
                custom,
                rules=[{"kind": "fold_conv_bn", "subject": conv, "batchnorm": bn}],
            )
        )

    inputs = keras.Input((5, 5, 2), dtype="float64", name="precise_input")
    value = keras.layers.Conv2D(4, 3, dtype="float64", name="precise_conv")(inputs)
    value = keras.layers.BatchNormalization(dtype="float64", name="precise_bn")(value)
    precise = keras.Model(inputs, value, name="precise_model")
    sample = tf.constant(np.arange(50).reshape(1, 5, 5, 2) / 97, dtype=tf.float64)
    folded = optimize(precise, output_dir=root / "precise-fold")
    assert folded.model(sample).dtype == tf.float64
    np.testing.assert_allclose(
        precise(sample, training=False),
        folded.model(sample, training=False),
        rtol=1e-12,
        atol=1e-12,
    )
    inverted = optimize(
        precise, objective="structure", output_dir=root / "precise-inverted"
    )
    restored = load_model(root / "precise-inverted")
    assert restored(sample).dtype == tf.float64
    np.testing.assert_array_equal(inverted.model(sample), restored(sample))

    inputs = keras.Input((2,), name="argmax_input")
    value = keras.layers.Dense(2, name="argmax_dense")(inputs)
    disconnected = keras.Model(
        inputs, keras.ops.argmax(value, axis=-1), name="disconnected"
    )
    report = optimize(
        disconnected, rules=[], example_args=(tf.ones((1, 2)),)
    ).validation
    assert report["backward"] == "not_assessed_no_differentiable_loss"
    assert report["smoke_update"] == "not_run"

    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    op = torch.optim.SGD(model.parameters(), lr=0.1)
    with Recorder(model, optimizer=op, logdir=root / "terminal-rejection") as rec:
        rec.finish()
        rejected(lambda: rec.before_update(optimizer=op, at={}))
        rejected(lambda: rec.record_snapshot())
        assert rec.pending == {} and rec.attempts == 0
    assert read_run(root / "terminal-rejection")["training"]["lifecycle"] == "ended"

    with Recorder(
        model, optimizer=op, logdir=root / "update-order", capture=("weights",), every=2
    ) as rec:
        rejected(lambda: rec.before_update(optimizer=op, at=[]))
        assert rec.pending == {} and rec.attempts == 0
        first = rec.before_update(optimizer=op, at={"ordinal": 0})
        second = rec.before_update(optimizer=op, at={"ordinal": 1})
        rejected(lambda: rec.after_update(first, optimizer=op, at="invalid"))
        assert first in rec.pending
        rec.after_update(second, optimizer=op, at={"ordinal": 1})
        rec.after_update(first, optimizer=op, at={"ordinal": 0})
    snapshots = [
        e
        for e in read_run(root / "update-order")["events"]
        if e["kind"] == "model_snapshot"
    ]
    assert len(snapshots) == 1 and snapshots[0]["at"]["ordinal"] == 0
    rejected(lambda: json_value({1: "first", "1": "second"}))

    manifest_path = root / "torch-fold" / "manifest.json"
    original_manifest = manifest_path.read_text()
    for mutation in ("identity", "missing-file", "reference-schema", "baseline"):
        document = json.loads(original_manifest)
        if mutation == "identity":
            document["snapshot_sha256"] = "0" * 64
        elif mutation == "missing-file":
            del document["files"]["state.json"]
        elif mutation == "reference-schema":
            document["evidence"]["weight_ir"]["schema"] = "deepbom.weight_ir.v1"
        else:
            document["baseline_sha256"] = "0" * 64
        document["manifest_sha256"] = content_hash(
            {k: v for k, v in document.items() if k != "manifest_sha256"}
        )
        try:
            manifest_path.write_text(json.dumps(document))
            rejected(lambda: load_evidence(root / "torch-fold"))
            rejected(lambda: load_model(root / "torch-fold"))
        finally:
            manifest_path.write_text(original_manifest)

    directory = root / "progress-export"
    with Recorder(logdir=directory) as rec:
        rec.record_metrics({"loss": 2.0}, at={"step": 0})
        rec.flush()
        first = export(
            directory, destination="tensorboard", logdir=root / "progress-tb"
        )
        precise = {"loss": 1.0, "precise": 1.0000000000000002, "large": 1e100}
        rec.record_metrics(precise, at={"step": 1})
        rec.flush()
        second = export(
            directory, destination="tensorboard", logdir=root / "progress-tb"
        )
        assert first["event_logdir"] != second["event_logdir"]
        assert (
            export(directory, destination="tensorboard", logdir=root / "progress-tb")
            == second
        )
    from tensorboard.util.tensor_util import make_ndarray

    first_values = EventAccumulator(first["event_logdir"]).Reload().Tensors("loss")
    second_values = EventAccumulator(second["event_logdir"]).Reload().Tensors("loss")
    assert [make_ndarray(v.tensor_proto).item() for v in first_values] == [2.0]
    assert [make_ndarray(v.tensor_proto).item() for v in second_values] == [2.0, 1.0]
    accumulator = EventAccumulator(second["event_logdir"]).Reload()
    for name, value in precise.items():
        observed = make_ndarray(accumulator.Tensors(name)[-1].tensor_proto)
        assert observed.dtype == np.dtype("float64") and observed.item() == value
        assert accumulator.SummaryMetadata(name).plugin_data.plugin_name == "scalars"
    import_logs(
        "tensorboard", output=root / "progress-import", logdir=second["event_logdir"]
    )
    imported = {
        key: value
        for event in read_run(root / "progress-import")["events"]
        for key, value in event["payload"].get("values", {}).items()
    }
    assert imported == precise


check(
    "Native compatibility boundaries and incremental export isolation",
    compatibility_edges,
)


def serialized_schema_contracts():
    from collections import Counter
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from deepbom._native.adapters import snapshot_input

    schemas = []
    for path in (ROOT / "docs/schemas").glob("*.json"):
        document = json.loads(path.read_text())
        if "$id" in document and "$schema" in document:
            schemas.append((document["$id"], Resource.from_contents(document)))
    registry = Registry().with_resources(schemas)
    schema = json.loads(
        (ROOT / "docs/schemas/deepbom-native-evidence-v1.schema.json").read_text()
    )
    validator = Draft202012Validator(schema, registry=registry)
    identities = {
        definition["properties"]["schema"]["const"]: name
        for name, definition in schema["$defs"].items()
    }
    counts = Counter()
    for path in root.rglob("*.json"):
        document = json.loads(path.read_text())
        kind = (
            identities.get(document.get("schema"))
            if isinstance(document, dict)
            else None
        )
        if kind:
            try:
                validator.validate(document)
            except Exception as error:
                raise AssertionError(f"Schema validation failed: {path}") from error
            counts[kind] += 1
    request = snapshot_input(torch.nn.Sequential(torch.nn.Linear(2, 2)))
    validator.validate(request)
    validator.validate(
        {
            "schema": "deepbom.training_input.v1",
            "run_id": "schema-check",
            "segment_id": "segment",
            "lifecycle": "running",
            "events": [],
            "previous": None,
        }
    )
    for kind in (
        "snapshot",
        "model",
        "weight",
        "activation",
        "training",
        "chunk",
        "candidate",
        "comparison",
        "result",
    ):
        assert counts[kind] > 0, f"No real serialized {kind} document was checked"
    print("SCHEMA_COUNTS", json.dumps(dict(sorted(counts.items()))), flush=True)


check(
    "Real serialized native inputs, candidates, result and all IR schemas",
    serialized_schema_contracts,
)

print(
    json.dumps(
        {
            "root": str(root),
            "passed": results,
            "torch": torch.__version__,
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
        },
        indent=2,
    )
)
