"""Real framework regressions; no private models or external tracking servers."""
import copy
import tempfile
from pathlib import Path
from unittest.mock import patch
import warnings
import os
import subprocess
import sys
import json
import random
import numpy as np
import torch
import tensorflow as tf
import keras
import deepbom
from deepbom._native.adapters import snapshot_input, independent_copy
from deepbom.optimization import optimize, refine, export, load_model
from deepbom.training.integrations import read_run

torch.set_num_threads(1)

# Explicit inspection/validation must not consume caller Python/NumPy/Torch
# default streams, including when the copy's forward raises. TF's stateful-op
# counters and arbitrary user side effects are outside this in-process guard.
from deepbom.optimization._report_probe import capture_shapes
from deepbom.optimization import _validate
class RandomProbe(torch.nn.Module):
    def forward(self, x, fail=False):
        random.random(); np.random.random()
        y = x + torch.rand_like(x)
        if fail:
            raise ValueError("probe failed")
        return y

def rng_state():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()

def assert_rng_same(before):
    after = rng_state()
    assert before[0] == after[0]
    assert before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1]) and before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])

for probe in (capture_shapes, _validate):
    for fail in (False, True):
        before = rng_state()
        try:
            probe(RandomProbe(), (torch.ones(1,2),), {"fail":fail})
        except ValueError as error:
            assert fail and "probe failed" in str(error)
        else:
            assert not fail
        assert_rng_same(before)
keras_probe = keras.Sequential([keras.Input((2,)), keras.layers.Dense(1)])
before = rng_state()
capture_shapes(keras_probe, (tf.ones((1,2)),), {})
assert_rng_same(before)


def transpose_oracle(x, weight, y, stride, padding, dilation, groups):
    # Enumerate contributing products, independent of the production formula.
    count = 0
    for a in range(x[2]):
        for b in range(x[3]):
            for u in range(weight[2]):
                for v in range(weight[3]):
                    if 0 <= a * stride[0] - padding[0] + u * dilation[0] < y[2] and 0 <= b * stride[1] - padding[1] + v * dilation[1] < y[3]:
                        count += x[0] * x[1] * weight[1]
    return count


with tempfile.TemporaryDirectory(prefix="deepbom-monitor-check-") as tmp:
    root = Path(tmp)
    for index, cfg in enumerate([
        {}, {"stride": 2, "padding": 1, "output_padding": 1},
        {"groups": 2, "padding": (1, 2), "dilation": (2, 1), "stride": (2, 1)},
    ]):
        layer = torch.nn.ConvTranspose2d(2, 4, (3, 2), **cfg)
        model = torch.nn.Sequential(layer)
        x = torch.ones(2, 2, 3, 4)
        before = snapshot_input(model, max_values=0)
        result = deepbom.inspect_native(model, example_args=(x,), max_values=0)
        y = model(x)
        assert int(result["structure"]["cost"]["macs"]) == transpose_oracle(x.shape, layer.weight.shape, y.shape, layer.stride, layer.padding, layer.dilation, layer.groups)
        assert snapshot_input(model, max_values=0) == before
        assert result["validation"]["runtime_latency"] == "not_measured"
    with patch("deepbom._native.adapters.tensor_row", side_effect=AssertionError("copied")):
        try:
            deepbom.inspect_native(model, max_state_bytes=1)
        except ValueError as error:
            assert "No state was copied" in str(error)
        else:
            raise AssertionError("Copy budget was bypassed")
    assert deepbom.inspect_native(model, max_values=0)["structure"]["cost"]["status"] == "not_assessed"
    # A call-time output_size overrides configured output_padding. Do not report
    # a configured-shape cost as a complete observation of that invocation.
    layer = torch.nn.ConvTranspose2d(2,4,3,stride=2)
    x = torch.ones(1,2,3,3)
    overridden = deepbom.inspect_native(layer,example_args=(x,),example_kwargs={"output_size":[1,4,8,8]},max_values=0)
    assert overridden["structure"]["cost"]["unmapped_invocations"] == 1
    assert overridden["structure"]["cost"]["status"] != "complete"
    factory=root/'inspect_fixture.py'
    factory.write_text('import torch\ndef build():\n    torch.manual_seed(17)\n    return torch.nn.Sequential(torch.nn.Linear(2,1)).eval()\n')
    probe_file=root/'probe.json';probe_file.write_text(json.dumps({'args':[{'shape':[1,2],'dtype':'float32'}]}))
    process=subprocess.run([sys.executable,'-m','deepbom','inspect-native','--factory','inspect_fixture:build','--inputs',str(probe_file)],
        env={**os.environ,'PYTHONPATH':str(root)+os.pathsep+os.environ.get('PYTHONPATH','')},capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    inspected=json.loads(process.stdout)
    torch.manual_seed(17)
    expected=deepbom.inspect_native(torch.nn.Sequential(torch.nn.Linear(2,1)).eval(),example_args=(torch.zeros(1,2),))
    assert inspected==expected,'Native CLI and API do not share the same result'

    # Observe identical deterministic updates, with no owned backward/step.
    baseline = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1))
    observed = copy.deepcopy(baseline)
    opt_a = torch.optim.SGD(baseline.parameters(), lr=.01)
    opt_b = torch.optim.SGD(observed.parameters(), lr=.01)
    x = torch.tensor([[1., 2.], [3., 4.]])
    for _ in range(3):
        opt_a.zero_grad(); baseline(x).square().sum().backward(); opt_a.step()
    context = deepbom.monitor(observed, optimizer=opt_b, logdir=root / "torch", every=1, preset="gradients", report=True)
    with context as session:
        for step in range(3):
            opt_b.zero_grad(); loss = observed(x).square().sum(); loss.backward(); opt_b.step()
            session.record_metrics({"loss": loss.item()}, at={"step": step, "phase": "train"})
    assert all(torch.equal(a, b) for a, b in zip(baseline.parameters(), observed.parameters()))
    assert all(torch.equal(a.grad, b.grad) for a, b in zip(baseline.parameters(), observed.parameters()))
    assert context.receipt["training_lifecycle"] == "collector_closed"
    assert not opt_b._optimizer_step_pre_hooks and not opt_b._optimizer_step_post_hooks
    assert Path(context.receipt["report"]).is_file()
    assert read_run(root / "torch")["events"]

    # An observer failure is visible, and does not mask a user exception.
    context = deepbom.monitor(observed, optimizer=opt_b, logdir=root / "failure", errors="record_and_disable")
    original = RuntimeError("user training failed")
    with warnings.catch_warnings(record=True) as caught:
        try:
            with context as session:
                session.recorder.flush = lambda: (_ for _ in ()).throw(OSError("test sink failure"))
                raise original
        except RuntimeError as error:
            assert error is original
        assert caught
    assert context.receipt["collection_status"] == "incomplete"
    assert not opt_b._optimizer_step_pre_hooks and not opt_b._optimizer_step_post_hooks
    assert not (root / "failure" / ".writer.lock").exists()

    # The same session wraps a manual GradientTape boundary, without applying
    # gradients or recreating the caller's tape.
    tape_base=keras.Sequential([keras.Input((2,)),keras.layers.Dense(1)])
    tape_observed=keras.models.model_from_json(tape_base.to_json());tape_observed.set_weights(tape_base.get_weights())
    def tape_step(model,optimizer,session=None):
        with tf.GradientTape() as tape: loss=tf.reduce_sum(model(tf.ones((1,2)))**2)
        gradients=tape.gradient(loss,model.trainable_variables)
        if session:token=session.before_update(optimizer=optimizer,at={"step":0},gradients=gradients,variables=model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        if session:session.after_update(token,optimizer=optimizer)
    tape_step(tape_base,keras.optimizers.SGD(.01))
    optimizer=keras.optimizers.SGD(.01)
    with deepbom.monitor(tape_observed,optimizer=optimizer,logdir=root/'tape-monitor',preset='gradients',every=1,errors='strict') as session:
        tape_step(tape_observed,optimizer,session)
    assert all((a==b).all() for a,b in zip(tape_base.get_weights(),tape_observed.get_weights()))
    assert session.receipt['collection_status']=='completed'

    # Duplicate local names in prebuilt nested models must not imply sharing.
    a = keras.Sequential([keras.layers.Dense(2, name="dense")], name="a")
    b = keras.Sequential([keras.layers.Dense(2, name="dense")], name="b")
    a(tf.ones((1, 2))); b(tf.ones((1, 2)))
    inp = keras.Input((2,)); model = keras.Model(inp, [a(inp), b(inp)])
    source = snapshot_input(model, max_values=0)
    assert len(source["tensors"]) == 4
    assert len({row["id"] for row in source["tensors"]}) == 4
    assert source == snapshot_input(independent_copy(model), max_values=0)
    candidate = optimize(model, rules=[], example_args=(tf.ones((1, 2)),))
    export(candidate, root / "candidate")
    loaded = load_model(root / "candidate")
    assert snapshot_input(loaded, max_values=0) == snapshot_input(candidate.model, max_values=0)
    refined = refine(candidate, rules=[], example_args=(tf.ones((1, 2)),))
    assert snapshot_input(refined.model, max_values=0) == snapshot_input(candidate.model, max_values=0)

    model = keras.Sequential([keras.layers.Input((2,)), keras.layers.Dense(1)])
    model.compile(optimizer="sgd", loss="mse")
    context = deepbom.monitor(model, logdir=root / "keras", preset="weights", every=1)
    with context as session:
        model.fit(tf.ones((2, 2)), tf.ones((2, 1)), batch_size=2, epochs=2,
                  validation_data=(tf.ones((2, 2)), tf.ones((2, 1))),
                  callbacks=[session.keras_callback()], verbose=0)
    assert context.receipt["training_lifecycle"] == "ended"
    assert context.receipt["collection_status"] == "completed", context.receipt
    assert any(event["kind"] == "model_snapshot" for event in read_run(root / "keras")["events"])
    validation = [event for event in read_run(root / "keras")["events"] if event["kind"] == "metrics" and event["at"].get("phase") == "validation"]
    assert {event["at"]["epoch"] for event in validation} == {0, 1}

print("Native inspection, common ConvTranspose MAC oracle, copy budget, monitor noninterference, cleanup and Keras identity/package checks passed.")
