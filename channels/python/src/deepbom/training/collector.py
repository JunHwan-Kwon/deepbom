from __future__ import annotations
import copy
from pathlib import Path
import threading
import uuid
import weakref
from .._native.core import engine, atomic_json, EvidenceStore, json_value, tensor_row
from .._native.adapters import framework_of, snapshot_input

_ACTIVE = weakref.WeakKeyDictionary()


class Recorder:
    def __init__(
        self,
        model=None,
        *,
        optimizer=None,
        logdir,
        observation="manual",
        capture=("metrics",),
        every=1,
        max_values=100_000,
        run_id=None,
        chunk_size=32,
    ):
        if observation not in ("auto", "manual"):
            raise ValueError("observation must be auto or manual")
        if not isinstance(every, int) or isinstance(every, bool) or every < 1:
            raise ValueError("every must be a positive integer")
        if (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or not 1 <= chunk_size <= 1000
        ):
            raise ValueError("chunk_size must be in [1,1000]")
        if set(capture) - {"metrics", "gradients", "weights", "activations"}:
            raise ValueError("Unknown capture kind")
        if observation == "auto" and (
            model is None or framework_of(model) != "pytorch" or optimizer is None
        ):
            raise ValueError(
                "Automatic update observation requires PyTorch model and optimizer"
            )
        self.model = model
        self.optimizer = optimizer
        self.observation = observation
        self.capture = set(capture)
        self.every = every
        self.max_values = max_values
        self.logdir = Path(logdir)
        self.store = EvidenceStore(self.logdir)
        if (self.logdir / "latest.json").exists():
            raise FileExistsError(
                "Use a new evidence directory; previous runs are immutable"
            )
        self.run_id = run_id or uuid.uuid4().hex
        self.segment_id = uuid.uuid4().hex
        self.sequence = 0
        self.events = []
        self.previous = None
        self.lifecycle = "running"
        self.chunk_size = chunk_size
        if (
            not isinstance(max_values, int)
            or isinstance(max_values, bool)
            or not 0 <= max_values <= 100_000_000
        ):
            raise ValueError("Invalid max_values")
        self.pending = {}
        self.attempts = 0
        self.handles = []
        self._entered = False
        self._closed = False
        self._thread = None
        self._snapshots = {}
        self._auto_token = None

    def __enter__(self):
        if self._entered or self._closed:
            raise RuntimeError("Recorder cannot be entered twice")
        if self.model is not None and self.model in _ACTIVE:
            raise ValueError("Model already has an active DEEPBOM collector")
        self._lock_path = self.logdir / ".writer.lock"
        with self._lock_path.open("x", encoding="utf-8") as lock:
            lock.write(self.run_id + "\n" + self.segment_id + "\n")
        try:
            if (self.logdir / "latest.json").exists():
                raise FileExistsError("Evidence directory already has a run")
            return self._begin()
        except BaseException:
            for handle in self.handles:
                handle.remove()
            if self.model is not None:
                _ACTIVE.pop(self.model, None)
            self._closed = True
            self._lock_path.unlink(missing_ok=True)
            raise

    def _begin(self):
        if self.model is not None:
            _ACTIVE[self.model] = self
        self._entered = True
        self._thread = threading.get_ident()
        if self.observation == "auto":

            def pre(optimizer, args, kwargs):
                if kwargs.get("closure") is not None or len(args) > 1:
                    raise ValueError("Optimizer closure observation is not qualified")
                self._auto_token = self._before_update(
                    optimizer=optimizer, at={}, gradient_stage="optimizer_pre_hook"
                )

            def post(optimizer, args, kwargs):
                self._after_update(self._auto_token, optimizer=optimizer)

            self.handles = [
                self.optimizer.register_step_pre_hook(pre),
                self.optimizer.register_step_post_hook(post),
            ]
        self.event(
            "collector_started",
            {},
            {
                "capture": sorted(self.capture),
                "observation": self.observation,
                "every": self.every,
                "max_values": self.max_values,
                "limits": [
                    "No inferred loss, data semantics, AMP-skipped calls, distributed aggregation or task quality.",
                    "Activations require explicit record_activations; fit callbacks do not observe internal forwards.",
                ],
            },
        )
        return self

    def _check(self, *, running=False):
        if not self._entered or self._closed:
            raise RuntimeError("Recorder is not active")
        if threading.get_ident() != self._thread:
            raise RuntimeError("Use separate collectors for concurrent streams")
        if running and self.lifecycle != "running":
            raise RuntimeError("Training lifecycle was already finalized")

    def event(self, kind, at, payload):
        self._check()
        if (
            not isinstance(kind, str)
            or not kind
            or not isinstance(at, dict)
            or not isinstance(payload, dict)
        ):
            raise ValueError(
                "An event requires a nonempty kind and object coordinates/payload"
            )
        if self.lifecycle != "running" and kind not in (
            "training_status",
            "collector_closed",
        ):
            raise RuntimeError(
                "Cannot append training observations after an explicit terminal lifecycle"
            )
        row = {
            "id": f"{self.segment_id}:{self.sequence}",
            "sequence": str(self.sequence),
            "kind": kind,
            "at": json_value(at),
            "payload": json_value(payload),
        }
        self.events.append(row)
        self.sequence += 1
        if len(self.events) >= self.chunk_size:
            self.flush()
        return row["id"]

    def flush(self):
        self._check()
        if (
            not self.events
            and self.previous
            and self.previous["lifecycle"] == self.lifecycle
        ):
            return self.previous
        result = engine(
            "training",
            {
                "schema": "deepbom.training_input.v1",
                "run_id": self.run_id,
                "segment_id": self.segment_id,
                "lifecycle": self.lifecycle,
                "events": self.events,
                "previous": self.previous,
            },
        )
        self.store.put(result["chunk"])
        self.store.put(result["document"])
        atomic_json(
            self.logdir / "latest.json",
            {
                "schema": "deepbom.training_head.v1",
                "training_ir_sha256": result["document"]["training_ir_sha256"],
            },
            replace=True,
        )
        self.previous = result["document"]
        self.events = []
        return self.previous

    def record_metrics(
        self, metrics, *, at=None, definition="user_supplied; aggregation not inferred"
    ):
        if "metrics" not in self.capture:
            return None
        return self.event(
            "metrics",
            at or {},
            {"values": metrics, "definition": definition, "model_state_binding": None},
        )

    def record_snapshot(self, *, at=None, advanced=False):
        self._check(running=True)
        if self.model is None:
            raise ValueError("Snapshot requires a model")
        raw = snapshot_input(self.model, max_values=self.max_values, advanced=advanced)
        bundle = engine("snapshot", raw)
        refs = {k: self.store.put(v) for k, v in bundle.items() if v is not None}
        digest = bundle["snapshot"]["snapshot_sha256"]
        self._snapshots = {digest: raw}
        self.event(
            "model_snapshot",
            at or {},
            {"evidence": refs, "binding": "adapter_consistent_host_copy"},
        )
        return digest

    def record_activations(self, values, *, snapshot_sha256, at, invocation, context):
        self._check(running=True)
        if "activations" not in self.capture:
            raise ValueError("Enable activations explicitly")
        if snapshot_sha256 not in self._snapshots:
            raise ValueError(
                "Activation needs a retained explicit before-forward snapshot"
            )
        if (
            not isinstance(context, dict)
            or not {"mode", "input_identity", "state_boundary"} <= context.keys()
        ):
            raise ValueError(
                "Activation context needs mode, input_identity and state_boundary"
            )
        if "at" in context and json_value(context["at"]) != json_value(at):
            raise ValueError("Activation context.at conflicts with event coordinates")
        raw = copy.deepcopy(self._snapshots[snapshot_sha256])
        rows = []
        for i, (subject, value) in enumerate(values.items()):
            rows.append(
                {
                    "subject_ref": subject,
                    "invocation": str(invocation),
                    "tensor": tensor_row(value, f"activation:{i}", role="activation"),
                }
            )
        raw["capture"] = {
            "context": {**json_value(context), "at": json_value(at)},
            "values": rows,
        }
        result = engine("snapshot", raw)
        ref = self.store.put(result["activation_ir"])
        self.event(
            "activation_capture",
            at,
            {
                "evidence": ref,
                "snapshot_sha256": snapshot_sha256,
                "binding": "explicit_context_not_attested",
            },
        )
        return ref

    def before_update(self, **kwargs):
        if self.observation != "manual":
            raise ValueError("Manual update calls cannot be mixed with automatic hooks")
        return self._before_update(**kwargs)

    def _before_update(
        self,
        *,
        optimizer,
        at,
        loss=None,
        gradients=None,
        variables=None,
        gradient_stage="computed",
        context=None,
    ):
        self._check(running=True)
        token = uuid.uuid4().hex
        if optimizer is not self.optimizer:
            raise ValueError("Unknown optimizer; use its own collector")
        observation_context = dict(context or {})
        if "optimizer_binding" in observation_context:
            raise ValueError("optimizer_binding is assigned by the collector")
        if framework_of(self.model) == "pytorch":
            members = {id(v): n for n, v in self.model.named_parameters()}
            selected = [v for group in optimizer.param_groups for v in group["params"]]
            if not selected or any(id(v) not in members for v in selected):
                raise ValueError("Optimizer parameters do not belong to the selected model")
            if len({id(v) for v in selected}) != len(selected):
                raise ValueError("Optimizer repeats a model parameter")
            observation_context["optimizer_binding"] = {"status":"model_parameter_subset", "native_state_refs":[members[id(v)] for v in selected], "model_parameter_count":len(members)}
        else:
            members = {id(v): getattr(v, "path", v.name) for v in self.model.weights}
            selected = getattr(optimizer, "_trainable_variables", ())
            if selected and any(id(v) not in members for v in selected):
                raise ValueError("Optimizer variables do not belong to the selected model")
            observation_context["optimizer_binding"] = {"status":"model_variable_subset" if selected else "not_observable_before_optimizer_build", "native_state_refs":[members[id(v)] for v in selected], "model_variable_count":len(members)}
        attempt_index = self.attempts
        payload = {
            "token": token,
            "gradient_stage": gradient_stage,
            "context": observation_context,
            "effect": "not_yet_observed",
        }
        if loss is not None and "metrics" in self.capture:
            payload["loss"] = json_value(loss)
        if "gradients" in self.capture and attempt_index % self.every == 0:
            if framework_of(self.model) == "pytorch":
                names = {id(v): n for n, v in self.model.named_parameters()}
            else:
                names = {id(v): getattr(v, "path", v.name) for v in self.model.weights}
            if variables is None:
                if framework_of(self.model) != "pytorch":
                    raise ValueError(
                        "GradientTape observations require variables and already-computed gradients"
                    )
                variables = tuple(self.model.parameters())
                gradients = tuple(v.grad for v in variables)
            if gradients is None or len(variables) != len(gradients):
                raise ValueError("Gradient/variable arity mismatch")
            if len({id(v) for v in variables}) != len(variables):
                raise ValueError("Gradient variables must be unique")
            gradient_rows = []
            transport_rows = []
            destinations = []
            for i, (v, g) in enumerate(zip(variables, gradients)):
                if id(v) not in names:
                    raise ValueError(
                        "Gradient variable does not belong to the selected model"
                    )
                subject = {
                    "index": i,
                    "native_state_ref": names[id(v)],
                    "snapshot_binding": None,
                }
                if g is None:
                    gradient_rows.append({**subject, "status": "missing_gradient"})
                    continue
                scope = "dense_tensor"
                indices = None
                dense_shape = list(v.shape)
                if hasattr(g, "is_sparse") and g.is_sparse:
                    indices = json_value(g._indices())
                    g = g._values()
                    scope = "sparse_stored_values_not_dense_gradient"
                elif type(g).__name__ == "IndexedSlices":
                    indices = json_value(g.indices)
                    g = g.values
                    scope = "indexed_slices_values_not_dense_gradient"
                if scope == "dense_tensor" and list(g.shape) != dense_shape:
                    raise ValueError("Dense gradient shape does not match its model variable")
                row = tensor_row(g, f"gradient:{i}", role="gradient")
                transport_rows.append(row)
                destinations.append(len(gradient_rows))
                gradient_rows.append(
                    {
                        **subject,
                        "status": "observed",
                        "scope": scope,
                        "indices": indices,
                        "dense_shape": dense_shape,
                        "evidence": None,
                    }
                )
            if transport_rows:
                batch = engine(
                    "tensor_batch",
                    {
                        "tensors": transport_rows,
                        "options": {"maxValues": self.max_values},
                    },
                )
                if len(batch["tensors"]) != len(destinations):
                    raise RuntimeError("Gradient batch correspondence mismatch")
                for index, detail in zip(destinations, batch["tensors"]):
                    gradient_rows[index]["evidence"] = detail
            payload["gradients"] = gradient_rows
        self.event("update_attempt", at, payload)
        self.pending[token] = {
            "optimizer": optimizer,
            "sampled": attempt_index % self.every == 0,
        }
        self.attempts += 1
        return token

    def after_update(
        self, token, *, optimizer, outcome="returned_effect_unverified", at=None
    ):
        if self.observation != "manual":
            raise ValueError("Manual update calls cannot be mixed with automatic hooks")
        return self._after_update(token, optimizer=optimizer, outcome=outcome, at=at)

    def _after_update(
        self, token, *, optimizer, outcome="returned_effect_unverified", at=None
    ):
        self._check(running=True)
        if (
            token not in self.pending
            or self.pending[token]["optimizer"] is not optimizer
        ):
            raise ValueError("Unknown, reused, or foreign optimizer token")
        if outcome not in (
            "returned_effect_unverified",
            "skipped_declared",
            "failed_declared",
        ):
            raise ValueError(
                "Unsupported update outcome; return alone does not prove changed weights"
            )
        self.event("update_returned", at or {}, {"token": token, "outcome": outcome})
        sampled = self.pending.pop(token)["sampled"]
        if "weights" in self.capture and sampled:
            self.record_snapshot(at=at)

    def finish(self, state="ended", *, origin="user_declared"):
        self._check()
        if self.lifecycle != "running":
            raise RuntimeError("Training lifecycle was already finalized")
        if state not in ("ended", "failed", "interrupted"):
            raise ValueError("Invalid training finish state")
        self.lifecycle = state
        self.event("training_status", {}, {"state": state, "origin": origin})
        self.flush()

    def __exit__(self, typ, value, tb):
        try:
            for handle in self.handles:
                handle.remove()
            if self.lifecycle == "running":
                self.lifecycle = "failed" if typ else "collector_closed"
            self.event(
                "collector_closed",
                {},
                {
                    "unfinished_update_tokens": list(self.pending),
                    "training_completion_inferred": False,
                },
            )
            self.flush()
        finally:
            if self.model is not None:
                _ACTIVE.pop(self.model, None)
            self._closed = True
            self._snapshots.clear()
            self._lock_path.unlink(missing_ok=True)
        return False


def watch(model, **options):
    options.setdefault("observation", "auto")
    return Recorder(model, **options)
