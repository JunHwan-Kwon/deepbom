"""Explicit local model candidate optimization; training remains user-owned."""

from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid
from .._native.adapters import framework_of, independent_copy, snapshot, snapshot_input
from .._native.core import (
    engine,
    atomic_json,
    content_hash,
    read_json,
    verify_document,
    EvidenceStore,
)
from ._models import apply_rules, model_spec, restore_model, has_sequential_call_path


def suggest(model, *, objective="cpu_inference"):
    """Return bounded candidates, not measured optimization benefits."""
    fw = framework_of(model)
    proposals = []
    if objective not in ("cpu_inference", "structure"):
        raise ValueError("Unsupported objective")
    if fw == "pytorch":
        import torch

        for path, parent in model.named_modules():
            if not isinstance(parent, torch.nn.Sequential):
                continue
            items = list(parent.named_children())
            for i, (name, layer) in enumerate(items):
                subject = path + "." + name if path else name
                if type(layer) is not torch.nn.Conv2d:
                    continue
                if objective == "cpu_inference" and i + 1 < len(items):
                    bnname, bn = items[i + 1]
                    if (
                        type(bn) is torch.nn.BatchNorm2d
                        and not layer.training
                        and not bn.training
                        and bn.running_mean is not None
                        and has_sequential_call_path(model, subject)
                    ):
                        proposals.append(
                            {
                                "kind": "fold_conv_bn",
                                "subject": subject,
                                "batchnorm": path + "." + bnname if path else bnname,
                            }
                        )
                        continue
                if (
                    objective == "structure"
                    and layer.groups == 1
                    and layer.padding_mode == "zeros"
                    and layer.kernel_size != (1, 1)
                ):
                    proposals.append(
                        {"kind": "inverted", "subject": subject, "expansion": 1}
                    )
    else:
        import keras

        for layer in model.layers:
            if (
                type(layer) is not keras.layers.Conv2D
                or layer.groups != 1
                or layer.data_format != "channels_last"
            ):
                continue
            if objective == "structure" and layer.kernel_size != (1, 1):
                proposals.append(
                    {"kind": "inverted", "subject": layer.name, "expansion": 1}
                )
            elif (
                objective == "cpu_inference"
                and keras.activations.serialize(layer.activation) == "linear"
                and len(layer._outbound_nodes) == 1
            ):
                for bn in model.layers:
                    if (
                        type(bn) is keras.layers.BatchNormalization
                        and bn.axis in (-1, 3)
                        and getattr(
                            getattr(bn.input, "_keras_history", None), "operation", None
                        )
                        is layer
                    ):
                        proposals.append(
                            {
                                "kind": "fold_conv_bn",
                                "subject": layer.name,
                                "batchnorm": bn.name,
                            }
                        )
    return proposals


class Candidate:
    def __init__(
        self,
        model,
        baseline,
        candidate,
        comparison,
        ledger,
        spec,
        rows,
        modes,
        root=None,
        parent=None,
    ):
        self.model = model
        self.baseline = baseline
        self.evidence = candidate
        self.comparison = comparison
        self.transformations = ledger
        self._spec = copy.deepcopy(spec)
        self._rows = copy.deepcopy(rows)
        self._modes = copy.deepcopy(modes)
        self._rules = [r["request"] for r in ledger]
        self._baseline_model = None
        self.parent = parent
        self.root = root or baseline["snapshot"]["snapshot_sha256"]
        self.report_path = None
        self.manifest_path = None

    @property
    def snapshot_sha256(self):
        return self.evidence["snapshot"]["snapshot_sha256"]

    def export(self, directory):
        return export(self, directory)

    def export_report(self, output, *, format=None):
        if self.manifest_path is None:
            raise ValueError(
                "Export the frozen candidate package before exporting its report"
            )
        from .report import export_report

        return export_report(Path(self.manifest_path).parent, output, format=format)


def _validate(model, example_args, example_kwargs, loss_fn=None):
    report = {
        "load": "not_run",
        "forward": "not_run",
        "backward": "not_run",
        "smoke_update": "not_run",
        "quality": "not_assessed",
        "latency": "not_measured",
    }
    if not example_args and not example_kwargs:
        return report
    test = independent_copy(model)
    fw = framework_of(test)
    if fw == "pytorch":
        import torch

        def clone_input(value):
            if torch.is_tensor(value):
                return value.detach().clone().requires_grad_(value.requires_grad)
            if isinstance(value, tuple):
                return tuple(clone_input(v) for v in value)
            if isinstance(value, list):
                return [clone_input(v) for v in value]
            if isinstance(value, dict):
                return {k: clone_input(v) for k, v in value.items()}
            return copy.deepcopy(value)

        example_args = clone_input(example_args)
        example_kwargs = clone_input(example_kwargs)

        with torch.random.fork_rng(devices=[]):
            out = test(*example_args, **example_kwargs)
            report["forward"] = "passed"
            tensors = []

            def flatten(x):
                if torch.is_tensor(x):
                    tensors.append(x)
                elif isinstance(x, dict):
                    for v in x.values():
                        flatten(v)
                elif isinstance(x, (list, tuple)):
                    for v in x:
                        flatten(v)

            flatten(out)
            report["probe_outputs"] = [
                {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)}
                for t in tensors
            ]
            if any(
                t.is_floating_point() and not torch.isfinite(t).all() for t in tensors
            ):
                raise ValueError(
                    "Candidate forward produced nonfinite values on the supplied probe"
                )
            loss = (
                loss_fn(out)
                if loss_fn
                else sum(
                    t.float().square().mean() for t in tensors if t.is_floating_point()
                )
            )
            if not torch.is_tensor(loss) or not loss.requires_grad:
                report["backward"] = "not_assessed_no_differentiable_loss"
                return report
            loss.backward()
            if not torch.isfinite(loss).all() or any(
                p.grad is not None
                and not torch.isfinite(
                    p.grad._values() if p.grad.is_sparse else p.grad
                ).all()
                for p in test.parameters()
            ):
                raise ValueError("Candidate probe produced nonfinite loss or gradients")
            report["backward"] = "passed"
            trainable = [p for p in test.parameters() if p.requires_grad]
            report["gradient_coverage"] = {
                "trainable": len(trainable),
                "observed": sum(p.grad is not None for p in trainable),
            }
            if trainable:
                torch.optim.SGD(trainable, lr=1e-4).step()
                report["smoke_update"] = "passed"
    else:
        import tensorflow as tf

        with tf.GradientTape() as tape:
            out = test(*example_args, **example_kwargs)
            values = tf.nest.flatten(out)
            report["probe_outputs"] = [
                {"shape": list(v.shape), "dtype": v.dtype.name} for v in values
            ]
            for value in values:
                if value.dtype.is_floating:
                    tf.debugging.assert_all_finite(
                        value, "Candidate forward produced nonfinite values"
                    )
            loss = (
                loss_fn(out)
                if loss_fn
                else sum(
                    tf.reduce_mean(tf.square(tf.cast(v, tf.float32))) for v in values
                )
            )
        report["forward"] = "passed"
        grads = tape.gradient(loss, test.trainable_variables)
        tf.debugging.assert_all_finite(loss, "Candidate probe loss is not finite")
        for gradient in grads:
            if gradient is not None:
                tf.debugging.assert_all_finite(
                    (
                        gradient.values
                        if isinstance(gradient, tf.IndexedSlices)
                        else gradient
                    ),
                    "Candidate gradient is not finite",
                )
        report["backward"] = "passed"
        report["gradient_coverage"] = {
            "trainable": len(grads),
            "observed": sum(g is not None for g in grads),
        }
        if not any(g is not None for g in grads):
            report["backward"] = "not_assessed_no_differentiable_loss"
        pairs = [
            (g, v) for g, v in zip(grads, test.trainable_variables) if g is not None
        ]
        if pairs:
            tf.keras.optimizers.SGD(1e-4).apply_gradients(pairs)
            report["smoke_update"] = "passed"
    report["loss_scope"] = "user_supplied" if loss_fn else "synthetic_not_task_quality"
    return report


def optimize(
    model,
    *,
    framework=None,
    example_args=(),
    example_kwargs=None,
    rules=None,
    objective="cpu_inference",
    target="current-cpu",
    output_dir=None,
    factory=None,
    loss_fn=None,
    advanced=False,
    rationale=None,
):
    fw = framework_of(model)
    if framework is not None and fw != framework:
        raise ValueError("Framework mismatch")
    if objective not in ("cpu_inference", "structure"):
        raise ValueError("Unsupported objective")
    if target != "current-cpu":
        raise ValueError("This native implementation qualifies current-cpu only")
    if fw == "pytorch":
        import torch

        for module in model.modules():
            if any(
                getattr(module, name, {})
                for name in (
                    "_forward_hooks",
                    "_forward_pre_hooks",
                    "_backward_hooks",
                    "_backward_pre_hooks",
                )
            ):
                raise ValueError(
                    "Native optimization cannot preserve arbitrary module hooks; remove them or use an explicit supported model definition"
                )
        if any(
            isinstance(p, torch.nn.parameter.UninitializedParameter)
            for p in model.parameters()
        ):
            raise ValueError("Materialize lazy model parameters before optimization")
    if rationale is not None and (
        not isinstance(rationale, str) or len(rationale) > 4096
    ):
        raise ValueError("rationale must be a string of at most 4096 characters")
    selection = "automatic_rules" if rules is None else "explicit_rules"
    rules = suggest(model, objective=objective) if rules is None else rules
    # No automatic source mutation, optimizer restoration or training.
    original = independent_copy(model)
    baseline = snapshot(original, advanced=advanced)
    spec = None
    try:
        spec = model_spec(original, factory)
    except ValueError:
        if output_dir:
            raise
    if spec is not None and fw == "pytorch":
        spec["modes"] = {
            n: m.training for n, m in original.named_modules(remove_duplicate=False)
        }
    candidate, ledger = apply_rules(original, rules)
    before = snapshot(candidate, advanced=advanced)
    validation = _validate(
        candidate, tuple(example_args), example_kwargs or {}, loss_fn
    )
    after = snapshot(candidate, advanced=advanced)
    if before["snapshot"]["snapshot_sha256"] != after["snapshot"]["snapshot_sha256"]:
        raise RuntimeError("Validation mutated the immutable candidate")
    comparison = engine("compare", {"baseline": baseline, "candidate": before})
    rows = snapshot_input(candidate)["tensors"]
    modes = {}
    if fw == "pytorch":
        modes = {
            n: m.training for n, m in candidate.named_modules(remove_duplicate=False)
        }
    result = Candidate(
        candidate, baseline, before, comparison, ledger, spec, rows, modes
    )
    result._baseline_model = original
    result.validation = validation
    result.factory = factory
    from ._report_probe import capture_shapes

    result.report_context = {
        "version": 1,
        "objective": objective,
        "target": target,
        "selection": selection,
        "user_rationale": rationale,
        "probes": {
            "baseline": capture_shapes(
                original, tuple(example_args), example_kwargs or {}
            ),
            "candidate": capture_shapes(
                candidate, tuple(example_args), example_kwargs or {}
            ),
        },
    }
    result.validation["candidate_status"] = (
        "transformed" if ledger else "no_applicable_transformation"
    )
    if output_dir:
        export(result, output_dir)
    return result


def refine(candidate, *, rules, output_dir=None, **options):
    if not isinstance(candidate, Candidate):
        raise TypeError("refine requires a Candidate; load packages explicitly")
    options.setdefault("factory", candidate.factory)
    result = optimize(
        candidate._baseline_model, rules=rules, output_dir=None, **options
    )
    result.parent = candidate.snapshot_sha256
    result.root = candidate.root
    if output_dir:
        export(result, output_dir)
    return result


def compare(baseline, candidate):
    def evidence(value):
        return value.evidence if isinstance(value, Candidate) else load_evidence(value)

    return engine(
        "compare", {"baseline": evidence(baseline), "candidate": evidence(candidate)}
    )


def export_report(directory, output, *, format=None, baseline=None):
    from .report import export_report as write_report

    return write_report(directory, output, format=format, baseline=baseline)


def export(candidate, directory):
    if not isinstance(candidate, Candidate) or candidate._spec is None:
        raise ValueError(
            "Export requires a reconstructable model specification or factory"
        )
    path = Path(directory).resolve()
    if path.exists():
        raise FileExistsError("Candidate export never overwrites a directory")
    temp = path.with_name(path.name + ".writing-" + uuid.uuid4().hex)
    temp.mkdir(parents=True)
    try:
        store = EvidenceStore(temp)
        engine("validate_bundle", candidate.evidence)
        evidence = {
            k: store.put(v) for k, v in candidate.evidence.items() if v is not None
        }
        state = {
            "spec": candidate._spec,
            "rows": candidate._rows,
            "modes": candidate._modes,
            "rules": candidate._rules,
        }
        atomic_json(temp / "state.json", state)
        atomic_json(temp / "comparison.json", candidate.comparison)
        verified_validation = {
            **candidate.validation,
            "load": "passed_exact_state_roundtrip",
        }
        atomic_json(temp / "validation.json", verified_validation)
        atomic_json(temp / "transformation.json", {"rules": candidate.transformations})
        atomic_json(
            temp / "report-source.json",
            {
                "schema": "deepbom.optimization_report_source.v1",
                "baseline": candidate.baseline,
                "context": candidate.report_context,
            },
        )
        code = '''"""Load the selected DEEPBOM candidate state; construct a new optimizer afterward."""\nfrom pathlib import Path\nfrom deepbom.optimization import load_model\n\ndef build_model():\n    return load_model(Path(__file__).resolve().parent)\n'''
        (temp / "model.py").write_text(code, encoding="utf-8")
        (temp / "model-code.md").write_text(
            "# Selected model state\n\nRequires deepbom and the declared framework; no optimizer state is restored.\n\n```python\n"
            + code
            + "```\n",
            encoding="utf-8",
        )
        files = {
            p.relative_to(temp).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in temp.rglob("*")
            if p.is_file()
        }
        body = {
            "schema": "deepbom.model_candidate_manifest.v1",
            "snapshot_sha256": candidate.snapshot_sha256,
            "baseline_sha256": candidate.root,
            "parent_sha256": candidate.parent,
            "evidence": evidence,
            "files": files,
            "optimizer_state": "not_restored",
        }
        manifest = {**body, "manifest_sha256": content_hash(body)}
        atomic_json(temp / "manifest.json", manifest)
        # Verify reconstitution before publishing the directory.
        restored = load_model(temp)
        observed = snapshot(restored)
        if observed["snapshot"]["snapshot_sha256"] != candidate.snapshot_sha256:
            raise ValueError("Candidate round-trip changed native state or definition")
        candidate.validation = verified_validation
        from .report import export_report

        export_report(temp, temp / "report.json")
        export_report(temp, temp / "report.html")
        # Reports are projections, explicitly outside the signed state manifest.
        temp.rename(path)
        candidate.report_path = str(path / "report.html")
        candidate.manifest_path = str(path / "manifest.json")
        return path
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def _manifest(directory):
    path = Path(directory).resolve()
    m = read_json(path / "manifest.json")
    verify_document(m)
    if m["schema"] != "deepbom.model_candidate_manifest.v1":
        raise ValueError("Unsupported candidate manifest")
    required = {
        "schema",
        "snapshot_sha256",
        "baseline_sha256",
        "parent_sha256",
        "evidence",
        "files",
        "optimizer_state",
        "manifest_sha256",
    }
    if set(m) != required or m["optimizer_state"] != "not_restored":
        raise ValueError("Invalid candidate manifest fields")
    if not isinstance(m["evidence"], dict) or set(m["evidence"]) != {
        "snapshot",
        "model_ir",
        "weight_ir",
    }:
        raise ValueError("Incomplete candidate evidence bindings")
    schemas = {
        "snapshot": "deepbom.model_state_snapshot.v1",
        "model_ir": "deepbom.model_ir.v2",
        "weight_ir": "deepbom.weight_ir.v2",
    }
    required_files = {
        "state.json",
        "comparison.json",
        "validation.json",
        "transformation.json",
        "model.py",
        "model-code.md",
    }
    for key, ref in m["evidence"].items():
        if (
            not isinstance(ref, dict)
            or set(ref) != {"schema", "sha256"}
            or ref["schema"] != schemas[key]
        ):
            raise ValueError("Invalid candidate evidence reference")
        sha = ref["sha256"]
        if (
            not isinstance(sha, str)
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise ValueError("Invalid candidate evidence digest")
        required_files.add(f"objects/{sha}.json")
    if m["snapshot_sha256"] != m["evidence"]["snapshot"]["sha256"]:
        raise ValueError("Candidate manifest and snapshot identity disagree")
    if not isinstance(m["files"], dict) or not required_files <= m["files"].keys():
        raise ValueError("Candidate manifest omits required package files")
    for name, digest in m["files"].items():
        raw = path / name
        p = raw.resolve()
        if not p.is_relative_to(path) or any(
            part.is_symlink() for part in [raw, *raw.parents] if part != path
        ):
            raise ValueError("Invalid package path")
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Package file digest mismatch: {name}")
    comparison = read_json(path / "comparison.json")
    verify_document(comparison)
    if (
        comparison.get("schema") != "deepbom.native_model_comparison.v1"
        or comparison.get("baseline") != m["baseline_sha256"]
        or comparison.get("candidate") != m["snapshot_sha256"]
    ):
        raise ValueError("Candidate comparison and manifest identities disagree")
    return path, m


def load_model(directory):
    path, m = _manifest(directory)
    _load_evidence(path, m)
    state = read_json(path / "state.json")
    # Explicit native API may import the factory named by this selected package.
    restored = restore_model(
        state["spec"], state["rows"], state["modes"], state["rules"]
    )
    if (
        snapshot(restored, max_values=0)["snapshot"]["snapshot_sha256"]
        != m["snapshot_sha256"]
    ):
        raise ValueError(
            "Loaded model definition or state differs from the selected manifest"
        )
    return restored


def load_evidence(directory):
    path, m = _manifest(directory)

    return _load_evidence(path, m)


def _load_evidence(path, m):
    store = EvidenceStore(path)
    bundle = {k: store.get(ref["sha256"]) for k, ref in m["evidence"].items()}
    bundle.setdefault("activation_ir", None)
    return engine("validate_bundle", bundle)
