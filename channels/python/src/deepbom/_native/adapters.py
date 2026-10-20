"""Native facts and independent state copies; no tensor statistics here."""

from __future__ import annotations
import copy
import hashlib
import inspect
from .core import engine, json_value, tensor_row
from .keras_identity import state_bindings, identity_policy, set_identity_policy, OWNER_POLICY


def _attribute_identity(value):
    """Preserve Python container/scalar distinctions in custom definitions."""
    if type(value) in (list, tuple):
        return {
            "type": type(value).__name__,
            "items": [_attribute_identity(v) for v in value],
        }
    if type(value) is dict:
        return {
            "type": "dict",
            "entries": [
                [_attribute_identity(k), _attribute_identity(v)]
                for k, v in value.items()
            ],
        }
    if value is None or type(value) in (str, bool, int, float):
        import math

        scalar = (
            "-0"
            if type(value) is float and value == 0 and math.copysign(1, value) < 0
            else json_value(value)
        )
        return {"type": type(value).__name__, "value": scalar}
    raise ValueError(
        "Custom attributes must be plain Python data; register tensors as parameters/buffers and callable layers as modules"
    )


def framework_of(model):
    # Inspect installed modules only when the optional native API is invoked.
    import sys

    torch = sys.modules.get("torch")
    if torch is not None and isinstance(model, torch.nn.Module):
        return "pytorch"
    import keras

    if isinstance(model, keras.Model):
        if keras.backend.backend() != "tensorflow":
            raise ValueError("Only TensorFlow Keras backend is qualified")
        return "tensorflow"
    raise TypeError("Expected torch.nn.Module or TensorFlow backend keras.Model")


def require_supported_execution(model):
    """One boundary for capture, cloning, rewriting and package restoration.

    Class definitions remain scoped adapter evidence, not a proof of arbitrary
    Python execution equivalence. Instance overrides/hooks cannot be represented
    by the existing declarative constructors and are rejected before copying.
    """
    if framework_of(model) == "pytorch":
        import torch
        modules = model.modules()
        from torch.nn.modules import module as module_runtime
        if any(getattr(module_runtime, key, {}) for key in (
            "_global_forward_hooks", "_global_forward_pre_hooks",
            "_global_backward_hooks", "_global_backward_pre_hooks",
        )):
            raise ValueError("Native capture cannot represent global module hooks")
        for module in modules:
            if any(key in module.__dict__ for key in ("forward", "_call_impl", "_compiled_call_impl")) and (
                "forward" in module.__dict__ or "_call_impl" in module.__dict__
                or module.__dict__.get("_compiled_call_impl") is not None
            ):
                raise ValueError("Native capture cannot represent an instance execution override")
            if any(getattr(module, key, {}) for key in (
                "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
            )):
                raise ValueError("Native capture cannot represent arbitrary module hooks")
    else:
        seen = set()
        def visit(layer):
            if id(layer) in seen:
                return
            seen.add(id(layer))
            if "call" in layer.__dict__:
                raise ValueError("Native capture cannot represent an instance call override")
            for child in getattr(layer, "layers", ()):
                visit(child)
        visit(model)


def independent_copy(model):
    require_supported_execution(model)
    framework = framework_of(model)
    if framework == "pytorch":
        import torch

        if hasattr(model, "_orig_mod") or isinstance(
            model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        ):
            raise ValueError("Unwrap compiled/distributed models before optimization")
        result = copy.deepcopy(model)
        for a, b in zip(
            [*model.parameters(), *model.buffers()],
            [*result.parameters(), *result.buffers()],
        ):
            if a.numel() and a.data_ptr() == b.data_ptr():
                raise ValueError("Candidate shares mutable parameter storage")
        return result
    import keras

    if not model.built:
        raise ValueError(
            "Build the Keras model with explicit inputs before optimization"
        )
    # clone_model can rename Functional input layers to symbolic tensor names.
    # Use the same declarative constructor as package restoration instead.
    result = keras.models.model_from_json(model.to_json())
    set_identity_policy(result, identity_policy(model))
    restore_rows(result, snapshot_input(model, max_values=0)["tensors"])
    # Validate trainability and state cardinality after clone, not only shapes.
    if len(result.weights) != len(model.weights):
        raise ValueError("Keras clone changed state cardinality")
    for old, new in zip(model.layers, result.layers):
        new.trainable = old.trainable
    return result


def state_byte_count(model):
    """Count unique state objects before any host copy; not peak memory."""
    if framework_of(model) == "pytorch":
        values = [*model.parameters(), *model.buffers()]
        seen, total = set(), 0
        for value in values:
            if id(value) not in seen:
                total += value.numel() * value.element_size()
                seen.add(id(value))
        return total
    import numpy as np
    return sum(int(np.prod(v.shape, dtype=object)) * np.dtype(v.dtype).itemsize
               for v, _, _ in state_bindings(model)[2])


def check_state_budget(model, maximum):
    if maximum is not None:
        if type(maximum) is not int or maximum < 0:
            raise ValueError("max_state_bytes must be a nonnegative integer")
        size = state_byte_count(model)
        if size > maximum:
            raise ValueError(f"Native state requires {size} bytes; max_state_bytes is {maximum}. No state was copied.")


def snapshot_input(model, *, input_spec=None, max_values=1_000_000, advanced=False, max_state_bytes=None):
    require_supported_execution(model)
    check_state_budget(model, max_state_bytes)
    framework = framework_of(model)
    nodes = []
    edges = []
    rows = []
    if framework == "pytorch":
        import torch

        seen = {}
        bindings = {}
        parameters = list(model.named_parameters(remove_duplicate=False))
        buffers = list(model.named_buffers(remove_duplicate=False))
        versions = [(v, v._version) for _, v in [*parameters, *buffers]]
        for role, items in (("parameter", parameters), ("buffer", buffers)):
            for name, value in items:
                if id(value) in seen:
                    rows[seen[id(value)]]["aliases"].append(name)
                    bindings[name] = rows[seen[id(value)]]["id"]
                    continue
                seen[id(value)] = len(rows)
                bindings[name] = name
                rows.append(
                    tensor_row(
                        value,
                        name,
                        role=role,
                        trainable=role == "parameter" and value.requires_grad,
                    )
                )
        for name, module in model.named_modules(remove_duplicate=False):
            config = {"native_configuration": module.extra_repr()}
            for key in (
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                "output_padding",
                "groups",
                "in_features",
                "out_features",
                "num_features",
                "eps",
                "momentum",
                "affine",
                "track_running_stats",
                "inplace",
                "p",
                "normalized_shape",
            ):
                if hasattr(module, key):
                    config[key] = json_value(getattr(module, key))
            if type(module).get_extra_state is not torch.nn.Module.get_extra_state:
                config["extra_state"] = json_value(module.get_extra_state())
            if not type(module).__module__.startswith("torch."):
                framework_fields = set(torch.nn.Module().__dict__)
                config["declared_attributes"] = {
                    k: _attribute_identity(v)
                    for k, v in module.__dict__.items()
                    if k not in framework_fields
                }
                try:
                    config["code_sha256"] = hashlib.sha256(
                        inspect.getsource(type(module)).encode()
                    ).hexdigest()
                except (OSError, TypeError):
                    raise ValueError(
                        "Custom model definition requires inspectable source"
                    )
            node_id = name or "$model"
            refs = [bindings[k] for k in bindings if k.rpartition(".")[0] == name]
            nodes.append(
                {
                    "id": node_id,
                    "kind": type(module).__module__ + "." + type(module).__qualname__,
                    "config": config,
                    "state_refs": list(dict.fromkeys(refs)),
                    "mode": "train" if module.training else "eval",
                }
            )
            if name:
                edges.append(
                    {
                        "from": name.rpartition(".")[0] or "$model",
                        "to": name,
                        "kind": "contains",
                    }
                )
        if any(v._version != version for v, version in versions):
            raise RuntimeError(
                "Model state changed during capture; use a quiescent update boundary"
            )
        fw = {"name": framework, "version": str(torch.__version__), "backend": "torch"}
        scope = "module_tree"
        limitations = [
            "Module containment is not an executed call graph.",
            "Custom Python side effects and external/global state are outside the adapter state scope; unsupported opaque attributes are rejected.",
        ]
    else:
        import keras, tensorflow as tf

        if not tf.executing_eagerly():
            raise ValueError(
                "Native host capture requires eager execution, not tf.function tracing"
            )
        if not model.built:
            raise ValueError("Unbuilt Keras model has no complete state")
        layers, edges, records = state_bindings(model)
        seen = {id(v): name for v, name, _ in records}
        for value, name, aliases in records:
            rows.append(tensor_row(value, name, role="variable", trainable=value.trainable, aliases=aliases))
        for layer, node_id in layers:
            config = json_value(layer.get_config())
            if node_id == "$model" and identity_policy(model) == OWNER_POLICY:
                config = {**config, "deepbom_state_identity": OWNER_POLICY}
            nodes.append(
                {
                    "id": node_id,
                    "kind": type(layer).__module__ + "." + type(layer).__name__,
                    "config": config,
                    "state_refs": [seen[id(v)] for v in layer.weights if id(v) in seen],
                    "mode": "call_dependent",
                }
            )
        fw = {
            "name": framework,
            "version": tf.__version__,
            "backend": keras.backend.backend(),
        }
        scope = "module_tree"
        limitations = [
            "Layer containment; data dependencies and repeated call occurrences require separate execution evidence.",
            "Only registered model variables are captured; optimizer state is excluded.",
        ]
    return {
        "schema": "deepbom.native_evidence_input.v1",
        "definition": {
            "framework": fw,
            "nodes": nodes,
            "relationships": edges,
            "inputs": input_spec or {},
            "scope": scope,
            "limitations": limitations,
        },
        "tensors": rows,
        "capture": None,
        "options": {"max_values": max_values, "advanced": bool(advanced)},
    }


def snapshot(model, **options):
    return engine("snapshot", snapshot_input(model, **options))


def arrays(model):
    """Independent, named state arrays, including nonpersistent torch buffers."""
    import numpy as np

    data = snapshot_input(model, max_values=0)
    return data


def restore_rows(model, rows):
    import base64, numpy as np

    framework = framework_of(model)
    if framework == "pytorch":
        import torch

        targets = dict(model.named_parameters(remove_duplicate=False))
        targets.update(dict(model.named_buffers(remove_duplicate=False)))
        with torch.no_grad():
            expected = set(targets)
            actual = {n for row in rows for n in [row["id"], *row["aliases"]]}
            if actual != expected:
                raise ValueError("State restoration has missing or unexpected keys")
            for row in rows:
                target = targets[row["id"]]
                raw = bytearray(base64.b64decode(row["data_base64"], validate=True))
                dtypes = {
                    "F16": torch.float16,
                    "BF16": torch.bfloat16,
                    "F32": torch.float32,
                    "F64": torch.float64,
                    "I8": torch.int8,
                    "U8": torch.uint8,
                    "I16": torch.int16,
                    "U16": torch.uint16,
                    "I32": torch.int32,
                    "U32": torch.uint32,
                    "I64": torch.int64,
                    "U64": torch.uint64,
                    "BOOL": torch.bool,
                }
                dtype = dtypes[row["dtype"]]
                if raw:
                    tensor = torch.frombuffer(raw, dtype=dtype).reshape(row["shape"])
                else:
                    tensor = torch.empty(row["shape"], dtype=dtype)
                if list(target.shape) != row["shape"]:
                    raise ValueError("State shape mismatch")
                target.data = tensor.to(target.device).clone()
                if row["role"] == "parameter":
                    target.requires_grad_(row["trainable"])
                for alias in row["aliases"]:
                    if targets[alias] is not target:
                        raise ValueError(
                            "Restored model lost tied parameter/buffer identity"
                        )
    else:
        records = state_bindings(model)[2]
        targets = {name: v for v, key, aliases in records for name in [key, *aliases]}
        if set(targets) != {name for r in rows for name in [r["id"], *r["aliases"]]}:
            raise ValueError("Keras variable correspondence changed")
        dtypes = {
            "F16": "<f2",
            "F32": "<f4",
            "F64": "<f8",
            "I8": "i1",
            "U8": "u1",
            "I16": "<i2",
            "U16": "<u2",
            "I32": "<i4",
            "U32": "<u4",
            "I64": "<i8",
            "U64": "<u8",
            "BOOL": "?",
        }
        import ml_dtypes

        dtypes["BF16"] = ml_dtypes.bfloat16
        assignments = []
        for row in rows:
            if row["dtype"] not in dtypes:
                raise ValueError("Unsupported Keras restoration dtype")
            value = np.frombuffer(
                base64.b64decode(row["data_base64"], validate=True),
                dtype=dtypes[row["dtype"]],
            ).reshape(row["shape"])
            target = targets[row["id"]]
            if list(target.shape) != row["shape"]:
                raise ValueError("Keras state shape mismatch")
            if str(target.dtype) != str(value.dtype):
                raise ValueError("Keras state dtype mismatch")
            if row["role"] != "variable" or bool(target.trainable) != bool(row["trainable"]):
                raise ValueError("Keras state role/trainability mismatch")
            if any(targets[alias] is not target for alias in row["aliases"]):
                raise ValueError("Restored model lost shared variable identity")
            assignments.append((target, value))
        for target, value in assignments:
            target.assign(value)
