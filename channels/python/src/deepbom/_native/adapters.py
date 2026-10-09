"""Native facts and independent state copies; no tensor statistics here."""

from __future__ import annotations
import copy
import hashlib
import inspect
from .core import engine, json_value, tensor_row


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


def independent_copy(model):
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
    result = keras.models.clone_model(model)
    result.set_weights([x.copy() for x in model.get_weights()])
    # Validate trainability and state cardinality after clone, not only shapes.
    if len(result.weights) != len(model.weights):
        raise ValueError("Keras clone changed state cardinality")
    for old, new in zip(model.layers, result.layers):
        new.trainable = old.trainable
    return result


def snapshot_input(model, *, input_spec=None, max_values=1_000_000, advanced=False):
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
        fw = {"name": framework, "version": torch.__version__, "backend": "torch"}
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
        seen = {}
        bindings = {}
        for i, v in enumerate(model.weights):
            name = getattr(v, "path", v.name)
            if name in bindings:
                raise ValueError("Ambiguous Keras variable path")
            bindings[name] = name
            seen[id(v)] = name
            rows.append(tensor_row(v, name, role="variable", trainable=v.trainable))
        layers = []
        visited = set()

        def walk(layer, path, parent=None):
            if id(layer) in visited:
                return
            visited.add(id(layer))
            layers.append((layer, path))
            if parent is not None:
                edges.append({"from": parent, "to": path, "kind": "contains"})
            children = layer._flatten_layers(include_self=False, recursive=False)
            for child in children:
                walk(
                    child,
                    child.name if layer is model else path + "/" + child.name,
                    path,
                )

        walk(model, "$model")
        for layer, node_id in layers:
            config = json_value(layer.get_config())
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
        targets = {getattr(v, "path", v.name): v for v in model.weights}
        if set(targets) != {r["id"] for r in rows}:
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
            target.assign(value)
