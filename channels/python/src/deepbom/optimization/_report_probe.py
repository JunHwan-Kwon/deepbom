"""Collect shapes on an independent copy during explicitly requested validation.

This adapter records facts only. The common JS engine owns sizes and MACs.
Report export itself never imports a model factory or runs a model.
"""

from __future__ import annotations
from collections import Counter
import copy
from .._native.adapters import framework_of, independent_copy


def capture_shapes(model, args, kwargs):
    if not args and not kwargs:
        return {
            "status": "not_requested",
            "invocations": [],
            "inputs": [],
            "outputs": [],
        }
    fw = framework_of(model)
    test = independent_copy(model)
    rows, counts = [], Counter()

    def tensors(value):
        if isinstance(value, dict):
            return [t for v in value.values() for t in tensors(v)]
        if isinstance(value, (list, tuple)):
            return [t for v in value for t in tensors(v)]
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return [
                {
                    "shape": [int(d) if d is not None else None for d in value.shape],
                    "dtype": str(value.dtype)
                    .replace("torch.", "")
                    .replace("<dtype: '", "")
                    .replace("'>", ""),
                }
            ]
        return []

    def record(subject, inputs, outputs):
        counts[subject] += 1
        rows.append(
            {
                "subject": subject,
                "invocation": counts[subject],
                "inputs": tensors(inputs),
                "outputs": tensors(outputs),
            }
        )

    if fw == "pytorch":
        import torch

        def clone(v):
            if torch.is_tensor(v):
                return v.detach().clone()
            if isinstance(v, dict):
                return {k: clone(x) for k, x in v.items()}
            if isinstance(v, tuple):
                return tuple(clone(x) for x in v)
            if isinstance(v, list):
                return [clone(x) for x in v]
            return copy.deepcopy(v)

        args, kwargs = clone(args), clone(kwargs)
        handles = []

        def hook(subject):
            return lambda module, a, kw, output: record(subject, (a, kw), output)

        try:
            for name, layer in test.named_modules():
                if not list(layer.children()):
                    handles.append(
                        layer.register_forward_hook(
                            hook(name or "$model"), with_kwargs=True
                        )
                    )
            devices = sorted({p.device.index for p in test.parameters() if p.is_cuda})
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                output = test(*args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
    else:
        import tensorflow as tf

        if not tf.executing_eagerly():
            raise ValueError("Report shape capture requires eager execution")
        changed, seen = [], set()

        def walk(layer, subject):
            if id(layer) in seen:
                return
            seen.add(id(layer))
            children = layer._flatten_layers(include_self=False, recursive=False)
            if children:
                for child in children:
                    walk(
                        child,
                        child.name if layer is test else subject + "/" + child.name,
                    )
                return
            original = layer.call
            had_instance_call = "call" in layer.__dict__

            def call(*a, **kw):
                result = original(*a, **kw)
                record(subject, (a, kw), result)
                return result

            changed.append((layer, original, had_instance_call))
            layer.call = call

        try:
            walk(test, "$model")
            output = test(*args, **kwargs)
        finally:
            for layer, original, had_instance_call in changed:
                if had_instance_call:
                    layer.call = original
                else:
                    del layer.call
    return {
        "status": "observed",
        "invocations": rows,
        "inputs": tensors((args, kwargs)),
        "outputs": tensors(output),
    }
