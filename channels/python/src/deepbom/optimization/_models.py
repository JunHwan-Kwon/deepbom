"""Explicit transformation and reconstruction rules. No hidden training loop."""

from __future__ import annotations
from collections import OrderedDict
import importlib
import math
from .._native.adapters import (
    framework_of,
    independent_copy,
    snapshot_input,
    restore_rows,
)
from .._native.core import json_value

TORCH_ARGUMENTS = {
    "Linear": ("in_features", "out_features"),
    "Conv1d": (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
    ),
    "Conv2d": (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
    ),
    "Conv3d": (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
    ),
    "BatchNorm1d": ("num_features", "eps", "momentum", "affine", "track_running_stats"),
    "BatchNorm2d": ("num_features", "eps", "momentum", "affine", "track_running_stats"),
    "BatchNorm3d": ("num_features", "eps", "momentum", "affine", "track_running_stats"),
    "ReLU": ("inplace",),
    "ReLU6": ("inplace",),
    "GELU": ("approximate",),
    "LeakyReLU": ("negative_slope", "inplace"),
    "Sigmoid": (),
    "Tanh": (),
    "Identity": (),
    "Flatten": ("start_dim", "end_dim"),
    "Dropout": ("p", "inplace"),
    "AdaptiveAvgPool2d": ("output_size",),
    "MaxPool2d": (
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "return_indices",
        "ceil_mode",
    ),
    "AvgPool2d": (
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
        "divisor_override",
    ),
    "LayerNorm": ("normalized_shape", "eps", "elementwise_affine"),
    "Embedding": (
        "num_embeddings",
        "embedding_dim",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ),
}


def has_sequential_call_path(model, subject):
    """Only built-in Sequential ancestors establish the serialized call order."""
    import torch

    parent = model
    for name in subject.split(".")[:-1]:
        if type(parent) is not torch.nn.Sequential:
            return False
        parent = parent.get_submodule(name)
    return type(parent) is torch.nn.Sequential


def model_spec(model, factory=None):
    fw = framework_of(model)
    if factory:
        if not isinstance(factory, str) or factory.count(":") != 1:
            raise ValueError("factory must be module:function")
        return {"framework": fw, "kind": "factory", "factory": factory}
    if fw == "tensorflow":
        import json

        return {"framework": fw, "kind": "keras", "config": json.loads(model.to_json())}
    import torch

    shared = {}

    def encode(module):
        if id(module) in shared:
            return {"alias": shared[id(module)]}
        key = str(len(shared))
        shared[id(module)] = key
        name = type(module).__name__
        if type(module) is torch.nn.Sequential:
            return {
                "id": key,
                "type": "Sequential",
                "children": [[n, encode(v)] for n, v in module._modules.items()],
            }
        if name not in TORCH_ARGUMENTS or type(module) is not getattr(torch.nn, name):
            raise ValueError(
                f"{type(module).__name__} package export requires an explicit factory; RAM candidates remain supported"
            )
        kwargs = {n: json_value(getattr(module, n)) for n in TORCH_ARGUMENTS[name]}
        if name.startswith("Conv") or name == "Linear":
            kwargs["bias"] = module.bias is not None
        if name == "LayerNorm":
            kwargs["bias"] = module.bias is not None
        return {"id": key, "type": name, "kwargs": kwargs}

    return {"framework": fw, "kind": "torch_builtin", "model": encode(model)}


def construct(spec):
    if spec["kind"] == "factory":
        module, name = spec["factory"].split(":")
        return getattr(importlib.import_module(module), name)()
    if spec["kind"] == "keras":
        import keras, json

        return keras.models.model_from_json(json.dumps(spec["config"]))
    if spec["kind"] != "torch_builtin":
        raise ValueError("Unknown model constructor")
    import torch

    refs = {}

    def decode(row):
        if "alias" in row:
            return refs[row["alias"]]
        name = row["type"]
        if name == "Sequential":
            result = torch.nn.Sequential(
                OrderedDict((n, decode(v)) for n, v in row["children"])
            )
        else:
            if name not in TORCH_ARGUMENTS:
                raise ValueError("Unregistered torch constructor")
            result = getattr(torch.nn, name)(**row["kwargs"])
        refs[row["id"]] = result
        return result

    return decode(spec["model"])


def apply_rules(model, rules):
    candidate = independent_copy(model)
    fw = framework_of(candidate)
    ledger = []
    if not isinstance(rules, list):
        raise TypeError("rules must be an ordered list")
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) - {
            "kind",
            "subject",
            "batchnorm",
            "expansion",
        }:
            raise ValueError("Unknown transformation option")
        kind = rule.get("kind")
        name = rule.get("subject")
        if kind not in ("inverted", "fold_conv_bn"):
            raise ValueError("Supported rules: inverted, fold_conv_bn")
        if not isinstance(name, str) or not name:
            raise ValueError("A transformation needs an exact native subject")
        expansion = rule.get("expansion", 1)
        if (
            isinstance(expansion, bool)
            or not isinstance(expansion, int)
            or not 1 <= expansion <= 16
        ):
            raise ValueError("expansion must be an integer in [1,16]")
        if fw == "pytorch":
            import torch

            conv = candidate.get_submodule(name)
            if type(conv) is not torch.nn.Conv2d:
                raise ValueError("Rule requires torch Conv2d")
            if (
                sum(
                    m is conv
                    for _, m in candidate.named_modules(remove_duplicate=False)
                )
                != 1
            ):
                raise ValueError(
                    "Shared Conv module requires explicit multi-call transformation mapping"
                )
            if kind == "inverted":
                if any(not p.requires_grad for p in conv.parameters()):
                    raise ValueError(
                        "Architecture replacement of frozen state requires an explicit training policy"
                    )
                all_parameters = list(
                    candidate.named_parameters(remove_duplicate=False)
                )
                if any(
                    sum(p is q for _, q in all_parameters) > 1
                    for p in conv.parameters()
                ):
                    raise ValueError(
                        "Tied convolution state requires explicit transformation mapping"
                    )
                if conv.groups != 1 or conv.padding_mode != "zeros":
                    raise ValueError(
                        "Inverted replacement requires dense zero-padded Conv2d"
                    )
                hidden = conv.in_channels * expansion
                replacement = torch.nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "expand",
                                torch.nn.Conv2d(
                                    conv.in_channels, hidden, 1, bias=False
                                ),
                            ),
                            ("expand_relu", torch.nn.ReLU()),
                            (
                                "depthwise",
                                torch.nn.Conv2d(
                                    hidden,
                                    hidden,
                                    conv.kernel_size,
                                    stride=conv.stride,
                                    padding=conv.padding,
                                    dilation=conv.dilation,
                                    groups=hidden,
                                    bias=False,
                                ),
                            ),
                            ("depthwise_relu", torch.nn.ReLU()),
                            (
                                "project",
                                torch.nn.Conv2d(
                                    hidden,
                                    conv.out_channels,
                                    1,
                                    bias=conv.bias is not None,
                                ),
                            ),
                        ]
                    )
                ).to(device=conv.weight.device, dtype=conv.weight.dtype)
                replacement.train(conv.training)
            else:
                bnname = rule.get("batchnorm")
                bn = candidate.get_submodule(bnname or "")
                if (
                    conv.training
                    or bn.training
                    or type(bn) is not torch.nn.BatchNorm2d
                    or bn.running_mean is None
                    or bn.running_var is None
                ):
                    raise ValueError(
                        "Conv-BN folding requires eval modules and running buffers"
                    )
                if (
                    sum(
                        m is bn
                        for _, m in candidate.named_modules(remove_duplicate=False)
                    )
                    != 1
                ):
                    raise ValueError(
                        "Shared BatchNorm requires explicit call-use analysis"
                    )
                # Only proven adjacent Sequential children; arbitrary graph rewrites need call-use analysis.
                parent_path = name.rpartition(".")[0]
                parent = candidate.get_submodule(parent_path)
                names = list(parent._modules)
                if (
                    not has_sequential_call_path(candidate, name)
                    or bnname.rpartition(".")[0] != parent_path
                    or names.index(bnname.rpartition(".")[2])
                    != names.index(name.rpartition(".")[2]) + 1
                ):
                    raise ValueError(
                        "Conv-BN folding requires adjacent children on a built-in Sequential call path"
                    )
                replacement = torch.nn.utils.fuse_conv_bn_eval(conv, bn)
                candidate.set_submodule(bnname, torch.nn.Identity())
            candidate.set_submodule(name, replacement)
        else:
            import keras, numpy as np

            conv = candidate.get_layer(name)
            if (
                type(conv) is not keras.layers.Conv2D
                or conv.groups != 1
                or conv.data_format != "channels_last"
            ):
                raise ValueError("Rule requires dense channels-last Keras Conv2D")
            bn = None
            if kind == "inverted" and not conv.trainable:
                raise ValueError(
                    "Architecture replacement of frozen state requires an explicit training policy"
                )
            if kind == "fold_conv_bn":
                bn = candidate.get_layer(rule.get("batchnorm", ""))
                if (
                    type(bn) is not keras.layers.BatchNormalization
                    or bn.axis not in (-1, 3)
                    or keras.activations.serialize(conv.activation) != "linear"
                ):
                    raise ValueError(
                        "Keras folding requires linear Conv2D and last-axis BatchNormalization"
                    )
                if (
                    getattr(
                        getattr(bn.input, "_keras_history", None), "operation", None
                    )
                    is not conv
                    or len(conv._outbound_nodes) != 1
                ):
                    raise ValueError("Folding requires a unique direct Conv-BN edge")
                if len(bn._inbound_nodes) != 1 or any(
                    output is conv.output for output in candidate.outputs
                ):
                    raise ValueError(
                        "Folding cannot change a separately exposed convolution output"
                    )
            in_channels = int(conv.kernel.shape[-2])
            hidden = in_channels * expansion

            def clone(layer):
                if kind == "fold_conv_bn":
                    if layer is conv:
                        config = layer.get_config()
                        config["use_bias"] = True
                        return keras.layers.Conv2D.from_config(config)
                    if layer is bn:
                        return keras.layers.Identity(
                            name=layer.name, dtype=layer.dtype_policy.name
                        )
                if layer is conv:
                    block = keras.Sequential(
                        [
                            keras.Input(
                                shape=tuple(conv.input.shape)[1:],
                                name=name + "_input",
                                dtype=conv.compute_dtype,
                            ),
                            keras.layers.Conv2D(
                                hidden,
                                1,
                                use_bias=False,
                                dtype=conv.dtype_policy.name,
                                name="expand",
                            ),
                            keras.layers.ReLU(
                                name="expand_relu", dtype=conv.dtype_policy.name
                            ),
                            keras.layers.DepthwiseConv2D(
                                conv.kernel_size,
                                strides=conv.strides,
                                padding=conv.padding,
                                dilation_rate=conv.dilation_rate,
                                use_bias=False,
                                dtype=conv.dtype_policy.name,
                                name="depthwise",
                            ),
                            keras.layers.ReLU(
                                name="depthwise_relu", dtype=conv.dtype_policy.name
                            ),
                            keras.layers.Conv2D(
                                conv.filters,
                                1,
                                use_bias=conv.use_bias,
                                activation=conv.activation,
                                dtype=conv.dtype_policy.name,
                                name="project",
                            ),
                        ],
                        name=name,
                    )
                    return block
                return layer.__class__.from_config(layer.get_config())

            def call(layer, *args, **kwargs):
                if kind == "fold_conv_bn" and layer.name == bn.name:
                    kwargs.pop("mask", None)
                    kwargs.pop("training", None)
                return layer(*args, **kwargs)

            previous = candidate
            if isinstance(previous, keras.Sequential):
                # A Sequential graph has no per-call mask arguments to preserve.
                candidate = keras.models.clone_model(
                    previous, clone_function=clone, recursive=False
                )
            else:
                candidate = keras.models.clone_model(
                    previous, clone_function=clone, call_function=call, recursive=False
                )
            for layer in candidate.layers:
                if kind == "fold_conv_bn" and layer.name == name:
                    scale = (
                        bn.gamma.numpy()
                        if bn.scale
                        else np.ones_like(bn.moving_mean.numpy())
                    ) / np.sqrt(bn.moving_variance.numpy() + bn.epsilon)
                    bias = conv.bias.numpy() if conv.use_bias else np.zeros_like(scale)
                    beta = bn.beta.numpy() if bn.center else np.zeros_like(scale)
                    layer.set_weights(
                        [
                            conv.kernel.numpy() * scale.reshape((1, 1, 1, -1)),
                            (bias - bn.moving_mean.numpy()) * scale + beta,
                        ]
                    )
                elif layer.name != name and (bn is None or layer.name != bn.name):
                    source = previous.get_layer(layer.name)
                    layer.set_weights(source.get_weights())
                    layer.trainable = source.trainable
        ledger.append(
            {
                "rule": kind,
                "rule_version": "1.0.0",
                "subject": name,
                "scope": (
                    "inference_equivalent_under_eval_preconditions"
                    if kind == "fold_conv_bn"
                    else "architecture_changed_requires_training_and_task_evaluation"
                ),
                "state": (
                    "transformed"
                    if kind == "fold_conv_bn"
                    else "replacement_initialized_unaffected_state_preserved"
                ),
                "request": rule,
            }
        )
    return candidate, ledger


def restore_model(spec, rows, modes, rules=()):
    model = construct(spec)
    if framework_of(model) == "pytorch":
        for name, mode in spec.get("modes", {}).items():
            model.get_submodule(name).training = mode
    if rules:
        model, _ = apply_rules(model, list(rules))
    restore_rows(model, rows)
    if framework_of(model) == "pytorch":
        modules = dict(model.named_modules(remove_duplicate=False))
        for name, mode in modes.items():
            modules[name].training = mode
    return model
