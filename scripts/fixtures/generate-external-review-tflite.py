#!/usr/bin/env python3
"""Generate small, redistributable TFLite correctness fixtures.

Generation is intentionally separate from ordinary CI because TensorFlow is not
a runtime dependency. CI consumes the generated, hash-pinned FlatBuffers.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb


EXPECTED_TENSORFLOW = "2.16.1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def convert(module: tf.Module, concrete: tf.types.experimental.ConcreteFunction, configure=None) -> bytes:
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], module)
    if configure:
        configure(converter)
    return converter.convert()


class SpaceToBatchModel(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([1, 20, 20, 16], tf.float32, name="input")])
    @tf.autograph.experimental.do_not_convert
    def __call__(self, value):
        blocked = tf.space_to_batch(value, paddings=[[0, 0], [0, 0]], block_shape=[2, 2])
        activated = tf.math.sin(blocked)
        kernel = tf.constant(np.ones((3, 3, 16, 16), dtype=np.float32))
        return {"output": tf.nn.conv2d(activated, kernel, strides=[1, 1, 1, 1], padding="VALID")}


class Conv16x8Model(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([1, 16, 16, 8], tf.float32, name="input")])
    @tf.autograph.experimental.do_not_convert
    def __call__(self, value):
        kernel = tf.constant(
            np.arange(3 * 3 * 8 * 8, dtype=np.float32).reshape(3, 3, 8, 8) / 576.0
        )
        return {"output": tf.nn.conv2d(value, kernel, strides=[1, 1, 1, 1], padding="VALID")}


class DynamicReshapeModel(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([1, None, 32], tf.float32, name="input")])
    @tf.autograph.experimental.do_not_convert
    def __call__(self, value):
        flattened = tf.reshape(value, [-1, 32])
        weight = tf.constant(
            np.arange(32 * 128, dtype=np.float32).reshape(32, 128) / 4096.0
        )
        projected = tf.matmul(flattened, weight)
        return {"output": tf.reshape(projected, [1, -1, 128])}


def patch_space_to_batch_placeholders(model_bytes: bytes) -> bytes:
    data = bytearray(model_bytes)
    model = schema_fb.Model.GetRootAsModel(data, 0)
    subgraph = model.Subgraphs(0)
    affected = set()
    for index in range(subgraph.OperatorsLength()):
        operator = subgraph.Operators(index)
        builtin = model.OperatorCodes(operator.OpcodeIndex()).BuiltinCode()
        if builtin in (3, 38, 66):  # CONV_2D, SPACE_TO_BATCH_ND, SIN
            affected.update(operator.Outputs(slot) for slot in range(operator.OutputsLength()))
    if len(affected) != 3:
        raise RuntimeError(f"unexpected SPACE_TO_BATCH fixture graph: outputs={sorted(affected)}")
    for index in sorted(affected):
        tensor = subgraph.Tensors(index)
        if tensor.ShapeLength() != 4 or tensor.Shape(0) != 4:
            raise RuntimeError(f"tensor {index} does not carry the expected batch=4 shape")
        shape_offset = tensor._tab.Offset(4)
        if shape_offset == 0:
            raise RuntimeError(f"tensor {index} shape vector is absent")
        vector = tensor._tab.Vector(shape_offset)
        data[vector : vector + 4] = (1).to_bytes(4, "little", signed=True)
    return bytes(data)


def configure_16x8(converter):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data():
        for index in range(8):
            yield [np.full((1, 16, 16, 8), (index - 4) / 8.0, dtype=np.float32)]

    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    ]
    converter.inference_input_type = tf.int16
    converter.inference_output_type = tf.int16


def write_fixture(output: Path, name: str, data: bytes, expected: dict) -> dict:
    path = output / name
    path.write_bytes(data)
    return {
        "path": name,
        "sha256": sha256(data),
        "bytes": len(data),
        "expected": expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if tf.__version__ != EXPECTED_TENSORFLOW:
        raise RuntimeError(f"TensorFlow {EXPECTED_TENSORFLOW} is required, found {tf.__version__}")
    args.output.mkdir(parents=True, exist_ok=True)

    space = SpaceToBatchModel()
    space_bytes = convert(space, space.__call__.get_concrete_function())
    stale_space_bytes = patch_space_to_batch_placeholders(space_bytes)
    quant = Conv16x8Model()
    quant_bytes = convert(quant, quant.__call__.get_concrete_function(), configure_16x8)
    dynamic = DynamicReshapeModel()
    dynamic_bytes = convert(dynamic, dynamic.__call__.get_concrete_function())

    fixtures = [
        write_fixture(
            args.output,
            "space-to-batch-stale-shapes.tflite",
            stale_space_bytes,
            {"serialized_conv_macs": 147456, "source_derived_conv_macs": 589824},
        ),
        write_fixture(
            args.output,
            "conv-16x8.tflite",
            quant_bytes,
            {"classification": "full_integer_16x8", "total_macs": 112896},
        ),
        write_fixture(
            args.output,
            "dynamic-reshape.tflite",
            dynamic_bytes,
            {"numeric_total_macs": None, "symbolic_total_macs": "4096*D2"},
        ),
    ]
    manifest = {
        "schema": "deepbom.external_review_tflite_fixtures.v1",
        "generator": {"tensorflow": tf.__version__, "numpy": np.__version__},
        "fixtures": fixtures,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
