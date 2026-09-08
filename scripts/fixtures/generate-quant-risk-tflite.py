#!/usr/bin/env python3
"""Generate the external-review quantization-risk fixture.

This generator is intentionally separate from the TensorFlow 2.16.1 boundary
fixtures. CI consumes the generated FlatBuffer and verifies its manifest; it
does not install TensorFlow.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


EXPECTED_TENSORFLOW = "2.11.1"
EXPECTED_NUMPY = "1.26.4"


class QuantizationRiskModel(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([1, 4, 4, 1], tf.float32, name="input")])
    @tf.autograph.experimental.do_not_convert
    def __call__(self, value):
        kernel = tf.constant(
            np.array([1.0e-7, 1.0], dtype=np.float32).reshape(1, 1, 1, 2)
        )
        return {"output": tf.nn.conv2d(value, kernel, strides=[1, 1, 1, 1], padding="VALID")}


def representative_data():
    for index in range(8):
        yield [np.full((1, 4, 4, 1), (index - 4) / 8.0, dtype=np.float32)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if tf.__version__ != EXPECTED_TENSORFLOW or np.__version__ != EXPECTED_NUMPY:
        raise RuntimeError(
            f"TensorFlow {EXPECTED_TENSORFLOW} and NumPy {EXPECTED_NUMPY} are required; "
            f"found TensorFlow {tf.__version__} and NumPy {np.__version__}"
        )

    model = QuantizationRiskModel()
    concrete = model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    data = converter.convert()

    args.output.mkdir(parents=True, exist_ok=True)
    fixture = args.output / "quant-scale-risk.tflite"
    fixture.write_bytes(data)
    manifest = {
        "schema": "deepbom.external_review_quant_risk_fixture.v1",
        "generator": {"tensorflow": tf.__version__, "numpy": np.__version__},
        "fixture": {
            "path": fixture.name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "expected": {
                "maximum_quantization_risk": "risk",
                "minimum_scale_ratio": 100000.0,
            },
        },
    }
    (args.output / "quant-scale-risk.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
