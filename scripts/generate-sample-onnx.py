"""Generate the deterministic browser ONNX sample used by contract checks."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "web" / "samples" / "sample_cnn_float.onnx"


def values(shape: tuple[int, ...], divisor: float, offset: int = 0) -> np.ndarray:
    count = int(np.prod(shape))
    raw = ((np.arange(count, dtype=np.float32) + offset) % 29 - 14) / divisor
    return raw.reshape(shape).astype(np.float32)


def main() -> None:
    w1 = values((16, 3, 3, 3), 96.0)
    b1 = values((16,), 192.0, 3)
    w2 = values((32, 16, 3, 3), 128.0, 7)
    b2 = values((32,), 256.0, 11)
    w2[7, :, :, :] = 0.0
    b2[7] = 0.125  # A zero kernel slice is not, by itself, a dead output channel.
    wf = values((10, 32), 160.0, 13)
    bf = values((10,), 320.0, 17)

    nodes = [
        helper.make_node("Conv", ["input", "w1", "b1"], ["c1"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node("MaxPool", ["r1"], ["p1"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Conv", ["p1", "w2", "b2"], ["c2"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c2"], ["r2"]),
        helper.make_node("GlobalAveragePool", ["r2"], ["g"]),
        helper.make_node("Flatten", ["g"], ["fl"]),
        helper.make_node("Gemm", ["fl", "wf", "bf"], ["logits"], transB=1),
        helper.make_node("Softmax", ["logits"], ["probs"]),
    ]
    graph = helper.make_graph(
        nodes,
        "sample_cnn_float",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, [1, 10])],
        [
            numpy_helper.from_array(w1, "w1"),
            numpy_helper.from_array(b1, "b1"),
            numpy_helper.from_array(w2, "w2"),
            numpy_helper.from_array(b2, "b2"),
            numpy_helper.from_array(wf, "wf"),
            numpy_helper.from_array(bf, "bf"),
        ],
    )
    graph.doc_string = "Deterministic floating-point CNN graph used for DeepBOM parser and report contract checks."
    model = helper.make_model(
        graph,
        producer_name="deepbom-sample",
        producer_version="1.0",
        domain="org.deepbom.samples",
        model_version=1,
        doc_string="Synthetic ONNX fixture; not a task-accuracy reference model.",
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    helper.set_model_props(
        model,
        {
            "deepbom.sample.input_basis": "synthetic NCHW FLOAT32 tensor; no image decoder contract",
            "deepbom.sample.purpose": "deterministic static-analysis contract verification",
        },
    )
    OUTPUT.write_bytes(model.SerializeToString())


if __name__ == "__main__":
    main()
