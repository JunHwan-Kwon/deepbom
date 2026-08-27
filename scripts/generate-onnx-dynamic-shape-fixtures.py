"""Generate deterministic ONNX fixtures for dynamic-shape cost contracts."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "scripts" / "fixtures"


def dynamic_conv() -> onnx.ModelProto:
    weights = (np.arange(16 * 3 * 3 * 3, dtype=np.float32) % 17 - 8).reshape(16, 3, 3, 3) / 64.0
    node = helper.make_node("Conv", ["input", "weight"], ["output"], name="dynamic_conv")
    graph = helper.make_graph(
        [node],
        "dynamic_conv_cost",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 32, 32])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 16, 30, 30])],
        [numpy_helper.from_array(weights, "weight")],
    )
    return helper.make_model(
        graph,
        producer_name="deepbom-dynamic-cost-fixture",
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 13)],
    )


def zero_dim_identity() -> onnx.ModelProto:
    node = helper.make_node("Identity", ["input"], ["output"], name="zero_dim_identity")
    graph = helper.make_graph(
        [node],
        "zero_dim_identity",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [0, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [0, 3])],
    )
    return helper.make_model(
        graph,
        producer_name="deepbom-zero-dimension-fixture",
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 13)],
    )


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "onnx_dynamic_conv.onnx").write_bytes(dynamic_conv().SerializeToString())
    (FIXTURES / "onnx_zero_dim_identity.onnx").write_bytes(zero_dim_identity().SerializeToString())


if __name__ == "__main__":
    main()
