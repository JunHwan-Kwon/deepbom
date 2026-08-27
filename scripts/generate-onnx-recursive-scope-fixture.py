"""Generate the deterministic ONNX recursive-scope parser fixture."""

from pathlib import Path

import onnx
from onnx import AttributeProto, TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "scripts" / "fixtures" / "onnx_recursive_scope.onnx"


def value(name: str, dtype: int, shape: list[int] | None):
    return helper.make_tensor_value_info(name, dtype, shape)


def main() -> None:
    transpose_node = helper.make_node("Transpose", ["formal_x"], ["formal_y"])
    transpose_node.attribute.append(helper.make_attribute_ref("perm", AttributeProto.INTS))
    transpose_function = helper.make_function(
        "local.deepbom",
        "TransposeBy",
        ["formal_x"],
        ["formal_y"],
        [transpose_node],
        [helper.make_opsetid("", 13)],
        attributes=["perm"],
        value_info=[value("formal_x", TensorProto.FLOAT, [None, None]), value("formal_y", TensorProto.FLOAT, [None, None])],
    )

    then_graph = helper.make_graph(
        [helper.make_node("Identity", ["function_out"], ["then_out"])],
        "then_branch",
        [],
        [value("then_out", TensorProto.FLOAT, [None, None])],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Identity", ["function_out"], ["else_out"])],
        "else_branch",
        [],
        [value("else_out", TensorProto.FLOAT, [None, None])],
    )
    loop_body = helper.make_graph(
        [
            helper.make_node("Identity", ["loop_cond_in"], ["loop_cond_out"]),
            helper.make_node("Identity", ["loop_state_in"], ["loop_state_out"]),
        ],
        "loop_body",
        [value("iter", TensorProto.INT64, []), value("loop_cond_in", TensorProto.BOOL, []), value("loop_state_in", TensorProto.FLOAT, [None, None])],
        [value("loop_cond_out", TensorProto.BOOL, []), value("loop_state_out", TensorProto.FLOAT, [None, None])],
    )
    scan_body = helper.make_graph(
        [
            helper.make_node("Identity", ["scan_state_in"], ["scan_state_out"]),
            helper.make_node("Identity", ["scan_element_in"], ["scan_element_out"]),
        ],
        "scan_body",
        [value("scan_state_in", TensorProto.FLOAT, [None, None]), value("scan_element_in", TensorProto.FLOAT, [None, None])],
        [value("scan_state_out", TensorProto.FLOAT, [None, None]), value("scan_element_out", TensorProto.FLOAT, [None, None])],
    )

    nodes = [
        helper.make_node("TransposeBy", ["x"], ["function_out"], domain="local.deepbom", perm=[1, 0]),
        helper.make_node("If", ["if_cond"], ["if_out"], then_branch=then_graph, else_branch=else_graph),
        helper.make_node("Loop", ["trip_count", "loop_cond", "loop_state"], ["loop_final"], body=loop_body),
        helper.make_node("Scan", ["scan_state", "scan_sequence"], ["scan_final", "scan_sequence_out"], body=scan_body, num_scan_inputs=1),
    ]
    graph = helper.make_graph(
        nodes,
        "recursive_scope_fixture",
        [
            value("x", TensorProto.FLOAT, [2, 3]),
            value("if_cond", TensorProto.BOOL, []),
            value("trip_count", TensorProto.INT64, []),
            value("loop_cond", TensorProto.BOOL, []),
            value("loop_state", TensorProto.FLOAT, [2, 3]),
            value("scan_state", TensorProto.FLOAT, [2, 3]),
            value("scan_sequence", TensorProto.FLOAT, [5, 2, 3]),
        ],
        [
            value("if_out", TensorProto.FLOAT, [None, None]),
            value("loop_final", TensorProto.FLOAT, [None, None]),
            value("scan_final", TensorProto.FLOAT, [None, None]),
            value("scan_sequence_out", TensorProto.FLOAT, [None, None, None]),
        ],
    )
    model = helper.make_model(
        graph,
        functions=[transpose_function],
        producer_name="deepbom-recursive-scope-fixture",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("local.deepbom", 1)],
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(model.SerializeToString())


if __name__ == "__main__":
    main()
