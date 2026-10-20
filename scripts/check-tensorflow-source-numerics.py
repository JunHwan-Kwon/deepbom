"""Compare serialized TensorProto evidence to TensorFlow's actual decoder."""
import json
import tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
import deepbom
from tensorflow.core.framework import graph_pb2, node_def_pb2, tensor_pb2, types_pb2


with tempfile.TemporaryDirectory(prefix="deepbom-tf-source-") as temporary:
    root = Path(temporary)
    protos = [
        tf.make_tensor_proto(np.array([[1., -2.], [0., 4.]], dtype=np.float32)),
        tensor_pb2.TensorProto(dtype=types_pb2.DT_FLOAT, float_val=[2.5]),
        tensor_pb2.TensorProto(dtype=types_pb2.DT_INT32, int_val=[-2, 3]),
        tensor_pb2.TensorProto(dtype=types_pb2.DT_INT64, int64_val=[-(2**60), 2**60]),
        tensor_pb2.TensorProto(dtype=types_pb2.DT_HALF, half_val=[0x3c00, 0xc000]),
        tensor_pb2.TensorProto(dtype=types_pb2.DT_BOOL, bool_val=[True, False]),
        tensor_pb2.TensorProto(dtype=types_pb2.DT_FLOAT),
    ]
    for index, proto in enumerate(protos):
        if index:
            proto.tensor_shape.dim.add(size=4)
        graph = graph_pb2.GraphDef()
        const = graph.node.add(name="value", op="Const")
        const.attr["value"].tensor.CopyFrom(proto)
        path = root / f"case-{index}.pb"
        path.write_bytes(graph.SerializeToString())
        result = deepbom.numerical_evidence(path, weights=True)["sections"]
        row = result["weight_ir"]["tensors"][0]
        values = tf.make_ndarray(proto).reshape(-1)
        assert row["status"] == "assessed", row
        assert row["statistics"]["value_count"] == str(values.size)
        assert row["statistics"]["zero_count"] == str(np.count_nonzero(values == 0))
        if index != 3:
            assert row["statistics"]["minimum"] == float(values.min())
            assert row["statistics"]["maximum"] == float(values.max())
        else:
            assert int(row["statistics"]["unsafe_integer_count"]) > 0
            assert result["weight_analysis"]["tensors"][0]["status"] == "not_assessed"

    # Shared layer calls and nonzero output ports must remain separate.
    a = keras.Input((4,), name="a")
    b = keras.Input((4,), name="b")
    shared = keras.layers.Dense(3, name="shared")
    nested = keras.Sequential([keras.layers.Dense(2, name="inner")], name="nested")
    model = keras.Model([a, b], [nested(shared(a)), nested(shared(b))])
    path = root / "shared.keras"
    model.save(path)
    summary = deepbom.model_ir(path)
    # model_ir() returns the selected common Model IR, not a framework graph.
    program = summary.get("program", summary.get("sections", {}).get("model_ir", {}).get("program"))
    assert program, summary.keys()
    nodes = program["operations"]
    names = [str(node) for node in nodes]
    assert len(nodes) == 6, len(nodes)
    assert any("shared@0" in n for n in names) and any("shared@1" in n for n in names)
    assert any("nested@0" in n for n in names) and any("nested@1" in n for n in names)
    print("TensorFlow raw/typed/repeated/half/bool/zero/unsafe-int evidence and actual Keras shared-call export passed.")
