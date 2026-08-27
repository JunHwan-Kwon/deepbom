from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "samples"


def values(shape, start=1, scale=0.01):
    count = int(np.prod(shape))
    data = (np.arange(start, start + count, dtype=np.float32) % 29 - 14) * scale
    return data.reshape(shape)


def initializer(name, shape, start=1, scale=0.01):
    return numpy_helper.from_array(values(shape, start, scale), name=name)


def scalar(name, value):
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def value_info(name, dtype, shape):
    return helper.make_tensor_value_info(name, dtype, shape)


def metadata(model, role, claim_boundary):
    model.producer_name = "DEEPBOM deterministic sample generator"
    model.producer_version = "1.0.0"
    model.domain = "org.deepbom.samples"
    model.model_version = 1
    for key, value in {
        "deepbom.sample.role": role,
        "deepbom.sample.claim_boundary": claim_boundary,
        "deepbom.sample.weights": "deterministic synthetic values; not trained parameters",
    }.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    return model


def save(model, filename):
    destination = OUT / filename
    destination.write_bytes(model.SerializeToString(deterministic=True))
    print(f"{destination.name}: {destination.stat().st_size} bytes")


def tensorrt_supported_probe():
    nodes = [
        helper.make_node("Conv", ["image", "conv_w", "conv_b"], ["conv"], name="conv", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv"], ["relu"], name="relu"),
        helper.make_node("GlobalAveragePool", ["relu"], ["pooled"], name="global_average_pool"),
        helper.make_node("Flatten", ["pooled"], ["flat"], name="flatten", axis=1),
        helper.make_node("Gemm", ["flat", "head_w", "head_b"], ["logits"], name="classifier"),
    ]
    graph = helper.make_graph(
        nodes,
        "deepbom_tensorrt_supported_probe",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4])],
        [
            initializer("conv_w", [4, 3, 3, 3], 1),
            initializer("conv_b", [4], 109),
            initializer("head_w", [4, 4], 113),
            initializer("head_b", [4], 129),
        ],
        value_info=[
            value_info("conv", TensorProto.FLOAT, [1, 4, 8, 8]),
            value_info("relu", TensorProto.FLOAT, [1, 4, 8, 8]),
            value_info("pooled", TensorProto.FLOAT, [1, 4, 1, 1]),
            value_info("flat", TensorProto.FLOAT, [1, 4]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    save(metadata(model, "tensorrt_supported_gpu_probe", "Parser acceptance is not engine-build, tactic, latency, accuracy, or safety evidence."), "tensorrt_supported_probe.onnx")


def gpu_boundary_probe():
    nodes = [
        helper.make_node("Conv", ["image", "conv_w", "conv_b"], ["conv"], name="conv", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv"], ["relu"], name="relu"),
        helper.make_node("NonZero", ["relu"], ["indices"], name="dynamic_nonzero_boundary"),
        helper.make_node("Cast", ["indices"], ["indices_f32"], name="cast_indices", to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["indices_f32", "axes"], ["index_sum"], name="reduce_indices", keepdims=0),
    ]
    graph = helper.make_graph(
        nodes,
        "deepbom_gpu_boundary_probe",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("index_sum", TensorProto.FLOAT, [])],
        [initializer("conv_w", [4, 3, 3, 3], 1), initializer("conv_b", [4], 109), numpy_helper.from_array(np.asarray([0, 1], dtype=np.int64), name="axes")],
        value_info=[
            value_info("conv", TensorProto.FLOAT, [1, 4, 8, 8]),
            value_info("relu", TensorProto.FLOAT, [1, 4, 8, 8]),
            value_info("indices", TensorProto.INT64, [4, None]),
            value_info("indices_f32", TensorProto.FLOAT, [4, None]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    save(metadata(model, "gpu_partition_boundary_probe", "The graph intentionally contains dynamic NonZero output; backend support must be observed per pinned runtime."), "gpu_partition_probe.onnx")


def tiny_decoder_llm_probe():
    nodes = [
        helper.make_node("Gather", ["token_embedding", "input_ids"], ["hidden"], name="token_embedding_lookup", axis=0),
        helper.make_node("LayerNormalization", ["hidden", "ln1_scale", "ln1_bias"], ["norm1"], name="attention_norm", axis=-1, epsilon=1e-5),
        helper.make_node("MatMul", ["norm1", "q_w"], ["query"], name="query_projection"),
        helper.make_node("MatMul", ["norm1", "k_w"], ["key"], name="key_projection"),
        helper.make_node("Transpose", ["key"], ["key_t"], name="key_transpose", perm=[0, 2, 1]),
        helper.make_node("MatMul", ["query", "key_t"], ["attention_scores"], name="attention_scores"),
        helper.make_node("Div", ["attention_scores", "attention_scale"], ["scaled_scores"], name="attention_scale"),
        helper.make_node("Softmax", ["scaled_scores"], ["attention_probabilities"], name="attention_softmax", axis=-1),
        helper.make_node("MatMul", ["norm1", "v_w"], ["value"], name="value_projection"),
        helper.make_node("MatMul", ["attention_probabilities", "value"], ["context"], name="attention_context"),
        helper.make_node("MatMul", ["context", "o_w"], ["attention_output"], name="output_projection"),
        helper.make_node("Add", ["hidden", "attention_output"], ["residual1"], name="attention_residual"),
        helper.make_node("LayerNormalization", ["residual1", "ln2_scale", "ln2_bias"], ["norm2"], name="feed_forward_norm", axis=-1, epsilon=1e-5),
        helper.make_node("MatMul", ["norm2", "ff_up_w"], ["ff_up"], name="feed_forward_up"),
        helper.make_node("Relu", ["ff_up"], ["ff_activation"], name="feed_forward_activation"),
        helper.make_node("MatMul", ["ff_activation", "ff_down_w"], ["ff_down"], name="feed_forward_down"),
        helper.make_node("Add", ["residual1", "ff_down"], ["last_hidden_state"], name="feed_forward_residual"),
        helper.make_node("Identity", ["past_key"], ["present_key"], name="serialized_state_passthrough"),
    ]
    graph = helper.make_graph(
        nodes,
        "deepbom_tiny_decoder_llm_probe",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 4]),
            helper.make_tensor_value_info("past_key", TensorProto.FLOAT, [1, 2, 2, 4]),
        ],
        [
            helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, [1, 4, 8]),
            helper.make_tensor_value_info("present_key", TensorProto.FLOAT, [1, 2, 2, 4]),
        ],
        [
            initializer("token_embedding", [32, 8], 1),
            numpy_helper.from_array(np.ones([8], dtype=np.float32), name="ln1_scale"),
            numpy_helper.from_array(np.zeros([8], dtype=np.float32), name="ln1_bias"),
            numpy_helper.from_array(np.ones([8], dtype=np.float32), name="ln2_scale"),
            numpy_helper.from_array(np.zeros([8], dtype=np.float32), name="ln2_bias"),
            initializer("q_w", [8, 8], 257),
            initializer("k_w", [8, 8], 321),
            initializer("v_w", [8, 8], 385),
            initializer("o_w", [8, 8], 449),
            initializer("ff_up_w", [8, 16], 513),
            initializer("ff_down_w", [16, 8], 641),
            scalar("attention_scale", np.sqrt(8.0)),
        ],
        value_info=[
            value_info("hidden", TensorProto.FLOAT, [1, 4, 8]),
            value_info("norm1", TensorProto.FLOAT, [1, 4, 8]),
            value_info("query", TensorProto.FLOAT, [1, 4, 8]),
            value_info("key", TensorProto.FLOAT, [1, 4, 8]),
            value_info("key_t", TensorProto.FLOAT, [1, 8, 4]),
            value_info("attention_scores", TensorProto.FLOAT, [1, 4, 4]),
            value_info("scaled_scores", TensorProto.FLOAT, [1, 4, 4]),
            value_info("attention_probabilities", TensorProto.FLOAT, [1, 4, 4]),
            value_info("value", TensorProto.FLOAT, [1, 4, 8]),
            value_info("context", TensorProto.FLOAT, [1, 4, 8]),
            value_info("attention_output", TensorProto.FLOAT, [1, 4, 8]),
            value_info("residual1", TensorProto.FLOAT, [1, 4, 8]),
            value_info("norm2", TensorProto.FLOAT, [1, 4, 8]),
            value_info("ff_up", TensorProto.FLOAT, [1, 4, 16]),
            value_info("ff_activation", TensorProto.FLOAT, [1, 4, 16]),
            value_info("ff_down", TensorProto.FLOAT, [1, 4, 8]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    save(metadata(model, "serialized_decoder_llm_graph_probe", "The graph establishes only serialized attention/state evidence, not architecture identity, tokenizer, generation policy, task quality, or clinical validity."), "tiny_decoder_llm.onnx")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    tensorrt_supported_probe()
    gpu_boundary_probe()
    tiny_decoder_llm_probe()
