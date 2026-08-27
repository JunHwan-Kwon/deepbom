export const TENSORRT_LLM_SOURCE_METADATA = Object.freeze({
  repository: "NVIDIA/TensorRT-LLM",
  release: "v1.2.0",
  commit: "51f5ef32b695f421dd27727f3935052015c945f9",
  files: Object.freeze([
    Object.freeze({
      path: "tensorrt_llm/builder.py",
      sha256: "306ced17abe40cc8d25fa1acc749a54f3896e7f97940c2f2d2e4ab8529913eb1",
      url: "https://github.com/NVIDIA/TensorRT-LLM/blob/51f5ef32b695f421dd27727f3935052015c945f9/tensorrt_llm/builder.py",
      purpose: "EngineConfig serialization, BuildConfig limits, KV-cache type, plugin configuration, and weight-streaming declaration.",
    }),
    Object.freeze({
      path: "tensorrt_llm/models/modeling_utils.py",
      sha256: "dc739812818f215be4ee625a11ae2418df9d18e79542385952f3a9f0f01ff442",
      url: "https://github.com/NVIDIA/TensorRT-LLM/blob/51f5ef32b695f421dd27727f3935052015c945f9/tensorrt_llm/models/modeling_utils.py",
      purpose: "PretrainedConfig architecture dimensions, distributed mapping, and quantization/KV-cache quantization fields.",
    }),
    Object.freeze({
      path: "tensorrt_llm/mapping.py",
      sha256: "ef149fde47ab5bd23a8a4d88da6fa61752d5213e5d425f1d08bead9545e57d42",
      url: "https://github.com/NVIDIA/TensorRT-LLM/blob/51f5ef32b695f421dd27727f3935052015c945f9/tensorrt_llm/mapping.py",
      purpose: "world_size = tp_size x pp_size x cp_size and pipeline-layer partition rules.",
    }),
    Object.freeze({
      path: "tensorrt_llm/llmapi/kv_cache_type.py",
      sha256: "761cd0a9412bba588c741c82dcbe64433eba5f863db323310c5f71c539b6e917",
      url: "https://github.com/NVIDIA/TensorRT-LLM/blob/51f5ef32b695f421dd27727f3935052015c945f9/tensorrt_llm/llmapi/kv_cache_type.py",
      purpose: "Serialized KV-cache type vocabulary.",
    }),
  ]),
});
