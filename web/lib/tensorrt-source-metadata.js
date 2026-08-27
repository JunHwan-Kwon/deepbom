export const TENSORRT_SOURCE_METADATA = Object.freeze({
  schema: "deepbom.tensorrt_source_metadata.v1",
  generated_at: "2026-08-19T00:00:00.000Z",
  tensorrt: {
    repository: "NVIDIA/TensorRT",
    source_commit: "10d15ae2f3b21437caf8248ba317cf76e0c0ebcb",
    files: [
      {
        path: "include/NvOnnxParser.h",
        sha256: "1443c9411e030ac35b8a549392745731515485367579ace586156806ece11d60",
        source_ref: "https://github.com/NVIDIA/TensorRT/blob/10d15ae2f3b21437caf8248ba317cf76e0c0ebcb/include/NvOnnxParser.h",
        scope: "supportsModelV2, getNbSubgraphs, isSubgraphSupported, getSubgraphNodes, and parser-error API contracts",
      },
      {
        path: "include/NvInferRuntime.h",
        sha256: "dd795beae7f612066897f8c3db2314fdfee219e7efa67e0a5781d1061dae3643",
        source_ref: "https://github.com/NVIDIA/TensorRT/blob/10d15ae2f3b21437caf8248ba317cf76e0c0ebcb/include/NvInferRuntime.h",
        scope: "runtime, engine-inspector, and deserialization trust-boundary API contracts",
      },
    ],
  },
  onnx_tensorrt_legacy_parser: {
    repository: "onnx/onnx-tensorrt",
    source_commit: "6ba67d3428e05f690145373ca87fb8d32f98df45",
    files: [
      {
        path: "NvOnnxParser.h",
        sha256: "9b9b1dea4f7f155775c8222c361721cf9aff15cb1cc40dde05202a5647c9c954",
        source_ref: "https://github.com/onnx/onnx-tensorrt/blob/6ba67d3428e05f690145373ca87fb8d32f98df45/NvOnnxParser.h",
        scope: "TensorRT 8.6.1 supportsModel and SubGraphCollection_t capability contract",
      },
      {
        path: "ModelImporter.cpp",
        sha256: "f931080006a4601d4dd08b32db2e838fcf1884c49cc97d4d1c867fed30b22f76",
        source_ref: "https://github.com/onnx/onnx-tensorrt/blob/6ba67d3428e05f690145373ca87fb8d32f98df45/ModelImporter.cpp",
        scope: "TensorRT 8.6.1 supported-subgraph collection membership and legacy boolean-flag behavior",
      },
    ],
  },
  onnxruntime_tensorrt_ep: {
    repository: "microsoft/onnxruntime",
    version: "1.23.2",
    source_commit: "a83fc4d58cb48eb68890dd689f94f28288cf2278",
    files: [
      {
        path: "include/onnxruntime/core/providers/tensorrt/tensorrt_provider_options.h",
        sha256: "6152c06856b7e23b2e229fc9445090970e068e3b1ea6c7aad1e0265dfbff7809",
        source_ref: "https://github.com/microsoft/onnxruntime/blob/a83fc4d58cb48eb68890dd689f94f28288cf2278/include/onnxruntime/core/providers/tensorrt/tensorrt_provider_options.h",
        scope: "public TensorRT EP provider-option structure",
      },
      {
        path: "onnxruntime/core/providers/tensorrt/tensorrt_execution_provider.h",
        sha256: "21a21952c2ced9c7e257b6be9ae3f9a9481667eeea03b1faf13079a6c8c112b6",
        source_ref: "https://github.com/microsoft/onnxruntime/blob/a83fc4d58cb48eb68890dd689f94f28288cf2278/onnxruntime/core/providers/tensorrt/tensorrt_execution_provider.h",
        scope: "TensorRT EP configuration and GetCapability implementation surface",
      },
      {
        path: "onnxruntime/core/providers/tensorrt/tensorrt_execution_provider.cc",
        sha256: "c5da53c6d0d0022b517601d52ba785780f7e010ae5adf67f80ef453457307b11",
        source_ref: "https://github.com/microsoft/onnxruntime/blob/a83fc4d58cb48eb68890dd689f94f28288cf2278/onnxruntime/core/providers/tensorrt/tensorrt_execution_provider.cc",
        scope: "provider partition, parser, profile, engine-cache, and runtime implementation basis",
      },
    ],
  },
  interpretation_boundary: "Pinned API and implementation sources define version-specific collector semantics and configuration fields. The observation records whether the selected SDK exposed supportsModel or supportsModelV2. These sources are not an operator support table and do not establish that a particular TensorRT build, CUDA device, plugin set, or ORT provider accepts an artifact.",
});
