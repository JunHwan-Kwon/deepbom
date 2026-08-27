# Public Try Samples

These are public trained artifacts and deterministic structural probes used by the production `Run verified example` path. `web/lib/sample-models.js` pins the exact byte length and SHA-256 verified by the browser before analysis. Large additional TFLite examples are fetched on demand from generation-pinned official Google Storage URLs and are not copied into the deployment bundle.

| Local file | Model and source | Pinned source revision | License | SHA-256 |
| --- | --- | --- | --- | --- |
| `mobilenet_v2_1.0_224_quant.tflite` | Google TensorFlow MobileNet V2 1.0 224 quantized, `https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz` | Published source archive; member hash pinned | Apache-2.0 | `f08d447cde49b4e0446428aa921aff0a14ea589fa9c5817b31f83128e9a43c1d` |
| Remote on demand: `efficientnet_lite0_int8.tflite` | Google MediaPipe EfficientNet-Lite0 INT8, `https://developers.google.com/edge/mediapipe/solutions/vision/image_classifier#models` | GCS generation `1682480006900522`; exact object length and hash pinned | Apache-2.0, embedded TFLite metadata | `bc2ffe19c1118de0c0c2a9088992da5589722656e0fba81421385300a4a34b16` |
| Remote on demand: `efficientdet_lite0_int8.tflite` | Google MediaPipe EfficientDet-Lite0 INT8, `https://developers.google.com/edge/mediapipe/solutions/vision/object_detector#models` | GCS generation `1682636013497226`; exact object length and hash pinned | Not embedded; linked from the official model page and not redistributed by DEEPBOM | `0720bf247bd76e6594ea28fa9c6f7c5242be774818997dbbeffc4da460c723bb` |
| Remote on demand: `blaze_face_short_range_fp16.tflite` | Google MediaPipe BlazeFace short-range FP16, `https://developers.google.com/edge/mediapipe/solutions/vision/face_detector#models` | GCS generation `1682480001338381`; exact object length and hash pinned | Apache-2.0, official model card | `b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f` |
| `mnist-8.onnx` | ONNX Model Zoo MNIST-8, `https://huggingface.co/onnxmodelzoo/mnist-8` | `a19f9a8c2333de1df9b03f10f5739f468b699a1a` | Apache-2.0 | `2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf` |
| `tensorrt_supported_probe.onnx` | Deterministic Conv-to-Gemm TensorRT parser-capability probe | `scripts/generate-accelerator-llm-samples.py` v1.0.0 | Apache-2.0 | `15e9cd555dfdedcc4ef6a313e40efcaaf3d18dd2ee502018f335fec084118e47` |
| `gpu_partition_probe.onnx` | Deterministic GPU partition-boundary probe with a serialized `NonZero` node | `scripts/generate-accelerator-llm-samples.py` v1.0.0 | Apache-2.0 | `82a2feef00eb6ab03d82f2b30cd17f4d826e2d8307cb059eccd6a0f3120059b2` |
| `tiny_decoder_llm.onnx` | Deterministic decomposed decoder-attention graph with external state candidates | `scripts/generate-accelerator-llm-samples.py` v1.0.0 | Apache-2.0 | `1fa4d75584011ef9223c6ccf815bc2c1830424b637533bc560a5be2f518f22ee` |
| `tinymqa1m.Q4_0.gguf` | shibatch TinyMQA 1M trained on TinyStories, `https://huggingface.co/shibatch/tinymqa1m` | `403668777b2a2708d2529f8913cabfcaf3d6b385` | MIT | `cb95a6e10f28b76a1dd71c15560dec5a5eee8943f591ef45d11c129786b22cff` |
| `nanofable-1m-fp16.safetensors` | NanoFable-1M FP16 trained on TinyStories, `https://huggingface.co/adrahmana/NanoFable-1M-fp16` | `0647a8c41e06c97e057eb6aceafe3d859eba853b` | MIT | `a35fd03f52c12f4e78a246bec1927e9a169377fbb8905dc13165d285010e7a44` |
| `MNISTClassifier.mlmodel` | Apple MNIST drawing classifier, `https://developer.apple.com/machine-learning/models/` | Apple-hosted artifact; exact artifact hash pinned | MIT, as embedded in model metadata | `816d1a222d5272109166ffc819d7ce44aff923a673884e34d982aa74985ba587` |

`sample_cnn_float.onnx` and `mobilenet_v1_025_224_float.tflite` remain development regression fixtures and are excluded from deployment.

## Expected evidence contract

The browser's expandable Verified Example Library and its `Expected evidence JSON` download are generated from `web/lib/sample-models.js`. The document uses schema `deepbom.public_sample_expected_evidence.v1` and binds each example to its exact filename, byte length, SHA-256, source revision, license state, analysis depth, applicable and not-applicable metric families, and deterministic numerical baseline. A baseline is a regression contract for that exact artifact, not a model-quality or runtime-performance claim.

`npm run check:public-samples` recalculates the five bundled public artifacts and three deterministic probes without network access. The accelerator probes additionally require hash-bound TensorRT build-profile and parser-observation companions. `npm run check:public-samples:remote` downloads the three generation-pinned Google artifacts, verifies byte identity, and recalculates their graph, MAC, quantization, liveness, arena, and source-rule XNNPACK invariants. Every example preserves runtime assignment as `not_observed`.

## Accelerator evidence boundary

The three generated ONNX files contain no trained model weights and make no model-quality claim. Their intermediate shapes are serialized in `value_info`, so graph cardinality, initializer bytes, static MACs, and liveness are independently reproducible without executing TensorRT.

TensorRT companion observations are captured by `native/tensorrt_collector` against the exact artifact bytes, SDK/runtime version, CUDA version, GPU compute capability, source pins, collector commit, and collector binary hash. A parser-supported subgraph is reported as `CONDITIONALLY_ELIGIBLE`; a node absent from the supported collection remains `UNRESOLVED`. Neither state establishes engine-build success, tactic selection, runtime placement, kernel execution, transfer traffic, latency, or numerical correctness.

The bundled observations use TensorRT 8.6.1, CUDA runtime 11.8, and an NVIDIA GeForce RTX 4060 Laptop GPU (compute capability 8.9). They were collected from clean commit `f3b8fb5adab237e490dbb930bd708049eb2764d8`. The shared profile and each observation are independently byte-counted and SHA-256 bound in the public sample manifest; changing any companion fails the example before analysis.
