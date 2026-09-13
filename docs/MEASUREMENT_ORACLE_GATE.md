# Measurement oracle gate

DeepBOM keeps parser measurements separate from summaries, policies, and
integration adapters. `corpus/external-review/measurement-oracles.v1.json`
pins one public GGUF, ONNX QDQ, and SafeTensors artifact with exact digests and
format-native expected values.

The primary checks are deliberately format-specific: GGUF tensor encodings and
block sizes, ONNX nodes and TensorProto parameters, and the SafeTensors header
directory. DeepBOM is the implementation under test. The tensor-assignment hash
is independently recomputed from the public table using the documented JCS
projection, so reproduction does not depend on a private schema or function.

The gate covers serialized facts only. It does not promote QDQ representation
to observed runtime fusion, infer lineage, or claim model quality, safety, or
regulatory conformity.
