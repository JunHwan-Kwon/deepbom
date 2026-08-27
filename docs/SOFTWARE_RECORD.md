# DEEPBOM Software Record

DEEPBOM is research software developed by Jun-Hwan Kwon, Ph.D.
The canonical hosted service is available at <https://deepbom.org/>.

## Citation scope

This record identifies version 1.94.0 of the software for citation,
attribution, and release provenance. It does not distribute the implementation
or define the license of components outside the deposit. Source code, JavaScript application modules,
WebAssembly binaries, native collectors, model artifacts, and production
configuration are not deposited in the software record.

## Analysis scope

The identified release analyzes deployment artifacts in TFLite, ONNX, GGUF,
SafeTensors, and Core ML formats. Depending on the serialized contract and
available source-pinned rules, it reports graph, tensor, quantization, memory,
execution-provider, XNNPACK delegation, SIMD/NEON, target-fit, integrity, and
deployment evidence.

Evidence is classified as OBSERVED, DERIVED, PREDICTED, ESTIMATED, or MEASURED.
A DOI identifies the software release; it does not by itself convert static
predictions into runtime measurements or establish universal format, producer,
operator, or hardware coverage.

## Public companion evidence

A separate validation-dataset record may publish hash-bound parser outputs,
worked reports, UI captures, and package manifests that contain no source code,
executable implementation, protected WebAssembly, credentials, or model
weights. The software and validation records should be linked with reciprocal
supplement relations.
