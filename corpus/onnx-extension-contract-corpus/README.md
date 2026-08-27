# ONNX Extension Contract Corpus

This purposeful conformance corpus covers two residual analyzer contracts without
claiming ecosystem prevalence:

- per-axis `QuantizeLinear` and `DequantizeLinear`, including the case where
  scale and zero-point values are runtime graph inputs; and
- non-ORT external custom-domain nodes from the pinned
  `microsoft/onnxruntime-extensions` upstream test corpus.

The runtime-parameter fixtures are source-derived from ONNX 1.22.0. Their
serialized dtype, shape, default axis, and cardinality are statically
assessable; their numerical values are not serialized and therefore remain
`RUNTIME_REQUIRED`. The complete static fixture independently checks decoded
per-axis values and Q/DQ payload conservation.

`onnxruntime-extensions-custom-op-test.onnx` is an exact upstream binary. Its
`ai.onnx.contrib` operators must remain external-registry requirements rather
than inheriting standard ONNX or `com.microsoft` semantics.

Run:

```shell
npm run build:onnx-extension-contract-corpus
npm run check:onnx-extension-contract-corpus
```

The manifest pins every artifact and upstream license by SHA-256.
