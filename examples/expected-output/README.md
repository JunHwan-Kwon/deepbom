# Expected Output Contract

This directory contains a small, hash-pinned public artifact contract for the
first CLI run. It is a regression fixture for exact artifact bytes, not a model
quality or runtime-performance claim.

Run:

```console
npx deepbom audit web/samples/gpu_partition_probe.onnx
npm run check:expected-output
```

Interpretation:

- `0` means an applicable, assessed count is exactly zero.
- `NOT_ASSESSABLE` means required evidence is absent or incomplete.
- `NOT_APPLICABLE` means the metric does not apply to that artifact or format.
- A finding is a review signal under its stated evidence class and boundary. It
  does not automatically establish model failure, task-accuracy loss, clinical
  invalidity, or release failure.

The artifact is the deterministic GPU partition probe documented in
`web/samples/README.md`. Its intermediate shapes are serialized, so graph and
MAC conservation are reproducible without executing an inference runtime.
