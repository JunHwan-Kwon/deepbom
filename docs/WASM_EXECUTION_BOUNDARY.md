# WebAssembly and Worker Execution Boundary

DEEPBOM assigns work by evidence ownership and browser responsiveness, not by a rule that every parser must be WebAssembly.

## Production boundary

- Rust/WebAssembly owns deterministic TFLite FlatBuffer analysis, target projections, redesign projections, tensor influence, tomography, landscape, and Haar computations.
- Every heavy public TFLite WebAssembly export is invoked through `workers/static-audit-worker.js`. The UI thread may initialize the signed public module and call only `runtime_guard` and `target_profiles`.
- GGUF, SafeTensors, and Core ML range readers run in that dedicated Worker. Their source-pinned JavaScript decoders remain JavaScript until a measured Rust/WebAssembly implementation proves equivalent and materially better.
- ONNX and ExecuTorch parsers remain isolated JavaScript Worker workloads. The protected DEEPBOM WebAssembly module retains its separate authorization-scoped Worker.

## Independent validation

Independent JavaScript validators are intentionally retained for quantization lattice, kernel witness, rounding equivalence, channel vitality, and related arithmetic checks. Moving a derivation and its validator into the same WebAssembly implementation would reduce implementation diversity and weaken the cross-check.

## Streaming candidates

Only payload kernels with bounded, contiguous inputs are candidates for a later streaming WebAssembly port:

1. incremental SHA-256;
2. source-pinned GGUF block decoding; and
3. SafeTensors scalar or packed-value decoding.

Run the screening benchmark with:

```text
npm run benchmark:streaming-wasm-candidates
```

The benchmark verifies digest and byte conservation while recording current JavaScript throughput. It is not a WebAssembly speed claim. A port is accepted only after a browser benchmark shows repeatable throughput or memory improvement and the new implementation passes every existing source-pinned decoder fixture. The independent JavaScript verifier remains available after any port.

## Enforcement

`npm run check:no-main-thread-heavy-wasm` fails when a heavy TFLite WebAssembly export is imported by the UI thread, when the Worker no longer owns a required export, or when a range analyzer is reintroduced into `web/app.js`.
