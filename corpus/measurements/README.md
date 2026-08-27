# Measurement Sources

This directory preserves the exact machine-readable measurement inputs used by
`scripts/build-standardization-evidence.mjs`. Keeping these sources beside the
derived standardization ledger allows a clean checkout to reproduce source
digests, population totals, and the final RFC 8785/JCS-bound ledger without
depending on ignored local report directories.

The two populations are intentionally separate and must not be summed:

- `quantization-interface-public-50-repeat2/` contains the repeated external
  interface-contract sweep over its predeclared TFLite population.
- `public-model-corpus-v1-final/` contains the pinned MediaPipe-generation
  public TFLite sweep used by the corresponding population record.

Each gzip member contains one complete JSON document with its own schema,
corpus identity, selection context, and artifact-level results. Compression
changes storage only; the derived ledger binds each stored gzip source by
SHA-256 and states the limits on population-level inference.
