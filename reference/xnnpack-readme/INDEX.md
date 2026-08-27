# XNNPACK delegate README — per-TFLite-tag snapshots

Source (per tag):
`https://raw.githubusercontent.com/tensorflow/tensorflow/<tag>/tensorflow/lite/delegates/xnnpack/README.md`

This is the authoritative, version-pinned source for DEEPBOM's XNNPACK
delegation rulepack: it documents exactly which ops the TFLite XNNPACK
delegate supports and under what dtype / dimensionality / quantization
constraints, at each TensorFlow release.

Fetched 2026-07-24.

## Snapshots

| Tag | Bytes | sha256[:12] | Content changed vs previous |
| --- | ----: | ----------- | --------------------------- |
| v2.10.0 | 29425 | e9e0caab739e | (baseline) |
| v2.11.0 | 29968 | 061e16414045 | yes |
| v2.12.0 | 28656 | 7c638200d2ec | yes |
| v2.13.0 | 28818 | 9a70a18d2248 | yes |
| v2.14.0 | 28818 | 9a70a18d2248 | no (identical to 2.13.0) |
| v2.15.0 | 30258 | 84fe54dbf893 | yes |
| v2.16.1 | 30258 | 84fe54dbf893 | no (identical to 2.15.0) |
| v2.17.0 | 30258 | 84fe54dbf893 | no (identical to 2.15.0) |
| v2.18.0 | 30258 | 84fe54dbf893 | no (identical to 2.15.0) |
| v2.19.0 | 29951 | 75f1e4c8d10f | yes |
| v2.20.0 | 29116 | 64cb3c7971fb | yes |
| v2.21.0 | 32825 | 85524b3e6acf | yes (latest) |

## What actually changes across versions

The **documented op set is stable from v2.13 onward**: 41 FP32 operators +
24 quantized operators, unchanged through v2.21.0. FP16 support is expressed
as full FP32 feature parity rather than repeated per-operator subsections.

- **v2.10 → v2.13**: op set expanded — added `SLICE`, `SPACE_TO_DEPTH`,
  `STRIDED_SLICE`, `TANH` (39 → 43 unique op names).
- **v2.13 → v2.21**: no op added or removed, and the normalized per-op
  constraint subsections in these snapshots are unchanged. The file churn is:
  - delegate flags (e.g. `TFLITE_XNNPACK_DELEGATE_FLAG_TRANSIENT_INDIRECTION_BUFFER`),
  - the **Weights Cache** documentation (substantially expanded in v2.21),
  - prose reflow / heading capitalization (much of the byte diff is cosmetic).

## Rulepack implication

DEEPBOM pins the rulepack to main commit `87bbf65b...` and records that its
README is byte-identical to the one at tag `v2.21.0` (tag commit
`a481b102...`). The generated semantic manifest maps all 133 artifact-visible
per-op constraints and records quantized build flags as runtime-only
requirements. `version-diff.json` is the machine-generated version comparison.
