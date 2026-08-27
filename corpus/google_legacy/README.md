# Google Legacy Corpus

This corpus is a reproducible, hash-pinned study set for separating architecture,
converter-generation, and quantization-recipe effects.

The repository stores metadata and code only. Archives and models are placed in
the user cache, not in the application bundle:

```text
%LOCALAPPDATA%\DeepBOM\google-legacy-corpus-v1
```

On Linux and macOS the cache falls back to `$XDG_CACHE_HOME` or `~/.cache`.

## Reproduce

Download and extract only the pinned TFLite members:

```bash
npm run corpus:google-legacy:download
```

Run every model twice in isolated Node processes and emit the research ledgers:

```bash
npm run corpus:google-legacy:sweep
```

Re-run without network access:

```bash
npm run corpus:google-legacy:sweep -- --offline
```

Outputs are written under `reports/google-legacy-hosted-models-2026-07-29/`:

- `google-legacy-sweep.json`: complete per-model and per-lab evidence
- `google-legacy-models.csv`: model-level paper table
- `google-legacy-labs.csv`: one row per model and research lab
- `google-legacy-reasons.csv`: grouped non-assessment reason counts

The extractor invokes `tar -xOf` for one manifest-pinned member at a time and
verifies both the archive and extracted member SHA-256. It never expands an
archive path into the workspace.

## Corrected Cohort

The downloaded evidence does not support the original proposed table verbatim:

- MobileNetV1/V2, InceptionV3, and SSD MobileNetV1 are quantized artifacts.
- The four official MnasNet members are FLOAT32. They are retained as negative
  controls and contribute no denominator to integer-contract defect rates.
- The official hosted NASNet-Mobile and SqueezeNet entries are float models.
- A prebuilt official SSD MobileNetV2 quantized TFLite member was not verified
  from this source and is therefore pending, not silently synthesized.

Likewise, activation labels are measured from serialized operator options during
the sweep. An architecture paper's usual activation is not treated as artifact
evidence.

## Study Denominators

There is deliberately no single global "channels assessed" number. Different
proof labs assess different channel domains. The JSON and lab CSV retain each
lab's own `assessed_channel_count`; the model CSV exposes the quantized-weight
output-channel denominator separately.

Report:

1. artifact class,
2. class-supported and artifact-applicable lab counts,
3. assessed, partial, not-assessed, and not-applicable counts,
4. per-lab assessed channel count,
5. grouped reason-code distribution.

`not_applicable` and `not_assessed` are never counted as defect absence.

## Converter Cohorts

`converter-cohorts.v1.json` records the four intended generations. Historical
hosted artifacts are immutable observations. MLIR, LiteRT Torch, and AI Edge
Quantizer rows require regenerated artifacts with complete recipe provenance.
No generated row is admitted merely because a package can be installed.

`paired-modern.v1.json` adds 16 hash-pinned modern comparison artifacts without
mislabeling them as MLIR outputs. It includes FP32/dynamic/static pairs for
MobileNetV2 and MobileNetV3, FP32/dynamic controls for MnasNet, EfficientNet-B0,
and SqueezeNet, plus EfficientDet-Lite0:

```bash
npm run corpus:google-modern:download
npm run corpus:google-modern:sweep
```

These repositories pin the model-card blob as well as each model file. Where the
upstream model license is absent, bytes remain cache-only and the row is marked
as unsuitable for redistribution.

`measurement-baseline.v1.json.gz` is the compact, committed historical result
ledger for all 24 artifacts. It is tied to both a clean source commit and the
executed analyzer WASM content hash; validation reads that exact WASM blob from
the recorded commit instead of conflating the ledger with the current build.
It records artifact identity, deterministic analysis hash, lab coverage,
lab-specific channel denominators, and distinct zero-slice/range/constant-output
signals. Regenerate it only from a clean worktree after both corpora complete
two zero-failure isolated runs:

```bash
npm run corpus:google:baseline
```
