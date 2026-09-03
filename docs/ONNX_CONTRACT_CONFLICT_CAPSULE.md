# ONNX Serialized Contract Conflict Capsules

DEEPBOM emits `deepbom.onnx_contract_conflict_capsule.v1` when an ONNX
artifact contains an unconditional or condition-bound contradiction in its
serialized tensor contract. The capsule is an artifact-bound diagnostic
record. It is not a repaired graph and it does not substitute an inferred
shape for a contradictory declaration.

## Evidence Contract

Each capsule binds:

- the primary artifact SHA-256 and source locator when available;
- the exact ONNX shape and dynamic-cost analyzer contract versions;
- unconditional declaration and semantic root conflicts;
- finite condition-bound invalid variants;
- affected values and downstream nodes;
- MAC-bearing rows withheld because an upstream contract is invalid;
- canonical Artifact IR operator/value subjects when they can be resolved; and
- a capsule SHA-256 over canonical JSON excluding `/capsule_sha256`.

The semantic validator checks count and histogram conservation, root lineage,
digest reproducibility, and optional Artifact IR referential integrity. A
consumer must treat `INVALID_CONTRACT` separately from `NOT_ASSESSED` and from
a valid but symbolically unresolved shape.

## Measured Corpus

`corpus/onnx-contract-conflict-corpus.v1.json.gz` contains every invalid-contract
artifact in the pinned 48-artifact ONNX residual population:

| Measure | Count |
| --- | ---: |
| Hash-identified artifacts | 9 |
| Unconditional conflict roots | 6 |
| Condition-bound invalid variants | 543 |
| Unconditionally invalid node outputs | 1,995 |
| Conditionally invalid node outputs | 400 |
| Downstream blocked nodes | 1,901 |
| MAC-bearing rows withheld | 213 |

The withheld MAC rows are 107 `Conv`, 88 `MatMul`, 17 `ConvTranspose`, and one
`LSTM`. Every affected or blocked record resolves to a root conflict; the
published corpus permits zero unresolved root references.

The source artifacts are revision- and SHA-256-bound. Corpus generation
re-analyzes the bytes, compares all aggregate counts with the independent
residual sweep, and records isolated runs of the official Python ONNX checker
and strict shape inference pinned by `requirements-onnx-corpus.txt`.

## Reference Boundary

The official ONNX runs are comparative evidence only. Some selected models
reference external tensor files that are intentionally absent from the
single-file cache, and strict native shape inference can terminate before
returning a diagnostic on this Windows validation host. `fail`, `crash`, and
`timeout` are preserved as distinct outcomes. They neither confirm nor erase a
DEEPBOM conflict, and they are never converted into a successful validation.

The nine-artifact corpus demonstrates reproducible failure modes within a
bounded population. It does not estimate ONNX ecosystem prevalence, runtime
compatibility, task correctness, or deployment safety.

## Reproduction

Install the pinned Python reference dependency, download the pinned model bytes
once, then rebuild and verify the corpus:

```powershell
python -m pip install -r requirements-onnx-corpus.txt
npm run build:onnx-contract-conflict-corpus -- --download
npm run check:onnx-contract-conflict-corpus
```

Subsequent rebuilds use the ignored `.local-validation` byte cache and do not
require another download.
