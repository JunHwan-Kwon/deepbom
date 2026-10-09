# Common Evidence IR flow consolidation

Date: 2026-10-09. Scope: the current development tree, including the native
optimization and training additions. This is an implementation/validation record,
not a deployment receipt or a claim that every possible input has been tested.

## Ownership and flow

```text
Artifact bytes / declared framework state
  → format-specific fact producer or explicit local framework adapter
  → common identity, Model/Artifact/Weight/Activation/Training contracts
  → shared calculation and semantic validation
  → channel-specific query, display and export
```

Adapters still own decoding, native layout and framework operations. Those are
not forced into a presentation IR or interpreted as runtime measurements. The
shared contract owns source binding, identities, reference closure, counts,
comparison semantics and numerical methods. A report, snapshot or view is not
an additional analytical IR merely because it is serialized.

## Implemented changes

| Area | Before | Current implementation |
| --- | --- | --- |
| Native tensor comparison | Generator and report importer had different semantic checks | `compareNativeTensorIdentities` is the shared owner. The importer checks the complete ordered inventory, duplicate/foreign/missing references, shape/dtype alignment, matching basis and payload equality against both saved storage inventories. |
| Explorer weight distributions | Independent Rust histogram, percentile and filter-rank semantics | Model IR selects the operation's bound storage. The worker produces common Weight IR and advanced analysis. The view reuses shared distributions, percentile bounds, channel, sparsity, similarity, quantization and singular-spectrum projections. |
| Legacy ABI | Returned a second numerical contract | Cached callers receive an explicit retirement error. The obsolete computation and browser RPC are removed. No silent fallback or second numerical engine remains in this path. |
| Web and CLI entry points | Could fall back to raw analysis when a context was absent | Presentation/export routing requires the common Artifact IR consumer view. Native producer objects remain internal facts; they are not used as a fallback presentation contract. |
| Hash/report ordering | TFLite could generate Markdown while only its hash existed | Hashing produces identity. Markdown generation occurs after context construction in the render flow. |
| Regression guard | Detected only direct `analysis.ops` / `analysis.tensors` | Also detects aliases and destructuring, bans retired histogram calls throughout browser sources, and checks the mandatory orchestration routes. |

The Weight reader caches one artifact's basic evidence and at most eight
operation projections. Source bytes, Model IR and Weight IR remain validated.
Foreign hashes are rejected; unsupported decoding or exhausted budgets stay
explicit. Selection changes do not allow late results to replace another
operation's evidence. Operations without bound storage do not trigger decoding.

Stored scalar distributions are kept separate from advanced features computed
from an explicitly dequantized representation. An all-zero UINT8 payload now
has exact zero percentile bounds. A histogram interval is never presented as an
exact median. Stored element counts are not labeled as trainable parameters,
and filter-energy concentration is not mislabeled as singular-spectrum rank.

## Compatibility

- Existing Model/Weight/Activation artifact contracts and the native experimental
  contracts retain their schema identities. No `Trained IR` was introduced.
- Valid generated optimization reports keep their format. Internally contradictory
  tensor comparison tables are now rejected, including when their hashes have
  been recomputed. This verifies internal consistency, not producer authenticity.
- Distribution comparison cannot be independently recomputed without the original
  numerical evidence. This limitation remains explicit.
- Provenance/OMOP/BOM adapters still cover their declared artifact-bound sources.
  Native Training evidence is not silently coerced into an artifact-file identity.
- Hosted ChatGPT/Claude tool inventories are unchanged. They inherit changes only
  through common implementations they actually consume.
- Compatibility catalog **0.14.1** adds traceability for the Explorer weight flow
  and advanced analysis as a supporting context, not a new IR. Earlier snapshots
  are preserved. The generated 0.14.0 draft grouped several IR field names into
  one selector; the field-closure check rejected that definition and 0.14.1 records
  individual selectors instead.

## Validation

Regression checks cover the following independent boundaries:

- Actual ONNX/TFLite operation selection, exact zero quantiles, digest mismatch,
  explicit unavailable evidence, cache isolation and retired ABI rejection.
- Actual Web audit → context → worker → Explorer charts, keyboard histogram bins,
  lazy advanced views, light/dark/mobile layout and rapid selection changes.
- Report mutations through common import, CLI, Web, local MCP and packaged Python
  PDF import; the three original counterexamples must all be rejected.
- Artifact/Model/Numerical IR, native evidence, shared arithmetic, six-format
  advanced Weight decoding, pruning, public SDK, MCP contracts and CLI exports.
- Real CPU PyTorch/TensorFlow model transformations and training collectors,
  GradientTape/fit, MLflow local store, TensorBoard round trip and W&B offline.
- Report generation, Before/After diff and PDF generation using the rebuilt
  packaged Python engine, rather than an independently implemented Python rule.

Execution logs, browser images and package outputs are kept under the ignored
`.local-validation/common-flow-*` paths. The final receipt separates initial
configuration/fixture failures from their corrected reruns. No GitHub Actions,
remote publication or deployment is performed by this change.

## Limits of enforcement

The static ownership checker is a conservative regression guard, not a complete
JavaScript type or information-flow proof. Runtime routing and semantic checks
are therefore required as well. Native fact producers and isolated compatibility
facades remain explicitly classified; their existence does not authorize a UI
or export route to bypass context construction. Future supported paths must add
source bindings and common owner checks before claiming cross-channel support.
