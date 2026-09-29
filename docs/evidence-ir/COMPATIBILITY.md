# Compatibility baseline and maintenance

Start compatibility tracking at the canonical Provenance IR transition. This is a reasonable boundary because the retired contract is intentionally rejected, while the other Evidence IR member identities remain distinct. Catalog 0.1.0 preserves the first draft baseline. Catalog 0.2.0 pins the implementation for the 2.0.0 release; both remain **draft implementation crosswalks**, without external standards-body approval.

## One definition, multiple views

`config/evidence-compatibility.mjs` imports the existing Evidence IR identities, public format maturity and audit output contracts. It describes mappings; it does not reimplement parsers, calculations or support decisions. `scripts/generate-evidence-compatibility.mjs` produces the GitHub table, immutable JSON snapshots and version index. The website loads the same snapshot bytes and checks the indexed SHA-256 before displaying them. The Pages build checks freshness and copies those exact bytes.

The explorer sits above separate input-schema and output-schema inventories. Search matches endpoint names, contracts, field groups, mapping IDs, limits, implementations and regression-check paths. A layer, direction or status filter narrows the catalog. Links retain the selected catalog version and mapping. The workspace links here from artifact selection, export and help.

## Boundaries reviewed

- Artifact IR normalizes observed native facts; Model IR derives a program/storage representation from its validated source. Native facts and projection losses remain traceable.
- GGUF and SafeTensors do not supply an executable operator DAG. HDF5 and PyTorch checkpoints expose a safe envelope. Preview declarative graphs do not establish framework execution.
- Weight IR needs a supported payload decoder and original source bytes. Activation IR needs a separately obtained capture. Provenance IR keeps declarations distinct from observed supplied-file consistency.
- Some exporters consume native analysis and finding ledgers alongside IR. Their paths are shown explicitly rather than implying an IR-only pipeline.
- CycloneDX 1.7 has a bounded provenance projection. The current SPDX export is a 2.3 file inventory, **not** SPDX 3 AI semantics. TEA exchange remains proposed. OMOP clinical meaning and FAIR assessment are not replaced by the catalog.
- A mapping's presence is not universal artifact support or a test receipt. Actual run coverage belongs in each resulting IR. Field groups are intentionally navigational, not exhaustive leaf-by-leaf transformation rules. Auxiliary command schemas are outside the first catalog.

## Four independent version axes

| Axis | Change rule |
| --- | --- |
| Member schema identity | Incompatible document shape or semantics requires a new schema identity and migration policy. |
| Calculation method | Numerical or derivation changes are tracked by the owning member's method version. |
| Compatibility catalog | New snapshot whenever mappings, source pins or published descriptions change. |
| Engine / channel release | Actual distributed implementation and build identity, independent of catalog publication. |

Use [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for the catalog. While 0.x remains a draft, use a new minor version for changed mapping semantics or capabilities and a patch for compatible clarifications. After 1.0, incompatible meaning or removals require a major version. Do not change a released snapshot under the same version. Existing snapshots and index digests are append-only; the generator rejects same-version rewrites, deletions and modified archive entries.

Source-file and schema SHA-256 pins identify the reviewed implementation, including when a current schema URL later changes. They are not publisher authentication. Historical source/schema bytes are recoverable from Git history; a current schema URL is not promised to serve historical bytes. Publish the catalog and implementation in the same reviewed public source release. Record release commit/tag externally rather than embedding a self-referential commit hash in generated files.

## Updating

1. Change the actual parser, common rule, IR validator or exporter in its owner; add relevant semantic regression coverage.
2. Update the mapping definition and increment `CATALOG_VERSION` when content or pinned source files change.
3. Run `node scripts/generate-evidence-compatibility.mjs`, then `node scripts/check-evidence-compatibility.mjs` and relevant referenced checks.
4. Review the generated diff, including losses, partial support and schema/source pins. Add new snapshots to the public export allowlist.
5. Build Pages; run the browser catalog check with `--dist` and public source export verification. Ship the same catalog alongside the actual compatible engine release.

The first baseline does not promise conversion among every pair of formats. It makes the implemented path and its limits inspectable. TEA may later deliver evidence documents without changing these common IR rules; see the [pinned TEA review](TEA_REVIEW.md) and [official draft](https://ecma-tc54.github.io/ECMA-xxx-TEA/).
