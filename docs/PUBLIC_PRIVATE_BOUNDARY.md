# DEEPBOM Public/Private Source Boundary

DEEPBOM uses a component boundary rather than treating every repository file
as having the same disclosure or license status.

## Public distribution

The following implementation classes are eligible for the public source
export and Apache-2.0 channel packages:

- the CLI and npm, PyPI, and Cargo channel adapters;
- the TFLite Rust static analyzer and its generated public WASM binding;
- JavaScript artifact parsers, deterministic calculations, report projections,
  viewer code, schemas, public fixtures, and release checks;
- public source-pin metadata and generated evidence snapshots already shipped
  to browsers; and
- reproducibility, package-integrity, and release-gate automation.

Public package publication does not imply that every file in the private
monorepo is licensed or disclosed. `channels/LICENSE` is the authoritative
license copied into public channel artifacts and the clean public source
export.

## Private implementation

The following remain outside the public source export:

- `protected/` source and `web/protected/` generated protected modules;
- rulepack generation programs for ORT, TFLite delegates, and XNNPACK;
- hosted authentication/database code, deployment workflows, and concrete
  infrastructure bindings;
- local private roadmap files and `docs/private/`; and
- unreleased research modules explicitly placed under a private boundary.

Generated public metadata may identify an upstream source commit, source file,
or digest. Those facts are evidence provenance, not the private generator
implementation.

## Enforcement

`config/public-source-files.v1.txt` is an exact allowlist. A new tracked file is
not public merely because it is added to the monorepo. The allowlist must be
reviewed and refreshed explicitly. `scripts/build-public-source-export.mjs`
rejects private prefixes, private generator paths, local files, and paths that
escape the repository before copying any member.

The channel build independently checks the complete esbuild input graph.
`scripts/check-public-package-boundary.mjs` then verifies the exact npm package
member set, license digest, repository identity, absence of source maps,
credentials, protected paths and symbols, and debug-bearing WASM custom
sections.

The existing private Git history must never be made public. A public repository
must be created from the clean export with new history. Removing private files
only from the latest commit is insufficient because earlier commits retain
their contents.

## Evidence boundary

Public source and package inspection establishes what was included in a
release. It does not establish that an analyzer is correct, that an upstream
runtime selected a predicted backend, or that a model is clinically valid.
Those claims remain governed by their own observed, derived, predicted, or
not-assessable evidence classes.

## Corpus boundary

Public corpus material is selected for independent reproduction: pinned source
identities, content hashes, licenses or explicit license gaps, compact fixtures,
deterministic sweep programs, and reviewed aggregate or per-artifact evidence.
Committing a manifest does not grant redistribution rights for the referenced
model bytes.

The private corpus boundary retains large downloaded bytes, restricted-license
artifacts, user-provided models, negative robustness inputs, raw runtime traces,
and unreleased research populations. Clinical or personally identifying data is
not committed to either repository. Private GitHub stores code, manifests, and
non-sensitive receipts; large or restricted bytes remain in access-controlled
object storage or verified local caches.

Public and private measurements must share the same schema, analyzer identity,
and deterministic verifier. A private result may be published only as a reviewed
aggregate or hash-bound record that does not disclose protected model content.
