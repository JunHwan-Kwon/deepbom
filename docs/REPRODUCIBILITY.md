# DEEPBOM Evidence and Reproducibility Statement

This document applies to the versioned DEEPBOM software citation record. The
record identifies the hosted software release; its deposited files are a
citation and provenance dossier, not a source or executable distribution.

## Release identity

Every Zenodo citation package contains:

- `ZENODO_RELEASE_MANIFEST.json`, binding the dossier to the semantic version,
  Git commit, repository state, package kind, and declared exclusions; and
- `SHA256SUMS`, binding every other archive member by path, byte length, and
  SHA-256.

A final package can only be built from a clean Git worktree. A package built
with `--allow-dirty` is named `DRAFT` and is not suitable for publication. The
validation dataset additionally requires a validation matrix whose worktree is
clean and whose Git commit matches the software release.

## Signature authority

Browser-generated contract and evidence packages use an origin-local P-256 key
stored in IndexedDB when available. Their `LOCAL_BROWSER_KEY` signature detects
member or digest changes, but it does not establish DEEPBOM authorship.

DEEPBOM-maintained release evidence is signed offline with a separately held
`OFFICIAL_RELEASE_KEY`. Verifiers must require all of the following, not merely
a valid ES256 signature:

- the fixed registry URL
  `https://deepbom.org/.well-known/deepbom-signing-keys.json`;
- a matching key ID, public JWK, and RFC 7638 SHA-256 thumbprint;
- an authorized signature scope and an issuance time within the key interval;
- a key status that is active, or retired for a historical in-range signature;
  and
- no revocation state.

The registry binds the release authority to the software DOI. The offline
private key is not shipped in the web application, repository, release dossier,
or validation dataset. Key rotation preserves historical verification through
retired records; revoked keys fail closed.

## Evidence interpretation

DEEPBOM distinguishes these classes:

- `OBSERVED`: read directly from the serialized artifact or imported runtime
  evidence.
- `DERIVED`: calculated deterministically from observed values and a stated
  method.
- `PREDICTED`: source-pinned static execution or delegation prediction.
- `ESTIMATED`: model-based quantity with disclosed assumptions.
- `MEASURED`: result of an identified runtime execution and environment.

Static delegation, roofline, packing, and target-fit results are not runtime
measurements. Imported runtime evidence is artifact-bound and retains its own
runtime, build, target, and capture provenance.

Representative-dataset captures additionally bind the exact artifact,
dataset-manifest identity, optional preprocessing-contract digest, runtime
identity, and audited external I/O contract. DEEPBOM independently reconstructs
storage endpoint counts, supplied-reference drift, repeated-run differences,
and the capture/ledger SHA-256 values. The contract and claim boundary are
specified in `REPRESENTATIVE_DATASET_CAPTURE.md`.

## Reproducibility boundary

The software DOI provides persistent release identity and attribution. Because
implementation source and executable artifacts are not distributed through the
record, the DOI alone does not provide an independently rebuildable analyzer.
That limitation is intentional and must remain explicit in papers and record
descriptions.

The public companion validation dataset supports narrower reproducibility of
published examples. It may include:

- `validation-matrix.json` and `artifact-catalog.json`;
- parser analysis/result JSON for advertised format paths;
- captured UI surfaces and their indexes;
- non-code public engineering evidence bundles; and
- generated package checksums and release provenance.

It excludes source model duplication, implementation source, JavaScript,
WebAssembly, native binaries, credentials, local logs, and protected report
logic. Synthetic Core ML package and sharded SafeTensors cases are declared as
deterministic envelopes, not independent upstream trained artifacts.

The validation matrix demonstrates only the identified artifacts and workflows.
It does not establish universal coverage of every producer, opset, custom
domain, external-data layout, quantization scheme, or hardware runtime.
