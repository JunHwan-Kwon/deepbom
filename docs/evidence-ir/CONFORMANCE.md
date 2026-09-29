# Conformance and compatibility

Status: draft. This document defines DEEPBOM-specific expectations; it is not a certification program.

## Required distinctions

1. **Structural validation:** load the family entry point and referenced member schemas into a local JSON Schema Draft 2020-12 registry. Resolve `$ref` by the declared absolute `$id`; no network fetch is required. Select the member using its serialized `schema` value.
2. **Semantic validation:** apply the member's validator to the exact source documents. Check digests, reference closure, source binding, exact counts, numeric mirrors, coverage conservation and supported method versions. A rehashed fabricated document is not thereby valid.
3. **Evidence interpretation:** retain declared versus observed/derived/predicted states, unsupported mappings and explicit unknown values. Provenance consistency is not relationship attestation. Imported execution evidence is not automatically authenticated.

Consumers must state which of these checks they perform. Producers must state which formats, methods, payload encodings and optional layers they can assess. Neither may describe one passing example as complete format support. Use the member-specific limits; do not broaden applicability merely because a field fits the common schema.

## Identity and migration

Schema identity, calculation method, software release and presentation name are separate. Preserve existing document bytes and digest rules when only a display name or selector alias changes. A new meaning, incompatible shape or canonicalization rule requires an explicit migration decision and an appropriate new contract/method identity. Never rewrite old documents in place to appear current.

The sole current selector is `provenance_ir`. Its document identity is `deepbom.provenance_ir.v1`, with digest field `provenance_ir_sha256`. Retired schemas and fields are rejected, not coerced. Regenerate evidence from the source model, current input contract and supplied files; do not rename fields on an old signed or hashed document. See [migration and coordinated delivery](MIGRATION.md).

## Reproducible checks

```sh
npm ci
node scripts/generate-provenance-schema.mjs --check
node scripts/generate-evidence-ir-catalog.mjs --check
node scripts/check-evidence-ir-family.mjs
node scripts/check-evidence-compatibility.mjs
node scripts/check-evidence-compatibility-browser.mjs
node scripts/check-provenance.mjs
node scripts/check-numerical-ir.mjs
node scripts/check-common-ir-semantics.mjs
node scripts/check-ir-hardening.mjs
```

The family check validates produced Artifact, Model, Weight, Activation and Provenance documents, including explicitly synthetic activation fixtures. It rejects unknown/retired identities and tests deterministic current hashes. Member regressions exercise rehashed tampering and incomplete evidence. CLI/MCP tests reject the retired selector; remote tests reject retired envelopes, summary schemas and fields. Examples and tests are reproducible fixtures, not independent adoption claims.

When citing a result, record the command, exact source revision, input digest, method identity and any unavailable fixture or environment. Immutable release archives are preferred for published results; a moving `main` branch is not a reproducibility identifier.

Reference: [JSON Schema identifiers, registries and modular references](https://json-schema.org/understanding-json-schema/structuring).
