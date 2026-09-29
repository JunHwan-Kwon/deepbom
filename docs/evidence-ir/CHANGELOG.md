# Evidence IR specification changes

## Unreleased — family catalog 1.0.0 (draft)

- Name the overall representation family **DEEPBOM Evidence IR**.
- Establish **Provenance IR** as the sole current connection contract: schema, digest field, generic input, summary, section, API and export names.
- Remove the retired implementation module, schema, compatibility facade, selector aliases and documentation stub.
- Preserve the other four IR contracts. Recompute new provenance documents under their own hash contract; never rewrite archived reports.
- Version remote result envelopes to v2 and update both widgets and server validators together.
- Generate the family catalog from shared identities and the Provenance IR schema from shared contracts.
- Add negative tests for retired contracts and explicit migration/release instructions.
- Review pinned TEA sources and document a proposed exchange adapter outside the common evidence core.

No package release, deployment, TEA integration, external endorsement or independent adoption is claimed by this entry.
