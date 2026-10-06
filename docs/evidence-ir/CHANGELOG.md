# Evidence IR specification changes

## DEEPBOM 2.0.0 — family catalog 1.0.0 (draft specification)

Released September 29, 2026. The package and hosted implementation are released; the project-maintained specification remains draft. Compatibility catalog 0.2.0 records the current mappings and preserves the earlier 0.1.0 snapshot.

- Name the overall representation family **DEEPBOM Evidence IR**.
- Establish **Provenance IR** as the sole current connection contract: schema, digest field, generic input, summary, section, API and export names.
- Remove the retired implementation module, schema, compatibility facade, selector aliases and documentation stub.
- Preserve the other four IR contracts. Recompute new provenance documents under their own hash contract; never rewrite archived reports.
- Add v2 remote result envelopes and update both widgets and server validators together. The September 29 ChatGPT compatibility repair also retains the original v1 base transport. Optional provenance requires v2; the browser/Claude envelope remains v2.
- Generate the family catalog from shared identities and the Provenance IR schema from shared contracts.
- Add negative tests for retired contracts and explicit migration/release instructions.
- Review pinned TEA sources and document a proposed exchange adapter outside the common evidence core.

See the [release record](RELEASE_2_0_0.md), [migration rules](MIGRATION.md) and [ChatGPT transport compatibility](../chatgpt-app/PUBLICATION_COMPATIBILITY.md). Neither publication nor deployment implies host directory approval, TEA integration, external standards endorsement or independent adoption.
