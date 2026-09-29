# DEEPBOM 2.0.0 release

This release establishes the canonical Evidence IR family and publishes a versioned compatibility explorer. The package major version changes because the earlier provenance contract and remote result envelopes are intentionally rejected, rather than silently converted.

- Provenance uses `deepbom.provenance_ir.v1`, the `provenance_ir` section and `provenance_ir_sha256`. Regenerate prior evidence from its original model and current metadata input; see [migration](MIGRATION.md).
- Agent contract `deepbom.agent_contract.v2` / 2.0.0 accompanies the v2 ChatGPT and browser result envelopes. The core artifact evidence envelope remains v1. Host submission/approval is separate from server deployment: the captured OpenAI/Claude metadata is a candidate, not an approval claim.
- The compatibility explorer at `/guides/evidence-ir/compatibility/` reads the same versioned JSON used by the GitHub catalog. Catalog 0.1.0 remains archived; 0.2.0 pins the release definitions. Search, schema inventories, connected mappings, source/check references and explicit support limits are included.
- Common scalar and exact-count rules, IR source binding, storage accounting and provenance consistency include the preceding correctness fixes. Static, derived, declared and imported runtime evidence remain distinct.
- Web assets and remote MCP validators are deployed together. Reconnect/refresh host metadata and open a new conversation/widget when moving from the older contract. Existing old evidence bytes remain usable with their original archived release.

No TEA client/server, external standards certification, automatic model execution or new clinical semantics are claimed. TEA exchange is recorded as proposed; SPDX remains a 2.3 artifact inventory, while CycloneDX 1.7 supports the bounded provenance projection.

Release validation runs locally; GitHub Actions is not used. Public source is generated through the exact reviewed allowlist. npm publication, production routing and registry metadata are verified after publication rather than inferred from a successful build.
