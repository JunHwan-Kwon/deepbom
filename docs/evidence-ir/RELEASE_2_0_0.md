# DEEPBOM 2.0.0 release

Released September 29, 2026. This release establishes the canonical Evidence IR family and publishes a versioned compatibility explorer. The package major version changes because the earlier provenance contract is intentionally rejected, rather than silently converted. Conversation transport compatibility is separate: ChatGPT accepts the original v1 base result and the v2 result, while optional provenance requires v2. Browser/Claude result envelopes use v2.

- Provenance uses `deepbom.provenance_ir.v1`, the `provenance_ir` section and `provenance_ir_sha256`. Regenerate prior evidence from its original model and current metadata input; see [migration](MIGRATION.md).
- Agent contract `deepbom.agent_contract.v2` / 2.0.0 accompanies these result transports. The core artifact evidence envelope remains v1. The ChatGPT v1 adapter does not accept retired provenance names or rewrite IR digests; see [publication compatibility](../chatgpt-app/PUBLICATION_COMPATIBILITY.md). Host submission/approval is separate from server deployment: captured OpenAI/Claude metadata is not an approval record.
- The compatibility explorer at `/guides/evidence-ir/compatibility/` reads the same versioned JSON used by the GitHub catalog. Catalog 0.1.0 remains archived; 0.2.0 pins the release definitions. Search, schema inventories, connected mappings, source/check references and explicit support limits are included.
- Common scalar and exact-count rules, IR source binding, storage accounting and provenance consistency include the preceding correctness fixes. Static, derived, declared and imported runtime evidence remain distinct.
- Web assets and remote MCP validators are deployed together. New query tools require updated host metadata; an existing basic ChatGPT report can continue through the supported v1 transport. Developer Mode reconnection does not update a published directory definition. Existing old provenance documents remain usable with their original archived release.

No TEA client/server, external standards certification, automatic model execution or new clinical semantics are claimed. TEA exchange is recorded as proposed; SPDX remains a 2.3 artifact inventory, while CycloneDX 1.7 supports the bounded provenance projection.

Release validation runs locally; GitHub Actions is not used. Public source is generated through the exact reviewed allowlist. npm publication, production routing and registry metadata are verified after publication rather than inferred from a successful build.
