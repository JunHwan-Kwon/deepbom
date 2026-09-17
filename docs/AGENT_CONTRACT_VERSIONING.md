# Engine and Agent contract versioning

DEEPBOM publishes three independent identities:

| Identity | Current value | Changes when |
| --- | --- | --- |
| Engine release | `package.json` / `release/version.json` | Parser, analysis, performance, packaging, or other product code is released |
| Public Agent contract | `deepbom.agent_contract.v1` / `1.0.2` | Reviewed tool metadata, schemas, annotations, instructions, UI resource identity or policy metadata, or the public Skill changes |
| Evidence contract | `deepbom.artifact_evidence_envelope.v1` / `1.0.0` | A removal or semantic incompatibility changes the stable evidence envelope |

An engine update may change `analyzer_version` and measured results while keeping
the Agent and evidence contracts unchanged. The public Skill resolves an engine,
checks both contract identities, runs the self-test, and records the exact engine
version it actually used. It does not silently accept a product merely because
its executable name is `deepbom`.

## Release gate

`npm run check:agent-contract-release` snapshots the following source material:

- OpenAI MCP tool names, titles, descriptions, input/output schemas,
  annotations, security schemes, instructions, resource URI, visibility,
  resource metadata, domain, and CSP;
- portable/Codex plugin manifests and every file in the public Skill;
- declared privacy and model/result transfer boundaries;
- the equivalent Claude remote MCP and listing surface; and
- the stable evidence-contract identity and compatibility boundary.

The snapshot intentionally excludes the engine version, live analysis values,
and backward-compatible widget content served at the existing resource URI.
Those are product/runtime changes, not new tool contracts.

The evidence identity also pins its required top-level fields and semantic
boundaries. An executable CLI test checks those fields against a real envelope;
additive conditional fields remain compatible within v1.

An ordinary release must match `release/agent-contract-baseline.v1.json`. A
contract mismatch fails the release gate. Do not rewrite the baseline to make a
routine release pass. A deliberate contract proposal uses:

```bash
node scripts/check-agent-contract-release.mjs \
  --write-baseline \
  --acknowledge-contract-review-candidate
```

Run that command only after reviewing the changed surface, selecting the next
Agent/evidence contract version where required, and preparing the applicable
platform review. The generated baseline remains a candidate record; it never
asserts platform approval.

## Platform operations

For a source-classified engine-only update, run the normal release suite and a
real host smoke test. Do not create a new OpenAI plugin version solely because
the DEEPBOM engine version changed.

For an OpenAI-facing contract change, keep the existing published surface
compatible, prepare a new plugin version, scan its tools, complete review, and
publish it through the account-owned portal. An origin change is treated as a
new plugin, while an endpoint-path change is treated as a reviewed new version.
The current operational source is:
https://developers.openai.com/plugins/deploy/app-review#how-published-mcp-metadata-versions-work

Claude remote MCP clients refresh tool surfaces on connection, so a tool change
does not by itself require connector resubmission. The refreshed surface still
requires compatibility and host validation, and listing metadata changes follow
Anthropic's listing-management process. The current operational sources are:
https://claude.com/docs/connectors/building/after-publishing and
https://claude.com/docs/connectors/building/managing-your-listing

Claude Desktop MCPB is an engine-bearing local package. Each engine release gets
a new MCPB version and asset. Directory installations may update automatically;
privately distributed bundles require user reinstallation. This is independent
of the remote Agent contract review state.

## Approval ledger boundary

Source files may record `candidate_unsubmitted`, `approved`, or `listed` only
from an account-owned observation. CI, a successful deployment, or a contract
fingerprint cannot establish approval. Host run templates likewise leave
`deepbom_version` blank until a real session records the engine it used.
