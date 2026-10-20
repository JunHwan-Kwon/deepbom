# ChatGPT publication compatibility

The engine version, Evidence IR identities, conversation transport and directory version have separate lifecycles. Publishing an approved directory entry connects users to the live MCP server; it does not restore the server used during submission.

## Stable basic reporting

The four tool names, original analysis-start arguments, resource URIs, authentication and CSP remain available. The current result bridge accepts both original base `deepbom.chatgpt_analysis_result.v1` and `deepbom.chatgpt_analysis_result.v2`. The v1 branch is frozen to the historical base fields. Both use the same numerical/identity validator and preserve the submitted schema, hashes and declared producer version. A producer version is a declaration, not attestation that the server reproduced the analysis.

The widget tries the v1 base envelope first. A host with a cached v2-only definition can reject this before calling the server; only an explicit schema rejection triggers one v2 retry. Network/authentication errors, tool failures and missing/mismatched acknowledgements are not treated as success. The server returns the same accepted envelope so the host's output-schema check also passes. Capability additions are nested inside the extensible `chatgpt_path` field, preserving the original closed top-level output schema.

## Optional provenance

Canonical Provenance IR is unchanged. Retired provenance fields and schema names remain rejected. Metadata reporting uses only the v2 envelope, with source/hash validation. A host still enforcing the original definition cannot accept that optional result. The widget explains the limitation and keeps the local canonical Provenance IR JSON export available; it never silently drops the metadata or reports an unacknowledged result as delivered.

## Publication and updates

An existing approved version can be published while later tool updates undergo continuous review, provided its live contracts still work. Under the current [OpenAI maintenance policy](https://developers.openai.com/plugins/deploy/app-review#how-published-mcp-metadata-versions-work), new/changed tool definitions appear after automated checks; previous definitions remain in use while changes are held. Listing information and imported Skills still require a new-version review. Do not claim that server deployment implies directory approval or that a refresh of a developer connection updates the published definition.

## Verification boundary

`scripts/fixtures/chatgpt-tools-20260916.json` pins the source commit and source digest of the historical four-tool definition. It is not an authenticated export of the approved portal snapshot. Server tests validate input and output against that frozen definition, the intervening v2-only definition and the updated union. Browser tests apply the historical schema on both sides of the actual result bridge, analyze ONNX and TFLite, test file exports, and exercise the v2-only retry. No portal approval is inferred from these tests. After publishing, run one attachment through the actual listed plugin and request its report in chat.

## 2.0.0 submission history

The 2.0.0 submission added `deepbom_query_file` and the component-only
`deepbom_publish_query`. The four original tools and their basic reporting
compatibility remain available. Query completion requires the new tools to be
available in the connection; a cached four-tool connection must be refreshed
in Developer Mode or await its reviewed directory metadata. A rejected query
publication is shown as undelivered, never fabricated as an accepted answer.

That submission also revised the imported Skill, description and multi-turn
test cases, requiring an updated Skill ZIP and directory draft for that change.
This is not a requirement for every subsequent engine release. The source
baseline is a review candidate, not a record of approval. See
[Conversational evidence queries](INTERACTIVE_REVIEW.md) for the data flow,
profile limits and recording sequence.

These changes affect the ChatGPT widget and remote bridge. They do not change
the analysis engine, CLI package, canonical IR, numeric calculations or Claude
result contract.

## Routine updates after 2.0.0

Prefer changes behind the existing tools and shared Evidence IR workflow.
Keep the plugin package, imported Skills, endpoint, resource URIs, and data
handling boundary stable unless the requested functionality requires a change.
Reuse the common engine, query builders, and renderers; do not create separate
ChatGPT calculations or parsing rules.

Apply improvements in this order:

1. Correctness, performance, result explanations, and widget navigation that
   preserve existing request/response schemas and their meanings.
2. Explicit MCP contract additions only when the existing API cannot express
   the request. Follow the [platform operations](../AGENT_CONTRACT_VERSIONING.md#platform-operations)
   for continuous review; do not rewrite Skills solely for a tool-only update.
3. Package/Skill changes when the user workflow actually needs new instructions
   or listing information. Group related changes into a deliberate submission.

For example, existing literal search and paginated operation details can be
improved internally. Numeric filters, a new hierarchy query section, or a
second attachment need an explicit contract assessment. Do not encode hidden
commands in `search` or change existing fields' meanings to avoid that review.
An optional request field is not automatically compatible: the query result
also echoes a closed request schema. Preserve old result validation, or
introduce an explicitly versioned contract with compatible handling.

For each release, run `npm run check:agent-contract-release` before deployment.
For runtime changes, exercise the affected widget/bridge path against the live
and historical schemas, including query pagination and artifact identity when
applicable. Keep real host checks separate from controlled browser tests.
A baseline update records a reviewed source candidate, never approval; verify
new tool availability in the portal before advertising it as live.
