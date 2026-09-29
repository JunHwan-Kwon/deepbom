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

This repair changes the ChatGPT widget and remote bridge. It does not change the analysis engine, CLI package, canonical IR, numeric calculations or Claude result contract.
