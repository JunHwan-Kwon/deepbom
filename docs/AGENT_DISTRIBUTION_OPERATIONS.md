# Agent distribution operations

This checklist begins after a versioned release and website deployment pass.
It covers account-owned platform actions that source automation must not claim
to have completed.

Before either deployment, run `npm run check:agent-contract-release`. A pass
means the engine update preserves the fingerprinted Agent and evidence
contracts; it is not an approval record. See `AGENT_CONTRACT_VERSIONING.md` for
the deliberate contract-change path.

## ChatGPT developer-mode validation

1. In ChatGPT developer mode, register `https://deepbom.org/mcp` as a
   Streamable HTTP plugin endpoint with no authentication.
2. Confirm that the public tools are `deepbom_analyze_file` and
   `deepbom_capabilities`; the result and error bridge tools must remain hidden
   from the model.
3. Attach one small supported public fixture and run the direct, indirect, and
   negative ChatGPT cases in
   `agent-evaluation/host-evaluation-cases.v1.json`.
4. Copy `agent-evaluation/host-evaluation-run.template.json` to
   `.local-validation/agent-host-evaluation/` and record the ChatGPT build,
   observed tool, arguments, result status, and evidence-boundary behavior.
5. Compare the successful attachment result with the same release's local CLI
   result for artifact SHA-256, format, analyzer version, and bounded summary.

Before public submission, the publisher must complete OpenAI identity
verification and hold **Apps Management: Write** permission in the submitting
Platform organization. Review `chatgpt-app/submission-profile.json`, policy
pages, icon, and a real new-session host evaluation containing at least five
positive and three negative cases. Do not submit the blank template as test
evidence.

When the OpenAI portal provides its domain challenge, enter it locally without
putting the token in source control or chat history:

```powershell
$challenge = Read-Host "OpenAI domain challenge"
$challenge | npx wrangler secret put OPENAI_APPS_CHALLENGE
Remove-Variable challenge
```

Verify that the exact value, with no HTML or extra newline, is returned from
`https://deepbom.org/.well-known/openai-apps-challenge`. In the plugin portal,
choose **With MCP**, enter the stable `https://deepbom.org/mcp` endpoint, use
**Scan Tools**, compare the detected public/private tool visibility and
annotations with the prepared profile, then submit. Approval and public
discovery must be recorded only after directory readback.

The portable root `plugin.json` and `mcp.json` package the same public Skill and
remote endpoint for Agent Plugins hosts. They are preparation artifacts, not a
substitute for MCP registration, review, or approval.

## Claude Desktop validation

1. Download `deepbom-1.99.2.mcpb` from the `channels-v1.99.2` GitHub Release.
2. Install it in Claude Desktop and select a dedicated, non-sensitive model
   directory as the only allowed root.
3. Confirm all four local MCP tools, execute the Claude indirect cases, and
   confirm that an outside-root path is rejected.
4. Record real observations in a separate copy of the shared run template and
   compare one MCP audit with the 1.99.2 CLI.
5. Review the current Anthropic directory requirements and submit the MCPB
   manually if the publisher account is eligible.

The MCPB manifest deliberately lists only `darwin` and `win32`. Source and CI
validate its manifest, packaged bytes, four runtime calls, annotations, and
outside-root rejection. Those checks do not replace installation through the
actual Claude Desktop UI. Record at least one Windows and one macOS installation
and tool call before describing both platforms as host-validated.

## Claude remote MCP App validation

The candidate remote endpoint is `https://deepbom.org/mcp/claude`. Do not use
the ChatGPT endpoint for this test. The Claude endpoint does not automatically
read conversation attachments. Its first tool opens an MCP App in which the
user explicitly selects one local file; analysis runs in that browser sandbox
and only a bounded result is returned through the connector.

1. Add `https://deepbom.org/mcp/claude` as a private custom connector in
   Claude.ai. No DEEPBOM account or test credential is required.
2. Confirm that the analyzer opens a file picker, does not claim automatic
   attachment access, and accurately describes its local Desktop alternative.
3. Use a non-confidential public fixture. Compare its SHA-256, format, analyzer
   version, finding counts, and evidence boundary with the same release's CLI.
4. Run the `claude_remote_browser_mcp_app` cases from the shared catalog in new
   sessions. Record whether the tool was selected, the picker opened, analysis
   completed, and the host accepted the returned model context.
5. Repeat on Claude Desktop and mobile only where that host exposes the MCP App.
   Treat a missing picker, unsupported UI capability, or failed context update
   as a host-specific limitation, not as a passing result.
6. Capture MCP Inspector results for all four endpoint tools and actual-host
   screenshots only from real sessions. Never fabricate directory evidence.

The remote result can contain hashes, model names, tensor names, architecture
facts, and findings even though the original bytes are not sent to DEEPBOM.
Use the local MCPB or CLI when those derived facts must remain on the computer.
Submit to Anthropic only after the publisher has rechecked current directory
requirements, privacy copy, screenshots, test evidence, and the account-owned
submission form. Directory approval must be recorded only after readback.

## Codex and Claude Code local selection

Preview, then apply the managed Skill:

```bash
npx -y deepbom@1.99.2 integrate codex
npx -y deepbom@1.99.2 integrate codex --apply
npx -y deepbom@1.99.2 integrate claude-code
npx -y deepbom@1.99.2 integrate claude-code --apply
```

Open a new session and use the brandless cases. The Skill's verifier must prefer
a contract-compatible local installation, record its exact observed engine
version, and never download a package unless `--allow-download` is explicitly
authorized. Record false selections as well as successful selections.

## Search discovery readback

1. Add the `deepbom.org` domain property in Google Search Console if it is not
   already verified.
2. Submit `https://deepbom.org/sitemap.xml`.
3. Inspect and request indexing for `/guides/`,
   `/guides/inspect-onnx-quantization/`,
   `/guides/inspect-gguf-tensor-encodings/`, and
   `/guides/compare-model-artifacts/`.
4. Confirm the live canonical URL, rendered text, and structured data before
   treating a URL as discoverable. Keep impressions and queries as observations
   over time; do not infer indexing from a successful deployment.

The guide pages answer concrete artifact questions and link to exact commands
and evidence limits. Do not generate near-duplicate pages by changing only a
model name or format keyword.

## Identity and claims

Use the descriptive listing title `DEEPBOM Artifact Evidence` while retaining
the package name `deepbom`. This reduces ambiguity with similarly named
software but is not a legal clearance determination. Every platform listing
must preserve these boundaries:

- static artifact evidence is not measured runtime or task evidence;
- a directory listing is not an endorsement or regulatory recognition;
- a host prompt catalog is not an observed evaluation run;
- model bytes and returned metadata have different transfer and confidentiality
  implications.
