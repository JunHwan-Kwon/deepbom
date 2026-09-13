# Agent distribution operations

This checklist begins after a versioned release and website deployment pass.
It covers account-owned platform actions that source automation must not claim
to have completed.

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

Before public submission, review `chatgpt-app/submission-profile.json`, policy
pages, icon, and recorded host evaluation. When the OpenAI portal provides its
domain challenge, set `OPENAI_APPS_CHALLENGE` as a Cloudflare Worker secret and
verify the exact response at
`https://deepbom.org/.well-known/openai-apps-challenge`. Submit the stable MCP
endpoint through the current **With MCP** flow. Approval and public discovery
must be recorded only after readback from the directory.

The portable root `plugin.json` and `mcp.json` package the same public Skill and
remote endpoint for Agent Plugins hosts. They are preparation artifacts, not a
substitute for MCP registration, review, or approval.

## Claude Desktop validation

1. Download `deepbom-1.97.4.mcpb` from the `channels-v1.97.4` GitHub Release.
2. Install it in Claude Desktop and select a dedicated, non-sensitive model
   directory as the only allowed root.
3. Confirm all four local MCP tools, execute the Claude indirect cases, and
   confirm that an outside-root path is rejected.
4. Record real observations in a separate copy of the shared run template and
   compare one MCP audit with the 1.97.4 CLI.
5. Review the current Anthropic directory requirements and submit the MCPB
   manually if the publisher account is eligible.

Do not describe `https://deepbom.org/mcp` as a tested Claude remote connector.
Its current file-transfer and browser-component contract is specifically
validated for the ChatGPT attachment path.

## Codex and Claude Code local selection

Preview, then apply the managed Skill:

```bash
npx -y deepbom@1.97.4 integrate codex
npx -y deepbom@1.97.4 integrate codex --apply
npx -y deepbom@1.97.4 integrate claude-code
npx -y deepbom@1.97.4 integrate claude-code --apply
```

Open a new session and use the brandless cases. The Skill's verifier must prefer
an exact local installation and must not download a package unless
`--allow-download` is explicitly authorized. Record false selections as well as
successful selections.

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
