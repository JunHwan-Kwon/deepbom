# DEEPBOM ChatGPT plugin review package

This directory contains product-facing material for connecting and reviewing
the DEEPBOM ChatGPT plugin. It is not a standards-conformance package and contains
no CycloneDX development-branch claims.

## Reviewed endpoint

- Streamable HTTP MCP: `https://deepbom.org/mcp`
- Authentication: none
- Public tools: `deepbom_analyze_file`, `deepbom_capabilities`
- Component-only tools: `deepbom_publish_analysis`, `deepbom_publish_error`
- Widget resource: `ui://deepbom/analyzer.html`
- Privacy: `https://deepbom.org/privacy`
- Terms: `https://deepbom.org/terms`
- Support: `https://deepbom.org/support`

The app accepts one ChatGPT-authorized deployment-artifact attachment. The
widget downloads and analyzes that attachment inside the ChatGPT browser
sandbox. The remote DEEPBOM MCP endpoint serves the tool and widget contracts
and validates a bounded result or error; it does not fetch or retain model
bytes.

Use the local CLI or stdio MCP when the artifact or returned metadata is
confidential, when the artifact is too large for the browser path, or when the
analysis requires a directory, shards, sidecars, ONNX external data, complete
exports, or repeat automation.

## Submission procedure

1. Deploy the exact reviewed source and confirm `/mcp` initialization,
   `tools/list`, widget resource loading, one successful attachment analysis,
   one structured failure, and the model-byte transfer boundary.
2. Run the prompts in `evals.json` and the shared brandless cases in
   `../agent-evaluation/host-evaluation-cases.v1.json`. Copy the blank run
   template to `.local-validation/agent-host-evaluation/`, record observed
   tool selection and arguments, and ensure no negative prompt starts an
   artifact analysis. The checked-in template is not execution evidence.
3. Confirm the public privacy, terms, and support pages match the deployed data
   flow and that the listing identity belongs to the submitting account.
4. When the submission portal issues a domain-verification token, configure it
   as the Cloudflare Worker secret `OPENAI_APPS_CHALLENGE`. Confirm that
   `https://deepbom.org/.well-known/openai-apps-challenge` returns only that
   token, with no JSON wrapper or newline.
5. Submit `submission-profile.json` and the current brand asset through the
   ChatGPT plugin submission portal. Do not claim public catalog availability before
   approval.
6. After approval, verify discovery by app name and by at least three generic
   artifact-analysis queries, then repeat one full attachment analysis.

The root `plugin.json` and `mcp.json` also form a portable Agent Plugins
package. They make the Skill and remote MCP identity reviewable together, but
they do not replace endpoint registration, domain verification, or public
submission. The descriptive listing title is `DEEPBOM Artifact Evidence`; this
file makes no trademark-clearance claim.

The submission itself is a manual account and attestation action. It must not
be automated from a source checkout.
