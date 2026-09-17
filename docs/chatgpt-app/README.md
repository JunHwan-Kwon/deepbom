# DEEPBOM ChatGPT plugin review package

This directory contains product-facing material for connecting and reviewing
the DEEPBOM ChatGPT plugin. It is not a standards-conformance package and contains
no CycloneDX development-branch claims.

## Reviewed endpoint

- Streamable HTTP MCP: `https://deepbom.org/mcp`
- Authentication: none
- Public tools: `deepbom_analyze_file`, `deepbom_capabilities`
- Component-only tools: `deepbom_publish_analysis`, `deepbom_publish_error`
- Widget resource: `ui://deepbom/analyzer-v2.html`
- Privacy: `https://deepbom.org/privacy`
- Terms: `https://deepbom.org/terms`
- Support: `https://deepbom.org/support`

The app accepts one ChatGPT-authorized deployment-artifact attachment. The
widget downloads and analyzes that attachment inside the ChatGPT browser
sandbox. The remote DEEPBOM MCP endpoint serves the tool and widget contracts
and validates a bounded result or error; it does not fetch or retain model
bytes.

Optional usage statistics are off by default. After explicit consent, a random
browser key and allowlisted activity events support browser/revisit estimates,
format counts, and export counts. The service stores a keyed hash of the key,
never model contents, filenames, artifact hashes, or conversation text in the
usage tables. Withdrawal, deletion, and a test-use exclusion control are in the
widget. Operational MCP counts include all callers and have no browser key.
See [USAGE_METRICS.md](USAGE_METRICS.md) for definitions, administrator access,
retention, and deployment order. Review materials and the deployed privacy
page must describe this collection before enabling it in production.

The widget also renders deterministic, monochrome Model IR views from the same
browser-local analysis. Users can download the selected canonical SVG, a
300-DPI PNG derivative, or a Word-ready ZIP containing all six view levels,
captions, hashes, and insertion metadata. `Send PNG to chat` is an explicit
user action: it uploads only the selected derived page to the current ChatGPT
conversation and makes that image visible to the model. It does not upload the
model artifact to DEEPBOM.

The file buttons appear above the visualization and before the evidence tables,
including in narrow ChatGPT panels. `Download SVG`, `Download PNG`, and
`Word-ready bundle` prepare actual files and retain a visible `Local download`
link for retry.
`CycloneDX 1.7 JSON` exports the existing artifact-evidence document;
`SPDX 2.3 JSON` exports a SHA-256-bound artifact inventory with DEEPBOM evidence
annotations. SPDX uses FILE-purpose packages, leaves license/copyright fields
as `NOASSERTION`, and does not infer a software dependency inventory or an SPDX
3 AI profile. These files are generated in the browser. When a host blocks local
downloads, `Save via ChatGPT` explicitly uploads only the generated export to
ChatGPT and opens its host-issued HTTPS download URL. `Send link to chat`
requests a reply containing that URL. These actions require host file-sharing
support and do not send the original model file to DEEPBOM.

`Send PNG to chat` explicitly requests an inline image and a Download PNG link,
using the temporary URL returned by ChatGPT for the uploaded derived image when
available. If that optional URL API is unavailable, the image ID is still
shared and the local PNG download remains available. A host accepting the
follow-up request is not proof that it rendered a new response; the button
stays usable. Hosts without image sharing show the disabled action and a local
download alternative. The component reports height changes to supported hosts
to avoid clipping controls.

The same result now includes a bounded `deepbom.model_summary_conversation.v1`
projection. ChatGPT can report native operation/storage rows, output contracts,
predecessor references, bound serialized bytes/elements, and static MACs without
reconstructing the graph itself. The widget shows the complete local
`deepbom.model_summary.v1` table; the conversation projection is row-bounded and
states when it is truncated. Neither form infers framework trainability or
runtime order.

Use the local CLI or stdio MCP when the artifact or returned metadata is
confidential, when the artifact is too large for the browser path, or when the
analysis requires a directory, shards, sidecars, ONNX external data, complete
exports, or repeat automation.

## Submission procedure

1. Deploy the exact reviewed source and confirm `/mcp` initialization,
   `tools/list`, widget resource loading, one successful attachment analysis,
   one deterministic SVG view, one user-initiated PNG handoff, one structured
   failure, and the model-byte transfer boundary.
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
5. Upload the repository-root `chatgpt-app-submission.json`, the 512 px
   directory icon, and the 256 px composer icon through the ChatGPT plugin
   submission portal. `submission-profile.json` remains the fuller internal
   preparation record; it is not the portal import format. Follow
   `PORTAL_VALUES.md` for the remaining account-owned fields. Do not claim
   public catalog availability before approval.
6. After approval, verify discovery by app name and by at least three generic
   artifact-analysis queries, then repeat one full attachment analysis.

The optional portal skill is
`skills/deepbom-artifact-evidence/SKILL.md` within this directory. Its workflow
uses the ChatGPT attachment tools and widget. Upload its containing folder or
the repository-root `deepbom-artifact-evidence-skill.zip` in the Skills tab.
The repository-root `skills/deepbom/` remains the separate local CLI skill.

The root `plugin.json` and `mcp.json` also form a portable Agent Plugins
package. They make the Skill and remote MCP identity reviewable together, but
they do not replace endpoint registration, domain verification, or public
submission. The descriptive listing title is `DEEPBOM Artifact Evidence`; this
file makes no trademark-clearance claim.

The submission itself is a manual account and attestation action. It must not
be automated from a source checkout.
