# ChatGPT plugin portal values

These values describe the `deepbom.agent_contract.v1` version `1.0.2` surface
served from `https://deepbom.org/mcp`. They are submission preparation, not an
approval or publication record.

## Info

- Directory icon: `assets/deepbom-directory-icon-512.png`
- Composer icon: `assets/deepbom-composer-icon-256.png` (256 x 256 PNG, no more than 10 KB)
- Name: `DEEPBOM Artifact Evidence`
- Version: `1.0.2` (new privacy/usage-statistics candidate; not an approval claim)
- Subtitle: `Inspect AI model artifacts`
- Category: `Developer Tools`
- Website: `https://deepbom.org/`
- Customer support: `https://deepbom.org/support`
- Privacy policy: `https://deepbom.org/privacy`
- Terms of Service: `https://deepbom.org/terms`
- Demo Recording URL: supplied privately by the publisher in the submission portal
- Commerce and purchasing: no; leave the external-purchase checkbox clear

Description:

> Inspect attached AI model deployment files without executing model code.
> DEEPBOM analyzes supported TFLite, ONNX, GGUF, SafeTensors, Core ML, and
> ExecuTorch files in the browser and reports SHA-256, serialized structure,
> tensor encodings, quantization evidence, and static findings. Explore Model IR
> tables and visualizations, and export SVG, PNG, Word-ready bundles, CycloneDX
> 1.7 artifact evidence, or SPDX 2.3 artifact inventories. Use Send PNG to chat to
> share a selected visualization or Save via ChatGPT to obtain a download link
> for a generated file when supported by the host. These actions share the
> generated export, not the model file. Results separate artifact defects,
> cautions, and evidence gaps, and explain what static analysis cannot establish.
> Use the local CLI or local MCP for confidential, very large, sharded, or
> multi-file artifacts.

Select the verified individual or business identity that legally owns this
submission. The Plugin Author value must exactly match that verified identity;
do not substitute the GitHub handle unless it is also the verified publisher
name.

The demo-recording URL above is the publisher-provided upload of
`DEEPBOM_review_live.mp4`. Enter it separately in the form; the root import JSON
does not contain this field. Keep the video accessible to reviewers without
sign-in or an expiring token. The recording demonstrates TFLite structure
analysis, a report in chat, visualization controls, the PNG handoff, and a
CycloneDX file opened through Save via ChatGPT. It does not demonstrate an SPDX
export or the deeper payload-integrity test. The scenarios in
`DEMO_RECORDING_SCRIPT.md` are recording guidance, not a record of tests run.

## MCP

- Submission type: `With MCP`
- URL type: `Universal`
- MCP server URL: `https://deepbom.org/mcp`
- Authentication: none
- Challenge base URL: `https://deepbom.org`

After the portal issues a token, expose exactly that token at
`https://deepbom.org/.well-known/openai-apps-challenge`, then run **Scan Tools**.
Review all four discovered tools and confirm that every annotation matches
`chatgpt-app-submission.json`.

## Skills

For the optional ChatGPT attachment workflow, upload the repository-root
`deepbom-artifact-evidence-skill.zip` or the folder
`docs/chatgpt-app/skills/deepbom-artifact-evidence/`. The bundle contains only
that folder's `SKILL.md`, covering browser analysis, evidence interpretation,
and explicit visualization/file sharing. Review the portal scan result before
continuing. Local bundle validation does not establish a portal scan result.

The separate `skills/deepbom/` skill targets the local CLI. Use the ChatGPT
bundle above for this browser-component submission. An MCP-only submission can
leave Skills empty.

## Prompts

Use these three starter prompts:

1. `Inspect this ONNX model's quantization evidence and limits.`
2. `List the tensor encodings in this GGUF artifact.`
3. `Check this TFLite model for serialized payload defects.`

## Testing

Import the five positive and three negative cases from the repository-root
`chatgpt-app-submission.json`. Cases 1–4 include pinned public download URLs in
both `file_attachment_urls` and the visible Scenario text, so the reviewer can
obtain and attach the model even if the portal does not display attachment URL
fields. Their instructions include waiting for `Static evidence ready` and
selecting `Report in chat`; the result-publication tool is called by the browser
component. Case 1 also describes the explicit image and file-sharing actions.

For case 4, `analysis_depth: payload_integrity` is an analyzer input and part of
the initial response, not a required field of the completed result. These cases
describe expected behavior; importing them is not evidence that a live ChatGPT
test run passed.

## Global

Select only countries or regions in which the individual publisher can provide
support and the linked policy and terms are intended to apply. This is a legal
and account-owned decision and is intentionally not preselected in source.

## Submit

Release notes:

> Initial submission of DEEPBOM Artifact Evidence. The plugin performs bounded,
> browser-local static inspection of one ChatGPT-authorized TFLite, ONNX, GGUF,
> SafeTensors, Core ML, or ExecuTorch deployment artifact. It returns only a
> validated, hash-bound result to the conversation and does not send model bytes
> to or retain them on the DEEPBOM service. The widget provides Model IR views,
> SVG, PNG, Word-ready bundles, CycloneDX 1.7 artifact evidence, and SPDX 2.3
> artifact inventories. Explicit controls can share a selected PNG or generated
> export with ChatGPT when the host supports file sharing. No authentication or
> demo account is required. Local CLI and MCP paths remain separate for
> confidential, large, or multi-file artifacts.

Complete policy attestations only after the Info, MCP, Skills, Prompts, Testing,
Global, and Submit tabs have been read back from the portal and the Developer
Mode recording and live test cases match the scanned contract.
