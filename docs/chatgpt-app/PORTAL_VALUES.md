# ChatGPT plugin portal values

These values describe the `deepbom.agent_contract.v1` version `1.0.0` surface
served from `https://deepbom.org/mcp`. They are submission preparation, not an
approval or publication record.

## Info

- Directory icon: `assets/deepbom-directory-icon-512.png`
- Composer icon: `assets/deepbom-composer-icon-256.png` (256 x 256 PNG, no more than 10 KB)
- Name: `DEEPBOM Artifact Evidence`
- Version: `1.0.0`
- Subtitle: `Inspect AI model artifacts`
- Category: `Developer Tools`
- Website: `https://deepbom.org/`
- Customer support: `https://deepbom.org/support`
- Privacy policy: `https://deepbom.org/privacy`
- Terms of Service: `https://deepbom.org/terms`
- Commerce and purchasing: no; leave the external-purchase checkbox clear

Description:

> Inspect attached AI model deployment files without executing model code.
> DEEPBOM reports artifact identity, serialized structure, tensor encodings,
> quantization evidence, and bounded findings for supported TFLite, ONNX, GGUF,
> SafeTensors, Core ML, and ExecuTorch files. Results separate artifact defects,
> cautions, and evidence gaps, and state what static analysis cannot establish.
> Use the local CLI or local MCP instead for confidential, very large, sharded,
> or multi-file artifacts.

Select the verified individual or business identity that legally owns this
submission. The Plugin Author value must exactly match that verified identity;
do not substitute the GitHub handle unless it is also the verified publisher
name.

The demo-recording URL remains an account-owned manual input. Record the final
Developer Mode interaction described in `DEMO_RECORDING_SCRIPT.md`, upload it to
an HTTPS URL accessible to reviewers without sign-in or an expiring token, and
paste that URL into the form.

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

## Prompts

Use these three starter prompts:

1. `Inspect this ONNX model's quantization evidence and limits.`
2. `List the tensor encodings in this GGUF artifact.`
3. `Check this TFLite model for serialized payload defects.`

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
> to or retain them on the DEEPBOM service. No authentication or demo account is
> required. Local CLI and MCP paths remain separate for confidential, large, or
> multi-file artifacts.

Complete policy attestations only after the Info, MCP, Skills, Prompts, Testing,
Global, and Submit tabs have been read back from the portal and the Developer
Mode recording and live test cases match the scanned contract.
