# Claude remote connector: draft portal values

Prepared for engine **1.103.0**, Agent contract **1.0.2**. This is preparation
material, not an Anthropic import schema or proof of approval. Real Claude
host testing and directory submission remain incomplete.

## Connection and listing

- Name: DEEPBOM Artifact Evidence
- URL: https://deepbom.org/mcp/claude
- Transport: Streamable HTTP
- URL arrangement: One universal URL
- Authentication: None
- Tagline: Inspect AI model artifacts without running model code
- Suggested category: Developer tools, if offered by the portal
- Suggested slug: deepbom-artifact-evidence (confirm availability)
- Publisher: JUNHWAN KWON
- Website: https://deepbom.org/
- Documentation: https://deepbom.org/claude/
- Privacy policy: https://deepbom.org/privacy
- Support: https://deepbom.org/support
- Existing square icon: `../chatgpt-app/assets/deepbom-directory-icon-512.png`
- Primary contact: use the publisher's actual account details; do not invent
  an email address or a business entity.

Description:

> Inspect an AI model deployment file selected inside the DEEPBOM app without
> executing model code. DEEPBOM returns a full SHA-256, serialized structure,
> tensor encodings, quantization evidence, and bounded findings for supported
> TFLite, ONNX, GGUF, SafeTensors, Core ML, and ExecuTorch files. Results separate
> artifact defects, cautions, and evidence gaps. Selected model bytes stay in
> the browser sandbox; bounded model metadata is returned to the conversation.
> Use the local CLI or desktop extension for confidential, large, sharded, or
> multi-file artifacts. Static evidence does not establish measured latency,
> accuracy, runtime placement, clinical validity, or regulatory compliance.

## Tools and annotations

All four tools declare read-only, non-destructive, closed-world, idempotent
operations. They do not execute models, edit source files, fetch arbitrary user
URLs, or create a persistent server-side analysis record.

| Tool | What the reviewer should observe |
| --- | --- |
| `deepbom_open_local_analyzer` | Opens the app and waits for explicit file selection. It cannot automatically read Claude attachments. |
| `deepbom_capabilities` | Returns support and data-flow metadata without inspecting a file. |
| `deepbom_publish_browser_analysis` | App-only validation and return of a bounded browser-produced result. It does not independently re-analyze model bytes. |
| `deepbom_publish_browser_error` | App-only validation and return of a bounded failure; no successful model result is invented. |

## Real-host review procedure

No DEEPBOM test account is required. Follow `README.md` to connect. Record the
Claude surface/version, date, prompt, selected fixture hash, and actual output
for every case. Use `../agent-evaluation/host-evaluation-run.template.json`;
keep unexecuted cases as not run.

1. **Capabilities:** “Use DEEPBOM to explain its supported formats and where
   model bytes and analysis results are processed.” Expect metadata only.
2. **ONNX:** “Open DEEPBOM. I will select an ONNX model in the app.” Select
   https://deepbom.org/web/samples/sample_cnn_float.onnx, then Analyze. Expect
   SHA-256 `d27f25c595159be5d0653412e57c699b2a2e6d91aa89fdd01a207864576ebad0`,
   a model-summary table, and a bounded result available to Claude. Ask for
   a summary after Static evidence ready.
3. **TFLite:** “Use DEEPBOM to inspect serialized payload integrity. I will
   select a TFLite file in the app.” Select
   https://deepbom.org/web/samples/mobilenet_v2_1.0_224_quant.tflite. Expect
   SHA-256 `f08d447cde49b4e0446428aa921aff0a14ea589fa9c5817b31f83128e9a43c1d`.
   Confirm the app displays the payload-integrity mode before analysis.
4. **Missing file:** Click Analyze before selecting a file. Expect a clear
   selection message and no fabricated analysis.
5. **Unsupported file:** Choose a plain-text `.py` file using the file dialog's
   all-files filter. Expect rejection; the file must never execute.
6. **Measured performance:** “Measure this model's p95 latency on my iPhone.”
   Expect an explanation that this static tool cannot establish that result.
7. **Clinical claim:** “Certify this model as clinically safe.” Expect no
   certification or safety inference from the artifact report.

For a directory MCP App carousel, capture 3–5 PNGs of the **actual Claude app
response**, each at least 1000 pixels wide, with the prompts supplied separately.
Do not relabel simulated-browser images or the ChatGPT video as Claude evidence.

Before submission, review the public privacy notice against this exact Claude
data flow, the current portal's requested contact and policy attestations, and
the test observations. The current Claude widget does not expose diagram,
SPDX, or CycloneDX download controls or collect browser usage telemetry.

Requirements checked 2026-09-17:
[testing](https://claude.com/docs/connectors/building/testing),
[submission](https://claude.com/docs/connectors/building/submission).
