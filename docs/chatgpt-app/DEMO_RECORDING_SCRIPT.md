# ChatGPT Developer Mode demo recording

Record the final deployed plugin in a fresh ChatGPT Developer Mode conversation.
Do not show account tokens, browser developer tools containing temporary file
URLs, private models, local paths, or unpublished source.

## Recording sequence

1. Show that the connected plugin is named `DEEPBOM Artifact Evidence` and uses
   `https://deepbom.org/mcp`.
2. Ask: `Which AI model artifact formats can this plugin inspect, where are
   attached bytes processed, and when should I use the local CLI instead?`
3. Expand the call and show `deepbom_capabilities` returning the browser/local
   processing distinction and static-evidence boundary.
4. Attach the pinned public `gpu_partition_probe.onnx` fixture referenced by
   `chatgpt-app-submission.json`.
5. Ask: `Inspect the attached ONNX deployment model. Show its Model IR
   visualization and report its SHA-256, serialized graph summary, and evidence
   limitations.`
6. Show `deepbom_analyze_file`, the analyzer component, and the completed
   `deepbom_publish_analysis` result. When the component says `Static evidence
   ready`, select `Report in chat` once. This explicit user action asks ChatGPT
   to interpret the completed bounded result in a new turn; it is not a retry
   of artifact analysis. Keep the filename, expected SHA-256, analyzer version,
   finding classes, and evidence limit visible. Do not describe the expected
   two-stage handoff as an artifact-analysis failure.
7. In the component, switch from `V1 Architecture overview` to `V3 Exhaustive
   graph / storage`. Show the page counter and the canonical SVG digest, then
   use the file buttons above the preview to download SVG, PNG, and the
   Word-ready ZIP. Download `CycloneDX 1.7 JSON` and `SPDX 2.3 JSON`, open the
   files, and confirm that each contains the same artifact SHA-256. SPDX is a
   file-purpose package inventory with explicit license unknowns; do not
   describe it as a complete software SBOM or SPDX 3 AI profile.
   Return to V1, select `Send PNG to chat`, and show the
   derived image and Download PNG link in the next ChatGPT response. If the
   host cannot supply a temporary image URL, show the local PNG download
   instead and record that host limitation. The bounded report must describe
   these as widget controls rather than claiming that DEEPBOM has no export
   operation. State that only this derived PNG, not the model bytes, was
   uploaded by that explicit action.
8. In a separate fresh conversation, ask: `Is this model clinically safe and
   FDA compliant?` Show that no DEEPBOM tool is invoked and that ChatGPT does not
   convert static evidence into a regulatory verdict.
9. End by showing the public Privacy, Terms, and Support URLs in the plugin
   listing or submission draft.

The recording is review evidence for the tested interaction, not evidence of
OpenAI approval. Host it at a stable HTTPS URL that reviewers can open without
credentials, MFA, cookies from a private account, or an expiring share token.
