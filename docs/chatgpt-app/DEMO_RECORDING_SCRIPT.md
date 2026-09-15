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
5. Ask: `Inspect the attached ONNX deployment model. Report its SHA-256,
   serialized graph summary, and evidence limitations.`
6. Show `deepbom_analyze_file`, the analyzer component, and the completed
   `deepbom_publish_analysis` result. Keep the filename, expected SHA-256,
   analyzer version, finding classes, and evidence limit visible.
7. In a separate fresh conversation, ask: `Is this model clinically safe and
   FDA compliant?` Show that no DEEPBOM tool is invoked and that ChatGPT does not
   convert static evidence into a regulatory verdict.
8. End by showing the public Privacy, Terms, and Support URLs in the plugin
   listing or submission draft.

The recording is review evidence for the tested interaction, not evidence of
OpenAI approval. Host it at a stable HTTPS URL that reviewers can open without
credentials, MFA, cookies from a private account, or an expiring share token.
