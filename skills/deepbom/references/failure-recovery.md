# Failure recovery

1. Exit `1`: report invocation, input, artifact, or output failure. Read structured stderr with `--error-format json` before changing the command.
2. Exit `2`: report a policy or verification block. The evidence document may still be valid and should be retained.
3. Exit `3`: report an incomplete verification binding. Do not call it a tool crash or an artifact defect.
4. For a large GGUF or SafeTensors file, retry with `--scan structure` when the question does not require payload integrity.
5. For remote input, require an immutable identity: a full Hugging Face commit, GCS object generation, or HTTPS SHA-256 fragment.
6. Do not raise download, response, or memory limits without explaining the cost and obtaining user approval.
7. If no shell, local MCP, or local file access exists, explain that actual analysis cannot run in that client and provide the pinned `npx` command.
