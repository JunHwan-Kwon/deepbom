# DEEPBOM CLI

Local deployment-artifact analysis for TFLite, ONNX, GGUF, SafeTensors, Core ML,
and ExecuTorch.

## Programmatic integration

The Node/TypeScript SDK is available from version **2.1.0**:

```sh
npm install deepbom@2.3.0
```

Follow the [integration walkthrough](https://github.com/JunHwan-Kwon/deepbom/blob/main/examples/integrations/README.md)
for runnable exporter, MLflow and release-review examples.

```js
import { inspect, audit, DeepBomPolicyBlocked } from "deepbom";

const summary = await inspect("model.onnx");
console.log(summary.artifact.sha256, summary.coverage);
try {
  const evidence = await audit("model.onnx", { gate: "defects" });
  // Store this native evidence; apply deployment policy separately.
} catch (error) {
  if (!(error instanceof DeepBomPolicyBlocked)) throw error;
  console.log(error.code, error.document);
}
```

This is an asynchronous ESM API for Node >=20 that invokes the bundled CLI
without a shell or engine download. It reuses the common analysis engine and
ships TypeScript declarations. Python uses the same engine in its wheel.
See the [SDK contract](https://github.com/JunHwan-Kwon/deepbom/blob/main/docs/SDK_CONTRACT.md)
for supported options, structured errors and independent IR versioning.

## Command line

For typed snapshots, lineage and model-change evaluation records:

```console
deepbom evidence-workflow request.json --file SHA256:./model.onnx --output result.json
deepbom evidence-workflow result.json --format html --output report.html
```

The SDK's `evidenceWorkflow(request, {files: [{sha256, path}]})` and local MCP's
`deepbom_evidence_workflow` use the same engine. Follow the
[complete synthetic example](https://github.com/JunHwan-Kwon/deepbom/tree/main/examples/evidence-workflow)
or open the [browser review](https://deepbom.org/reports/evidence/).
Missing bindings stay unresolved; external measurements are not recomputed.

For a local Codex or Claude Code project, preview and install the bundled Agent
Skill without operating an analysis server:

```console
npx -y deepbom@2.3.0 integrate codex
npx -y deepbom@2.3.0 integrate codex --apply
npx -y deepbom@2.3.0 integrate claude-code
npx -y deepbom@2.3.0 integrate claude-code --apply
```

```console
npx deepbom audit model.onnx --format cyclonedx
npx deepbom audit model.onnx --format sarif --output deepbom.sarif --fail-on high
npx deepbom capabilities --compact
npx deepbom gguf model.gguf --context 8192 --memory-mib 8192
npx deepbom audit Model.mlpackage --compact
npx deepbom audit safetensors-repository/ --compact
npx deepbom verify model.onnx --bom supplied.cdx.json --render markdown
npx deepbom contract capture model.onnx -o baseline.interface-contract.json
npx deepbom verify model.tflite --contract production-interface.json
npx deepbom diff baseline.gguf candidate.gguf --tensors --render markdown
npx deepbom explore model.tflite --target-profile target-profile.json
npx deepbom audit model.pte --executorch-build deepbom.executorch-build.json --compact
npx deepbom capabilities --format agent-json
npx deepbom capabilities --format agent-text
npx -y deepbom@2.3.0 mcp
```

Claude Desktop can install the version-matched `deepbom-2.3.0.mcpb` asset
from the corresponding GitHub Release as a local desktop extension.

ChatGPT developer-mode users can connect `https://deepbom.org/mcp` for one
authorized attachment. Analysis runs in the ChatGPT browser sandbox; the
DEEPBOM service receives only a bounded result and does not fetch or retain
model bytes. Public ChatGPT discovery requires separate OpenAI review. Use the
local package for confidential or large files, directories, shards, sidecars,
complete exports, and repeat automation.

On a Claude host that supports remote MCP Apps, private testers can add the
separate `https://deepbom.org/mcp/claude` connector. It requires an explicit
file selection inside the app and does not automatically access Claude
attachments. The selected bytes stay in the browser sandbox; only a bounded
result is returned to the conversation. Public Claude discovery remains
subject to Anthropic review. Use the local package or MCPB when model-derived
facts must not leave the computer.

The default output is a bounded human-readable summary. Use `--json` or
`--compact` for the complete analysis document, `--format envelope` for the
canonical cross-format contract, `--format cyclonedx` for CycloneDX 1.7, or
`--format sarif` for OASIS SARIF 2.1.0. `--policy-output` records a deterministic
finding gate when `--fail-on` is selected.

`verify --bom` reconciles an exactly selected CycloneDX 1.7 component with
artifact-observable facts; publisher declarations remain not assessable from
the file. `contract capture` creates an artifact-derived interface baseline,
not an approved production contract. `verify --contract` fails closed on
interface contradictions, `diff` preserves the
canonical multi-target TFLite delta ledger, and `explore` exposes deterministic
WASM Pareto candidates. These commands do not add a second analysis engine.

The package contains one generated JavaScript analysis bundle and the canonical
TFLite WebAssembly module. It does not send model bytes or results over the
network, and it does not depend on a hosted DEEPBOM analysis endpoint. A plain
chat client without local shell or MCP access cannot execute an audit.
TensorRT parser observations and build profiles can be imported with
`--tensorrt-parser-evidence` and `--tensorrt-profile`.

ONNX external data next to the model is discovered only from safe serialized
references. Use `--external-data-dir` to bind a different explicit root.
ExecuTorch PTE audits use the same option for PTD data; `--executorch-build`
binds a duplicate-key-checked selected-build and binary inventory without
promoting it to observed execution.
Core ML packages and sharded SafeTensors repositories are hashed as canonical
multi-file artifact sets rather than collapsed to one unqualified file hash.

Static provider/delegate placement remains predicted or conditionally eligible
unless an identity-bound runtime or parser observation is imported.

For the installed version, `deepbom --help` and `deepbom capabilities --compact`
are authoritative. The current source inventory is maintained in the
[generated CLI reference](https://github.com/JunHwan-Kwon/deepbom/blob/main/docs/CLI_REFERENCE.md).

This public channel package is licensed under Apache-2.0. Protected analyzers
and private rulepack-generation sources are not included.

## Optional metadata and OMOP connections

After inspecting a model, create a model-bound input with `deepbom audit model.onnx --metadata-template omop -o metadata.json`. Fill the selected OMOP 5.4/5.5 `CDM_SOURCE` row and institution/release identifiers, then use `--metadata metadata.json --evidence-files ./evidence --section provenance_ir --json`. A generic template connects non-OMOP datasets, cohorts, code, runs and reports through the same IR.

The Web **Provenance** workspace provides import, record creation, an interactive relationship diagram and JSON/CycloneDX downloads. Local MCP `deepbom_audit` accepts `metadata_template`, `metadata` and `evidence_files`; ChatGPT/Claude widgets share only explicitly requested counts and digests. Connections remain declarations: file hashes and reference consistency do not authenticate lineage or clinical suitability. See the [OMOP guide](https://deepbom.org/guides/omop-metadata/) and [common IR contract](https://github.com/JunHwan-Kwon/deepbom/blob/main/docs/PROVENANCE_IR.md).
