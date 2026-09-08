# CLI automation contract

DEEPBOM exposes one analysis implementation through npm, Python, Cargo, and the
repository CLI. Python and Cargo are verified launchers; they do not contain a
second parser or a second numerical rule set.

The complete human-readable command and option inventory is generated directly
from the executable and capability document in
[the CLI reference](CLI_REFERENCE.md). Run `npm run generate:cli-docs` after a
CLI contract change; `npm run check:cli-docs` rejects a stale reference.

## Discovery

Use capability discovery instead of parsing help text:

```console
deepbom capabilities --compact
```

The `deepbom.cli_capabilities.v1` document declares commands, supported file and
package classes, output contracts, target-profile support, policy levels,
privacy behavior, and exit codes. Breaking machine-contract changes require a
new schema identifier. Additive fields may be introduced within the current
schema.

## Machine outputs

| Option | Contract | Intended consumer |
| --- | --- | --- |
| `--compact` | Complete format-specific analysis JSON | Deep format inspection |
| `--format envelope` | `deepbom.artifact_evidence_envelope.v1` | Cross-format automation |
| `--format cyclonedx` | CycloneDX 1.7 JSON | ML-BOM and supply-chain tooling |
| `--format sarif` | OASIS SARIF 2.1.0 Errata 01 | CI findings and security dashboards |

The envelope is the canonical cross-format contract. It binds artifact identity,
capability status, interfaces, graph totals, external files, normalized findings,
format extensions, provenance, and its own reproducible SHA-256. Format-specific
analysis remains intentionally richer and can evolve independently.

SARIF results point to the binary model artifact rather than inventing source
line locations. Each result carries a stable artifact/finding fingerprint,
evidence class, severity, interpretation, recommendation, and evidence pointers.
The emitted document is tested offline against the source-pinned OASIS schema.

## CI policy gate

`--fail-on` evaluates normalized findings rather than parser warning strings:

```console
deepbom audit model.onnx --format sarif --output deepbom.sarif \
  --fail-on high --policy-output deepbom-policy.json
```

Accepted thresholds are `informational`, `low`, `medium`, and `high`. A finding
at or above the threshold returns exit code `2`. The requested evidence document
is still completed, so CI can archive or upload it. The policy result binds the
finding counts and blocking IDs to the canonical envelope SHA-256.

No threshold is enabled by default. Organizations must choose a threshold that
matches their review policy; DEEPBOM does not silently turn research limitations
or predicted deployment risks into a release policy.

For repeat review, `--review-policy policy.json` replaces `--fail-on`. Its
result keeps command completion, required-analysis coverage, and finding policy
as independent states. Exceptions require an exact artifact scope, owner,
reason, creation time, and optional expiry; target, analyzer, and rulepack
identity constraints are re-evaluated on every run. Coverage loss is never
reported as a resolved finding.

```console
deepbom audit model.tflite --format envelope \
  --review-policy review-policy.json \
  --policy-output review-result.json
```

## Assistant tool access

`deepbom mcp` speaks the Model Context Protocol over stdio. It declares four
tools: `deepbom_capabilities`, `deepbom_audit`, `deepbom_diff`, and
`deepbom_explain_rule`. Every call re-enters this CLI in a cancellable child
process on the same machine. There is no
hosted analysis endpoint and no artifact upload; the privacy contract is the one
`capabilities` reports.

```json
{ "mcpServers": { "deepbom": { "command": "npx", "args": ["deepbom", "mcp"] } } }
```

The tool schemas reject undeclared arguments, so an assistant cannot pass an
option that has not been reviewed here. Tool descriptions carry the finding-kind
distinction and the static-evidence boundary, because the model reads them
before it writes a summary. `scripts/check-mcp-server.mjs` rejects a version
that drops either statement.

`deepbom_audit` defaults to a bounded human summary. Complete evidence requires
an explicit `output_format`, `section`, or `pointer`; `scan` lets the caller
select `structure` or streamed `integrity` work for large GGUF and SafeTensors
artifacts. Local paths are restricted to the launch directory by default. Set
`DEEPBOM_MCP_ALLOWED_ROOTS` to a platform-delimited allowlist when the server
must inspect other directories. `DEEPBOM_MCP_MAX_RESPONSE_BYTES`,
`DEEPBOM_MCP_MAX_FRAME_BYTES`, `DEEPBOM_MCP_TOOL_TIMEOUT_MS`,
`DEEPBOM_MCP_MAX_CONCURRENT`, and `DEEPBOM_MCP_MAX_QUEUE` are bounded controls;
raising one is a deliberate host policy decision.

MCP is a transport, not an analysis command: it produces no evidence document of
its own and therefore declares no output contract in
`deepbom.cli_capabilities.v1`.

Agent-facing usage guidance lives in [the DEEPBOM skill](../skills/deepbom/SKILL.md).

## CPU and accelerator bindings

`--target` and `--target-profile` bind a TFLite CPU cost model. The envelope
records whether the target came from the default assumption, an explicit id, or
a profile file; it always records `host_observed: false`.

Accelerator evidence is separate and uses the shared
`deepbom.accelerator_binding.v1` lifecycle. Supported imports include an
MLComputePlan (`--coreml-compute-plan`), an Edge TPU compiler operation ledger
(`--edgetpu-compiler-evidence`), TensorRT parser/engine evidence, and an NVIDIA
host profile. A compiled plan never becomes observed assignment without a
separate runtime trace.

For deterministic N-way source/compiled-plan comparison, use:

```console
deepbom placement model.tflite --profiles xnnpack_cpu,tflite_coreml_delegate,litert_qualcomm_qnn --compact
```

The result is `deepbom.placement_comparison.v1`. Every selected row must
classify the complete canonical graph; no provider order, physical transfer,
generated kernel, latency, or task-correctness claim is inferred.

## Reproducibility and output safety

```console
SOURCE_DATE_EPOCH=0 deepbom audit model.onnx --format envelope --compact
deepbom audit model.onnx --format cyclonedx --timestamp 2026-08-30T00:00:00Z
deepbom audit model.onnx --format envelope --output evidence.json --no-clobber
```

`--timestamp` takes precedence over `SOURCE_DATE_EPOCH`. The canonical envelope
uses `null` when neither is supplied, avoiding an implicit clock dependency.
CycloneDX requires a timestamp and therefore uses the current time only when no
reproducible source is supplied.

File output is written to a same-directory temporary file, synchronized, and
then replaced. `--no-clobber` reserves the destination exclusively and refuses
to replace existing bytes. Use `--output -` for explicit stdout.

Stable artifact identity, safe package traversal, sidecar discovery, and range
reads require a regular file or directory. Symbolic stdin is deliberately not
accepted because buffering multi-gigabyte artifacts would weaken those
contracts.

## Errors and exit codes

`--error-format json` emits `deepbom.cli_error.v1` to stderr. Machine evidence
continues to use stdout, so diagnostics cannot corrupt JSON or SARIF output.

| Exit | Meaning |
| --- | --- |
| `0` | Command and requested gate passed |
| `1` | Invocation, input, analysis, or output failure |
| `2` | Finding policy or interface verification blocked |
| `3` | Interface verification could not establish a complete binding |

`verify`, `diff`, and `explore` retain their command-specific evidence schemas.
They reject unrelated output formats and policy options instead of silently
ignoring them.

## Evidence boundary

CLI execution is local and does not upload artifact bytes or analysis results.
Static eligibility is not actual provider/delegate assignment. SARIF severity is
an engineering triage level, not a probability of failure, clinical risk score,
or release-readiness decision. Runtime assignment, task accuracy, clinical
validity, and physical transfer or latency require separately identified
evidence.
