---
name: deepbom
description: Audit an on-device AI model artifact (.tflite, .onnx, .gguf, .safetensors, .mlmodel, .pte, .ptd) with the local DEEPBOM CLI, and read its evidence output without overstating it. Use when asked to inspect, audit, diff, or produce an ML-BOM/SARIF record for a deployed model file.
---

# DEEPBOM artifact audit

DEEPBOM reports what a model artifact's serialized bytes actually state. It runs
entirely in the local process: artifact bytes are never uploaded and there is no
hosted analysis endpoint. To analyse a user's model, run the CLI on their
machine.

## Before running anything

Discover the contract instead of parsing help text, because the option surface
moves between versions:

```bash
npx deepbom capabilities --json
```

The `deepbom.cli_capabilities.v1` document declares `commands`, `inputs`
(supported extensions and package kinds), `output_contracts`, `scan_policies`,
`targets`, `automation`, `privacy`, and `exit_codes`. Read it once per session
and use it — not this file — as the authority for what the installed version
supports.

If the user has no installation, `npx deepbom` fetches it. `npx deepbom
self-test` confirms the installed engine reproduces its pinned fixture before
its output is relied on.

## Auditing one artifact

```bash
npx deepbom audit model.tflite --output-format json          # complete format-specific analysis
npx deepbom audit model.onnx  --output-format envelope       # deepbom.artifact_evidence_envelope.v1
npx deepbom audit model.onnx  --output-format cyclonedx      # CycloneDX 1.7 ML-BOM
npx deepbom audit model.onnx  --output-format sarif          # SARIF 2.1.0 for CI findings
```

Choose by consumer: **envelope** for cross-format automation (it is the stable
canonical contract and carries its own reproducible SHA-256), **json** when a
format-specific detail is the actual question, **cyclonedx** for supply-chain
tooling, **sarif** for a CI findings surface.

Large analysis JSON does not need to be read whole:

```bash
npx deepbom audit model.tflite --list-sections
npx deepbom audit model.tflite --section quantization --output-format json
npx deepbom audit model.tflite --pointer /operator_count
```

Sidecars are explicit, never guessed: ONNX `external_data` and ExecuTorch `.ptd`
resolve through `--external-data-dir <directory>`.

Two artifacts:

```bash
npx deepbom diff baseline.tflite candidate.tflite --json
```

## Reading the output

Findings carry a `kind`, and the three kinds must stay separate in whatever you
report back:

| kind | meaning | correct response |
| --- | --- | --- |
| `artifact_defect` | a problem in the artifact itself | report it as a defect |
| `caution` | something a reviewer should look at | surface it, do not call it a failure |
| `evidence_gap` | a claim the artifact cannot settle alone | say what evidence would close it |

An `evidence_gap` is **not** a defect and must never be summarised as one. It
means the answer is not in the file — closing it requires importing a runtime,
lineage, or build-evidence document (`--coreml-compute-plan`,
`--edgetpu-compiler-evidence`, `--litert-qualcomm-evidence`,
`--tensorrt-parser-evidence`, `--executorch-build`, `--accelerator-profile`,
`--conversion-receipt`). The browser review summary reports the same group under
the field name `evidence_needed`.

## What this output does not establish

Do not write these claims into a summary, even when the analysis looks
suggestive. Static artifact analysis never establishes:

- which accelerator or delegate actually executed an operator at runtime
- latency, throughput, energy, or thermal behaviour
- task accuracy, clinical validity, or fitness for a specific device
- physical memory transfer

Delegate or provider *eligibility* is a static derivation about what a runtime
could accept. It is not observed assignment. `deepbom placement <artifact>
--profiles <ids|all>` compares eligibility across profiles deterministically and
still infers no executed assignment.

Target-bound numbers (`--target`, `--target-profile`) are bound to an assumed
CPU cost model; the envelope records `host_observed: false`. Report them as
"under target profile X", never as measured.

## CI gating

```bash
npx deepbom audit model.onnx --output-format sarif -o deepbom.sarif --gate defects
```

`--gate defects` exits 2 only on an `artifact_defect`. `--fail-on <severity>`
(`informational|low|medium|high`) is the severity-threshold alternative; the two
are separate policy sources. Nothing is gated by default — an evidence gap must
not silently become a release block. The requested document is still written
when a gate blocks, so CI can archive it.

Exit codes: `0` pass, `1` invalid invocation or analysis failure, `2` policy or
verification block, `3` incomplete verification binding. Treat `2` as a policy
result to report, not a crash.

## Reproducibility

```bash
npx deepbom audit model.onnx --output-format envelope --timestamp 2026-01-01T00:00:00Z
SOURCE_DATE_EPOCH=0 npx deepbom audit model.onnx --output-format envelope
npx deepbom audit model.onnx --output-format envelope -o evidence.json --no-clobber
```

`--timestamp` wins over `SOURCE_DATE_EPOCH`. With neither, the envelope records
`null` rather than depending on the clock. Output is written atomically;
`--no-clobber` refuses to replace existing bytes.

## MCP

For tool-call access instead of shell invocation, the same engine is exposed
over stdio JSON-RPC:

```bash
npx deepbom mcp
```

It provides `deepbom_capabilities`, `deepbom_audit`, `deepbom_diff`, and
`deepbom_explain_rule`, and runs locally on the same terms as the CLI. Audit
calls return the bounded human summary unless the task requires an explicit
`envelope`, `section`, or `pointer`. For large GGUF or SafeTensors artifacts,
prefer `scan: structure` for inventory questions and request `integrity` or
`full` only when payload evidence is necessary. Do not increase response or
download limits merely to obtain a complete dump; narrow the evidence first.

## Reference

- Machine contract: `npx deepbom capabilities --json`
- Plain-text overview for agents: <https://deepbom.org/llms.txt>
- CLI reference: <https://github.com/JunHwan-Kwon/deepbom/blob/main/docs/CLI_REFERENCE.md>
