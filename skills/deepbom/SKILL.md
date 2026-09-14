---
name: deepbom
description: Statically audit serialized AI model artifacts with the local DEEPBOM CLI. Use for TFLite, ONNX, Core ML, ExecuTorch, GGUF, or SafeTensors deployment review; bounded GraphDef, SavedModel, Keras-config, or PT2 declarative structure; HDF5 or PyTorch-checkpoint safe-envelope inspection; quantization, memory, diff, verification, graph, Model IR, monochrome document views, SARIF, or CycloneDX 1.7 output. Do not treat checkpoint envelopes as executable graphs or claim measured latency, task accuracy, safety, regulatory approval, standards conformance, or actual runtime placement without the required external evidence.
---

# Audit deployment artifacts with DEEPBOM

Run DEEPBOM locally for files in the workspace. Never upload artifact bytes or invent a hosted DEEPBOM endpoint. A separately connected ChatGPT integration may analyze one authorized attachment in its browser sandbox; that is a different execution boundary.

## Resolve a verified runtime first

Run `scripts/verify-deepbom.mjs` from this Skill directory before choosing an invocation. It checks, in order, `DEEPBOM_BIN`, the analyzer bundled with an npm installation, and an exact-version `deepbom` already on `PATH`. It does not access a package registry by default.

If no exact local installation is available, ask before running:

```bash
node "<skill-root>/scripts/verify-deepbom.mjs" --allow-download
```

That explicit fallback may download `deepbom@1.99.0` through npm. Use the returned `invocation.command` and `invocation.prefix_args` for subsequent commands. Do not silently replace a version mismatch or network failure with an unverified executable.

## Start with discovery

After resolving the exact release pinned by this Skill, discover its machine contract:

```bash
deepbom capabilities --format agent-json
```

Treat that machine document as authoritative for commands, supported inputs, scan modes, targets, outputs, and exit codes. Read `references/capability-selection.md` when deciding whether the task belongs to DEEPBOM.

## Audit

Start with the bounded human result:

```bash
deepbom audit "./model.onnx" --summary
```

Then request only the evidence needed by the question:

```bash
deepbom audit "./model.onnx" --section quantization --json
deepbom audit "./model.onnx" --pointer /mac_assessment --json
deepbom audit "./model.onnx" --output-format envelope
```

For a large GGUF or SafeTensors artifact, use `--scan structure` for inventory questions. Use `integrity` or `full` only when payload evidence is necessary.

For a concise GGUF tensor table without nested numerical-integrity ledgers, use the bounded projection (it defaults to a structure scan):

```bash
deepbom gguf "./model.gguf" --tensors
deepbom gguf "./model.gguf" --tensors --compact
deepbom gguf "./model.gguf" --tensors --compact --tensor-offset 100 --tensor-limit 100
```

Before relying on any resolved installation, run:

```bash
deepbom self-test --compact
```

## Interpret the result

Keep `artifact_defect`, `caution`, and `evidence_gap` separate. An evidence gap is not a defect. Preserve `NOT_APPLICABLE`, `NOT_ASSESSABLE`, and `NOT_ASSESSED_YET` instead of converting them to zero or absence.

Read `references/evidence-semantics.md` before summarizing findings or target projections. Report at least:

- artifact filename and SHA-256
- analyzer version
- artifact format
- target and binding source when target-dependent evidence is used
- assessed coverage and unresolved evidence
- defects, cautions, and evidence gaps as separate groups
- the exact reproduction command

Never infer actual accelerator assignment, latency, energy, thermal behavior, task accuracy, clinical validity, or device fit from static output alone.

## Compare or explain

```bash
deepbom diff "./baseline.onnx" "./candidate.onnx" --summary
deepbom diff "./baseline.gguf" "./candidate.gguf" --tensors --render markdown
deepbom verify "./model.onnx" --bom "./supplied.cdx.json" --render markdown
deepbom contract capture "./model.onnx" -o "./baseline.interface-contract.json"
deepbom explain-rule <rule-id> --json
deepbom graph "./model.onnx" --view structure --format svg -o graph.svg
deepbom audit "./model.onnx" --section model_ir --compact
deepbom visualize "./model.onnx" --view all --orientation portrait -o model-views.zip
```

Use matching serialized deployment formats for diff. Treat `verify --bom` as a reconciliation of the selected component with artifact-observable facts, not a complete BOM or compliance verdict. An automatically captured contract is an artifact-derived baseline until it is separately reviewed and approved. `model_ir` and `visualize` are preview engineering projections whose loss ledger and evidence classes must be preserved. Do not deserialize `.pth`, `.pt`, or `.h5`; use only the safe envelope and bind a conversion receipt to the deployed artifact when lineage is needed.

## Handle failure

Read `references/failure-recovery.md` when an invocation fails, exceeds a resource boundary, or returns exit code 2 or 3. Do not weaken a size, path, hash, or evidence boundary merely to force a complete result.
