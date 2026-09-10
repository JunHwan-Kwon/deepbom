---
name: deepbom
description: Statically audit serialized AI deployment artifacts with the local DEEPBOM CLI. Use for TFLite, ONNX, Core ML, ExecuTorch, GGUF, or SafeTensors deployment review; MAC and symbolic-shape coverage; quantization contracts; memory feasibility; accelerator eligibility; artifact diff; graph export; SARIF; or CycloneDX 1.7 ML-BOM generation. Do not use for training checkpoints, measured latency, task accuracy, or claims about actual runtime placement without imported evidence.
---

# Audit deployment artifacts with DEEPBOM

Run DEEPBOM locally. Never upload artifact bytes or invent a hosted DEEPBOM endpoint.

## Start with discovery

Use the release pinned by this skill:

```bash
npx -y deepbom@1.96.14 capabilities --format agent-json
```

Treat that machine document as authoritative for commands, supported inputs, scan modes, targets, outputs, and exit codes. Read `references/capability-selection.md` when deciding whether the task belongs to DEEPBOM.

## Audit

Start with the bounded human result:

```bash
npx -y deepbom@1.96.14 audit "./model.onnx" --summary
```

Then request only the evidence needed by the question:

```bash
npx -y deepbom@1.96.14 audit "./model.onnx" --section quantization --json
npx -y deepbom@1.96.14 audit "./model.onnx" --pointer /mac_assessment --json
npx -y deepbom@1.96.14 audit "./model.onnx" --output-format envelope
```

For a large GGUF or SafeTensors artifact, use `--scan structure` for inventory questions. Use `integrity` or `full` only when payload evidence is necessary.

For a concise GGUF tensor table without nested numerical-integrity ledgers, use the bounded projection (it defaults to a structure scan):

```bash
npx -y deepbom@1.96.14 gguf "./model.gguf" --tensors
npx -y deepbom@1.96.14 gguf "./model.gguf" --tensors --compact
```

Before relying on a newly fetched package, run:

```bash
npx -y deepbom@1.96.14 self-test --compact
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
npx -y deepbom@1.96.14 diff "./baseline.onnx" "./candidate.onnx" --summary
npx -y deepbom@1.96.14 explain-rule <rule-id> --json
npx -y deepbom@1.96.14 graph "./model.onnx" --view structure --format svg -o graph.svg
```

Use matching serialized deployment formats for diff. Do not deserialize `.pth`, `.pt`, or `.h5`; bind a conversion receipt to the deployed artifact instead.

## Handle failure

Read `references/failure-recovery.md` when an invocation fails, exceeds a resource boundary, or returns exit code 2 or 3. Do not weaken a size, path, hash, or evidence boundary merely to force a complete result.
