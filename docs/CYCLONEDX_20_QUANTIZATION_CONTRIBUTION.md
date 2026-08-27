# CycloneDX 2.0 Quantization Fixture Contribution

This package is bound to CycloneDX/specification PR #990 head
`49a945618811213e55686a23fa63b287940071c6`. It is contribution evidence, not
a claim that the draft has been accepted or released.

## Ready Fixtures

The candidate upstream fixtures in
`corpus/cyclonedx-20-quantization-fixtures/` are complete CycloneDX 2.0 BOM
documents. They are validated against the exact modular draft schema graph,
stored together as one deterministic gzip member:

- valid per-channel parameter quantization with a named input and axis;
- valid custom scheme/granularity with fractional nominal bit width;
- invalid zero bit width;
- invalid zero group size; and
- invalid unregistered scheme string.

These cases do not require a new cross-field policy. The three invalid cases
are already rejected by the pinned draft, while the two valid cases fill
coverage gaps in the current PR fixtures.

## Policy Probe

The evidence ledger retains twelve explicit schema probes. The pinned draft
accepts eight and rejects four. In particular, it currently accepts an empty
quantization object and several semantically questionable granularity/axis/
group-size combinations. Those accepted cases are observations, not proposed
invalid fixtures. They should become invalid fixtures only after the working
group decides whether this object is partial descriptive metadata or a complete
normalized quantization contract.

The draft currently rejects negative axes. ONNX permits framework-native
negative axes, but changing the schema requires a documented normalization
rule. Parameter-level objects have a shape against which normalization can be
defined; model-level weight summaries do not.

## Empirical Basis

The linked DEEPBOM population contains 50 predeclared TFLite artifacts and 114
external parameters: 62 complete affine contracts and 52 explicitly
unquantized parameters. This supports parameter identity binding and an
explicit not-quantized state. It is not a probability sample and does not
establish ecosystem prevalence or per-group behavior.

## Reproduction

```text
npm run build:cyclonedx20-quant-fixtures
npm run check:cyclonedx20-quant-fixtures
```

The build command requires the pinned PR branch at
`.local-validation/upstream/cyclonedx-specification-990`. The check command is
offline: it decompresses and validates against the committed pinned schema set,
verifies every fixture result and hash, replays the twelve probes, and
reconstructs the evidence-ledger SHA-256.

The PR-head bundled schema is retained in the schema archive for provenance but
is not used for validation: it does not yet contain the draft `modelProperties`
structure. The ledger records that distinction explicitly.
