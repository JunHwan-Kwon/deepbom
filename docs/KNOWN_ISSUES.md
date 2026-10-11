# Known Issues

## DEEPBOM 2.3.0: tuple-valued native layer configuration

Reproduced on 2026-10-11 in the installed Linux x86-64 wheel while executing
the Python development example. `AdaptiveAvgPool2d((1, 1))` stores a tuple in
`output_size`; the native package's JSON reconstruction changes it to a list.
The state-definition identity preserves that distinction, so export fails with
`Loaded model definition or state differs from the selected manifest`.

This is an unresolved native reconstruction defect, not evidence of altered
model outputs or a static artifact calculation error. The identity check must
remain enabled. The documented toy example uses `AdaptiveAvgPool2d(1)`, whose
scalar configuration restores correctly; that is not a general solution for
tuple-valued layer settings. A fix needs explicit type-preserving constructor
configuration and round-trip checks across supported layer arguments.

The existing native guide's statement about preserving custom attribute types
does not establish the same property for built-in constructor configurations.
No runtime fix or new package release is included in this documentation update.

## DEEPBOM 1.95.0: partial ONNX MAC ledgers

DEEPBOM 1.95.0 could present the assessed MAC subtotal of an incomplete ONNX
cost ledger in a position that readers could interpret as the complete graph
total. The affected scope is limited to ONNX analyses where one or more
MAC-bearing operators could not be assessed from the serialized shape
contract.

DEEPBOM 1.96.0 corrects the contract as follows:

- `total_macs` is `null` unless every candidate compute operator is assessed.
- `mac_assessment.total_assessed_macs` remains available as the assessed
  subtotal.
- assessed and unassessed operator counts, the denominator, and the assessment
  status remain explicit.
- CLI, browser, graph, report, review, and CycloneDX projections use the same
  nullable-total meaning.

This issue does not change artifact identity, parsed operator identity, or the
MAC values of operators that were assessed. It changes whether an incomplete
subtotal may be read as a complete total.
