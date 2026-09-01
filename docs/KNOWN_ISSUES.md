# Known Issues

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
