# Product direction candidates from external review

This document preserves product and API proposals that are deliberately kept
outside the correctness triage in `EXTERNAL_REVIEW_TRIAGE.md`. None of these
items is a release blocker until it is accepted into a versioned product plan.

## Library APIs

- Consider a thin Python subprocess wrapper with `audit()`, typed results, and
  gate helpers. Do not duplicate the JavaScript/WASM engine.
- Consider an npm programmatic API with `audit()`, `diff()`, and
  `explainRule()`, backed by the same canonical envelope as the CLI.
- Any public API must define timeout, cancellation, output-size, and schema
  compatibility contracts before it is declared stable.

## Product audiences

The external review identified three distinct uses: interactive engineering
inspection, CI gating, and reusable evidence packaging. They should share one
evidence engine but may need separate summaries and gate policies.

## Format focus

Do not infer equal maturity from an accepted extension. Publish capability and
evidence-depth matrices, and concentrate correctness fixtures on TFLite, ONNX,
and GGUF before broadening claims for other formats.

## Public and protected implementation

Protected analyzers require public mathematical definitions, source identities,
and hash-bound golden fixtures if third parties are expected to reproduce or
audit their outputs. A future licensing or hosting decision is separate from
the correctness of current releases.

## Adoption and sustainability

- Provide a short installed-package success path and defect-injection tutorial.
- Evaluate integrations only after the core result and schema contracts are
  stable.
- Treat funding, foundation participation, and commercial packaging as product
  decisions rather than evidence about analyzer correctness.
