# Contributing to DEEPBOM

DEEPBOM accepts focused changes to the public analyzer, evidence contracts,
documentation, fixtures, and validation tooling. Do not submit proprietary
models, patient data, credentials, raw production traces, or files whose
redistribution rights are unclear.

## Before opening a change

1. Reproduce the behavior with a public or locally generated minimal artifact.
2. Record the artifact format, exact SHA-256, command, DEEPBOM version, target
   or accelerator binding, and expected versus observed evidence state.
3. Preserve `0`, `NOT_ASSESSABLE`, and `NOT_APPLICABLE` as distinct values.
4. Do not convert source eligibility into observed runtime assignment or a
   static lower bound into a fit, latency, quality, or clinical claim.

## Validation

Use the smallest relevant check first. Before proposing a release-facing
change, run:

```console
npm run check:version
npm run check:expected-output
npm run check:cli
npm run check:evidence-workbench
npm run check:public-source
```

Large corpus and runtime checks are intentionally separate. A pull request
must state which checks were run and which were not available.

## Reporting without model bytes

Prefer the canonical evidence envelope, analyzer/rulepack identities, the
failing JSON Pointer, and a minimized synthetic fixture. If a defect cannot be
reproduced without a restricted artifact, open a private security report or
contact the maintainer before transmitting any bytes.
The complete model-free template is in
[`docs/MODEL_FREE_BUG_REPORTING.md`](docs/MODEL_FREE_BUG_REPORTING.md).

## Public and private boundaries

The public repository is generated from an exact reviewed allowlist. Private
rulepack generators, hosted infrastructure, restricted corpora, and unreleased
research modules are not accepted into the public tree. See
[`docs/PUBLIC_PRIVATE_BOUNDARY.md`](docs/PUBLIC_PRIVATE_BOUNDARY.md).
