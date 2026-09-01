# Model-Free Bug Reporting

Do not upload a proprietary model to report a DEEPBOM defect. A useful report can be built from bounded identities and a synthetic reproducer.

## Include

1. The exact DEEPBOM version and installation channel: npm, PyPI, Cargo, or web.
2. Artifact format, byte length, and SHA-256. The original artifact is not required.
3. Command or browser workflow, target binding source, and accelerator evidence stage if applicable.
4. Structured error output (`--error-format json`) or the affected evidence JSON Pointer.
5. Analyzer, rulepack, policy, target-profile, selected-build, compiled-plan, and runtime-evidence digests that were bound to the run.
6. A redacted structure manifest or the smallest synthetic artifact that reproduces the behavior.

## Do Not Include

- Original model bytes, raw weights, patient data, prompts, labels, credentials, signing keys, or private download URLs.
- Unredacted GPU UUID, PCI address, account identifier, or internal filesystem path unless it is necessary and approved for disclosure.
- A claim that static eligibility proves runtime assignment.

## Minimal Template

```text
DEEPBOM version/channel:
Format:
Artifact SHA-256 / byte length:
Command or browser steps:
Target binding source / profile SHA-256:
Accelerator profile / evidence stage / binding SHA-256:
Expected result:
Observed status or error code:
Affected JSON Pointer:
Synthetic reproducer SHA-256 or generation script:
```

Security-sensitive reports should follow [SECURITY.md](../SECURITY.md) instead of a public issue.
