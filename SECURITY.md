# Security Policy

## Supported releases

Security fixes target the latest published DEEPBOM release. Older releases may
be documented as affected without receiving a backport.

## Reporting a vulnerability

Use GitHub private vulnerability reporting for
`JunHwan-Kwon/deepbom` when available. Otherwise contact `kjh0442@yuhs.ac`
with the subject `DEEPBOM security report`.

Include the DEEPBOM version, platform, command or browser path, artifact format,
failure class, and a minimal reproduction. Do not send proprietary model bytes,
patient data, secrets, or production runtime traces by email. Begin with hashes,
sizes, relevant JSON pointers, and a synthetic reproducer; a protected transfer
method can be agreed separately if the original artifact is essential.

Relevant classes include parser crashes or resource exhaustion, path traversal,
unsafe package extraction, artifact identity bypass, signature or digest
verification bypass, unintended network transfer of model data, and disclosure
across the documented public/private source boundary.

## System and scope

DEEPBOM statically parses untrusted deployment artifacts and therefore treats
malformed input as hostile. A clean rejection, timeout, or configured resource
limit is not itself a vulnerability. Executable pickle formats are not
deserialized.

This policy covers the browser application, CLI and standalone engine, Python,
npm and Cargo launchers, local MCP server, artifact/cache/package resolvers,
export and verification code, build/release workflows, and files included in an
official DeepBOM distribution. The hosted browser application performs analysis
locally; the CLI accesses the network only for an explicitly supplied immutable
remote source.

## Threat model and trust boundaries

Artifact bytes, sidecars, manifests, BOMs, model repositories, archives, file
names, paths, and MCP arguments are attacker-controlled. Registry credentials,
release signing identities, the build workflow, pinned schemas/rulepacks, and
independently supplied expected digests are security-sensitive assets. An
analysis result is engineering evidence, not trusted executable model content.

## Security invariants

- Parsing must be bounded, deterministic, and non-executing. Python pickle and
  equivalent executable checkpoint formats must never be imported or evaluated.
- Local artifact analysis must not upload model bytes or emit telemetry.
- Remote inputs must have an immutable revision, object generation, or SHA-256;
  cache reuse must revalidate identity.
- Archive extraction, external-data resolution, package closure, and MCP local
  paths must remain inside their explicitly allowed roots.
- An independently supplied digest mismatch must fail closed and remain distinct
  from a finding-policy decision.
- Verification must not bind a BOM component by a fuzzy name. Ambiguous hashes
  or `bom-ref` values remain unresolved.
- Machine output and exit status must not hide parser, renderer, analysis,
  verification, or output failures.
- Public distributions must exclude private evidence, credentials, unpublished
  research material, and protected implementation sources.

## Reportable findings and severity context

Report unintended code execution, sandbox or path-boundary escape, arbitrary
file overwrite, credential exposure, release or artifact identity bypass,
unintended model-byte transfer, cross-request disclosure, decompression/resource
exhaustion that bypasses configured bounds, or a fail-open result that can cause
automation to accept a failed or contradicted audit. Severity depends on actual
reachability through a supported public surface and the affected identity,
confidentiality, availability, or release-gating property.

## Out of scope and limitations

Static compatibility predictions do not establish runtime safety, model quality,
clinical validity, legal conformity, or release readiness. Incorrect task
predictions, model bias, adversarial robustness, training-data provenance, and
runtime performance are out of scope unless DeepBOM falsely reports that it
measured them or a defect breaks a documented security invariant above.

Third-party vulnerabilities are reportable when they are reachable through a
shipped DeepBOM path. Development-only oracle dependencies are assessed against
their use in release gates and are not automatically treated as runtime
exposure. No vulnerability class is suppressed solely because a dependency is
development-only.
