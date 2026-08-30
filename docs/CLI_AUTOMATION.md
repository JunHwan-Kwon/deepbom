# CLI automation contract

DEEPBOM exposes one analysis implementation through npm, Python, Cargo, and the
repository CLI. Python and Cargo are verified launchers; they do not contain a
second parser or a second numerical rule set.

## Discovery

Use capability discovery instead of parsing help text:

```console
deepbom capabilities --compact
```

The `deepbom.cli_capabilities.v1` document declares commands, supported file and
package classes, output contracts, target-profile support, policy levels,
privacy behavior, and exit codes. Breaking machine-contract changes require a
new schema identifier. Additive fields may be introduced within the current
schema.

## Machine outputs

| Option | Contract | Intended consumer |
| --- | --- | --- |
| `--compact` | Complete format-specific analysis JSON | Deep format inspection |
| `--format envelope` | `deepbom.artifact_evidence_envelope.v1` | Cross-format automation |
| `--format cyclonedx` | CycloneDX 1.7 JSON | ML-BOM and supply-chain tooling |
| `--format sarif` | OASIS SARIF 2.1.0 Errata 01 | CI findings and security dashboards |

The envelope is the canonical cross-format contract. It binds artifact identity,
capability status, interfaces, graph totals, external files, normalized findings,
format extensions, provenance, and its own reproducible SHA-256. Format-specific
analysis remains intentionally richer and can evolve independently.

SARIF results point to the binary model artifact rather than inventing source
line locations. Each result carries a stable artifact/finding fingerprint,
evidence class, severity, interpretation, recommendation, and evidence pointers.
The emitted document is tested offline against the source-pinned OASIS schema.

## CI policy gate

`--fail-on` evaluates normalized findings rather than parser warning strings:

```console
deepbom audit model.onnx --format sarif --output deepbom.sarif \
  --fail-on high --policy-output deepbom-policy.json
```

Accepted thresholds are `informational`, `low`, `medium`, and `high`. A finding
at or above the threshold returns exit code `2`. The requested evidence document
is still completed, so CI can archive or upload it. The policy result binds the
finding counts and blocking IDs to the canonical envelope SHA-256.

No threshold is enabled by default. Organizations must choose a threshold that
matches their review policy; DEEPBOM does not silently turn research limitations
or predicted deployment risks into a release policy.

## Reproducibility and output safety

```console
SOURCE_DATE_EPOCH=0 deepbom audit model.onnx --format envelope --compact
deepbom audit model.onnx --format cyclonedx --timestamp 2026-08-30T00:00:00Z
deepbom audit model.onnx --format envelope --output evidence.json --no-clobber
```

`--timestamp` takes precedence over `SOURCE_DATE_EPOCH`. The canonical envelope
uses `null` when neither is supplied, avoiding an implicit clock dependency.
CycloneDX requires a timestamp and therefore uses the current time only when no
reproducible source is supplied.

File output is written to a same-directory temporary file, synchronized, and
then replaced. `--no-clobber` reserves the destination exclusively and refuses
to replace existing bytes. Use `--output -` for explicit stdout.

Stable artifact identity, safe package traversal, sidecar discovery, and range
reads require a regular file or directory. Symbolic stdin is deliberately not
accepted because buffering multi-gigabyte artifacts would weaken those
contracts.

## Errors and exit codes

`--error-format json` emits `deepbom.cli_error.v1` to stderr. Machine evidence
continues to use stdout, so diagnostics cannot corrupt JSON or SARIF output.

| Exit | Meaning |
| --- | --- |
| `0` | Command and requested gate passed |
| `1` | Invocation, input, analysis, or output failure |
| `2` | Finding policy or interface verification blocked |
| `3` | Interface verification could not establish a complete binding |

`verify`, `diff`, and `explore` retain their command-specific evidence schemas.
They reject unrelated output formats and policy options instead of silently
ignoring them.

## Evidence boundary

CLI execution is local and does not upload artifact bytes or analysis results.
Static eligibility is not actual provider/delegate assignment. SARIF severity is
an engineering triage level, not a probability of failure, clinical risk score,
or release-readiness decision. Runtime assignment, task accuracy, clinical
validity, and physical transfer or latency require separately identified
evidence.
