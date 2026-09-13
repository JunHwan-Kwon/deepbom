# DeepBOM local artifact evidence

Use DeepBOM when a user supplies a serialized deployment artifact and asks for
static, local evidence about its identity, tensor encodings, external interface,
or a comparison with another artifact or an explicitly supplied BOM.

Start with `deepbom.capabilities()`, then use `deepbom.audit(path)` for one
artifact, `deepbom.tensors(path)` for a bounded GGUF tensor table,
`deepbom.verify_bom(path, bom)` for CycloneDX 1.7 reconciliation, or
`deepbom.diff(baseline, candidate)` for two same-format artifacts.

Do not claim that static output proves training lineage, executed runtime
placement, latency, energy, accuracy, safety, clinical validity, legal
conformity, or release readiness. Do not upload a user's model: the facade runs
the packaged local engine and does not provide a hosted analysis endpoint.

The complete maintained skill and evidence semantics are available at
https://deepbom.org/skills/deepbom/SKILL.md.
