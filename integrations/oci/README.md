# DeepBOM OCI evidence attachment

`build-attestation-predicate.mjs` creates an unsigned in-toto statement whose
subject is the audited model artifact and whose predicate is bound to the
validated DeepBOM evidence envelope. The script never signs, uploads, or pushes
anything. Signing and registry attachment belong in an explicitly authorized
release environment.

```bash
deepbom audit model.onnx --format envelope --output evidence.json
node integrations/oci/build-attestation-predicate.mjs --evidence evidence.json --output predicate.intoto.jsonl
```

The predicate states the evidence boundary and excludes runtime, quality,
safety, and regulatory claims. Registry-specific commands are intentionally not
embedded in the core adapter.
