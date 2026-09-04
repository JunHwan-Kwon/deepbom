# Public TFLite AI/ML-BOM example

This example is generated from Google's public `mobilenet_v2_1.0_224_quant.tflite` artifact through DEEPBOM's WASM analyzer.

- Source archive: `https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz`
- Archive SHA-256: `d6a04d780f76f656c902413be432eb349ec4a458240e3739119eb44977f77a79`
- TFLite member SHA-256: `f08d447cde49b4e0446428aa921aff0a14ea589fa9c5817b31f83128e9a43c1d`
- CycloneDX schema: official JSON schema v1.7, pinned by `../../schema-lock.json`
- Analyzer execution identity: recorded in every document as semantic version, source commit/state, and exact analyzer-bundle SHA-256. A dirty development regeneration is labeled `working-tree-dirty`; publishable snapshots must be regenerated from a clean tree.

`deepbom_cyclonedx_evidence.cdx.json` is the primary schema-valid AI/ML-BOM. The other files are hash-bound companion evidence for the Artifact Evidence IR, tensor contracts, observed formulation, runtime requirements, and provenance gaps. Public product output targets the stable CycloneDX 1.7 schema. Missing source declarations are reported as missing or not assessable; they are not fabricated to make the example appear complete.

In this snapshot, both external parameters have complete per-tensor affine contracts. The lifecycle-provenance ledger separately reports 14 missing and 4 partial fields. "Complete example" therefore means a schema-valid, artifact-bound, closed evidence package, not a claim that an opaque deployment artifact contains a complete training and release history.

Regenerate after building the WASM package:

```sh
node scripts/generate-cyclonedx-17-example.mjs
npm run validate:cyclonedx -- reference/cyclonedx/1.7/examples/mobilenet-v2-quant/deepbom_cyclonedx_evidence.cdx.json
```
