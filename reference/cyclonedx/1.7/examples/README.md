# Producer-bound export references

`current-reference.json` selects the current regression set and records the actual
analyzer metadata, source state, rulepack, WASM SHA-256 and every document digest.
Its `producer.json` and documents are stored together in a content-addressed
`mobilenet-v2-quant-<sha256>/` directory. The checker compares every complete
canonical document and validates the companion-reference hash closure.

Regenerate explicitly after building the engine and its metadata:

```sh
node scripts/write-build-metadata.mjs
node scripts/generate-cyclonedx-17-example.mjs
node scripts/check-export-contract-documents.mjs
```

Regeneration preserves existing sets and atomically rotates the current index.
A dirty-source set is marked `development_reference_not_released`. A passing
development regression is not a clean-release qualification or publication.
Formal release generation still requires the release pipeline's clean-source gate.

`mobilenet-v2-quant/` is the historical example. Its eight JSON files are preserved
byte-for-byte by `historical-reference.json`; their recorded producer is never
injected into the current engine's output. Use the current index for current
regression results, and the historical set only with its recorded context.

A commit that stores generated references changes Git context. The checker may
reuse a golden set across commit/dirty-state changes only when WASM, complete
source-content identity, rulepack, versions and all other producer metadata are
identical. Actual-run assertions use the current producer; the complete golden
comparison uses the recorded metadata as an explicit test input. The archived
files are never relabeled, and a changed engine or rulepack requires rotation.
