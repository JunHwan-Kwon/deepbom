# Public TFLite Corpus

## Public Cross-Format Measured Corpus

`public-multiformat-corpus.v1.json` expands the real-file validation population
to 113 immutable path records representing 113 unique primary artifacts:

- 48 ONNX graph files across task and precision strata. Fourteen records also
  bind a matching external-data sidecar no larger
  than 100,000,000 bytes; larger sidecars remain explicitly unbound.
- 20 real GGUF files spanning serialized architecture and mixed GGML storage
  encodings.
- 18 real SafeTensors checkpoints, one for every on-device LLM architecture
  family currently decoded by the adapter.
- 27 public Apple Core ML assets: the complete set linked from the Apple
  Developer model catalog snapshot on 2026-08-18, spanning classification,
  detection, segmentation, depth, question answering, legacy neural networks,
  ML Program packages, headless variants, palettization, and an updatable
  pipeline.

The manifest binds 145 source files and 2,926,971,262 declared download bytes.
Hugging Face files are repository-revision and SHA-256 bound. Apple catalog
assets have no exposed source revision, so the manifest records that limitation
and binds the downloaded content by URL, size, and SHA-256. Original bytes stay
in the user-local cache and are not committed or deployed.

ONNX, GGUF, and SafeTensors form a purposeful stratified validation population,
not a random sample. Core ML instead enumerates the complete public Apple model
catalog snapshot declared above. SafeTensors tiny/random checkpoints establish
parser and architecture-contract behavior, not task quality. Missing license
metadata is not treated as a reuse grant. A graph whose large ONNX sidecar is
unbound contributes graph/interface evidence but never decoded-weight evidence.

```powershell
npm run build:public-multiformat-corpus
npm run corpus:multiformat:download -- --max-total-gib 4
npm run corpus:multiformat:sweep -- --offline
npm run build:cyclonedx-generalization-evidence
npm run check:public-multiformat-corpus
```

## ONNX extension contracts

`onnx-extension-contract-corpus.v1.json` is a small, hash-bound conformance
population for residual ONNX contracts that were not positively represented in
the public application corpus. It includes source-derived per-axis Q/DQ with
runtime-supplied affine values, a complete static per-axis Q/DQ contract, and an
exact upstream `ai.onnx.contrib` custom-op binary. These records establish parser
and fail-closed behavior only; they are not ecosystem prevalence or runtime
provider evidence.

```shell
npm run build:onnx-extension-contract-corpus
npm run check:onnx-extension-contract-corpus
```

The sweep analyzes every path record twice and stores full local analyses plus
compact receipts. `cyclonedx-generalization-evidence.v1.json` retains the
aggregate ledger and a digest-bound index; the complete per-artifact records
are stored without information loss in
`cyclonedx-generalization-evidence.records.v1.json.gz`. Together they retain one
machine-readable observation per unique artifact and derive format-level
counts only from those rows. It separates observed bytes, deterministic
derivations, external provenance, unavailable values, runtime-required claims,
and out-of-scope task claims. In particular, it does not collapse SafeTensors
dtype storage, GGUF block encoding, tensor-scoped affine mappings, and observed
runtime placement into one generic `quantization` assertion.

The generated [per-artifact review](../docs/PUBLIC_MULTIFORMAT_CORPUS.md)
summarizes every hash-bound result; the index and compressed records together
retain the complete machine-readable observations for schema design and fixture
generation.

`public-tflite-corpus.v1.json` pins 20 public Google MediaPipe TFLite
artifacts by GCS object generation, byte size, and SHA-256. Model bytes are
downloaded into a user-local cache and are not committed or deployed.

Run the deterministic sweep twice in isolated processes per artifact:

```powershell
npm run sweep:public-model-corpus -- --output-dir reports/public-model-corpus-v1
```

Re-run without network access after the cache has been populated:

```powershell
npm run sweep:public-model-corpus -- --offline --output-dir reports/public-model-corpus-v1-offline
```

The command fails if an artifact identity changes, either isolated analysis
digest differs, a quant-research denominator does not conserve all 15 labs,
the analysis contains a non-finite number, or a report references a phantom
operator.

The manifest records technical provenance, not a license grant. Use of each
downloaded model remains subject to its upstream terms.

## Quantizer Policy-Boundary Corpus

`quant_policy/manifest.v1.json` composes four predeclared public subcohorts
into a 50-artifact target: 20 MediaPipe models, 11 MCUNet models, four
Google-hosted legacy quantized models, and 15 revision- and LFS-SHA-pinned
static W8A8 channelwise LiteRT models. Model bytes remain in the existing
user-local caches.

Run every target twice in isolated analyzer processes:

```powershell
npm run corpus:quant-policy:sweep
```

After the caches are populated, reproduce the sweep without network access:

```powershell
npm run corpus:quant-policy:sweep -- --offline
```

Independently recompute every interface denominator and cross-table from the
sweep rows, verify the current analyzer identity and per-artifact ledgers, and
write a compact proposal-facing review:

```powershell
npm run verify:interface-boundary-corpus -- `
  --sweep .local-validation/interface-boundary-corpus-2026-08-05/quant-policy-boundary-sweep.json
```

The non-redistributable DeepLab case-study anchor is optional and never
counted in the public denominator:

```powershell
$env:DEEPBOM_CASE_STUDY_TFLITE = "C:\path\to\pinned-deeplab.tflite"
npm run corpus:quant-policy:sweep -- --offline
```

The sweep records embedded `CONVERSION_METADATA` optimization modes, keeps an
absent QAT/PTQ declaration as `unknown`, derives kernel granularity from the
artifact, and measures stored INT32 bias codes against
`floor(INT32_MAX / 2)`. That comparison is a source-backed policy reference,
not evidence that a converter pass executed or skipped. Strict exceedance,
the 129-code float32 guard-adjacent class, and material exceedance are reported
separately. The sweep also reports exact full-code-domain INT32 envelope
exceedance, which is a storage-domain witness rather than proof of graph
reachability or executed-kernel overflow.

Each row also binds every model input/output to its file SHA-256, parameter
identity, dtype, shape, affine scales, zero-points, quantized dimension,
cardinality status, and scalar real code domain. The corpus summary groups
identical `direction + dtype + shape` signatures and records when multiple
serialized affine contracts occur under the same signature.

Interface coverage uses mutually exclusive artifact denominators:
`fully_affine_quantized`, `mixed_quantized_unquantized`,
`fully_unquantized`, `invalid_or_incomplete`, and `not_declared`. A FLOAT32
input or output is an explicit unquantized boundary fact, not missing affine
metadata. These boundary contracts do not establish channel order, source
value normalization, mean/standard-deviation transforms, resize
interpolation, application tensor layout, labels, or task accuracy.

Optionally cross-check all cached public artifacts with the official LiteRT
Interpreter metadata API. This calls only `get_input_details()` and
`get_output_details()`; it does not allocate tensors or run inference:

```powershell
python scripts/verify-litert-interface-corpus.py `
  --sweep reports/quant-policy-boundary-public-50-2026-07-30/quant-policy-boundary-sweep.json `
  --output reports/quant-policy-boundary-public-50-2026-07-30/litert-interface-crosscheck.json
```

## Hugging Face Community Corpus

`huggingface-community-corpus.v1.json.gz` inventories every repository visible
through the public API for `litert-community` and `onnx-community`. Each entry
pins the observed repository commit and every file's size plus LFS SHA-256 or
Git blob identity. The readable `.summary.json` binds the compressed snapshot
by SHA-256.

Refresh all metadata:

```powershell
npm run corpus:hf:sync
```

Export flat repository and artifact catalogs for spreadsheet or research use:

```powershell
npm run corpus:hf:list
npm run corpus:hf:list -- --tier mid --format onnx
```

Inspect a bounded download plan without transferring model bytes:

```powershell
npm run corpus:hf:plan -- --tier micro --scope testable
npm run corpus:hf:plan -- --tier mid --format onnx --scope testable
npm run corpus:hf:plan -- --tier large --scope model
```

Download and verify a bounded selection into the user-local DeepBOM cache:

```powershell
npm run corpus:hf:download -- --tier micro --scope testable --download --max-total-gib 8
npm run corpus:hf:download -- --repo onnx-community/resnet-50 --scope testable --download --max-total-gib 2
```

An intentional full mirror requires both `--download` and `--yes-unbounded`.
This is deliberately explicit because the organizations contain LLM/VLM
artifacts and repository-wide transfer can require very large storage:

```powershell
npm run corpus:hf:download -- --scope repository --download --yes-unbounded
```

Run deterministic isolated-process analysis over files already present in the
verified cache:

```powershell
npm run corpus:hf:sweep -- --tier micro
npm run corpus:hf:sweep -- --tier mid --format tflite --format onnx
```

Scale tiers are planning labels:

- `micro`: explicit TFLite Micro/TinyML/MCU evidence, or a size-bounded TFLite
  MCU candidate. Candidate classification is not proof that arena, op resolver,
  or latency fits a particular microcontroller.
- `mid`: standalone TFLite/ONNX and similar per-model runtime artifacts without
  large-model evidence.
- `large`: LLM/VLM/generative runtime evidence, a large-model container,
  sharding, or at least 1 GiB of model payload.

The metadata snapshot is tracked. Downloaded files and sweep outputs stay in
the user-local cache and `reports/`; they are never copied into the web build.

## Residual Coverage Priorities

### SafeTensors architecture anchors

`safetensors-architecture-corpus.v1.json` pins the immutable repository
revision, `config.json`, and `model.safetensors` bytes for one dense Mistral,
one Mixtral sparse-MoE, and one Mamba recurrent-SSM checkpoint. All three are
small randomly initialized public checkpoints. They validate parser,
canonical tensor-layout, state-cardinality, and architecture compute formulas;
they do not establish model quality, runtime performance, ecosystem prevalence,
or reuse permission. The source model cards currently declare no license, so
downloaded bytes remain only in the user-local DeepBOM cache.

Download, hash-verify, analyze, and compare all three baselines:

```powershell
npm run corpus:safetensors:architecture
```

Reproduce from verified cached bytes without network access:

```powershell
npm run corpus:safetensors:architecture -- --offline
```

### SafeTensors AWQ and GPTQ contracts

`safetensors-quantization-contract-corpus.v1.json` binds two public Apache-2.0
repositories at immutable revisions. The builder range-reads each complete
SafeTensors header, verifies full-file size and LFS SHA-256 from revision-bound
repository metadata, binds `config.json` plus the producer quantization
sidecar, and checks every packed module against pinned AutoAWQ or AutoGPTQ
layout source. The committed corpus contains source configs and compact
measurement ledgers, not model payloads. Detailed module receipts remain in
`.local-validation/`.

```powershell
npm run build:safetensors-quantization-corpus
npm run check:safetensors-quantization-corpus
```

The current anchors establish 322/322 valid 4-bit, group-128 module layouts
and exact packed-code conservation. They do not scan scale or zero-point
payload values and do not establish sharded, HQQ, bitsandbytes, or
compressed-tensors behavior.

### GGUF architecture and storage anchors

`gguf-architecture-encoding-corpus.v1.json` pins eight real GGUF files. Five
TinyMQA files probe quantizer-label strata, while GPT-2, OLMo, and StableLM add
serialized architecture strata. Baselines are built from each tensor header,
not from the filename: the current set covers four serialized architectures
and ten observed GGML storage types with no unsupported tensor encoding. This
also records the important fact that a file-level quantization label does not
imply one uniform tensor type.

```powershell
npm run corpus:gguf:architecture-encoding
npm run corpus:gguf:architecture-encoding -- --offline
```

### Core ML MLProgram contract fixtures

`coreml-mlprogram-contract-corpus.v1.json` binds five deterministic
`.mlpackage` fixtures generated from the same commit- and content-pinned Apple
Core ML protobuf contracts used by the analyzer. They cover a static MLProgram
with an externally bound blob-v2 weight, enumerated multi-array shapes, and
bounded shape ranges with unknown MIL spatial dimensions, plus iOS 18
blockwise affine compression and vector LUT palettization contracts.

```powershell
npm run corpus:coreml:mlprogram-contracts
```

### Core ML legacy per-channel quantization contract

`coreml-legacy-quantization-corpus.v1.json` binds one deterministic legacy
NeuralNetwork `.mlmodel` fixture to the pinned `NeuralNetwork.proto` and
`quantization_utils.py` source commit and content digests. The fixture contains
one INT4 convolution weight payload with two output-channel scale/bias pairs.
Its checker independently regenerates the protobuf bytes, verifies the artifact
and payload SHA-256 values, decodes all 18 packed codes, and rejects a one-scale
mutation against the two-channel axis. This is positive format-conformance
evidence, not a public-model sample or an ecosystem-frequency claim.

```bash
npm run build:coreml-legacy-quantization-corpus
npm run check:coreml-legacy-quantization-corpus
```

The fixtures establish package resolution, exact byte identity, source-schema
conformance, flexible-interface decoding, and fail-closed dynamic cost
handling. They are not public model samples and do not establish ecosystem
prevalence, task quality, runtime device placement, or latency.

`residual-coverage-priorities.v1.json` separates measured analyzer residuals
from validation-population gaps. Its ONNX population now covers all 48
unique-byte ONNX artifacts in the public multiformat corpus, drawn from 40
revision-pinned repositories. `public-onnx-residual-sweep.v1.json.gz` retains
the compact per-artifact residual rows and binds every artifact and analysis
SHA-256; this is still a purposeful validation population, not an ecosystem
prevalence sample. The GGUF,
SafeTensors, and Core ML populations contain eight architecture/encoding
anchors, three family anchors, and five generated MLProgram contract anchors,
respectively. Those fixture populations remain useful contract regressions but
must not be substituted for the newer real-file population or presented as an
ecosystem-prevalence sample.

Regenerate after repeat corpus sweeps have emitted
`deepbom.corpus_coverage_residuals.v1.3` rows. The builder rejects older rows
instead of silently publishing zero-valued symbolic-contract fields:

```powershell
npm run corpus:public-onnx-residuals -- `
  --sweep .local-validation/residual-coverage/public-multiformat-onnx-v1.35/public-multiformat-corpus-sweep.json `
  --output corpus/public-onnx-residual-sweep.v1.json.gz
npm run corpus:residual-coverage
```

The ranking puts observed residuals before corpus-breadth gaps, then sorts by
affected artifacts, affected entities, deterministic closability, and
deployment impact. It never converts an unbound dynamic dimension into a
numeric MAC, memory, or latency value. Shape-contract ratios use node outputs,
not nodes, as their denominator because an ONNX node may have multiple outputs.

## Curated Micro Corpus

`curated-micro-corpus.v1.json` resolves actual model artifacts discovered
through the commit-pinned `umitkacar/awesome-tinyml` index. The discovery list
itself contains no model binaries, so linked projects are independently
validated rather than inheriting a TinyML label or license.

The first resolved source is the official MIT HAN Lab MCUNet model index:
11 downloadable INT8 TFLite artifacts across ImageNet classification, Visual
Wake Words, and person detection. Every artifact is pinned by byte size and
SHA-256. The source commit, model-index digest, README digest, and repository
license evidence are recorded separately.

Download missing bytes and run every artifact twice in isolated analyzer
processes:

```powershell
npm run sweep:curated-micro-corpus
```

Re-run entirely from the verified user-local cache:

```powershell
npm run sweep:curated-micro-corpus -- --offline
```

Select one or more models:

```powershell
npm run sweep:curated-micro-corpus -- --artifact mcunet-in2 --artifact mcunet-vww0
```

The sweep verifies artifact identity, full-integer classification, finite
analysis output, phantom-op absence, quant-lab denominator conservation,
isolated-process determinism, and agreement within 1% of the rounded upstream
MAC figure. Current target-profile timing is explicitly not interpreted as an
STM32 result. Upstream SRAM/flash numbers remain published evidence until a
pinned MCU runtime and device reproduce them.

EtinyNet remains in the discovery ledger but is not downloaded: its pinned
repository exposes MXNet parameters rather than TFLite/ONNX and has no
repository license file. This distinction keeps "not analyzable" separate
from "analyzed with no finding."
# Google Legacy Converter Cohort

`google_legacy/` pins eight official Google-hosted archives and their exact
TFLite members. Four are measured quantized artifacts and four proposed MnasNet
quantized entries are retained as measured FLOAT32 controls. The corpus includes
a safe downloader, isolated repeat sweep, per-lab denominators, and converter
generation methodology. Model bytes remain in the user cache.
