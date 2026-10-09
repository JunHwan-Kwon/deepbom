# Three local integration examples

These examples implement **export → evidence → external decision**, using the
same analysis engine as the CLI. Read the [SDK contract](../../docs/SDK_CONTRACT.md)
for formats, errors, output semantics and version policy.

The current checkout also includes the [Weight and activation detail guide](../../docs/NUMERICAL_EVIDENCE_GUIDE.md),
with executable capture/import steps and Node/Python numerical API examples.

For development-time use, start with the
[fusion and inverted-residual experiment (Korean)](MLFLOW_ARCHITECTURE_SCENARIO.ko.md)
or the
[Weight IR → local redesign → MLflow scenario (Korean)](MLFLOW_DEVELOPMENT_SCENARIO.ko.md).
`mlflow_development.py` runs the checkout example, proposes two small internal
channel changes using the common workbench engines, and records the evidence,
plots and untrained structure scaffolds. It does not train or deploy a model.

For the proposed trainable-model interface, see
[file, factory and Python-object entry points (Korean)](TRAINABLE_ENTRY_POINTS.ko.md).
Its `deepbom.optimization` and `deepbom.training` commands and APIs are design examples, **not implemented
or available in the released SDK**. It specifies both PyTorch and TensorFlow/Keras
inputs, the call point before optimizer construction, and the common before/after
comparison output. Candidate comparison, constraint revision, regeneration and
selection form an independent loop before training; the same proposal includes
CLI/API examples for that loop. Model optimization produces a selected structure
and model state for training with a new user-owned optimizer. The separate
training observation path records evidence; it does not optimize the training
procedure or require a resumable checkpoint.

The [training lifecycle design](../../docs/evidence-ir/TRAINING_LIFECYCLE.md)
owns the proposed Training IR terminology, generation/reference diagrams and
binding rules. It distinguishes run observations from checkpoint-bound numerical
evidence and defines the shared observation semantics for Keras `fit`, TensorFlow
`GradientTape` and PyTorch loops. This extension is not implemented yet.
The same design provides optional presentation/storage adapters for DeepBoard
(the proposed local evidence viewer), MLflow, TensorBoard and W&B. Existing
MLflow artifact examples below do not imply that those full adapters or the
live training collector already exist.

For a separate model-replacement scenario with three comparable MLflow runs,
see [the MLflow release-review walkthrough (Korean)](MLFLOW_RELEASE_SCENARIO.ko.md).
`mlflow_release_review.py` consumes a completed bundle from the examples below,
logs the existing evidence and decisions, and verifies them by reading them back.

**Minimum SDK version: 2.1.0.** Install the Node SDK with
`npm install deepbom@2.1.0` and use the matching Python platform wheel from the
release assets. The source walkthrough below builds and verifies both packages
without publishing them. These additions are absent from earlier 2.0.0 packages.

## Install and run from this checkout

Requires Node >=20, Python >=3.10 for the example dependencies, and the checked-in
public WASM bundle. Python >=3.9 remains the SDK minimum. The first installation
downloads build/example dependencies; analysis runs locally. Commands below
use a POSIX shell; on Windows activate the venv with `.local-validation/sdk-venv/Scripts/Activate.ps1`.

```sh
python3 -m venv .local-validation/sdk-venv
. .local-validation/sdk-venv/bin/activate
python -m pip install build==1.3.0
npm ci
npm run build:channels
node scripts/run-sdk-integration-examples.mjs --python .local-validation/sdk-venv/bin/python
```

The runner installs the generated wheel into the specified environment and
the generated npm tarball into `.local-validation/sdk-consumer`. It installs
the pinned example requirements into that environment, then runs the examples
from the installed consumer. It does not publish packages, use GitHub Actions,
connect to a production registry/tracking server, or release a model.
Use a dedicated virtual environment as shown: the runner installs packages.

## 1. Inspect immediately after model export

`after_export.py` creates three deterministic toy ONNX artifacts using ONNX's
serializer. They are untrained examples, not clinical models. The candidate
changes one weight while preserving the interface; the incompatible candidate
changes the output width. The script computes a file digest independently,
then calls `inspect(..., expected_sha256=digest, scan="full")` and `audit`.
It checks that both results bind the same file and envelope, then saves the
original JSON documents without translating findings.

For your exporter, insert those calls immediately after saving the final
deployment artifact. A model that has no observed defect can still have
missing preprocessing, lineage or external validation evidence.

Expected files in `.local-validation/sdk-examples`:
`baseline.onnx`, `candidate.onnx`, `incompatible.onnx`, `inputs.json`, and a
`.summary.json` plus `.evidence.json` for each artifact.

## 2. Store and retrieve evidence in MLflow

`mlflow_evidence.py` audits the candidate against its recorded digest. It logs
the native envelope and review summary, the artifact/envelope digests and
engine version, the three separate finding counts and capability coverage into
a local MLflow run. It downloads both documents and checks their equality,
then checks the recorded identity and metrics. Unavailable capabilities are
separate from evidence-gap findings. These guards also run under Python `-O`.

The demo explicitly selects local SQLite tracking and a local artifact
directory rather than inheriting `MLFLOW_TRACKING_URI`. For a real integration,
select your approved tracking endpoint and credentials explicitly. Generated
evidence contains names/structure; it is not anonymous. Model bytes are not
logged by this example. No task accuracy or runtime metric is invented.

Expected files: `mlflow.db`, `mlflow-artifacts/`, and `mlflow-run.json` with the
run ID and tracking URI. To inspect the run interactively from the environment:

```sh
mlflow ui --backend-store-uri sqlite:///.local-validation/sdk-examples/mlflow.db --host 127.0.0.1
```

Open the local URL printed by MLflow and select `deepbom-sdk-example`.

## 3. Compare baseline and candidate before release

`release_review.mjs` uses the installed Node SDK. It checks each summary's
expected digest, captures and verifies the baseline interface against that
baseline, and obtains the engine's semantic diff for both candidates. It
checks that comparison identities agree with the inspected files.

An example policy outside the engine rejects an interface change or an
observed defect, and otherwise holds the candidate for external evaluation:

```text
candidate: hold_for_external_evaluation
incompatible: reject
```

The captured baseline contract also binds the original artifact hash. Applying
it directly to a changed candidate correctly rejects the changed hash;
candidate interface comparison instead uses the common engine's `graph_delta`.
The demo does not strip identities to force successful verification.

Each invocation creates a new `release-reviews/run-*` directory with copies of
these self-contained toy ONNX inputs. Expected files: `baseline.contract.json`, `baseline.verification.json`,
`candidate.comparison.json`, `incompatible.comparison.json`, and separate
`.decision.json` files with every reason and evidence-file digests. Only a
fully completed run receives `completed.json`; incomplete folders must not be
used as release evidence. Payload comparison coverage is explicit: missing
digests do not establish equal weights. These decisions belong to this example application,
not to a new DEEPBOM IR or universal release policy. Task quality, output
equivalence, operating environment and measured service requirements remain
separate approval inputs.

## Rerun individual examples

After the runner installs the packages, execute its copied scripts:

```sh
python .local-validation/sdk-consumer/after_export.py .local-validation/sdk-examples
python .local-validation/sdk-consumer/mlflow_evidence.py .local-validation/sdk-examples
node .local-validation/sdk-consumer/release_review.mjs .local-validation/sdk-examples
```

The first command replaces generated toy inputs; the third creates a fresh
review bundle. MLflow creates a new run each time. Do not point these demo commands
at a production release directory.

### Native optimization and training

[Executable API guide](../../docs/NATIVE_MODEL_TRAINING_GUIDE.md): independent PyTorch/TensorFlow candidates, frozen package export, manual GradientTape and PyTorch optimizer observations, DeepBoard and logger projections. The older trainable design examples remain historical proposals; use this guide for current signatures.
