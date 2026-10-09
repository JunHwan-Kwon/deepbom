# Native model optimization and training evidence

Status: implemented experimental local extension, qualified on CPU with PyTorch
2.8.0, TensorFlow 2.20.0 and Keras 3.11.3. This is an additive Python entry point;
the existing artifact CLI, SDK and public remote MCP contracts remain unchanged.
The extension is packaged starting with 2.2.0. Native platform availability is
determined by published wheel/release assets, not by this CPU qualification.

The [implementation validation record](evidence-ir/NATIVE_IMPLEMENTATION_VALIDATION_2026-10-07.md)
lists the executed source, installed-package, schema, browser and compatibility
checks, their results and the unqualified environments.
The [follow-up compatibility audit](evidence-ir/COMPATIBILITY_BUG_AUDIT_2026-10-08.md)
records additional reproduced defects, corrections and broader regression checks.

## Responsibility and compatibility

DEEPBOM proposes and materializes model candidates. The application owns training,
losses, optimizers, data and deployment decisions. A returned `Candidate.model` is
an independent mutable model; its selected package contains the earlier frozen
state, even if that returned model is subsequently trained. Use a new optimizer.
There is no optimizer, RNG, sampler or training-resume restoration.

| Document | Source and responsibility |
| --- | --- |
| Model State Snapshot v1 | Definition in the adapter's stated scope, named state, dtype, shape, exact byte and element counts, payload digests and aliases |
| Model IR v2 | Native module containment and state references; not an observed execution schedule |
| Weight IR v2 | Values from that exact state; common scalar decoder, statistics, distribution, matrix and sparsity engine |
| Activation IR v2 | Explicit invocation and input context, declared before-forward state and captured subjects |
| Training IR v1 | Append-only event chunks, run/segment identity, coordinates, lifecycle and references to already fixed evidence |
| Training Result Manifest v1 | Selection of a recorded state and its evidence, without asserting quality or completion |

There is no Trained IR. Artifact-based Model/Weight/Activation v1 documents and
Evidence IR family v1 remain unchanged. Native documents use explicit new schema
identities; they are not silently accepted by artifact-only consumers. Family v2
is an additive union. Snapshot and chunk documents support the family and are not
new analytical layers. See [native catalog](evidence-ir/native-catalog.json) and
[schemas](schemas/deepbom-native-evidence-v1.schema.json).

Hash normalization uses the common JavaScript canonical JSON owner. JSON numeric
statistics use binary64; exact counts and large integer extrema use decimal
strings. Python preserves binary64 serialization when validating returned hashes.
A content digest detects changes, not truthfulness or source authenticity.

## Installation and local entry points

For the qualified Linux x86-64 release wheel:

```sh
python -m pip install 'deepbom[torch,tensorflow] @ https://github.com/JunHwan-Kwon/deepbom/releases/download/channels-v2.2.0/deepbom-2.2.0-py3-none-linux_x86_64.whl'
# Install only the framework extra you use. GPU/OS wheel choice is your environment's responsibility.
python -m deepbom.optimization --help
python -m deepbom.training --help
```

Use a matching published asset for another platform, or build its wheel from
source; the Linux wheel is not portable to Windows or macOS. This installation
does not assume that the release is available on PyPI.
Optional imports leave the dependency-free static API usable without frameworks.
The framework-qualified range is narrower than the static package's Python range;
the verification environment uses Python 3.12. The package's verified engine is
required; the native path never downloads an unverified executable.

```sh
# Explicitly importing this trusted factory executes your local Python code.
python -m deepbom.optimization suggest --factory my_model:build_model --objective structure
python -m deepbom.optimization optimize --factory my_model:build_model \
  --objective structure --inputs inputs.json --output candidate-a
# inputs.json: {"args":[{"shape":[1,3,32,32],"dtype":"float32"}],"kwargs":{}}

# A PyTorch state dictionary needs its model definition; keys must match exactly.
python -m deepbom.optimization optimize --factory my_model:build_model \
  --state-file weights.pt --rules rules.json --output candidate-b
# .keras uses compile=False, safe_mode=True; no saved optimizer is restored.
python -m deepbom.optimization optimize --keras model.keras --output candidate-c
python -m deepbom.optimization compare candidate-a candidate-b
```

The Python `deepbom optimize ...` launcher also routes to the local native entry
point. The npm-only static CLI does not import PyTorch or TensorFlow.

## Candidate selection, refinement and export

```python
from deepbom.optimization import optimize, refine, load_model

candidate = optimize(model, objective="structure", example_args=(example_input,),
                     advanced=True, output_dir="candidate-a")
print(candidate.comparison["totals"])
print(candidate.validation)
# Change the proposal without beginning training. Each refinement starts from
# the frozen baseline; it does not accidentally compound the previous rewrite.
alternative = refine(candidate,
    rules=[{"kind": "inverted", "subject": "features.0", "expansion": 2}],
    example_args=(example_input,), output_dir="candidate-b")
selected = load_model("candidate-b")
# optimizer = torch.optim.Adam(selected.parameters(), lr=...)
```

`subject` must be an exact native module/layer identifier. `rules=[]` explicitly
requests an unchanged candidate. Omitted rules invoke bounded discovery:

- `objective="structure"`: eligible dense spatial Conv2D replacements with an
  expand → depthwise → project block. Affected weights are newly initialized;
  unaffected state is preserved. This changes the model and requires training
  and task evaluation. It is not a guaranteed improvement.
- `objective="cpu_inference"`: eligible Conv–BN folding. PyTorch requires eval
  modules with running statistics and adjacent children reached through built-in
  Sequential ancestors from the model root. A custom parent can bypass or reorder
  registered children, so its containment alone is not accepted as a call graph.
  Keras requires a linear Conv2D followed by a unique BN edge. Its equivalence
  boundary is `training=False`; training-mode BN behavior is not preserved.
- No eligible rule yields `no_applicable_transformation`, not an optimization claim.

Shared convolution call sites, arbitrary graph rewrites, frozen-backbone
architecture replacement and unsupported layouts fail explicitly. Candidate
validation performs forward/backward and a disposable fresh-optimizer smoke step
on a clone when examples are provided. It does not train the selected model or
establish task quality. Without examples those checks remain `not_run`.
State bytes are not peak memory; parameter reduction is not measured acceleration.
The [official optimization report](OPTIMIZATION_REPORT.md) exports a monochrome
PDF, HTML and JSON with before/after structure, exact change locations, applied
rule reasons, state correspondence and bounded static cost evidence. Use
`--report-format pdf` while optimizing, or
`deepbom optimization-report candidate --output report.pdf` afterward.
Delegate selection, kernel scheduling and measured latency remain separate runtime
workflows. This extension does not silently equate folding to delegation.

Exports contain a verified manifest, exact state payloads, source/evidence hashes,
comparison and validation, `model.py`, `model-code.md`, and `report.html` with code
copy/download and before/after structure. These are packages: the generated `.py`
loads adjacent verified state and is not a dependency-free standalone model.
Built-in PyTorch Sequential trees reconstruct directly; custom model packaging
requires an explicit local factory. Custom attributes preserve Python type distinctions; opaque attributes require an explicit adapter. External/global Python side effects remain outside snapshot scope. Restoration is checked before a package is published to
its local destination. Existing directories are never overwritten.

## Training: PyTorch and GradientTape

```python
from deepbom.training import watch

with watch(model, optimizer=optimizer, logdir="run-evidence",
           capture=("metrics", "weights", "gradients"), every=10) as evidence:
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        loss = loss_fn(model(batch.inputs), batch.labels)
        loss.backward()
        optimizer.step()  # hooks observe; DEEPBOM never calls backward or step
        evidence.record_metrics({"loss": loss}, at={"step": step})
    evidence.finish("ended")  # an explicit lifecycle declaration
```

```python
with watch(model, optimizer=optimizer, observation="manual", logdir="tape-evidence",
           capture=("metrics", "weights", "gradients")) as evidence:
    with tf.GradientTape() as tape:
        prediction = model(x, training=True)
        loss = loss_fn(y, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    token = evidence.before_update(optimizer=optimizer, at={"step": 0}, loss=loss,
        gradients=gradients, variables=model.trainable_variables,
        gradient_stage="computed")
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    evidence.after_update(token, optimizer=optimizer)
    evidence.record_metrics({"loss": loss}, at={"step": 0})
```

Automatic and manual update observation cannot be mixed. Update tokens cannot be
reused or attached to another optimizer. A returned optimizer call is recorded as
`returned_effect_unverified`, not proof that weights changed. Missing gradients
remain missing; sparse gradients retain stored indices, dense shape and a clear
stored-values scope, without silently densifying them. Gradient names with no
snapshot binding do not claim attachment to a particular weight state.

Use eager host observation for GradientTape; do not put recording calls inside a
traced `tf.function`. PyTorch compiled/distributed optimizer closures, AMP skipped
updates and distributed aggregation are not qualified automatic observations.
Use a quiescent boundary for snapshots; concurrent mutation is unsupported. A directory accepts one active writer. A stale lock after a process crash requires a new destination, not silent overwrite.
Capturing full state is opt-in, performs host copies and can be expensive. The
value budget is per tensor; an exceeded budget yields `not_assessed`, never a
sample masquerading as an exact full distribution.

For `fit`, `deepbom.training.callback(logdir=..., capture=("metrics", "weights"))`
records framework logs and selected batch-end states. It cannot claim internal
activations or gradients. Call `callback.close()` in `finally` if `fit` aborts.
Exiting a recorder without `finish` is `collector_closed`, not successful training.

## Explicit activations and selected results

```python
with watch(model, optimizer=optimizer, observation="manual", logdir="observations",
           capture=("metrics", "activations")) as evidence:
    state = evidence.record_snapshot(at={"step": 0}, advanced=True)
    output = model(x)  # the caller owns execution and context
    evidence.record_activations({"$model": output}, snapshot_sha256=state,
        at={"step": 0}, invocation="forward-0",
        context={"mode": "eval", "input_identity": "my-explicit-input-reference",
                 "state_boundary": "before_forward"})
```

Use exact native subjects for intermediate activations and separate calls for
repeated occurrences. Inputs and captures can contain sensitive information; the
native recorder writes locally. Input identity is caller-declared, not attested.
BatchNorm buffers may change during a train-mode forward, so before/after state
must not be conflated. There is no inferred causal activation flow.

```python
from deepbom.training import select_result
manifest = select_result("run-evidence", snapshot_sha256=recorded_state_digest,
                         output="selected-result.json")
```

Selection requires a state actually recorded in that Training snapshot. It does
not export a resumable training checkpoint or infer completion from a model file.

## DeepBoard and external record systems

```sh
python -m deepbom.training view run-evidence
python -m deepbom.training view run-evidence --output report.html
python -m deepbom.training inspect run-evidence
python -m deepbom.training export run-evidence --destination mlflow \
  --tracking-uri http://localhost:5000 --run-id EXISTING_RUN_ID
python -m deepbom.training export run-evidence --destination tensorboard --logdir tb
python -m deepbom.training import tensorboard \
  --logdir tb/deepbom/TRAINING_IR_SHA256 --output imported-run
```

DeepBoard serves on loopback only, reads validated evidence and never loads model
code. Observation selection, exact coordinates, weight/activation selection,
histograms, singular spectra, storage-axis slices, similarity and value/zero maps
have explicit unavailable states. SVG and PNG can be downloaded. Refresh reads a
new complete head; a failed/truncated object is not displayed as valid evidence.
The candidate report separately presents structure/state correspondence and code.

MLflow uploads only selected evidence to the explicitly specified existing run.
TensorBoard charts use event sequence, with original coordinates in sidecars.
Scalar summaries use float64 tensors so finite binary64 metrics do not overflow
or round prematurely through the legacy float32 `simple_value` field.
Projection 2.0.0 writes each complete immutable Training snapshot to its own
`deepbom/<training-ir-sha256>` run directory, returned as `event_logdir` in the
receipt. This prevents repeated observations when exporting successive snapshots
or multiple training runs to one parent log directory. Older flat event directories
remain readable by the importer; they are not overwritten or mixed with new runs.
W&B requires an explicitly passed `wandb_run` to `training.export`; local export
receipts distinguish destinations. Repeated local requests for the same immutable
snapshot reuse their receipt; remote exactly-once delivery is not claimed.
No integration uploads raw model state or input bytes. Metrics and model metadata
remain potentially sensitive and are sent only through an explicit export call.
CSV, TensorBoard, MLflow and W&B imports preserve logger-defined coordinates and
remain run-only/unbound unless separate evidence establishes a state connection.

## Validation and support limits

`check-native-evidence.mjs` checks schemas, exact calculations, SVD, budgets,
reference/digest tampering and chunk continuity. `check-native-training.py` uses
actual CPU frameworks, serialization, independent candidates, real learning loops,
Keras fit, local MLflow and TensorBoard and offline W&B. The browser test checks
interaction, downloads, dark mode and narrow layout. Existing numerical, SDK,
Python, MCP and browser regression checks remain required.

These tests do not qualify arbitrary custom layers, accelerators, every framework
version or distributed training. W&B cloud delivery requires credentials and was
not exercised. No claim of task-accuracy preservation or measured performance
benefit follows from this development validation.
