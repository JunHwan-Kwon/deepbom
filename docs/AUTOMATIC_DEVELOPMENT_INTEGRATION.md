# Automatic development integration

These local development adapters reuse DEEPBOM's native collector and common
evidence engine. They do not own `backward`, optimizer steps, model selection,
patient-level aggregation or training completion. A collector that closed
normally is not proof that training or evaluation passed.

## WASM execution policy and migration status

Portable analysis and calculation must be owned by the common Rust/WASM engine.
This applies to existing JavaScript calculations and new built-in metrics, not
only to TFLite analysis. Web, CLI, SDK and MCP must call the same qualified WASM
implementation for a migrated operation. Framework adapters collect declared
state and observations; host code handles files, scheduling, transport and
presentation. They must not introduce parallel numerical or semantic rules.

The common portable engine owns exact binary64 moments, fixed-edge histograms,
cosine/comparison reductions and shape products in Rust/WASM. Built-in
SSIM and Generalized Dice also use WASM, with separately identified binary64
methods. Singular-spectrum iteration and its derived rank/error metrics also
use WASM. Exact count aggregation, structural field/node comparison, connected
partitioning, port-boundary projection and topology now share WASM owners too.
The supported static ONNX contraction rules (also used by native reports),
guarded integer evaluation, Einsum equation parsing and protobuf wire framing
are in that engine. Static ONNX, symbolic costs, Core ML and ExecuTorch reuse
one ConvTranspose contributing-pair counter. The host
adapters retain format transport and source collection. ONNX node/container/ML
semantics and Artifact/Model/Weight/Activation/conversion/provenance admission
now use Rust/WASM as well. Lexical graph/function/control-flow orchestration,
I/O, lifecycle and presentation remain shared host responsibilities; see
[the execution boundary](WASM_EXECUTION_BOUNDARY.md).
Framework hooks, gradient capture and trusted model execution require their
framework runtime. No migrated numerical operation falls back to JS/Python.

Migration must preserve the existing IR, method definitions, exact counts,
missing-value states and source bindings. Each operation needs independent
numerical checks, old-result reader tests, actual browser and packaged-channel
execution, and end-to-end time/memory measurements including tensor transfer.
Documented numerical changes need method versioning. A migrated operation must
not silently fall back to a second JS/Python implementation when WASM fails.
The retired legacy histogram ABI remains retired: the common Weight IR contract
is the migration target, rather than the obsolete histogram semantics.

## Managed analysis and qualification scope

For Linux single-process CPU models, the `weights` monitor owns admission,
immutable capture, the engine process, result attachment and shutdown. `basic`
collects lifecycle and supplied scalars; detailed capture is opt-in. Distributed,
sharded and GPU copying are not qualified by this managed path. Explicit
`record_snapshot()` retains its synchronous Snapshot SHA return contract.

`max_pending`, `max_queue_bytes` and `close_timeout` bound admission and the
shutdown attempt. Counts include capturing, queued and running work. The byte
reservation accounts for raw capture plus bounded encoded requests; it is not
a claim about total process RSS. A full queue skips capture before copying and
records the skipped count. Completed evidence is not deleted automatically.

Shutdown uses a shared monotonic deadline for drain, cancellation and engine
calls, reserving time to seal the training record. Linux engine process groups
can be killed; a non-reaped controller retains its lock and is reported as
pending. OS/disk stalls prevent an unconditional real-time guarantee. Failed
sealing leaves an explicitly incomplete raw journal rather than fabricating a
valid final Training IR.

Automatic update snapshots retain the returned optimizer token and its original
attempt number. Multiple outstanding manual update tokens do not borrow the
latest attempt's identity. The outcome still does not establish a weight change.

A Ctrl+C during drain still triggers cancellation and cleanup before the
interrupt is propagated. An existing training exception takes precedence over
cleanup errors. A disabled observer does not replace a subsequent training
failure with an interrupted state; a training KeyboardInterrupt is recorded as
interrupted. These cleanup paths cannot run after SIGKILL; the explicit recovery
workflow below handles persisted evidence from a dead process. An unresponsive OS
is outside the controlled shutdown guarantee.

Results completed after an explicit training terminal state are supporting
`analysis-bindings.json` records. `read_analysis_view()` and DeepBoard validate
the same snapshot/model/weight bindings and display late results separately;
they do not append invented events to immutable training history.

Persisted commit intents can be recovered after process death. On POSIX, a
kernel file lock protects a persistent guard inode; a dead owner marker can be
reclaimed only after acquiring that lock. Legacy markers still require explicit
inspection. Do not remove guard files while another process may use the directory.

```python
from deepbom.training import recover
receipt = recover("evidence/run-1", timeout=30)
```

The Python CLI exposes the same workflow:

```bash
deepbom training recover evidence/run-1 --timeout 30
```

Recovery validates the persisted Training intent through the common engine,
acknowledges already committed events without duplication, and closes an abandoned
running collection as interrupted. Completed snapshot jobs without a Training
event stay late supporting evidence. Missing buffered observations remain unknown;
recovery does not resume training, reconstruct gradients, or invent a model state.
Immutable queued inputs require the same producer/engine identity. Active writers,
conflicting heads and altered input/result digests are rejected. `--skip-analysis`
seals the persisted observations without executing pending snapshot analysis.
The timeout governs controlled engine work; host filesystem stalls are not a
hard real-time guarantee.

`scripts/benchmark-monitor-overhead.py` measures off, scalar, capture-only,
capture+worker and explicit activation cases with warmup, repeated order changes,
actual completed/skipped counts, trainer/worker RSS and close time. It requires
`psutil` in the experiment environment. Synthetic CPU measurements do not
establish GPU, distributed or production overhead guarantees.
The [CPU measurement record](MONITOR_OVERHEAD.md) reports 60 installed-build runs,
including substantial small-model overhead and captures skipped under pressure.
It does not establish a universal recommended sampling period.

## Minimal connection

Install only the adapters you use:

```bash
pip install 'deepbom[torch,report]'
# Alternatives: deepbom[tensorflow,report], deepbom[autolog],
# deepbom[lightning], deepbom[metrics], deepbom[tracking].
```

```python
import deepbom

# Call outside a concurrent optimizer update. Optional examples run a copy.
inspection = deepbom.inspect_native(model, example_args=(example_batch,))

with deepbom.monitor(model, optimizer=optimizer, logdir='evidence/run-1',
                     preset='weights', every=100, report=True) as session:
    train()  # The application still owns its complete training loop.
```

`basic` records lightweight optimizer lifecycle observations; `weights` adds
selected snapshots; `gradients` records gradients at the declared optimizer
boundary. `activations` permits explicit captures. It does not discover an
arbitrary task's input, target, loss, validation split or activation points.
Use `session.record_metrics(values, at={...})` and
`session.record_activations(...)` with the collector's explicit capture contract.
The returned `session.receipt` identifies its directory, requested capture kinds,
errors and completion state. Actual observations are in the Training IR events;
a requested capture kind does not establish that any value was collected.
Hook handles and writer locks are removed on exit.
`errors='record_and_disable'` preserves training and records collection failures;
`errors='strict'` propagates a collection error. Neither masks an original
training exception. Observer overhead and host I/O must be measured on your task.

For a manual TensorFlow `GradientTape` loop, `monitor` and the existing
`deepbom.training.watch(..., observation='manual')` use the same recorder.
Explicitly record the tape gradients and update boundary. DEEPBOM never calls
a second backward pass or applies gradients on your behalf.

```python
with deepbom.monitor(model, logdir='evidence/tape', preset='gradients') as session:
    with tf.GradientTape() as tape:
        loss = loss_fn(target, model(inputs, training=True))
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    token = session.before_update(optimizer=optimizer, at={'step': 0},
                                  gradients=gradients, variables=variables)
    optimizer.apply_gradients(zip(gradients, variables))
    session.after_update(token, optimizer=optimizer)
```

For Keras `fit`, keep the public callback or use `session.keras_callback()`.
Keras callbacks observe logged metrics and selected batch-end weights; they do
not automatically obtain internal gradients or activations. A callback belongs
to one fit/session and must not be reused after closing.

```python
with deepbom.autolog(framework='keras', logdir='evidence/fits', preset='basic'):
    history = model.fit(x, y, callbacks=user_callbacks)
```

Autolog is qualified for TensorFlow 2.20.x, TensorFlow-backed Keras 3.11.x and
wrapt 2.5.0. It installs one optional wrapper around the supported public fit
entry point, preserves the existing callbacks, return value and exceptions,
and restores its own wrapper on disable. It can be enabled before importing
Keras. Repeated installation is rejected; an already monitored model is not
instrumented twice. Only the last 100 in-memory receipts are retained; persisted
evidence remains on disk. The terminal wrapper accepts an explicit Python script:

```bash
deepbom instrument --framework keras --logdir evidence/fits -- python train.py
```

Lightning uses `deepbom.training.lightning.lightning_callback(...)` and the
official callback events. The initial adapter is for a single-process,
single-optimizer automatic-optimization trainer. Gradients carry the observed
stage (after unscale, before Lightning clipping), and completed step hooks
count optimizer updates rather than batches. Distributed and manual-optimization
training is rejected. CPU fp32 and bf16 have regression coverage; other precision
modes are rejected until separately qualified.

Validation observations preserve the epoch and, for Lightning, dataloader index.
Lightning's `logged_metrics` is a framework snapshot: it can retain previously
logged values. Its observation coordinates do not establish that every metric
was newly computed in that batch or phase. Separate `validate`, `test` and
`predict` runs are not automatically instrumented by this fit callback.

DeepBoard filters tensors using recorded module ownership and snapshot identities.
Its module selector includes descendants; bounded pages and search remain available
for large captures. A module tree is not a complete functional-operation graph.

## Immutable background analysis

`deepbom.jobs.AnalysisQueue` runs the same existing engine operations in one
bounded worker. Submit JSON transports created at a quiescent snapshot boundary,
never a live mutable model, autograd tensor or arbitrary Python callable.
Requests include operation, configuration and producer identity; canonical
hashes identify jobs. Request and result files are immutable. Status updates
are atomic, completed results are digest-checked, and recovery retries only
declared transient failures (timeouts/interrupted or blocked I/O). Validation,
identity and budget failures require caller review, not an automatic retry loop.
Cancellation of running work requests termination of the managed engine process
group and suppresses its result. `close()` uses the duration configured through
`close_timeout` on construction; `close(deadline=...)` can share an enclosing
monotonic deadline. Its receipt reports pending jobs, elapsed time and whether
controllers stopped. POSIX guard locks prevent concurrent recovery and release
on process death; legacy writer markers require explicit inspection. `recover()`
is explicit and requires the retained immutable request. A result committed before
its progress acknowledgement is verified and acknowledged without re-execution.

`deepbom.completed_files.mark_complete` writes a producer completion manifest
beside a finished artifact. `CompletedFileWatcher` polls these manifests and
verifies hash, size and safe relative path before staging an immutable copy and
submitting `audit_file`. A filesystem event or stable size alone is not a
completion signal. Multi-file producers must provide the exact closure and the
common Artifact Set identity; no directory traversal guesses additional files.
Polling also handles missed NAS notifications. Per-artifact and total staging
budgets fail explicitly; archiving is a separate operator action.

External export remains explicit. Existing `training.integrations.export`
targets an existing MLflow run, TensorBoard directory or W&B run; it does not
create an inferred experiment. For file jobs, use the completed result path
with your tracker's artifact API after verifying job success. No watcher or
monitor implicitly sends model bytes to DEEPBOM or a tracking service.

## Evaluation and optional analyzers

`deepbom.metrics.evaluate` defaults to built-in WASM SSIM and Generalized Dice
for explicit NumPy, eager TensorFlow or PyTorch tensors. Select
`backend="torchmetrics"` for the existing pinned 1.9.0 external adapter. Supply prediction, target, unique sample IDs and all
metric options, including range, class/background and missing-value policies.
It hashes the selected tensor transports and records configuration, version,
sample denominator and omissions. Samples are analyzed independently. Built-in
configuration v2 records the WASM digest, ABI and numerical method; external
configuration v1 remains readable. Evaluation v1 still imports an observation
as `externally_reported`; its validator does not recompute predictions.
The result is a sample-weighted measurement, not automatically a patient-level
estimate, binary Dice, confidence interval or clinical verdict.
The built-in NumPy path needs `deepbom[native]`; it does not need TorchMetrics
or PyTorch. The existing `deepbom[metrics]` extra is retained for users of the
explicit external TorchMetrics backend.

`configuration_document(observation)` returns a context-ready sealed settings
document. `evaluation_record(observation, template, metric_id=..., documents=...)`
uses the existing Evaluation workflow; the template must explicitly bind the
model, cohort, protocol, data, run and exact configuration. Unavailable context
is not silently promoted into verified evidence.

Optional analyzers use the standard `deepbom.analyzers` entry-point group and
a distribution-owned `deepbom-analyzers.json` descriptor. Discovery reads only
metadata; only an explicitly selected entry point is imported. Name, version,
required inputs, execution behavior, cost, dependencies, operation and result
schema are checked. Plugins produce supported common workflow inputs; they
cannot replace the MAC definition or silently overwrite the core IR.

Plugin descriptor v1 retains its explicitly trusted in-process behavior. V2 adds
Python/dependency ranges, platform, network/GPU declarations and resource limits.
Selected v2 code imports only inside a Linux child after RLIMIT_CPU, RLIMIT_AS
and RLIMIT_FSIZE are applied. Wall deadlines terminate its process group.
`run(..., return_receipt=True)` returns the workflow result and execution facts.
Network deny, GPU isolation and filesystem allowlists are not supported by this
trusted runner: mandatory requests are rejected before import. Address-space
limits are not RSS or VRAM quotas. This is not a malicious-code sandbox.

TorchLens 2.36.1 is an **experimental qualification probe**, not a supported
automatic backend. `scripts/check-torchlens-prototype.py` checks a copied CPU
eval model, output/state preservation, module-address mappings, functional
unmapped rows and capture overhead. This does not qualify arbitrary models,
training gradients, other devices or per-batch collection. TorchLens's own
FLOP values are not substituted for DEEPBOM MACs.

## Reports, scope and correspondence

The existing report query v1 remains the default. Opt in to v2 for validation,
probe conditions, source pointers, whole-scope invocation sorting and byte-bounded
paging:

```bash
deepbom optimization-report report.json --query-version 2 --section summary
deepbom optimization-report report.json --query-version 2 --section invocations --sort macs_desc --limit 25 --max-bytes 65536
deepbom optimization-report report.json --query-version 2 --section facts
```

Python `deepbom.optimization_query` and Node `optimizationQuery` call this same
owner. Local MCP exposes the same optional version and selectors. `next_offset`
continues the exact filtered/sorted sequence. A page export includes only its
returned rows and source references; complete report exports remain separate.
Web and PDF use the same report facts. State elements, physical storage,
logical bytes, observed module MAC subtotals and measured runtime cost remain
different quantities. A validation failure appears alongside execution completion.

Native inspection returns `deepbom.native_inspection.v1` using the existing
Model/Snapshot/Weight IR, and the common native cost method 1.1.0. This adds the
qualified ConvTranspose2d mapping; historical optimization reports retain their
default method 1.0.0. Unknown or inconsistent call geometry is not assigned zero
MACs. Module hooks do not establish a complete functional-operation graph.

Without probe inputs, inspection does not run a forward pass. With inputs, it
runs a copied model in the caller's process and restores Python/NumPy default
RNGs plus Torch CPU and selected CUDA RNGs on success or exception. Call at a
quiescent boundary. Arbitrary user side effects, user-owned generators and
TensorFlow stateful-op RNG counters are not isolated; use a separate process
when those are involved. This is model-state isolation, not a guarantee of
process-wide noninterference for arbitrary executable models.

`deepbom.correspondence.correspondence` accepts source and target Model IR
references, explicit many-to-many binding groups and the source documents.
Declared, exporter-observed, ambiguous and unmapped subjects remain distinct.
`mapped_comparison` returns aligned native-module and serialized-operation records;
it never infers functional equivalence or subtracts native trainable parameters
from exported constant storage. The original same-format semantic diff is unchanged.

The supporting contracts are in
[`deepbom-development-analysis-v1.schema.json`](schemas/deepbom-development-analysis-v1.schema.json).
Schema validation does not replace digest, source binding, conservation and
common-engine recomputation checks.

## Static format qualifications

- ZIP64 and pickle memo indices are bounded syntax decoding; no Python unpickling.
- Keras config preserves reused layer calls and tensor ports separately. Nested
  bodies are inventoried, not inlined. Build input shapes are not output shapes.
- SavedModel inventories all MetaGraphs, functions, resources and signatures;
  `--tf-metagraph N` selects the graph (default 0). Function bodies are not counted
  as observed calls. Variable values use the selected TensorBundle index/shards,
  offsets, CRCs and common numeric decoder. Unsupported slices/types stay unassessed.
- HDF5 uses pinned h5wasm 0.10.3 in an isolated selected-file MEMFS, not host
  filesystem access. Files are limited to 128 MiB, objects/links to 20,000, numeric
  datasets to 64 MiB per decode. Hard-link aliases are deduplicated; external and
  soft links are not followed. Compound, variable-length and virtual datasets are
  not numerically decoded. Physical storage size is unknown when not exposed;
  logical bytes are not used as a substitute. `.keras` uses the selected
  `model.weights.h5` member. Dataset-to-layer binding is not guessed from names.
- Core ML and sharded SafeTensors use the common verified selected-member
  resolver before numerical inspection. Shared storage is not evidence of shared
  training parameters. All these paths remain static; hosted MCP capabilities
  are not expanded to execute native code.

## Rechecking the qualified integrations

Run `python scripts/check-development-integrations.py` in the optional framework
environment described in the script. It executes actual Torch/Keras/Lightning
training, metric and parser checks, generated-model comparisons and browser
navigation tests. `--torchlens` adds the separately pinned experimental probe.
This suite is separate from base Node checks so installing DEEPBOM does not
require every framework. No patient data or external tracking server is needed.
