# Native Runtime Selector Capture

DeepBOM closes selector ambiguity only for one bound tuple:

`artifact SHA-256 x runtime binary SHA-256 x XNNPACK source/build identity x host CPU features x runtime options x concrete input shapes x lowering x dispatched microkernel`

The native path has four explicit responsibilities:

- `native/instrumentation`: thread-safe C ABI event sink linked into an instrumented runtime.
- `scripts/bootstrap-instrumented-runtime.mjs`: verified archive download, transactional pinned-source patching, and optional Bazel build.
- `scripts/capture-pinned-runtime.mjs`: pinned host build, benchmark invocation, event capture, collection, and package verification.
- `scripts/run-native-capture.mjs`: build/run/capture orchestrator.
- `native/runtime_collector`: fail-closed event validator and canonical assignment producer.
- `native/runtime_probe`: synthetic contract probe used only to test the plumbing.

Pinned TensorFlow and XNNPACK commits live in `native/pins.json`. The run-config
contract lives in `native/capture-run.schema.json`.

## Reproducible Validation

Run the full device-free validation path with:

```text
npm run check:native-capture
```

The check builds the collector and synthetic probe, runs the same fixed capture
twice, compares the complete assignment documents, independently checks timing
sum/mean/sample count, verifies package digests, confirms that the browser
rejects synthetic output as runtime evidence, and injects conflicting lowering
identity to prove fail-closed behavior.

Synthetic output uses:

```text
source.kind = deepbom_native_runtime_contract_probe
source.validation_scope = collector_contract_only_not_runtime_evidence
```

It cannot be imported by the viewer. It validates the collection contract, not
TFLite placement, XNNPACK lowering, CPU dispatch, latency, or a microkernel.

## Automatic Capture

For a complete device-free host capture, build and run the pinned instrumented
runtime in one command. Every concrete input and the selected target-profile
digest are part of the evidence identity:

```text
npm run capture:pinned-runtime -- --model model.tflite --output-dir capture-out --target-profile-id x86_avx2 --target-profile-sha256 <64-hex> --input 0:input:1x224x224x3
```

On Linux, bind the runtime worker pool to an explicit CPU set and collect the
OS-side evidence in the same package:

```text
npm run capture:pinned-runtime -- --model model.tflite --output-dir capture-out --target-profile-id rpi5_a76 --target-profile-sha256 <64-hex> --input 0:input:1x224x224x3 --threads 2 --cpu-set 2-3 --isolation-expectation exclusive_cpuset
```

`affinity_only` requires every sampled thread mask to remain inside the
requested set. `exclusive_cpuset` additionally requires the process cgroup v2
`cpuset.cpus.partition` state to be `isolated` and its effective CPU set to
match exactly. The collector does not infer exclusivity from `taskset` or
`isolcpus` alone. It records the complete timestamped affinity-sample ledger,
plus derived thread masks and processors, cgroup state,
kernel `isolcpus`/`nohz_full`/`irqaffinity`/`rcu_nocbs` values, frequency
policy, and cache-sharing CPU lists in `resource-partition-observation.json`.
The collector and package verifier independently reproduce every summary from
that raw ledger before its SHA-256 is bound into `runtime-assignment.json` and
the package index. A missing thread sample, an out-of-set mask, a divergent
summary, or an unobserved requested exclusive partition fails the capture.

This command verifies TensorFlow and XNNPACK archive SHA-256 values, verifies
every patch target's original SHA-256, applies the patch transactionally,
builds `benchmark_model`, records binary/build identity, runs the attribution
capture with XNN operator fusion disabled, invokes the collector, and verifies
the completed package. A transform or manifest failure restores every original
patch target and removes generated instrumentation files. Source caches are
accepted only with an archive-SHA-bound completion marker and all required
TensorFlow/XNNPACK sentinels. Bazelisk 1.29.0 is downloaded per host
OS/architecture from its pinned release URL, verified by SHA-256, and recorded
in the build identity; `--bazel <path>` is the explicit override.
On MSVC, the matching `benchmark_model.pdb` is required, hashed into the build
identity, rechecked before capture, and disclosed by filename and SHA-256 in
the packaged runtime build manifest.
The default source/build cache is `%LOCALAPPDATA%\DeepBOM\native-runtime` on
Windows and `$XDG_CACHE_HOME/deepbom/native-runtime` (or
`~/.cache/deepbom/native-runtime`) elsewhere; `--cache-dir` overrides it.

To prepare source only, or to prepare and build without running a model:

```text
npm run bootstrap:instrumented-runtime
npm run bootstrap:instrumented-runtime -- --build
```

For custom runners, create a run config conforming to
`deepbom.native_capture_run.v1.2`, then run:

```text
npm run capture:native -- --config path/to/capture-run.json
```

The orchestrator performs these steps without hand-authored evidence files:

1. Runs the optional `runtime.build_command`.
2. Hashes the artifact, runtime binary, and microkernel build-identity file.
3. Writes canonical build-manifest and runtime-options documents.
4. Exposes capture ID, event path, artifact identity, XNNPACK commit, and build ID through `DEEPBOM_*` environment variables.
5. Invokes the instrumented runtime with expanded `${artifact_path}`, `${events_path}`, `${capture_dir}`, and `${config_dir}` arguments; an optional Linux CPU-set request is enforced with `taskset` and sampled from procfs/cgroupfs.
6. Writes and validates the optional resource-partition observation before accepting the run.
7. Generates `deepbom.native_capture_manifest.v4` from the exact run inputs.
8. Builds and invokes the Rust collector.
9. Independently checks assignment schema, source kind, hashes, and sorted unique original-op rows.
10. Writes `deepbom.native_capture_package.v1.1` with SHA-256 for every capture file.

The output directory is created once and must not already exist. This prevents a
new run from silently inheriting stale evidence. The importable artifact is
`runtime-assignment.json`; the remaining files preserve the replay trail in one
compact directory.

Verify a copied or archived directory independently with:

```text
npm run verify:native-capture -- path/to/capture-directory
```

The verifier recomputes every indexed digest and checks the manifest, event
stream, build manifest, runtime options, and assignment identity chain.

## Instrumented Runtime Patch

The source transformer is bound to TensorFlow
`87bbf65b8d23d3f06912b1b2183587e1884bc45c` and XNNPACK
`23a67314f7afdbb76191589ae090d82bf55afbfa`. It patches the actual TFLite
XNNPACK delegate visitor, copies original-op provenance into every emitted XNN
subgraph node and runtime operator, wraps all 89 pinned `operator-run.c`
dispatch call sites, and matches the executed function pointer against 9,643
pinned microkernel symbols. ELF/Mach-O builds use linked weak addresses with a
dynamic-symbol fallback. MSVC builds resolve the live address through the
matching PDB with `DbgHelp`, require zero symbol displacement, and then require
an exact pinned-catalog name. An unresolved pointer is never promoted to
microkernel evidence.

One TFLite op may lower to multiple XNN nodes, and one runtime operator may
dispatch multiple compute invocations. Schema `deepbom.runtime_assignment.v1.9`
therefore preserves `lowerings[]` and `dispatches[]`; singular top-level
lowering/kernel fields exist only for a one-row inventory.

Native dispatch rows use `unique_context_function_selection_per_process`
sampling. The trace emits the first observed function pointer for each original
op, runtime node, compute invocation, and microkernel address tuple. A changed
runtime selector creates another row, while repeated tiles and benchmark runs
do not inflate the evidence file or masquerade as timing samples.

## Runtime Integration

Link `native/instrumentation/deepbom_runtime_trace.cc` into the pinned native
runtime and call:

1. `deepbom_runtime_trace_open()` before inference.
2. `deepbom_runtime_trace_emit()` for each executed original-op observation.
3. `deepbom_runtime_trace_close()` after the final invocation.

The sink writes directly to `DEEPBOM_RUNTIME_EVENTS_PATH`, injects the hashed
microkernel build identifier supplied by the orchestrator, serializes floating
point timing with round-trip precision, flushes each event, and rejects partial
kernel identity. It is safe for concurrent emitters.

An event declares `event_kind` as `placement`, `lowering`, `dispatch`, or
`execution`. It may contain `op_index`, `op_name`, `provider`, optional
`delegated`/`partition_id`, `runtime_node_id`, `compute_invocation_id`, optional
`duration_us`, and selector fields. All
microkernel fields must be present together and require `lowering_id`:

```json
{"op_index":12,"op_name":"CONV_2D","provider":"XNNPACK","delegated":true,"partition_id":"xnn-0","lowering_id":"convolution_to_igemm","kernel_id":"f32-igemm-4x8-neonfma","kernel":"xnn_f32_igemm_minmax_ukernel_4x8__neonfma","kernel_source_ref":"google/XNNPACK@23a67314f7afdbb76191589ae090d82bf55afbfa/src/f32-igemm/f32-igemm-4x8-neonfma.c","kernel_build_identifier_sha256":"<injected by sink>","duration_us":18.25}
```

The sink is deliberately not a selector oracle. Original-op mapping must come
from the TFLite execution/delegate plan, lowering IDs from the delegate lowering
path, and the symbol from the XNNPACK dispatch site that actually supplied the
function pointer. A generic profiler name or source-enumerated candidate must
not be passed as an executed microkernel.

Arena instrumentation uses separate C ABI events: one `memory_snapshot` after
each successful `ArenaPlanner::ExecuteAllocations` commit, followed by the
snapshot's complete `memory_allocation` and `memory_alias` inventory. Each
allocation records arena kind, tensor index, offset, size, and first/last live
execution nodes. Aliases record the in-place tensor/root relation. The pinned
MSVC patch suppresses only the upstream weak no-op arena debug definition when
DeepBOM instrumentation is enabled, so the matching strong debug implementation
links without changing non-instrumented builds. Patch manifest v1.3 binds both
`arena_planner.cc` and `simple_memory_arena.cc` by SHA-256.

## Collector Rules

The collector hashes the model, runtime binary, collector executable, raw event
stream, build manifest, microkernel build-identity file, and runtime options. It
probes CPU features natively and aggregates repeated events by original op
index. It rejects:

- mixed timed and untimed rows for one op;
- conflicting placement identity or malformed duplicate selector observations;
- partial kernel identity or a non-XNNPACK microkernel symbol;
- source references not pinned to the configured XNNPACK commit;
- event build identity that differs from the hashed build-identity file;
- unsorted or ambiguous compile definitions;
- missing `XNN_BUILD_ALL_MICROKERNELS` or `XNN_ENABLE_ASSEMBLY` values.
- arena snapshot count, buffer aggregate, allocation bound, lifetime, live-range
  overlap, alias-root, tensor/execution-node count, or ledger-digest mismatch.

An importable selector-aware export uses:

- `source.kind = deepbom_native_runtime_capture`
- `source.collector.schema = deepbom.native_runtime_collector.v1.1`
- `selector_context.schema = deepbom.runtime_selector_context.v1.1`
- `selector_context.backend_library = XNNPACK`

## Evidence Boundary

An op reaches `OBSERVED_MICROKERNEL` only when its dispatch inventory contains
native original-op mapping, lowering ID, stable kernel ID, executed kernel
symbol, commit-pinned source ref, and matching build identifier. CPU/build context without lowering is
`OBSERVED_RUNTIME_CONTEXT`; lowering without dispatch is `OBSERVED_LOWERING`.

`runtime_memory` is `OBSERVED_RUNTIME` only after collector validation. The
browser recomputes the canonical snapshot ledger SHA-256 and derives
`deepbom.arena_runtime_reconciliation.v1` from the final snapshot and static
tensor-index plan. A smaller observed TFLite arena after delegation is not
automatically a defect: delegated activations may live in XNNPACK-owned memory,
which this ledger deliberately excludes.

Collector attestation profile v1 remains unattested. The browser verifies internal identities
and binds them to the active artifact and target-profile hashes, but it does not
remotely attest the native producer. A real ARM/NEON claim still requires running
the instrumented ARM binary on that environment; host-only validation cannot
substitute for it.
