# Shared numerical engine boundary

The migrated numerical operations use the same `pkg/tflite_wasm_audit_bg.wasm`
in the browser, Node CLI, packaged SDK engine and local MCP. The host loader is
`web/lib/numerical-wasm.js`; Rust implementations are `portable_numerics.rs`,
`portable_spectrum.rs`, `portable_weight.rs`, `portable_distribution.rs`,
`portable_affine.rs`, `portable_pruning.rs`, `portable_metrics.rs`,
`portable_contracts.rs`, `portable_graph.rs`, `portable_cost.rs`, `portable_integer.rs`,
`portable_einsum.rs`, `portable_protobuf.rs`, `portable_shape.rs`,
`portable_redesign.rs`, `portable_evidence.rs`, `portable_identity.rs`,
`portable_workflow.rs` and `portable_decimal.rs`. This is an internal calculation
ABI, not a new Evidence IR family or a replacement for the existing artifact
formats.

## Readiness and identity

An asynchronous host must await `initializeNumericalWasm()` before calling a
synchronous numerical consumer. `web/app.js`, embedded widgets, workers and
asynchronous native/weight analysis entrypoints establish readiness. Synchronous
functions retain synchronous returns. A missing module, stale ABI or absent
required method fails; there is no fallback implementation in JavaScript or
Python. Node 20 hosts without `process.getBuiltinModule` load their built-ins
at the asynchronous boundary first.

The ABI marker is `deepbom.numerical_abi.v1:exact_binary64_v1`. The loader also
checks required exports, so an older binary with the marker alone is insufficient.
The numerical identity includes the SHA-256 of the loaded bytes. Packaged engines
compare it against their build-pinned digest. In an unpinned browser/source host,
the digest identifies the fetched bytes; it is not a signature or authenticity
attestation. The persistent native session also identifies the shared analyzer
bundle, because not all surrounding semantic rules have migrated to WASM.

Model/Snapshot/Weight/Activation/Training identity and metric configuration
versions are independent from the ABI marker. A calculation migration does not
make old evidence refer to a new producer. Built-in metric configurations use v2;
external metric v1 and Evaluation IR v1 remain readable.

## Data and numerical contracts

Hosts decode the supported source representation and pass contiguous binary64
arrays through wasm-bindgen. Source dtype, shape, storage order and exact source
identity remain in the existing input/IR contracts. Integer observations outside
binary64's exact range use the existing decimal-integer statistics path, rather
than first rounding them into floats. Existing framework snapshot transport
remains bounded JSON/base64; passing a live model to a worker is not supported.

- Exact moments accumulate binary64 values using arbitrary-precision integers,
  then round at the result boundary. Each streaming chunk is limited to 65,536
  values and rejects nonfinite inputs. Statistics separately count NaN and
  positive/negative infinity rather than treating them as zero.
- Counts remain exact through 2^53−1 observations; decimal-integer inputs have
  an explicit size budget. Shape products are arbitrary precision, bounded to
  100,000 bits, and returned as decimal strings. Dynamic/invalid dimensions are
  handled by the existing host contract before the finite-dimension ABI.
- Histograms preserve the existing fixed binary64 edge contract. Sampling and
  missing-payload status still belong to the common Weight/Activation workflow.
- Singular spectra accept at most 10,000,000 finite values, matching the public
  per-tensor Weight analysis limit, validate rows × columns,
  and preserve the existing work/sweep/convergence, underflow and overflow
  outcomes. A budget or convergence failure is `not_assessed`, not zero rank.
- Matrix unfolding, complete signed channel similarity and clustering, exact
  tile aggregation, serialized-block and selected-axis 2:4 sparsity, and paired
  weight comparison use the same portable owner. Nonfinite/invalid inputs fail
  before calculation. Similarity limits both arithmetic work and result cells,
  including a matrix with zero columns. Tile means reuse exact moments. An
  overflowing paired difference remains nonfinite in the working array and
  unavailable in JSON metrics; it is never treated as zero error.
- Distribution projections use exact decimal counts, including aggregates beyond
  u128. Percentiles remain nearest-rank bounds from fixed bins, not interpolated
  values. Total variation rounds one exact rational; unknown/overflow stays null.
- Affine reconstruction validates shape, storage order, integer range, cardinality
  and scale indexing before declaring dequantized values. Invalid or overflowing
  reconstruction retains the original codes and an explicit unassessed reason.
- Magnitude pruning preserves stable position tie breaks, existing zeros and the
  compensated energy method. It processes at most 2,000,000 values/4,096 channels,
  verifies supplied rankings, and changes no model file. Axis-permuted weight
  comparisons use the same bounded portable tensor indexing.
- SSIM and Generalized Dice require explicit shape, range/class/background and
  aggregation options. SSIM checks intermediate convolution expansion as well
  as input size. The portable implementation has separately identified methods;
  TorchMetrics remains an explicit optional external adapter and test oracle.
- The JSON `numerical_contract` entrypoint accepts at most 128 MiB and preserves
  unknown counts separately from zero. Exact count products, packed byte sizes,
  structural differences, graph partitions and boundary edges share this owner.
  Graph traversal uses an explicit stack rather than the host call stack.
- Standard ONNX contraction costs use exact integer arithmetic in Rust. The
  host supplies serialized shapes, attributes and operation classification;
  unsupported, dynamic or non-dense inputs retain an unassessed result. These
  nominal MACs are not latency measurements. ConvTranspose contributing-pair
  enumeration is bounded to one million terms per axis. Einsum expansion rejects
  ranks above 100,000 and more than one million expanded labels.
- Guarded integer expressions use the same exact arithmetic and ConvTranspose
  pair counter as static ONNX, Core ML and ExecuTorch consumers. Invalid bindings
  and failed divisibility/equality guards remain unavailable. A request is bounded
  to 100,000 expression visits and 131,072-bit intermediate integers. Symbolic
  text rendering remains a host function; it does not evaluate the expression.
- Dimension algebra, Reshape cardinality, broadcasting, Conv/ConvTranspose,
  spatial pooling, Flatten/Gemm/MatMul, Concat, Pad and Slice arithmetic use portable
  shape kernels. Dynamic dimension expressions share guarded integer arithmetic
  and Slice normalization. Concrete arithmetic is exact; a shape dimension outside
  the existing safe-integer representation remains unknown rather than rounded.
  Host dimension decoding keeps null, boolean and empty text unknown, including
  Size inference and declared/inferred type merging. Shape rank is bounded to 100,000 and a derived symbol to one MiB; a broadcast
  request has at most one million axis/input comparisons. Format/opset admission and control-flow traversal still
  belong to the common ONNX adapter.
- Redesign executable admission validates producer/consumer shapes, identities,
  exact cardinality, emitter input arity, spatial output, concat and broadcast
  geometry through these same kernels. PyTorch/Keras emitters consume the validated plan. Linear feature counts
  are emitted from the shared exact product without host floating-point arithmetic.
- Evidence subject references, bounded unique IDs and millisecond RFC3339 calendar
  checks use one portable leaf validator. Evidence/native document canonical
  identity uses RFC 8785 UTF-16 key ordering, ECMAScript binary64 spelling and
  SHA-256 in WASM. Host admission rejects non-JSON objects, undefined, cycles,
  nonfinite numbers and lone surrogates before transport, without invoking
  `toJSON`. Sealing captures a detached value once, preserving original key
  order for existing JSON exports; later caller mutation cannot invalidate its
  content hash. Historical admitted document digests are preserved. General-purpose
  publication/bootstrap hashing and JCS helpers have not yet migrated.
- Snapshot scope/content/path checks, Evaluation protocol/measurement contracts,
  comparison input admission and Deployment half-open interval selection use
  portable validators. Hosts supply explicitly resolved documents/bytes; missing
  references stay unresolved. Overlapping deployments and clock uncertainty
  crossing a boundary cannot establish a unique deployment. These declarations
  do not authenticate execution or clinical use.
- Reported-decimal differences, sums, direction reversal, rounding intervals and
  threshold comparisons use one exact decimal owner. Its input is the shortest
  ECMAScript spelling of an already reported number, not recovered measurement
  precision. Overflow/underflow remains null in the numeric projection while the
  exact decimal text survives. Decimal operands are bounded to 32,768 coefficient
  characters and exponents from -10,000 through 10,000. Existing evidence JSON
  requests share the 128 MiB ABI budget and serde JSON nesting limit; excessive
  inputs fail without a host calculation fallback.
- Shared protobuf framing validates uint64 varints, field numbers and complete
  wire spans before exposing offsets into the original bytes. It supports wire
  types 0, 1, 2 and 5, with at most two million fields per message. Packed floats
  reject partial elements and preserve nonfinite values for the consumer to
  classify. Format-specific field meaning remains with the format adapter.

Adapters free streaming accumulators deterministically. A returned JS view must
not be retained across memory growth or reused after a transfer. The current
native session requests and replies are sequential, sequence-checked, and bounded
to 128 MiB; repeated snapshots are separate immutable jobs and bindings.

## Execution and limits

Monitor owns the CPU analysis queue and session. Admission reserves capture count
and raw-plus-encoded bytes before copying. Queue bytes are not total process RSS.
The process runner supports Linux process-group termination and explicit file,
address-space and CPU-time limits. An address-space limit is neither resident
memory nor GPU memory. Managed defaults do not claim a total host/worker RSS cap.
Unsupported mandatory plugin isolation is rejected before loading plugin code.
The Linux launcher also installs a parent-death signal for the direct engine
and checks for a parent exit during startup. A controller killed before Python
cleanup therefore does not leave that engine running. This does not guarantee
cleanup of arbitrary descendants spawned by a plugin; the Linux setting is
cleared in forked children. See the [Linux parent-death signal contract](https://man7.org/linux/man-pages/man2/PR_SET_PDEATHSIG.2const.html).

Close shares an overall monotonic deadline across drain and engine requests.
It attempts cancellation and actual process termination, then records pending or
incomplete work. File I/O stalls, bounded JSON parsing on the owner thread and
uninterruptible OS work prevent a hard real-time guarantee. An unsealed journal
is not a valid final Training IR. Explicit `deepbom.training.recover` acquires the
same writer guard, validates and seals pending evidence, and recovers immutable
analysis jobs. POSIX crash and interrupted-publication tests qualify this path;
legacy locks without a kernel guard fail closed. Recovery does not resume
training or invent observations lost before persistence. Results completed after
the collection ended remain supporting evidence unless their original event was
already committed. Recovery is user-invoked, not automatic training restart.

## Current migration boundary

Exact statistics, histograms, shape products, cosine/comparison reductions,
singular-spectrum iteration/summary, weight unfolding/similarity/sparsity/tile
aggregation/comparison, distribution projections, affine reconstruction, pruning
selection, built-in SSIM/Generalized Dice, exact aggregation, structural field
differences, graph topology/partitions/boundaries, standard ONNX contraction
costs, dimension algebra and principal shape kernels, guarded and dynamic dimension
integer evaluation, executable redesign admission, evidence leaf contracts,
Evidence/native canonical identity, Snapshot/Evaluation/Deployment admission,
deployment interval selection, reported-decimal arithmetic, Einsum parsing and
shared protobuf framing have WASM owners. ONNX operator/container contracts,
conditional shapes, source-pinned ORT and ML semantics, higher-level Artifact/Model
and numerical family validators, conversion receipts, provenance graphs and
runtime reconciliation also use the portable engine. Host adapters retain format
decoding, lexical scope/function/control-flow orchestration, dependency I/O,
change-review composition, publication/bootstrap byte hashing, rendering,
framework hooks, explicit model execution and process scheduling.

The loader requires `onnx_semantics`, `semantic_validation`,
`conversion_receipt` and `provenance` in the feature handshake. An older binary
is rejected instead of activating a host fallback. Numeric payload transport
preserves INT64 decimal strings and NaN/Infinity/negative-zero tokens. ONNX
shape-analysis method v1.31 records corrected value propagation and 64-bit
bounds; Evidence IR wire identities remain unchanged.


Use [the integration guide](AUTOMATIC_DEVELOPMENT_INTEGRATION.md) for supported
entrypoints and [the CPU measurement record](MONITOR_OVERHEAD.md) for measured
costs and limitations. Exact numerical tests, independent metric oracles,
Chromium worker tests and installed-channel tests must accompany each migrated
scope; their success cannot qualify an unported calculation by association.

The HDF5/Keras decoder is the pinned `h5wasm` browser build, operating on selected
bytes in MEMFS. The application reuses the copied vendor module rather than
embedding another copy in its chunks; the isolated ChatGPT audit worker remains
self-contained. This container decoder does not provide a model execution engine.
Runtime UI availability follows the same format capability registry as the other
consumers, so a static-only file never acquires a synthetic benchmark contract.

With the Python test dependencies (`numpy`, `h5py`, Keras and its backend), the
intended DEEPBOM package and Playwright installed, run
`DEEPBOM_HDF5_DIST_ROOT=dist python scripts/check-hdf5-source.py` after the pages
build. This generates actual HDF5/Keras files, compares installed-engine and browser
Weight IR, checks production UI exports at root and subdirectory URLs, and verifies
that a subsequent malformed artifact produces a visible error. External/soft
links, aliases, big-endian floats and unsupported scalar types are included.

## 2026-10-20 ownership extension

Required contract features now include `onnx_type` and `native_tensor`, and the
byte ABI includes `evidence_sha256`. The public ABI version and admitted IR
schemas remain unchanged; a stale module fails the feature handshake.

ONNX TypeProto validation and joins/unification, native tensor identity/count
validation, numerical-IR field sets/count admission and numerical-IR seals use
Rust/WASM. Python `content_hash` calls the same identity contract through the
native bridge, reusing a Monitor's owned session when present. Job request and
completion digest work participates in its cancellation scope. Host bootstrap
hashes still verify the packaged executable before it is trusted.

The native producer now rejects an alias equal to its own tensor ID at intake;
imported identity validation already rejected that contradictory representation.
No model value, precision conversion or statistical definition changes here.
See `WASM_EXECUTION_BOUNDARY.md` for the host execution boundary.

## Portable IR semantics

`numerical_contract("ir_semantics", input)` owns `value_type`, `elements`,
`metric`, `new_metric`, `aggregate_metrics`, `exact_integer`,
`exact_count_value`, `validate_exact_integer`, `safe_bigint_number`, `logical_inventory`,
`member_bounds`, `member_overlaps`, `members`, `input_manifest` and
`runtime_provenance`. Host JSON admission rejects values that would be lost in
transport; semantic errors come from Rust. Validation operations return null on
success and fail on inconsistent identities, counts, references or digests.

Ranges are half-open, exact decimal integers. Empty ranges have no overlap,
adjacent ranges are distinct, and transitive overlap groups preserve the member
and offset basis. Member completeness is checked against actual bindings.
A known zero dimension establishes zero elements only after every dimension's
contract has been validated. Metric totals with unknown contributors stay null;
the assessed subtotal is separate. Duplicate source references cannot be summed.

`numerical_contract("coreml_deployment", input)` applies the pinned source tables
passed by the host. Unknown MIL opsets preserve the declared load floor but leave
the observed feature floor unassessed. The host adds source metadata and display
text. Neither operation requires a JS calculation fallback.
