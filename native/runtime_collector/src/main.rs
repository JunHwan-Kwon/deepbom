use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const MANIFEST_SCHEMA: &str = "deepbom.native_capture_manifest.v4";
const OUTPUT_SCHEMA: &str = "deepbom.runtime_assignment.v1.9";
const COLLECTOR_SCHEMA: &str = "deepbom.native_runtime_collector.v1.1";
const SELECTOR_SCHEMA: &str = "deepbom.runtime_selector_context.v1.1";
const RUNTIME_MEMORY_SCHEMA: &str = "deepbom.runtime_memory.v1";
const TFLITE_DELEGATE_BUILD_SCHEMA: &str = "deepbom.tflite_delegate_build_inventory.v1";
const TENSORFLOW_SOURCE_COMMIT: &str = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const TFLITE_CMAKE_SOURCE_SHA256: &str =
    "bc8574b999dc15f8ce34939303afa70fa026ba2085f290e4f73c9a73163b7694";
const TFLITE_GPU_OPTIONS_SOURCE_SHA256: &str =
    "8db9e012233f6ca9f58de9acc5f8e351fbef4d29b4b852ca887a4e0c364abde1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureManifest {
    schema: String,
    capture_mode: CaptureMode,
    artifact_path: PathBuf,
    artifact_sha256: String,
    target_profile_id: String,
    target_profile_sha256: String,
    runtime: RuntimeManifest,
    source: SourceManifest,
    device: DeviceManifest,
    build: BuildManifest,
    invocation: InvocationManifest,
    instrumentation: Instrumentation,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum CaptureMode {
    RuntimeCapture,
    SyntheticContractProbe,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeManifest {
    name: String,
    version: String,
    backend: String,
    build: String,
    binary_path: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceManifest {
    collected_at: String,
    capture_id: String,
    collector_source_commit: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeviceManifest {
    identity: String,
    #[serde(default)]
    nnapi_runtime_feature_level: Option<u32>,
    #[serde(default)]
    nnapi_accelerator_identity: Option<String>,
    #[serde(default)]
    nnapi_capability_source: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BuildManifest {
    tensorflow_source_commit: String,
    xnnpack_source_commit: String,
    microkernel_build_identifier_path: PathBuf,
    build_manifest_path: PathBuf,
    compile_definitions: Vec<CompileDefinition>,
    #[serde(default)]
    cmake_system_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CompileDefinition {
    name: String,
    value: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InvocationManifest {
    inputs: Vec<InputShape>,
    thread_count: u32,
    runtime_options_path: PathBuf,
    #[serde(default)]
    resource_partition_request: Option<ResourcePartitionRequest>,
    #[serde(default)]
    resource_partition_observation_path: Option<PathBuf>,
    #[serde(default)]
    tflite_gpu_experimental_flags: Option<u64>,
    #[serde(default)]
    tflite_gpu_max_delegated_partitions: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResourcePartitionRequest {
    requested_cpu_ids: Vec<u64>,
    affinity_mode: String,
    isolation_expectation: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct InputShape {
    tensor_index: usize,
    name: String,
    shape: Vec<u64>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Instrumentation {
    lowering_ids: bool,
    microkernel_ids: bool,
    arena_allocations: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RuntimeEvent {
    #[serde(default)]
    event_kind: EventKind,
    op_index: usize,
    op_name: String,
    provider: String,
    delegated: Option<bool>,
    partition_id: Option<String>,
    lowering_id: Option<String>,
    kernel_id: Option<String>,
    kernel: Option<String>,
    kernel_source_ref: Option<String>,
    kernel_build_identifier_sha256: Option<String>,
    duration_us: Option<f64>,
    runtime_node_id: Option<usize>,
    compute_invocation_id: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum EventKind {
    Placement,
    Lowering,
    Dispatch,
    Execution,
    #[default]
    Observation,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MemorySnapshotEvent {
    event_kind: String,
    memory_snapshot_id: u64,
    non_persistent_arena_bytes: u64,
    persistent_arena_bytes: u64,
    tensor_count: usize,
    execution_node_count: usize,
    allocation_count: usize,
    alias_count: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MemoryAllocationEvent {
    event_kind: String,
    memory_snapshot_id: u64,
    tensor_index: usize,
    arena: String,
    offset_bytes: u64,
    size_bytes: u64,
    first_node: i32,
    last_node: i32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MemoryAliasEvent {
    event_kind: String,
    memory_snapshot_id: u64,
    tensor_index: usize,
    shared_with_tensor_index: usize,
}

#[derive(Default)]
struct ParsedEvents {
    op_events: Vec<RuntimeEvent>,
    memory_snapshots: Vec<MemorySnapshotEvent>,
    memory_allocations: Vec<MemoryAllocationEvent>,
    memory_aliases: Vec<MemoryAliasEvent>,
}

#[derive(Debug, Serialize)]
struct RuntimeMemoryEvidence {
    schema: &'static str,
    status: &'static str,
    evidence_class: &'static str,
    tensorflow_source_commit: String,
    snapshot_count: usize,
    peak_non_persistent_arena_bytes: u64,
    peak_persistent_arena_bytes: u64,
    peak_combined_arena_bytes: u64,
    final_non_persistent_arena_bytes: u64,
    final_persistent_arena_bytes: u64,
    final_combined_arena_bytes: u64,
    allocation_ledger_sha256: String,
    snapshots: Vec<RuntimeMemorySnapshot>,
    method: &'static str,
    interpretation_boundary: &'static str,
}

#[derive(Debug, Serialize)]
struct RuntimeMemorySnapshot {
    memory_snapshot_id: u64,
    non_persistent_arena_bytes: u64,
    persistent_arena_bytes: u64,
    combined_arena_bytes: u64,
    tensor_count: usize,
    execution_node_count: usize,
    allocation_count: usize,
    alias_count: usize,
    allocated_interval_bytes: u64,
    allocations: Vec<RuntimeMemoryAllocation>,
    aliases: Vec<RuntimeMemoryAlias>,
}

#[derive(Clone, Debug, Serialize)]
struct RuntimeMemoryAllocation {
    tensor_index: usize,
    arena: String,
    offset_bytes: u64,
    size_bytes: u64,
    first_node: usize,
    last_node: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
struct RuntimeMemoryAlias {
    tensor_index: usize,
    shared_with_tensor_index: usize,
}

#[derive(Debug, Serialize)]
struct Assignment {
    op_index: usize,
    op_name: String,
    provider: String,
    delegated: Option<bool>,
    partition_id: Option<String>,
    mapping_method: &'static str,
    lowering_id: Option<String>,
    kernel_id: Option<String>,
    kernel: Option<String>,
    kernel_source_ref: Option<String>,
    kernel_build_identifier_sha256: Option<String>,
    duration_us: Option<f64>,
    duration_sum_us: Option<f64>,
    sample_count: Option<u64>,
    lowerings: Vec<LoweringObservation>,
    dispatches: Vec<DispatchObservation>,
}

#[derive(Clone, Debug, Serialize)]
struct LoweringObservation {
    lowering_id: String,
    runtime_node_id: Option<usize>,
    observation_count: u64,
}

#[derive(Clone, Debug, Serialize)]
struct DispatchObservation {
    lowering_id: String,
    runtime_node_id: Option<usize>,
    compute_invocation_id: Option<usize>,
    kernel_id: String,
    kernel: String,
    kernel_source_ref: String,
    kernel_build_identifier_sha256: String,
    duration_us: Option<f64>,
    duration_sum_us: Option<f64>,
    sample_count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DispatchKey {
    lowering_id: String,
    runtime_node_id: Option<usize>,
    compute_invocation_id: Option<usize>,
    kernel_id: String,
    kernel: String,
    kernel_source_ref: String,
    kernel_build_identifier_sha256: String,
}

#[derive(Debug)]
struct DispatchAggregate {
    duration_sum_us: Option<f64>,
    sample_count: u64,
}

#[derive(Debug)]
struct OpAggregate {
    op_name: String,
    provider: String,
    delegated: Option<bool>,
    partition_id: Option<String>,
    lowerings: BTreeMap<(String, Option<usize>), u64>,
    dispatches: BTreeMap<DispatchKey, DispatchAggregate>,
    execution_duration_sum_us: Option<f64>,
    execution_sample_count: u64,
    legacy_identity: Option<RuntimeEvent>,
}

struct MemorySnapshotAggregate {
    header: MemorySnapshotEvent,
    allocations: Vec<MemoryAllocationEvent>,
    aliases: Vec<MemoryAliasEvent>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("deepbom-runtime-collector: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        return Err("usage: deepbom-runtime-collector <capture-manifest.json> <runtime-events.ndjson> <output.json>".to_string());
    }
    let manifest_path = Path::new(&args[1]);
    let event_path = Path::new(&args[2]);
    let output_path = Path::new(&args[3]);
    let manifest: CaptureManifest =
        serde_json::from_slice(&read_file(manifest_path, "capture manifest")?)
            .map_err(|error| format!("invalid capture manifest JSON: {error}"))?;
    validate_manifest(&manifest)?;
    let event_bytes = read_file(event_path, "runtime event stream")?;
    let events = parse_events(&event_bytes)?;
    let runtime_memory = aggregate_memory_events(&events, &manifest)?;
    let assignments = aggregate_events(events.op_events, &manifest)?;
    let collector_binary = env::current_exe()
        .map_err(|error| format!("cannot resolve collector executable: {error}"))?;
    let runtime_binary_sha = hash_checked_file(&manifest.runtime.binary_path, "runtime binary")?;
    let artifact_sha = hash_checked_file(&manifest.artifact_path, "artifact")?;
    if artifact_sha != manifest.artifact_sha256 {
        return Err(format!(
            "artifact SHA-256 mismatch: manifest {}, actual {artifact_sha}",
            manifest.artifact_sha256
        ));
    }
    let microkernel_build_sha = hash_checked_file(
        &manifest.build.microkernel_build_identifier_path,
        "microkernel build identifier",
    )?;
    let build_manifest_sha =
        hash_checked_file(&manifest.build.build_manifest_path, "build manifest")?;
    let runtime_options_sha =
        hash_checked_file(&manifest.invocation.runtime_options_path, "runtime options")?;
    let resource_partition = load_resource_partition_observation(
        manifest
            .invocation
            .resource_partition_observation_path
            .as_deref(),
        manifest.invocation.resource_partition_request.as_ref(),
        manifest.invocation.thread_count,
    )?;
    let collector_binary_sha = hash_checked_file(&collector_binary, "collector executable")?;
    let event_stream_sha = sha256_hex(&event_bytes);
    if assignments
        .iter()
        .filter_map(|item| item.kernel_build_identifier_sha256.as_deref())
        .any(|value| value != microkernel_build_sha)
    {
        return Err(
            "an event microkernel build identifier does not match the hashed build-identity file"
                .to_string(),
        );
    }
    let architecture = native_architecture()?;
    let cpu_features = native_cpu_features();
    if cpu_features.is_empty() {
        return Err(format!(
            "native CPU feature detection returned no features for {architecture}"
        ));
    }
    let timing_collected = assignments.iter().any(|item| item.duration_us.is_some());
    let source_kind = match manifest.capture_mode {
        CaptureMode::RuntimeCapture => "deepbom_native_runtime_capture",
        CaptureMode::SyntheticContractProbe => "deepbom_native_runtime_contract_probe",
    };
    let tflite_delegate_build_inventory = build_tflite_delegate_inventory(
        &manifest,
        &artifact_sha,
        &runtime_binary_sha,
        &build_manifest_sha,
    );
    let document = json!({
        "schema": OUTPUT_SCHEMA,
        "artifact_sha256": artifact_sha,
        "target_profile_id": manifest.target_profile_id,
        "target_profile_sha256": manifest.target_profile_sha256,
        "runtime": {
            "name": manifest.runtime.name,
            "version": manifest.runtime.version,
            "backend": manifest.runtime.backend,
            "build": manifest.runtime.build,
            "binary_sha256": runtime_binary_sha,
        },
        "source": {
            "kind": source_kind,
            "validation_scope": match manifest.capture_mode {
                CaptureMode::RuntimeCapture => "runtime_observation",
                CaptureMode::SyntheticContractProbe => "collector_contract_only_not_runtime_evidence",
            },
            "collected_at": manifest.source.collected_at,
            "capture_id": manifest.source.capture_id,
            "capture_binding_semantics": "NATIVE_PROCESS_INVOCATION_IDENTIFIER",
            "assignment_semantics": "original_graph_op_assignment",
            "partition_semantics": "partition_id_identifies_runtime_partition_when_present",
            "dispatch_sample_semantics": "unique_context_function_selection_per_process",
            "duration_semantics": if timing_collected { "per_original_op_exclusive" } else { "not_collected" },
            "duration_statistic": if timing_collected { Some("mean_of_instrumented_original_op_events") } else { None },
            "profile_sha256": event_stream_sha,
            "collector": {
                "schema": COLLECTOR_SCHEMA,
                "name": "deepbom-runtime-collector",
                "version": env!("CARGO_PKG_VERSION"),
                "source_commit": manifest.source.collector_source_commit,
                "binary_sha256": collector_binary_sha,
                "attestation_status": "not_attested",
                "instrumentation": manifest.instrumentation,
            }
        },
        "selector_context": {
            "schema": SELECTOR_SCHEMA,
            "backend_library": "XNNPACK",
            "device": {
                "architecture": architecture,
                "identity": manifest.device.identity,
                "cpu_feature_source": "rust_std_runtime_detection",
                "cpu_features": cpu_features,
            },
            "build": {
                "runtime_binary_sha256": runtime_binary_sha,
                "tensorflow_source_commit": manifest.build.tensorflow_source_commit,
                "xnnpack_source_commit": manifest.build.xnnpack_source_commit,
                "microkernel_build_identifier_sha256": microkernel_build_sha,
                "build_manifest_sha256": build_manifest_sha,
                "compile_definitions": manifest.build.compile_definitions,
            },
            "invocation": {
                "inputs": manifest.invocation.inputs,
                "thread_count": manifest.invocation.thread_count,
                "runtime_options_sha256": runtime_options_sha,
                "resource_partition": resource_partition,
            }
        },
        "tflite_delegate_build_inventory": tflite_delegate_build_inventory,
        "runtime_memory": runtime_memory,
        "assignments": assignments,
    });
    let encoded = serde_json::to_vec_pretty(&document)
        .map_err(|error| format!("cannot encode output: {error}"))?;
    fs::write(output_path, encoded)
        .map_err(|error| format!("cannot write {}: {error}", output_path.display()))?;
    Ok(())
}

fn load_resource_partition_observation(
    path: Option<&Path>,
    request: Option<&ResourcePartitionRequest>,
    thread_count: u32,
) -> Result<Option<serde_json::Value>, String> {
    let (path, request) = match (path, request) {
        (None, None) => return Ok(None),
        (Some(path), Some(request)) => (path, request),
        _ => {
            return Err(
                "resource partition request and observation path must be declared together"
                    .to_string(),
            )
        }
    };
    let bytes = read_file(path, "resource partition observation")?;
    let mut value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid resource partition observation JSON: {error}"))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| "resource partition observation must be an object".to_string())?;
    if object.get("schema").and_then(|item| item.as_str())
        != Some("deepbom.resource_partition_observation.v1")
        || object.get("evidence_class").and_then(|item| item.as_str())
            != Some("OBSERVED_OS_RESOURCE_PARTITION")
    {
        return Err(
            "resource partition observation schema or evidence class is invalid".to_string(),
        );
    }
    let requested = resource_cpu_ids(object, "requested_cpu_ids", false)?;
    if requested != request.requested_cpu_ids
        || object.get("affinity_mode").and_then(|item| item.as_str())
            != Some(request.affinity_mode.as_str())
        || object
            .get("isolation_expectation")
            .and_then(|item| item.as_str())
            != Some(request.isolation_expectation.as_str())
    {
        return Err(
            "resource partition observation does not match the manifest request".to_string(),
        );
    }
    if thread_count as usize > requested.len() {
        return Err(
            "resource partition thread count exceeds requested CPU-set cardinality".to_string(),
        );
    }
    if object.get("affinity_status").and_then(|item| item.as_str())
        != Some("observed_all_sampled_threads_within_requested_set")
    {
        return Err(
            "resource partition observation did not establish sampled-thread affinity".to_string(),
        );
    }
    let declared_sample_count = object
        .get("sample_count")
        .and_then(|item| item.as_u64())
        .unwrap_or(0);
    let declared_maximum_thread_count = object
        .get("maximum_observed_thread_count")
        .and_then(|item| item.as_u64())
        .unwrap_or(0);
    let samples = object
        .get("affinity_samples")
        .and_then(|item| item.as_array())
        .ok_or_else(|| "resource partition observation requires affinity_samples".to_string())?;
    if samples.is_empty() || declared_sample_count != samples.len() as u64 {
        return Err(
            "resource partition sample_count must exactly match a non-empty affinity sample ledger"
                .to_string(),
        );
    }
    let requested_set: BTreeSet<u64> = requested.iter().copied().collect();
    let mut allowed_union = BTreeSet::new();
    let mut observed_processors = BTreeSet::new();
    let mut final_thread_masks = BTreeMap::<u64, Vec<u64>>::new();
    let mut maximum_thread_count = 0_u64;
    let mut previous_time = 0_u64;
    for (sample_index, sample) in samples.iter().enumerate() {
        let sample = sample
            .as_object()
            .ok_or_else(|| "resource partition affinity sample must be an object".to_string())?;
        let observed_at = sample
            .get("observed_at_unix_ms")
            .and_then(|item| item.as_u64())
            .unwrap_or(0);
        if observed_at == 0 || (sample_index > 0 && observed_at < previous_time) {
            return Err(
                "resource partition affinity sample timestamps must be positive and non-decreasing"
                    .to_string(),
            );
        }
        previous_time = observed_at;
        let threads = sample
            .get("threads")
            .and_then(|item| item.as_array())
            .ok_or_else(|| "resource partition affinity sample requires threads".to_string())?;
        if threads.is_empty() {
            return Err("resource partition affinity sample requires observed threads".to_string());
        }
        maximum_thread_count = maximum_thread_count.max(threads.len() as u64);
        let mut previous_tid = 0_u64;
        for (thread_index, row) in threads.iter().enumerate() {
            let row = row
                .as_object()
                .ok_or_else(|| "resource partition sampled thread must be an object".to_string())?;
            let tid = row.get("tid").and_then(|item| item.as_u64()).unwrap_or(0);
            if tid == 0 || (thread_index > 0 && tid <= previous_tid) {
                return Err("resource partition sampled thread IDs must be positive, unique, and ascending per sample".to_string());
            }
            previous_tid = tid;
            let allowed = resource_cpu_ids(row, "allowed_cpu_ids", false)?;
            if allowed.iter().any(|cpu| !requested_set.contains(cpu)) {
                return Err(
                    "resource partition sampled thread mask exceeds the requested CPU set"
                        .to_string(),
                );
            }
            if let Some(processor) = row.get("processor") {
                if !processor.is_null() {
                    let processor = processor.as_u64().ok_or_else(|| {
                        "resource partition sampled processor must be a non-negative integer or null".to_string()
                    })?;
                    if !requested_set.contains(&processor) {
                        return Err(
                            "resource partition sampled processor exceeds the requested CPU set"
                                .to_string(),
                        );
                    }
                    observed_processors.insert(processor);
                }
            }
            allowed_union.extend(allowed.iter().copied());
            final_thread_masks.insert(tid, allowed);
        }
    }
    if maximum_thread_count != declared_maximum_thread_count
        || maximum_thread_count < u64::from(thread_count)
    {
        return Err(
            "resource partition maximum thread count does not reproduce the affinity sample ledger"
                .to_string(),
        );
    }
    let sampled = object
        .get("sampled_threads")
        .and_then(|item| item.as_array())
        .ok_or_else(|| "resource partition observation requires sampled_threads".to_string())?;
    if sampled.len() != final_thread_masks.len() {
        return Err("resource partition sampled-thread summary does not reproduce the affinity sample ledger".to_string());
    }
    for ((expected_tid, expected_mask), row) in final_thread_masks.iter().zip(sampled.iter()) {
        let row = row.as_object().ok_or_else(|| {
            "resource partition sampled thread summary must be an object".to_string()
        })?;
        let tid = row.get("tid").and_then(|item| item.as_u64()).unwrap_or(0);
        let mask = resource_cpu_ids(row, "allowed_cpu_ids", false)?;
        if tid != *expected_tid || mask != *expected_mask {
            return Err("resource partition sampled-thread summary does not reproduce the affinity sample ledger".to_string());
        }
    }
    let declared_union = resource_cpu_ids(object, "observed_allowed_cpu_ids_union", false)?;
    if allowed_union.iter().copied().collect::<Vec<_>>() != declared_union
        || declared_union != requested
    {
        return Err("resource partition sampled thread-mask union must exactly reproduce the requested CPU set".to_string());
    }
    let online = resource_cpu_ids(object, "online_cpu_ids", false)?;
    if requested.iter().any(|cpu| !online.contains(cpu)) {
        return Err(
            "resource partition requested CPU is absent from the observed online CPU set"
                .to_string(),
        );
    }
    let processors = resource_cpu_ids(object, "observed_processor_ids", true)?;
    if processors != observed_processors.iter().copied().collect::<Vec<_>>() {
        return Err(
            "resource partition processor summary does not reproduce the affinity sample ledger"
                .to_string(),
        );
    }
    let expectation = object
        .get("isolation_expectation")
        .and_then(|item| item.as_str())
        .ok_or_else(|| {
            "resource partition observation requires isolation_expectation".to_string()
        })?;
    if !["affinity_only", "exclusive_cpuset"].contains(&expectation) {
        return Err("resource partition isolation expectation is unsupported".to_string());
    }
    let isolation_status = object
        .get("exclusive_isolation_status")
        .and_then(|item| item.as_str())
        .ok_or_else(|| {
            "resource partition observation requires exclusive_isolation_status".to_string()
        })?;
    if ![
        "observed_cgroup_v2_isolated_partition",
        "not_observed_affinity_only",
    ]
    .contains(&isolation_status)
    {
        return Err("resource partition exclusive isolation status is unsupported".to_string());
    }
    if expectation == "exclusive_cpuset"
        && isolation_status != "observed_cgroup_v2_isolated_partition"
    {
        return Err("resource partition requested exclusive cpuset was not observed".to_string());
    }
    if isolation_status == "observed_cgroup_v2_isolated_partition" {
        let effective = resource_cpu_ids(object, "observed_effective_cpu_ids", false)?;
        if object
            .get("cgroup_v2_partition_state")
            .and_then(|item| item.as_str())
            != Some("isolated")
            || effective != requested
        {
            return Err("resource partition isolated status requires an exact isolated cgroup v2 effective CPU set".to_string());
        }
    }
    object.remove("affinity_samples");
    object.insert(
        "observation_sha256".to_string(),
        serde_json::Value::String(sha256_hex(&bytes)),
    );
    Ok(Some(value))
}

fn resource_cpu_ids(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    allow_empty: bool,
) -> Result<Vec<u64>, String> {
    let values = object
        .get(field)
        .and_then(|item| item.as_array())
        .ok_or_else(|| format!("resource partition observation requires {field}"))?;
    let ids: Vec<u64> = values
        .iter()
        .map(|item| item.as_u64())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            format!("resource partition {field} must contain non-negative integer CPU IDs")
        })?;
    if (!allow_empty && ids.is_empty())
        || ids.len() > 4096
        || ids.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(format!(
            "resource partition {field} must be {}unique and ascending",
            if allow_empty { "" } else { "non-empty, " }
        ));
    }
    Ok(ids)
}

fn build_tflite_delegate_inventory(
    manifest: &CaptureManifest,
    artifact_sha256: &str,
    runtime_binary_sha256: &str,
    build_manifest_sha256: &str,
) -> serde_json::Value {
    let gpu_enabled =
        declared_build_boolean(&manifest.build.compile_definitions, "TFLITE_ENABLE_GPU");
    let nnapi_enabled =
        declared_build_boolean(&manifest.build.compile_definitions, "TFLITE_ENABLE_NNAPI");
    let cmake_system = manifest.build.cmake_system_name.as_deref();
    let gpu_status = match gpu_enabled {
        Some(true) => "enabled_by_declared_cmake_option",
        Some(false) => "disabled_by_declared_cmake_option",
        None => "not_declared",
    };
    let nnapi_status = match (nnapi_enabled, cmake_system) {
        (Some(false), _) => "disabled_by_declared_cmake_option",
        (Some(true), Some(system)) if system.eq_ignore_ascii_case("Android") => {
            "enabled_by_declared_cmake_option_and_android_gate"
        }
        (Some(true), Some(_)) => "disabled_by_non_android_cmake_gate",
        (Some(true), None) => "unresolved_cmake_system_name",
        (None, _) => "not_declared",
    };
    let gpu_flags = manifest.invocation.tflite_gpu_experimental_flags;
    let quantized_model_flag_status = match gpu_flags {
        Some(flags) if flags & 1 == 1 => "enabled_by_declared_runtime_option",
        Some(_) => "disabled_by_declared_runtime_option",
        None => "not_declared",
    };
    json!({
        "schema": TFLITE_DELEGATE_BUILD_SCHEMA,
        "evidence_class": "DECLARED_BUILD_AND_RUNTIME_OPTION_INVENTORY",
        "artifact_sha256": artifact_sha256,
        "tensorflow_source_commit": manifest.build.tensorflow_source_commit,
        "runtime_binary_sha256": runtime_binary_sha256,
        "build_manifest_sha256": build_manifest_sha256,
        "cmake_system_name": cmake_system,
        "build_options": [
            {
                "name": "TFLITE_ENABLE_GPU",
                "declared_value": declared_build_value(&manifest.build.compile_definitions, "TFLITE_ENABLE_GPU"),
                "normalized_enabled": gpu_enabled,
                "effective_status": gpu_status,
            },
            {
                "name": "TFLITE_ENABLE_NNAPI",
                "declared_value": declared_build_value(&manifest.build.compile_definitions, "TFLITE_ENABLE_NNAPI"),
                "normalized_enabled": nnapi_enabled,
                "effective_status": nnapi_status,
            }
        ],
        "gpu": {
            "compiled_status": gpu_status,
            "experimental_flags": gpu_flags,
            "quantized_model_flag_bit": 1,
            "quantized_model_flag_status": quantized_model_flag_status,
            "max_delegated_partitions": manifest.invocation.tflite_gpu_max_delegated_partitions,
            "option_source": "capture runtime-options.json",
        },
        "nnapi": {
            "compiled_status": nnapi_status,
            "runtime_feature_level": manifest.device.nnapi_runtime_feature_level,
            "accelerator_identity": manifest.device.nnapi_accelerator_identity,
            "capability_source": manifest.device.nnapi_capability_source.as_deref().unwrap_or("not_collected"),
        },
        "source_files": [
            {
                "id": "tflite_cmake_build_options",
                "source_ref": format!("https://github.com/tensorflow/tensorflow/blob/{TENSORFLOW_SOURCE_COMMIT}/tensorflow/lite/CMakeLists.txt"),
                "sha256": TFLITE_CMAKE_SOURCE_SHA256,
            },
            {
                "id": "tflite_gpu_delegate_options",
                "source_ref": format!("https://github.com/tensorflow/tensorflow/blob/{TENSORFLOW_SOURCE_COMMIT}/tensorflow/lite/delegates/gpu/delegate_options.h"),
                "sha256": TFLITE_GPU_OPTIONS_SOURCE_SHA256,
            }
        ],
        "interpretation_boundary": "Build options and delegate runtime options are bound to the hashed runtime binary and build manifest. They establish declared selected-build prerequisites only; successful delegate initialization, device acceptance, partition assignment, and execution remain runtime observations.",
    })
}

fn declared_build_value<'a>(definitions: &'a [CompileDefinition], name: &str) -> Option<&'a str> {
    definitions
        .iter()
        .find(|item| item.name == name)
        .map(|item| item.value.as_str())
}

fn declared_build_boolean(definitions: &[CompileDefinition], name: &str) -> Option<bool> {
    declared_build_value(definitions, name).and_then(|value| {
        match value.to_ascii_uppercase().as_str() {
            "1" | "ON" | "TRUE" => Some(true),
            "0" | "OFF" | "FALSE" => Some(false),
            _ => None,
        }
    })
}

fn validate_manifest(manifest: &CaptureManifest) -> Result<(), String> {
    if manifest.schema != MANIFEST_SCHEMA {
        return Err(format!("manifest schema must be {MANIFEST_SCHEMA}"));
    }
    require_sha(&manifest.artifact_sha256, "artifact_sha256", 64)?;
    require_sha(&manifest.target_profile_sha256, "target_profile_sha256", 64)?;
    require_sha(
        &manifest.build.tensorflow_source_commit,
        "tensorflow_source_commit",
        40,
    )?;
    if manifest.build.tensorflow_source_commit != TENSORFLOW_SOURCE_COMMIT {
        return Err(format!(
            "tensorflow_source_commit must match the pinned {TENSORFLOW_SOURCE_COMMIT}"
        ));
    }
    require_sha(
        &manifest.build.xnnpack_source_commit,
        "xnnpack_source_commit",
        40,
    )?;
    if manifest.target_profile_id.trim().is_empty() || manifest.device.identity.trim().is_empty() {
        return Err("target profile ID and device identity are required".to_string());
    }
    if manifest
        .build
        .cmake_system_name
        .as_ref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err("cmake_system_name must be non-empty when declared".to_string());
    }
    if manifest
        .device
        .nnapi_runtime_feature_level
        .is_some_and(|value| value < 27)
    {
        return Err("nnapi_runtime_feature_level must be at least 27".to_string());
    }
    if (manifest.device.nnapi_runtime_feature_level.is_some()
        || manifest.device.nnapi_accelerator_identity.is_some())
        && manifest.device.nnapi_capability_source.is_none()
    {
        return Err(
            "NNAPI feature or accelerator identity requires nnapi_capability_source".to_string(),
        );
    }
    if let Some(source) = manifest.device.nnapi_capability_source.as_deref() {
        if ![
            "android_nnapi_runtime_query",
            "declared_capture_configuration",
        ]
        .contains(&source)
        {
            return Err("nnapi_capability_source is unsupported".to_string());
        }
    }
    if manifest
        .device
        .nnapi_accelerator_identity
        .as_ref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err("nnapi_accelerator_identity must be non-empty when declared".to_string());
    }
    if manifest
        .invocation
        .tflite_gpu_max_delegated_partitions
        .is_some_and(|value| value == 0)
    {
        return Err("tflite_gpu_max_delegated_partitions must be positive".to_string());
    }
    if !manifest
        .runtime
        .backend
        .to_ascii_lowercase()
        .contains("xnnpack")
    {
        return Err("runtime.backend must identify the instrumented XNNPACK path".to_string());
    }
    if manifest.capture_mode == CaptureMode::RuntimeCapture
        && manifest
            .runtime
            .name
            .to_ascii_lowercase()
            .contains("contract probe")
    {
        return Err(
            "the synthetic contract probe cannot produce runtime_capture evidence".to_string(),
        );
    }
    if manifest.source.capture_id.trim().is_empty()
        || !manifest.source.collected_at.contains('T')
        || !(manifest.source.collected_at.ends_with('Z')
            || manifest.source.collected_at.contains('+'))
    {
        return Err(
            "source capture ID and timezone-qualified collected_at are required".to_string(),
        );
    }
    if manifest.source.collector_source_commit.trim().is_empty() {
        return Err("collector_source_commit is required".to_string());
    }
    if manifest.invocation.inputs.is_empty()
        || manifest.invocation.thread_count == 0
        || manifest.invocation.inputs.iter().any(|input| {
            input.name.is_empty() || input.shape.is_empty() || input.shape.contains(&0)
        })
    {
        return Err(
            "invocation requires named positive input shapes and a positive thread count"
                .to_string(),
        );
    }
    match (
        manifest.invocation.resource_partition_request.as_ref(),
        manifest
            .invocation
            .resource_partition_observation_path
            .as_ref(),
    ) {
        (None, None) => {}
        (Some(request), Some(_)) => {
            if request.affinity_mode != "taskset_process_and_descendants"
                || !["affinity_only", "exclusive_cpuset"]
                    .contains(&request.isolation_expectation.as_str())
                || request.requested_cpu_ids.is_empty()
                || request.requested_cpu_ids.len() > 4096
                || request
                    .requested_cpu_ids
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
                || manifest.invocation.thread_count as usize > request.requested_cpu_ids.len()
            {
                return Err("resource partition request is invalid or oversubscribed".to_string());
            }
        }
        _ => {
            return Err(
                "resource partition request and observation path must be declared together"
                    .to_string(),
            )
        }
    }
    let names: Vec<&str> = manifest
        .build
        .compile_definitions
        .iter()
        .map(|item| item.name.as_str())
        .collect();
    if names.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(
            "compile definitions must have unique lexicographically sorted names".to_string(),
        );
    }
    for required in ["XNN_BUILD_ALL_MICROKERNELS", "XNN_ENABLE_ASSEMBLY"] {
        let value = manifest
            .build
            .compile_definitions
            .iter()
            .find(|item| item.name == required)
            .ok_or_else(|| format!("compile definitions must include {required}"))?;
        if !["0", "1", "OFF", "ON", "FALSE", "TRUE"]
            .contains(&value.value.to_ascii_uppercase().as_str())
        {
            return Err(format!("{required} must carry an explicit boolean value"));
        }
    }
    for optional in ["TFLITE_ENABLE_GPU", "TFLITE_ENABLE_NNAPI"] {
        if let Some(value) = manifest
            .build
            .compile_definitions
            .iter()
            .find(|item| item.name == optional)
        {
            if !["0", "1", "OFF", "ON", "FALSE", "TRUE"]
                .contains(&value.value.to_ascii_uppercase().as_str())
            {
                return Err(format!("{optional} must carry an explicit boolean value"));
            }
        }
    }
    Ok(())
}

fn parse_events(bytes: &[u8]) -> Result<ParsedEvents, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("runtime event stream is not UTF-8: {error}"))?;
    let mut events = ParsedEvents::default();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .map_err(|error| format!("invalid event line {}: {error}", index + 1))?;
        let event_kind = value
            .get("event_kind")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("observation");
        match event_kind {
            "memory_snapshot" => {
                let event: MemorySnapshotEvent =
                    serde_json::from_value(value).map_err(|error| {
                        format!("invalid memory snapshot line {}: {error}", index + 1)
                    })?;
                events.memory_snapshots.push(event);
                continue;
            }
            "memory_allocation" => {
                let event: MemoryAllocationEvent =
                    serde_json::from_value(value).map_err(|error| {
                        format!("invalid memory allocation line {}: {error}", index + 1)
                    })?;
                events.memory_allocations.push(event);
                continue;
            }
            "memory_alias" => {
                let event: MemoryAliasEvent = serde_json::from_value(value)
                    .map_err(|error| format!("invalid memory alias line {}: {error}", index + 1))?;
                events.memory_aliases.push(event);
                continue;
            }
            _ => {}
        }
        let event: RuntimeEvent = serde_json::from_value(value)
            .map_err(|error| format!("invalid op event line {}: {error}", index + 1))?;
        if event.op_name.trim().is_empty() || event.provider.trim().is_empty() {
            return Err(format!(
                "event line {} requires op_name and provider",
                index + 1
            ));
        }
        if event
            .duration_us
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(format!(
                "event line {} duration_us must be finite and non-negative",
                index + 1
            ));
        }
        events.op_events.push(event);
    }
    if events.op_events.is_empty() {
        return Err("runtime event stream contains no original-op events".to_string());
    }
    Ok(events)
}

fn aggregate_memory_events(
    events: &ParsedEvents,
    manifest: &CaptureManifest,
) -> Result<Option<RuntimeMemoryEvidence>, String> {
    let memory_event_count = events.memory_snapshots.len()
        + events.memory_allocations.len()
        + events.memory_aliases.len();
    if memory_event_count == 0 {
        if manifest.instrumentation.arena_allocations {
            return Err(
                "arena allocation instrumentation was declared but emitted no memory snapshot"
                    .to_string(),
            );
        }
        return Ok(None);
    }
    if !manifest.instrumentation.arena_allocations {
        return Err(
            "memory events exceed the declared arena allocation instrumentation capability"
                .to_string(),
        );
    }

    let mut groups = BTreeMap::<u64, MemorySnapshotAggregate>::new();
    for header in &events.memory_snapshots {
        if header.event_kind != "memory_snapshot" {
            return Err("memory snapshot event_kind is invalid".to_string());
        }
        if header.tensor_count == 0 || header.execution_node_count == 0 {
            return Err(format!(
                "memory snapshot {} requires positive tensor and execution-node counts",
                header.memory_snapshot_id
            ));
        }
        let snapshot_id = header.memory_snapshot_id;
        if groups
            .insert(
                snapshot_id,
                MemorySnapshotAggregate {
                    header: header.clone(),
                    allocations: Vec::new(),
                    aliases: Vec::new(),
                },
            )
            .is_some()
        {
            return Err(format!("duplicate memory snapshot {snapshot_id}"));
        }
    }
    for allocation in &events.memory_allocations {
        if allocation.event_kind != "memory_allocation" {
            return Err("memory allocation event_kind is invalid".to_string());
        }
        groups
            .get_mut(&allocation.memory_snapshot_id)
            .ok_or_else(|| {
                format!(
                    "memory allocation references missing snapshot {}",
                    allocation.memory_snapshot_id
                )
            })?
            .allocations
            .push(allocation.clone());
    }
    for alias in &events.memory_aliases {
        if alias.event_kind != "memory_alias" {
            return Err("memory alias event_kind is invalid".to_string());
        }
        groups
            .get_mut(&alias.memory_snapshot_id)
            .ok_or_else(|| {
                format!(
                    "memory alias references missing snapshot {}",
                    alias.memory_snapshot_id
                )
            })?
            .aliases
            .push(alias.clone());
    }
    if groups.is_empty() {
        return Err("memory allocation stream has no snapshot header".to_string());
    }

    let mut snapshots = Vec::with_capacity(groups.len());
    for (expected_id, (snapshot_id, mut group)) in groups.into_iter().enumerate() {
        if snapshot_id != expected_id as u64 {
            return Err(format!(
                "memory snapshot IDs must be contiguous from zero; expected {expected_id}, received {snapshot_id}"
            ));
        }
        if group.allocations.len() != group.header.allocation_count
            || group.aliases.len() != group.header.alias_count
        {
            return Err(format!(
                "memory snapshot {snapshot_id} declared {}/{} allocation/alias rows but emitted {}/{}",
                group.header.allocation_count,
                group.header.alias_count,
                group.allocations.len(),
                group.aliases.len()
            ));
        }
        group
            .allocations
            .sort_by_key(|allocation| allocation.tensor_index);
        group.aliases.sort_by_key(|alias| alias.tensor_index);
        let allocation_tensors = group
            .allocations
            .iter()
            .map(|allocation| allocation.tensor_index)
            .collect::<BTreeSet<_>>();
        let alias_tensors = group
            .aliases
            .iter()
            .map(|alias| alias.tensor_index)
            .collect::<BTreeSet<_>>();
        if allocation_tensors.len() != group.allocations.len()
            || alias_tensors.len() != group.aliases.len()
            || !allocation_tensors.is_disjoint(&alias_tensors)
        {
            return Err(format!(
                "memory snapshot {snapshot_id} has duplicate or owning-plus-alias tensor rows"
            ));
        }

        let mut allocations = Vec::with_capacity(group.allocations.len());
        let mut allocated_interval_bytes = 0_u64;
        for allocation in group.allocations {
            if allocation.tensor_index >= group.header.tensor_count
                || allocation.size_bytes == 0
                || allocation.first_node < 0
                || allocation.last_node < allocation.first_node
            {
                return Err(format!(
                    "memory snapshot {snapshot_id} allocation T{} has invalid tensor, size, or lifetime",
                    allocation.tensor_index
                ));
            }
            let arena_limit = match allocation.arena.as_str() {
                "kTfLiteArenaRw" => group.header.non_persistent_arena_bytes,
                "kTfLiteArenaRwPersistent" => group.header.persistent_arena_bytes,
                _ => {
                    return Err(format!(
                        "memory snapshot {snapshot_id} allocation T{} has unknown arena {}",
                        allocation.tensor_index, allocation.arena
                    ));
                }
            };
            let allocation_end = allocation
                .offset_bytes
                .checked_add(allocation.size_bytes)
                .ok_or_else(|| {
                    format!(
                        "memory snapshot {snapshot_id} allocation T{} byte range overflows",
                        allocation.tensor_index
                    )
                })?;
            if allocation_end > arena_limit {
                return Err(format!(
                    "memory snapshot {snapshot_id} allocation T{} exceeds {}",
                    allocation.tensor_index, allocation.arena
                ));
            }
            let first_node = allocation.first_node as usize;
            let last_node = if allocation.last_node == i32::MAX {
                None
            } else {
                Some(allocation.last_node as usize)
            };
            if first_node >= group.header.execution_node_count
                || last_node.is_some_and(|last| last >= group.header.execution_node_count)
            {
                return Err(format!(
                    "memory snapshot {snapshot_id} allocation T{} lifetime exceeds the execution plan",
                    allocation.tensor_index
                ));
            }
            allocated_interval_bytes = allocated_interval_bytes
                .checked_add(allocation.size_bytes)
                .ok_or_else(|| {
                    format!("memory snapshot {snapshot_id} allocation byte total overflows")
                })?;
            allocations.push(RuntimeMemoryAllocation {
                tensor_index: allocation.tensor_index,
                arena: allocation.arena,
                offset_bytes: allocation.offset_bytes,
                size_bytes: allocation.size_bytes,
                first_node,
                last_node,
            });
        }
        validate_memory_allocation_overlap(snapshot_id, &allocations)?;

        let mut aliases = Vec::with_capacity(group.aliases.len());
        for alias in group.aliases {
            if alias.tensor_index >= group.header.tensor_count
                || alias.shared_with_tensor_index >= group.header.tensor_count
                || alias.tensor_index == alias.shared_with_tensor_index
                || !allocation_tensors.contains(&alias.shared_with_tensor_index)
                || alias_tensors.contains(&alias.shared_with_tensor_index)
            {
                return Err(format!(
                    "memory snapshot {snapshot_id} alias T{} -> T{} is not rooted in one owning allocation",
                    alias.tensor_index, alias.shared_with_tensor_index
                ));
            }
            aliases.push(RuntimeMemoryAlias {
                tensor_index: alias.tensor_index,
                shared_with_tensor_index: alias.shared_with_tensor_index,
            });
        }
        let combined_arena_bytes = group
            .header
            .non_persistent_arena_bytes
            .checked_add(group.header.persistent_arena_bytes)
            .ok_or_else(|| format!("memory snapshot {snapshot_id} arena total overflows"))?;
        snapshots.push(RuntimeMemorySnapshot {
            memory_snapshot_id: snapshot_id,
            non_persistent_arena_bytes: group.header.non_persistent_arena_bytes,
            persistent_arena_bytes: group.header.persistent_arena_bytes,
            combined_arena_bytes,
            tensor_count: group.header.tensor_count,
            execution_node_count: group.header.execution_node_count,
            allocation_count: allocations.len(),
            alias_count: aliases.len(),
            allocated_interval_bytes,
            allocations,
            aliases,
        });
    }

    let final_snapshot = snapshots
        .last()
        .ok_or_else(|| "runtime memory snapshot aggregation is empty".to_string())?;
    let final_non_persistent_arena_bytes = final_snapshot.non_persistent_arena_bytes;
    let final_persistent_arena_bytes = final_snapshot.persistent_arena_bytes;
    let final_combined_arena_bytes = final_snapshot.combined_arena_bytes;
    let peak_non_persistent_arena_bytes = snapshots
        .iter()
        .map(|snapshot| snapshot.non_persistent_arena_bytes)
        .max()
        .unwrap_or(0);
    let peak_persistent_arena_bytes = snapshots
        .iter()
        .map(|snapshot| snapshot.persistent_arena_bytes)
        .max()
        .unwrap_or(0);
    let peak_combined_arena_bytes = snapshots
        .iter()
        .map(|snapshot| snapshot.combined_arena_bytes)
        .max()
        .unwrap_or(0);
    let allocation_ledger_sha256 = sha256_hex(
        &serde_json::to_vec(&snapshots)
            .map_err(|error| format!("cannot encode runtime memory ledger: {error}"))?,
    );
    Ok(Some(RuntimeMemoryEvidence {
        schema: RUNTIME_MEMORY_SCHEMA,
        status: "assessed",
        evidence_class: "OBSERVED_RUNTIME",
        tensorflow_source_commit: manifest.build.tensorflow_source_commit.clone(),
        snapshot_count: snapshots.len(),
        peak_non_persistent_arena_bytes,
        peak_persistent_arena_bytes,
        peak_combined_arena_bytes,
        final_non_persistent_arena_bytes,
        final_persistent_arena_bytes,
        final_combined_arena_bytes,
        allocation_ledger_sha256,
        snapshots,
        method: "Instrumented post-commit ArenaPlanner snapshots with complete owning-allocation and in-place-alias ledgers; collector independently validates byte bounds, counts, roots, lifetimes, and overlapping live ranges.",
        interpretation_boundary: "Observed TFLite arena buffers for the captured build and invocation. Delegate-owned buffers, kernel scratch outside TfLite arenas, allocator metadata/alignment slack outside GetBufferSize(), thread stacks, code, constants, and process RSS are excluded.",
    }))
}

fn validate_memory_allocation_overlap(
    snapshot_id: u64,
    allocations: &[RuntimeMemoryAllocation],
) -> Result<(), String> {
    for (index, left) in allocations.iter().enumerate() {
        for right in allocations.iter().skip(index + 1) {
            if left.arena != right.arena {
                continue;
            }
            let left_last = left.last_node.unwrap_or(usize::MAX);
            let right_last = right.last_node.unwrap_or(usize::MAX);
            const fn ranges_overlap(start_a: u64, size_a: u64, start_b: u64, size_b: u64) -> bool {
                start_a < start_b.saturating_add(size_b) && start_b < start_a.saturating_add(size_a)
            }
            const fn lifetimes_overlap(
                first_a: usize,
                last_a: usize,
                first_b: usize,
                last_b: usize,
            ) -> bool {
                first_a <= last_b && first_b <= last_a
            }
            if lifetimes_overlap(left.first_node, left_last, right.first_node, right_last)
                && ranges_overlap(
                    left.offset_bytes,
                    left.size_bytes,
                    right.offset_bytes,
                    right.size_bytes,
                )
            {
                return Err(format!(
                    "memory snapshot {snapshot_id} has overlapping live allocations T{} and T{} in {}",
                    left.tensor_index, right.tensor_index, left.arena
                ));
            }
        }
    }
    Ok(())
}

fn aggregate_events(
    events: Vec<RuntimeEvent>,
    manifest: &CaptureManifest,
) -> Result<Vec<Assignment>, String> {
    let mut groups: BTreeMap<usize, OpAggregate> = BTreeMap::new();
    for event in events {
        validate_selector_event(&event, manifest)?;
        let entry = groups.entry(event.op_index).or_insert_with(|| OpAggregate {
            op_name: event.op_name.clone(),
            provider: event.provider.clone(),
            delegated: event.delegated,
            partition_id: event.partition_id.clone(),
            lowerings: BTreeMap::new(),
            dispatches: BTreeMap::new(),
            execution_duration_sum_us: None,
            execution_sample_count: 0,
            legacy_identity: None,
        });
        if entry.op_name != event.op_name
            || entry.provider != event.provider
            || entry.delegated != event.delegated
            || entry.partition_id != event.partition_id
        {
            return Err(format!(
                "conflicting placement identity for original op #{}",
                event.op_index
            ));
        }
        match event.event_kind {
            EventKind::Placement => {}
            EventKind::Lowering => {
                let key = (
                    event.lowering_id.clone().unwrap_or_default(),
                    event.runtime_node_id,
                );
                *entry.lowerings.entry(key).or_insert(0) += 1;
            }
            EventKind::Dispatch => {
                let key = DispatchKey::from_event(&event);
                let dispatch =
                    entry
                        .dispatches
                        .entry(key.clone())
                        .or_insert_with(|| DispatchAggregate {
                            duration_sum_us: event.duration_us.map(|_| 0.0),
                            sample_count: 0,
                        });
                if dispatch.duration_sum_us.is_some() != event.duration_us.is_some() {
                    return Err(format!(
                        "mixed timed and untimed dispatches for original op #{} kernel {}",
                        event.op_index, key.kernel
                    ));
                }
                if let Some(duration) = event.duration_us {
                    dispatch.duration_sum_us =
                        Some(dispatch.duration_sum_us.unwrap_or(0.0) + duration);
                }
                dispatch.sample_count += 1;
                *entry
                    .lowerings
                    .entry((key.lowering_id, key.runtime_node_id))
                    .or_insert(0) += 1;
            }
            EventKind::Execution => {
                if entry.execution_duration_sum_us.is_none() && entry.execution_sample_count > 0 {
                    return Err(format!(
                        "mixed timed and untimed execution events for original op #{}",
                        event.op_index
                    ));
                }
                entry.execution_duration_sum_us = Some(
                    entry.execution_duration_sum_us.unwrap_or(0.0)
                        + event.duration_us.unwrap_or(0.0),
                );
                entry.execution_sample_count += 1;
            }
            EventKind::Observation => {
                let static_identity = RuntimeEvent {
                    duration_us: None,
                    ..event.clone()
                };
                if let Some(identity) = &entry.legacy_identity {
                    if identity != &static_identity {
                        return Err(format!(
                            "conflicting provider/selector identity for original op #{}",
                            event.op_index
                        ));
                    }
                } else {
                    entry.legacy_identity = Some(static_identity);
                    entry.execution_duration_sum_us = event.duration_us.map(|_| 0.0);
                }
                if entry.execution_duration_sum_us.is_some() != event.duration_us.is_some() {
                    return Err(format!(
                        "mixed timed and untimed events for original op #{}",
                        event.op_index
                    ));
                }
                if let Some(duration) = event.duration_us {
                    entry.execution_duration_sum_us =
                        Some(entry.execution_duration_sum_us.unwrap_or(0.0) + duration);
                }
                entry.execution_sample_count += 1;
                if let Some(lowering) = &event.lowering_id {
                    *entry
                        .lowerings
                        .entry((lowering.clone(), event.runtime_node_id))
                        .or_insert(0) += 1;
                }
                if event.kernel.is_some() {
                    let key = DispatchKey::from_event(&event);
                    entry.dispatches.entry(key).or_insert(DispatchAggregate {
                        duration_sum_us: None,
                        sample_count: 1,
                    });
                }
            }
        }
    }
    Ok(groups
        .into_iter()
        .map(|(op_index, entry)| {
            let lowerings = entry
                .lowerings
                .into_iter()
                .map(
                    |((lowering_id, runtime_node_id), observation_count)| LoweringObservation {
                        lowering_id,
                        runtime_node_id,
                        observation_count,
                    },
                )
                .collect::<Vec<_>>();
            let dispatches = entry
                .dispatches
                .into_iter()
                .map(|(key, aggregate)| DispatchObservation {
                    lowering_id: key.lowering_id,
                    runtime_node_id: key.runtime_node_id,
                    compute_invocation_id: key.compute_invocation_id,
                    kernel_id: key.kernel_id,
                    kernel: key.kernel,
                    kernel_source_ref: key.kernel_source_ref,
                    kernel_build_identifier_sha256: key.kernel_build_identifier_sha256,
                    duration_us: aggregate
                        .duration_sum_us
                        .map(|sum| sum / aggregate.sample_count as f64),
                    duration_sum_us: aggregate.duration_sum_us,
                    sample_count: aggregate.sample_count,
                })
                .collect::<Vec<_>>();
            let unique_lowering = if lowerings.len() == 1 {
                Some(lowerings[0].lowering_id.clone())
            } else {
                None
            };
            let unique_dispatch = match (unique_lowering.as_deref(), dispatches.as_slice()) {
                (Some(lowering), [dispatch]) if dispatch.lowering_id == lowering => Some(dispatch),
                _ => None,
            };
            let average = entry
                .execution_duration_sum_us
                .map(|sum| sum / entry.execution_sample_count as f64);
            Assignment {
                op_index,
                op_name: entry.op_name,
                provider: entry.provider,
                delegated: entry.delegated,
                partition_id: entry.partition_id,
                mapping_method: "native_runtime_original_op_instrumentation",
                lowering_id: unique_lowering,
                kernel_id: unique_dispatch.map(|item| item.kernel_id.clone()),
                kernel: unique_dispatch.map(|item| item.kernel.clone()),
                kernel_source_ref: unique_dispatch.map(|item| item.kernel_source_ref.clone()),
                kernel_build_identifier_sha256: unique_dispatch
                    .map(|item| item.kernel_build_identifier_sha256.clone()),
                duration_us: average,
                duration_sum_us: entry.execution_duration_sum_us,
                sample_count: average.map(|_| entry.execution_sample_count),
                lowerings,
                dispatches,
            }
        })
        .collect())
}

impl DispatchKey {
    fn from_event(event: &RuntimeEvent) -> Self {
        Self {
            lowering_id: event.lowering_id.clone().unwrap_or_default(),
            runtime_node_id: event.runtime_node_id,
            compute_invocation_id: event.compute_invocation_id,
            kernel_id: event.kernel_id.clone().unwrap_or_default(),
            kernel: event.kernel.clone().unwrap_or_default(),
            kernel_source_ref: event.kernel_source_ref.clone().unwrap_or_default(),
            kernel_build_identifier_sha256: event
                .kernel_build_identifier_sha256
                .clone()
                .unwrap_or_default(),
        }
    }
}

fn validate_selector_event(event: &RuntimeEvent, manifest: &CaptureManifest) -> Result<(), String> {
    let kernel_fields = [
        &event.kernel_id,
        &event.kernel,
        &event.kernel_source_ref,
        &event.kernel_build_identifier_sha256,
    ];
    let kernel_count = kernel_fields.iter().filter(|value| value.is_some()).count();
    if kernel_count != 0 && kernel_count != kernel_fields.len() {
        return Err(format!(
            "op #{} must emit all microkernel identity fields together",
            event.op_index
        ));
    }
    if event.lowering_id.is_some() && !manifest.instrumentation.lowering_ids {
        return Err(format!(
            "op #{} emits lowering evidence beyond declared instrumentation",
            event.op_index
        ));
    }
    match event.event_kind {
        EventKind::Placement
            if event.lowering_id.is_some()
                || kernel_count > 0
                || event.duration_us.is_some()
                || event.runtime_node_id.is_some()
                || event.compute_invocation_id.is_some() =>
        {
            return Err(format!(
                "op #{} placement event carries lowering, dispatch, timing, or runtime-node fields",
                event.op_index
            ));
        }
        EventKind::Lowering
            if event.lowering_id.is_none()
                || kernel_count > 0
                || event.duration_us.is_some()
                || event.runtime_node_id.is_none()
                || event.compute_invocation_id.is_some() =>
        {
            return Err(format!("op #{} lowering event requires lowering_id/runtime_node_id and no dispatch or timing fields", event.op_index));
        }
        EventKind::Dispatch
            if event.lowering_id.is_none()
                || kernel_count != 4
                || event.runtime_node_id.is_none()
                || event.compute_invocation_id.is_none() =>
        {
            return Err(format!("op #{} dispatch event requires lowering, runtime node, compute invocation, and complete kernel identity", event.op_index));
        }
        EventKind::Execution
            if event.duration_us.is_none()
                || event.lowering_id.is_some()
                || kernel_count > 0
                || event.runtime_node_id.is_some()
                || event.compute_invocation_id.is_some() =>
        {
            return Err(format!(
                "op #{} execution event requires only an exclusive duration and placement identity",
                event.op_index
            ));
        }
        _ => {}
    }
    if kernel_count > 0 {
        if !manifest.instrumentation.microkernel_ids || event.lowering_id.is_none() {
            return Err(format!("op #{} microkernel evidence requires declared microkernel instrumentation and a lowering ID", event.op_index));
        }
        let kernel = event.kernel.as_deref().unwrap_or_default();
        if !kernel.starts_with("xnn_") || !kernel.contains("ukernel") {
            return Err(format!(
                "op #{} kernel is not a concrete XNNPACK microkernel symbol",
                event.op_index
            ));
        }
        let source_prefix = format!(
            "google/XNNPACK@{}/src/",
            manifest.build.xnnpack_source_commit
        );
        if !event
            .kernel_source_ref
            .as_deref()
            .unwrap_or_default()
            .starts_with(&source_prefix)
        {
            return Err(format!(
                "op #{} kernel source is not pinned to the manifest XNNPACK commit",
                event.op_index
            ));
        }
        require_sha(
            event
                .kernel_build_identifier_sha256
                .as_deref()
                .unwrap_or_default(),
            "event kernel_build_identifier_sha256",
            64,
        )?;
    }
    if matches!(event.event_kind, EventKind::Lowering | EventKind::Dispatch)
        && matches!(
            event.lowering_id.as_deref(),
            Some("<unknown>" | "<invalid-node-type>")
        )
    {
        return Err(format!(
            "op #{} carries an unresolved XNNPACK lowering identifier",
            event.op_index
        ));
    }
    Ok(())
}

fn read_file(path: &Path, label: &str) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|error| format!("cannot read {label} {}: {error}", path.display()))
}

fn hash_checked_file(path: &Path, label: &str) -> Result<String, String> {
    Ok(sha256_hex(&read_file(path, label)?))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn require_sha(value: &str, field: &str, length: usize) -> Result<(), String> {
    if value.len() != length
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!(
            "{field} must be a lowercase {length}-character hexadecimal identity"
        ));
    }
    Ok(())
}

fn native_architecture() -> Result<&'static str, String> {
    match env::consts::ARCH {
        "x86_64" => Ok("x86_64"),
        "aarch64" => Ok("aarch64"),
        "wasm32" => Ok("wasm32"),
        other => Err(format!("unsupported collector architecture {other}")),
    }
}

fn native_cpu_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        for (name, present) in [
            ("sse2", std::is_x86_feature_detected!("sse2")),
            ("sse3", std::is_x86_feature_detected!("sse3")),
            ("ssse3", std::is_x86_feature_detected!("ssse3")),
            ("sse4.1", std::is_x86_feature_detected!("sse4.1")),
            ("sse4.2", std::is_x86_feature_detected!("sse4.2")),
            ("avx", std::is_x86_feature_detected!("avx")),
            ("avx2", std::is_x86_feature_detected!("avx2")),
            ("avxvnni", std::is_x86_feature_detected!("avxvnni")),
            ("f16c", std::is_x86_feature_detected!("f16c")),
            ("fma", std::is_x86_feature_detected!("fma")),
            ("avx512bf16", std::is_x86_feature_detected!("avx512bf16")),
            ("avx512bw", std::is_x86_feature_detected!("avx512bw")),
            ("avx512dq", std::is_x86_feature_detected!("avx512dq")),
            ("avx512f", std::is_x86_feature_detected!("avx512f")),
            ("avx512fp16", std::is_x86_feature_detected!("avx512fp16")),
            ("avx512vbmi", std::is_x86_feature_detected!("avx512vbmi")),
            ("avx512vbmi2", std::is_x86_feature_detected!("avx512vbmi2")),
            ("avx512vl", std::is_x86_feature_detected!("avx512vl")),
            ("avx512vnni", std::is_x86_feature_detected!("avx512vnni")),
        ] {
            if present {
                features.push(name);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        for (name, present) in [
            ("aes", std::arch::is_aarch64_feature_detected!("aes")),
            ("bf16", std::arch::is_aarch64_feature_detected!("bf16")),
            ("crc", std::arch::is_aarch64_feature_detected!("crc")),
            (
                "dotprod",
                std::arch::is_aarch64_feature_detected!("dotprod"),
            ),
            ("fp16", std::arch::is_aarch64_feature_detected!("fp16")),
            ("i8mm", std::arch::is_aarch64_feature_detected!("i8mm")),
            ("neon", std::arch::is_aarch64_feature_detected!("neon")),
            ("sve", std::arch::is_aarch64_feature_detected!("sve")),
            ("sve2", std::arch::is_aarch64_feature_detected!("sve2")),
        ] {
            if present {
                features.push(name);
            }
        }
    }
    features.sort_unstable();
    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregation_is_sorted_and_uses_exact_mean() {
        let manifest = test_manifest();
        let events = vec![event(2, 3.0), event(1, 2.0), event(2, 5.0)];
        let rows = aggregate_events(events, &manifest).unwrap();
        assert_eq!(
            rows.iter().map(|row| row.op_index).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(rows[1].duration_sum_us, Some(8.0));
        assert_eq!(rows[1].duration_us, Some(4.0));
        assert_eq!(rows[1].sample_count, Some(2));
    }

    #[test]
    fn selector_conflicts_fail_closed() {
        let manifest = test_manifest();
        let mut changed = event(0, 2.0);
        changed.provider = "CPU".to_string();
        let error = aggregate_events(vec![event(0, 1.0), changed], &manifest).unwrap_err();
        assert!(error.contains("conflicting placement identity"));
    }

    #[test]
    fn memory_snapshots_conserve_bytes_lifetimes_and_alias_roots() {
        let mut manifest = test_manifest();
        manifest.instrumentation.arena_allocations = true;
        let events = memory_events();
        let memory = aggregate_memory_events(&events, &manifest)
            .unwrap()
            .unwrap();
        assert_eq!(memory.snapshot_count, 1);
        assert_eq!(memory.peak_non_persistent_arena_bytes, 4096);
        assert_eq!(memory.peak_persistent_arena_bytes, 256);
        assert_eq!(memory.peak_combined_arena_bytes, 4352);
        assert_eq!(memory.snapshots[0].allocation_count, 2);
        assert_eq!(memory.snapshots[0].alias_count, 1);
        assert_eq!(memory.snapshots[0].allocations[1].last_node, None);
        assert_eq!(memory.allocation_ledger_sha256.len(), 64);
    }

    #[test]
    fn overlapping_live_runtime_allocations_fail_closed() {
        let mut manifest = test_manifest();
        manifest.instrumentation.arena_allocations = true;
        let mut events = memory_events();
        events.memory_snapshots[0].allocation_count = 3;
        events.memory_allocations.push(MemoryAllocationEvent {
            event_kind: "memory_allocation".to_string(),
            memory_snapshot_id: 0,
            tensor_index: 3,
            arena: "kTfLiteArenaRw".to_string(),
            offset_bytes: 512,
            size_bytes: 512,
            first_node: 0,
            last_node: 1,
        });
        let error = aggregate_memory_events(&events, &manifest).unwrap_err();
        assert!(error.contains("overlapping live allocations"));
    }

    #[test]
    fn declared_memory_instrumentation_requires_a_snapshot() {
        let mut manifest = test_manifest();
        manifest.instrumentation.arena_allocations = true;
        let error = aggregate_memory_events(&ParsedEvents::default(), &manifest).unwrap_err();
        assert!(error.contains("emitted no memory snapshot"));
    }

    #[test]
    fn resource_partition_requires_exact_sampled_mask_union() {
        let value = resource_partition_fixture();
        let path = write_resource_partition_fixture(&value);
        let request = resource_partition_request_fixture();
        let parsed = load_resource_partition_observation(Some(&path), Some(&request), 2)
            .unwrap()
            .unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(
            parsed["exclusive_isolation_status"],
            "observed_cgroup_v2_isolated_partition"
        );

        let mut incomplete = resource_partition_fixture();
        incomplete["affinity_samples"] = json!([
            { "observed_at_unix_ms": 1000, "threads": [{ "tid": 100, "allowed_cpu_ids": [2], "processor": 2 }, { "tid": 101, "allowed_cpu_ids": [2], "processor": 2 }] },
            { "observed_at_unix_ms": 1020, "threads": [{ "tid": 100, "allowed_cpu_ids": [2], "processor": 2 }, { "tid": 101, "allowed_cpu_ids": [2], "processor": 2 }] },
            { "observed_at_unix_ms": 1040, "threads": [{ "tid": 100, "allowed_cpu_ids": [2], "processor": 2 }, { "tid": 101, "allowed_cpu_ids": [2], "processor": 2 }] },
            { "observed_at_unix_ms": 1060, "threads": [{ "tid": 100, "allowed_cpu_ids": [2], "processor": 2 }, { "tid": 101, "allowed_cpu_ids": [2], "processor": 2 }] }
        ]);
        incomplete["sampled_threads"] =
            json!([{ "tid": 100, "allowed_cpu_ids": [2] }, { "tid": 101, "allowed_cpu_ids": [2] }]);
        incomplete["observed_allowed_cpu_ids_union"] = json!([2]);
        incomplete["observed_processor_ids"] = json!([2]);
        let path = write_resource_partition_fixture(&incomplete);
        let request = resource_partition_request_fixture();
        let error =
            load_resource_partition_observation(Some(&path), Some(&request), 2).unwrap_err();
        fs::remove_file(&path).unwrap();
        assert!(error.contains("exactly reproduce"));
    }

    #[test]
    fn resource_partition_rejects_unmet_exclusive_request() {
        let mut value = resource_partition_fixture();
        value["exclusive_isolation_status"] = json!("not_observed_affinity_only");
        value["cgroup_v2_partition_state"] = json!("member");
        let path = write_resource_partition_fixture(&value);
        let request = resource_partition_request_fixture();
        let error =
            load_resource_partition_observation(Some(&path), Some(&request), 2).unwrap_err();
        fs::remove_file(&path).unwrap();
        assert!(error.contains("exclusive cpuset was not observed"));
    }

    #[test]
    fn preserves_multiple_lowerings_and_dispatches_without_fabricating_a_single_kernel() {
        let mut manifest = test_manifest();
        manifest.instrumentation.lowering_ids = true;
        manifest.instrumentation.microkernel_ids = true;
        manifest.build.xnnpack_source_commit = "c".repeat(40);
        let mut first = event(7, 1.0);
        first.event_kind = EventKind::Dispatch;
        first.duration_us = None;
        first.runtime_node_id = Some(3);
        first.compute_invocation_id = Some(0);
        first.lowering_id = Some("convolution_2d".to_string());
        first.kernel_id = Some("f32-igemm-4x8".to_string());
        first.kernel = Some("xnn_f32_igemm_minmax_ukernel_4x8__scalar".to_string());
        first.kernel_source_ref = Some(format!(
            "google/XNNPACK@{}/src/f32-igemm/gen/f32-igemm-4x8-minmax.c",
            "c".repeat(40)
        ));
        first.kernel_build_identifier_sha256 = Some("d".repeat(64));
        let mut second = first.clone();
        second.runtime_node_id = Some(4);
        second.compute_invocation_id = Some(1);
        second.lowering_id = Some("clamp".to_string());
        second.kernel_id = Some("f32-vclamp".to_string());
        second.kernel = Some("xnn_f32_vclamp_ukernel__scalar".to_string());
        second.kernel_source_ref = Some(format!(
            "google/XNNPACK@{}/src/f32-vclamp/gen/f32-vclamp.c",
            "c".repeat(40)
        ));
        let rows = aggregate_events(vec![first, second], &manifest).unwrap();
        assert_eq!(rows[0].lowerings.len(), 2);
        assert_eq!(rows[0].dispatches.len(), 2);
        assert!(rows[0].lowering_id.is_none());
        assert!(rows[0].kernel.is_none());
    }

    #[test]
    fn one_dispatch_with_multiple_lowerings_has_no_singular_selector_claim() {
        let mut manifest = test_manifest();
        manifest.instrumentation.lowering_ids = true;
        manifest.instrumentation.microkernel_ids = true;
        manifest.build.xnnpack_source_commit = "c".repeat(40);
        let mut dispatch = event(7, 1.0);
        dispatch.event_kind = EventKind::Dispatch;
        dispatch.duration_us = None;
        dispatch.runtime_node_id = Some(3);
        dispatch.compute_invocation_id = Some(0);
        dispatch.lowering_id = Some("fully_connected".to_string());
        dispatch.kernel_id = Some("qs8-gemm-1x8".to_string());
        dispatch.kernel = Some("xnn_qs8_gemm_minmax_ukernel_1x8__scalar".to_string());
        dispatch.kernel_source_ref = Some(format!(
            "google/XNNPACK@{}/src/qs8-gemm/gen/qs8-gemm-1x8-minmax.c",
            "c".repeat(40)
        ));
        dispatch.kernel_build_identifier_sha256 = Some("d".repeat(64));
        let mut reshape = dispatch.clone();
        reshape.event_kind = EventKind::Lowering;
        reshape.runtime_node_id = Some(2);
        reshape.compute_invocation_id = None;
        reshape.lowering_id = Some("static_reshape".to_string());
        reshape.kernel_id = None;
        reshape.kernel = None;
        reshape.kernel_source_ref = None;
        reshape.kernel_build_identifier_sha256 = None;
        let rows = aggregate_events(vec![dispatch, reshape], &manifest).unwrap();
        assert_eq!(rows[0].lowerings.len(), 2);
        assert_eq!(rows[0].dispatches.len(), 1);
        assert!(rows[0].lowering_id.is_none());
        assert!(rows[0].kernel_id.is_none());
        assert!(rows[0].kernel.is_none());
        assert!(rows[0].kernel_source_ref.is_none());
        assert!(rows[0].kernel_build_identifier_sha256.is_none());
    }

    fn event(op_index: usize, duration_us: f64) -> RuntimeEvent {
        RuntimeEvent {
            event_kind: EventKind::Observation,
            op_index,
            op_name: "CONV_2D".to_string(),
            provider: "XNNPACK".to_string(),
            delegated: Some(true),
            partition_id: Some("xnn-0".to_string()),
            lowering_id: None,
            kernel_id: None,
            kernel: None,
            kernel_source_ref: None,
            kernel_build_identifier_sha256: None,
            duration_us: Some(duration_us),
            runtime_node_id: None,
            compute_invocation_id: None,
        }
    }

    fn memory_events() -> ParsedEvents {
        ParsedEvents {
            memory_snapshots: vec![MemorySnapshotEvent {
                event_kind: "memory_snapshot".to_string(),
                memory_snapshot_id: 0,
                non_persistent_arena_bytes: 4096,
                persistent_arena_bytes: 256,
                tensor_count: 4,
                execution_node_count: 2,
                allocation_count: 2,
                alias_count: 1,
            }],
            memory_allocations: vec![
                MemoryAllocationEvent {
                    event_kind: "memory_allocation".to_string(),
                    memory_snapshot_id: 0,
                    tensor_index: 0,
                    arena: "kTfLiteArenaRw".to_string(),
                    offset_bytes: 0,
                    size_bytes: 1024,
                    first_node: 0,
                    last_node: 1,
                },
                MemoryAllocationEvent {
                    event_kind: "memory_allocation".to_string(),
                    memory_snapshot_id: 0,
                    tensor_index: 1,
                    arena: "kTfLiteArenaRwPersistent".to_string(),
                    offset_bytes: 0,
                    size_bytes: 256,
                    first_node: 0,
                    last_node: i32::MAX,
                },
            ],
            memory_aliases: vec![MemoryAliasEvent {
                event_kind: "memory_alias".to_string(),
                memory_snapshot_id: 0,
                tensor_index: 2,
                shared_with_tensor_index: 0,
            }],
            ..ParsedEvents::default()
        }
    }

    fn resource_partition_fixture() -> serde_json::Value {
        json!({
            "schema": "deepbom.resource_partition_observation.v1",
            "evidence_class": "OBSERVED_OS_RESOURCE_PARTITION",
            "requested_cpu_ids": [2, 3],
            "affinity_mode": "taskset_process_and_descendants",
            "isolation_expectation": "exclusive_cpuset",
            "affinity_status": "observed_all_sampled_threads_within_requested_set",
            "exclusive_isolation_status": "observed_cgroup_v2_isolated_partition",
            "sample_count": 4,
            "maximum_observed_thread_count": 2,
            "sampled_threads": [
                { "tid": 100, "allowed_cpu_ids": [2, 3] },
                { "tid": 101, "allowed_cpu_ids": [2, 3] }
            ],
            "affinity_samples": [
                { "observed_at_unix_ms": 1000, "threads": [
                    { "tid": 100, "allowed_cpu_ids": [2, 3], "processor": 2 },
                    { "tid": 101, "allowed_cpu_ids": [2, 3], "processor": 3 }
                ] },
                { "observed_at_unix_ms": 1020, "threads": [
                    { "tid": 100, "allowed_cpu_ids": [2, 3], "processor": 3 },
                    { "tid": 101, "allowed_cpu_ids": [2, 3], "processor": 2 }
                ] },
                { "observed_at_unix_ms": 1040, "threads": [
                    { "tid": 100, "allowed_cpu_ids": [2, 3], "processor": 2 },
                    { "tid": 101, "allowed_cpu_ids": [2, 3], "processor": 3 }
                ] },
                { "observed_at_unix_ms": 1060, "threads": [
                    { "tid": 100, "allowed_cpu_ids": [2, 3], "processor": 3 },
                    { "tid": 101, "allowed_cpu_ids": [2, 3], "processor": 2 }
                ] }
            ],
            "observed_allowed_cpu_ids_union": [2, 3],
            "observed_processor_ids": [2, 3],
            "observed_effective_cpu_ids": [2, 3],
            "cgroup_v2_path": "/deepbom.slice/capture.scope",
            "cgroup_v2_partition_state": "isolated",
            "online_cpu_ids": [0, 1, 2, 3],
            "kernel_command_line": "quiet nohz_full=2-3",
            "kernel_isolation_parameters": {},
            "cpu_frequency_policy": [],
            "cache_shared_cpu_lists": [],
            "interpretation_boundary": "fixture"
        })
    }

    fn resource_partition_request_fixture() -> ResourcePartitionRequest {
        ResourcePartitionRequest {
            requested_cpu_ids: vec![2, 3],
            affinity_mode: "taskset_process_and_descendants".to_string(),
            isolation_expectation: "exclusive_cpuset".to_string(),
        }
    }

    fn write_resource_partition_fixture(value: &serde_json::Value) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = env::temp_dir().join(format!(
            "deepbom-resource-partition-{}-{nonce}.json",
            std::process::id()
        ));
        fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
        path
    }

    fn test_manifest() -> CaptureManifest {
        CaptureManifest {
            schema: MANIFEST_SCHEMA.to_string(),
            capture_mode: CaptureMode::RuntimeCapture,
            artifact_path: PathBuf::new(),
            artifact_sha256: "a".repeat(64),
            target_profile_id: "x86_desktop_avx2".to_string(),
            target_profile_sha256: "b".repeat(64),
            runtime: RuntimeManifest {
                name: "TFLite".to_string(),
                version: "pinned".to_string(),
                backend: "XNNPACK".to_string(),
                build: "test".to_string(),
                binary_path: PathBuf::new(),
            },
            source: SourceManifest {
                collected_at: "2026-07-16T00:00:00Z".to_string(),
                capture_id: "test".to_string(),
                collector_source_commit: "deepbom@test".to_string(),
            },
            device: DeviceManifest {
                identity: "test-host".to_string(),
                nnapi_runtime_feature_level: None,
                nnapi_accelerator_identity: None,
                nnapi_capability_source: None,
            },
            build: BuildManifest {
                tensorflow_source_commit: TENSORFLOW_SOURCE_COMMIT.to_string(),
                xnnpack_source_commit: "c".repeat(40),
                microkernel_build_identifier_path: PathBuf::new(),
                build_manifest_path: PathBuf::new(),
                compile_definitions: vec![],
                cmake_system_name: None,
            },
            invocation: InvocationManifest {
                inputs: vec![],
                thread_count: 1,
                runtime_options_path: PathBuf::new(),
                resource_partition_request: None,
                resource_partition_observation_path: None,
                tflite_gpu_experimental_flags: None,
                tflite_gpu_max_delegated_partitions: None,
            },
            instrumentation: Instrumentation {
                lowering_ids: false,
                microkernel_ids: false,
                arena_allocations: false,
            },
        }
    }
}
