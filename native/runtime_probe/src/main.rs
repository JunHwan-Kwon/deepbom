use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::Path;

const PLAN_SCHEMA: &str = "deepbom.native_contract_probe_plan.v1.1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbePlan {
    schema: String,
    artifact_sha256: String,
    iterations: usize,
    events: Vec<ProbeEvent>,
    #[serde(default)]
    memory_snapshots: Vec<ProbeMemorySnapshot>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbeMemorySnapshot {
    non_persistent_arena_bytes: u64,
    persistent_arena_bytes: u64,
    tensor_count: usize,
    execution_node_count: usize,
    allocations: Vec<ProbeMemoryAllocation>,
    aliases: Vec<ProbeMemoryAlias>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbeMemoryAllocation {
    tensor_index: usize,
    arena: String,
    offset_bytes: u64,
    size_bytes: u64,
    first_node: usize,
    last_node: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbeMemoryAlias {
    tensor_index: usize,
    shared_with_tensor_index: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProbeEvent {
    op_index: usize,
    op_name: String,
    provider: String,
    delegated: Option<bool>,
    partition_id: Option<String>,
    lowering_id: Option<String>,
    kernel_id: Option<String>,
    kernel: Option<String>,
    kernel_source_ref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kernel_build_identifier_sha256: Option<String>,
    #[serde(skip_serializing)]
    duration_us_samples: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_us: Option<f64>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("deepbom-runtime-contract-probe: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        return Err(
            "usage: deepbom-runtime-contract-probe <plan.json> <artifact> <events.ndjson>"
                .to_string(),
        );
    }
    let plan: ProbePlan = serde_json::from_slice(
        &fs::read(&args[1]).map_err(|error| format!("cannot read probe plan: {error}"))?,
    )
    .map_err(|error| format!("invalid probe plan: {error}"))?;
    validate_plan(&plan)?;
    let artifact = fs::read(&args[2]).map_err(|error| format!("cannot read artifact: {error}"))?;
    let actual_artifact_sha = sha256_hex(&artifact);
    if actual_artifact_sha != plan.artifact_sha256 {
        return Err(format!(
            "artifact SHA-256 mismatch: plan {}, actual {actual_artifact_sha}",
            plan.artifact_sha256
        ));
    }
    let build_id = env::var("DEEPBOM_MICROKERNEL_BUILD_ID_SHA256")
        .map_err(|_| "DEEPBOM_MICROKERNEL_BUILD_ID_SHA256 is required".to_string())?;
    require_sha(&build_id, "DEEPBOM_MICROKERNEL_BUILD_ID_SHA256")?;
    let encoded = render_events(&plan, &build_id)?;
    fs::write(Path::new(&args[3]), encoded)
        .map_err(|error| format!("cannot write event stream: {error}"))?;
    Ok(())
}

fn validate_plan(plan: &ProbePlan) -> Result<(), String> {
    if plan.schema != PLAN_SCHEMA {
        return Err(format!("probe plan schema must be {PLAN_SCHEMA}"));
    }
    require_sha(&plan.artifact_sha256, "artifact_sha256")?;
    if plan.iterations == 0 || plan.events.is_empty() {
        return Err("probe plan requires positive iterations and at least one event".to_string());
    }
    for event in &plan.events {
        if event.op_name.trim().is_empty()
            || event.provider.trim().is_empty()
            || event.duration_us_samples.is_empty()
            || event
                .duration_us_samples
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(format!("invalid event plan for op #{}", event.op_index));
        }
        if event.duration_us.is_some() || event.kernel_build_identifier_sha256.is_some() {
            return Err(
                "duration_us and kernel_build_identifier_sha256 are generated fields".to_string(),
            );
        }
        let kernel_fields = [&event.kernel_id, &event.kernel, &event.kernel_source_ref];
        let kernel_count = kernel_fields.iter().filter(|value| value.is_some()).count();
        if kernel_count != 0 && kernel_count != kernel_fields.len() {
            return Err(format!(
                "op #{} must define all synthetic kernel identity fields together",
                event.op_index
            ));
        }
    }
    for (snapshot_index, snapshot) in plan.memory_snapshots.iter().enumerate() {
        if snapshot.tensor_count == 0 || snapshot.execution_node_count == 0 {
            return Err(format!(
                "memory snapshot {snapshot_index} requires positive tensor and execution-node counts"
            ));
        }
        for allocation in &snapshot.allocations {
            if allocation.tensor_index >= snapshot.tensor_count
                || allocation.size_bytes == 0
                || allocation.first_node >= snapshot.execution_node_count
                || allocation
                    .last_node
                    .is_some_and(|last| last < allocation.first_node || last >= snapshot.execution_node_count)
                || !["kTfLiteArenaRw", "kTfLiteArenaRwPersistent"]
                    .contains(&allocation.arena.as_str())
            {
                return Err(format!(
                    "memory snapshot {snapshot_index} has an invalid allocation"
                ));
            }
        }
        for alias in &snapshot.aliases {
            if alias.tensor_index >= snapshot.tensor_count
                || alias.shared_with_tensor_index >= snapshot.tensor_count
                || alias.tensor_index == alias.shared_with_tensor_index
            {
                return Err(format!("memory snapshot {snapshot_index} has an invalid alias"));
            }
        }
    }
    Ok(())
}

fn render_events(plan: &ProbePlan, build_id: &str) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    for (snapshot_id, snapshot) in plan.memory_snapshots.iter().enumerate() {
        serde_json::to_writer(
            &mut output,
            &serde_json::json!({
                "event_kind": "memory_snapshot",
                "memory_snapshot_id": snapshot_id,
                "non_persistent_arena_bytes": snapshot.non_persistent_arena_bytes,
                "persistent_arena_bytes": snapshot.persistent_arena_bytes,
                "tensor_count": snapshot.tensor_count,
                "execution_node_count": snapshot.execution_node_count,
                "allocation_count": snapshot.allocations.len(),
                "alias_count": snapshot.aliases.len(),
            }),
        )
        .map_err(|error| format!("cannot encode memory snapshot: {error}"))?;
        output.push(b'\n');
        for allocation in &snapshot.allocations {
            serde_json::to_writer(
                &mut output,
                &serde_json::json!({
                    "event_kind": "memory_allocation",
                    "memory_snapshot_id": snapshot_id,
                    "tensor_index": allocation.tensor_index,
                    "arena": allocation.arena,
                    "offset_bytes": allocation.offset_bytes,
                    "size_bytes": allocation.size_bytes,
                    "first_node": allocation.first_node,
                    "last_node": allocation.last_node.unwrap_or(i32::MAX as usize),
                }),
            )
            .map_err(|error| format!("cannot encode memory allocation: {error}"))?;
            output.push(b'\n');
        }
        for alias in &snapshot.aliases {
            serde_json::to_writer(
                &mut output,
                &serde_json::json!({
                    "event_kind": "memory_alias",
                    "memory_snapshot_id": snapshot_id,
                    "tensor_index": alias.tensor_index,
                    "shared_with_tensor_index": alias.shared_with_tensor_index,
                }),
            )
            .map_err(|error| format!("cannot encode memory alias: {error}"))?;
            output.push(b'\n');
        }
    }
    for iteration in 0..plan.iterations {
        for template in &plan.events {
            let mut event = template.clone();
            event.duration_us =
                Some(event.duration_us_samples[iteration % event.duration_us_samples.len()]);
            if event.kernel.is_some() {
                event.kernel_build_identifier_sha256 = Some(build_id.to_string());
            }
            serde_json::to_writer(&mut output, &event)
                .map_err(|error| format!("cannot encode event: {error}"))?;
            output.push(b'\n');
        }
    }
    Ok(output)
}

fn require_sha(value: &str, field: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!("{field} must be a lowercase SHA-256"));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> ProbePlan {
        ProbePlan {
            schema: PLAN_SCHEMA.to_string(),
            artifact_sha256: "a".repeat(64),
            iterations: 2,
            events: vec![ProbeEvent {
                op_index: 4,
                op_name: "CONV_2D".to_string(),
                provider: "XNNPACK".to_string(),
                delegated: Some(true),
                partition_id: Some("xnn-0".to_string()),
                lowering_id: Some("convolution_to_igemm".to_string()),
                kernel_id: Some("f32-igemm-4x8".to_string()),
                kernel: Some("xnn_f32_igemm_minmax_ukernel_4x8__scalar".to_string()),
                kernel_source_ref: Some("google/XNNPACK@23a67314f7afdbb76191589ae090d82bf55afbfa/src/f32-igemm/gen/f32-igemm-4x8-minmax.c".to_string()),
                kernel_build_identifier_sha256: None,
                duration_us_samples: vec![4.0, 6.0],
                duration_us: None,
            }],
            memory_snapshots: vec![ProbeMemorySnapshot {
                non_persistent_arena_bytes: 4096,
                persistent_arena_bytes: 256,
                tensor_count: 3,
                execution_node_count: 2,
                allocations: vec![
                    ProbeMemoryAllocation {
                        tensor_index: 0,
                        arena: "kTfLiteArenaRw".to_string(),
                        offset_bytes: 0,
                        size_bytes: 1024,
                        first_node: 0,
                        last_node: Some(1),
                    },
                    ProbeMemoryAllocation {
                        tensor_index: 1,
                        arena: "kTfLiteArenaRwPersistent".to_string(),
                        offset_bytes: 0,
                        size_bytes: 256,
                        first_node: 0,
                        last_node: None,
                    },
                ],
                aliases: vec![ProbeMemoryAlias {
                    tensor_index: 2,
                    shared_with_tensor_index: 0,
                }],
            }],
        }
    }

    #[test]
    fn produces_deterministic_repeated_events() {
        let value = plan();
        validate_plan(&value).unwrap();
        let output = String::from_utf8(render_events(&value, &"b".repeat(64)).unwrap()).unwrap();
        let rows: Vec<serde_json::Value> = output
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(rows.len(), 6);
        assert_eq!(rows[0]["event_kind"], "memory_snapshot");
        assert_eq!(rows[1]["event_kind"], "memory_allocation");
        assert_eq!(rows[3]["event_kind"], "memory_alias");
        assert_eq!(rows[4]["duration_us"], 4.0);
        assert_eq!(rows[5]["duration_us"], 6.0);
        assert_eq!(rows[4]["kernel_build_identifier_sha256"], "b".repeat(64));
    }

    #[test]
    fn rejects_generated_fields_in_plan() {
        let mut value = plan();
        value.events[0].duration_us = Some(1.0);
        assert!(validate_plan(&value).is_err());
    }
}
