use super::{
    fmt_mb, ArtifactByteIntegrityLedger, InputContract, MovementAnalysis, OpInfo,
    PredictedPartitionBoundaryInventory, TensorInfo, TensorLiveness, WeightIntegrityReport,
};
use serde::Serialize;

#[derive(Clone, Serialize)]
pub(super) struct FindingEvidence {
    pub(super) source: String,
    pub(super) text: String,
}

#[derive(Clone, Serialize)]
pub(super) struct Finding {
    pub(super) id: String,
    pub(super) title: String,
    pub(super) category: String,
    pub(super) severity: String,
    pub(super) confidence: String,
    pub(super) evidence: Vec<FindingEvidence>,
    pub(super) impact: String,
    pub(super) actions: Vec<String>,
}

pub(super) struct FindingAnalysisContext<'a> {
    pub(super) ops: &'a [OpInfo],
    pub(super) tensors: &'a [TensorInfo],
    pub(super) liveness: &'a TensorLiveness,
    pub(super) movement: &'a MovementAnalysis,
    pub(super) contracts: &'a [InputContract],
    pub(super) boundaries: &'a PredictedPartitionBoundaryInventory,
    pub(super) weight_integrity: &'a WeightIntegrityReport,
    pub(super) byte_integrity: &'a ArtifactByteIntegrityLedger,
}

pub(super) fn append_quantized_artifact_findings(
    findings: &mut Vec<Finding>,
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    weight_integrity: &WeightIntegrityReport,
) {
    if weight_integrity.zero_kernel_slice_count > 0 {
        let examples = weight_integrity
            .zero_kernel_slice_details
            .iter()
            .filter(|detail| {
                weight_integrity.exact_zero_kernel_slice_count == 0
                    || !detail.exact_zero_channels.is_empty()
            })
            .take(4)
            .map(|detail| {
                let channels = if weight_integrity.exact_zero_kernel_slice_count > 0 {
                    &detail.exact_zero_channels
                } else {
                    &detail.channels
                };
                format!(
                    "T{} {}: channel(s) {}",
                    detail.tensor_index,
                    detail.tensor_name,
                    channels
                        .iter()
                        .map(|channel| channel.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        findings.push(Finding {
            id: "weight-integrity-zero-kernel-slices".to_string(),
            title: if weight_integrity.exact_zero_kernel_slice_count > 0 {
                format!(
                    "{} exact-zero stored kernel slice(s) require functional review",
                    weight_integrity.exact_zero_kernel_slice_count
                )
            } else {
                format!(
                    "{} near-zero decoded kernel slice(s) require functional review",
                    weight_integrity.zero_kernel_slice_count
                )
            },
            category: "quant".to_string(),
            severity: if weight_integrity.exact_zero_kernel_slice_count > 0 {
                "medium"
            } else {
                "informational"
            }
            .to_string(),
            confidence: "static+wasm".to_string(),
            evidence: vec![
                FindingEvidence {
                    source: "OBSERVED".to_string(),
                    text: format!(
                        "{} tensor(s), {} output-channel slice(s), after applying artifact scale/zero-point metadata",
                        weight_integrity.zero_kernel_slice_tensors,
                        weight_integrity.zero_kernel_slice_count
                    ),
                },
                FindingEvidence {
                    source: "Exact-zero subset".to_string(),
                    text: format!(
                        "{} tensor(s), {} output-channel slice(s) have every stored centered weight code exactly zero",
                        weight_integrity.exact_zero_kernel_slice_tensors,
                        weight_integrity.exact_zero_kernel_slice_count
                    ),
                },
                FindingEvidence {
                    source: "Examples".to_string(),
                    text: examples,
                },
                FindingEvidence {
                    source: "Scope".to_string(),
                    text: "Near-zero decoded and exact-zero stored slices are separate artifact observations. Neither alone proves model-output inactivity or task-accuracy loss; bias, fused activation, residual/downstream behavior, and representative outputs remain separate evidence.".to_string(),
                },
            ],
            impact: "An exact-zero stored kernel slice contributes no input-dependent dot product, but bias and downstream graph behavior can still produce a non-zero or consequential channel; model-output inactivity is not established by this check".to_string(),
            actions: vec![
                "Review the listed bias values, fused activations, residual/downstream consumers, and representative outputs before classifying a channel as inactive".to_string(),
            ],
        });
    }

    if weight_integrity.low_grid_utilization_tensors > 0
        || weight_integrity.saturated_quantized_tensors > 0
    {
        findings.push(Finding {
            id: "quant-grid-utilization".to_string(),
            title: "Quantized constant grid utilization requires calibration review".to_string(),
            category: "quant".to_string(),
            severity: "medium".to_string(),
            confidence: "static+wasm".to_string(),
            evidence: vec![FindingEvidence {
                source: "DERIVED".to_string(),
                text: weight_integrity.quant_grid_detail.clone(),
            }],
            impact: "Low level utilization is consistent with an over-wide quantization range, while endpoint saturation is consistent with an under-wide range; neither proves task-accuracy loss".to_string(),
            actions: vec![
                "Review calibration or QAT ranges for the flagged constants and compare representative outputs against the source model".to_string(),
            ],
        });
    }

    let depthwise_per_tensor = ops
        .iter()
        .filter(|op| op.name == "DEPTHWISE_CONV_2D")
        .filter_map(|op| {
            let tensor_index = *op.inputs.get(1)?;
            let tensor = tensors.get(usize::try_from(tensor_index).ok()?)?;
            (tensor.constant_buffer
                && matches!(tensor.dtype.as_str(), "INT8" | "UINT8")
                && tensor.quant_scales == 1)
                .then_some((op, tensor))
        })
        .collect::<Vec<_>>();
    if !depthwise_per_tensor.is_empty() {
        let examples = depthwise_per_tensor
            .iter()
            .take(4)
            .map(|(op, tensor)| {
                format!(
                    "#{} {} -> T{} {}",
                    op.index, op.name, tensor.index, tensor.name
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        findings.push(Finding {
            id: "depthwise-per-tensor-weights".to_string(),
            title: format!(
                "{} quantized DEPTHWISE_CONV_2D op(s) use per-tensor weights",
                depthwise_per_tensor.len()
            ),
            category: "quant".to_string(),
            severity: "medium".to_string(),
            confidence: "static+wasm".to_string(),
            evidence: vec![
                FindingEvidence {
                    source: "OBSERVED".to_string(),
                    text: format!(
                        "Each listed constant has exactly one embedded scale. Per-channel quantized tensors in the artifact: {}.",
                        tensors.iter().filter(|tensor| tensor.quant_scales > 1).count()
                    ),
                },
                FindingEvidence {
                    source: "Examples".to_string(),
                    text: examples,
                },
            ],
            impact: "A single scale across depthwise channels can reduce effective precision for channels with different ranges; actual task impact requires representative comparison".to_string(),
            actions: vec![
                "Prefer a supported per-channel weight export and compare representative outputs and target-runtime compatibility before release".to_string(),
            ],
        });
    }

    let asymmetric_uint8 = ops
        .iter()
        .filter(|op| {
            matches!(
                op.name.as_str(),
                "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED"
            )
        })
        .filter_map(|op| {
            let tensor_index = *op.inputs.get(1)?;
            let tensor = tensors.get(usize::try_from(tensor_index).ok()?)?;
            (tensor.constant_buffer
                && tensor.dtype == "UINT8"
                && tensor.quant_zero_points == 1
                && tensor
                    .zero_point_sample
                    .first()
                    .is_some_and(|zero_point| *zero_point != 0))
            .then_some((op, tensor))
        })
        .collect::<Vec<_>>();
    if !asymmetric_uint8.is_empty() {
        let examples = asymmetric_uint8
            .iter()
            .take(4)
            .map(|(op, tensor)| {
                format!(
                    "#{} {} -> T{} {}, zp {}",
                    op.index,
                    op.name,
                    tensor.index,
                    tensor.name,
                    tensor
                        .zero_point_sample
                        .first()
                        .copied()
                        .unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        findings.push(Finding {
            id: "asymmetric-uint8-weights".to_string(),
            title: format!(
                "{} quantized kernel op(s) use asymmetric UINT8 weights",
                asymmetric_uint8.len()
            ),
            category: "quant".to_string(),
            severity: "medium".to_string(),
            confidence: "static+wasm".to_string(),
            evidence: vec![
                FindingEvidence {
                    source: "OBSERVED".to_string(),
                    text: examples,
                },
                FindingEvidence {
                    source: "Contract".to_string(),
                    text: "The observed UINT8 zero-points are legal dtype values. Exact converter generation or lineage is not inferable from this artifact fact alone.".to_string(),
                },
            ],
            impact: "This quantization design may not use modern per-channel symmetric INT8 kernel paths; runtime support and performance remain build-dependent".to_string(),
            actions: vec![
                "Record converter lineage and runtime compatibility, and compare a modern per-channel symmetric INT8 export when the target runtime supports it".to_string(),
            ],
        });
    }
}

pub(super) fn build_findings_from_analysis(context: FindingAnalysisContext<'_>) -> Vec<Finding> {
    let FindingAnalysisContext {
        ops,
        tensors,
        liveness,
        movement,
        contracts,
        boundaries,
        weight_integrity,
        byte_integrity,
    } = context;
    let mut findings: Vec<Finding> = Vec::new();

    // 1. XNNPACK predicted partition breaks
    let break_ops: Vec<&OpInfo> = ops.iter().filter(|op| op.xnnpack_chain_break).collect();
    if !break_ops.is_empty() {
        let families = break_ops
            .iter()
            .map(|op| op.name.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let classes = break_ops
            .iter()
            .map(|op| op.xnnpack_break_class.as_str())
            .filter(|value| !value.is_empty())
            .collect::<std::collections::BTreeSet<_>>();
        let assessed_bytes = boundaries
            .edges
            .iter()
            .filter_map(|edge| edge.payload_bytes)
            .sum::<usize>();
        let unassessed_payloads = boundaries
            .edges
            .iter()
            .filter(|edge| edge.payload_bytes.is_none())
            .count();
        findings.push(Finding {
            id: "xnn-break-summary".to_string(),
            title: format!("{} XNNPACK predicted partition breaks total", break_ops.len()),
            category: "delegation".to_string(),
            severity: "medium".to_string(),
            confidence: "static".to_string(),
            evidence: vec![
                FindingEvidence {
                    source: "Static assignment".to_string(),
                    text: format!(
                        "{} break op(s): {}; classes {}",
                        break_ops.len(),
                        families.into_iter().collect::<Vec<_>>().join(", "),
                        classes.into_iter().collect::<Vec<_>>().join(", ")
                    ),
                },
                FindingEvidence {
                    source: "Predicted boundary edges".to_string(),
                    text: format!(
                        "{} producer/consumer edge(s), {} logical payload bytes assessed{}",
                        boundaries.edges.len(),
                        assessed_bytes,
                        if unassessed_payloads > 0 {
                            format!("; {} payload(s) unassessed", unassessed_payloads)
                        } else {
                            String::new()
                        }
                    ),
                },
            ],
            impact: "Multiple predicted partition breaks can add fallback/interface overhead; runtime materialization and latency require target profiling".to_string(),
            actions: vec![
                "Use the emitted break families, rule reasons, and Delegation Repair Lab scenarios to address the observed graph cause; do not apply a generic Q/DQ rewrite when Q/DQ ops are absent".to_string(),
                "Confirm assignment and materialized boundary traffic in target-runtime delegate logs and profiling".to_string(),
            ],
        });
    }

    // 2. Quant holes
    let holes: Vec<&OpInfo> = ops.iter().filter(|op| op.quant_hole).collect();
    if !holes.is_empty() {
        let list = holes
            .iter()
            .take(3)
            .map(|op| format!("#{} {}", op.index, op.name))
            .collect::<Vec<_>>()
            .join(" · ");
        let suffix = if holes.len() > 3 {
            format!(" +{} more", holes.len() - 3)
        } else {
            String::new()
        };
        findings.push(Finding {
            id: "quant-holes".to_string(),
            title: format!("{} FP32 island{} in INT8 graph", holes.len(), if holes.len() > 1 { "s" } else { "" }),
            category: "quant".to_string(),
            severity: if holes.len() > 2 { "high" } else { "medium" }.to_string(),
            confidence: "static".to_string(),
            evidence: vec![
                FindingEvidence { source: "Graph".to_string(), text: format!("{}{}", list, suffix) },
                FindingEvidence { source: "Static".to_string(), text: "FP32 compute wrapped in DEQUANTIZE/QUANTIZE boundary".to_string() },
            ],
            impact: "FP32 islands create predicted delegate-segment boundaries, double buffer pressure, and reduce INT8 benefits".to_string(),
            actions: vec![
                "Apply PTQ or QAT to quant-hole ops if accuracy allows".to_string(),
                "If FP32 is intentional, document the fallback path and confirm it is acceptable on target".to_string(),
            ],
        });
    }

    // 3. Peak activation memory
    let peak_mb = liveness.peak_bytes as f64 / (1024.0 * 1024.0);
    if peak_mb > 10.0 {
        findings.push(Finding {
            id: "peak-activation-memory".to_string(),
            title: format!("Peak activation memory ≈ {}", fmt_mb(liveness.peak_bytes)),
            category: "memory".to_string(),
            severity: if peak_mb > 80.0 {
                "high"
            } else if peak_mb > 30.0 {
                "medium"
            } else {
                "low"
            }
            .to_string(),
            confidence: "static".to_string(),
            evidence: vec![
                FindingEvidence {
                    source: "Liveness".to_string(),
                    text: format!(
                        "Peak at op #{} {} — {} concurrent activations",
                        liveness.peak_at_op,
                        liveness.peak_at_op_name,
                        fmt_mb(liveness.peak_bytes)
                    ),
                },
                FindingEvidence {
                    source: "Note".to_string(),
                    text: "Live-payload bound; in-place execution can reduce physical arena demand for eligible operators"
                        .to_string(),
                },
            ],
            impact: format!(
                "Devices with < {} free SRAM may OOM or thrash the memory bus",
                fmt_mb((liveness.peak_bytes as f64 * 1.5) as usize)
            ),
            actions: vec![
                "Profile actual heap on device with the TFLite profiler".to_string(),
                format!(
                    "Reduce intermediate channel width around op #{} to lower the peak",
                    liveness.peak_at_op
                ),
            ],
        });
    }

    // 4. Low-intensity memory-traffic majority
    let mem_bound = ops
        .iter()
        .filter(|op| op.static_bound_guess == "memory-bound")
        .count();
    if !ops.is_empty() && mem_bound * 100 / ops.len() > 45 {
        let pct = mem_bound * 100 / ops.len();
        findings.push(Finding {
            id: "low-intensity-majority".to_string(),
            title: format!("{}% of ops are low-intensity memory-traffic candidates (static posture)", pct),
            category: "latency".to_string(),
            severity: if pct > 70 { "medium" } else { "low" }.to_string(),
            confidence: "static".to_string(),
            evidence: vec![
                FindingEvidence { source: "Roofline".to_string(), text: format!("{}/{} ops fall into the low-intensity static posture band", mem_bound, ops.len()) },
                FindingEvidence { source: "Implication".to_string(), text: "Memory traffic may limit latency more than peak compute for these candidates".to_string() },
            ],
            impact: "Higher peak compute alone may not improve latency if target profiling confirms memory-traffic dominance".to_string(),
            actions: vec![
                "Reduce activation footprint: fewer/smaller intermediate tensors".to_string(),
                "Reduce activation traffic or fuse materializing transforms where target profiling identifies a bottleneck".to_string(),
            ],
        });
    }

    // 5. Heavy data movement
    if !ops.is_empty() && movement.movement_op_ratio > 0.12 && movement.status == "assessed" {
        let mov_mb = fmt_mb(movement.total_movement_bytes);
        let brk_mb = fmt_mb(movement.xnn_break_movement_bytes);
        findings.push(Finding {
            id: "heavy-data-movement".to_string(),
            title: format!("{} layout/movement ops — {} total", movement.movement_op_count, mov_mb),
            category: "latency".to_string(),
            severity: if movement.movement_op_ratio > 0.22 { "medium" } else { "low" }.to_string(),
            confidence: "static".to_string(),
            evidence: vec![
                FindingEvidence { source: "Graph".to_string(), text: format!("{} zero-MAC ops (TRANSPOSE/CONCAT/RESIZE/Q·DQ/PAD/SLICE)", movement.movement_op_count) },
                FindingEvidence { source: "Static".to_string(), text: format!("{} moved · {} at XNNPACK break points", mov_mb, brk_mb) },
            ],
            impact: "Tensor traffic adds latency and cache pressure without contributing to compute".to_string(),
            actions: vec![
                "Audit TRANSPOSE ops — may indicate NHWC/NCHW conversion overhead from framework mismatch".to_string(),
                "Fuse or merge consecutive QUANTIZE/DEQUANTIZE boundary ops where possible".to_string(),
            ],
        });
    }

    // 6. High quant-risk ops
    let risk_ops: Vec<&OpInfo> = ops.iter().filter(|op| op.quant_risk == "risk").collect();
    if !risk_ops.is_empty() {
        let list = risk_ops
            .iter()
            .take(3)
            .map(|op| {
                format!(
                    "#{} {} (scale ratio {:.3e}, CV {:.2})",
                    op.index, op.name, op.quant_scale_ratio, op.quant_scale_cv
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        let suffix = if risk_ops.len() > 3 {
            format!(" +{} more", risk_ops.len() - 3)
        } else {
            String::new()
        };
        findings.push(Finding {
            id: "quant-risk-ops".to_string(),
            title: format!(
                "{} op{} with high quantization risk",
                risk_ops.len(),
                if risk_ops.len() > 1 { "s" } else { "" }
            ),
            category: "quant".to_string(),
            severity: if risk_ops.len() > 4 { "high" } else { "medium" }.to_string(),
            confidence: "static".to_string(),
            evidence: vec![
                FindingEvidence {
                    source: "Static".to_string(),
                    text: format!("{}{}", list, suffix),
                },
                FindingEvidence {
                    source: "Signal".to_string(),
                    text: "The op risk predicate is driven by the emitted scale-ratio, zero-point, and distribution fields; CV alone is not the decision rule".to_string(),
                },
            ],
            impact: "The artifact carries an unusual quantization contract that warrants source-checkpoint and output-drift review; scale spread alone is not a task-accuracy conclusion".to_string(),
            actions: vec![
                "Inspect the exact flagged scale channels, bias-scale contract, stored centered codes, and source checkpoint before selecting a remedy".to_string(),
                "Use QAT or calibration changes only when representative output comparisons identify the corresponding error mechanism".to_string(),
            ],
        });
    }

    // 7. Input contract risks
    let risky: Vec<&InputContract> = contracts.iter().filter(|c| !c.risks.is_empty()).collect();
    if !risky.is_empty() {
        let evidence: Vec<FindingEvidence> = risky
            .iter()
            .map(|c| FindingEvidence {
                source: format!("T{}", c.tensor_index),
                text: format!(
                    "{} {} — {}",
                    c.dtype,
                    c.shape
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join("×"),
                    c.range_note
                ),
            })
            .collect();
        findings.push(Finding {
            id: "input-contract".to_string(),
            title: "Input preprocessing contract not encoded in model".to_string(),
            category: "input".to_string(),
            severity: "low".to_string(),
            confidence: "static".to_string(),
            evidence,
            impact:
                "Silent accuracy degradation if normalization or channel order mismatches training"
                    .to_string(),
            actions: vec![
                "Document expected input range and normalization in TFLite model metadata"
                    .to_string(),
                "Verify channel order (RGB vs BGR) matches the training preprocessing pipeline"
                    .to_string(),
            ],
        });
    }

    append_quantized_artifact_findings(&mut findings, ops, tensors, weight_integrity);

    if byte_integrity.status != "assessed_clean" {
        let severe = byte_integrity.flatbuffer_archive_overlap_bytes > 0
            || byte_integrity.partial_buffer_overlap_count > 0
            || byte_integrity.metadata_archive_status == "malformed"
            || byte_integrity.conservation_status != "exact";
        findings.push(Finding {
            id: "artifact-byte-integrity".to_string(),
            title: if severe {
                "Artifact byte ownership or container integrity is inconsistent"
            } else {
                "Artifact byte ledger requires release review"
            }
            .to_string(),
            category: "integrity".to_string(),
            severity: if severe { "high" } else { "medium" }.to_string(),
            confidence: "static+wasm".to_string(),
            evidence: vec![
                FindingEvidence {
                    source: "DERIVED byte conservation".to_string(),
                    text: byte_integrity.detail.clone(),
                },
                FindingEvidence {
                    source: "Exact issues".to_string(),
                    text: byte_integrity.issues.join("; "),
                },
            ],
            impact: "Unowned suffix bytes, overlapping buffer ownership, malformed terminal metadata archives, or failed byte conservation can hide payloads or make downstream identity and provenance interpretations incomplete".to_string(),
            actions: vec![
                "Inspect the emitted byte offsets, remove unexplained suffix data, and rebuild the artifact from a known converter pipeline".to_string(),
                "Treat a terminal ZIP as TFLite associated-file metadata only when its bounded central directory, local records, names, and payload ranges validate".to_string(),
            ],
        });
    }

    // Sort: high first, then medium, then low
    let sev_rank = |s: &str| match s {
        "high" => 0u8,
        "medium" => 1,
        "low" => 2,
        _ => 3,
    };
    findings.sort_by_key(|f| sev_rank(&f.severity));
    findings
}
