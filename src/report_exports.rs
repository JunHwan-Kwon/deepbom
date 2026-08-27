use super::Analysis;

pub(super) fn render_stage_mermaid(analysis: &Analysis) -> String {
    let mut lines = vec!["flowchart LR".to_string()];
    for stage in &analysis.stages {
        let channels = if stage.channels.is_empty() {
            "-".to_string()
        } else {
            stage
                .channels
                .iter()
                .map(|channel| channel.to_string())
                .collect::<Vec<_>>()
                .join("/")
        };
        lines.push(format!(
            "  S{}[\"#{} ops {}-{}<br/>{}<br/>ops={} C={}<br/>MACs={} ({:.1}%)<br/>MAC-weighted delegated={:.1}% fallback={:.1}%<br/>op-count delegated={:.1}% fallback={:.1}%<br/>breaks={} patterns={}\"]",
            stage.index,
            stage.index,
            stage.first_op,
            stage.last_op,
            stage.key,
            stage.op_count,
            channels,
            fmt_number(stage.macs),
            stage.mac_percent * 100.0,
            stage.delegated_mac_percent * 100.0,
            stage.fallback_mac_percent * 100.0,
            stage.delegated_op_percent * 100.0,
            stage.fallback_op_percent * 100.0,
            stage.xnnpack_chain_breaks,
            if stage.patterns.is_empty() { "-".to_string() } else { stage.patterns.join("/") }
        ));
        if stage.index > 0 {
            lines.push(format!("  S{} --> S{}", stage.index - 1, stage.index));
        }
    }
    lines.join("\n")
}

pub(super) fn render_roofline_csv(analysis: &Analysis) -> String {
    let mut lines = Vec::<String>::new();
    lines.push("op_index,op_name,op_version,output_shapes,macs,ops,estimated_bytes,fallback_byte_percent,intensity_ops_per_byte,static_bound_guess,roofline_reason,row_working_set_kb,row_working_set_ratio,row_working_set_severity,xnnpack_supported,xnnpack_reason,xnnpack_chain_id,xnnpack_chain_break,xnnpack_break_class,xnnpack_kernel_candidate,xnnpack_kernel_candidate_count,xnnpack_kernel_candidate_families,xnnpack_kernel_architecture_conditions,xnnpack_kernel_compile_conditions,xnnpack_kernel_runtime_conditions,xnnpack_kernel_source_refs,xnnpack_kernel_source_file_sha256s,xnnpack_kernel_alignment_multiples,xnnpack_kernel_tile_mr,xnnpack_kernel_tile_nr,xnnpack_kernel_channel_tile,xnnpack_kernel_primary_tile,xnnpack_kernel_source,xnnpack_kernel_evidence_class,xnnpack_kernel_selector_status,xnnpack_selector_artifact_facts,xnnpack_unresolved_selector_dimensions,xnnpack_no_match_reason_code,xnnpack_candidate_tail_projections,xnnpack_build_requirement,chain_break_impact_mac_percent,chain_break_overhead_us_low,chain_break_overhead_us_high,weight_bytes,weight_packing_overhead_us,weight_packing_risk,output_channels,channel_alignment_multiple,channel_alignment_status,channel_tail_overhead_percent,channel_tail_overhead_percent_min,channel_tail_overhead_percent_max,quantized_path,quantized_compute_path,quantization_state,quantization_detail,quant_scale_mode,quant_scale_ratio_meaningful,quant_scale_ratio,quant_scale_cv,quant_zero_point_offset,quant_zero_point_status,quant_zero_point_risk,quant_risk,fused_activation,fusion_status,patterns,static_action".to_string());
    for op in &analysis.ops {
        let row = vec![
            op.index.to_string(),
            csv_escape(&op.name),
            op.version.to_string(),
            csv_escape(&format!("{:?}", op.output_shapes)),
            format!("{:.0}", op.macs),
            format!("{:.0}", op.ops),
            format!("{:.0}", op.estimated_bytes),
            format!("{:.6}", op.fallback_byte_percent),
            format!("{:.6}", op.intensity_ops_per_byte),
            csv_escape(&op.static_bound_guess),
            csv_escape(&op.roofline_reason),
            format!("{:.3}", op.row_working_set_bytes / 1024.0),
            format!("{:.6}", op.row_working_set_ratio),
            csv_escape(&op.row_working_set_severity),
            op.xnnpack_supported.to_string(),
            csv_escape(&op.xnnpack_reason),
            op.xnnpack_chain_id.to_string(),
            op.xnnpack_chain_break.to_string(),
            csv_escape(&op.xnnpack_break_class),
            csv_escape(&op.xnnpack_kernel_candidate),
            op.xnnpack_kernel_candidates.len().to_string(),
            csv_escape(
                &op.xnnpack_kernel_candidates
                    .iter()
                    .map(|item| item.family.as_str())
                    .collect::<Vec<_>>()
                    .join(" || "),
            ),
            csv_escape(
                &op.xnnpack_kernel_candidates
                    .iter()
                    .map(|item| item.architecture_condition.as_str())
                    .collect::<Vec<_>>()
                    .join(" || "),
            ),
            csv_escape(
                &op.xnnpack_kernel_candidates
                    .iter()
                    .map(|item| item.compile_condition.as_str())
                    .collect::<Vec<_>>()
                    .join(" || "),
            ),
            csv_escape(
                &op.xnnpack_kernel_candidates
                    .iter()
                    .map(|item| item.runtime_condition.as_str())
                    .collect::<Vec<_>>()
                    .join(" || "),
            ),
            csv_escape(
                &op.xnnpack_kernel_candidates
                    .iter()
                    .map(|item| item.source_ref.as_str())
                    .collect::<Vec<_>>()
                    .join(" || "),
            ),
            csv_escape(
                &op.xnnpack_kernel_candidates
                    .iter()
                    .map(|item| item.source_file_sha256.as_str())
                    .collect::<Vec<_>>()
                    .join(" || "),
            ),
            csv_escape(
                &op.xnnpack_kernel_alignment_multiples
                    .iter()
                    .map(|value| value.to_string())
                    .collect::<Vec<_>>()
                    .join("|"),
            ),
            op.xnnpack_kernel_tile_mr.to_string(),
            op.xnnpack_kernel_tile_nr.to_string(),
            op.xnnpack_kernel_channel_tile.to_string(),
            op.xnnpack_kernel_primary_tile.to_string(),
            csv_escape(&op.xnnpack_kernel_source),
            csv_escape(&op.xnnpack_kernel_evidence_class),
            csv_escape(&op.xnnpack_kernel_selector_status),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            csv_escape(&op.xnnpack_build_requirement),
            format!("{:.6}", op.chain_break_impact_mac_percent),
            format!("{:.1}", op.chain_break_overhead_us_low),
            format!("{:.1}", op.chain_break_overhead_us_high),
            format!("{:.0}", op.weight_bytes),
            format!("{:.2}", op.weight_packing_overhead_us),
            csv_escape(&op.weight_packing_risk),
            op.output_channels.to_string(),
            op.channel_alignment_multiple.to_string(),
            csv_escape(&op.channel_alignment_status),
            format!("{:.6}", op.channel_tail_overhead_percent),
            format!("{:.6}", op.channel_tail_overhead_percent_min),
            format!("{:.6}", op.channel_tail_overhead_percent_max),
            op.quantized_path.to_string(),
            op.quantized_compute_path.to_string(),
            csv_escape(&op.quantization_state),
            csv_escape(&op.quantization_detail),
            csv_escape(&op.quant_scale_mode),
            op.quant_scale_ratio_meaningful.to_string(),
            format!("{:.6}", op.quant_scale_ratio),
            format!("{:.6}", op.quant_scale_cv),
            op.quant_zero_point_offset.to_string(),
            csv_escape(&op.quant_zero_point_status),
            csv_escape(&op.quant_zero_point_risk),
            csv_escape(&op.quant_risk),
            csv_escape(&op.fused_activation),
            csv_escape(&op.fusion_status),
            csv_escape(&op.patterns.join("|")),
            csv_escape(&op.static_action),
        ];
        lines.push(row.join(","));
    }
    lines.join("\n")
}

fn fmt_number(value: f64) -> String {
    let rounded = value.round() as i128;
    let raw = rounded.abs().to_string();
    let mut output = String::new();
    for (index, character) in raw.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            output.push(',');
        }
        output.push(character);
    }
    let result = output.chars().rev().collect::<String>();
    if rounded < 0 {
        format!("-{}", result)
    } else {
        result
    }
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}
