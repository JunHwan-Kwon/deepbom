use super::*;

fn haar_decompose_patch(patch: &[f64], kh: usize, kw: usize) -> KernelHaarEnergy {
    if patch.is_empty() || kh < 2 || kw < 2 {
        return KernelHaarEnergy {
            ll: 1.0,
            lh: 0.0,
            hl: 0.0,
            hh: 0.0,
            center_surround: 0.0,
        };
    }
    let n = (kh * kw) as f64;

    // LL: mean (DC)
    let ll_val = patch.iter().sum::<f64>() / n;

    // LH: horizontal edge — top half mean minus bottom half mean
    let h2 = kh / 2;
    let top_sum: f64 = (0..h2)
        .map(|r| patch[r * kw..(r + 1) * kw].iter().sum::<f64>())
        .sum();
    let bot_sum: f64 = (h2..kh)
        .map(|r| patch[r * kw..(r + 1) * kw].iter().sum::<f64>())
        .sum();
    let lh_val = (top_sum / (h2 * kw) as f64 - bot_sum / ((kh - h2) * kw) as f64) * 0.5;

    // HL: vertical edge — left half mean minus right half mean
    let w2 = kw / 2;
    let left_sum: f64 = (0..kh)
        .map(|r| patch[r * kw..r * kw + w2].iter().sum::<f64>())
        .sum();
    let right_sum: f64 = (0..kh)
        .map(|r| patch[r * kw + w2..(r + 1) * kw].iter().sum::<f64>())
        .sum();
    let hl_val = (left_sum / (kh * w2) as f64 - right_sum / (kh * (kw - w2)) as f64) * 0.5;

    // HH: checkerboard correlation
    let hh_val: f64 = patch
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if (i / kw + i % kw).is_multiple_of(2) {
                v
            } else {
                -v
            }
        })
        .sum::<f64>()
        / n;

    // Center-surround (only for odd-size kernels)
    let cs_val = if kh % 2 == 1 && kw % 2 == 1 {
        let center = patch[(kh / 2) * kw + kw / 2];
        let surround = (patch.iter().sum::<f64>() - center) / (n - 1.0).max(1.0);
        center - surround
    } else {
        0.0
    };

    let e_ll = ll_val * ll_val;
    let e_lh = lh_val * lh_val;
    let e_hl = hl_val * hl_val;
    let e_hh = hh_val * hh_val;
    let total_e = (e_ll + e_lh + e_hl + e_hh).max(1e-30);

    KernelHaarEnergy {
        ll: (e_ll / total_e) as f32,
        lh: (e_lh / total_e) as f32,
        hl: (e_hl / total_e) as f32,
        hh: (e_hh / total_e) as f32,
        center_surround: cs_val as f32,
    }
}

#[wasm_bindgen]
pub fn compute_kernel_haar_decomposition(
    bytes: &[u8],
    filename: &str,
    target_id: &str,
) -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    let tid = if target_id.is_empty() {
        "android_mid_a55"
    } else {
        target_id
    };
    let analysis = analyze_with_target_without_step_response(bytes, filename, tid)
        .map_err(|e| JsValue::from_str(&e))?;

    let mut op_results: Vec<KernelHaarOpResult> = Vec::new();

    for op in &analysis.ops {
        let op_name = op.name.as_str();
        if op_name != "CONV_2D" && op_name != "DEPTHWISE_CONV_2D" {
            continue;
        }

        let w_idx = match op.inputs.get(1).copied() {
            Some(i) if i >= 0 => i as usize,
            _ => continue,
        };
        if w_idx >= analysis.tensors.len() {
            continue;
        }
        let wt = &analysis.tensors[w_idx];
        if !wt.constant_buffer || wt.sparse_storage || wt.shape.len() != 4 {
            continue;
        }

        let out_ch = wt.shape[0] as usize;
        let k_h = wt.shape[1] as usize;
        let k_w = wt.shape[2] as usize;
        let in_ch = wt.shape[3] as usize;
        if k_h < 2 || k_w < 2 {
            continue;
        }

        let end = wt.buffer_data_offset + wt.buffer_data_length;
        if end > bytes.len() {
            continue;
        }
        let raw = &bytes[wt.buffer_data_offset..end];

        let elem_count = out_ch * k_h * k_w * in_ch;
        let filter_size = k_h * k_w * in_ch;
        let is_depthwise = op_name == "DEPTHWISE_CONV_2D";

        // Per-channel dequantization (quant axis = out_ch for CONV_2D, in_ch for DEPTHWISE)
        let per_ch_scales: Vec<f64> = wt.scale_sample.iter().map(|&s| s as f64).collect();
        let per_ch_zps: Vec<f64> = wt.zero_point_sample.iter().map(|&z| z as f64).collect();
        let quant_ch_count = if is_depthwise { in_ch } else { out_ch };
        let is_per_channel = !per_ch_scales.is_empty() && per_ch_scales.len() == quant_ch_count;
        let global_scale = per_ch_scales.first().copied().unwrap_or(1.0);
        let global_zp = per_ch_zps.first().copied().unwrap_or(0.0);
        // dequant: (raw_int - zero_point) * scale; no-op for float dtypes (scale=1, zp=0)
        let dequant = |raw_val: f64, flat_idx: usize| -> f64 {
            if !is_per_channel {
                (raw_val - global_zp) * global_scale
            } else {
                let ch = if is_depthwise {
                    flat_idx % in_ch
                } else {
                    flat_idx / filter_size
                };
                let s = per_ch_scales.get(ch).copied().unwrap_or(global_scale);
                let z = per_ch_zps.get(ch).copied().unwrap_or(global_zp);
                (raw_val - z) * s
            }
        };

        let values: Vec<f64> = match wt.dtype.as_str() {
            "INT8" if raw.len() >= elem_count => (0..elem_count)
                .map(|i| dequant((raw[i] as i8) as f64, i))
                .collect(),
            "UINT8" if raw.len() >= elem_count => {
                (0..elem_count).map(|i| dequant(raw[i] as f64, i)).collect()
            }
            "INT16" if raw.len() >= elem_count * 2 => (0..elem_count)
                .map(|i| {
                    let o = i * 2;
                    dequant(i16::from_le_bytes([raw[o], raw[o + 1]]) as f64, i)
                })
                .collect(),
            "FLOAT16" if raw.len() >= elem_count * 2 => (0..elem_count)
                .map(|i| {
                    let o = i * 2;
                    f16_to_f32(u16::from_le_bytes([raw[o], raw[o + 1]])) as f64
                })
                .collect(),
            "FLOAT32" if raw.len() >= elem_count * 4 => (0..elem_count)
                .map(|i| {
                    let o = i * 4;
                    f32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]) as f64
                })
                .collect(),
            _ => continue,
        };

        // Single-pass for max_abs + total_abs; separate near-zero count fixes all-zero case
        let (max_abs, total_abs) = values.iter().fold((0.0f64, 0.0f64), |(mx, tot), &v| {
            (mx.max(v.abs()), tot + v.abs())
        });
        let near_zero_count = if max_abs == 0.0 {
            values.len() // all dequantized weights are zero → fully sparse
        } else {
            let t = max_abs * 0.01;
            values.iter().filter(|&&v| v.abs() < t).count()
        };
        let filter_vol = k_h * k_w * in_ch;
        let energy_proxy =
            (total_abs / values.len().max(1) as f64 / (filter_vol as f64).sqrt()) as f32;
        let sparsity_proxy = near_zero_count as f32 / values.len().max(1) as f32;

        let num_filters = if op_name == "DEPTHWISE_CONV_2D" {
            in_ch
        } else {
            out_ch
        };
        let mut haar_energies: Vec<KernelHaarEnergy> = Vec::with_capacity(num_filters);

        if op_name == "DEPTHWISE_CONV_2D" {
            for ic in 0..in_ch {
                let patch: Vec<f64> = (0..k_h * k_w)
                    .map(|i| {
                        let ky = i / k_w;
                        let kx = i % k_w;
                        values
                            .get(ky * k_w * in_ch + kx * in_ch + ic)
                            .copied()
                            .unwrap_or(0.0)
                    })
                    .collect();
                haar_energies.push(haar_decompose_patch(&patch, k_h, k_w));
            }
        } else {
            for oc in 0..out_ch {
                // Average across input channels to get the dominant spatial pattern per filter
                let patch: Vec<f64> = (0..k_h * k_w)
                    .map(|i| {
                        let ky = i / k_w;
                        let kx = i % k_w;
                        let base = oc * k_h * k_w * in_ch + ky * k_w * in_ch + kx * in_ch;
                        values[base..base + in_ch].iter().sum::<f64>() / in_ch.max(1) as f64
                    })
                    .collect();
                haar_energies.push(haar_decompose_patch(&patch, k_h, k_w));
            }
        }

        let n = haar_energies.len().max(1) as f32;
        let mean_ll = haar_energies.iter().map(|e| e.ll).sum::<f32>() / n;
        let mean_lh = haar_energies.iter().map(|e| e.lh).sum::<f32>() / n;
        let mean_hl = haar_energies.iter().map(|e| e.hl).sum::<f32>() / n;
        let mean_hh = haar_energies.iter().map(|e| e.hh).sum::<f32>() / n;
        let mean_cs = haar_energies.iter().map(|e| e.center_surround).sum::<f32>() / n;

        let (mut dominant_ll, mut dominant_lh, mut dominant_hl, mut dominant_hh) =
            (0usize, 0usize, 0usize, 0usize);
        for e in &haar_energies {
            let max_f = e.ll.max(e.lh).max(e.hl).max(e.hh);
            // max_f is one of the four f32 values; >= comparison is exact
            if e.ll >= max_f {
                dominant_ll += 1;
            } else if e.lh >= max_f {
                dominant_lh += 1;
            } else if e.hl >= max_f {
                dominant_hl += 1;
            } else {
                dominant_hh += 1;
            } // e.hh == max_f
        }

        let dominant = *["ll", "lh", "hl", "hh"]
            .iter()
            .zip([mean_ll, mean_lh, mean_hl, mean_hh])
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(s, _)| s)
            .unwrap_or(&"ll");

        op_results.push(KernelHaarOpResult {
            op_index: op.index,
            op_name: op.name.clone(),
            num_filters,
            kernel_h: k_h,
            kernel_w: k_w,
            in_channels: in_ch,
            mean_energy: KernelHaarEnergy {
                ll: mean_ll,
                lh: mean_lh,
                hl: mean_hl,
                hh: mean_hh,
                center_surround: mean_cs,
            },
            dominant: dominant.to_string(),
            dominant_ll,
            dominant_lh,
            dominant_hl,
            dominant_hh,
            orientation_ratio: mean_lh / mean_hl.max(1e-9),
            edge_dc_ratio: (mean_lh + mean_hl + mean_hh) / mean_ll.max(1e-9),
            energy_proxy,
            sparsity_proxy,
        });
    }

    // Global summary
    let conv_op_count = op_results.len();
    let edge_heavy_ops = op_results.iter().filter(|o| o.edge_dc_ratio > 0.5).count();
    let dc_heavy_ops = op_results.iter().filter(|o| o.mean_energy.ll > 0.6).count();
    let lh_dominant_ops = op_results.iter().filter(|o| o.dominant == "lh").count();
    let hl_dominant_ops = op_results.iter().filter(|o| o.dominant == "hl").count();
    let hh_dominant_ops = op_results.iter().filter(|o| o.dominant == "hh").count();
    let ll_dominant_ops = op_results.iter().filter(|o| o.dominant == "ll").count();

    let orientation_bias = if conv_op_count > 0 {
        op_results
            .iter()
            .map(|o| o.mean_energy.lh - o.mean_energy.hl)
            .sum::<f32>()
            / conv_op_count as f32
    } else {
        0.0
    };

    let global_dominant = *[
        ("ll", ll_dominant_ops),
        ("lh", lh_dominant_ops),
        ("hl", hl_dominant_ops),
        ("hh", hh_dominant_ops),
    ]
    .iter()
    .max_by_key(|&&(_, c)| c)
    .map(|(s, _)| s)
    .unwrap_or(&"ll");

    let result = KernelHaarResult {
        ops: op_results,
        summary: KernelHaarSummary {
            conv_op_count,
            edge_heavy_ops,
            dc_heavy_ops,
            lh_dominant_ops,
            hl_dominant_ops,
            hh_dominant_ops,
            ll_dominant_ops,
            orientation_bias,
            global_dominant: global_dominant.to_string(),
        },
    };

    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Activation Haar Decomposition — synthetic forward pass in f64
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Serialize)]
struct ActivationHaarOpResult {
    op_index: usize,
    op_name: String,
    output_tensor_index: usize,
    output_tensor_name: String,
    output_shape: Vec<usize>,
    ll_energy: f64,
    lh_energy: f64,
    hl_energy: f64,
    hh_energy: f64,
    dominant_band: String,
    spatial_smoothness: f64,
    spatial_too_small: bool,
    skipped: bool,
    skip_reason: String,
}

// Read op builtin_options table for any op table
pub(super) fn op_options_table(fb: &Fb, op_table: usize) -> Option<usize> {
    fb.table_field(op_table, 4)
}

// conv2d opts: (same_pad, sh, sw, dh, dw, act_code)
pub(super) fn parse_conv2d_opts(
    fb: &Fb,
    opts: Option<usize>,
) -> (bool, usize, usize, usize, usize, i8) {
    let Some(t) = opts else {
        return (false, 1, 1, 1, 1, 0);
    };
    let pad = fb.field_pos(t, 0).and_then(|p| fb.i8(p)).unwrap_or(0);
    let sw = fb
        .field_pos(t, 1)
        .and_then(|p| fb.i32(p))
        .unwrap_or(1)
        .max(1) as usize;
    let sh = fb
        .field_pos(t, 2)
        .and_then(|p| fb.i32(p))
        .unwrap_or(1)
        .max(1) as usize;
    let act = fb.field_pos(t, 3).and_then(|p| fb.i8(p)).unwrap_or(0);
    let dw = fb
        .field_pos(t, 4)
        .and_then(|p| fb.i32(p))
        .unwrap_or(1)
        .max(1) as usize;
    let dh = fb
        .field_pos(t, 5)
        .and_then(|p| fb.i32(p))
        .unwrap_or(1)
        .max(1) as usize;
    (padding_is_same(pad), sh, sw, dh, dw, act)
}

// depthwise opts: (same_pad, sh, sw, dh, dw, act_code)
pub(super) fn parse_dw_conv_opts(
    fb: &Fb,
    opts: Option<usize>,
) -> (bool, usize, usize, usize, usize, i8) {
    let Some(t) = opts else {
        return (false, 1, 1, 1, 1, 0);
    };
    let pad = fb.field_pos(t, 0).and_then(|p| fb.i8(p)).unwrap_or(0);
    let sw = fb
        .field_pos(t, 1)
        .and_then(|p| fb.i32(p))
        .unwrap_or(1)
        .max(1) as usize;
    let sh = fb
        .field_pos(t, 2)
        .and_then(|p| fb.i32(p))
        .unwrap_or(1)
        .max(1) as usize;
    let act = fb.field_pos(t, 4).and_then(|p| fb.i8(p)).unwrap_or(0);
    let dw = fb
        .field_pos(t, 5)
        .and_then(|p| fb.i32(p))
        .unwrap_or(1)
        .max(1) as usize;
    let dh = fb
        .field_pos(t, 6)
        .and_then(|p| fb.i32(p))
        .unwrap_or(1)
        .max(1) as usize;
    (padding_is_same(pad), sh, sw, dh, dw, act)
}

// pool opts: (same_pad, sh, sw, fh, fw, act_code)
fn parse_pool2d_opts(fb: &Fb, opts: Option<usize>) -> (bool, usize, usize, usize, usize, i8) {
    let Some(t) = opts else {
        return (false, 1, 1, 2, 2, 0);
    };
    let pad = fb.field_pos(t, 0).and_then(|p| fb.i8(p)).unwrap_or(0);
    let sw = fb
        .field_pos(t, 1)
        .and_then(|p| fb.i32(p))
        .unwrap_or(1)
        .max(1) as usize;
    let sh = fb
        .field_pos(t, 2)
        .and_then(|p| fb.i32(p))
        .unwrap_or(1)
        .max(1) as usize;
    let fw = fb
        .field_pos(t, 3)
        .and_then(|p| fb.i32(p))
        .unwrap_or(2)
        .max(1) as usize;
    let fh = fb
        .field_pos(t, 4)
        .and_then(|p| fb.i32(p))
        .unwrap_or(2)
        .max(1) as usize;
    let act = fb.field_pos(t, 5).and_then(|p| fb.i8(p)).unwrap_or(0);
    (padding_is_same(pad), sh, sw, fh, fw, act)
}

#[inline(always)]
fn padding_is_same(value: i8) -> bool {
    // TensorFlow Lite schema enum Padding: SAME = 0, VALID = 1.
    value == 0
}

#[inline(always)]
fn fused_act_f64(v: f64, act: i8) -> f64 {
    match act {
        1 => v.max(0.0),
        2 => v.clamp(-1.0, 1.0),
        3 => v.clamp(0.0, 6.0),
        4 => v.tanh(),
        _ => v,
    }
}

// Compute output size and top/left padding for SAME/VALID
pub(super) fn conv_out_size(
    in_size: usize,
    k: usize,
    stride: usize,
    dilation: usize,
    same: bool,
) -> (usize, usize) {
    let eff_k = (k - 1) * dilation + 1;
    if same {
        let out = in_size.div_ceil(stride);
        let pad_total = ((out.saturating_sub(1)) * stride + eff_k).saturating_sub(in_size);
        (out, pad_total / 2)
    } else {
        if in_size < eff_k {
            return (0, 0);
        }
        ((in_size - eff_k) / stride + 1, 0)
    }
}

// Dequantize a constant weight tensor to Vec<f64>.
// Returns None if not a constant buffer or dtype unsupported.
pub(super) fn weight_to_f64(bytes: &[u8], t: &TensorInfo) -> Option<Vec<f64>> {
    let raw = extract_tensor_buffer(bytes, t)?;
    let shape: Vec<usize> = t.shape.iter().map(|&d| d.max(1) as usize).collect();
    let elem: usize = shape.iter().product();
    match t.dtype.as_str() {
        "FLOAT32" if raw.len() >= elem * 4 => Some(
            (0..elem)
                .map(|i| {
                    f32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]])
                        as f64
                })
                .collect(),
        ),
        "FLOAT16" if raw.len() >= elem * 2 => Some(
            (0..elem)
                .map(|i| f16_to_f32(u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]])) as f64)
                .collect(),
        ),
        "INT8" if raw.len() >= elem => {
            // Weight layout [oc, kH, kW, ic] for CONV_2D; [1, kH, kW, ic] for DW
            let oc = shape.first().copied().unwrap_or(1);
            let ic = shape.last().copied().unwrap_or(1);
            let is_dw = oc == 1 && ic > 1 && t.scale_sample.len() == ic;
            let filter_size = if elem > 0 && oc > 0 { elem / oc } else { 1 };
            Some(
                (0..elem)
                    .map(|i| {
                        let raw_v = (raw[i] as i8) as f64;
                        if is_dw {
                            let ch = i % ic;
                            let s = t.scale_sample.get(ch).copied().unwrap_or(1.0) as f64;
                            let z = t.zero_point_sample.get(ch).copied().unwrap_or(0) as f64;
                            (raw_v - z) * s
                        } else {
                            let ch = i / filter_size;
                            let s = t
                                .scale_sample
                                .get(ch)
                                .copied()
                                .or_else(|| t.scale_sample.first().copied())
                                .unwrap_or(1.0) as f64;
                            let z = t.zero_point_sample.get(ch).copied().unwrap_or(0) as f64;
                            (raw_v - z) * s
                        }
                    })
                    .collect(),
            )
        }
        "UINT8" if raw.len() >= elem => {
            let s0 = t.scale_sample.first().copied().unwrap_or(1.0) as f64;
            let z0 = t.zero_point_sample.first().copied().unwrap_or(0) as f64;
            Some(
                raw.iter()
                    .take(elem)
                    .map(|&b| (b as f64 - z0) * s0)
                    .collect(),
            )
        }
        _ => None,
    }
}

// Decode bias tensor to f64, scaling INT32 biases by input_scale * per-channel weight_scale
pub(super) fn bias_to_f64(
    bytes: &[u8],
    t: &TensorInfo,
    in_scale: f64,
    wt: &TensorInfo,
) -> Vec<f64> {
    let raw = match extract_tensor_buffer(bytes, t) {
        Some(r) => r,
        None => return Vec::new(),
    };
    let len = raw.len();
    match t.dtype.as_str() {
        "FLOAT32" if len >= 4 => (0..len / 4)
            .map(|i| {
                f32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]])
                    as f64
            })
            .collect(),
        "INT32" if len >= 4 => (0..len / 4)
            .map(|i| {
                let v = i32::from_le_bytes([
                    raw[i * 4],
                    raw[i * 4 + 1],
                    raw[i * 4 + 2],
                    raw[i * 4 + 3],
                ]);
                let ws = wt
                    .scale_sample
                    .get(i)
                    .copied()
                    .or_else(|| wt.scale_sample.first().copied())
                    .unwrap_or(1.0) as f64;
                v as f64 * in_scale * ws
            })
            .collect(),
        _ => Vec::new(),
    }
}

// Build synthetic input for a model input tensor, capped to max_hw per spatial dim.
// Returns (data_f64, actual_h, actual_w, c)
fn make_synthetic_input(t: &TensorInfo, max_hw: usize) -> (Vec<f64>, usize, usize, usize) {
    let shape: Vec<usize> = t.shape.iter().map(|&d| d as usize).collect();
    // Support [N,H,W,C] and [N,C] (batch=1)
    let (h, w, c) = match shape.len() {
        4 => (
            shape[1].max(1).min(max_hw),
            shape[2].max(1).min(max_hw),
            shape[3].max(1),
        ),
        3 => (shape[1].max(1).min(max_hw), shape[2].max(1).min(max_hw), 1),
        2 => (1, 1, shape[1].max(1)),
        _ => (1, 1, 1),
    };
    let s0 = t.scale_sample.first().copied().unwrap_or(1.0) as f64;
    let z0 = t.zero_point_sample.first().copied().unwrap_or(0) as f64;
    // Amplitude: ~30% of representable range for the dtype
    let amp = match t.dtype.to_uppercase().as_str() {
        "INT8" => 30.0,
        "UINT8" => 30.0,
        "FLOAT32" | "FLOAT16" => 0.3,
        "INT32" => 1000.0,
        _ => 0.3,
    };
    let mut data = Vec::with_capacity(h * w * c);
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                // Deterministic spatial pattern with per-channel phase shift
                #[allow(clippy::approx_constant)]
                let t_val = (y as f64 * 0.785 + x as f64 * 0.5236 + ch as f64 * 0.3141)
                    % (2.0 * std::f64::consts::PI);
                let sig = t_val.sin() * amp;
                let v = match t.dtype.to_uppercase().as_str() {
                    "INT8" => (sig + z0).clamp(-128.0, 127.0),
                    "UINT8" => (sig + z0).clamp(0.0, 255.0),
                    _ => sig,
                };
                // Dequantize to float domain if quantized
                let f = match t.dtype.to_uppercase().as_str() {
                    "INT8" | "UINT8" => (v - z0) * s0,
                    "FLOAT32" | "FLOAT16" => v,
                    _ => v * s0,
                };
                data.push(f);
            }
        }
    }
    (data, h, w, c)
}

// Single-level 2D Haar energy (LL, LH, HL, HH) across all channels.
// buf layout: [h, w, c] flattened in HWC order.
fn haar_energy_hwc(buf: &[f64], h: usize, w: usize, c: usize) -> [f64; 4] {
    let ph = h & !1;
    let pw = w & !1;
    if ph < 2 || pw < 2 || c == 0 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let (mut ll, mut lh, mut hl, mut hh) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for hy in (0..ph).step_by(2) {
        for hx in (0..pw).step_by(2) {
            for ch in 0..c {
                let a = buf[hy * w * c + hx * c + ch];
                let b = buf[hy * w * c + (hx + 1) * c + ch];
                let cv = buf[(hy + 1) * w * c + hx * c + ch];
                let d = buf[(hy + 1) * w * c + (hx + 1) * c + ch];
                ll += ((a + b + cv + d) * 0.25).powi(2);
                lh += ((a + b - cv - d) * 0.25).powi(2); // LH: top-bot diff (horizontal edge)
                hl += ((a - b + cv - d) * 0.25).powi(2); // HL: left-right diff (vertical edge)
                hh += ((a - b - cv + d) * 0.25).powi(2);
            }
        }
    }
    [ll, lh, hl, hh]
}

// CONV_2D forward pass in f64 (HWC layout throughout, weight [oc, kh, kw, ic])
#[allow(clippy::too_many_arguments)]
fn fwd_conv2d(
    inp: &[f64],
    ih: usize,
    iw: usize,
    ic: usize,
    wt: &[f64],
    oc: usize,
    kh: usize,
    kw: usize,
    bias: &[f64],
    sh: usize,
    sw: usize,
    dh: usize,
    dw: usize,
    same: bool,
    act: i8,
) -> (Vec<f64>, usize, usize) {
    let (oh, ph) = conv_out_size(ih, kh, sh, dh, same);
    let (ow, pw) = conv_out_size(iw, kw, sw, dw, same);
    if oh == 0 || ow == 0 || oc == 0 {
        return (vec![], 0, 0);
    }
    let mut out = vec![0.0f64; oh * ow * oc];
    for o_h in 0..oh {
        for o_w in 0..ow {
            for o_c in 0..oc {
                let mut acc = bias.get(o_c).copied().unwrap_or(0.0);
                for k_h in 0..kh {
                    let i_h = (o_h * sh + k_h * dh) as isize - ph as isize;
                    if i_h < 0 || i_h >= ih as isize {
                        continue;
                    }
                    let i_h = i_h as usize;
                    for k_w in 0..kw {
                        let i_w = (o_w * sw + k_w * dw) as isize - pw as isize;
                        if i_w < 0 || i_w >= iw as isize {
                            continue;
                        }
                        let i_w = i_w as usize;
                        for i_c in 0..ic {
                            acc += inp[i_h * iw * ic + i_w * ic + i_c]
                                * wt[o_c * kh * kw * ic + k_h * kw * ic + k_w * ic + i_c];
                        }
                    }
                }
                out[o_h * ow * oc + o_w * oc + o_c] = fused_act_f64(acc, act);
            }
        }
    }
    (out, oh, ow)
}

// DEPTHWISE_CONV_2D forward (weight layout [1, kh, kw, ic] strided by ic)
#[allow(clippy::too_many_arguments)]
fn fwd_dw_conv2d(
    inp: &[f64],
    ih: usize,
    iw: usize,
    ic: usize,
    wt: &[f64],
    kh: usize,
    kw: usize,
    bias: &[f64],
    sh: usize,
    sw: usize,
    dh: usize,
    dw: usize,
    same: bool,
    act: i8,
) -> (Vec<f64>, usize, usize) {
    let (oh, ph) = conv_out_size(ih, kh, sh, dh, same);
    let (ow, pw) = conv_out_size(iw, kw, sw, dw, same);
    if oh == 0 || ow == 0 {
        return (vec![], 0, 0);
    }
    let mut out = vec![0.0f64; oh * ow * ic];
    for o_h in 0..oh {
        for o_w in 0..ow {
            for ch in 0..ic {
                let mut acc = bias.get(ch).copied().unwrap_or(0.0);
                for k_h in 0..kh {
                    let i_h = (o_h * sh + k_h * dh) as isize - ph as isize;
                    if i_h < 0 || i_h >= ih as isize {
                        continue;
                    }
                    let i_h = i_h as usize;
                    for k_w in 0..kw {
                        let i_w = (o_w * sw + k_w * dw) as isize - pw as isize;
                        if i_w < 0 || i_w >= iw as isize {
                            continue;
                        }
                        let i_w = i_w as usize;
                        acc += inp[i_h * iw * ic + i_w * ic + ch] * wt[(k_h * kw + k_w) * ic + ch];
                    }
                }
                out[o_h * ow * ic + o_w * ic + ch] = fused_act_f64(acc, act);
            }
        }
    }
    (out, oh, ow)
}

// AVERAGE_POOL_2D
#[allow(clippy::too_many_arguments)]
fn fwd_avg_pool(
    inp: &[f64],
    ih: usize,
    iw: usize,
    ic: usize,
    fh: usize,
    fw: usize,
    sh: usize,
    sw: usize,
    same: bool,
    act: i8,
) -> (Vec<f64>, usize, usize) {
    let (oh, ph) = conv_out_size(ih, fh, sh, 1, same);
    let (ow, pw) = conv_out_size(iw, fw, sw, 1, same);
    if oh == 0 || ow == 0 {
        return (vec![], 0, 0);
    }
    let mut out = vec![0.0f64; oh * ow * ic];
    for o_h in 0..oh {
        for o_w in 0..ow {
            for ch in 0..ic {
                let mut acc = 0.0f64;
                let mut cnt = 0usize;
                for f_h in 0..fh {
                    let i_h = (o_h * sh + f_h) as isize - ph as isize;
                    if i_h < 0 || i_h >= ih as isize {
                        continue;
                    }
                    let i_h = i_h as usize;
                    for f_w in 0..fw {
                        let i_w = (o_w * sw + f_w) as isize - pw as isize;
                        if i_w < 0 || i_w >= iw as isize {
                            continue;
                        }
                        let i_w = i_w as usize;
                        acc += inp[i_h * iw * ic + i_w * ic + ch];
                        cnt += 1;
                    }
                }
                let v = if cnt > 0 { acc / cnt as f64 } else { 0.0 };
                out[o_h * ow * ic + o_w * ic + ch] = fused_act_f64(v, act);
            }
        }
    }
    (out, oh, ow)
}

// MAX_POOL_2D
#[allow(clippy::too_many_arguments)]
fn fwd_max_pool(
    inp: &[f64],
    ih: usize,
    iw: usize,
    ic: usize,
    fh: usize,
    fw: usize,
    sh: usize,
    sw: usize,
    same: bool,
    act: i8,
) -> (Vec<f64>, usize, usize) {
    let (oh, ph) = conv_out_size(ih, fh, sh, 1, same);
    let (ow, pw) = conv_out_size(iw, fw, sw, 1, same);
    if oh == 0 || ow == 0 {
        return (vec![], 0, 0);
    }
    let mut out = vec![0.0f64; oh * ow * ic];
    for o_h in 0..oh {
        for o_w in 0..ow {
            for ch in 0..ic {
                let mut best = f64::NEG_INFINITY;
                for f_h in 0..fh {
                    let i_h = (o_h * sh + f_h) as isize - ph as isize;
                    if i_h < 0 || i_h >= ih as isize {
                        continue;
                    }
                    let i_h = i_h as usize;
                    for f_w in 0..fw {
                        let i_w = (o_w * sw + f_w) as isize - pw as isize;
                        if i_w < 0 || i_w >= iw as isize {
                            continue;
                        }
                        let i_w = i_w as usize;
                        let v = inp[i_h * iw * ic + i_w * ic + ch];
                        if v > best {
                            best = v;
                        }
                    }
                }
                let v = if best == f64::NEG_INFINITY { 0.0 } else { best };
                out[o_h * ow * ic + o_w * ic + ch] = fused_act_f64(v, act);
            }
        }
    }
    (out, oh, ow)
}

// FULLY_CONNECTED [N, units] → [N, oc]
// input: [h*w*ic] flattened; weight: [oc, ic]; bias: [oc]
fn fwd_fc(inp: &[f64], ic: usize, wt: &[f64], oc: usize, bias: &[f64], act: i8) -> Vec<f64> {
    if ic == 0 || oc == 0 {
        return vec![];
    }
    (0..oc)
        .map(|o_c| {
            let mut acc = bias.get(o_c).copied().unwrap_or(0.0);
            for i_c in 0..ic.min(wt.len() / oc) {
                acc += inp.get(i_c).copied().unwrap_or(0.0) * wt[o_c * ic + i_c];
            }
            fused_act_f64(acc, act)
        })
        .collect()
}

// Element-wise ADD with broadcast (both tensors must have same total size or scalar second)
fn fwd_add(a: &[f64], b: &[f64], act: i8) -> Vec<f64> {
    if a.is_empty() {
        return b.to_vec();
    }
    if b.is_empty() {
        return a.to_vec();
    }
    if b.len() == 1 {
        let bv = b[0];
        return a.iter().map(|&av| fused_act_f64(av + bv, act)).collect();
    }
    a.iter()
        .zip(b.iter().cycle().take(a.len()))
        .map(|(&av, &bv)| fused_act_f64(av + bv, act))
        .collect()
}

fn fwd_mul(a: &[f64], b: &[f64], act: i8) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    if b.len() == 1 {
        let bv = b[0];
        return a.iter().map(|&av| fused_act_f64(av * bv, act)).collect();
    }
    a.iter()
        .zip(b.iter().cycle().take(a.len()))
        .map(|(&av, &bv)| fused_act_f64(av * bv, act))
        .collect()
}

fn fwd_relu(inp: &[f64], act: i8) -> Vec<f64> {
    inp.iter().map(|&v| fused_act_f64(v, act)).collect()
}

fn dominant_band(ll: f64, lh: f64, hl: f64, hh: f64) -> &'static str {
    let vals = [(ll, "LL"), (lh, "LH"), (hl, "HL"), (hh, "HH")];
    vals.iter()
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .map(|x| x.1)
        .unwrap_or("LL")
}

/// Run a synthetic forward pass through all ops and compute 2D Haar energy on each
/// CONV_2D / DEPTHWISE_CONV_2D output activation.
fn run_activation_haar(bytes: &[u8], analysis: &Analysis) -> Vec<ActivationHaarOpResult> {
    const MAX_HW: usize = 32; // cap spatial dims for synthetic input to keep pass fast

    let fb = match Fb::verified_tflite(bytes) {
        Ok(fb) => fb,
        Err(_) => return vec![],
    };
    let model = match fb.root_table() {
        Ok(r) => r,
        Err(_) => return vec![],
    };
    let subgraph_tables = fb.vector_tables(model, 2);
    let subgraph = match subgraph_tables.first().copied() {
        Some(s) => s,
        None => return vec![],
    };
    let op_tables = fb.vector_tables(subgraph, 3);

    // tensor_index → (data_f64, h, w, c)
    let mut buffers: HashMap<usize, (Vec<f64>, usize, usize, usize)> = HashMap::new();

    // Seed model inputs with synthetic data
    for &idx in &analysis.input_tensor_indices {
        let idx = idx as usize;
        if let Some(t) = analysis.tensors.get(idx) {
            let (data, h, w, c) = make_synthetic_input(t, MAX_HW);
            buffers.insert(idx, (data, h, w, c));
        }
    }

    // Pre-seed constant tensors (weights/biases) are loaded on demand via analysis.tensors.

    let mut results: Vec<ActivationHaarOpResult> = Vec::new();

    for (op_idx, (&op_table, op)) in op_tables.iter().zip(analysis.ops.iter()).enumerate() {
        let name = op.name.as_str();
        let opts = op_options_table(&fb, op_table);
        let inp_idx = |n: usize| op.inputs.get(n).copied().unwrap_or(-1) as usize;
        let out_idx = |n: usize| op.outputs.get(n).copied().unwrap_or(-1) as usize;

        let get_buf = |idx: usize| buffers.get(&idx).cloned();
        let get_tensor = |idx: usize| analysis.tensors.get(idx);

        match name {
            "CONV_2D" => {
                let (same, sh, sw, dh, dw, act) = parse_conv2d_opts(&fb, opts);
                let Some((inp, ih, iw, ic)) = get_buf(inp_idx(0)) else {
                    push_skipped(
                        &mut results,
                        op_idx,
                        op,
                        out_idx(0),
                        &analysis.tensors,
                        "input buffer missing",
                    );
                    continue;
                };
                let wt_info = match get_tensor(inp_idx(1)) {
                    Some(t) => t,
                    None => {
                        push_skipped(
                            &mut results,
                            op_idx,
                            op,
                            out_idx(0),
                            &analysis.tensors,
                            "weight tensor missing",
                        );
                        continue;
                    }
                };
                let wt_data = match weight_to_f64(bytes, wt_info) {
                    Some(d) => d,
                    None => {
                        push_skipped(
                            &mut results,
                            op_idx,
                            op,
                            out_idx(0),
                            &analysis.tensors,
                            "weight dtype unsupported",
                        );
                        continue;
                    }
                };
                let oc = wt_info.shape.first().copied().unwrap_or(0) as usize;
                let kh = wt_info.shape.get(1).copied().unwrap_or(1) as usize;
                let kw = wt_info.shape.get(2).copied().unwrap_or(1) as usize;
                let in_scale = get_tensor(inp_idx(0))
                    .and_then(|t| t.scale_sample.first().copied())
                    .unwrap_or(1.0) as f64;
                let bias: Vec<f64> = get_tensor(inp_idx(2))
                    .map(|bt| bias_to_f64(bytes, bt, in_scale, wt_info))
                    .unwrap_or_default();
                let (out_data, oh, ow) = fwd_conv2d(
                    &inp, ih, iw, ic, &wt_data, oc, kh, kw, &bias, sh, sw, dh, dw, same, act,
                );
                if oh == 0 {
                    push_skipped(
                        &mut results,
                        op_idx,
                        op,
                        out_idx(0),
                        &analysis.tensors,
                        "output size 0",
                    );
                    continue;
                }
                let energy = haar_energy_hwc(&out_data, oh, ow, oc);
                push_haar_result(
                    &mut results,
                    op_idx,
                    op,
                    out_idx(0),
                    &analysis.tensors,
                    oh,
                    ow,
                    oc,
                    energy,
                );
                buffers.insert(out_idx(0), (out_data, oh, ow, oc));
            }

            "DEPTHWISE_CONV_2D" => {
                let (same, sh, sw, dh, dw, act) = parse_dw_conv_opts(&fb, opts);
                let Some((inp, ih, iw, ic)) = get_buf(inp_idx(0)) else {
                    push_skipped(
                        &mut results,
                        op_idx,
                        op,
                        out_idx(0),
                        &analysis.tensors,
                        "input buffer missing",
                    );
                    continue;
                };
                let wt_info = match get_tensor(inp_idx(1)) {
                    Some(t) => t,
                    None => {
                        push_skipped(
                            &mut results,
                            op_idx,
                            op,
                            out_idx(0),
                            &analysis.tensors,
                            "weight tensor missing",
                        );
                        continue;
                    }
                };
                let wt_data = match weight_to_f64(bytes, wt_info) {
                    Some(d) => d,
                    None => {
                        push_skipped(
                            &mut results,
                            op_idx,
                            op,
                            out_idx(0),
                            &analysis.tensors,
                            "weight dtype unsupported",
                        );
                        continue;
                    }
                };
                let kh = wt_info.shape.get(1).copied().unwrap_or(1) as usize;
                let kw = wt_info.shape.get(2).copied().unwrap_or(1) as usize;
                let in_scale = get_tensor(inp_idx(0))
                    .and_then(|t| t.scale_sample.first().copied())
                    .unwrap_or(1.0) as f64;
                let bias: Vec<f64> = get_tensor(inp_idx(2))
                    .map(|bt| bias_to_f64(bytes, bt, in_scale, wt_info))
                    .unwrap_or_default();
                let (out_data, oh, ow) = fwd_dw_conv2d(
                    &inp, ih, iw, ic, &wt_data, kh, kw, &bias, sh, sw, dh, dw, same, act,
                );
                if oh == 0 {
                    push_skipped(
                        &mut results,
                        op_idx,
                        op,
                        out_idx(0),
                        &analysis.tensors,
                        "output size 0",
                    );
                    continue;
                }
                let energy = haar_energy_hwc(&out_data, oh, ow, ic);
                push_haar_result(
                    &mut results,
                    op_idx,
                    op,
                    out_idx(0),
                    &analysis.tensors,
                    oh,
                    ow,
                    ic,
                    energy,
                );
                buffers.insert(out_idx(0), (out_data, oh, ow, ic));
            }

            "AVERAGE_POOL_2D" => {
                let (same, sh, sw, fh, fw, act) = parse_pool2d_opts(&fb, opts);
                if let Some((inp, ih, iw, ic)) = get_buf(inp_idx(0)) {
                    let (out_data, oh, ow) =
                        fwd_avg_pool(&inp, ih, iw, ic, fh, fw, sh, sw, same, act);
                    if oh > 0 {
                        buffers.insert(out_idx(0), (out_data, oh, ow, ic));
                    }
                }
            }

            "MAX_POOL_2D" => {
                let (same, sh, sw, fh, fw, act) = parse_pool2d_opts(&fb, opts);
                if let Some((inp, ih, iw, ic)) = get_buf(inp_idx(0)) {
                    let (out_data, oh, ow) =
                        fwd_max_pool(&inp, ih, iw, ic, fh, fw, sh, sw, same, act);
                    if oh > 0 {
                        buffers.insert(out_idx(0), (out_data, oh, ow, ic));
                    }
                }
            }

            "FULLY_CONNECTED" => {
                let act = opts
                    .and_then(|t| fb.field_pos(t, 0))
                    .and_then(|p| fb.i8(p))
                    .unwrap_or(0);
                if let Some((inp, ih, iw, ic)) = get_buf(inp_idx(0)) {
                    let flat_ic = ih * iw * ic;
                    if let Some(wt_info) = get_tensor(inp_idx(1)) {
                        let oc = wt_info.shape.first().copied().unwrap_or(0) as usize;
                        if let Some(wt_data) = weight_to_f64(bytes, wt_info) {
                            let in_scale = get_tensor(inp_idx(0))
                                .and_then(|t| t.scale_sample.first().copied())
                                .unwrap_or(1.0) as f64;
                            let bias: Vec<f64> = get_tensor(inp_idx(2))
                                .map(|bt| bias_to_f64(bytes, bt, in_scale, wt_info))
                                .unwrap_or_default();
                            let out_data = fwd_fc(&inp, flat_ic, &wt_data, oc, &bias, act);
                            buffers.insert(out_idx(0), (out_data, 1, 1, oc));
                        }
                    }
                }
            }

            "ADD" => {
                let act = opts
                    .and_then(|t| fb.field_pos(t, 0))
                    .and_then(|p| fb.i8(p))
                    .unwrap_or(0);
                let a = get_buf(inp_idx(0));
                let b = get_buf(inp_idx(1));
                if let (Some((ad, ah, aw, ac)), Some((bd, _, _, _))) = (a, b) {
                    let out_data = fwd_add(&ad, &bd, act);
                    buffers.insert(out_idx(0), (out_data, ah, aw, ac));
                }
            }

            "MUL" => {
                let act = opts
                    .and_then(|t| fb.field_pos(t, 0))
                    .and_then(|p| fb.i8(p))
                    .unwrap_or(0);
                let a = get_buf(inp_idx(0));
                let b = get_buf(inp_idx(1));
                if let (Some((ad, ah, aw, ac)), Some((bd, _, _, _))) = (a, b) {
                    let out_data = fwd_mul(&ad, &bd, act);
                    buffers.insert(out_idx(0), (out_data, ah, aw, ac));
                }
            }

            "RELU" => {
                if let Some((inp, ih, iw, ic)) = get_buf(inp_idx(0)) {
                    let out_data = fwd_relu(&inp, 1);
                    buffers.insert(out_idx(0), (out_data, ih, iw, ic));
                }
            }
            "RELU6" | "RELU_N1_TO_1" => {
                let act: i8 = if name == "RELU6" { 3 } else { 2 };
                if let Some((inp, ih, iw, ic)) = get_buf(inp_idx(0)) {
                    let out_data = fwd_relu(&inp, act);
                    buffers.insert(out_idx(0), (out_data, ih, iw, ic));
                }
            }

            "RESHAPE" | "SQUEEZE" | "EXPAND_DIMS" => {
                // Reinterpret shape; use output tensor shape for h/w/c if available
                if let Some((inp, ih, iw, ic)) = get_buf(inp_idx(0)) {
                    let out_shape: Vec<usize> = get_tensor(out_idx(0))
                        .map(|t| t.shape.iter().map(|&d| d as usize).collect())
                        .unwrap_or_else(|| vec![1, ih, iw, ic]);
                    let (nh, nw, nc) = match out_shape.len() {
                        4 => (
                            out_shape[1].max(1),
                            out_shape[2].max(1),
                            out_shape[3].max(1),
                        ),
                        3 => (out_shape[1].max(1), out_shape[2].max(1), 1),
                        2 => (1, 1, out_shape[1].max(1)),
                        _ => (ih, iw, ic),
                    };
                    buffers.insert(out_idx(0), (inp, nh, nw, nc));
                }
            }

            "SOFTMAX" | "L2_NORMALIZATION" => {
                // Pass activation through (identity for Haar propagation purposes)
                if let Some(buf) = get_buf(inp_idx(0)) {
                    buffers.insert(out_idx(0), buf);
                }
            }

            "DEQUANTIZE" | "QUANTIZE" => {
                // Treat as identity (already working in float domain)
                if let Some(buf) = get_buf(inp_idx(0)) {
                    buffers.insert(out_idx(0), buf);
                }
            }

            "PAD" | "PADV2" => {
                // Simple zero-padding: just propagate without full pad computation
                // (acceptable for Haar analysis purposes)
                if let Some(buf) = get_buf(inp_idx(0)) {
                    buffers.insert(out_idx(0), buf);
                }
            }

            "MEAN" => {
                // Global average (usually [N, H, W, C] → [N, 1, 1, C])
                if let Some((inp, ih, iw, ic)) = get_buf(inp_idx(0)) {
                    let (out_data, oh, ow) = fwd_avg_pool(&inp, ih, iw, ic, ih, iw, 1, 1, false, 0);
                    if oh > 0 {
                        buffers.insert(out_idx(0), (out_data, oh, ow, ic));
                    }
                }
            }

            "CONCATENATION" => {
                // Concatenate along last axis (axis=3 for HWC — typical for channel concat)
                let axis = opts
                    .and_then(|t| fb.field_pos(t, 0))
                    .and_then(|p| fb.i32(p))
                    .unwrap_or(3);
                if axis == 3 || axis == -1 {
                    let parts: Vec<(Vec<f64>, usize, usize, usize)> = op
                        .inputs
                        .iter()
                        .filter_map(|&idx| get_buf(idx as usize))
                        .collect();
                    if !parts.is_empty() {
                        let (fh, fw) = (parts[0].1, parts[0].2);
                        let total_c: usize = parts.iter().map(|p| p.3).sum();
                        let mut concat = vec![0.0f64; fh * fw * total_c];
                        for (y, x) in (0..fh).flat_map(|y| (0..fw).map(move |x| (y, x))) {
                            let mut ch_off = 0;
                            for (pd, _, _, pc) in &parts {
                                for ch in 0..*pc {
                                    concat[y * fw * total_c + x * total_c + ch_off + ch] =
                                        pd[y * fw * pc + x * pc + ch];
                                }
                                ch_off += pc;
                            }
                        }
                        buffers.insert(out_idx(0), (concat, fh, fw, total_c));
                    }
                } else {
                    // Non-channel concat: just pass first input through
                    if let Some(buf) = get_buf(inp_idx(0)) {
                        buffers.insert(out_idx(0), buf);
                    }
                }
            }

            _ => {
                // Unknown op: propagate first input tensor as-is so downstream ops
                // have something to work with (graceful degradation)
                if let Some(buf) = get_buf(inp_idx(0)) {
                    buffers.insert(out_idx(0), buf);
                }
            }
        }
    }

    results
}

#[allow(clippy::too_many_arguments)]
fn push_haar_result(
    results: &mut Vec<ActivationHaarOpResult>,
    op_idx: usize,
    op: &OpInfo,
    out_tensor_idx: usize,
    tensors: &[TensorInfo],
    h: usize,
    w: usize,
    c: usize,
    energy: [f64; 4],
) {
    let total = energy[0] + energy[1] + energy[2] + energy[3];
    let (ll, lh, hl, hh) = if total > 0.0 {
        (
            energy[0] / total,
            energy[1] / total,
            energy[2] / total,
            energy[3] / total,
        )
    } else {
        (1.0, 0.0, 0.0, 0.0)
    };
    let dom = dominant_band(energy[0], energy[1], energy[2], energy[3]).to_string();
    let too_small = h < 2 || w < 2;
    results.push(ActivationHaarOpResult {
        op_index: op_idx,
        op_name: op.name.clone(),
        output_tensor_index: out_tensor_idx,
        output_tensor_name: tensors
            .get(out_tensor_idx)
            .map(|t| t.name.clone())
            .unwrap_or_default(),
        output_shape: vec![1, h, w, c],
        ll_energy: ll,
        lh_energy: lh,
        hl_energy: hl,
        hh_energy: hh,
        dominant_band: dom,
        spatial_smoothness: ll,
        spatial_too_small: too_small,
        skipped: false,
        skip_reason: String::new(),
    });
}

fn push_skipped(
    results: &mut Vec<ActivationHaarOpResult>,
    op_idx: usize,
    op: &OpInfo,
    out_tensor_idx: usize,
    tensors: &[TensorInfo],
    reason: &str,
) {
    results.push(ActivationHaarOpResult {
        op_index: op_idx,
        op_name: op.name.clone(),
        output_tensor_index: out_tensor_idx,
        output_tensor_name: tensors
            .get(out_tensor_idx)
            .map(|t| t.name.clone())
            .unwrap_or_default(),
        output_shape: vec![],
        ll_energy: 0.0,
        lh_energy: 0.0,
        hl_energy: 0.0,
        hh_energy: 0.0,
        dominant_band: String::new(),
        spatial_smoothness: 0.0,
        spatial_too_small: false,
        skipped: true,
        skip_reason: reason.to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::padding_is_same;

    #[test]
    fn tflite_padding_enum_is_not_reversed() {
        assert!(padding_is_same(0));
        assert!(!padding_is_same(1));
        assert!(!padding_is_same(2));
    }
}

/// Compute activation Haar energy for each CONV/DEPTHWISE op via synthetic forward pass.
/// Uses a 32×32 (max) synthetic input with a deterministic spatial pattern.
/// Works entirely in f64; INT8 weights are dequantized using full per-channel scales.
#[wasm_bindgen]
pub fn compute_activation_haar(
    bytes: &[u8],
    filename: &str,
    target_id: &str,
) -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    let tid = if target_id.is_empty() {
        "android_mid_a55"
    } else {
        target_id
    };
    let analysis = analyze_with_target_without_step_response(bytes, filename, tid)
        .map_err(|e| JsValue::from_str(&e))?;
    let results = run_activation_haar(bytes, &analysis);
    serde_wasm_bindgen::to_value(&results).map_err(|e| JsValue::from_str(&e.to_string()))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Loss Landscape — multi-seed 2D weight-space perturbation (direction generation)
// JS applies the actual perturbation via TypedArray arithmetic to avoid 405×WASM overhead.
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-tensor metadata needed by JS to apply alpha*d1 + beta*d2 perturbations.
#[derive(Serialize)]
struct LscWeightMeta {
    buf_offset: usize,  // byte offset of weight data in model file
    elem_count: usize,  // number of elements (not bytes)
    dtype: String,      // "FLOAT32" or "INT8"
    oc: usize,          // first dimension (output channels; 1 for depthwise)
    filter_size: usize, // elements per output channel = elem_count / oc
    scales: Vec<f32>,   // per-channel quant scales (or length-1 global)
    zps: Vec<i32>,      // per-channel zero points
}

#[derive(Serialize)]
struct LscDirs {
    d1: Vec<f32>, // flat filter-normalized direction 1 across all weight tensors
    d2: Vec<f32>, // flat filter-normalized direction 2
    metas: Vec<LscWeightMeta>,
    total_params: usize,
}

// XorShift64 deterministic RNG + Box-Muller Gaussian
struct Xsr64(u64);
impl Xsr64 {
    fn new(seed: u64) -> Self {
        Xsr64(seed | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f64_01(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    fn gaussian(&mut self) -> f64 {
        let u1 = self.f64_01().max(1e-15);
        let u2 = self.f64_01();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// Parse only the weight tensors needed for landscape — much faster than analyze_with_target.
fn lsc_parse_weight_tensors(fb: &Fb) -> Result<Vec<TensorInfo>, String> {
    let model = fb.root_table()?;
    let buf_locs = read_buffer_locations(fb, model);
    let sg_tables = fb.vector_tables(model, 2);
    let subgraph = sg_tables
        .first()
        .copied()
        .ok_or_else(|| "Model has no subgraph".to_string())?;
    let tensors = fb
        .vector_tables(subgraph, 0)
        .iter()
        .enumerate()
        .map(|(i, &t)| read_tensor(fb, i, t, &buf_locs))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(tensors
        .into_iter()
        .filter(|t| {
            t.constant_buffer
                && !t.sparse_storage
                && t.buffer_data_length > 0
                && (t.dtype == "FLOAT32" || t.dtype == "INT8")
        })
        .collect())
}

// Filter-wise normalized random direction for a single weight tensor.
// For FLOAT32: use raw values. For INT8: dequantize first.
fn lsc_filter_dir(bytes: &[u8], t: &TensorInfo, rng: &mut Xsr64) -> Vec<f32> {
    let elem = t.buffer_data_length / if t.dtype == "FLOAT32" { 4 } else { 1 };
    // For DEPTHWISE [1,kH,kW,IC]: shape[0]=1 but IC channels each get their own scale.
    // Detect DW the same way weight_to_f64 does: oc==1 with multi-element scale array.
    let shape_oc = t.shape.first().copied().unwrap_or(1).max(1) as usize;
    let ic = t.shape.get(3).copied().unwrap_or(1).max(1) as usize;
    let is_dw = shape_oc == 1 && t.shape.len() >= 4 && t.scale_sample.len() > 1;
    let (oc, filter_size) = if is_dw {
        (ic, elem / ic.max(1))
    } else {
        (shape_oc, elem / shape_oc.max(1))
    };
    let is_per_ch = t.scale_sample.len() >= oc;
    let raw = &bytes[t.buffer_data_offset..t.buffer_data_offset + t.buffer_data_length];

    let mut dir = Vec::with_capacity(elem);
    for oc_idx in 0..oc {
        let start = oc_idx * filter_size;
        let end = start + filter_size;

        // Filter values in float domain
        let filt: Vec<f64> = if t.dtype == "FLOAT32" {
            (start..end)
                .map(|i| {
                    f32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]])
                        as f64
                })
                .collect()
        } else {
            let s = if is_per_ch {
                t.scale_sample[oc_idx.min(t.scale_sample.len() - 1)] as f64
            } else {
                t.scale_sample.first().copied().unwrap_or(1.0) as f64
            };
            let z = if is_per_ch {
                t.zero_point_sample[oc_idx.min(t.zero_point_sample.len().max(1) - 1)] as f64
            } else {
                t.zero_point_sample.first().copied().unwrap_or(0) as f64
            };
            (start..end)
                .map(|i| ((raw[i] as i8) as f64 - z) * s)
                .collect()
        };

        // Random Gaussian raw direction
        let raw_dir: Vec<f64> = (0..filter_size).map(|_| rng.gaussian()).collect();
        let filt_norm = filt.iter().map(|v| v * v).sum::<f64>().sqrt();
        let dir_norm = raw_dir.iter().map(|v| v * v).sum::<f64>().sqrt();

        if dir_norm > 1e-12 {
            let scale = if filt_norm > 1e-12 {
                filt_norm / dir_norm
            } else {
                1.0 / dir_norm
            };
            dir.extend(raw_dir.iter().map(|&v| (v * scale) as f32));
        } else {
            dir.extend(std::iter::repeat_n(0.0f32, filter_size));
        }
    }
    dir
}

/// Generate two filter-normalized random directions in weight space (one per seed parameter).
/// Returns d1, d2 as flat Float32Arrays and per-tensor metadata needed for JS-side perturbation.
/// Call once per seed; JS applies alpha*d1+beta*d2 directly on the model Uint8Array.
#[wasm_bindgen]
pub fn landscape_directions(bytes: &[u8], seed1: u32, seed2: u32) -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    // This is the one weight-tensor reader that does not reach the graph through
    // analyze_with_target*, so it applies the same truncation gate itself.
    // Without it a truncated artifact yields a silently shorter direction set.
    let fb = Fb::verified_tflite(bytes).map_err(|error| JsValue::from_str(&error))?;
    let tensors = lsc_parse_weight_tensors(&fb).map_err(|error| JsValue::from_str(&error))?;
    let mut rng1 = Xsr64::new(seed1 as u64 | 1);
    let mut rng2 = Xsr64::new(seed2 as u64 | 1);
    let mut d1: Vec<f32> = Vec::new();
    let mut d2: Vec<f32> = Vec::new();
    let mut metas: Vec<LscWeightMeta> = Vec::new();

    for t in &tensors {
        let elem = t.buffer_data_length / if t.dtype == "FLOAT32" { 4 } else { 1 };
        // Mirror DW detection from lsc_filter_dir so meta oc/filter_size matches direction layout
        let shape_oc = t.shape.first().copied().unwrap_or(1).max(1) as usize;
        let ic = t.shape.get(3).copied().unwrap_or(1).max(1) as usize;
        let is_dw = shape_oc == 1 && t.shape.len() >= 4 && t.scale_sample.len() > 1;
        let (oc, filter_size) = if is_dw {
            (ic, elem / ic.max(1))
        } else {
            (shape_oc, elem / shape_oc.max(1))
        };
        let dir1 = lsc_filter_dir(bytes, t, &mut rng1);
        let dir2 = lsc_filter_dir(bytes, t, &mut rng2);
        d1.extend_from_slice(&dir1);
        d2.extend_from_slice(&dir2);
        metas.push(LscWeightMeta {
            buf_offset: t.buffer_data_offset,
            elem_count: elem,
            dtype: t.dtype.clone(),
            oc,
            filter_size,
            scales: t.scale_sample.to_vec(),
            zps: t.zero_point_sample.iter().map(|&z| z as i32).collect(),
        });
    }
    let total_params = d1.len();
    serde_wasm_bindgen::to_value(&LscDirs {
        d1,
        d2,
        metas,
        total_params,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ── Loss Landscape f64 path ──────────────────────────────────────────────────

fn weight_to_f64_delta(bytes: &[u8], t: &TensorInfo, delta: Option<&[f64]>) -> Option<Vec<f64>> {
    let mut w = weight_to_f64(bytes, t)?;
    if let Some(d) = delta {
        for (wi, di) in w.iter_mut().zip(d.iter()) {
            *wi += di;
        }
    }
    Some(w)
}

// Synthetic f64 forward pass with optional per-tensor weight deltas (keyed by buffer_data_offset).
// Returns concatenated final output tensor values.
fn synth_fwd_with_deltas(
    bytes: &[u8],
    analysis: &Analysis,
    deltas: &std::collections::HashMap<usize, Vec<f64>>,
) -> Vec<f64> {
    const MAX_HW: usize = 32;
    let fb = match Fb::verified_tflite(bytes) {
        Ok(fb) => fb,
        Err(_) => return vec![],
    };
    let model = match fb.root_table() {
        Ok(r) => r,
        Err(_) => return vec![],
    };
    let sg_tables = fb.vector_tables(model, 2);
    let subgraph = match sg_tables.first().copied() {
        Some(s) => s,
        None => return vec![],
    };
    let op_tables = fb.vector_tables(subgraph, 3);

    let mut buffers: HashMap<usize, (Vec<f64>, usize, usize, usize)> = HashMap::new();

    for &idx in &analysis.input_tensor_indices {
        let idx = idx as usize;
        if let Some(t) = analysis.tensors.get(idx) {
            let (data, h, w, c) = make_synthetic_input(t, MAX_HW);
            buffers.insert(idx, (data, h, w, c));
        }
    }

    for (&op_table, op) in op_tables.iter().zip(analysis.ops.iter()) {
        let name = op.name.as_str();
        let opts = op_options_table(&fb, op_table);
        let inp_idx = |n: usize| op.inputs.get(n).copied().unwrap_or(-1) as usize;
        let out_idx = |n: usize| op.outputs.get(n).copied().unwrap_or(-1) as usize;

        macro_rules! get_buf {
            ($i:expr) => {
                buffers.get(&$i).cloned()
            };
        }
        macro_rules! get_ten {
            ($i:expr) => {
                analysis.tensors.get($i)
            };
        }
        macro_rules! wt_f64d {
            ($t:expr) => {
                weight_to_f64_delta(
                    bytes,
                    $t,
                    deltas.get(&$t.buffer_data_offset).map(|v| v.as_slice()),
                )
            };
        }
        macro_rules! in_scale {
            ($i:expr) => {
                get_ten!($i)
                    .and_then(|t| t.scale_sample.first().copied())
                    .unwrap_or(1.0) as f64
            };
        }
        macro_rules! bias_v {
            ($bi:expr, $wt:expr) => {
                get_ten!($bi)
                    .map(|bt| bias_to_f64(bytes, bt, in_scale!(inp_idx(0)), $wt))
                    .unwrap_or_default()
            };
        }

        match name {
            "CONV_2D" => {
                let (same, sh, sw, dh, dw, act) = parse_conv2d_opts(&fb, opts);
                let Some((inp, ih, iw, ic)) = get_buf!(inp_idx(0)) else {
                    continue;
                };
                let Some(wt_info) = get_ten!(inp_idx(1)) else {
                    continue;
                };
                let Some(wt_data) = wt_f64d!(wt_info) else {
                    continue;
                };
                let oc = wt_info.shape.first().copied().unwrap_or(0) as usize;
                let kh = wt_info.shape.get(1).copied().unwrap_or(1) as usize;
                let kw = wt_info.shape.get(2).copied().unwrap_or(1) as usize;
                let bias = bias_v!(inp_idx(2), wt_info);
                let (out_data, oh, ow) = fwd_conv2d(
                    &inp, ih, iw, ic, &wt_data, oc, kh, kw, &bias, sh, sw, dh, dw, same, act,
                );
                if oh > 0 {
                    buffers.insert(out_idx(0), (out_data, oh, ow, oc));
                }
            }
            "DEPTHWISE_CONV_2D" => {
                let (same, sh, sw, dh, dw, act) = parse_dw_conv_opts(&fb, opts);
                let Some((inp, ih, iw, ic)) = get_buf!(inp_idx(0)) else {
                    continue;
                };
                let Some(wt_info) = get_ten!(inp_idx(1)) else {
                    continue;
                };
                let Some(wt_data) = wt_f64d!(wt_info) else {
                    continue;
                };
                let kh = wt_info.shape.get(1).copied().unwrap_or(1) as usize;
                let kw = wt_info.shape.get(2).copied().unwrap_or(1) as usize;
                let bias = bias_v!(inp_idx(2), wt_info);
                let (out_data, oh, ow) = fwd_dw_conv2d(
                    &inp, ih, iw, ic, &wt_data, kh, kw, &bias, sh, sw, dh, dw, same, act,
                );
                if oh > 0 {
                    buffers.insert(out_idx(0), (out_data, oh, ow, ic));
                }
            }
            "AVERAGE_POOL_2D" => {
                let (same, sh, sw, fh, fw, act) = parse_pool2d_opts(&fb, opts);
                if let Some((inp, ih, iw, ic)) = get_buf!(inp_idx(0)) {
                    let (out_data, oh, ow) =
                        fwd_avg_pool(&inp, ih, iw, ic, fh, fw, sh, sw, same, act);
                    if oh > 0 {
                        buffers.insert(out_idx(0), (out_data, oh, ow, ic));
                    }
                }
            }
            "MAX_POOL_2D" => {
                let (same, sh, sw, fh, fw, act) = parse_pool2d_opts(&fb, opts);
                if let Some((inp, ih, iw, ic)) = get_buf!(inp_idx(0)) {
                    let (out_data, oh, ow) =
                        fwd_max_pool(&inp, ih, iw, ic, fh, fw, sh, sw, same, act);
                    if oh > 0 {
                        buffers.insert(out_idx(0), (out_data, oh, ow, ic));
                    }
                }
            }
            "FULLY_CONNECTED" => {
                let act = opts
                    .and_then(|t| fb.field_pos(t, 0))
                    .and_then(|p| fb.i8(p))
                    .unwrap_or(0);
                if let Some((inp, ih, iw, ic)) = get_buf!(inp_idx(0)) {
                    let flat_ic = ih * iw * ic;
                    if let Some(wt_info) = get_ten!(inp_idx(1)) {
                        let oc = wt_info.shape.first().copied().unwrap_or(0) as usize;
                        if let Some(wt_data) = wt_f64d!(wt_info) {
                            let bias = bias_v!(inp_idx(2), wt_info);
                            let out_data = fwd_fc(&inp, flat_ic, &wt_data, oc, &bias, act);
                            buffers.insert(out_idx(0), (out_data, 1, 1, oc));
                        }
                    }
                }
            }
            "ADD" => {
                let act = opts
                    .and_then(|t| fb.field_pos(t, 0))
                    .and_then(|p| fb.i8(p))
                    .unwrap_or(0);
                if let (Some((ad, ah, aw, ac)), Some((bd, _, _, _))) =
                    (get_buf!(inp_idx(0)), get_buf!(inp_idx(1)))
                {
                    buffers.insert(out_idx(0), (fwd_add(&ad, &bd, act), ah, aw, ac));
                }
            }
            "MUL" => {
                let act = opts
                    .and_then(|t| fb.field_pos(t, 0))
                    .and_then(|p| fb.i8(p))
                    .unwrap_or(0);
                if let (Some((ad, ah, aw, ac)), Some((bd, _, _, _))) =
                    (get_buf!(inp_idx(0)), get_buf!(inp_idx(1)))
                {
                    buffers.insert(out_idx(0), (fwd_mul(&ad, &bd, act), ah, aw, ac));
                }
            }
            "RELU" => {
                if let Some((inp, ih, iw, ic)) = get_buf!(inp_idx(0)) {
                    buffers.insert(out_idx(0), (fwd_relu(&inp, 1), ih, iw, ic));
                }
            }
            "RELU6" | "RELU_N1_TO_1" => {
                let act: i8 = if name == "RELU6" { 3 } else { 2 };
                if let Some((inp, ih, iw, ic)) = get_buf!(inp_idx(0)) {
                    buffers.insert(out_idx(0), (fwd_relu(&inp, act), ih, iw, ic));
                }
            }
            "RESHAPE" | "SQUEEZE" | "EXPAND_DIMS" => {
                if let Some((inp, ih, iw, ic)) = get_buf!(inp_idx(0)) {
                    let out_shape: Vec<usize> = analysis
                        .tensors
                        .get(out_idx(0))
                        .map(|t| t.shape.iter().map(|&d| d as usize).collect())
                        .unwrap_or_else(|| vec![1, ih, iw, ic]);
                    let (nh, nw, nc) = match out_shape.len() {
                        4 => (
                            out_shape[1].max(1),
                            out_shape[2].max(1),
                            out_shape[3].max(1),
                        ),
                        3 => (out_shape[1].max(1), out_shape[2].max(1), 1),
                        2 => (1, 1, out_shape[1].max(1)),
                        _ => (ih, iw, ic),
                    };
                    buffers.insert(out_idx(0), (inp, nh, nw, nc));
                }
            }
            "SOFTMAX" | "L2_NORMALIZATION" | "DEQUANTIZE" | "QUANTIZE" => {
                if let Some(buf) = get_buf!(inp_idx(0)) {
                    buffers.insert(out_idx(0), buf);
                }
            }
            "PAD" | "PADV2" => {
                if let Some(buf) = get_buf!(inp_idx(0)) {
                    buffers.insert(out_idx(0), buf);
                }
            }
            "MEAN" => {
                if let Some((inp, ih, iw, ic)) = get_buf!(inp_idx(0)) {
                    let (out_data, oh, ow) = fwd_avg_pool(&inp, ih, iw, ic, ih, iw, 1, 1, false, 0);
                    if oh > 0 {
                        buffers.insert(out_idx(0), (out_data, oh, ow, ic));
                    }
                }
            }
            "CONCATENATION" => {
                let axis = opts
                    .and_then(|t| fb.field_pos(t, 0))
                    .and_then(|p| fb.i32(p))
                    .unwrap_or(3);
                if axis == 3 || axis == -1 {
                    let parts: Vec<(Vec<f64>, usize, usize, usize)> = op
                        .inputs
                        .iter()
                        .filter_map(|&i| buffers.get(&(i as usize)).cloned())
                        .collect();
                    if !parts.is_empty() {
                        let (fh, fw) = (parts[0].1, parts[0].2);
                        let total_c: usize = parts.iter().map(|p| p.3).sum();
                        let mut concat = vec![0.0f64; fh * fw * total_c];
                        for y in 0..fh {
                            for x in 0..fw {
                                let mut ch_off = 0;
                                for (pd, _, _, pc) in &parts {
                                    for ch in 0..*pc {
                                        concat[y * fw * total_c + x * total_c + ch_off + ch] =
                                            pd[y * fw * pc + x * pc + ch];
                                    }
                                    ch_off += pc;
                                }
                            }
                        }
                        buffers.insert(out_idx(0), (concat, fh, fw, total_c));
                    }
                } else if let Some(buf) = get_buf!(inp_idx(0)) {
                    buffers.insert(out_idx(0), buf);
                }
            }
            _ => {
                if let Some(buf) = get_buf!(inp_idx(0)) {
                    buffers.insert(out_idx(0), buf);
                }
            }
        }
    }

    analysis
        .output_tensor_indices
        .iter()
        .filter(|&&i| i >= 0)
        .filter_map(|&i| buffers.get(&(i as usize)))
        .flat_map(|(data, _, _, _)| data.iter().copied())
        .collect()
}

#[derive(Serialize)]
struct SynthGrid {
    grid: Vec<f64>, // G*G flat, row-major: grid[bi*G + ai], bi=beta, ai=alpha
    axes: Vec<f64>,
}

/// Compute full G×G synthetic f64 loss landscape for one seed.
/// Weights perturbed in float domain (no requantization). Returns drift vs center.
#[wasm_bindgen]
pub fn synthetic_landscape_grid(
    bytes: &[u8],
    seed1: u32,
    seed2: u32,
    grid_size: u32,
    radius: f32,
) -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    let g = (grid_size as usize).max(3);
    let analysis = analyze_with_target_without_step_response(bytes, "", "android_mid_a55")
        .map_err(|e| JsValue::from_str(&e))?;
    let fb = Fb::verified_tflite(bytes).map_err(|error| JsValue::from_str(&error))?;
    let weight_tensors =
        lsc_parse_weight_tensors(&fb).map_err(|error| JsValue::from_str(&error))?;
    let mut rng1 = Xsr64::new(seed1 as u64 | 1);
    let mut rng2 = Xsr64::new(seed2 as u64 | 1);

    // Build per-filter-normalized direction maps keyed by buffer_data_offset
    let mut dir1_map: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut dir2_map: HashMap<usize, Vec<f64>> = HashMap::new();
    for t in &weight_tensors {
        dir1_map.insert(
            t.buffer_data_offset,
            lsc_filter_dir(bytes, t, &mut rng1)
                .into_iter()
                .map(|v| v as f64)
                .collect(),
        );
        dir2_map.insert(
            t.buffer_data_offset,
            lsc_filter_dir(bytes, t, &mut rng2)
                .into_iter()
                .map(|v| v as f64)
                .collect(),
        );
    }

    let center_out = synth_fwd_with_deltas(bytes, &analysis, &HashMap::new());
    let n = center_out.len();

    let step = if g > 1 {
        2.0 * radius as f64 / (g - 1) as f64
    } else {
        0.0
    };
    let axes: Vec<f64> = (0..g).map(|i| -radius as f64 + i as f64 * step).collect();
    let mut grid = vec![f64::NAN; g * g];

    for (bi, &beta) in axes.iter().enumerate() {
        for (ai, &alpha) in axes.iter().enumerate() {
            let deltas: HashMap<usize, Vec<f64>> = dir1_map
                .iter()
                .map(|(&off, d1)| {
                    let d2 = &dir2_map[&off];
                    let delta = d1
                        .iter()
                        .zip(d2.iter())
                        .map(|(&a, &b)| alpha * a + beta * b)
                        .collect();
                    (off, delta)
                })
                .collect();
            let out = synth_fwd_with_deltas(bytes, &analysis, &deltas);
            if out.len() == n && n > 0 {
                let rms = (out
                    .iter()
                    .zip(center_out.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    / n as f64)
                    .sqrt();
                grid[bi * g + ai] = rms;
            }
        }
    }

    serde_wasm_bindgen::to_value(&SynthGrid { grid, axes })
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ── Model Tomography ──────────────────────────────────────────────────────────

#[derive(Serialize)]
struct LayerTomographyEntry {
    op_index: usize,
    op_name: String,
    tensor_name: String,
    layer_type: String, // "conv2d" | "depthwise" | "fc"
    oc: usize,          // effective output channels (= IC for depthwise)
    kh: usize,
    kw: usize,
    ic: usize,
    dtype: String,
    l2_per_oc: Vec<f32>,    // [OC] RMS per filter (normalized by kH*kW*IC)
    spatial_flat: Vec<f32>, // [OC * kH * kW] IC-averaged spatial filter weights
    haar_ll: f32,
    haar_lh: f32,
    haar_hl: f32,
    haar_hh: f32,
    // Simulated per-tensor symmetric int8 weight-quantization SNR (dB). Proxy for
    // how much this layer's weights would distort under int8; NOT an accuracy claim.
    // Valid only for float weights — already-quantized layers report valid=false.
    quant_snr_db: f32,
    quant_snr_valid: bool,
}

/// Simulated int8 weight-quantization SNR for a float weight tensor.
/// Per-tensor symmetric quantization (scale = max|w| / 127), reconstruction MSE
/// against the original weights. Returns None for non-float or degenerate tensors.
fn weight_quant_snr_db(bytes: &[u8], wt: &TensorInfo) -> Option<f64> {
    if wt.dtype != "FLOAT32" && wt.dtype != "FLOAT16" {
        return None;
    }
    let w = weight_to_f64(bytes, wt)?;
    if w.is_empty() {
        return None;
    }
    let max_abs = w.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    if max_abs <= 0.0 || !max_abs.is_finite() {
        return None;
    }
    let scale = max_abs / 127.0;
    let mut signal = 0.0_f64;
    let mut noise = 0.0_f64;
    for &v in &w {
        let q = (v / scale).round().clamp(-127.0, 127.0);
        let err = v - q * scale;
        signal += v * v;
        noise += err * err;
    }
    if signal <= 0.0 {
        return None;
    }
    if noise <= 0.0 {
        return Some(99.0); // lossless within f64 rounding — cap instead of infinity
    }
    Some((10.0 * (signal / noise).log10()).clamp(-99.0, 99.0))
}

/// Per-filter statistics: (oc, kh, kw, ic, l2_per_oc, spatial_flat[oc*kh*kw])
fn layer_weight_stats(
    bytes: &[u8],
    wt: &TensorInfo,
    is_dw: bool,
) -> (usize, usize, usize, usize, Vec<f32>, Vec<f32>) {
    let shape: Vec<usize> = wt.shape.iter().map(|&d| (d as usize).max(1)).collect();

    let (oc, kh, kw, ic, real_ic) = if is_dw {
        // [1, kH, kW, IC]: treat each input channel as independent filter
        let kh = shape.get(1).copied().unwrap_or(1);
        let kw = shape.get(2).copied().unwrap_or(1);
        let ic = shape.get(3).copied().unwrap_or(1);
        (ic, kh, kw, 1usize, ic)
    } else if shape.len() == 2 {
        // FC: [OC, IC]
        (shape[0], 1, 1, shape[1], shape[1])
    } else {
        // CONV_2D: [OC, kH, kW, IC]
        let oc = shape.first().copied().unwrap_or(1);
        let kh = shape.get(1).copied().unwrap_or(1);
        let kw = shape.get(2).copied().unwrap_or(1);
        let ic = shape.get(3).copied().unwrap_or(1);
        (oc, kh, kw, ic, ic)
    };

    let w = match weight_to_f64(bytes, wt) {
        Some(w) => w,
        None => return (oc, kh, kw, ic, vec![0.0f32; oc], vec![0.0f32; oc * kh * kw]),
    };

    let mut l2s = vec![0.0f32; oc];
    let mut spatial = vec![0.0f32; oc * kh * kw];

    for o in 0..oc {
        let mut sum_sq = 0.0f64;
        for h in 0..kh {
            for ww in 0..kw {
                let sp: f32 = if is_dw {
                    // [1, kH, kW, real_ic]: element at [0,h,ww,o]
                    let v = w
                        .get(h * kw * real_ic + ww * real_ic + o)
                        .copied()
                        .unwrap_or(0.0);
                    sum_sq += v * v;
                    v as f32
                } else {
                    // CONV_2D [OC,kH,kW,IC] or FC[OC,IC] (kh=kw=1): average over IC
                    let base = o * kh * kw * ic + h * kw * ic + ww * ic;
                    let mut ic_sum = 0.0f64;
                    for c in 0..ic {
                        let v = w.get(base + c).copied().unwrap_or(0.0);
                        sum_sq += v * v;
                        ic_sum += v;
                    }
                    (ic_sum / ic.max(1) as f64) as f32
                };
                spatial[o * kh * kw + h * kw + ww] = sp;
            }
        }
        let elem = kh * kw * (if is_dw { 1 } else { ic });
        l2s[o] = (sum_sq / elem.max(1) as f64).sqrt() as f32;
    }

    (oc, kh, kw, ic, l2s, spatial)
}

/// Per-layer weight statistics + activation Haar for triplanar model viewer.
/// Returns one entry per CONV_2D / DEPTHWISE_CONV_2D / FULLY_CONNECTED op.
#[wasm_bindgen]
pub fn compute_model_tomography(
    bytes: &[u8],
    filename: &str,
    target_id: &str,
) -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    let tid = if target_id.is_empty() {
        "android_mid_a55"
    } else {
        target_id
    };
    let analysis = analyze_with_target_without_step_response(bytes, filename, tid)
        .map_err(|e| JsValue::from_str(&e))?;

    let haar_results = run_activation_haar(bytes, &analysis);
    let haar_map: HashMap<usize, (f32, f32, f32, f32)> = haar_results
        .iter()
        .filter(|r| !r.skipped && !r.spatial_too_small)
        .map(|r| {
            (
                r.op_index,
                (
                    r.ll_energy as f32,
                    r.lh_energy as f32,
                    r.hl_energy as f32,
                    r.hh_energy as f32,
                ),
            )
        })
        .collect();

    let mut entries: Vec<LayerTomographyEntry> = Vec::new();

    for (op_idx, op) in analysis.ops.iter().enumerate() {
        let name = op.name.as_str();
        let is_dw = name == "DEPTHWISE_CONV_2D";
        let layer_type = match name {
            "CONV_2D" => "conv2d",
            "DEPTHWISE_CONV_2D" => "depthwise",
            "FULLY_CONNECTED" => "fc",
            _ => continue,
        };

        let wt_idx = op.inputs.get(1).copied().unwrap_or(-1);
        if wt_idx < 0 {
            continue;
        }
        let wt = match analysis.tensors.get(wt_idx as usize) {
            Some(t) if t.constant_buffer => t,
            _ => continue,
        };

        let (oc, kh, kw, ic, l2_per_oc, spatial_flat) = layer_weight_stats(bytes, wt, is_dw);
        if l2_per_oc.is_empty() {
            continue;
        }

        let (haar_ll, haar_lh, haar_hl, haar_hh) = haar_map
            .get(&op_idx)
            .copied()
            .unwrap_or((0.0, 0.0, 0.0, 0.0));

        let snr = weight_quant_snr_db(bytes, wt);

        entries.push(LayerTomographyEntry {
            op_index: op_idx,
            op_name: op.name.clone(),
            tensor_name: wt.name.clone(),
            layer_type: layer_type.to_string(),
            oc,
            kh,
            kw,
            ic,
            dtype: wt.dtype.clone(),
            l2_per_oc,
            spatial_flat,
            haar_ll,
            haar_lh,
            haar_hl,
            haar_hh,
            quant_snr_db: snr.unwrap_or(0.0) as f32,
            quant_snr_valid: snr.is_some(),
        });
    }

    serde_wasm_bindgen::to_value(&entries).map_err(|e| JsValue::from_str(&e.to_string()))
}

// ── Loss Landscape Tomography ─────────────────────────────────────────────────

#[derive(Serialize)]
struct TomoResult {
    /// num_projections × (G*G) flat row-major grids (beta outer, alpha inner)
    grids: Vec<Vec<f64>>,
    axes: Vec<f64>,
    center_rms: f64,
}

/// Run K independent 2D landscape projections in a single WASM call.
/// analyze_with_target is called only once; K × G² forward passes follow.
/// Each projection uses a distinct filter-normalized direction pair.
#[wasm_bindgen]
pub fn landscape_tomography(
    bytes: &[u8],
    num_projections: u32,
    grid_size: u32,
    radius: f32,
) -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    let k = (num_projections as usize).max(1);
    let g = (grid_size as usize).max(3);

    let analysis = analyze_with_target_without_step_response(bytes, "", "android_mid_a55")
        .map_err(|e| JsValue::from_str(&e))?;
    let fb = Fb::verified_tflite(bytes).map_err(|error| JsValue::from_str(&error))?;
    let weight_tensors =
        lsc_parse_weight_tensors(&fb).map_err(|error| JsValue::from_str(&error))?;

    let step = if g > 1 {
        2.0 * radius as f64 / (g - 1) as f64
    } else {
        0.0
    };
    let axes: Vec<f64> = (0..g).map(|i| -radius as f64 + i as f64 * step).collect();

    let center_out = synth_fwd_with_deltas(bytes, &analysis, &HashMap::new());
    let n = center_out.len();
    let center_rms = {
        let c = g / 2;
        // placeholder; real center is alpha=beta=0 which gives 0 drift
        let _ = c;
        0.0f64
    };

    let mut all_grids: Vec<Vec<f64>> = Vec::with_capacity(k);

    for proj in 0..k {
        let s1 = ((2000u64).wrapping_add(11 * proj as u64)) | 1;
        let s2 = ((7000u64).wrapping_add(17 * proj as u64)) | 1;
        let mut rng1 = Xsr64::new(s1);
        let mut rng2 = Xsr64::new(s2);

        let mut dir1_map: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut dir2_map: HashMap<usize, Vec<f64>> = HashMap::new();
        for t in &weight_tensors {
            dir1_map.insert(
                t.buffer_data_offset,
                lsc_filter_dir(bytes, t, &mut rng1)
                    .into_iter()
                    .map(|v| v as f64)
                    .collect(),
            );
            dir2_map.insert(
                t.buffer_data_offset,
                lsc_filter_dir(bytes, t, &mut rng2)
                    .into_iter()
                    .map(|v| v as f64)
                    .collect(),
            );
        }

        let mut grid = vec![f64::NAN; g * g];
        for (bi, &beta) in axes.iter().enumerate() {
            for (ai, &alpha) in axes.iter().enumerate() {
                let deltas: HashMap<usize, Vec<f64>> = dir1_map
                    .iter()
                    .map(|(&off, d1)| {
                        let d2 = &dir2_map[&off];
                        let delta = d1
                            .iter()
                            .zip(d2.iter())
                            .map(|(&a, &b)| alpha * a + beta * b)
                            .collect();
                        (off, delta)
                    })
                    .collect();
                let out = synth_fwd_with_deltas(bytes, &analysis, &deltas);
                if out.len() == n && n > 0 {
                    let rms = (out
                        .iter()
                        .zip(center_out.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        / n as f64)
                        .sqrt();
                    grid[bi * g + ai] = rms;
                }
            }
        }
        all_grids.push(grid);
    }

    serde_wasm_bindgen::to_value(&TomoResult {
        grids: all_grids,
        axes,
        center_rms,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Single-layer 2D landscape grid — perturbs only the weight tensor of the given op.
/// Much faster than full-model landscape for interactive per-layer exploration.
#[wasm_bindgen]
pub fn layer_landscape_grid(
    bytes: &[u8],
    op_index: u32,
    seed1: u32,
    seed2: u32,
    grid_size: u32,
    radius: f32,
) -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    let g = (grid_size as usize).max(3);
    let analysis = analyze_with_target_without_step_response(bytes, "", "android_mid_a55")
        .map_err(|e| JsValue::from_str(&e))?;

    let op_idx = op_index as usize;
    let op = analysis
        .ops
        .get(op_idx)
        .ok_or_else(|| JsValue::from_str("op_index out of range"))?;

    let wt_idx = op.inputs.get(1).copied().unwrap_or(-1);
    if wt_idx < 0 {
        return Err(JsValue::from_str("op has no weight tensor at input[1]"));
    }
    let wt = analysis
        .tensors
        .get(wt_idx as usize)
        .filter(|t| t.constant_buffer && !t.sparse_storage)
        .ok_or_else(|| JsValue::from_str("weight tensor not found or not constant"))?;

    let mut rng1 = Xsr64::new(seed1 as u64 | 1);
    let mut rng2 = Xsr64::new(seed2 as u64 | 1);
    let mut dir1_map: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut dir2_map: HashMap<usize, Vec<f64>> = HashMap::new();
    dir1_map.insert(
        wt.buffer_data_offset,
        lsc_filter_dir(bytes, wt, &mut rng1)
            .into_iter()
            .map(|v| v as f64)
            .collect(),
    );
    dir2_map.insert(
        wt.buffer_data_offset,
        lsc_filter_dir(bytes, wt, &mut rng2)
            .into_iter()
            .map(|v| v as f64)
            .collect(),
    );

    let center_out = synth_fwd_with_deltas(bytes, &analysis, &HashMap::new());
    let n = center_out.len();

    let step = if g > 1 {
        2.0 * radius as f64 / (g - 1) as f64
    } else {
        0.0
    };
    let axes: Vec<f64> = (0..g).map(|i| -radius as f64 + i as f64 * step).collect();
    let mut grid = vec![f64::NAN; g * g];

    for (bi, &beta) in axes.iter().enumerate() {
        for (ai, &alpha) in axes.iter().enumerate() {
            let deltas: HashMap<usize, Vec<f64>> = dir1_map
                .iter()
                .map(|(&off, d1)| {
                    let d2 = &dir2_map[&off];
                    let delta = d1
                        .iter()
                        .zip(d2.iter())
                        .map(|(&a, &b)| alpha * a + beta * b)
                        .collect();
                    (off, delta)
                })
                .collect();
            let out = synth_fwd_with_deltas(bytes, &analysis, &deltas);
            if out.len() == n && n > 0 {
                let rms = (out
                    .iter()
                    .zip(center_out.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    / n as f64)
                    .sqrt();
                grid[bi * g + ai] = rms;
            }
        }
    }

    serde_wasm_bindgen::to_value(&SynthGrid { grid, axes })
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
