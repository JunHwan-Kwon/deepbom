use super::*;

pub(super) fn uf_find(parent: &mut [i32], mut x: i32) -> i32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize]; // path halving
        x = parent[x as usize];
    }
    x
}

// Spatial influence computation implemented in the Rust/WASM analysis core.
// Ports the JS computeInputInfluence / computeOutputInfluence BFS algorithms.

fn infl_resample(map: &[f32], h: usize, w: usize, th: usize, tw: usize) -> Vec<f32> {
    if h == th && w == tw {
        return map.to_vec();
    }
    let th = th.max(1);
    let tw = tw.max(1);
    let mut out = vec![0.0f32; th * tw];
    for y in 0..th {
        let sy = y as f32 * (h as f32 - 1.0) / (th as f32 - 1.0).max(1.0);
        let y0 = (sy.floor() as usize).min(h.saturating_sub(1));
        let y1 = (y0 + 1).min(h.saturating_sub(1));
        let fy = sy - y0 as f32;
        for x in 0..tw {
            let sx = x as f32 * (w as f32 - 1.0) / (tw as f32 - 1.0).max(1.0);
            let x0 = (sx.floor() as usize).min(w.saturating_sub(1));
            let x1 = (x0 + 1).min(w.saturating_sub(1));
            let fx = sx - x0 as f32;
            out[y * tw + x] = map[y0 * w + x0] * (1.0 - fy) * (1.0 - fx)
                + map[y0 * w + x1] * (1.0 - fy) * fx
                + map[y1 * w + x0] * fy * (1.0 - fx)
                + map[y1 * w + x1] * fy * fx;
        }
    }
    out
}

fn infl_add(a: &InflMap, b: &InflMap) -> InflMap {
    if a.h == b.h && a.w == b.w {
        InflMap {
            map: a.map.iter().zip(&b.map).map(|(&x, &y)| x + y).collect(),
            h: a.h,
            w: a.w,
        }
    } else {
        let rb = infl_resample(&b.map, b.h, b.w, a.h, a.w);
        InflMap {
            map: a.map.iter().zip(&rb).map(|(&x, &y)| x + y).collect(),
            h: a.h,
            w: a.w,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn infl_forward_conv(
    inf: &InflMap,
    kimp: &[f32],
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    th: usize,
    tw: usize,
) -> InflMap {
    let pad_h = ((th.saturating_sub(1)) * sh + kh).saturating_sub(inf.h) / 2;
    let pad_w = ((tw.saturating_sub(1)) * sw + kw).saturating_sub(inf.w) / 2;
    let mut out = vec![0.0f32; th * tw];
    for yo in 0..th {
        for xo in 0..tw {
            let mut s = 0.0f32;
            for ky in 0..kh {
                let yi = yo * sh + ky;
                if yi < pad_h || yi >= inf.h + pad_h {
                    continue;
                }
                let yi = yi - pad_h;
                for kx in 0..kw {
                    let xi = xo * sw + kx;
                    if xi < pad_w || xi >= inf.w + pad_w {
                        continue;
                    }
                    s += inf.map[yi * inf.w + (xi - pad_w)] * kimp[ky * kw + kx];
                }
            }
            out[yo * tw + xo] = s;
        }
    }
    InflMap {
        map: out,
        h: th,
        w: tw,
    }
}

#[allow(clippy::too_many_arguments)]
fn infl_transposed_conv(
    inf: &InflMap,
    kimp: &[f32],
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    th: usize,
    tw: usize,
) -> InflMap {
    let raw_h = inf.h.saturating_sub(1) * sh + kh;
    let raw_w = inf.w.saturating_sub(1) * sw + kw;
    let mut raw = vec![0.0f32; raw_h * raw_w];
    for y in 0..inf.h {
        for x in 0..inf.w {
            let v = inf.map[y * inf.w + x];
            if v == 0.0 {
                continue;
            }
            for ky in 0..kh {
                for kx in 0..kw {
                    let ry = y * sh + ky;
                    let rx = x * sw + kx;
                    if ry < raw_h && rx < raw_w {
                        raw[ry * raw_w + rx] += v * kimp[ky * kw + kx];
                    }
                }
            }
        }
    }
    let crop_t = raw_h.saturating_sub(th) / 2;
    let crop_l = raw_w.saturating_sub(tw) / 2;
    if crop_t == 0 && crop_l == 0 && raw_h == th && raw_w == tw {
        return InflMap {
            map: raw,
            h: raw_h,
            w: raw_w,
        };
    }
    let mut out = vec![0.0f32; th * tw];
    for y in 0..th {
        for x in 0..tw {
            out[y * tw + x] =
                raw[(y + crop_t).min(raw_h - 1) * raw_w + (x + crop_l).min(raw_w - 1)];
        }
    }
    InflMap {
        map: out,
        h: th,
        w: tw,
    }
}

// Kernel importance: collapses [outCh, kH, kW, inCh] weights → [kH, kW] mean abs importance
fn infl_conv_kernel_imp(
    bytes: &[u8],
    tensor: &TensorInfo,
    op_name: &str,
) -> Option<(Vec<f32>, usize, usize)> {
    if tensor.shape.len() != 4 || !tensor.constant_buffer || tensor.sparse_storage {
        return None;
    }
    let (out_ch, k_h, k_w, in_ch) = (
        tensor.shape[0] as usize,
        tensor.shape[1] as usize,
        tensor.shape[2] as usize,
        tensor.shape[3] as usize,
    );
    let end = tensor.buffer_data_offset + tensor.buffer_data_length;
    if end > bytes.len() {
        return None;
    }
    let raw = &bytes[tensor.buffer_data_offset..end];
    let mut imp = vec![0.0f32; k_h * k_w];

    if op_name == "DEPTHWISE_CONV_2D" {
        match tensor.dtype.as_str() {
            "INT8" => {
                for ky in 0..k_h {
                    for kx in 0..k_w {
                        let mut s = 0.0f32;
                        for c in 0..in_ch {
                            let idx = ky * k_w * in_ch + kx * in_ch + c;
                            if idx < raw.len() {
                                s += (raw[idx] as i8).unsigned_abs() as f32;
                            }
                        }
                        imp[ky * k_w + kx] = s / in_ch.max(1) as f32;
                    }
                }
            }
            "FLOAT32" => {
                for ky in 0..k_h {
                    for kx in 0..k_w {
                        let mut s = 0.0f32;
                        for c in 0..in_ch {
                            let idx = (ky * k_w * in_ch + kx * in_ch + c) * 4;
                            if idx + 4 <= raw.len() {
                                s += f32::from_le_bytes([
                                    raw[idx],
                                    raw[idx + 1],
                                    raw[idx + 2],
                                    raw[idx + 3],
                                ])
                                .abs();
                            }
                        }
                        imp[ky * k_w + kx] = s / in_ch.max(1) as f32;
                    }
                }
            }
            _ => return None,
        }
    } else {
        let norm = (out_ch * in_ch).max(1) as f32;
        match tensor.dtype.as_str() {
            "INT8" if raw.len() >= out_ch * k_h * k_w * in_ch => {
                for oc in 0..out_ch {
                    for ky in 0..k_h {
                        for kx in 0..k_w {
                            let mut s = 0.0f32;
                            for ic in 0..in_ch {
                                s += (raw
                                    [oc * k_h * k_w * in_ch + ky * k_w * in_ch + kx * in_ch + ic]
                                    as i8)
                                    .unsigned_abs() as f32;
                            }
                            imp[ky * k_w + kx] += s / norm;
                        }
                    }
                }
            }
            "FLOAT32" if raw.len() >= out_ch * k_h * k_w * in_ch * 4 => {
                for oc in 0..out_ch {
                    for ky in 0..k_h {
                        for kx in 0..k_w {
                            let mut s = 0.0f32;
                            for ic in 0..in_ch {
                                let idx =
                                    (oc * k_h * k_w * in_ch + ky * k_w * in_ch + kx * in_ch + ic)
                                        * 4;
                                s += f32::from_le_bytes([
                                    raw[idx],
                                    raw[idx + 1],
                                    raw[idx + 2],
                                    raw[idx + 3],
                                ])
                                .abs();
                            }
                            imp[ky * k_w + kx] += s / norm;
                        }
                    }
                }
            }
            _ => return None,
        }
    }
    Some((imp, k_h, k_w))
}

fn infl_tensor_hw(
    tensors: &[TensorInfo],
    idx: i32,
    fallback_h: usize,
    fallback_w: usize,
) -> (usize, usize) {
    if idx < 0 || idx as usize >= tensors.len() {
        return (fallback_h, fallback_w);
    }
    let t = &tensors[idx as usize];
    let h = t.shape.get(1).copied().unwrap_or(1).max(1) as usize;
    let w = t.shape.get(2).copied().unwrap_or(1).max(1) as usize;
    (h, w)
}

fn op_backward_infl(
    op: &OpInfo,
    out_inf: &InflMap,
    tensors: &[TensorInfo],
    bytes: &[u8],
) -> Vec<(usize, InflMap)> {
    let mut res: Vec<(usize, InflMap)> = Vec::new();
    let name = op.name.as_str();

    macro_rules! in_hw {
        ($idx:expr) => {
            infl_tensor_hw(tensors, $idx, out_inf.h, out_inf.w)
        };
    }
    macro_rules! resamp_to {
        ($idx:expr) => {{
            let (th, tw) = in_hw!($idx);
            InflMap {
                map: infl_resample(&out_inf.map, out_inf.h, out_inf.w, th, tw),
                h: th,
                w: tw,
            }
        }};
    }

    match name {
        "CONV_2D" | "DEPTHWISE_CONV_2D" | "TRANSPOSE_CONV" => {
            let act_in = op.inputs.first().copied().unwrap_or(-1);
            if act_in < 0 {
                return res;
            }
            let (in_h, in_w) = in_hw!(act_in);
            let w_idx = op.inputs.get(1).copied().unwrap_or(-1);
            let mut done = false;
            if w_idx >= 0
                && (w_idx as usize) < tensors.len()
                && tensors[w_idx as usize].constant_buffer
            {
                if let Some((ki, kh, kw)) =
                    infl_conv_kernel_imp(bytes, &tensors[w_idx as usize], name)
                {
                    let sh = ((in_h as f32 / out_inf.h.max(1) as f32).round() as usize).max(1);
                    let sw = ((in_w as f32 / out_inf.w.max(1) as f32).round() as usize).max(1);
                    res.push((
                        act_in as usize,
                        infl_transposed_conv(out_inf, &ki, kh, kw, sh, sw, in_h, in_w),
                    ));
                    done = true;
                }
            }
            if !done {
                res.push((act_in as usize, resamp_to!(act_in)));
            }
        }
        "FULLY_CONNECTED" => {
            let act_in = op.inputs.first().copied().unwrap_or(-1);
            if act_in < 0 {
                return res;
            }
            let w_idx = op.inputs.get(1).copied().unwrap_or(-1);
            let mut done = false;
            if w_idx >= 0 && (w_idx as usize) < tensors.len() {
                let wt = &tensors[w_idx as usize];
                if wt.constant_buffer && !wt.sparse_storage && wt.shape.len() == 2 {
                    let out_f = wt.shape[0] as usize;
                    let in_f = wt.shape[1] as usize;
                    let end = wt.buffer_data_offset + wt.buffer_data_length;
                    if end <= bytes.len() {
                        let raw = &bytes[wt.buffer_data_offset..end];
                        let out_vec = &out_inf.map;
                        let mut new_imp = vec![0.0f32; in_f];
                        let ok = match wt.dtype.as_str() {
                            "INT8" if raw.len() >= out_f * in_f => {
                                for i in 0..in_f {
                                    let mut s = 0.0f32;
                                    for o in 0..out_f {
                                        let v = out_vec.get(o).copied().unwrap_or(
                                            out_vec.first().copied().unwrap_or(0.0) / out_f as f32,
                                        );
                                        s += (raw[o * in_f + i] as i8).unsigned_abs() as f32 * v;
                                    }
                                    new_imp[i] = s / out_f.max(1) as f32;
                                }
                                true
                            }
                            "FLOAT32" if raw.len() >= out_f * in_f * 4 => {
                                #[allow(clippy::needless_range_loop)]
                                for i in 0..in_f {
                                    let mut s = 0.0f32;
                                    for o in 0..out_f {
                                        let v = out_vec.get(o).copied().unwrap_or(
                                            out_vec.first().copied().unwrap_or(0.0) / out_f as f32,
                                        );
                                        let idx = (o * in_f + i) * 4;
                                        s += f32::from_le_bytes([
                                            raw[idx],
                                            raw[idx + 1],
                                            raw[idx + 2],
                                            raw[idx + 3],
                                        ])
                                        .abs()
                                            * v;
                                    }
                                    new_imp[i] = s / out_f.max(1) as f32;
                                }
                                true
                            }
                            _ => false,
                        };
                        if ok {
                            res.push((
                                act_in as usize,
                                InflMap {
                                    map: new_imp,
                                    h: in_f,
                                    w: 1,
                                },
                            ));
                            done = true;
                        }
                    }
                }
            }
            if !done {
                res.push((act_in as usize, out_inf.clone()));
            }
        }
        "MEAN" => {
            let act_in = op.inputs.first().copied().unwrap_or(-1);
            if act_in < 0 {
                return res;
            }
            let (in_h, in_w) = in_hw!(act_in);
            let mean_val = if !out_inf.map.is_empty() {
                out_inf.map.iter().sum::<f32>() / out_inf.map.len() as f32
            } else {
                0.0
            };
            res.push((
                act_in as usize,
                InflMap {
                    map: vec![mean_val; in_h * in_w],
                    h: in_h,
                    w: in_w,
                },
            ));
        }
        "ADD" | "SUB" | "MUL" | "MAXIMUM" | "MINIMUM" | "CONCATENATION" => {
            for &idx in &op.inputs {
                if idx < 0 {
                    continue;
                }
                res.push((idx as usize, resamp_to!(idx)));
            }
        }
        "AVERAGE_POOL_2D" | "MAX_POOL_2D" => {
            let act_in = op.inputs.first().copied().unwrap_or(-1);
            if act_in < 0 {
                return res;
            }
            let (in_h, in_w) = in_hw!(act_in);
            let sh = ((in_h as f32 / out_inf.h.max(1) as f32).round() as usize).max(1);
            let sw = ((in_w as f32 / out_inf.w.max(1) as f32).round() as usize).max(1);
            let kh = sh.max(in_h.saturating_sub(out_inf.h.saturating_sub(1) * sh));
            let kw = sw.max(in_w.saturating_sub(out_inf.w.saturating_sub(1) * sw));
            let kern = vec![1.0f32 / (kh * kw).max(1) as f32; kh * kw];
            res.push((
                act_in as usize,
                infl_transposed_conv(out_inf, &kern, kh, kw, sh, sw, in_h, in_w),
            ));
        }
        _ => {
            let act_in = op.inputs.first().copied().unwrap_or(-1);
            if act_in >= 0 {
                res.push((act_in as usize, resamp_to!(act_in)));
            }
        }
    }
    res
}

fn op_forward_infl(
    op: &OpInfo,
    in_inf: &InflMap,
    tensors: &[TensorInfo],
    bytes: &[u8],
) -> Vec<(usize, InflMap)> {
    let mut res: Vec<(usize, InflMap)> = Vec::new();
    let name = op.name.as_str();

    let first_out = op.outputs.first().copied().unwrap_or(-1);
    if first_out < 0 {
        return res;
    }

    let out_hw = |idx: i32| infl_tensor_hw(tensors, idx, in_inf.h, in_inf.w);

    let push_resampled = |res: &mut Vec<(usize, InflMap)>, idx: i32, inf: &InflMap| {
        if idx < 0 {
            return;
        }
        let (oh, ow) = out_hw(idx);
        if oh == inf.h && ow == inf.w {
            res.push((idx as usize, inf.clone()));
        } else {
            res.push((
                idx as usize,
                InflMap {
                    map: infl_resample(&inf.map, inf.h, inf.w, oh, ow),
                    h: oh,
                    w: ow,
                },
            ));
        }
    };

    let (def_h, def_w) = out_hw(first_out);

    match name {
        "CONV_2D" | "DEPTHWISE_CONV_2D" => {
            let w_idx = op.inputs.get(1).copied().unwrap_or(-1);
            let mut done = false;
            if w_idx >= 0
                && (w_idx as usize) < tensors.len()
                && tensors[w_idx as usize].constant_buffer
            {
                if let Some((ki, kh, kw)) =
                    infl_conv_kernel_imp(bytes, &tensors[w_idx as usize], name)
                {
                    let sh = ((in_inf.h as f32 / def_h.max(1) as f32).round() as usize).max(1);
                    let sw = ((in_inf.w as f32 / def_w.max(1) as f32).round() as usize).max(1);
                    res.push((
                        first_out as usize,
                        infl_forward_conv(in_inf, &ki, kh, kw, sh, sw, def_h, def_w),
                    ));
                    done = true;
                }
            }
            if !done {
                push_resampled(&mut res, first_out, in_inf);
            }
        }
        "TRANSPOSE_CONV" => {
            let w_idx = op.inputs.get(1).copied().unwrap_or(-1);
            let mut done = false;
            if w_idx >= 0
                && (w_idx as usize) < tensors.len()
                && tensors[w_idx as usize].constant_buffer
            {
                if let Some((ki, kh, kw)) =
                    infl_conv_kernel_imp(bytes, &tensors[w_idx as usize], name)
                {
                    let sh = ((def_h as f32 / in_inf.h.max(1) as f32).round() as usize).max(1);
                    let sw = ((def_w as f32 / in_inf.w.max(1) as f32).round() as usize).max(1);
                    res.push((
                        first_out as usize,
                        infl_transposed_conv(in_inf, &ki, kh, kw, sh, sw, def_h, def_w),
                    ));
                    done = true;
                }
            }
            if !done {
                push_resampled(&mut res, first_out, in_inf);
            }
        }
        "FULLY_CONNECTED" => {
            let w_idx = op.inputs.get(1).copied().unwrap_or(-1);
            let mut done = false;
            if w_idx >= 0 && (w_idx as usize) < tensors.len() {
                let wt = &tensors[w_idx as usize];
                if wt.constant_buffer && !wt.sparse_storage && wt.shape.len() == 2 {
                    let out_f = wt.shape[0] as usize;
                    let in_f = wt.shape[1] as usize;
                    let end = wt.buffer_data_offset + wt.buffer_data_length;
                    if end <= bytes.len() {
                        let raw = &bytes[wt.buffer_data_offset..end];
                        let in_vec = &in_inf.map;
                        let mut out_imp = vec![0.0f32; out_f];
                        let ok = match wt.dtype.as_str() {
                            "INT8" if raw.len() >= out_f * in_f => {
                                for o in 0..out_f {
                                    let mut s = 0.0f32;
                                    for i in 0..in_f {
                                        let v = in_vec.get(i).copied().unwrap_or(0.0);
                                        s += (raw[o * in_f + i] as i8).unsigned_abs() as f32 * v;
                                    }
                                    out_imp[o] = s / in_f.max(1) as f32;
                                }
                                true
                            }
                            "FLOAT32" if raw.len() >= out_f * in_f * 4 => {
                                #[allow(clippy::needless_range_loop)]
                                for o in 0..out_f {
                                    let mut s = 0.0f32;
                                    for i in 0..in_f {
                                        let v = in_vec.get(i).copied().unwrap_or(0.0);
                                        let idx = (o * in_f + i) * 4;
                                        s += f32::from_le_bytes([
                                            raw[idx],
                                            raw[idx + 1],
                                            raw[idx + 2],
                                            raw[idx + 3],
                                        ])
                                        .abs()
                                            * v;
                                    }
                                    out_imp[o] = s / in_f.max(1) as f32;
                                }
                                true
                            }
                            _ => false,
                        };
                        if ok {
                            res.push((
                                first_out as usize,
                                InflMap {
                                    map: out_imp,
                                    h: out_f,
                                    w: 1,
                                },
                            ));
                            done = true;
                        }
                    }
                }
            }
            if !done {
                push_resampled(&mut res, first_out, in_inf);
            }
        }
        "MEAN" => {
            let v = if !in_inf.map.is_empty() {
                in_inf.map.iter().sum::<f32>() / in_inf.map.len() as f32
            } else {
                0.0
            };
            res.push((
                first_out as usize,
                InflMap {
                    map: vec![v],
                    h: 1,
                    w: 1,
                },
            ));
        }
        "AVERAGE_POOL_2D" | "MAX_POOL_2D" => {
            let sh = ((in_inf.h as f32 / def_h.max(1) as f32).round() as usize).max(1);
            let sw = ((in_inf.w as f32 / def_w.max(1) as f32).round() as usize).max(1);
            let kh = sh.max(in_inf.h.saturating_sub(def_h.saturating_sub(1) * sh));
            let kw = sw.max(in_inf.w.saturating_sub(def_w.saturating_sub(1) * sw));
            let kern = vec![1.0f32 / (kh * kw).max(1) as f32; kh * kw];
            res.push((
                first_out as usize,
                infl_forward_conv(in_inf, &kern, kh, kw, sh, sw, def_h, def_w),
            ));
        }
        _ => {
            for &oi in &op.outputs {
                push_resampled(&mut res, oi, in_inf);
            }
        }
    }
    res
}

const INFL_STOP_OPS: &[&str] = &["SOFTMAX"];
const INFL_MAX_OPS: usize = 80;

pub(super) fn compute_influence_bwd(
    analysis: &Analysis,
    bytes: &[u8],
    start_op_idx: usize,
) -> Option<InfluenceMapResult> {
    let mut producers: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for op in &analysis.ops {
        for &oi in &op.outputs {
            if oi >= 0 {
                producers.insert(oi as usize, op.index);
            }
        }
    }

    let start_op = analysis.ops.iter().find(|o| o.index == start_op_idx)?;
    let mut tensor_inf: std::collections::HashMap<usize, InflMap> =
        std::collections::HashMap::new();

    for &out_idx in &start_op.outputs {
        if out_idx < 0 || out_idx as usize >= analysis.tensors.len() {
            continue;
        }
        let t = &analysis.tensors[out_idx as usize];
        let h = t.shape.get(1).copied().unwrap_or(1).max(1) as usize;
        let w = t.shape.get(2).copied().unwrap_or(1).max(1) as usize;
        tensor_inf.insert(
            out_idx as usize,
            InflMap {
                map: vec![1.0f32; h * w],
                h,
                w,
            },
        );
    }

    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    queue.push_back(start_op_idx);
    let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();

    while let Some(op_idx) = queue.pop_front() {
        if visited.len() >= INFL_MAX_OPS {
            break;
        }
        if !visited.insert(op_idx) {
            continue;
        }

        let op = match analysis.ops.iter().find(|o| o.index == op_idx) {
            Some(o) => o,
            None => continue,
        };
        if INFL_STOP_OPS.contains(&op.name.as_str()) {
            continue;
        }

        let out_inf = op
            .outputs
            .iter()
            .find_map(|&oi| {
                if oi >= 0 {
                    tensor_inf.get(&(oi as usize))
                } else {
                    None
                }
            })
            .cloned()
            .or_else(|| {
                // Fallback seed for start_op itself
                if op.index == start_op_idx {
                    let oi = op.outputs.first().copied().unwrap_or(-1);
                    if oi >= 0 && (oi as usize) < analysis.tensors.len() {
                        let t = &analysis.tensors[oi as usize];
                        let h = t.shape.get(1).copied().unwrap_or(1).max(1) as usize;
                        let w = t.shape.get(2).copied().unwrap_or(1).max(1) as usize;
                        return Some(InflMap {
                            map: vec![1.0f32; h * w],
                            h,
                            w,
                        });
                    }
                }
                None
            });
        let out_inf = match out_inf {
            Some(i) => i,
            None => continue,
        };

        for (tid, inf) in op_backward_infl(op, &out_inf, &analysis.tensors, bytes) {
            tensor_inf
                .entry(tid)
                .and_modify(|e| *e = infl_add(e, &inf))
                .or_insert(inf);
            if let Some(&prod) = producers.get(&tid) {
                if !visited.contains(&prod) {
                    queue.push_back(prod);
                }
            }
        }
    }

    let best = tensor_inf
        .iter()
        .max_by_key(|(tid, inf)| {
            let is_root = !producers.contains_key(tid);
            inf.h * inf.w * if is_root { 2 } else { 1 }
        })
        .map(|(_, inf)| inf)?;
    if best.h * best.w <= 1 {
        return None;
    }
    let max_val = best.map.iter().cloned().fold(0.0f32, f32::max);
    Some(InfluenceMapResult {
        map: best.map.clone(),
        h: best.h,
        w: best.w,
        max_val,
        chain_len: visited.len(),
    })
}

pub(super) fn compute_influence_fwd(
    analysis: &Analysis,
    bytes: &[u8],
    start_op_idx: usize,
) -> Option<InfluenceMapResult> {
    let mut consumers: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for op in &analysis.ops {
        for &ii in &op.inputs {
            if ii >= 0 {
                consumers.entry(ii as usize).or_default().push(op.index);
            }
        }
    }

    let start_op = analysis.ops.iter().find(|o| o.index == start_op_idx)?;
    let start_out_set: std::collections::HashSet<usize> = start_op
        .outputs
        .iter()
        .filter(|&&i| i >= 0)
        .map(|&i| i as usize)
        .collect();

    let mut tensor_inf: std::collections::HashMap<usize, InflMap> =
        std::collections::HashMap::new();
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

    for &out_idx in &start_op.outputs {
        if out_idx < 0 || out_idx as usize >= analysis.tensors.len() {
            continue;
        }
        let t = &analysis.tensors[out_idx as usize];
        let h = t.shape.get(1).copied().unwrap_or(1).max(1) as usize;
        let w = t.shape.get(2).copied().unwrap_or(1).max(1) as usize;
        tensor_inf.insert(
            out_idx as usize,
            InflMap {
                map: vec![1.0f32; h * w],
                h,
                w,
            },
        );
        if let Some(c) = consumers.get(&(out_idx as usize)) {
            for &cop in c {
                queue.push_back(cop);
            }
        }
    }

    let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();

    while let Some(op_idx) = queue.pop_front() {
        if visited.len() >= INFL_MAX_OPS {
            break;
        }
        if !visited.insert(op_idx) {
            continue;
        }

        let op = match analysis.ops.iter().find(|o| o.index == op_idx) {
            Some(o) => o,
            None => continue,
        };
        if INFL_STOP_OPS.contains(&op.name.as_str()) {
            continue;
        }

        let in_inf = op
            .inputs
            .iter()
            .filter(|&&ii| ii >= 0)
            .filter_map(|&ii| tensor_inf.get(&(ii as usize)).cloned())
            .reduce(|a, b| infl_add(&a, &b));
        let in_inf = match in_inf {
            Some(i) => i,
            None => continue,
        };

        for (tid, inf) in op_forward_infl(op, &in_inf, &analysis.tensors, bytes) {
            tensor_inf
                .entry(tid)
                .and_modify(|e| *e = infl_add(e, &inf))
                .or_insert(inf.clone());
            if let Some(c) = consumers.get(&tid) {
                for &cop in c {
                    if !visited.contains(&cop) {
                        queue.push_back(cop);
                    }
                }
            }
        }
    }

    let best = tensor_inf
        .iter()
        .filter(|(tid, _)| !start_out_set.contains(tid))
        .max_by_key(|(tid, inf)| {
            let is_leaf = consumers
                .get(tid)
                .map(|c| c.iter().all(|ci| visited.contains(ci)))
                .unwrap_or(true);
            inf.h * inf.w * if is_leaf { 2 } else { 1 }
        })
        .map(|(_, inf)| inf)?;
    if best.h * best.w <= 1 {
        return None;
    }
    let max_val = best.map.iter().cloned().fold(0.0f32, f32::max);
    Some(InfluenceMapResult {
        map: best.map.clone(),
        h: best.h,
        w: best.w,
        max_val,
        chain_len: visited.len(),
    })
}

// Findings engine implemented in the Rust/WASM analysis core.
