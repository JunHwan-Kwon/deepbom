//! Bounded weight projections and comparisons. Host code only packages inputs.
use crate::portable_numerics::{NumericalComparison, NumericalMoments};
use serde::Serialize;
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

// Match the public per-tensor Weight analysis limit; do not narrow it in WASM.
const MAX_VALUES: usize = 10_000_000;
fn error(s: &str) -> JsValue {
    JsValue::from_str(s)
}
fn output(v: &Value) -> Result<JsValue, JsValue> {
    v.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|e| error(&e.to_string()))
}
fn missing(reason: &str) -> Value {
    json!({"status":"not_assessed","reason":reason})
}
fn finite(data: &[f64]) -> Result<(), JsValue> {
    if data.len() > MAX_VALUES || data.iter().any(|x| !x.is_finite()) {
        return Err(error(
            "Weight analysis requires bounded finite binary64 values",
        ));
    }
    Ok(())
}
fn dimension(x: f64) -> Result<usize, JsValue> {
    if !x.is_finite() || x < 0.0 || x.fract() != 0.0 || x > MAX_VALUES as f64 {
        return Err(error(
            "Weight dimension exceeds the bounded matrix contract",
        ));
    }
    Ok(x as usize)
}
fn matrix(data: &[f64], rows: f64, columns: f64) -> Result<(usize, usize), JsValue> {
    finite(data)?;
    let (r, c) = (dimension(rows)?, dimension(columns)?);
    if r.checked_mul(c) != Some(data.len()) {
        return Err(error("Weight matrix shape differs from its values"));
    }
    Ok((r, c))
}
fn layout(
    data: &[f64],
    shape: &[f64],
    axis: f64,
    order: &str,
) -> Result<(usize, usize, usize), JsValue> {
    finite(data)?;
    if shape.len() > 64 || !matches!(order, "first_axis_fastest" | "last_axis_fastest") {
        return Err(error("Unsupported weight shape rank or storage order"));
    }
    let a = dimension(axis)?;
    if a >= shape.len().max(1) {
        return Err(error("Weight axis outside shape"));
    }
    let dims = shape
        .iter()
        .map(|x| dimension(*x))
        .collect::<Result<Vec<_>, _>>()?;
    let count = if dims.contains(&0) {
        0
    } else {
        dims.iter()
            .try_fold(1usize, |n, x| n.checked_mul(*x))
            .ok_or_else(|| error("Weight shape product overflow"))?
    };
    if count != data.len() {
        return Err(error("Weight shape differs from its values"));
    }
    let rows = dims.get(a).copied().unwrap_or(1);
    let columns = if rows == 0 { 0 } else { data.len() / rows };
    // Empty matrices need no stride and must not overflow unrelated dimensions.
    let axes = if order == "first_axis_fastest" {
        &dims[..a]
    } else {
        &dims[(a + 1).min(dims.len())..]
    };
    let stride = if data.is_empty() {
        1
    } else {
        axes.iter().product()
    };
    Ok((rows, columns, stride))
}
fn unfold(data: &[f64], rows: usize, columns: usize, stride: usize) -> Vec<f64> {
    let mut values = vec![0.0; data.len()];
    let mut offsets = vec![0; rows];
    for (i, &value) in data.iter().enumerate() {
        let row = (i / stride) % rows;
        values[row * columns + offsets[row]] = value;
        offsets[row] += 1;
    }
    values
}
pub(crate) fn max_abs(data: &[f64]) -> f64 {
    data.iter().fold(0.0_f64, |v, x| v.max(x.abs()))
}

#[wasm_bindgen]
pub fn numerical_weight_permute(
    data: &[f64],
    shape: &[f64],
    order: &str,
    axes: &[f64],
) -> Result<Vec<f64>, JsValue> {
    layout(data, shape, 0.0, order)?;
    let permutation = axes
        .iter()
        .map(|x| dimension(*x))
        .collect::<Result<Vec<_>, _>>()?;
    let mut sorted = permutation.clone();
    sorted.sort_unstable();
    if sorted != (0..shape.len()).collect::<Vec<_>>() {
        return Err(error("invalid axis permutation"));
    }
    let dims = shape.iter().map(|x| *x as usize).collect::<Vec<_>>();
    let mut native = vec![1usize; shape.len()];
    let mut stride = 1usize;
    let storage: Vec<_> = if order == "first_axis_fastest" {
        (0..shape.len()).collect()
    } else {
        (0..shape.len()).rev().collect()
    };
    if data.is_empty() {
        return Ok(vec![]);
    }
    for i in storage {
        native[i] = stride;
        stride *= dims[i];
    }
    let mut values = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        let mut rest = i;
        let mut offset = 0;
        for &axis in permutation.iter().rev() {
            let c = rest % dims[axis];
            rest /= dims[axis];
            offset += c * native[axis];
        }
        values.push(data[offset]);
    }
    Ok(values)
}
fn cosine(a: &[f64], b: &[f64]) -> Result<Option<f64>, JsValue> {
    let mut comparison = NumericalComparison::new();
    for (left, right) in a.chunks(65536).zip(b.chunks(65536)) {
        comparison.add_chunk(left, right)?;
    }
    Ok(comparison.cosine())
}
#[wasm_bindgen]
pub fn numerical_weight_magnitude(data: &[f64]) -> Result<f64, JsValue> {
    finite(data)?;
    Ok(max_abs(data))
}
#[wasm_bindgen]
pub fn numerical_weight_matrix(
    data: &[f64],
    shape: &[f64],
    axis: f64,
    order: &str,
) -> Result<Vec<f64>, JsValue> {
    let (rows, columns, stride) = layout(data, shape, axis, order)?;
    Ok(unfold(data, rows, columns, stride))
}
fn root(parent: &[usize], mut i: usize) -> usize {
    while parent[i] != i {
        i = parent[i];
    }
    i
}
fn similarity(
    data: &[f64],
    rows: usize,
    columns: usize,
    limit: usize,
    threshold: f64,
) -> Result<Value, JsValue> {
    if rows > limit
        || rows
            .checked_mul(rows)
            .and_then(|n| n.checked_mul(columns))
            .unwrap_or(usize::MAX)
            > 20_000_000
    {
        return Ok(missing("complete_pairwise_matrix_exceeds_budget"));
    }
    // A zero-column matrix still has a quadratic result. Bound that allocation.
    if rows.saturating_mul(rows) > MAX_VALUES {
        return Ok(missing("complete_pairwise_matrix_exceeds_budget"));
    }
    let mut cells = vec![None; rows * rows];
    let mut pairs = Vec::new();
    let mut parent: Vec<usize> = (0..rows).collect();
    for a in 0..rows {
        for b in a..rows {
            let value = cosine(
                &data[a * columns..(a + 1) * columns],
                &data[b * columns..(b + 1) * columns],
            )?;
            cells[a * rows + b] = value;
            cells[b * rows + a] = value;
            if a != b && value.is_some_and(|v| v >= threshold) {
                pairs.push(json!({"a":a,"b":b,"cosine":value}));
                let target = root(&parent, b);
                parent[target] = root(&parent, a);
            }
        }
    }
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for i in 0..rows {
        let r = root(&parent, i);
        match groups.iter_mut().find(|g| g.0 == r) {
            Some(g) => g.1.push(i),
            None => groups.push((r, vec![i])),
        }
    }
    Ok(
        json!({"status":"assessed","method":"complete_signed_cosine_no_sampling","size":rows,"values":cells,
        "threshold":threshold,"pairs":pairs,"clusters":groups.into_iter().filter(|g|g.1.len()>1).map(|g|g.1).collect::<Vec<_>>(),
        "cluster_method":"connected_components_of_threshold_edges_not_all_pairs_equivalence","undefined_reason":"zero_norm_channels"}),
    )
}
#[wasm_bindgen]
pub fn numerical_weight_similarity(
    data: &[f64],
    rows: f64,
    columns: f64,
    limit: f64,
    threshold: f64,
) -> Result<JsValue, JsValue> {
    let (r, c) = matrix(data, rows, columns)?;
    let limit = dimension(limit)?;
    if !threshold.is_finite() || !(-1.0..=1.0).contains(&threshold) {
        return Err(error("Invalid cosine threshold"));
    }
    output(&similarity(data, r, c, limit, threshold)?)
}
#[derive(Serialize)]
struct Cell {
    count: usize,
    zero_count: usize,
    mean: Option<f64>,
    minimum: Option<f64>,
    maximum: Option<f64>,
}
fn tiles(data: &[f64], rows: usize, columns: usize, maximum: usize) -> Result<Value, JsValue> {
    let (height, width) = (rows.min(maximum), columns.min(maximum));
    if height == 0 || width == 0 {
        return Ok(missing("empty_matrix"));
    }
    if height.saturating_mul(width) > 65536 {
        return Err(error("Weight projection exceeds cell budget"));
    }
    let mut cells: Vec<Cell> = (0..height * width)
        .map(|_| Cell {
            count: 0,
            zero_count: 0,
            mean: None,
            minimum: None,
            maximum: None,
        })
        .collect();
    let mut sums: Vec<_> = (0..cells.len())
        .map(|_| NumericalMoments::new(false))
        .collect();
    for r in 0..rows {
        for c in 0..columns {
            let index = ((r as u64 * height as u64 / rows as u64) as usize) * width
                + (c as u64 * width as u64 / columns as u64) as usize;
            let v = data[r * columns + c];
            let cell = &mut cells[index];
            cell.count += 1;
            cell.zero_count += usize::from(v == 0.0);
            sums[index].add_chunk(&[v])?;
            cell.minimum = Some(cell.minimum.map_or(v, |a| {
                if a == 0.0 && v == 0.0 {
                    if a.is_sign_negative() || v.is_sign_negative() {
                        -0.0
                    } else {
                        0.0
                    }
                } else {
                    a.min(v)
                }
            }));
            cell.maximum = Some(cell.maximum.map_or(v, |a| {
                if a == 0.0 && v == 0.0 {
                    if a.is_sign_positive() || v.is_sign_positive() {
                        0.0
                    } else {
                        -0.0
                    }
                } else {
                    a.max(v)
                }
            }));
        }
    }
    for (cell, sum) in cells.iter_mut().zip(sums) {
        cell.mean = sum.mean(cell.count as f64)?;
    }
    Ok(
        json!({"status":"assessed","method":if height==rows&&width==columns{"exact_cells"}else{"complete_area_aggregation_no_sampling"},
        "rows":height,"columns":width,"source_rows":rows,"source_columns":columns,"cells":cells}),
    )
}
#[wasm_bindgen]
pub fn numerical_weight_tiles(
    data: &[f64],
    rows: f64,
    columns: f64,
    maximum: f64,
) -> Result<JsValue, JsValue> {
    let (r, c) = matrix(data, rows, columns)?;
    output(&tiles(data, r, c, dimension(maximum)?)?)
}
fn sparsity(
    data: &[f64],
    rows: usize,
    columns: usize,
    stride: usize,
    axis: usize,
    block: usize,
) -> Value {
    let zeros = data.iter().filter(|&&x| x == 0.0).count();
    let zero_blocks = data
        .chunks_exact(block)
        .filter(|xs| xs.iter().all(|&x| x == 0.0))
        .count();
    let values = unfold(data, rows, columns, stride);
    let (mut groups, mut compliant) = (0, 0);
    for c in 0..columns {
        for r in (0..rows / 4 * 4).step_by(4) {
            let nonzero = (0..4)
                .filter(|k| values[(r + k) * columns + c] != 0.0)
                .count();
            groups += 1;
            compliant += usize::from(nonzero <= 2);
        }
    }
    json!({"status":"assessed","zero_count":zeros.to_string(),"value_count":data.len().to_string(),
        "zero_fraction":if data.is_empty(){None}else{Some(zeros as f64/data.len() as f64)},
        "block":{"basis":"contiguous_serialized_values","size":block,"complete_blocks":data.len()/block,"all_zero_blocks":zero_blocks,"trailing_values":data.len()%block},
        "structured_2_4":{"axis":axis,"groups":groups,"compliant_groups":compliant,"trailing_values":rows%4*columns,
        "pattern_satisfied":if groups>0&&rows%4==0{Some(compliant==groups)}else{None},"scope":"selected_axis_pattern_only_not_backend_eligibility_or_speedup"}})
}
#[wasm_bindgen]
pub fn numerical_weight_sparsity(
    data: &[f64],
    shape: &[f64],
    order: &str,
    axis: f64,
    block: f64,
) -> Result<JsValue, JsValue> {
    let (r, c, s) = layout(data, shape, axis, order)?;
    let block = dimension(block)?;
    if block == 0 {
        return Err(error("Block size must be positive"));
    }
    output(&sparsity(data, r, c, s, axis as usize, block))
}
fn comparison(a: &[f64], b: &[f64]) -> Result<Value, JsValue> {
    let scale = max_abs(a).max(max_abs(b));
    let reference_scale = max_abs(a);
    let differences: Vec<_> = a.iter().zip(b).map(|(x, y)| y - x).collect();
    let difference_scale = if differences.iter().all(|x| x.is_finite()) {
        max_abs(&differences)
    } else {
        scale
    };
    let normalized = |i: usize| {
        if difference_scale == 0.0 {
            0.0
        } else if differences[i].is_finite() {
            differences[i] / difference_scale
        } else {
            b[i] / difference_scale - a[i] / difference_scale
        }
    };
    let (mut maximum, mut reference, mut changed) = (0.0_f64, 0.0, 0);
    for i in 0..a.len() {
        maximum = maximum.max(normalized(i).abs());
        changed += usize::from(a[i] != b[i]);
        if reference_scale != 0.0 {
            reference += (a[i] / reference_scale).powi(2);
        }
    }
    let (mut square, mut absolute) = (0.0, 0.0);
    for i in 0..a.len() {
        let unit = if maximum == 0.0 {
            0.0
        } else {
            normalized(i) / maximum
        };
        square += unit * unit;
        absolute += unit.abs();
    }
    let n = a.len() as f64;
    let rmse = if n > 0.0 {
        Some((square / n).sqrt() * maximum * difference_scale)
    } else {
        None
    };
    let reference_log = if reference_scale != 0.0 {
        Some(reference_scale.ln() + reference.ln() / 2.0)
    } else {
        None
    };
    let error_log = if maximum != 0.0 {
        Some(maximum.ln() + difference_scale.ln() + square.ln() / 2.0)
    } else {
        None
    };
    let relative = reference_log.map(|r| error_log.map_or(0.0, |e| (e - r).exp()));
    Ok(
        json!({"status":"assessed","count":a.len().to_string(),"changed_count":changed.to_string(),
        "mean_absolute_error":if n>0.0{Some(absolute/n*maximum*difference_scale)}else{None},"maximum_absolute_error":maximum*difference_scale,
        "rmse":rmse,"mse":rmse.map(|x|x*x),"relative_l2":relative,
        "relative_l2_status":if reference_log.is_none(){"zero_reference_norm"}else if relative.is_some_and(|x|x.is_finite()){"assessed"}else{"overflow"},
        "cosine":cosine(a,b)?,"sqnr_db":reference_log.zip(error_log).map(|(r,e)|20.0/std::f64::consts::LN_10*(r-e)),
        "sqnr_status":if reference_log.is_none(){"zero_reference_energy"}else if error_log.is_none(){"zero_error"}else{"finite"}}),
    )
}
#[wasm_bindgen]
pub fn numerical_weight_comparison(a: &[f64], b: &[f64]) -> Result<JsValue, JsValue> {
    finite(a)?;
    finite(b)?;
    if a.len() != b.len() {
        return output(&missing("value_count_mismatch"));
    }
    output(&comparison(a, b)?)
}
#[wasm_bindgen]
pub fn numerical_weight_differences(a: &[f64], b: &[f64]) -> Result<Vec<f64>, JsValue> {
    finite(a)?;
    finite(b)?;
    if a.len() != b.len() {
        return Err(error("Weight value count mismatch"));
    }
    Ok(a.iter().zip(b).map(|(x, y)| y - x).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn axis_and_storage() {
        assert_eq!(
            unfold(&[0., 1., 2., 3., 4., 5.], 3, 2, 1),
            vec![0., 3., 1., 4., 2., 5.]
        );
    }
    #[test]
    fn signs_and_empty_channels() {
        let s = similarity(&[1., 2., -1., -2., 0., 0.], 3, 2, 10, 0.5).unwrap();
        assert_eq!(
            s["values"],
            json!([1., -1., null, -1., 1., null, null, null, null])
        );
        assert_eq!(
            similarity(&[], 4000, 0, 5000, 0.5).unwrap()["status"],
            "not_assessed"
        );
    }
    #[test]
    fn exact_tile_mean_and_zeros() {
        let t = tiles(&[1e308, 1., -1e308], 1, 3, 1).unwrap();
        assert_eq!(t["cells"][0]["mean"].as_f64(), Some(1.0 / 3.0));
        let z = tiles(&[0., -0.], 1, 2, 1).unwrap();
        assert!(z["cells"][0]["minimum"]
            .as_f64()
            .unwrap()
            .is_sign_negative());
    }
    #[test]
    fn sparse_tail_is_not_compliant() {
        let s = sparsity(&[0., 1., 0., 2., 0.], 5, 1, 1, 0, 4);
        assert_eq!(s["structured_2_4"]["pattern_satisfied"], Value::Null);
        assert_eq!(s["block"]["trailing_values"], 1);
    }
    #[test]
    fn comparison_extremes() {
        let c = comparison(&[1e308], &[-1e308]).unwrap();
        assert_eq!(c["maximum_absolute_error"], Value::Null);
        assert_eq!(c["cosine"], -1.0);
        let c = comparison(&[0.], &[f64::from_bits(1)]).unwrap();
        assert_eq!(c["rmse"].as_f64(), Some(f64::from_bits(1)));
    }
}
