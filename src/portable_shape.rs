//! Exact dimension algebra and tensor shape kernels shared by all hosts.
//! Format/opset admission and evidence attachment remain adapter responsibilities.
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive, Zero};
use serde_json::{json, Value};
use std::collections::BTreeMap;

const SAFE: i128 = 9_007_199_254_740_991;
const MAX_RANK: usize = 100_000;
#[derive(Clone, Debug, PartialEq)]
enum Dim {
    Value(i64),
    Symbol(String),
    Unknown,
}
impl Dim {
    fn read(v: &Value) -> Self {
        if v["kind"] == "symbolic" {
            if let Some(s) = v["parameter"]
                .as_str()
                .filter(|s| !s.is_empty() && s.len() <= 1_048_576)
            {
                return Self::Symbol(s.into());
            }
        }
        let number = if v["kind"] == "value" { &v["value"] } else { v };
        number
            .as_i64()
            .filter(|x| (*x as i128).abs() <= SAFE)
            .map(Self::Value)
            .unwrap_or(Self::Unknown)
    }
    fn json(&self) -> Value {
        match self {
            Self::Value(n) => {
                json!({"kind":"value","value":n,"parameter":"","denotation":"","valueFieldCount":1})
            }
            Self::Symbol(s) => {
                json!({"kind":"symbolic","value":null,"parameter":s,"denotation":"","valueFieldCount":1})
            }
            Self::Unknown => {
                json!({"kind":"unknown","value":null,"parameter":"","denotation":"","valueFieldCount":0})
            }
        }
    }
    fn number(&self) -> Option<i64> {
        if let Self::Value(n) = self {
            Some(*n)
        } else {
            None
        }
    }
    fn known(&self) -> bool {
        matches!(self, Self::Value(n) if *n >= 0) || matches!(self, Self::Symbol(_))
    }
    fn key(&self) -> String {
        match self {
            Self::Value(n) => format!("v:{n}"),
            Self::Symbol(s) => format!("s:{s}"),
            _ => "?".into(),
        }
    }
}
fn symbol(s: String) -> Dim {
    if s.len() <= 1_048_576 {
        Dim::Symbol(s)
    } else {
        Dim::Unknown
    }
}
fn exact(n: i128) -> Dim {
    if n.abs() <= SAFE {
        Dim::Value(n as i64)
    } else {
        Dim::Unknown
    }
}
fn dims(value: &Value) -> Option<Vec<Dim>> {
    let a = value.as_array()?;
    (a.len() <= MAX_RANK).then(|| a.iter().map(Dim::read).collect())
}
fn shape_json(a: Option<Vec<Dim>>) -> Value {
    a.map(|a| json!(a.iter().map(Dim::json).collect::<Vec<_>>()))
        .unwrap_or(Value::Null)
}
fn signed(v: &Value) -> Option<i64> {
    v.as_i64().filter(|x| (*x as i128).abs() <= SAFE)
}
fn add(a: &Dim, b: &Dim) -> Dim {
    match (a.number(), b.number()) {
        (Some(x), Some(y)) => exact(x as i128 + y as i128),
        (Some(0), _) => b.clone(),
        (_, Some(0)) => a.clone(),
        _ if a.known() && b.known() => symbol(format!("deepbom_expr:add({},{})", a.key(), b.key())),
        _ => Dim::Unknown,
    }
}
fn offset(a: &Dim, n: i128) -> Dim {
    if n.abs() > SAFE || !a.known() {
        return Dim::Unknown;
    }
    match a.number() {
        Some(v) if v as i128 + n >= 0 => exact(v as i128 + n),
        Some(_) => Dim::Unknown,
        None if n == 0 => a.clone(),
        None => symbol(format!("deepbom_expr:add({},{n})", a.key())),
    }
}
fn multiply(a: &Dim, b: &Dim) -> Dim {
    match (a.number(), b.number()) {
        (Some(x), Some(y)) => {
            let n = x as i128 * y as i128;
            if n >= 0 {
                exact(n)
            } else {
                Dim::Unknown
            }
        }
        (Some(0), _) | (_, Some(0)) => Dim::Value(0),
        (Some(1), _) => b.clone(),
        (_, Some(1)) => a.clone(),
        _ if a.known() && b.known() => symbol(format!("deepbom_expr:mul({},{})", a.key(), b.key())),
        _ => Dim::Unknown,
    }
}
fn factor(a: &Dim, n: i64) -> Dim {
    if n < 0 || !a.known() {
        return Dim::Unknown;
    }
    match a.number() {
        Some(x) => exact(x as i128 * n as i128),
        None if n == 0 => Dim::Value(0),
        None if n == 1 => a.clone(),
        None => symbol(format!("deepbom_expr:mul({},{n})", a.key())),
    }
}
fn product(a: &[Dim]) -> Dim {
    a.iter().fold(Dim::Value(1), |acc, n| multiply(&acc, n))
}
fn factors(a: &[Dim]) -> Option<(BigInt, Vec<String>)> {
    let mut numeric = vec![];
    let mut symbols = vec![];
    for d in a {
        match d {
            Dim::Value(v) if *v >= 0 => numeric.push(json!(v)),
            Dim::Symbol(s) => symbols.push(s.clone()),
            _ => return None,
        }
    }
    let n = BigInt::from(crate::portable_contracts::count_product(&numeric).ok()??);
    symbols.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
    Some((n, symbols))
}
fn expression(n: &BigInt, symbols: &[String]) -> String {
    let mut fields = vec![];
    if !n.is_one() || symbols.is_empty() {
        fields.push(format!("v:{n}"));
    }
    fields.extend(symbols.iter().map(|s| format!("s:{s}")));
    if fields.len() == 1 {
        fields[0].clone()
    } else {
        format!("deepbom_expr:mul({})", fields.join(","))
    }
}
fn quotient(a: &[Dim], b: &[Dim]) -> Dim {
    let (Some((n, ns)), Some((d, ds))) = (factors(a), factors(b)) else {
        return Dim::Unknown;
    };
    if d.is_zero() {
        return Dim::Unknown;
    }
    let mut counts = BTreeMap::new();
    for s in ds {
        *counts.entry(s).or_insert(0usize) += 1;
    }
    let mut numerator = vec![];
    for s in ns {
        if let Some(c) = counts.get_mut(&s).filter(|c| **c > 0) {
            *c -= 1;
        } else {
            numerator.push(s);
        }
    }
    let mut denominator: Vec<String> = counts
        .into_iter()
        .flat_map(|(s, n)| std::iter::repeat(s).take(n))
        .collect();
    denominator.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
    if denominator.is_empty() && (&n % &d).is_zero() {
        if let Some(v) = (&n / &d).to_i64().filter(|v| *v as i128 <= SAFE) {
            return numerator
                .iter()
                .fold(Dim::Value(v), |a, s| multiply(&a, &Dim::Symbol(s.clone())));
        }
    }
    symbol(format!(
        "deepbom_expr:reshape_quotient({},{})",
        expression(&n, &numerator),
        expression(&d, &denominator)
    ))
}
fn broadcast(shapes: &[Vec<Dim>]) -> Option<Vec<Dim>> {
    let rank = shapes.iter().map(Vec::len).max().unwrap_or(0);
    if shapes.len().saturating_mul(rank) > 1_000_000 {
        return None;
    }
    let mut out = vec![Dim::Unknown; rank];
    for offset in 1..=rank {
        let ds: Vec<_> = shapes
            .iter()
            .map(|s| {
                if offset > s.len() {
                    Dim::Value(1)
                } else {
                    s[s.len() - offset].clone()
                }
            })
            .filter(|d| d.number() != Some(1))
            .collect();
        if ds.is_empty() {
            out[rank - offset] = Dim::Value(1);
            continue;
        }
        let mut concrete: Vec<_> = ds.iter().filter_map(Dim::number).collect();
        concrete.sort();
        concrete.dedup();
        if concrete.len() > 1 {
            return None;
        }
        if concrete.len() == 1 {
            out[rank - offset] = Dim::Value(concrete[0]);
            continue;
        }
        if ds.iter().any(|d| !d.known()) {
            continue;
        }
        let mut symbols: Vec<_> = ds
            .iter()
            .filter_map(|d| {
                if let Dim::Symbol(s) = d {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .collect();
        symbols.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
        symbols.dedup();
        out[rank - offset] = match symbols.len() {
            0 => Dim::Unknown,
            1 => Dim::Symbol(symbols[0].clone()),
            _ => symbol(format!(
                "deepbom_expr:broadcast_dim({})",
                symbols
                    .iter()
                    .map(|s| format!("s:{s}"))
                    .collect::<Vec<_>>()
                    .join(",")
            )),
        };
    }
    Some(out)
}
fn reshape(v: &Value) -> Value {
    let fail = |s| json!({"shape":null,"reason":s});
    let Some(input) = dims(&v["input"]) else {
        return fail("reshape_data_shape_unknown");
    };
    let Some(request) = dims(&v["requested"]) else {
        return fail("reshape_shape_input_not_static_integer_data");
    };
    let allow = v["allow_zero"].as_bool().unwrap_or(false);
    if allow && request.contains(&Dim::Value(0)) && request.contains(&Dim::Value(-1)) {
        return fail("reshape_allowzero_one_cannot_mix_zero_and_minus_one");
    }
    let mut target = vec![];
    let mut infer = None;
    for (i, d) in request.iter().enumerate() {
        match d {
            Dim::Value(0) if !allow => {
                let Some(d) = input.get(i) else {
                    return fail("reshape_zero_copy_axis_out_of_range");
                };
                target.push(d.clone());
            }
            Dim::Value(-1) => {
                if infer.is_some() {
                    return fail("reshape_multiple_minus_one_dimensions");
                }
                infer = Some(i);
                target.push(Dim::Unknown);
            }
            Dim::Value(n) if *n >= 0 => target.push(d.clone()),
            Dim::Symbol(_) => target.push(d.clone()),
            _ => return fail("reshape_negative_dimension_invalid"),
        }
    }
    let known: Vec<_> = target
        .iter()
        .enumerate()
        .filter(|(i, _)| Some(*i) != infer)
        .map(|(_, d)| d.clone())
        .collect();
    if input.iter().all(|d| matches!(d,Dim::Value(n) if *n>=0))
        && known.iter().all(|d| d.number().is_some())
    {
        let (Some((n, _)), Some((d, _))) = (factors(&input), factors(&known)) else {
            return fail("reshape_input_cardinality_invalid");
        };
        if let Some(i) = infer {
            if d.is_zero() || !(&n % &d).is_zero() {
                return fail("reshape_inferred_dimension_not_integral");
            }
            let Some(x) = (&n / &d).to_i64().filter(|x| *x as i128 <= SAFE) else {
                return fail("reshape_inferred_dimension_outside_exact_static_integer_range");
            };
            target[i] = Dim::Value(x);
        } else if n != d {
            return fail("reshape_element_count_mismatch");
        }
    }
    if let Some(i) = infer {
        if !target[i].known() {
            target[i] = quotient(&input, &known);
        }
    }
    json!({"shape":shape_json(Some(target)),"reason":""})
}
fn ints(v: &Value) -> Option<Vec<i64>> {
    v.as_array()?.iter().map(signed).collect()
}
fn padded(v: &Value, len: usize, fallback: i64, zero: bool) -> Vec<i64> {
    let values = ints(v).unwrap_or_default();
    (0..len)
        .map(|i| {
            values
                .get(i)
                .copied()
                .filter(|x| if zero { *x >= 0 } else { *x > 0 })
                .unwrap_or(fallback)
        })
        .collect()
}
fn floor(n: i128, d: i128) -> i128 {
    n.div_euclid(d)
}
fn ceil(n: i128, d: i128) -> i128 {
    -floor(-n, d)
}
// Each axis is calculated independently, including mixed concrete/symbolic input.
fn spatial(input: &[Dim], kernels: &[i64], attrs: &Value) -> Option<Vec<Dim>> {
    let rank = input.len();
    if rank != kernels.len() || kernels.iter().any(|k| *k <= 0) {
        return None;
    }
    let strides = padded(&attrs["strides"], rank, 1, false);
    let dilations = padded(&attrs["dilations"], rank, 1, false);
    let mut pads = padded(&attrs["pads"], rank * 2, 0, true);
    let auto = attrs["auto_pad"].as_str().unwrap_or("NOTSET");
    if !["NOTSET", "VALID", "", "SAME_UPPER", "SAME_LOWER"].contains(&auto) {
        return None;
    }
    if auto == "VALID" {
        pads.fill(0);
    }
    let ceiling = attrs["ceil_mode"] == 1;
    let same = auto.starts_with("SAME_");
    let mut out = vec![];
    for (i, d) in input.iter().enumerate() {
        let s = strides[i] as i128;
        let adjust = pads[i] as i128 + pads[i + rank] as i128
            - dilations[i] as i128 * (kernels[i] as i128 - 1)
            - 1;
        if let Some(n) = d.number() {
            let n = n as i128;
            let mut result = if same {
                ceil(n, s)
            } else {
                (if ceiling {
                    ceil(n + adjust, s)
                } else {
                    floor(n + adjust, s)
                }) + 1
            };
            if !same && ceiling && (result - 1) * s >= n + pads[i] as i128 {
                result -= 1;
            }
            if result < 0 {
                return None;
            }
            out.push(exact(result));
            continue;
        }
        if !d.known() {
            out.push(Dim::Unknown);
            continue;
        }
        if same {
            out.push(if s == 1 {
                d.clone()
            } else {
                symbol(format!("deepbom_expr:ceil_div({},{s})", d.key()))
            });
            continue;
        }
        if s == 1 && !ceiling {
            out.push(offset(d, adjust + 1));
            continue;
        }
        let round = if ceiling { "ceil_div" } else { "floor_div" };
        let candidate = format!(
            "deepbom_expr:add({round}(deepbom_expr:add({},{adjust}),{s}),1)",
            d.key()
        );
        // Legal Pool padding can remove the final ceil-mode window. Use the same
        // correction for symbolic dimensions, rather than asserting the raw size.
        let expr = if ceiling {
            format!(
                "deepbom_expr:min({candidate},ceil_div(deepbom_expr:add({},{}),{s}))",
                d.key(),
                pads[i]
            )
        } else {
            candidate
        };
        out.push(symbol(expr));
    }
    Some(out)
}
fn can_match(a: &Dim, b: &Dim) -> bool {
    a.number().is_none() || b.number().is_none() || a.number() == b.number()
}
fn matmul(a: &[Dim], b: &[Dim]) -> Option<Vec<Dim>> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let av = a.len() == 1;
    let bv = b.len() == 1;
    let aa = if av {
        vec![Dim::Value(1), a[0].clone()]
    } else {
        a.to_vec()
    };
    let bb = if bv {
        vec![b[0].clone(), Dim::Value(1)]
    } else {
        b.to_vec()
    };
    if !can_match(aa.last()?, &bb[bb.len() - 2]) {
        return None;
    }
    let mut result = broadcast(&[aa[..aa.len() - 2].to_vec(), bb[..bb.len() - 2].to_vec()])?;
    if !av {
        result.push(aa[aa.len() - 2].clone());
    }
    if !bv {
        result.push(bb.last()?.clone());
    }
    Some(result)
}
fn strict(
    attrs: &Value,
    name: &str,
    len: usize,
    fallback: i64,
    zero: bool,
    optional: bool,
) -> Option<Vec<i64>> {
    if attrs.get(name).is_none() {
        return optional.then(|| vec![fallback; len]);
    }
    let values = ints(&attrs[name])?;
    (values.len() == len && values.iter().all(|n| if zero { *n >= 0 } else { *n > 0 }))
        .then_some(values)
}
fn conv(v: &Value, transpose: bool) -> Option<Vec<Dim>> {
    let w = ints(&v["weight"])?;
    if w.len() < 3 || w.len() > MAX_RANK {
        return None;
    }
    let x = if v["input"].is_null() {
        vec![Dim::Unknown; w.len()]
    } else {
        dims(&v["input"])?
    };
    if x.len() != w.len() {
        return None;
    }
    let a = &v["attributes"];
    let r = x.len() - 2;
    let group = signed(&a["group"]).unwrap_or(1);
    if group <= 0 || w[0] <= 0 || w[1] <= 0 {
        return None;
    }
    if !transpose {
        if x[1]
            .number()
            .is_some_and(|n| n as i128 != w[1] as i128 * group as i128)
            || w[0] % group != 0
        {
            return None;
        }
        let declared = ints(&a["kernel_shape"]).unwrap_or_default();
        let kernels: Vec<_> = w[2..]
            .iter()
            .enumerate()
            .map(|(i, d)| {
                if *d > 0 {
                    *d
                } else {
                    declared.get(i).copied().unwrap_or(0)
                }
            })
            .collect();
        let s = spatial(&x[2..], &kernels, a)?;
        let mut out = vec![x[0].clone(), Dim::Value(w[0])];
        out.extend(s);
        return Some(out);
    }
    if x[1].number().is_some_and(|n| n != w[0]) || w[2..].iter().any(|d| *d <= 0) {
        return None;
    }
    if a.get("kernel_shape").is_some() && strict(a, "kernel_shape", r, 0, false, false)? != w[2..] {
        return None;
    }
    let strides = strict(a, "strides", r, 1, false, true)?;
    let dilations = strict(a, "dilations", r, 1, false, true)?;
    let padding = strict(a, "output_padding", r, 0, true, true)?;
    if padding
        .iter()
        .enumerate()
        .any(|(i, p)| *p >= strides[i] && *p >= dilations[i])
    {
        return None;
    }
    let s: Vec<Dim> = if a.get("output_shape").is_some() {
        strict(a, "output_shape", r, 0, true, false)?
            .into_iter()
            .map(Dim::Value)
            .collect()
    } else {
        let auto = a["auto_pad"].as_str().unwrap_or("NOTSET");
        if auto == "SAME_UPPER" || auto == "SAME_LOWER" {
            x[2..]
                .iter()
                .enumerate()
                .map(|(i, d)| factor(d, strides[i]))
                .collect()
        } else {
            if auto != "NOTSET" && auto != "VALID" {
                return None;
            }
            let pads = if auto == "VALID" {
                vec![0; 2 * r]
            } else {
                strict(a, "pads", 2 * r, 0, true, true)?
            };
            x[2..]
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let off =
                        padding[i] as i128 + (w[i + 2] as i128 - 1) * dilations[i] as i128 + 1
                            - pads[i] as i128
                            - pads[i + r] as i128
                            - strides[i] as i128;
                    // Calculate once in the wider integer type; intermediate overflow
                    // of a display dimension must not destroy a representable result.
                    match d.number() {
                        Some(n) => {
                            let z = n as i128 * strides[i] as i128 + off;
                            if z >= 0 {
                                exact(z)
                            } else {
                                Dim::Unknown
                            }
                        }
                        None => offset(&factor(d, strides[i]), off),
                    }
                })
                .collect()
        }
    };
    let mut out = vec![x[0].clone(), exact(w[1] as i128 * group as i128)];
    out.extend(s);
    Some(out)
}
fn bound(v: &Value) -> Option<crate::portable_integer::Bound> {
    use crate::portable_integer::Bound;
    match v.as_str() {
        Some("-inf") => Some(Bound::Infinity(-1)),
        Some("+inf") => Some(Bound::Infinity(1)),
        _ => {
            let n = match v {
                Value::String(s) if s.len() <= 32768 => s.parse::<BigInt>().ok()?,
                _ => BigInt::from(signed(v)?),
            };
            Some(Bound::Finite(n))
        }
    }
}
fn slice(v: &Value) -> Option<Value> {
    let size = signed(&v["size"])?;
    let step = signed(&v["step"])?;
    let (a, b, n) = crate::portable_integer::slice_range(
        &BigInt::from(size),
        &bound(&v["start"])?,
        &bound(&v["end"])?,
        &BigInt::from(step),
    )?;
    Some(json!({"start":a.to_i64()?,"end":b.to_i64()?,"step":step,"length":n.to_i64()?}))
}
pub(crate) fn run(v: &Value) -> Result<Value, String> {
    let a = Dim::read(&v["left"]);
    let b = Dim::read(&v["right"]);
    Ok(match v["operation"].as_str().unwrap_or("") {
        "concat" => {
            let result = (|| {
                let shapes = v["shapes"]
                    .as_array()?
                    .iter()
                    .map(dims)
                    .collect::<Option<Vec<_>>>()?;
                let first = shapes.first()?;
                let axis = signed(&v["axis"])?;
                let axis = if axis < 0 {
                    axis + first.len() as i64
                } else {
                    axis
                };
                if axis < 0 || axis >= first.len() as i64 {
                    return None;
                }
                let axis = axis as usize;
                let mut out = first.clone();
                out[axis] = Dim::Value(0);
                for shape in &shapes {
                    if shape.len() != first.len()
                        || shape
                            .iter()
                            .enumerate()
                            .any(|(i, d)| i != axis && !can_match(d, &first[i]))
                    {
                        return None;
                    }
                    out[axis] = add(&out[axis], &shape[axis]);
                }
                Some(out)
            })();
            shape_json(result)
        }
        "add" => add(&a, &b).json(),
        "offset" => signed(&v["offset"])
            .map(|n| offset(&a, n as i128))
            .unwrap_or(Dim::Unknown)
            .json(),
        "factor" => signed(&v["factor"])
            .map(|n| factor(&a, n))
            .unwrap_or(Dim::Unknown)
            .json(),
        "multiply" => multiply(&a, &b).json(),
        "product" => dims(&v["dimensions"])
            .map(|a| product(&a))
            .unwrap_or(Dim::Unknown)
            .json(),
        "quotient" => match (dims(&v["input"]), dims(&v["target"])) {
            (Some(a), Some(b)) => quotient(&a, &b),
            _ => Dim::Unknown,
        }
        .json(),
        "broadcast" => {
            let shapes = v["shapes"]
                .as_array()
                .and_then(|a| a.iter().map(dims).collect::<Option<Vec<_>>>());
            shape_json(shapes.and_then(|a| broadcast(&a)))
        }
        "pad" => {
            let result = (|| {
                let mut input = dims(&v["input"])?;
                let pads = ints(&v["pads"])?;
                let axes = ints(&v["axes"])?;
                if pads.len() != axes.len() * 2 {
                    return None;
                }
                let mut seen = std::collections::BTreeSet::new();
                for (i, axis) in axes.iter().enumerate() {
                    if *axis < 0 || *axis >= input.len() as i64 || !seen.insert(*axis) {
                        return None;
                    }
                    let axis = *axis as usize;
                    let delta = pads[i] as i128 + pads[i + axes.len()] as i128;
                    input[axis] = if let Some(n) = input[axis].number() {
                        let n = n as i128 + delta;
                        if n < 0 {
                            return None;
                        }
                        exact(n)
                    } else if delta.abs() <= SAFE {
                        add(&input[axis], &Dim::Value(delta as i64))
                    } else {
                        Dim::Unknown
                    };
                }
                Some(input)
            })();
            shape_json(result)
        }
        "reshape" => reshape(v),
        "spatial" => shape_json(
            dims(&v["input"])
                .and_then(|d| ints(&v["kernels"]).and_then(|k| spatial(&d, &k, &v["attributes"]))),
        ),
        "conv" => shape_json(conv(v, false)),
        "conv_transpose" => shape_json(conv(v, true)),
        "matmul" => shape_json(
            dims(&v["left"]).and_then(|a| dims(&v["right"]).and_then(|b| matmul(&a, &b))),
        ),
        "flatten" => shape_json(dims(&v["input"]).and_then(|a| {
            let n = signed(&v["axis"])?;
            let n = if n < 0 { n + a.len() as i64 } else { n };
            (n >= 0 && n <= a.len() as i64)
                .then(|| vec![product(&a[..n as usize]), product(&a[n as usize..])])
        })),
        "gemm" => shape_json(dims(&v["left"]).and_then(|a| {
            dims(&v["right"]).and_then(|b| {
                if a.len() != 2 || b.len() != 2 {
                    return None;
                }
                let ta = usize::from(v["trans_a"] == true);
                let tb = usize::from(v["trans_b"] == true);
                can_match(&a[1 - ta], &b[tb]).then(|| vec![a[ta].clone(), b[1 - tb].clone()])
            })
        })),
        "slice" => slice(v).unwrap_or(Value::Null),
        _ => return Err("unknown portable shape operation".into()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dimensions_never_round_or_coerce_unknown() {
        assert_eq!(add(&Dim::Value(SAFE as i64), &Dim::Value(1)), Dim::Unknown);
        assert_eq!(Dim::read(&Value::Null), Dim::Unknown);
        assert_eq!(factor(&Dim::Value(SAFE as i64), 2), Dim::Unknown);
        assert_eq!(
            broadcast(&[vec![Dim::Value(0)], vec![Dim::Value(1)]]),
            Some(vec![Dim::Value(0)])
        );
        assert_eq!(broadcast(&[vec![Dim::Value(2)], vec![Dim::Value(3)]]), None);
    }
    #[test]
    fn reshape_exact_and_symbolic() {
        assert_eq!(
            reshape(&json!({"input":[3,4],"requested":[-1,2]}))["shape"][0]["value"],
            6
        );
        assert_eq!(
            reshape(&json!({"input":[3,4],"requested":[-1,5]}))["reason"],
            "reshape_inferred_dimension_not_integral"
        );
        assert_eq!(
            quotient(&[Dim::Symbol("N".into()), Dim::Value(12)], &[Dim::Value(3)]),
            Dim::Symbol("deepbom_expr:mul(v:4,s:N)".into())
        );
    }
    #[test]
    fn exact_spatial_does_not_round_near_safe_limit() {
        let result = spatial(
            &[Dim::Value(SAFE as i64)],
            &[1],
            &json!({"strides":[1],"pads":[1,0]}),
        )
        .unwrap();
        assert_eq!(result[0], Dim::Unknown);
        assert_eq!(
            spatial(
                &[Dim::Value(3)],
                &[2],
                &json!({"strides":[2],"pads":[0,1],"ceil_mode":1})
            )
            .unwrap(),
            vec![Dim::Value(2)]
        );
    }
    #[test]
    fn slices_keep_large_negative_bounds() {
        assert_eq!(
            slice(&json!({"size":5,"start":"+inf","end":"-9223372036854775808","step":-1}))
                .unwrap()["length"],
            5
        );
        assert_eq!(
            slice(&json!({"size":0,"start":"+inf","end":"-inf","step":-1})).unwrap()["length"],
            0
        );
    }
}
