use super::{OpInfo, TensorInfo};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

const SCHEMA: &str = "deepbom.dynamic_shape_cost_contract.v2";

#[derive(Clone, Serialize)]
pub(crate) struct DynamicShapeCostContract {
    schema: String,
    status: String,
    evidence_class: String,
    format: String,
    dynamic_tensor_count: usize,
    symbol_count: usize,
    symbols: Vec<DynamicShapeSymbol>,
    tensor_formula_count: usize,
    tensor_formulas: Vec<DynamicTensorFormula>,
    dynamic_compute_op_count: usize,
    op_formula_count: usize,
    unresolved_dynamic_compute_op_count: usize,
    unresolved_dynamic_compute_ops: Vec<DynamicOpIssue>,
    total_macs_unresolved_op_count: usize,
    total_macs_unresolved_ops: Vec<DynamicOpIssue>,
    op_formulas: Vec<DynamicOpFormula>,
    total_macs_formula: Option<DynamicIntegerFormula>,
    total_macs_formula_status: String,
    liveness: DynamicLivenessContract,
    arena_projection_status: String,
    dimension_bounds_status: String,
    method: String,
    interpretation_boundary: String,
}

#[derive(Clone, Serialize)]
struct DynamicShapeSymbol {
    symbol_id: String,
    source: String,
    declared_name: String,
    lower_bound: Option<usize>,
    upper_bound: Option<usize>,
    bounds_status: String,
    occurrences: Vec<DynamicShapeOccurrence>,
}

#[derive(Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct DynamicShapeOccurrence {
    tensor_index: usize,
    tensor_name: String,
    axis: usize,
}

#[derive(Clone, Serialize)]
struct DynamicFormulaFactor {
    symbol_id: String,
    exponent: usize,
}

#[derive(Clone, Serialize)]
struct DynamicFormulaTerm {
    coefficient_decimal: String,
    factors: Vec<DynamicFormulaFactor>,
}

#[derive(Clone, Serialize)]
struct DynamicIntegerFormula {
    status: String,
    unit: String,
    expression: String,
    terms: Vec<DynamicFormulaTerm>,
    symbol_ids: Vec<String>,
    method: String,
}

#[derive(Clone, Serialize)]
struct DynamicTensorFormula {
    tensor_index: usize,
    tensor_name: String,
    dtype: String,
    shape: Vec<i32>,
    shape_signature: Vec<i32>,
    value_kind: String,
    constant_buffer: bool,
    formula_status: String,
    element_count_formula: Option<DynamicIntegerFormula>,
    payload_bits_formula: Option<DynamicIntegerFormula>,
    payload_bytes_formula: Option<DynamicIntegerFormula>,
    payload_bytes_expression: String,
    declared_shape_projection_bytes: Option<usize>,
    declared_shape_projection_status: String,
    reason: String,
}

#[derive(Clone, Serialize)]
struct DynamicOpFormula {
    op_index: usize,
    op_name: String,
    formula_status: String,
    macs_formula: DynamicIntegerFormula,
    declared_shape_projection_macs: Option<u64>,
    declared_shape_projection_status: String,
    reason: String,
}

#[derive(Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct DynamicOpIssue {
    op_index: usize,
    op_name: String,
    reason: String,
}

fn group_op_issues(issues: Vec<DynamicOpIssue>) -> Vec<DynamicOpIssue> {
    let mut grouped = BTreeMap::<(usize, String), BTreeSet<String>>::new();
    for issue in issues {
        grouped
            .entry((issue.op_index, issue.op_name))
            .or_default()
            .insert(issue.reason);
    }
    grouped
        .into_iter()
        .map(|((op_index, op_name), reasons)| DynamicOpIssue {
            op_index,
            op_name,
            reason: reasons.into_iter().collect::<Vec<_>>().join("; "),
        })
        .collect()
}

#[derive(Clone, Serialize)]
struct DynamicLivenessCandidate {
    first_program_point: usize,
    last_program_point: usize,
    occurrence_count: usize,
    live_payload_formula: DynamicIntegerFormula,
}

#[derive(Clone, Serialize)]
struct DynamicFormulaMaximum {
    status: String,
    expression: String,
    candidates: Vec<DynamicIntegerFormula>,
}

#[derive(Clone, Serialize)]
struct DynamicLivenessContract {
    status: String,
    candidate_program_point_count: usize,
    exact_candidate_program_point_count: usize,
    unresolved_candidate_program_point_count: usize,
    distinct_exact_formula_count: usize,
    candidates: Vec<DynamicLivenessCandidate>,
    peak_selection_status: String,
    peak_live_payload_formula: Option<DynamicIntegerFormula>,
    peak_live_payload_max_formula: Option<DynamicFormulaMaximum>,
    interpretation_boundary: String,
}

#[derive(Clone, Default, PartialEq, Eq)]
struct Polynomial(BTreeMap<Vec<(String, usize)>, u128>);

impl Polynomial {
    fn constant(value: u128) -> Self {
        let mut terms = BTreeMap::new();
        if value > 0 {
            terms.insert(Vec::new(), value);
        }
        Self(terms)
    }

    fn monomial(coefficient: u128, factors: BTreeMap<String, usize>) -> Self {
        let mut terms = BTreeMap::new();
        if coefficient > 0 {
            terms.insert(
                factors
                    .into_iter()
                    .filter(|(_, exponent)| *exponent > 0)
                    .collect(),
                coefficient,
            );
        }
        Self(terms)
    }

    fn add(&self, other: &Self) -> Option<Self> {
        let mut terms = self.0.clone();
        for (factors, coefficient) in &other.0 {
            let entry = terms.entry(factors.clone()).or_default();
            *entry = entry.checked_add(*coefficient)?;
        }
        terms.retain(|_, coefficient| *coefficient > 0);
        Some(Self(terms))
    }

    fn multiply(&self, other: &Self) -> Option<Self> {
        let mut result = Self::default();
        for (left_factors, left_coefficient) in &self.0 {
            for (right_factors, right_coefficient) in &other.0 {
                let coefficient = left_coefficient.checked_mul(*right_coefficient)?;
                let mut factors = left_factors.iter().cloned().collect::<BTreeMap<_, _>>();
                for (symbol, exponent) in right_factors {
                    let entry = factors.entry(symbol.clone()).or_default();
                    *entry = entry.checked_add(*exponent)?;
                }
                result = result.add(&Self::monomial(coefficient, factors))?;
            }
        }
        Some(result)
    }

    fn scale(&self, numerator: u128, denominator: u128) -> Option<Self> {
        if denominator == 0 {
            return None;
        }
        let mut terms = BTreeMap::new();
        for (factors, coefficient) in &self.0 {
            let product = coefficient.checked_mul(numerator)?;
            if product % denominator != 0 {
                return None;
            }
            let scaled = product / denominator;
            if scaled > 0 {
                terms.insert(factors.clone(), scaled);
            }
        }
        Some(Self(terms))
    }

    fn dominates(&self, other: &Self) -> bool {
        other
            .0
            .iter()
            .all(|(factors, coefficient)| self.0.get(factors).copied().unwrap_or(0) >= *coefficient)
    }

    fn expression(&self) -> String {
        if self.0.is_empty() {
            return "0".to_string();
        }
        self.0
            .iter()
            .map(|(factors, coefficient)| {
                let factor_text = factors
                    .iter()
                    .map(|(symbol, exponent)| {
                        if *exponent == 1 {
                            symbol.clone()
                        } else {
                            format!("{}^{}", symbol, exponent)
                        }
                    })
                    .collect::<Vec<_>>();
                if factor_text.is_empty() {
                    coefficient.to_string()
                } else if *coefficient == 1 {
                    factor_text.join("*")
                } else {
                    format!("{}*{}", coefficient, factor_text.join("*"))
                }
            })
            .collect::<Vec<_>>()
            .join(" + ")
    }

    fn key(&self) -> String {
        self.0
            .iter()
            .map(|(factors, coefficient)| {
                format!(
                    "{}:{}",
                    coefficient,
                    factors
                        .iter()
                        .map(|(symbol, exponent)| format!("{}^{}", symbol, exponent))
                        .collect::<Vec<_>>()
                        .join("*")
                )
            })
            .collect::<Vec<_>>()
            .join("|")
    }

    fn serialize(&self, unit: &str, method: &str) -> DynamicIntegerFormula {
        let symbol_ids = self
            .0
            .keys()
            .flat_map(|factors| factors.iter().map(|(symbol, _)| symbol.clone()))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        DynamicIntegerFormula {
            status: "exact_symbolic_integer_polynomial".to_string(),
            unit: unit.to_string(),
            expression: self.expression(),
            terms: self
                .0
                .iter()
                .map(|(factors, coefficient)| DynamicFormulaTerm {
                    coefficient_decimal: coefficient.to_string(),
                    factors: factors
                        .iter()
                        .map(|(symbol_id, exponent)| DynamicFormulaFactor {
                            symbol_id: symbol_id.clone(),
                            exponent: *exponent,
                        })
                        .collect(),
                })
                .collect(),
            symbol_ids,
            method: method.to_string(),
        }
    }
}

#[derive(Default)]
struct SymbolRegistry {
    rows: Vec<DynamicShapeSymbol>,
    by_key: HashMap<String, usize>,
    occurrences: HashSet<(usize, usize, usize)>,
}

impl SymbolRegistry {
    fn symbol_for(&mut self, tensor: &TensorInfo, axis: usize) -> String {
        let key = format!("tflite:{}:{}", tensor.index, axis);
        let row_index = if let Some(index) = self.by_key.get(&key) {
            *index
        } else {
            let index = self.rows.len();
            self.by_key.insert(key, index);
            self.rows.push(DynamicShapeSymbol {
                symbol_id: format!("D{}", index),
                source: "tflite_shape_signature_unknown".to_string(),
                declared_name: String::new(),
                lower_bound: None,
                upper_bound: None,
                bounds_status: "not_embedded_in_artifact".to_string(),
                occurrences: Vec::new(),
            });
            index
        };
        if self.occurrences.insert((row_index, tensor.index, axis)) {
            self.rows[row_index]
                .occurrences
                .push(DynamicShapeOccurrence {
                    tensor_index: tensor.index,
                    tensor_name: tensor.name.clone(),
                    axis,
                });
        }
        self.rows[row_index].symbol_id.clone()
    }
}

fn tensor_is_dynamic(tensor: &TensorInfo) -> bool {
    tensor.shape.iter().any(|dimension| *dimension < 0)
        || tensor
            .shape_signature
            .iter()
            .any(|dimension| *dimension < 0)
}

fn dimension_polynomial(
    tensor: &TensorInfo,
    axis: usize,
    symbols: &mut SymbolRegistry,
) -> Option<Polynomial> {
    if axis >= tensor.shape.len() {
        return None;
    }
    let signature_unknown = tensor.shape_signature.len() == tensor.shape.len()
        && tensor.shape_signature.get(axis).copied().unwrap_or(0) < 0;
    let shape_dimension = tensor.shape[axis];
    if !signature_unknown && shape_dimension >= 0 {
        return Some(Polynomial::constant(shape_dimension as u128));
    }
    let symbol = symbols.symbol_for(tensor, axis);
    Some(Polynomial::monomial(1, BTreeMap::from([(symbol, 1usize)])))
}

fn dimensions_polynomial(
    tensor: &TensorInfo,
    axes: &[usize],
    symbols: &mut SymbolRegistry,
) -> Option<Polynomial> {
    let mut result = Polynomial::constant(1);
    for axis in axes {
        result = result.multiply(&dimension_polynomial(tensor, *axis, symbols)?)?;
    }
    Some(result)
}

fn tensor_element_polynomial(
    tensor: &TensorInfo,
    symbols: &mut SymbolRegistry,
) -> Option<Polynomial> {
    dimensions_polynomial(
        tensor,
        &(0..tensor.shape.len()).collect::<Vec<_>>(),
        symbols,
    )
}

fn dtype_storage_bits(dtype: &str) -> Option<u128> {
    match dtype {
        "BOOL" | "UINT8" | "INT8" => Some(8),
        "UINT16" | "INT16" | "FLOAT16" | "BFLOAT16" => Some(16),
        "UINT32" | "INT32" | "FLOAT32" => Some(32),
        "UINT64" | "INT64" | "FLOAT64" | "COMPLEX64" => Some(64),
        "COMPLEX128" => Some(128),
        "UINT4" | "INT4" => Some(4),
        "UINT2" | "INT2" => Some(2),
        _ => None,
    }
}

fn tensor_payload_polynomials(
    tensor: &TensorInfo,
    symbols: &mut SymbolRegistry,
) -> (
    Option<Polynomial>,
    Option<Polynomial>,
    Option<Polynomial>,
    String,
) {
    let Some(elements) = tensor_element_polynomial(tensor, symbols) else {
        return (None, None, None, String::new());
    };
    let Some(storage_bits) = dtype_storage_bits(&tensor.dtype) else {
        return (Some(elements), None, None, String::new());
    };
    let Some(bits) = elements.scale(storage_bits, 1) else {
        return (Some(elements), None, None, String::new());
    };
    let bytes = bits.scale(1, 8);
    let expression = bytes
        .as_ref()
        .map(Polynomial::expression)
        .unwrap_or_else(|| format!("ceil(({})/8)", bits.expression()));
    (Some(elements), Some(bits), bytes, expression)
}

fn declared_shape_projection_bytes(tensor: &TensorInfo) -> Option<usize> {
    let bits = dtype_storage_bits(&tensor.dtype)?;
    let elements = tensor.shape.iter().try_fold(1u128, |product, dimension| {
        (*dimension >= 0).then_some(())?;
        product.checked_mul(*dimension as u128)
    })?;
    let payload_bits = elements.checked_mul(bits)?;
    let bytes = payload_bits.checked_add(7)?.checked_div(8)?;
    usize::try_from(bytes).ok()
}

fn tensor_formula_rows(
    tensors: &[TensorInfo],
    symbols: &mut SymbolRegistry,
) -> Vec<DynamicTensorFormula> {
    tensors
        .iter()
        .filter(|tensor| tensor_is_dynamic(tensor))
        .map(|tensor| {
            let (elements, bits, bytes, byte_expression) =
                tensor_payload_polynomials(tensor, symbols);
            let formula_status = if bytes.is_some() {
                "exact_symbolic_integer_polynomial"
            } else if bits.is_some() {
                "exact_symbolic_ceil_expression"
            } else if elements.is_some() {
                "not_assessed_dtype_storage_width"
            } else {
                "not_assessed_shape_missing"
            };
            let projection = declared_shape_projection_bytes(tensor);
            DynamicTensorFormula {
                tensor_index: tensor.index,
                tensor_name: tensor.name.clone(),
                dtype: tensor.dtype.clone(),
                shape: tensor.shape.clone(),
                shape_signature: tensor.shape_signature.clone(),
                value_kind: "tensor".to_string(),
                constant_buffer: tensor.constant_buffer,
                formula_status: formula_status.to_string(),
                element_count_formula: elements
                    .as_ref()
                    .map(|formula| formula.serialize("elements", "product of serialized tensor dimensions with negative shape-signature axes replaced by explicit symbols")),
                payload_bits_formula: bits
                    .as_ref()
                    .map(|formula| formula.serialize("bits", "element count times the fixed dtype storage width")),
                payload_bytes_formula: bytes
                    .as_ref()
                    .map(|formula| formula.serialize("bytes", "exact division of payload bits by 8")),
                payload_bytes_expression: byte_expression,
                declared_shape_projection_bytes: projection,
                declared_shape_projection_status: if projection.is_some() {
                    "available_example_not_bound".to_string()
                } else {
                    "not_available_unknown_declared_shape".to_string()
                },
                reason: if formula_status.starts_with("exact_") {
                    "The expression is exact after every symbol is bound to a non-negative runtime dimension; the concrete TFLite shape, when present, is an example projection and not a bound.".to_string()
                } else {
                    "A complete tensor rank and fixed-width dtype are required.".to_string()
                },
            }
        })
        .collect()
}

fn tensor(tensors: &[TensorInfo], index: i32) -> Option<&TensorInfo> {
    (index >= 0).then(|| tensors.get(index as usize)).flatten()
}

fn op_has_dynamic_tensor(op: &OpInfo, tensors: &[TensorInfo]) -> bool {
    op.inputs
        .iter()
        .chain(&op.outputs)
        .filter_map(|index| tensor(tensors, *index))
        .any(tensor_is_dynamic)
}

fn dynamic_compute_op(name: &str) -> bool {
    matches!(
        name,
        "CONV_2D"
            | "DEPTHWISE_CONV_2D"
            | "FULLY_CONNECTED"
            | "BATCH_MATMUL"
            | "TRANSPOSE_CONV"
            | "CONV_3D"
            | "CONV_3D_TRANSPOSE"
            | "STABLEHLO_CONVOLUTION"
            | "STABLEHLO_DOT_GENERAL"
            | "LSTM"
            | "UNIDIRECTIONAL_SEQUENCE_LSTM"
            | "BIDIRECTIONAL_SEQUENCE_LSTM"
            | "RNN"
            | "UNIDIRECTIONAL_SEQUENCE_RNN"
            | "SVDF"
    )
}

fn multiply_all(polynomials: &[Polynomial]) -> Option<Polynomial> {
    polynomials
        .iter()
        .try_fold(Polynomial::constant(1), |result, item| {
            result.multiply(item)
        })
}

fn tflite_op_mac_polynomial(
    op: &OpInfo,
    tensors: &[TensorInfo],
    symbols: &mut SymbolRegistry,
) -> Result<(Polynomial, String), String> {
    let output = op.outputs.first().and_then(|index| tensor(tensors, *index));
    match op.name.as_str() {
        "CONV_2D" => {
            let output = output.ok_or("CONV_2D output tensor is missing")?;
            let weight = op
                .inputs
                .get(1)
                .and_then(|index| tensor(tensors, *index))
                .ok_or("CONV_2D weight tensor is missing")?;
            if output.shape.len() != 4 || weight.shape.len() != 4 {
                return Err(
                    "CONV_2D requires rank-4 NHWC output and OHWI weight tensors".to_string(),
                );
            }
            let volume = dimensions_polynomial(output, &[0, 1, 2], symbols)
                .ok_or("CONV_2D output dimensions are unresolved")?;
            let kernel = dimensions_polynomial(weight, &[0, 1, 2, 3], symbols)
                .ok_or("CONV_2D weight dimensions are unresolved")?;
            Ok((
                multiply_all(&[volume, kernel]).ok_or("CONV_2D polynomial overflow")?,
                "N*Hout*Wout*Cout*Kh*Kw*Cin from NHWC output and OHWI weight dimensions"
                    .to_string(),
            ))
        }
        "DEPTHWISE_CONV_2D" => {
            let output = output.ok_or("DEPTHWISE_CONV_2D output tensor is missing")?;
            let weight = op
                .inputs
                .get(1)
                .and_then(|index| tensor(tensors, *index))
                .ok_or("DEPTHWISE_CONV_2D weight tensor is missing")?;
            if output.shape.len() != 4 || weight.shape.len() != 4 {
                return Err(
                    "DEPTHWISE_CONV_2D requires rank-4 output and weight tensors".to_string(),
                );
            }
            let volume = dimensions_polynomial(output, &[0, 1, 2, 3], symbols)
                .ok_or("DEPTHWISE_CONV_2D output dimensions are unresolved")?;
            let kernel = dimensions_polynomial(weight, &[1, 2], symbols)
                .ok_or("DEPTHWISE_CONV_2D kernel dimensions are unresolved")?;
            Ok((
                multiply_all(&[volume, kernel]).ok_or("DEPTHWISE_CONV_2D polynomial overflow")?,
                "N*Hout*Wout*Cout*Kh*Kw from output and depthwise weight dimensions".to_string(),
            ))
        }
        "FULLY_CONNECTED" => {
            let output = output.ok_or("FULLY_CONNECTED output tensor is missing")?;
            let weight = op
                .inputs
                .get(1)
                .and_then(|index| tensor(tensors, *index))
                .ok_or("FULLY_CONNECTED weight tensor is missing")?;
            let input_depth_axis = weight
                .shape
                .len()
                .checked_sub(1)
                .ok_or("FULLY_CONNECTED weight rank is zero")?;
            let output_elements = tensor_element_polynomial(output, symbols)
                .ok_or("FULLY_CONNECTED output dimensions are unresolved")?;
            let input_depth = dimensions_polynomial(weight, &[input_depth_axis], symbols)
                .ok_or("FULLY_CONNECTED input depth is unresolved")?;
            Ok((
                multiply_all(&[output_elements, input_depth])
                    .ok_or("FULLY_CONNECTED polynomial overflow")?,
                "output_element_count*weight_input_depth, including every runtime batch dimension"
                    .to_string(),
            ))
        }
        "BATCH_MATMUL" => {
            let output = output.ok_or("BATCH_MATMUL output tensor is missing")?;
            let lhs = op
                .inputs
                .first()
                .and_then(|index| tensor(tensors, *index))
                .ok_or("BATCH_MATMUL left tensor is missing")?;
            let k_axis = lhs
                .shape
                .len()
                .checked_sub(1)
                .ok_or("BATCH_MATMUL left rank is zero")?;
            let output_elements = tensor_element_polynomial(output, symbols)
                .ok_or("BATCH_MATMUL output dimensions are unresolved")?;
            let reduction = dimensions_polynomial(lhs, &[k_axis], symbols)
                .ok_or("BATCH_MATMUL K dimension is unresolved")?;
            Ok((
                multiply_all(&[output_elements, reduction])
                    .ok_or("BATCH_MATMUL polynomial overflow")?,
                "output_element_count*K for batched matrix multiplication".to_string(),
            ))
        }
        "TRANSPOSE_CONV" => {
            let output = output.ok_or("TRANSPOSE_CONV output tensor is missing")?;
            let weight = op
                .inputs
                .get(1)
                .and_then(|index| tensor(tensors, *index))
                .ok_or("TRANSPOSE_CONV filter tensor is missing")?;
            if output.shape.len() != 4 || weight.shape.len() != 4 {
                return Err(
                    "TRANSPOSE_CONV requires rank-4 output and OHWI filter tensors".to_string(),
                );
            }
            let spatial = dimensions_polynomial(output, &[0, 1, 2], symbols)
                .ok_or("TRANSPOSE_CONV output dimensions are unresolved")?;
            let kernel = dimensions_polynomial(weight, &[0, 1, 2, 3], symbols)
                .ok_or("TRANSPOSE_CONV filter dimensions are unresolved")?;
            Ok((
                multiply_all(&[spatial, kernel]).ok_or("TRANSPOSE_CONV polynomial overflow")?,
                "N*Hout*Wout*Cout*Kh*Kw*Cin from output and filter dimensions".to_string(),
            ))
        }
        "CONV_3D" | "CONV_3D_TRANSPOSE" => {
            let output = output.ok_or("3D convolution output tensor is missing")?;
            let weight = op
                .inputs
                .get(1)
                .and_then(|index| tensor(tensors, *index))
                .ok_or("3D convolution filter tensor is missing")?;
            if output.shape.len() != 5 || weight.shape.len() != 5 {
                return Err("3D convolution requires rank-5 output and filter tensors".to_string());
            }
            let spatial = dimensions_polynomial(output, &[0, 1, 2, 3], symbols)
                .ok_or("3D convolution output dimensions are unresolved")?;
            let kernel = dimensions_polynomial(weight, &[0, 1, 2, 3, 4], symbols)
                .ok_or("3D convolution filter dimensions are unresolved")?;
            Ok((
                multiply_all(&[spatial, kernel]).ok_or("3D convolution polynomial overflow")?,
                "N*Dout*Hout*Wout*Cout*Kd*Kh*Kw*Cin from output and filter dimensions".to_string(),
            ))
        }
        _ => Err(format!(
            "{} dynamic MAC formula is not implemented",
            op.name
        )),
    }
}

fn exact_projected_macs(op: &OpInfo) -> Option<u64> {
    (op.macs.is_finite() && op.macs >= 0.0 && op.macs.fract() == 0.0 && op.macs <= u64::MAX as f64)
        .then_some(op.macs as u64)
}

fn op_formula_rows(
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    symbols: &mut SymbolRegistry,
) -> (Vec<(DynamicOpFormula, Polynomial)>, Vec<DynamicOpIssue>) {
    let mut rows = Vec::new();
    let mut unresolved = Vec::new();
    for op in ops {
        if !dynamic_compute_op(&op.name) || !op_has_dynamic_tensor(op, tensors) {
            continue;
        }
        match tflite_op_mac_polynomial(op, tensors, symbols) {
            Ok((polynomial, reason)) => rows.push((
                DynamicOpFormula {
                    op_index: op.index,
                    op_name: op.name.clone(),
                    formula_status: "exact_symbolic_integer_polynomial".to_string(),
                    macs_formula: polynomial.serialize("MACs", &reason),
                    declared_shape_projection_macs: exact_projected_macs(op),
                    declared_shape_projection_status: if exact_projected_macs(op).is_some() {
                        "available_example_not_bound".to_string()
                    } else {
                        "not_available".to_string()
                    },
                    reason,
                },
                polynomial,
            )),
            Err(reason) => unresolved.push(DynamicOpIssue {
                op_index: op.index,
                op_name: op.name.clone(),
                reason,
            }),
        }
    }
    (rows, unresolved)
}

fn total_mac_polynomial(
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    dynamic_rows: &[(DynamicOpFormula, Polynomial)],
) -> (Option<Polynomial>, Vec<DynamicOpIssue>) {
    let dynamic_by_index = dynamic_rows
        .iter()
        .map(|(row, polynomial)| (row.op_index, polynomial))
        .collect::<HashMap<_, _>>();
    let mut total = Polynomial::default();
    let mut issues = Vec::new();
    for op in ops {
        if let Some(polynomial) = dynamic_by_index.get(&op.index) {
            let Some(next) = total.add(polynomial) else {
                issues.push(DynamicOpIssue {
                    op_index: op.index,
                    op_name: op.name.clone(),
                    reason: "total MAC polynomial coefficient overflow".to_string(),
                });
                continue;
            };
            total = next;
        } else if dynamic_compute_op(&op.name) && op_has_dynamic_tensor(op, tensors) {
            issues.push(DynamicOpIssue {
                op_index: op.index,
                op_name: op.name.clone(),
                reason: "dynamic compute op has no exact symbolic formula".to_string(),
            });
        } else if op.macs > 0.0 {
            let Some(value) = exact_projected_macs(op) else {
                issues.push(DynamicOpIssue {
                    op_index: op.index,
                    op_name: op.name.clone(),
                    reason: "static MAC value is not an exact non-negative integer".to_string(),
                });
                continue;
            };
            let Some(next) = total.add(&Polynomial::constant(value as u128)) else {
                issues.push(DynamicOpIssue {
                    op_index: op.index,
                    op_name: op.name.clone(),
                    reason: "total MAC polynomial coefficient overflow".to_string(),
                });
                continue;
            };
            total = next;
        }
    }
    if issues.is_empty() {
        (Some(total), issues)
    } else {
        (None, issues)
    }
}

fn liveness_contract(
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    input_indices: &[i32],
    output_indices: &[i32],
    symbols: &mut SymbolRegistry,
) -> DynamicLivenessContract {
    let mut producer = HashMap::<usize, isize>::new();
    let mut last_use = HashMap::<usize, usize>::new();
    for index in input_indices.iter().filter(|index| **index >= 0) {
        producer.insert(*index as usize, -1);
    }
    for op in ops {
        for index in op.outputs.iter().filter(|index| **index >= 0) {
            producer.insert(*index as usize, op.index as isize);
        }
        for index in op.inputs.iter().filter(|index| **index >= 0) {
            let index = *index as usize;
            if tensors
                .get(index)
                .is_some_and(|tensor| !tensor.constant_buffer)
            {
                last_use
                    .entry(index)
                    .and_modify(|last| *last = (*last).max(op.index))
                    .or_insert(op.index);
            }
        }
    }
    for index in output_indices.iter().filter(|index| **index >= 0) {
        last_use
            .entry(*index as usize)
            .and_modify(|last| *last = (*last).max(ops.len()))
            .or_insert(ops.len());
    }
    let mut payloads = HashMap::<usize, Option<Polynomial>>::new();
    for tensor in tensors {
        let (_, _, bytes, _) = tensor_payload_polynomials(tensor, symbols);
        payloads.insert(tensor.index, bytes);
    }
    let mut groups = BTreeMap::<String, (Polynomial, usize, usize, usize)>::new();
    let mut unresolved_points = 0usize;
    for point in 0..=ops.len() {
        let mut total = Polynomial::default();
        let mut unresolved = false;
        for tensor in tensors {
            if tensor.constant_buffer {
                continue;
            }
            let Some(first) = producer.get(&tensor.index) else {
                continue;
            };
            let Some(last) = last_use.get(&tensor.index) else {
                continue;
            };
            if *first > point as isize || *last < point {
                continue;
            }
            let Some(Some(payload)) = payloads.get(&tensor.index) else {
                unresolved = true;
                continue;
            };
            let Some(next) = total.add(payload) else {
                unresolved = true;
                continue;
            };
            total = next;
        }
        if unresolved {
            unresolved_points += 1;
            continue;
        }
        let key = total.key();
        groups
            .entry(key)
            .and_modify(|(_, _, last, count)| {
                *last = point;
                *count += 1;
            })
            .or_insert((total, point, point, 1));
    }
    let candidates = groups.into_values().collect::<Vec<_>>();
    let dominant = if unresolved_points == 0 {
        candidates
            .iter()
            .find(|(candidate, _, _, _)| {
                candidates
                    .iter()
                    .all(|(other, _, _, _)| candidate.dominates(other))
            })
            .map(|(polynomial, _, _, _)| polynomial.clone())
    } else {
        None
    };
    let peak_maximum = if unresolved_points == 0 && !candidates.is_empty() {
        Some(DynamicFormulaMaximum {
            status: "exact_symbolic_max_of_integer_polynomials".to_string(),
            expression: format!(
                "max({})",
                candidates
                    .iter()
                    .map(|(polynomial, _, _, _)| polynomial.expression())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            candidates: candidates
                .iter()
                .map(|(polynomial, _, _, _)| {
                    polynomial.serialize("bytes", "exact live-set candidate")
                })
                .collect(),
        })
    } else {
        None
    };
    DynamicLivenessContract {
        status: if unresolved_points == 0 {
            "assessed"
        } else if candidates.is_empty() {
            "not_assessed"
        } else {
            "partial"
        }
        .to_string(),
        candidate_program_point_count: ops.len() + 1,
        exact_candidate_program_point_count: ops.len() + 1 - unresolved_points,
        unresolved_candidate_program_point_count: unresolved_points,
        distinct_exact_formula_count: candidates.len(),
        candidates: candidates
            .iter()
            .map(|(polynomial, first, last, count)| DynamicLivenessCandidate {
                first_program_point: *first,
                last_program_point: *last,
                occurrence_count: *count,
                live_payload_formula: polynomial.serialize(
                    "bytes",
                    "sum of exact live dense-tensor payload polynomials at this program point",
                ),
            })
            .collect(),
        peak_selection_status: if unresolved_points > 0 {
            "not_assessed_incomplete_payload_formula_coverage"
        } else if dominant.is_some() {
            "exact_by_nonnegative_coefficient_dominance"
        } else {
            "requires_runtime_dimension_binding"
        }
        .to_string(),
        peak_live_payload_formula: dominant.map(|polynomial| {
            polynomial.serialize(
                "bytes",
                "one live-set polynomial coefficient-wise dominates every candidate for all non-negative symbol assignments",
            )
        }),
        peak_live_payload_max_formula: peak_maximum,
        interpretation_boundary: "Tensor lifetimes are graph-derived. Complete candidate coverage preserves the exact peak as max(P1,...,Pn). A single polynomial is emitted only when coefficient-wise dominance proves it for every non-negative symbol assignment; otherwise numeric peak bytes require symbol binding.".to_string(),
    }
}

fn static_contract() -> DynamicShapeCostContract {
    DynamicShapeCostContract {
        schema: SCHEMA.to_string(),
        status: "not_applicable_static_shapes".to_string(),
        evidence_class: "DERIVED".to_string(),
        format: "tflite".to_string(),
        dynamic_tensor_count: 0,
        symbol_count: 0,
        symbols: Vec::new(),
        tensor_formula_count: 0,
        tensor_formulas: Vec::new(),
        dynamic_compute_op_count: 0,
        op_formula_count: 0,
        unresolved_dynamic_compute_op_count: 0,
        unresolved_dynamic_compute_ops: Vec::new(),
        total_macs_unresolved_op_count: 0,
        total_macs_unresolved_ops: Vec::new(),
        op_formulas: Vec::new(),
        total_macs_formula: None,
        total_macs_formula_status: "not_applicable_static_shapes".to_string(),
        liveness: DynamicLivenessContract {
            status: "not_applicable_static_shapes".to_string(),
            candidate_program_point_count: 0,
            exact_candidate_program_point_count: 0,
            unresolved_candidate_program_point_count: 0,
            distinct_exact_formula_count: 0,
            candidates: Vec::new(),
            peak_selection_status: "not_applicable_static_shapes".to_string(),
            peak_live_payload_formula: None,
            peak_live_payload_max_formula: None,
            interpretation_boundary:
                "No tensor carries a negative shape or shape-signature dimension.".to_string(),
        },
        arena_projection_status: "not_applicable_static_shapes".to_string(),
        dimension_bounds_status: "not_applicable_static_shapes".to_string(),
        method: "No tensor carries an unknown dimension.".to_string(),
        interpretation_boundary:
            "Static-shape costs remain in the ordinary MAC, payload, liveness, and arena sections."
                .to_string(),
    }
}

pub(crate) fn build_tflite_dynamic_shape_cost_contract(
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    input_indices: &[i32],
    output_indices: &[i32],
) -> DynamicShapeCostContract {
    let dynamic_tensor_count = tensors
        .iter()
        .filter(|tensor| tensor_is_dynamic(tensor))
        .count();
    if dynamic_tensor_count == 0 {
        return static_contract();
    }
    let mut symbols = SymbolRegistry::default();
    let tensor_formulas = tensor_formula_rows(tensors, &mut symbols);
    let (op_rows, unresolved) = op_formula_rows(ops, tensors, &mut symbols);
    let (total, total_issues) = total_mac_polynomial(ops, tensors, &op_rows);
    let unresolved = group_op_issues(unresolved);
    let total_issues = group_op_issues(total_issues);
    let liveness = liveness_contract(ops, tensors, input_indices, output_indices, &mut symbols);
    let formula_status_partial = tensor_formulas
        .iter()
        .any(|row| !row.formula_status.starts_with("exact_"));
    let status = if unresolved.is_empty()
        && total_issues.is_empty()
        && !formula_status_partial
        && liveness.status == "assessed"
    {
        "assessed"
    } else {
        "partial"
    };
    DynamicShapeCostContract {
        schema: SCHEMA.to_string(),
        status: status.to_string(),
        evidence_class: "DERIVED".to_string(),
        format: "tflite".to_string(),
        dynamic_tensor_count,
        symbol_count: symbols.rows.len(),
        symbols: symbols.rows,
        tensor_formula_count: tensor_formulas.len(),
        tensor_formulas,
        dynamic_compute_op_count: op_rows.len() + unresolved.len(),
        op_formula_count: op_rows.len(),
        unresolved_dynamic_compute_op_count: unresolved.len(),
        unresolved_dynamic_compute_ops: unresolved,
        total_macs_unresolved_op_count: total_issues.len(),
        total_macs_unresolved_ops: total_issues,
        op_formulas: op_rows.into_iter().map(|(row, _)| row).collect(),
        total_macs_formula: total.as_ref().map(|formula| {
            formula.serialize(
                "MACs",
                "sum of every static exact MAC term and every exact dynamic compute-op polynomial",
            )
        }),
        total_macs_formula_status: if total.is_some() {
            "exact_symbolic_integer_polynomial"
        } else {
            "not_assessed_incomplete_compute_formula_coverage"
        }
        .to_string(),
        liveness,
        arena_projection_status: "declared_shape_projection_available_runtime_binding_required".to_string(),
        dimension_bounds_status: "not_embedded_in_artifact".to_string(),
        method: "Negative TFLite shape-signature axes become explicit independent symbols. Tensor payload and supported convolution/matrix MACs are exact non-negative integer polynomials. Concrete shape values are retained only as unbound example projections; no numeric min/max is invented without artifact bounds.".to_string(),
        interpretation_boundary: "The formulas become numeric only after every symbol is bound to a runtime dimension satisfying the interpreter contract. ArenaPlanner ordering, delegate assignment, executed kernels, copies, latency, and task accuracy remain runtime evidence.".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(index: usize, shape: &[i32], signature: &[i32], dtype: &str) -> TensorInfo {
        TensorInfo {
            index,
            name: format!("T{}", index),
            shape: shape.to_vec(),
            shape_signature: signature.to_vec(),
            dtype: dtype.to_string(),
            buffer_index: 0,
            buffer_data_offset: 0,
            buffer_data_length: 0,
            constant_buffer: false,
            sparse_storage: false,
            sparse_encoding: None,
            quant_scales: 0,
            quant_zero_points: 0,
            quantized_dimension: 0,
            scale_sample: Vec::new(),
            zero_point_sample: Vec::new(),
            scale_min: 0.0,
            scale_max: 0.0,
            scale_ratio: 0.0,
            scale_mean: 0.0,
            scale_stddev: 0.0,
            scale_cv: 0.0,
            zero_point_min: 0,
            zero_point_max: 0,
            zero_point_offset_max: 0,
            zero_point_status: String::new(),
            zero_point_detail: String::new(),
            scale_mode: String::new(),
            scale_ratio_meaningful: false,
            quant_risk: String::new(),
            buffer_hash: String::new(),
            is_variable: false,
        }
    }

    #[test]
    fn payload_formula_keeps_zero_static_and_negative_signature_symbolic() {
        let mut symbols = SymbolRegistry::default();
        let zero = tensor(0, &[0, 3], &[0, 3], "FLOAT32");
        assert!(!tensor_is_dynamic(&zero));
        let dynamic = tensor(1, &[1, 3], &[-1, 3], "FLOAT32");
        let (_, _, bytes, expression) = tensor_payload_polynomials(&dynamic, &mut symbols);
        assert_eq!(expression, "12*D0");
        assert_eq!(bytes.unwrap().expression(), "12*D0");
        assert_eq!(symbols.rows.len(), 1);
    }
}
