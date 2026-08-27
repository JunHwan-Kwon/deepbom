use crate::xnnpack_rulepack_generated::XNNPACK_DOCUMENT_RULES;
use crate::{Fb, TensorInfo};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum XnnpackPrecision {
    Fp32,
    Quantized,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum XnnpackPredicate {
    BiasMandatory,
    BlockSizeGreaterThanOne,
    ConvIoQuant8BiasInt32,
    DequantizeInputPerTensorQuant8,
    FilterBiasStatic,
    FullyConnectedIoQuant8BiasInt32,
    FusedActivationSupported,
    InputCountTwo,
    InputCountTwoToFour,
    InputOneStatic,
    InputsOneTwoStatic,
    InputsOneTwoThreeStatic,
    IoFp32,
    IoQuant8,
    IoSignedInt8,
    LeakyReluScaleRatio,
    LeakyReluSlopeScaleProduct,
    MeanAxesSupported,
    OutputCountTwoToFour,
    OutputFp32,
    PaddingValuesNonnegative,
    PoolOneByOneRequiresUnitStride,
    PreluSlopeShapeSupported,
    PrimaryIoFp32,
    PrimaryIoFp32Rank4,
    PrimaryIoQuant8,
    PrimaryIoQuant8Rank4,
    QuantizeInputFp32OrQuant8,
    QuantizeOutputPerTensorQuant8,
    QuantizeRequantizationContract,
    ReshapeShapeStaticOrAbsent,
    SoftmaxBetaOne,
    StridedSliceMasksZero,
    StridedSliceStridesOne,
    TransposeConvIoFp32,
    TransposeConvIoQuant8BiasInt32,
    TransposeConvParametersStatic,
}

pub(crate) struct XnnpackDocumentRule {
    pub precision: XnnpackPrecision,
    pub op: &'static str,
    pub predicates: &'static [XnnpackPredicate],
    pub source_lines: &'static [usize],
    pub source_text_sha256: &'static [&'static str],
}

pub(crate) struct XnnpackOpContext<'a> {
    pub name: &'a str,
    pub inputs: &'a [i32],
    pub outputs: &'a [i32],
    pub tensors: &'a [TensorInfo],
    pub model_bytes: &'a [u8],
    pub fb: &'a Fb<'a>,
    pub options_table: Option<usize>,
    pub fused_activation: &'a str,
}

pub(crate) fn support_for_op(context: &XnnpackOpContext<'_>) -> (bool, String) {
    let Some(precision) = precision_for_op(context) else {
        return (
            false,
            "dtype is outside the pinned FP32/8-bit XNNPACK delegate rulepack".to_string(),
        );
    };
    let Some(rule) = XNNPACK_DOCUMENT_RULES
        .iter()
        .find(|rule| rule.precision == precision && rule.op == context.name)
    else {
        return (false, unsupported_reason(precision, context.name));
    };
    debug_assert_eq!(rule.predicates.len(), rule.source_lines.len());
    debug_assert_eq!(rule.predicates.len(), rule.source_text_sha256.len());

    for predicate in rule.predicates {
        if !evaluate_predicate(*predicate, context) {
            return (
                false,
                format!("XNNP:{}:COND:{}", precision_code(precision), predicate.id()),
            );
        }
    }
    (true, format!("XNNP:{}:OK", precision_code(precision)))
}

pub(crate) fn quantized_fully_connected_has_rank_predicate() -> bool {
    XNNPACK_DOCUMENT_RULES
        .iter()
        .find(|rule| rule.precision == XnnpackPrecision::Quantized && rule.op == "FULLY_CONNECTED")
        .is_none_or(|rule| {
            rule.predicates.iter().any(|predicate| {
                matches!(
                    predicate,
                    XnnpackPredicate::PrimaryIoFp32Rank4 | XnnpackPredicate::PrimaryIoQuant8Rank4
                )
            })
        })
}

fn precision_for_op(context: &XnnpackOpContext<'_>) -> Option<XnnpackPrecision> {
    if context.name == "DEQUANTIZE" {
        return tensor_input(context, 0)
            .filter(|tensor| is_quant8(&tensor.dtype))
            .map(|_| XnnpackPrecision::Quantized);
    }
    if context.name == "QUANTIZE" {
        return tensor_output(context, 0)
            .filter(|tensor| is_quant8(&tensor.dtype))
            .map(|_| XnnpackPrecision::Quantized);
    }
    let data_input = match context.name {
        "SPLIT" => 1,
        "TRANSPOSE_CONV" => 2,
        _ => 0,
    };
    let dtype = tensor_input(context, data_input)
        .or_else(|| tensor_output(context, 0))
        .map(|tensor| tensor.dtype.as_str())?;
    if is_quant8(dtype) {
        Some(XnnpackPrecision::Quantized)
    } else if dtype == "FLOAT32" {
        Some(XnnpackPrecision::Fp32)
    } else {
        None
    }
}

fn evaluate_predicate(predicate: XnnpackPredicate, context: &XnnpackOpContext<'_>) -> bool {
    match predicate {
        XnnpackPredicate::BiasMandatory => tensor_input(context, 2).is_some(),
        XnnpackPredicate::BlockSizeGreaterThanOne => option_i32(context, 0) > 1,
        XnnpackPredicate::ConvIoQuant8BiasInt32 => {
            tensors_have_dtype(
                [
                    tensor_input(context, 0),
                    tensor_input(context, 1),
                    tensor_output(context, 0),
                ],
                is_quant8,
            ) && tensor_input(context, 2)
                .map(|tensor| tensor.dtype == "INT32")
                .unwrap_or(false)
        }
        XnnpackPredicate::DequantizeInputPerTensorQuant8 => tensor_input(context, 0)
            .map(|tensor| is_quant8(&tensor.dtype) && has_scalar_quantization(tensor))
            .unwrap_or(false),
        XnnpackPredicate::FilterBiasStatic => {
            tensor_input(context, 1)
                .map(|tensor| tensor.constant_buffer)
                .unwrap_or(false)
                && tensor_input(context, 2)
                    .map(|tensor| tensor.constant_buffer)
                    .unwrap_or(true)
        }
        XnnpackPredicate::FullyConnectedIoQuant8BiasInt32 => {
            tensors_have_dtype(
                [
                    tensor_input(context, 0),
                    tensor_input(context, 1),
                    tensor_output(context, 0),
                ],
                is_quant8,
            ) && tensor_input(context, 2)
                .map(|tensor| tensor.dtype == "INT32")
                .unwrap_or(true)
        }
        XnnpackPredicate::FusedActivationSupported => matches!(
            context.fused_activation,
            "NONE" | "RELU" | "RELU_N1_TO_1" | "RELU6"
        ),
        XnnpackPredicate::InputCountTwo => valid_input_count(context) == 2,
        XnnpackPredicate::InputCountTwoToFour => (2..=4).contains(&valid_input_count(context)),
        XnnpackPredicate::InputOneStatic => tensor_input(context, 1)
            .map(|tensor| tensor.constant_buffer)
            .unwrap_or(false),
        XnnpackPredicate::InputsOneTwoStatic => static_inputs(context, &[1, 2]),
        XnnpackPredicate::InputsOneTwoThreeStatic => static_inputs(context, &[1, 2, 3]),
        XnnpackPredicate::IoFp32 => io_contract(context, |tensor| tensor.dtype == "FLOAT32"),
        XnnpackPredicate::IoQuant8 => io_contract(context, |tensor| is_quant8(&tensor.dtype)),
        XnnpackPredicate::IoSignedInt8 => io_contract(context, |tensor| tensor.dtype == "INT8"),
        XnnpackPredicate::LeakyReluScaleRatio => scale_ratio(context)
            .map(|ratio| (1.0 / 256.0..=128.0).contains(&ratio))
            .unwrap_or(false),
        XnnpackPredicate::LeakyReluSlopeScaleProduct => {
            let value = scale_ratio(context).map(|ratio| ratio * option_f32(context, 0) as f64);
            value
                .map(|product| {
                    (-127.99609375..=-1.0 / 256.0).contains(&product)
                        || (1.0 / 256.0..=128.0).contains(&product)
                })
                .unwrap_or(false)
        }
        XnnpackPredicate::MeanAxesSupported => constant_integer_values(context, 1)
            .map(|axes| matches!(axes.as_slice(), [1, 2] | [2, 1] | [2]))
            .unwrap_or(false),
        XnnpackPredicate::OutputCountTwoToFour => (2..=4).contains(&valid_output_count(context)),
        XnnpackPredicate::OutputFp32 => tensor_output(context, 0)
            .map(|tensor| tensor.dtype == "FLOAT32")
            .unwrap_or(false),
        XnnpackPredicate::PaddingValuesNonnegative => constant_integer_values(context, 1)
            .map(|values| !values.is_empty() && values.iter().all(|value| *value >= 0))
            .unwrap_or(false),
        XnnpackPredicate::PoolOneByOneRequiresUnitStride => {
            let stride_w = option_i32(context, 1);
            let stride_h = option_i32(context, 2);
            let filter_w = option_i32(context, 3);
            let filter_h = option_i32(context, 4);
            stride_w > 0
                && stride_h > 0
                && filter_w > 0
                && filter_h > 0
                && (filter_w != 1 || filter_h != 1 || (stride_w == 1 && stride_h == 1))
        }
        XnnpackPredicate::PreluSlopeShapeSupported => tensor_input(context, 1)
            .map(|tensor| {
                tensor.shape.len() == 1
                    || (tensor.shape.len() > 1
                        && tensor.shape[..tensor.shape.len() - 1]
                            .iter()
                            .all(|dimension| *dimension == 1))
            })
            .unwrap_or(false),
        XnnpackPredicate::PrimaryIoFp32 => primary_io(context, "FLOAT32", false),
        XnnpackPredicate::PrimaryIoFp32Rank4 => primary_io(context, "FLOAT32", true),
        XnnpackPredicate::PrimaryIoQuant8 => primary_quant8_io(context, false),
        XnnpackPredicate::PrimaryIoQuant8Rank4 => primary_quant8_io(context, true),
        XnnpackPredicate::QuantizeInputFp32OrQuant8 => tensor_input(context, 0)
            .map(|tensor| tensor.dtype == "FLOAT32" || is_quant8(&tensor.dtype))
            .unwrap_or(false),
        XnnpackPredicate::QuantizeOutputPerTensorQuant8 => tensor_output(context, 0)
            .map(|tensor| is_quant8(&tensor.dtype) && has_scalar_quantization(tensor))
            .unwrap_or(false),
        XnnpackPredicate::QuantizeRequantizationContract => {
            let Some(input) = tensor_input(context, 0) else {
                return false;
            };
            let Some(output) = tensor_output(context, 0) else {
                return false;
            };
            if input.dtype == "FLOAT32" {
                true
            } else if is_quant8(&input.dtype) && input.dtype == output.dtype {
                scalar_scale(input)
                    .zip(scalar_scale(output))
                    .map(|(input_scale, output_scale)| {
                        (2_f64.powi(-8)..=2_f64.powi(7)).contains(&(input_scale / output_scale))
                    })
                    .unwrap_or(false)
            } else {
                false
            }
        }
        XnnpackPredicate::ReshapeShapeStaticOrAbsent => {
            optional_input_absent(context, 1)
                || tensor_input(context, 1)
                    .map(|tensor| tensor.constant_buffer)
                    .unwrap_or(false)
        }
        XnnpackPredicate::SoftmaxBetaOne => {
            (option_f32(context, 0) as f64 - 1.0).abs() <= f32::EPSILON as f64
        }
        XnnpackPredicate::StridedSliceMasksZero => {
            option_i32(context, 2) == 0
                && option_i32(context, 3) == 0
                && option_i32(context, 4) == 0
        }
        XnnpackPredicate::StridedSliceStridesOne => constant_integer_values(context, 3)
            .map(|values| !values.is_empty() && values.iter().all(|value| *value == 1))
            .unwrap_or(false),
        XnnpackPredicate::TransposeConvIoFp32 => {
            tensors_have_dtype(
                [
                    tensor_input(context, 1),
                    tensor_input(context, 2),
                    tensor_output(context, 0),
                ],
                |dtype| dtype == "FLOAT32",
            ) && tensor_input(context, 3)
                .map(|tensor| tensor.dtype == "FLOAT32")
                .unwrap_or(true)
        }
        XnnpackPredicate::TransposeConvIoQuant8BiasInt32 => {
            tensors_have_dtype(
                [
                    tensor_input(context, 1),
                    tensor_input(context, 2),
                    tensor_output(context, 0),
                ],
                is_quant8,
            ) && tensor_input(context, 3)
                .map(|tensor| tensor.dtype == "INT32")
                .unwrap_or(true)
        }
        XnnpackPredicate::TransposeConvParametersStatic => {
            static_inputs(context, &[0, 1])
                && tensor_input(context, 3)
                    .map(|tensor| tensor.constant_buffer)
                    .unwrap_or(true)
        }
    }
}

fn io_tensors<'a>(context: &'a XnnpackOpContext<'_>) -> Vec<&'a TensorInfo> {
    let input_positions: Vec<usize> = match context.name {
        "MEAN" | "PAD" | "RESHAPE" | "RESIZE_BILINEAR" | "SLICE" | "STRIDED_SLICE"
        | "TRANSPOSE" => vec![0],
        "SPLIT" => vec![1],
        _ => (0..context.inputs.len()).collect(),
    };
    input_positions
        .into_iter()
        .filter_map(|position| tensor_input(context, position))
        .chain((0..context.outputs.len()).filter_map(|position| tensor_output(context, position)))
        .collect()
}

fn io_contract(context: &XnnpackOpContext<'_>, predicate: impl Fn(&TensorInfo) -> bool) -> bool {
    let required_data_input = if context.name == "SPLIT" { 1 } else { 0 };
    if tensor_input(context, required_data_input).is_none() || tensor_output(context, 0).is_none() {
        return false;
    }
    io_tensors(context).into_iter().all(predicate)
}

fn primary_io(context: &XnnpackOpContext<'_>, dtype: &str, rank4: bool) -> bool {
    let Some(input) = tensor_input(context, 0) else {
        return false;
    };
    let Some(output) = tensor_output(context, 0) else {
        return false;
    };
    input.dtype == dtype
        && output.dtype == dtype
        && (!rank4 || (input.shape.len() == 4 && output.shape.len() == 4))
}

fn primary_quant8_io(context: &XnnpackOpContext<'_>, rank4: bool) -> bool {
    let Some(input) = tensor_input(context, 0) else {
        return false;
    };
    let Some(output) = tensor_output(context, 0) else {
        return false;
    };
    is_quant8(&input.dtype)
        && is_quant8(&output.dtype)
        && (!rank4 || (input.shape.len() == 4 && output.shape.len() == 4))
}

fn tensors_have_dtype<const N: usize>(
    tensors: [Option<&TensorInfo>; N],
    predicate: impl Fn(&str) -> bool,
) -> bool {
    tensors
        .into_iter()
        .all(|tensor| tensor.map(|value| predicate(&value.dtype)).unwrap_or(false))
}

fn static_inputs(context: &XnnpackOpContext<'_>, positions: &[usize]) -> bool {
    positions.iter().all(|position| {
        tensor_input(context, *position)
            .map(|tensor| tensor.constant_buffer)
            .unwrap_or(false)
    })
}

fn tensor_input<'a>(context: &'a XnnpackOpContext<'_>, position: usize) -> Option<&'a TensorInfo> {
    tensor_for_index(context.tensors, context.inputs.get(position).copied())
}

fn tensor_output<'a>(context: &'a XnnpackOpContext<'_>, position: usize) -> Option<&'a TensorInfo> {
    tensor_for_index(context.tensors, context.outputs.get(position).copied())
}

fn tensor_for_index(tensors: &[TensorInfo], index: Option<i32>) -> Option<&TensorInfo> {
    let index = index?;
    if index < 0 {
        return None;
    }
    tensors.get(index as usize)
}

fn optional_input_absent(context: &XnnpackOpContext<'_>, position: usize) -> bool {
    context
        .inputs
        .get(position)
        .map(|index| *index < 0)
        .unwrap_or(true)
}

fn valid_input_count(context: &XnnpackOpContext<'_>) -> usize {
    context.inputs.iter().filter(|index| **index >= 0).count()
}

fn valid_output_count(context: &XnnpackOpContext<'_>) -> usize {
    context.outputs.iter().filter(|index| **index >= 0).count()
}

fn constant_integer_values(context: &XnnpackOpContext<'_>, position: usize) -> Option<Vec<i64>> {
    let tensor = tensor_input(context, position)?;
    if !tensor.constant_buffer || tensor.sparse_storage || tensor.buffer_data_length == 0 {
        return None;
    }
    let end = tensor
        .buffer_data_offset
        .checked_add(tensor.buffer_data_length)?;
    let bytes = context.model_bytes.get(tensor.buffer_data_offset..end)?;
    match tensor.dtype.as_str() {
        "INT32" if bytes.len() % 4 == 0 => Some(
            bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()) as i64)
                .collect(),
        ),
        "INT64" if bytes.len() % 8 == 0 => Some(
            bytes
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ),
        _ => None,
    }
}

fn option_i32(context: &XnnpackOpContext<'_>, field: usize) -> i32 {
    context
        .options_table
        .and_then(|table| context.fb.field_pos(table, field))
        .and_then(|position| context.fb.i32(position))
        .unwrap_or(0)
}

fn option_f32(context: &XnnpackOpContext<'_>, field: usize) -> f32 {
    context
        .options_table
        .and_then(|table| context.fb.field_pos(table, field))
        .and_then(|position| context.fb.f32(position))
        .unwrap_or(0.0)
}

fn scale_ratio(context: &XnnpackOpContext<'_>) -> Option<f64> {
    scalar_scale(tensor_input(context, 0)?)
        .zip(scalar_scale(tensor_output(context, 0)?))
        .map(|(input, output)| input / output)
}

fn scalar_scale(tensor: &TensorInfo) -> Option<f64> {
    if !has_scalar_quantization(tensor) {
        return None;
    }
    tensor
        .scale_sample
        .first()
        .copied()
        .map(|value| value as f64)
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn has_scalar_quantization(tensor: &TensorInfo) -> bool {
    tensor.quant_scales == 1 && tensor.scale_sample.len() == 1
}

fn is_quant8(dtype: &str) -> bool {
    matches!(dtype, "INT8" | "UINT8")
}

fn precision_code(precision: XnnpackPrecision) -> &'static str {
    match precision {
        XnnpackPrecision::Fp32 => "F",
        XnnpackPrecision::Quantized => "Q",
    }
}

fn unsupported_reason(precision: XnnpackPrecision, name: &str) -> String {
    match (precision, name) {
        (XnnpackPrecision::Fp32, "BATCH_MATMUL") => "XNNP:F:MM".to_string(),
        (XnnpackPrecision::Fp32, "GATHER") => "XNNP:F:GATHER".to_string(),
        (
            XnnpackPrecision::Fp32,
            "REDUCE_MAX" | "REDUCE_PROD" | "REDUCE_MIN" | "REDUCE_ANY" | "REDUCE_ALL",
        ) => "XNNP:F:REDUCE".to_string(),
        (XnnpackPrecision::Fp32, "SHAPE") => "XNNP:F:META".to_string(),
        (XnnpackPrecision::Fp32, _) => format!("XNNP:F:REJECT:{name}"),
        (XnnpackPrecision::Quantized, "AVERAGE_POOL_2D") => "XNNP:Q:POOL".to_string(),
        (XnnpackPrecision::Quantized, "SOFTMAX") => "XNNP:Q:SOFTMAX".to_string(),
        (XnnpackPrecision::Quantized, "BATCH_MATMUL") => "XNNP:Q:MM".to_string(),
        (XnnpackPrecision::Quantized, "GATHER") => "XNNP:Q:GATHER".to_string(),
        (
            XnnpackPrecision::Quantized,
            "REDUCE_MAX" | "REDUCE_PROD" | "REDUCE_MIN" | "REDUCE_ANY" | "REDUCE_ALL",
        ) => "XNNP:Q:REDUCE".to_string(),
        (XnnpackPrecision::Quantized, "SQUARED_DIFFERENCE" | "SHAPE" | "EXPAND_DIMS") => {
            "XNNP:Q:META".to_string()
        }
        (XnnpackPrecision::Quantized, _) => "XNNP:Q:REJECT".to_string(),
    }
}

impl XnnpackPredicate {
    fn id(self) -> &'static str {
        match self {
            Self::BiasMandatory => "bias_mandatory",
            Self::BlockSizeGreaterThanOne => "block_size_greater_than_one",
            Self::ConvIoQuant8BiasInt32 => "conv_io_quant8_bias_int32",
            Self::DequantizeInputPerTensorQuant8 => "dequantize_input_per_tensor_quant8",
            Self::FilterBiasStatic => "filter_bias_static",
            Self::FullyConnectedIoQuant8BiasInt32 => "fully_connected_io_quant8_bias_int32",
            Self::FusedActivationSupported => "fused_activation_supported",
            Self::InputCountTwo => "input_count_two",
            Self::InputCountTwoToFour => "input_count_two_to_four",
            Self::InputOneStatic => "input_one_static",
            Self::InputsOneTwoStatic => "inputs_one_two_static",
            Self::InputsOneTwoThreeStatic => "inputs_one_two_three_static",
            Self::IoFp32 => "io_fp32",
            Self::IoQuant8 => "io_quant8",
            Self::IoSignedInt8 => "io_signed_int8",
            Self::LeakyReluScaleRatio => "leaky_relu_scale_ratio",
            Self::LeakyReluSlopeScaleProduct => "leaky_relu_slope_scale_product",
            Self::MeanAxesSupported => "mean_axes_supported",
            Self::OutputCountTwoToFour => "output_count_two_to_four",
            Self::OutputFp32 => "output_fp32",
            Self::PaddingValuesNonnegative => "padding_values_nonnegative",
            Self::PoolOneByOneRequiresUnitStride => "pool_one_by_one_requires_unit_stride",
            Self::PreluSlopeShapeSupported => "prelu_slope_shape_supported",
            Self::PrimaryIoFp32 => "primary_io_fp32",
            Self::PrimaryIoFp32Rank4 => "primary_io_fp32_rank4",
            Self::PrimaryIoQuant8 => "primary_io_quant8",
            Self::PrimaryIoQuant8Rank4 => "primary_io_quant8_rank4",
            Self::QuantizeInputFp32OrQuant8 => "quantize_input_fp32_or_quant8",
            Self::QuantizeOutputPerTensorQuant8 => "quantize_output_per_tensor_quant8",
            Self::QuantizeRequantizationContract => "quantize_requantization_contract",
            Self::ReshapeShapeStaticOrAbsent => "reshape_shape_static_or_absent",
            Self::SoftmaxBetaOne => "softmax_beta_one",
            Self::StridedSliceMasksZero => "strided_slice_masks_zero",
            Self::StridedSliceStridesOne => "strided_slice_strides_one",
            Self::TransposeConvIoFp32 => "transpose_conv_io_fp32",
            Self::TransposeConvIoQuant8BiasInt32 => "transpose_conv_io_quant8_bias_int32",
            Self::TransposeConvParametersStatic => "transpose_conv_parameters_static",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(index: usize, dtype: &str, shape: Vec<i32>) -> TensorInfo {
        TensorInfo {
            index,
            name: format!("T{index}"),
            shape_signature: shape.clone(),
            shape,
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
            zero_point_status: "none".to_string(),
            zero_point_detail: "none".to_string(),
            scale_mode: "none".to_string(),
            scale_ratio_meaningful: false,
            quant_risk: "none".to_string(),
            buffer_hash: String::new(),
            is_variable: false,
        }
    }

    fn static_tensor(index: usize, dtype: &str, shape: Vec<i32>) -> TensorInfo {
        let mut value = tensor(index, dtype, shape);
        value.constant_buffer = true;
        value.buffer_data_length = 4;
        value
    }

    #[test]
    fn pinned_quantized_fully_connected_rule_has_no_rank_predicate() {
        assert!(!quantized_fully_connected_has_rank_predicate());
    }

    #[test]
    fn split_uses_data_input_instead_of_axis_dtype_for_precision() {
        let tensors = vec![
            static_tensor(0, "INT32", vec![1]),
            tensor(1, "INT8", vec![1, 4]),
            tensor(2, "INT8", vec![1, 2]),
            tensor(3, "INT8", vec![1, 2]),
        ];
        let bytes = [0_u8; 8];
        let fb = Fb::new_for_test(&bytes);
        let decision = support_for_op(&XnnpackOpContext {
            name: "SPLIT",
            inputs: &[0, 1],
            outputs: &[2, 3],
            tensors: &tensors,
            model_bytes: &bytes,
            fb: &fb,
            options_table: None,
            fused_activation: "NONE",
        });
        assert_eq!(decision, (true, "XNNP:Q:OK".to_string()));
    }

    #[test]
    fn dynamic_quantized_conv_filter_fails_documented_static_constraint() {
        let mut bias = static_tensor(2, "INT32", vec![4]);
        bias.buffer_data_length = 16;
        let tensors = vec![
            tensor(0, "INT8", vec![1, 4, 4, 3]),
            tensor(1, "INT8", vec![4, 3, 3, 3]),
            bias,
            tensor(3, "INT8", vec![1, 2, 2, 4]),
        ];
        let bytes = [0_u8; 32];
        let fb = Fb::new_for_test(&bytes);
        let decision = support_for_op(&XnnpackOpContext {
            name: "CONV_2D",
            inputs: &[0, 1, 2],
            outputs: &[3],
            tensors: &tensors,
            model_bytes: &bytes,
            fb: &fb,
            options_table: None,
            fused_activation: "NONE",
        });
        assert_eq!(
            decision,
            (false, "XNNP:Q:COND:filter_bias_static".to_string())
        );
    }

    #[test]
    fn mean_axes_are_decoded_from_static_tensor_bytes() {
        let mut axes = static_tensor(1, "INT32", vec![2]);
        axes.buffer_data_length = 8;
        let tensors = vec![
            tensor(0, "INT8", vec![1, 8, 8, 4]),
            axes,
            tensor(2, "INT8", vec![1, 1, 1, 4]),
        ];
        let bytes = [1_u8, 0, 0, 0, 2, 0, 0, 0];
        let fb = Fb::new_for_test(&bytes);
        let decision = support_for_op(&XnnpackOpContext {
            name: "MEAN",
            inputs: &[0, 1],
            outputs: &[2],
            tensors: &tensors,
            model_bytes: &bytes,
            fb: &fb,
            options_table: None,
            fused_activation: "NONE",
        });
        assert_eq!(decision, (true, "XNNP:Q:OK".to_string()));
    }

    #[test]
    fn transpose_conv_uses_activation_input_and_static_parameter_contract() {
        let mut output_size = static_tensor(0, "INT32", vec![4]);
        output_size.buffer_data_length = 16;
        let mut filter = static_tensor(1, "INT8", vec![4, 3, 3, 3]);
        filter.buffer_data_length = 108;
        let mut bias = static_tensor(3, "INT32", vec![4]);
        bias.buffer_data_length = 16;
        let tensors = vec![
            output_size,
            filter,
            tensor(2, "INT8", vec![1, 8, 8, 3]),
            bias,
            tensor(4, "INT8", vec![1, 16, 16, 4]),
        ];
        let bytes = vec![0_u8; 160];
        let fb = Fb::new_for_test(&bytes);
        let decision = support_for_op(&XnnpackOpContext {
            name: "TRANSPOSE_CONV",
            inputs: &[0, 1, 2, 3],
            outputs: &[4],
            tensors: &tensors,
            model_bytes: &bytes,
            fb: &fb,
            options_table: None,
            fused_activation: "NONE",
        });
        assert_eq!(decision, (true, "XNNP:Q:OK".to_string()));
    }

    #[test]
    fn dequantize_rejects_per_channel_input() {
        let mut input = tensor(0, "INT8", vec![1, 4]);
        input.quant_scales = 2;
        input.scale_sample = vec![0.1, 0.2];
        let tensors = vec![input, tensor(1, "FLOAT32", vec![1, 4])];
        let bytes = [0_u8; 8];
        let fb = Fb::new_for_test(&bytes);
        let decision = support_for_op(&XnnpackOpContext {
            name: "DEQUANTIZE",
            inputs: &[0],
            outputs: &[1],
            tensors: &tensors,
            model_bytes: &bytes,
            fb: &fb,
            options_table: None,
            fused_activation: "NONE",
        });
        assert_eq!(
            decision,
            (
                false,
                "XNNP:Q:COND:dequantize_input_per_tensor_quant8".to_string()
            )
        );
    }

    #[test]
    fn pool_fused_activation_uses_schema_field_five() {
        let mut bytes = vec![0_u8; 44];
        bytes[0..2].copy_from_slice(&16_u16.to_le_bytes());
        bytes[2..4].copy_from_slice(&28_u16.to_le_bytes());
        for (field, offset) in [4_u16, 8, 12, 16, 20, 24].into_iter().enumerate() {
            let start = 4 + field * 2;
            bytes[start..start + 2].copy_from_slice(&offset.to_le_bytes());
        }
        bytes[16..20].copy_from_slice(&16_i32.to_le_bytes());
        bytes[24..28].copy_from_slice(&1_i32.to_le_bytes());
        bytes[28..32].copy_from_slice(&1_i32.to_le_bytes());
        bytes[32..36].copy_from_slice(&2_i32.to_le_bytes());
        bytes[36..40].copy_from_slice(&2_i32.to_le_bytes());
        bytes[40] = 3;
        let fb = Fb::new_for_test(&bytes);
        assert_eq!(
            crate::fused_activation_for_op(&fb, "MAX_POOL_2D", Some(16)),
            "RELU6"
        );
    }
}
