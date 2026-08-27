const ONNX_SCHEMA = "deepbom.dynamic_shape_cost_contract.v2.2";
const TFLITE_SCHEMA = "deepbom.dynamic_shape_cost_contract.v2";
import { isOnnxMacBearingOperation } from "./onnx-operation-cost.js";
import { onnxDimensionExpressionDependencies, parseOnnxDimensionExpression } from "./onnx-dimension-expression.js";
import { parseOnnxEinsumEquation } from "./onnx-einsum-contract.js";
import {
  divisibleGuard,
  equalityGuard,
  integerCall,
  integerConstant,
  integerSymbol,
  nonnegativeGuard,
  serializeGuardedIntegerFormula,
} from "./guarded-integer-expression.js";

const TFLITE_MAC_OPS = new Set([
  "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "BATCH_MATMUL",
  "TRANSPOSE_CONV", "CONV_3D", "CONV_3D_TRANSPOSE", "STABLEHLO_CONVOLUTION",
  "STABLEHLO_DOT_GENERAL", "LSTM", "UNIDIRECTIONAL_SEQUENCE_LSTM",
  "BIDIRECTIONAL_SEQUENCE_LSTM", "RNN", "UNIDIRECTIONAL_SEQUENCE_RNN", "SVDF",
]);

const DTYPE_BITS = new Map([
  ["BOOL", 8], ["UINT8", 8], ["INT8", 8], ["FLOAT8E4M3FN", 8], ["FLOAT8E4M3FNUZ", 8],
  ["FLOAT8E5M2", 8], ["FLOAT8E5M2FNUZ", 8], ["FLOAT8E8M0", 8], ["UINT16", 16],
  ["INT16", 16], ["FLOAT16", 16], ["BFLOAT16", 16], ["UINT32", 32], ["INT32", 32],
  ["FLOAT32", 32], ["UINT64", 64], ["INT64", 64], ["FLOAT64", 64], ["COMPLEX64", 64],
  ["COMPLEX128", 128], ["UINT4", 4], ["INT4", 4], ["FLOAT4E2M1", 4], ["UINT2", 2],
  ["INT2", 2],
]);

function tensorIndex(tensor, fallback = -1) {
  const value = Number(tensor?.index);
  return Number.isSafeInteger(value) && value >= 0 ? value : fallback;
}

function denseTensor(tensor) {
  return tensor?.contract_status !== "invalid" && tensor?.contractStatus !== "invalid"
    && tensor?.conditional_shape_contract?.status !== "assessed_partial"
    && tensor?.conditionalShapeContract?.status !== "assessed_partial"
    && String(tensor?.value_kind || tensor?.valueKind || "tensor") === "tensor";
}

function dynamicAxes(tensor, format) {
  const shape = Array.isArray(tensor?.shape) ? tensor.shape : [];
  if (format === "onnx") return shape.flatMap((dimension, axis) => Number(dimension) < 0 ? [axis] : []);
  const signature = Array.isArray(tensor?.shape_signature) && tensor.shape_signature.length === shape.length
    ? tensor.shape_signature : [];
  return shape.flatMap((dimension, axis) => Number(dimension) < 0 || Number(signature[axis]) < 0 ? [axis] : []);
}

function shapeDeclared(tensor, format) {
  return format === "tflite" || tensor?.shape_declared === true || tensor?.shapeDeclared === true;
}

function factorKey(factors) {
  return [...factors.entries()]
    .filter(([, exponent]) => exponent > 0)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([symbol, exponent]) => `${symbol}^${exponent}`)
    .join("*");
}

function polynomial(terms = []) {
  const result = new Map();
  for (const term of terms) {
    const coefficient = BigInt(term.coefficient);
    if (coefficient === 0n) continue;
    const factors = new Map(term.factors || []);
    const key = factorKey(factors);
    result.set(key, (result.get(key) || 0n) + coefficient);
  }
  return [...result.entries()]
    .filter(([, coefficient]) => coefficient !== 0n)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, coefficient]) => ({
      coefficient,
      factors: key ? key.split("*").map((token) => {
        const [symbol, exponent] = token.split("^");
        return [symbol, Number(exponent)];
      }) : [],
    }));
}

function constant(value) {
  return polynomial([{ coefficient: BigInt(value), factors: [] }]);
}

function add(...values) {
  return polynomial(values.flat());
}

function multiply(...values) {
  let result = constant(1);
  for (const value of values) {
    if (!value) return null;
    const products = [];
    for (const left of result) {
      for (const right of value) {
        const factors = new Map(left.factors);
        for (const [symbol, exponent] of right.factors) {
          factors.set(symbol, (factors.get(symbol) || 0) + exponent);
        }
        products.push({ coefficient: left.coefficient * right.coefficient, factors });
      }
    }
    result = polynomial(products);
  }
  return result;
}

function scale(value, numerator, denominator = 1n) {
  const result = [];
  for (const term of value || []) {
    const product = term.coefficient * BigInt(numerator);
    if (product % BigInt(denominator) !== 0n) return null;
    result.push({ coefficient: product / BigInt(denominator), factors: term.factors });
  }
  return polynomial(result);
}

function termExpression(term) {
  const factors = term.factors.map(([symbol, exponent]) => exponent === 1 ? symbol : `${symbol}^${exponent}`);
  if (!factors.length) return term.coefficient.toString();
  return term.coefficient === 1n ? factors.join("*") : `${term.coefficient}*${factors.join("*")}`;
}

function expression(value) {
  return value?.length ? value.map(termExpression).join(" + ") : "0";
}

function polynomialIr(value) {
  return integerCall("add", polynomial(value || []).map((term) => integerCall("mul", [
    integerConstant(term.coefficient),
    ...term.factors.flatMap(([symbol, exponent]) => Array(exponent).fill(null).map(() => integerSymbol(symbol))),
  ])));
}

function polynomialKey(value) {
  return polynomial(value).map((term) => `${term.coefficient}:${factorKey(new Map(term.factors))}`).join("|");
}

function dominates(left, right) {
  const coefficients = new Map(polynomial(left).map((term) => [factorKey(new Map(term.factors)), term.coefficient]));
  return polynomial(right).every((term) => (coefficients.get(factorKey(new Map(term.factors))) || 0n) >= term.coefficient);
}

function formulaMatches(actual, expected, unit) {
  if (!actual || actual.status !== "exact_symbolic_integer_polynomial" || actual.unit !== unit) return false;
  if (actual.expression !== expression(expected)) return false;
  const actualTerms = [];
  try {
    for (const term of actual.terms || []) {
      if (!/^(?:0|[1-9][0-9]*)$/.test(String(term?.coefficient_decimal || ""))) return false;
      const factors = new Map();
      for (const factor of term.factors || []) {
        if (!/^D[0-9]+$/.test(String(factor?.symbol_id || ""))
          || !Number.isSafeInteger(Number(factor?.exponent)) || Number(factor.exponent) <= 0
          || factors.has(factor.symbol_id)) return false;
        factors.set(factor.symbol_id, Number(factor.exponent));
      }
      actualTerms.push({ coefficient: BigInt(term.coefficient_decimal), factors });
    }
  } catch {
    return false;
  }
  if (polynomialKey(actualTerms) !== polynomialKey(expected)) return false;
  const symbols = [...new Set(polynomial(expected).flatMap((term) => term.factors.map(([symbol]) => symbol)))].sort();
  return JSON.stringify(actual.symbol_ids || []) === JSON.stringify(symbols);
}

function guardedFormulaMatches(actual, expected, unit) {
  if (!expected) return false;
  const serialized = serializeGuardedIntegerFormula(expected.expression_ir, unit, expected.method, expected.preconditions);
  return JSON.stringify(actual) === JSON.stringify(serialized);
}

function expectedSymbolRegistry(tensors, format) {
  const rows = [];
  const byKey = new Map();
  const symbolByAxis = new Map();
  for (const [fallback, tensor] of (tensors || []).entries()) {
    if (!shapeDeclared(tensor, format)) continue;
    const index = tensorIndex(tensor, fallback);
    for (const axis of dynamicAxes(tensor, format)) {
      let key = `${format}:${index}:${axis}`;
      let source = "tflite_shape_signature_unknown";
      let declaredName = "";
      if (format === "onnx") {
        const dimension = tensor?.type_proto?.shapeDimensions?.[axis]
          || tensor?.typeProto?.shapeDimensions?.[axis] || null;
        declaredName = dimension?.kind === "symbolic" ? String(dimension.parameter || "") : "";
        const derived = declaredName.startsWith("deepbom_expr:");
        const runtimeValue = declaredName.startsWith("deepbom_runtime:");
        key = declaredName
          ? `${derived ? "onnx:derived" : runtimeValue ? "onnx:runtime" : "onnx:param"}:${declaredName}`
          : `onnx:anonymous:${index}:${axis}`;
        source = declaredName
          ? derived ? "onnx_derived_dimension_expression" : runtimeValue ? "onnx_runtime_value_dimension" : "onnx_dim_param"
          : "onnx_anonymous_dimension";
      }
      let row = byKey.get(key);
      if (!row) {
        const expressionIr = source === "onnx_derived_dimension_expression" ? parseOnnxDimensionExpression(declaredName) : null;
        row = {
          symbol_id: `D${rows.length}`,
          source,
          declared_name: declaredName,
          lower_bound: null,
          upper_bound: null,
          upper_bound_expression: null,
          bounds_status: "not_embedded_in_artifact",
          expression_ir: expressionIr,
          expression_dependencies: expressionIr ? onnxDimensionExpressionDependencies(expressionIr) : [],
          occurrences: [],
        };
        byKey.set(key, row);
        rows.push(row);
      }
      if (format === "onnx") mergeExpectedBounds(row, tensor, axis, declaredName);
      row.occurrences.push({ tensor_index: index, tensor_name: tensor?.name || `T${index}`, axis });
      symbolByAxis.set(`${index}:${axis}`, row.symbol_id);
    }
  }
  return { rows, symbolByAxis };
}

function mergeExpectedBounds(row, tensor, axis, parameter) {
  const rows = tensor?.runtime_dimension_bounds || tensor?.runtimeDimensionBounds || [];
  const bound = rows.find((item) => Number(item?.axis) === axis
    && (!item?.symbol || String(item.symbol) === parameter));
  if (!bound) return;
  const parseBound = (value) => {
    const text = String(value ?? "");
    return /^(0|[1-9][0-9]*)$/.test(text) && BigInt(text) <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(text) : null;
  };
  const values = {
    lower_bound: parseBound(bound.lower_bound_decimal),
    upper_bound: parseBound(bound.upper_bound_decimal),
    upper_bound_expression: String(bound.upper_bound_expression || "") || null,
  };
  for (const key of Object.keys(values)) {
    const value = values[key];
    if (value == null) continue;
    if (row[key] != null && row[key] !== value) {
      row.bounds_status = "artifact_contract_bounds_conflict";
      return;
    }
    row[key] = value;
  }
  row.bounds_status = row.lower_bound != null && (row.upper_bound != null || row.upper_bound_expression)
    ? "artifact_derived_lower_and_upper" : "artifact_derived_partial";
}

function symbolsMatch(actual, expected) {
  const normalize = (rows) => (rows || []).map((row) => ({
    symbol_id: row.symbol_id,
    source: row.source,
    declared_name: row.declared_name || "",
    lower_bound: row.lower_bound ?? null,
    upper_bound: row.upper_bound ?? null,
    upper_bound_expression: row.upper_bound_expression ?? null,
    bounds_status: row.bounds_status,
    expression_ir: row.expression_ir || null,
    expression_dependencies: row.expression_dependencies || [],
    occurrences: (row.occurrences || []).map((item) => ({
      tensor_index: Number(item.tensor_index), tensor_name: item.tensor_name || "", axis: Number(item.axis),
    })),
  }));
  return JSON.stringify(normalize(actual)) === JSON.stringify(normalize(expected));
}

function dimensionPolynomial(tensor, axis, format, symbols, fallback) {
  const shape = tensor?.shape || [];
  if (!shapeDeclared(tensor, format) || axis < 0 || axis >= shape.length) return null;
  const index = tensorIndex(tensor, fallback);
  const isDynamic = dynamicAxes(tensor, format).includes(axis);
  if (!isDynamic && Number.isSafeInteger(Number(shape[axis])) && Number(shape[axis]) >= 0) {
    return constant(Number(shape[axis]));
  }
  const symbol = symbols.get(`${index}:${axis}`);
  return symbol ? polynomial([{ coefficient: 1n, factors: new Map([[symbol, 1]]) }]) : null;
}

function dimensionsPolynomial(tensor, axes, format, symbols, fallback) {
  return multiply(...axes.map((axis) => dimensionPolynomial(tensor, axis, format, symbols, fallback)));
}

function tensorPolynomial(tensor, format, symbols, fallback) {
  if (!Array.isArray(tensor?.shape) || !shapeDeclared(tensor, format)) return null;
  return dimensionsPolynomial(tensor, tensor.shape.map((_, axis) => axis), format, symbols, fallback);
}

function tensorPayload(tensor, format, symbols, fallback) {
  if (!denseTensor(tensor)) return { status: "not_assessed_non_dense_value", elements: null, bits: null, bytes: null, byteExpression: "" };
  const elements = tensorPolynomial(tensor, format, symbols, fallback);
  if (!elements) return { status: "not_assessed_shape_missing", elements: null, bits: null, bytes: null, byteExpression: "" };
  const storageBits = DTYPE_BITS.get(String(tensor?.dtype || ""));
  if (!storageBits) return { status: "not_assessed_dtype_storage_width", elements, bits: null, bytes: null, byteExpression: "" };
  const bits = scale(elements, BigInt(storageBits));
  const bytes = scale(bits, 1n, 8n);
  return {
    status: bytes ? "exact_symbolic_integer_polynomial" : "exact_symbolic_ceil_expression",
    elements,
    bits,
    bytes,
    byteExpression: bytes ? expression(bytes) : `ceil((${expression(bits)})/8)`,
  };
}

function declaredProjectionBytes(tensor) {
  const bits = DTYPE_BITS.get(String(tensor?.dtype || ""));
  if (!bits || !Array.isArray(tensor?.shape) || tensor.shape.some((dimension) => !Number.isSafeInteger(Number(dimension)) || Number(dimension) < 0)) return null;
  let elements = 1n;
  for (const dimension of tensor.shape) elements *= BigInt(Number(dimension));
  const bytes = (elements * BigInt(bits) + 7n) / 8n;
  return bytes <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(bytes) : null;
}

function attributeInt(op, name, fallback) {
  const row = (op?.onnx_attributes || []).find((attribute) => attribute.name === name);
  return Number.isSafeInteger(row?.int_value) ? row.int_value : fallback;
}

function attributeInts(op, name) {
  const row = (op?.onnx_attributes || []).find((attribute) => attribute.name === name);
  return Array.isArray(row?.int_values) && row.int_values.every(Number.isSafeInteger) ? row.int_values : [];
}

function attributeString(op, name, fallback = "") {
  const row = (op?.onnx_attributes || []).find((attribute) => attribute.name === name);
  return typeof row?.string_value === "string" && row.string_value ? row.string_value : fallback;
}

function tensorAt(byIndex, index) {
  const value = Number(index);
  return Number.isSafeInteger(value) && value >= 0 ? byIndex.get(value) || null : null;
}

function opHasDynamicTensor(op, byIndex, format) {
  return [...(op?.inputs || []), ...(op?.outputs || [])]
    .map((index) => tensorAt(byIndex, index))
    .filter(Boolean)
    .some((tensor) => dynamicAxes(tensor, format).length > 0);
}

function dimensionIr(tensor, axis, symbols, fallback) {
  if (!Array.isArray(tensor?.shape) || !shapeDeclared(tensor, "onnx") || axis < 0 || axis >= tensor.shape.length) return null;
  const value = Number(tensor.shape[axis]);
  if (Number.isSafeInteger(value) && value >= 0) return integerConstant(value);
  const symbol = symbols.get(`${tensorIndex(tensor, fallback)}:${axis}`);
  return symbol ? integerSymbol(symbol) : null;
}

function exactSpatialValues(op, name, rank, fallback, allowZero = false) {
  const values = attributeInts(op, name);
  if (!values.length) return Array(rank).fill(fallback);
  return values.length === rank && values.every((value) => allowZero ? value >= 0 : value > 0) ? values : null;
}

function hasAttribute(op, name) {
  return (op?.onnx_attributes || []).some((attribute) => attribute.name === name);
}

function expectedGuardedConvTranspose(op, input, weight, output, symbols, fallbackByIndex) {
  const fallback = (tensor) => fallbackByIndex.get(tensorIndex(tensor)) ?? -1;
  const group = attributeInt(op, "group", 1);
  const autoPad = attributeString(op, "auto_pad", "NOTSET");
  const inputDeclared = shapeDeclared(input, "onnx");
  const rank = inputDeclared ? input.shape?.length || 0 : weight.shape?.length || 0;
  if (rank < 3 || weight.shape?.length !== rank || output.shape?.length !== rank || group <= 0
    || !["NOTSET", "VALID"].includes(autoPad) || hasAttribute(op, "output_shape")
    || autoPad === "VALID" && hasAttribute(op, "pads")
    || weight.shape.some((value) => !Number.isSafeInteger(Number(value)) || Number(value) <= 0)) return null;
  const spatialRank = rank - 2;
  const strides = exactSpatialValues(op, "strides", spatialRank, 1);
  const dilations = exactSpatialValues(op, "dilations", spatialRank, 1);
  const outputPadding = exactSpatialValues(op, "output_padding", spatialRank, 0, true);
  const pads = autoPad === "VALID" ? Array(spatialRank * 2).fill(0) : exactSpatialValues(op, "pads", spatialRank * 2, 0, true);
  const kernelShape = hasAttribute(op, "kernel_shape") ? attributeInts(op, "kernel_shape") : weight.shape.slice(2).map(Number);
  if (![strides, dilations, outputPadding, pads].every(Boolean) || kernelShape.length !== spatialRank
    || kernelShape.some((value, axis) => value !== Number(weight.shape[axis + 2]))
    || outputPadding.some((value, axis) => value >= strides[axis] && value >= dilations[axis])) return null;
  const weightChannels = Number(weight.shape[0]);
  const weightOutputChannels = Number(weight.shape[1]);
  if (weightChannels % group !== 0) return null;
  const outputBatch = dimensionIr(output, 0, symbols, fallback(output));
  const outputChannels = dimensionIr(output, 1, symbols, fallback(output));
  if (!outputBatch || !outputChannels) return null;
  const preconditions = [equalityGuard(outputChannels, integerConstant(weightOutputChannels * group), "ConvTranspose output channels")];
  const factors = [outputBatch, integerConstant(weightChannels), integerConstant(weightOutputChannels)];
  if (!inputDeclared) {
    if ((input.shape?.length || 0) !== 0 || pads.some((value) => value !== 0)) return null;
    for (let axis = 0; axis < spatialRank; axis += 1) {
      const outputDimension = dimensionIr(output, axis + 2, symbols, fallback(output));
      if (!outputDimension) return null;
      const effectiveKernel = (kernelShape[axis] - 1) * dilations[axis] + 1;
      const numerator = integerCall("add", [outputDimension, integerConstant(-outputPadding[axis] - effectiveKernel + strides[axis])]);
      const stride = integerConstant(strides[axis]);
      preconditions.push(nonnegativeGuard(numerator, `ConvTranspose inverse input axis ${axis}`));
      preconditions.push(divisibleGuard(numerator, stride, `ConvTranspose inverse input axis ${axis}`));
      factors.push(integerCall("mul", integerCall("exact_div", numerator, stride), integerConstant(kernelShape[axis])));
    }
    return {
      expression_ir: integerCall("mul", factors), preconditions,
      method: "Exact ONNX ConvTranspose MAC count from N*C*(M/group) and the guarded inverse output-size relation for every uncropped spatial axis. Each inferred input extent is required to be non-negative and exactly divisible by stride.",
    };
  }
  if (!pads.some((value) => value !== 0)) return null;
  const inputBatch = dimensionIr(input, 0, symbols, fallback(input));
  const inputChannels = dimensionIr(input, 1, symbols, fallback(input));
  if (!inputBatch || !inputChannels) return null;
  preconditions.push(equalityGuard(inputBatch, outputBatch, "ConvTranspose batch"));
  preconditions.push(equalityGuard(inputChannels, integerConstant(weightChannels), "ConvTranspose input channels"));
  factors[0] = inputBatch;
  for (let axis = 0; axis < spatialRank; axis += 1) {
    const inputDimension = dimensionIr(input, axis + 2, symbols, fallback(input));
    const outputDimension = dimensionIr(output, axis + 2, symbols, fallback(output));
    if (!inputDimension || !outputDimension) return null;
    const effectiveKernel = (kernelShape[axis] - 1) * dilations[axis] + 1;
    preconditions.push(equalityGuard(outputDimension, integerCall("add", [
      integerCall("mul", integerCall("sub", inputDimension, integerConstant(1)), integerConstant(strides[axis])),
      integerConstant(outputPadding[axis] + effectiveKernel - pads[axis] - pads[axis + spatialRank]),
    ]), `ConvTranspose output axis ${axis}`));
    factors.push(integerCall("conv_transpose_pairs", [inputDimension, integerConstant(kernelShape[axis]), integerConstant(strides[axis]), integerConstant(dilations[axis]), integerConstant(pads[axis]), outputDimension]));
  }
  return {
    expression_ir: integerCall("mul", factors), preconditions,
    method: "Exact contributing input/kernel pair count per ONNX ConvTranspose spatial axis, multiplied by N*C*(M/group). The emitted equality guards preserve the source output-size equation and prevent a cropped output-volume approximation.",
  };
}

function onnxMacFormula(op, byIndex, symbols, fallbackByIndex) {
  const input = tensorAt(byIndex, op?.inputs?.[0]);
  const weightInput = ["QLinearConv", "QLinearMatMul"].includes(op?.name) ? op?.inputs?.[3] : op?.inputs?.[1];
  const weight = tensorAt(byIndex, weightInput);
  const output = tensorAt(byIndex, op?.outputs?.[0]);
  if (![input, weight, output].every(denseTensor)) return null;
  const fallback = (tensor) => fallbackByIndex.get(tensorIndex(tensor)) ?? -1;
  if (["Conv", "QLinearConv", "ConvInteger"].includes(op.name)) {
    const rank = weight.shape?.length || 0, inputRank = input.shape?.length || 0, group = attributeInt(op, "group", 1);
    const dimension = (tensor, axis) => {
      const value = Number(tensor?.shape?.[axis]);
      return Number.isSafeInteger(value) && value >= 0 ? value : null;
    };
    const conflict = (left, right) => left != null && right != null && left !== right;
    const weightInputChannels = dimension(weight, 1), weightOutputChannels = dimension(weight, 0);
    const inputChannels = dimension(input, 1), outputChannels = dimension(output, 1);
    const inputBatch = dimension(input, 0), outputBatch = dimension(output, 0);
    if (rank < 3 || output.shape?.length !== rank || shapeDeclared(input, "onnx") && inputRank !== rank
      || !Number.isSafeInteger(group) || group <= 0
      || weightInputChannels == null || weightInputChannels <= 0 || weightOutputChannels == null || weightOutputChannels <= 0
      || weight.shape.slice(2).some((value) => !Number.isSafeInteger(Number(value)) || Number(value) <= 0)
      || weightOutputChannels % group !== 0
      || conflict(inputChannels, weightInputChannels * group)
      || conflict(outputChannels, weightOutputChannels)
      || conflict(inputBatch, outputBatch)) return null;
    return multiply(
      dimensionsPolynomial(output, output.shape.map((_, axis) => axis).filter((axis) => axis !== 1), "onnx", symbols, fallback(output)),
      tensorPolynomial(weight, "onnx", symbols, fallback(weight)),
    );
  }
  if (op.name === "ConvTranspose") {
    const rank = input.shape?.length || 0, group = attributeInt(op, "group", 1), autoPad = attributeString(op, "auto_pad", "NOTSET");
    const pads = attributeInts(op, "pads"), kernelShape = attributeInts(op, "kernel_shape"), outputShape = attributeInts(op, "output_shape");
    const channels = Number(input.shape?.[1]), weightChannels = Number(weight.shape?.[0]), outputChannels = Number(output.shape?.[1]);
    if (rank < 3 || weight.shape?.length !== rank || output.shape?.length !== rank || group <= 0 || channels <= 0 || channels !== weightChannels
      || channels % group !== 0 || outputChannels !== Number(weight.shape?.[1]) * group || outputShape.length
      || !["NOTSET", "VALID"].includes(autoPad) || pads.some((value) => value !== 0)
      || kernelShape.length && (kernelShape.length !== rank - 2 || kernelShape.some((value, axis) => value !== Number(weight.shape?.[axis + 2])))) return null;
    return multiply(
      tensorPolynomial(input, "onnx", symbols, fallback(input)),
      dimensionsPolynomial(weight, weight.shape.slice(1).map((_, axis) => axis + 1), "onnx", symbols, fallback(weight)),
    );
  }
  if (op.name === "Attention") {
    const valueTensor = tensorAt(byIndex, op?.inputs?.[2]);
    const pastKey = tensorAt(byIndex, op?.inputs?.[4]);
    const pastValue = tensorAt(byIndex, op?.inputs?.[5]);
    const rank = input.shape?.length || 0;
    if (![input, weight, valueTensor, output].every(denseTensor)
      || ![3, 4].includes(rank) || weight.shape?.length !== rank || valueTensor.shape?.length !== rank || output.shape?.length !== rank) return null;
    let base, sequence, qHeads, qkHeadSize, valueHeadSize;
    if (rank === 4) {
      qHeads = Number(input.shape[1]);
      const kvHeads = Number(weight.shape[1]);
      qkHeadSize = Number(input.shape[3]);
      valueHeadSize = Number(valueTensor.shape[3]);
      if (![qHeads, kvHeads, qkHeadSize, valueHeadSize].every((value) => Number.isSafeInteger(value) && value > 0)
        || qHeads < kvHeads || qHeads % kvHeads !== 0 || Number(weight.shape[3]) !== qkHeadSize) return null;
      base = dimensionsPolynomial(input, [0, 2], "onnx", symbols, fallback(input));
      sequence = dimensionsPolynomial(weight, [2], "onnx", symbols, fallback(weight));
    } else {
      qHeads = attributeInt(op, "q_num_heads", 0);
      const kvHeads = attributeInt(op, "kv_num_heads", 0);
      const qHidden = Number(input.shape[2]), kHidden = Number(weight.shape[2]), vHidden = Number(valueTensor.shape[2]);
      if (![qHeads, kvHeads, qHidden, kHidden, vHidden].every((value) => Number.isSafeInteger(value) && value > 0)
        || qHeads < kvHeads || qHeads % kvHeads !== 0 || qHidden % qHeads !== 0 || kHidden % kvHeads !== 0
        || vHidden % kvHeads !== 0 || qHidden / qHeads !== kHidden / kvHeads) return null;
      qkHeadSize = qHidden / qHeads;
      valueHeadSize = vHidden / kvHeads;
      base = dimensionsPolynomial(input, [0, 1], "onnx", symbols, fallback(input));
      sequence = dimensionsPolynomial(weight, [1], "onnx", symbols, fallback(weight));
    }
    const terms = [scale(multiply(base, sequence), qHeads * (qkHeadSize + valueHeadSize))];
    if (pastKey || pastValue) {
      if (!(pastKey && pastValue && denseTensor(pastKey) && denseTensor(pastValue))
        || pastKey.shape?.length !== 4 || pastValue.shape?.length !== 4) return null;
      terms.push(scale(multiply(base, dimensionsPolynomial(pastKey, [2], "onnx", symbols, fallback(pastKey))), qHeads * (qkHeadSize + valueHeadSize)));
    }
    return add(...terms);
  }
  if (op.name === "DeformConv") {
    const offset = tensorAt(byIndex, op?.inputs?.[2]);
    const rank = input.shape?.length || 0;
    const group = attributeInt(op, "group", 1), offsetGroup = attributeInt(op, "offset_group", 1);
    if (![input, weight, offset, output].every(denseTensor) || rank < 3 || weight.shape?.length !== rank
      || offset.shape?.length !== rank || output.shape?.length !== rank || group <= 0 || offsetGroup <= 0
      || weight.shape.slice(0, 2).some((value) => !Number.isSafeInteger(Number(value)) || Number(value) <= 0)
      || weight.shape.slice(2).some((value) => !Number.isSafeInteger(Number(value)) || Number(value) <= 0)
      || Number(weight.shape[0]) % group !== 0) return null;
    return multiply(
      dimensionsPolynomial(output, output.shape.map((_, axis) => axis).filter((axis) => axis !== 1), "onnx", symbols, fallback(output)),
      tensorPolynomial(weight, "onnx", symbols, fallback(weight)),
    );
  }
  if (op.name === "Einsum") {
    const inputs = (op.inputs || []).map((index) => tensorAt(byIndex, index)).filter(Boolean);
    if (!inputs.length || inputs.some((tensor) => !denseTensor(tensor) || !shapeDeclared(tensor, "onnx"))) return null;
    const parsed = parseOnnxEinsumEquation(attributeString(op, "equation", ""), inputs.map((tensor) => tensor.shape.length));
    if (parsed.status !== "assessed") return null;
    if (inputs.length === 1) return [];
    if (inputs.length !== 2) return null;
    const domains = new Map();
    for (let inputIndex = 0; inputIndex < inputs.length; inputIndex += 1) {
      const operand = inputs[inputIndex];
      for (let axis = 0; axis < parsed.operands[inputIndex].length; axis += 1) {
        const label = parsed.operands[inputIndex][axis];
        const candidate = dimensionsPolynomial(operand, [axis], "onnx", symbols, fallback(operand));
        const previous = domains.get(label);
        if (!previous) {
          domains.set(label, candidate);
          continue;
        }
        const previousStatic = previous.length === 1 && previous[0].factors.length === 0 ? previous[0].coefficient : null;
        const candidateStatic = candidate.length === 1 && candidate[0].factors.length === 0 ? candidate[0].coefficient : null;
        if (label.startsWith("@ellipsis:")) {
          if (previousStatic === 1n) domains.set(label, candidate);
          else if (candidateStatic === 1n || polynomialKey(previous) === polynomialKey(candidate)) continue;
          else return null;
        } else if (previousStatic != null && candidateStatic != null && previousStatic !== candidateStatic) return null;
      }
    }
    return multiply(...parsed.all_labels.map((label) => domains.get(label)));
  }
  if (["RNN", "GRU", "LSTM"].includes(op.name)) {
    const recurrent = tensorAt(byIndex, op?.inputs?.[2]);
    const gates = op.name === "LSTM" ? 4 : op.name === "GRU" ? 3 : 1;
    const hidden = attributeInt(op, "hidden_size", 0), layout = attributeInt(op, "layout", 0), direction = attributeString(op, "direction", "forward");
    const directions = direction === "bidirectional" ? 2 : ["forward", "reverse"].includes(direction) ? 1 : 0;
    if (![input, weight, recurrent].every(denseTensor) || input.shape?.length !== 3 || weight.shape?.length !== 3 || recurrent.shape?.length !== 3
      || ![0, 1].includes(layout) || !directions || hidden <= 0 || weight.shape[0] !== directions || recurrent.shape[0] !== directions
      || weight.shape[1] !== gates * hidden || recurrent.shape[1] !== gates * hidden || weight.shape[2] !== input.shape[2] || recurrent.shape[2] !== hidden) return null;
    return scale(
      dimensionsPolynomial(input, [layout ? 1 : 0, layout ? 0 : 1], "onnx", symbols, fallback(input)),
      directions * gates * hidden * (input.shape[2] + hidden),
    );
  }
  if (op.name === "Gemm") {
    if (input.shape?.length !== 2 || weight.shape?.length !== 2) return null;
    const transAValue = attributeInt(op, "transA", 0);
    const transBValue = attributeInt(op, "transB", 0);
    if (![0, 1].includes(transAValue) || ![0, 1].includes(transBValue)) return null;
    const transA = transAValue === 1;
    const transB = transBValue === 1;
    return multiply(
      dimensionsPolynomial(input, [transA ? 1 : 0], "onnx", symbols, fallback(input)),
      dimensionsPolynomial(weight, [transB ? 0 : 1], "onnx", symbols, fallback(weight)),
      dimensionsPolynomial(input, [transA ? 0 : 1], "onnx", symbols, fallback(input)),
    );
  }
  if (["MatMul", "QLinearMatMul", "MatMulInteger"].includes(op.name)) {
    if ((input.shape?.length || 0) < 1 || (weight.shape?.length || 0) < 1 || !Array.isArray(output.shape)) return null;
    return multiply(
      tensorPolynomial(output, "onnx", symbols, fallback(output)),
      dimensionsPolynomial(input, [input.shape.length - 1], "onnx", symbols, fallback(input)),
    );
  }
  return null;
}

function tfliteMacFormula(op, byIndex, symbols, fallbackByIndex) {
  const output = tensorAt(byIndex, op?.outputs?.[0]);
  const weight = tensorAt(byIndex, op?.inputs?.[1]);
  const fallback = (tensor) => fallbackByIndex.get(tensorIndex(tensor)) ?? -1;
  if (op.name === "CONV_2D") {
    if (output?.shape?.length !== 4 || weight?.shape?.length !== 4) return null;
    return multiply(
      dimensionsPolynomial(output, [0, 1, 2], "tflite", symbols, fallback(output)),
      dimensionsPolynomial(weight, [0, 1, 2, 3], "tflite", symbols, fallback(weight)),
    );
  }
  if (op.name === "DEPTHWISE_CONV_2D") {
    if (output?.shape?.length !== 4 || weight?.shape?.length !== 4) return null;
    return multiply(
      dimensionsPolynomial(output, [0, 1, 2, 3], "tflite", symbols, fallback(output)),
      dimensionsPolynomial(weight, [1, 2], "tflite", symbols, fallback(weight)),
    );
  }
  if (op.name === "FULLY_CONNECTED") {
    if (!output || !weight?.shape?.length) return null;
    return multiply(
      tensorPolynomial(output, "tflite", symbols, fallback(output)),
      dimensionsPolynomial(weight, [weight.shape.length - 1], "tflite", symbols, fallback(weight)),
    );
  }
  if (op.name === "BATCH_MATMUL") {
    const input = tensorAt(byIndex, op?.inputs?.[0]);
    if (!output || !input?.shape?.length) return null;
    return multiply(
      tensorPolynomial(output, "tflite", symbols, fallback(output)),
      dimensionsPolynomial(input, [input.shape.length - 1], "tflite", symbols, fallback(input)),
    );
  }
  if (op.name === "TRANSPOSE_CONV") {
    if (output?.shape?.length !== 4 || weight?.shape?.length !== 4) return null;
    return multiply(
      dimensionsPolynomial(output, [0, 1, 2], "tflite", symbols, fallback(output)),
      dimensionsPolynomial(weight, [0, 1, 2, 3], "tflite", symbols, fallback(weight)),
    );
  }
  if (["CONV_3D", "CONV_3D_TRANSPOSE"].includes(op.name)) {
    if (output?.shape?.length !== 5 || weight?.shape?.length !== 5) return null;
    return multiply(
      dimensionsPolynomial(output, [0, 1, 2, 3], "tflite", symbols, fallback(output)),
      dimensionsPolynomial(weight, [0, 1, 2, 3, 4], "tflite", symbols, fallback(weight)),
    );
  }
  return null;
}

function groupIssues(issues) {
  const grouped = new Map();
  for (const issue of issues) {
    const key = `${issue.op_index}:${issue.op_name}`;
    if (!grouped.has(key)) grouped.set(key, { op_index: issue.op_index, op_name: issue.op_name });
  }
  return [...grouped.values()].sort((left, right) => left.op_index - right.op_index || left.op_name.localeCompare(right.op_name));
}

function expectedOpEvidence(analysis, format, symbols) {
  const tensors = analysis?.tensors || [];
  const byIndex = new Map(tensors.map((tensor, fallback) => [tensorIndex(tensor, fallback), tensor]));
  const fallbackByIndex = new Map(tensors.map((tensor, fallback) => [tensorIndex(tensor, fallback), fallback]));
  const formulas = [];
  const unresolved = [];
  for (const op of analysis?.ops || []) {
    const eligible = format === "onnx"
      ? isOnnxMacBearingOperation(op?.name, op?.standard_domain !== false)
      : TFLITE_MAC_OPS.has(op?.name);
    if (!eligible || !opHasDynamicTensor(op, byIndex, format)) continue;
    const value = format === "onnx"
      ? onnxMacFormula(op, byIndex, symbols, fallbackByIndex)
      : tfliteMacFormula(op, byIndex, symbols, fallbackByIndex);
    const guarded = format === "onnx" && !value && op.name === "ConvTranspose"
      ? expectedGuardedConvTranspose(
        op,
        tensorAt(byIndex, op?.inputs?.[0]),
        tensorAt(byIndex, op?.inputs?.[1]),
        tensorAt(byIndex, op?.outputs?.[0]),
        symbols,
        fallbackByIndex,
      )
      : null;
    if (value || guarded) formulas.push({ op_index: Number(op.index), op_name: op.name, polynomial: value, guarded, op });
    else unresolved.push({ op_index: Number(op.index), op_name: op.name || "UNKNOWN" });
  }
  return { formulas, unresolved: groupIssues(unresolved), byIndex, symbols, fallbackByIndex };
}

function expectedTotalMacs(analysis, format, opEvidence) {
  const dynamicByIndex = new Map(opEvidence.formulas.map((row) => [row.op_index, row]));
  const issues = [];
  const terms = [];
  const expressionTerms = [];
  const preconditions = [];
  let guarded = false;
  for (const op of analysis?.ops || []) {
    const eligible = format === "onnx"
      ? isOnnxMacBearingOperation(op?.name, op?.standard_domain !== false)
      : TFLITE_MAC_OPS.has(op?.name);
    if (!eligible) continue;
    if (dynamicByIndex.has(Number(op.index))) {
      const row = dynamicByIndex.get(Number(op.index));
      if (row.polynomial) {
        terms.push(row.polynomial);
        expressionTerms.push(polynomialIr(row.polynomial));
      } else {
        guarded = true;
        expressionTerms.push(row.guarded.expression_ir);
        preconditions.push(...row.guarded.preconditions);
      }
      continue;
    }
    if (opHasDynamicTensor(op, opEvidence.byIndex, format)) {
      issues.push({ op_index: Number(op.index), op_name: op.name || "UNKNOWN" });
      continue;
    }
    if (format === "onnx") {
      const exactMacs = String(op.macs_decimal ?? (Number.isSafeInteger(Number(op.macs)) ? op.macs : ""));
      if (op.macs_status !== "assessed" || !/^(?:0|[1-9]\d*)$/.test(exactMacs)) {
        const conditional = op.name === "ConvTranspose" ? expectedGuardedConvTranspose(
          op,
          tensorAt(opEvidence.byIndex, op?.inputs?.[0]),
          tensorAt(opEvidence.byIndex, op?.inputs?.[1]),
          tensorAt(opEvidence.byIndex, op?.outputs?.[0]),
          opEvidence.symbols,
          opEvidence.fallbackByIndex,
        ) : null;
        if (conditional) {
          guarded = true;
          expressionTerms.push(conditional.expression_ir);
          preconditions.push(...conditional.preconditions);
        } else issues.push({ op_index: Number(op.index), op_name: op.name || "UNKNOWN" });
      } else {
        const value = constant(exactMacs);
        terms.push(value);
        expressionTerms.push(integerConstant(exactMacs));
      }
    } else if (Number(op.macs) > 0) {
      const macs = Number(op.macs);
      if (!Number.isSafeInteger(macs)) issues.push({ op_index: Number(op.index), op_name: op.name || "UNKNOWN" });
      else terms.push(constant(macs));
    }
  }
  const groupedIssues = groupIssues(issues);
  if (groupedIssues.length) return { polynomial: null, guarded: null, issues: groupedIssues };
  if (guarded) return {
    polynomial: null,
    guarded: {
      expression_ir: integerCall("add", expressionTerms), preconditions,
      method: "Sum of every exact static MAC count, symbolic polynomial, and guarded integer expression after all source-defined preconditions are satisfied.",
    },
    issues: groupedIssues,
  };
  return { polynomial: add(...terms), guarded: null, issues: groupedIssues };
}

function expectedLiveness(analysis, format, symbols) {
  const tensors = analysis?.tensors || [];
  const ops = analysis?.ops || [];
  const byIndex = new Map(tensors.map((tensor, fallback) => [tensorIndex(tensor, fallback), tensor]));
  const fallbackByIndex = new Map(tensors.map((tensor, fallback) => [tensorIndex(tensor, fallback), fallback]));
  const inputIndices = format === "onnx"
    ? tensors.filter((tensor) => tensor?.role === "input").map(tensorIndex)
    : analysis?.input_tensor_indices || [];
  const outputIndices = format === "onnx"
    ? tensors.filter((tensor) => tensor?.role === "output").map(tensorIndex)
    : analysis?.output_tensor_indices || [];
  const producer = new Map(inputIndices.filter((index) => Number(index) >= 0).map((index) => [Number(index), -1]));
  const lastUse = new Map();
  for (const op of ops) {
    for (const index of op.outputs || []) if (Number(index) >= 0) producer.set(Number(index), Number(op.index));
    for (const index of op.inputs || []) {
      const tensor = tensorAt(byIndex, index);
      if (tensor && tensor.constant_buffer !== true) lastUse.set(Number(index), Math.max(lastUse.get(Number(index)) ?? -1, Number(op.index)));
    }
  }
  for (const index of outputIndices) {
    if (Number(index) >= 0) lastUse.set(Number(index), Math.max(lastUse.get(Number(index)) ?? -1, ops.length));
  }
  const payloads = new Map(tensors.map((tensor, fallback) => [
    tensorIndex(tensor, fallback), tensorPayload(tensor, format, symbols, fallback).bytes,
  ]));
  const grouped = new Map();
  let unresolvedPoints = 0;
  for (let point = 0; point <= ops.length; point += 1) {
    const values = [];
    let unresolved = false;
    for (const [fallback, tensor] of tensors.entries()) {
      const index = tensorIndex(tensor, fallback);
      if (tensor.constant_buffer === true || !producer.has(index) || !lastUse.has(index)) continue;
      if (producer.get(index) > point || lastUse.get(index) < point) continue;
      const payload = payloads.get(index);
      if (!payload) unresolved = true;
      else values.push(payload);
    }
    if (unresolved) {
      unresolvedPoints += 1;
      continue;
    }
    const value = add(...values);
    const key = polynomialKey(value);
    const row = grouped.get(key) || { polynomial: value, first: point, last: point, count: 0 };
    row.last = point;
    row.count += 1;
    grouped.set(key, row);
  }
  const candidates = [...grouped.values()].sort((left, right) => {
    const leftKey = polynomialKey(left.polynomial);
    const rightKey = polynomialKey(right.polynomial);
    return leftKey < rightKey ? -1 : leftKey > rightKey ? 1 : 0;
  });
  const dominant = unresolvedPoints ? null
    : candidates.find((candidate) => candidates.every((other) => dominates(candidate.polynomial, other.polynomial))) || null;
  return { candidates, unresolvedPoints, dominant, programPoints: ops.length + 1 };
}

function tensorRowsMatch(contract, analysis, format, symbols) {
  const tensors = analysis?.tensors || [];
  const expectedTensors = tensors.filter((tensor) => dynamicAxes(tensor, format).length > 0);
  const rows = contract?.tensor_formulas || [];
  if (rows.length !== expectedTensors.length || Number(contract?.tensor_formula_count) !== rows.length) return false;
  return expectedTensors.every((tensor, rowIndex) => {
    const fallback = tensors.indexOf(tensor);
    const row = rows[rowIndex];
    const payload = tensorPayload(tensor, format, symbols, fallback);
    const projection = format === "tflite" ? declaredProjectionBytes(tensor) : null;
    return Number(row?.tensor_index) === tensorIndex(tensor, fallback)
      && row?.tensor_name === (tensor?.name || `T${tensorIndex(tensor, fallback)}`)
      && row?.dtype === (tensor?.dtype || "UNKNOWN")
      && JSON.stringify(row?.shape || []) === JSON.stringify(tensor?.shape || [])
      && row?.formula_status === payload.status
      && (payload.elements ? formulaMatches(row.element_count_formula, payload.elements, "elements") : row.element_count_formula == null)
      && (payload.bits ? formulaMatches(row.payload_bits_formula, payload.bits, "bits") : row.payload_bits_formula == null)
      && (payload.bytes ? formulaMatches(row.payload_bytes_formula, payload.bytes, "bytes") : row.payload_bytes_formula == null)
      && row?.payload_bytes_expression === payload.byteExpression
      && row?.declared_shape_projection_bytes === projection;
  });
}

function opRowsMatch(contract, opEvidence) {
  const rows = contract?.op_formulas || [];
  if (rows.length !== opEvidence.formulas.length || Number(contract?.op_formula_count) !== rows.length) return false;
  if (Number(contract?.dynamic_compute_op_count) !== rows.length + opEvidence.unresolved.length
    || Number(contract?.unresolved_dynamic_compute_op_count) !== opEvidence.unresolved.length) return false;
  const unresolvedIndices = (contract?.unresolved_dynamic_compute_ops || []).map((row) => Number(row.op_index));
  if (JSON.stringify(unresolvedIndices) !== JSON.stringify(opEvidence.unresolved.map((row) => row.op_index))) return false;
  return opEvidence.formulas.every((expected, index) => {
    const row = rows[index];
    return Number(row?.op_index) === expected.op_index
      && row?.op_name === expected.op_name
      && (expected.polynomial
        ? row?.formula_status === "exact_symbolic_integer_polynomial"
          && formulaMatches(row?.macs_formula, expected.polynomial, "MACs")
        : row?.formula_status === "exact_guarded_integer_expression"
          && guardedFormulaMatches(row?.macs_formula, expected.guarded, "MACs"));
  });
}

function totalMatches(contract, expected) {
  if (Number(contract?.total_macs_unresolved_op_count) !== expected.issues.length) return false;
  const actualIndices = (contract?.total_macs_unresolved_ops || []).map((row) => Number(row.op_index));
  if (JSON.stringify(actualIndices) !== JSON.stringify(expected.issues.map((row) => row.op_index))) return false;
  if (expected.guarded) {
    return contract?.total_macs_formula_status === "exact_guarded_integer_expression"
      && guardedFormulaMatches(contract?.total_macs_formula, expected.guarded, "MACs");
  }
  if (!expected.polynomial) {
    return contract?.total_macs_formula == null
      && contract?.total_macs_formula_status === "not_assessed_incomplete_compute_formula_coverage";
  }
  return contract?.total_macs_formula_status === "exact_symbolic_integer_polynomial"
    && formulaMatches(contract?.total_macs_formula, expected.polynomial, "MACs");
}

function livenessMatches(actual, expected) {
  const expectedStatus = expected.unresolvedPoints ? expected.candidates.length ? "partial" : "not_assessed" : "assessed";
  if (actual?.status !== expectedStatus
    || Number(actual?.candidate_program_point_count) !== expected.programPoints
    || Number(actual?.exact_candidate_program_point_count) !== expected.programPoints - expected.unresolvedPoints
    || Number(actual?.unresolved_candidate_program_point_count) !== expected.unresolvedPoints
    || Number(actual?.distinct_exact_formula_count) !== expected.candidates.length
    || (actual?.candidates || []).length !== expected.candidates.length) return false;
  if (!expected.candidates.every((candidate, index) => {
    const row = actual.candidates[index];
    return Number(row?.first_program_point) === candidate.first
      && Number(row?.last_program_point) === candidate.last
      && Number(row?.occurrence_count) === candidate.count
      && formulaMatches(row?.live_payload_formula, candidate.polynomial, "bytes");
  })) return false;
  const peakStatus = expected.unresolvedPoints
    ? "not_assessed_incomplete_payload_formula_coverage"
    : expected.dominant ? "exact_by_nonnegative_coefficient_dominance" : "requires_runtime_dimension_binding";
  const maxFormula = actual?.peak_live_payload_max_formula;
  const maxValid = expected.unresolvedPoints || !expected.candidates.length
    ? maxFormula == null
    : maxFormula?.status === "exact_symbolic_max_of_integer_polynomials"
      && maxFormula.candidates?.length === expected.candidates.length
      && expected.candidates.every((candidate, index) => formulaMatches(maxFormula.candidates[index], candidate.polynomial, "bytes"));
  return maxValid && actual?.peak_selection_status === peakStatus
    && (expected.dominant
      ? formulaMatches(actual?.peak_live_payload_formula, expected.dominant.polynomial, "bytes")
      : actual?.peak_live_payload_formula == null);
}

function mlBomProperties(document) {
  return [...(document?.metadata?.component?.properties || []), ...(document?.properties || [])];
}

function propertyValue(document, name) {
  return mlBomProperties(document).find((item) => item?.name === name)?.value;
}

function compactEvidencePointerValid(document) {
  return propertyValue(document, "deepbom:compatibility:detailLocation")
    === "engineering_evidence.json#/evidence/static_analysis";
}

function findingRows(register) {
  return Array.isArray(register) ? register : register?.findings || [];
}

export function verifyDynamicShapeCostEvidence({ analysis, engineeringReport, mlBomDocument, findingsRegister } = {}) {
  const contract = analysis?.dynamic_shape_cost_contract;
  const format = String(analysis?.format || "tflite").toLowerCase();
  const expectedSchema = format === "onnx" ? ONNX_SCHEMA : TFLITE_SCHEMA;
  const dynamicTensors = (analysis?.tensors || []).filter((tensor) => dynamicAxes(tensor, format).length > 0);
  if (!contract) {
    return {
      contract_present: false,
      static_non_applicability_valid: false,
      symbols_valid: false,
      tensor_formulas_valid: false,
      op_formulas_valid: false,
      total_macs_valid: false,
      liveness_valid: false,
      report_valid: false,
      mlbom_valid: false,
      finding_valid: false,
    };
  }
  if (!dynamicTensors.length) {
    const staticValid = contract.schema === expectedSchema
      && contract.status === "not_applicable_static_shapes"
      && Number(contract.dynamic_tensor_count) === 0
      && Number(contract.symbol_count) === 0
      && !(contract.symbols || []).length
      && !(contract.tensor_formulas || []).length
      && !(contract.op_formulas || []).length
      && contract.total_macs_formula == null;
    return {
      contract_present: true,
      static_non_applicability_valid: staticValid,
      symbols_valid: staticValid,
      tensor_formulas_valid: staticValid,
      op_formulas_valid: staticValid,
      total_macs_valid: staticValid,
      liveness_valid: contract.liveness?.status === "not_applicable_static_shapes",
      report_valid: String(engineeringReport || "").includes("## Dynamic Shape Cost Contract (DERIVED)")
        && String(engineeringReport || "").includes("not_applicable_static_shapes"),
      mlbom_valid: compactEvidencePointerValid(mlBomDocument) || propertyValue(mlBomDocument, "deepbom:model:dynamicShapeCostContractEvidence") === JSON.stringify(contract)
        && propertyValue(mlBomDocument, "deepbom:model:dynamicShapeCostStatus") === String(contract.status)
        && propertyValue(mlBomDocument, "deepbom:model:dynamicShapeSymbolCount") === "0"
        && propertyValue(mlBomDocument, "deepbom:model:dynamicShapeTotalMacBlockers") === "0"
        && propertyValue(mlBomDocument, "deepbom:model:dynamicShapeTotalMacFormula") === String(contract.total_macs_formula_status)
        && propertyValue(mlBomDocument, "deepbom:model:dynamicShapePeakLivePayloadFormula") === String(contract.liveness?.peak_selection_status),
      finding_valid: !findingRows(findingsRegister).some((finding) => finding.finding_id === "EA-DYN-0001"),
    };
  }

  const expectedSymbols = expectedSymbolRegistry(analysis?.tensors || [], format);
  const opEvidence = expectedOpEvidence(analysis, format, expectedSymbols.symbolByAxis);
  const total = expectedTotalMacs(analysis, format, opEvidence);
  const liveness = expectedLiveness(analysis, format, expectedSymbols.symbolByAxis);
  const tensorValid = tensorRowsMatch(contract, analysis, format, expectedSymbols.symbolByAxis);
  const opValid = opRowsMatch(contract, opEvidence);
  const totalValid = totalMatches(contract, total);
  const livenessValid = livenessMatches(contract.liveness, liveness);
  const formulaPartial = (contract.tensor_formulas || []).some((row) => !String(row.formula_status || "").startsWith("exact_"));
  const expectedStatus = opEvidence.unresolved.length || total.issues.length || formulaPartial || contract.liveness?.status !== "assessed"
    ? "partial" : "assessed";
  const report = String(engineeringReport || "");
  const reportValid = report.includes("## Dynamic Shape Cost Contract (DERIVED)")
    && report.includes(contract.schema)
    && report.includes(contract.status)
    && report.includes(contract.total_macs_formula?.expression || contract.total_macs_formula_status)
    && report.includes(contract.liveness?.peak_selection_status)
    && report.includes(contract.liveness?.peak_live_payload_formula?.expression || contract.liveness?.peak_selection_status)
    && (contract.op_formulas || []).every((row) => report.includes(row.macs_formula?.expression || "__missing__"))
    && (contract.total_macs_unresolved_ops || []).every((row) => report.includes(row.reason || "__missing__"));
  const mlbomValid = compactEvidencePointerValid(mlBomDocument) || propertyValue(mlBomDocument, "deepbom:model:dynamicShapeCostContractEvidence") === JSON.stringify(contract)
    && propertyValue(mlBomDocument, "deepbom:model:dynamicShapeCostStatus") === String(contract.status)
    && propertyValue(mlBomDocument, "deepbom:model:dynamicShapeSymbolCount") === String(contract.symbol_count)
    && propertyValue(mlBomDocument, "deepbom:model:dynamicShapeTotalMacBlockers") === String(contract.total_macs_unresolved_op_count)
    && propertyValue(mlBomDocument, "deepbom:model:dynamicShapeTotalMacFormula") === String(contract.total_macs_formula?.expression || contract.total_macs_formula_status)
    && propertyValue(mlBomDocument, "deepbom:model:dynamicShapePeakLivePayloadFormula") === String(contract.liveness?.peak_live_payload_formula?.expression || contract.liveness?.peak_selection_status);
  const finding = findingRows(findingsRegister).find((item) => item.finding_id === "EA-DYN-0001");
  const findingValid = Boolean(finding)
    && finding.evidence_class === "DERIVED"
    && (finding.evidence_json_pointers || []).includes("/evidence/static_analysis/dynamic_shape_cost_contract");
  return {
    contract_present: contract.schema === expectedSchema && contract.format === format
      && contract.evidence_class === "DERIVED" && contract.status === expectedStatus
      && Number(contract.dynamic_tensor_count) === dynamicTensors.length,
    static_non_applicability_valid: true,
    symbols_valid: Number(contract.symbol_count) === expectedSymbols.rows.length
      && symbolsMatch(contract.symbols, expectedSymbols.rows),
    tensor_formulas_valid: tensorValid,
    op_formulas_valid: opValid,
    total_macs_valid: totalValid,
    liveness_valid: livenessValid,
    report_valid: reportValid,
    mlbom_valid: mlbomValid,
    finding_valid: findingValid,
  };
}
