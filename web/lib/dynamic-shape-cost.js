import { isOnnxMacBearingOperation } from "./onnx-operation-cost.js";
import { onnxDimensionExpressionDependencies, parseOnnxDimensionExpression } from "./onnx-dimension-expression.js";
import { parseOnnxEinsumEquation } from "./onnx-einsum-contract.js";
import {
  divisibleGuard,
  equalityGuard,
  evaluateGuardedIntegerFormula,
  integerCall,
  integerConstant,
  integerSymbol,
  nonnegativeGuard,
  serializeGuardedIntegerFormula,
} from "./guarded-integer-expression.js";

export const DYNAMIC_SHAPE_COST_SCHEMA = "deepbom.dynamic_shape_cost_contract.v2.2";

const DTYPE_STORAGE_BITS = new Map([
  ["BOOL", 8], ["UINT8", 8], ["INT8", 8], ["FLOAT8E4M3FN", 8], ["FLOAT8E4M3FNUZ", 8],
  ["FLOAT8E5M2", 8], ["FLOAT8E5M2FNUZ", 8], ["FLOAT8E8M0", 8], ["UINT16", 16],
  ["INT16", 16], ["FLOAT16", 16], ["BFLOAT16", 16], ["UINT32", 32], ["INT32", 32],
  ["FLOAT32", 32], ["UINT64", 64], ["INT64", 64], ["FLOAT64", 64], ["COMPLEX64", 64],
  ["COMPLEX128", 128], ["UINT4", 4], ["INT4", 4], ["FLOAT4E2M1", 4], ["UINT2", 2],
  ["INT2", 2],
]);

export const ONNX_DYNAMIC_COST_SOURCE = Object.freeze({
  repository: "onnx/onnx",
  release: "v1.21.0",
  commit: "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b",
  path: "onnx/defs/nn/defs.cc",
  source_ref: "https://raw.githubusercontent.com/onnx/onnx/be2b5fde82d9c8874f3d19328bdfe3b6962dc67b/onnx/defs/nn/defs.cc",
  sha256: "1619dd419d2eaa1da3ad4155206d58d86432829a534d5a8c587269abf5c1df02",
  documents: Object.freeze([
    Object.freeze({
      role: "neural_network_operator_schemas",
      path: "onnx/defs/nn/defs.cc",
      source_ref: "https://raw.githubusercontent.com/onnx/onnx/be2b5fde82d9c8874f3d19328bdfe3b6962dc67b/onnx/defs/nn/defs.cc",
      sha256: "1619dd419d2eaa1da3ad4155206d58d86432829a534d5a8c587269abf5c1df02",
    }),
    Object.freeze({
      role: "matrix_and_einsum_operator_schemas",
      path: "onnx/defs/math/defs.cc",
      source_ref: "https://raw.githubusercontent.com/onnx/onnx/be2b5fde82d9c8874f3d19328bdfe3b6962dc67b/onnx/defs/math/defs.cc",
      sha256: "0428224a3cb2b5aabf87dab3dfca94988c3a913d73b6f39fa295980060b97594",
    }),
  ]),
});

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

function compactInvalidTensorContract(tensor, role) {
  if (tensor?.contract_status !== "invalid" && tensor?.contractStatus !== "invalid") return null;
  const conflict = tensor.contract_conflict || tensor.contractConflict || {};
  const root = conflict.root_conflict || conflict.rootConflict || conflict;
  return {
    resolution_class: "artifact_contract_conflict",
    blocking_tensor_role: role,
    blocking_tensor_index: tensorIndex(tensor),
    blocking_tensor_name: String(tensor?.name || ""),
    root_conflict: {
      node_index: Number.isSafeInteger(Number(root?.node_index)) ? Number(root.node_index) : null,
      op_name: String(root?.op_name || ""),
      tensor_name: String(root?.tensor_name || tensor?.name || ""),
      field: String(root?.field || ""),
      reason: String(root?.reason || conflict?.reason || "invalid_serialized_tensor_contract"),
    },
  };
}

function compactConditionalTensorConflict(tensor, role) {
  const contract = tensor?.conditional_shape_contract || tensor?.conditionalShapeContract || {};
  const failures = Array.isArray(contract.variant_failures) ? contract.variant_failures : [];
  const invalidFailures = failures.filter((row) => row?.status === "invalid");
  if (!invalidFailures.length) return null;
  const first = invalidFailures[0] || {};
  return {
    resolution_class: "artifact_contract_conflict",
    blocking_tensor_role: role,
    blocking_tensor_index: tensorIndex(tensor),
    blocking_tensor_name: String(tensor?.name || ""),
    root_conflict: {
      node_index: null,
      op_name: "conditional_shape_contract",
      tensor_name: String(tensor?.name || ""),
      field: "conditional_shape_contract.variant_failures",
      reason: String(first.reason || "conditionally_invalid_tensor_contract"),
      invalid_variant_count: invalidFailures.length,
      conditions: structuredClone(first.conditions || []),
    },
  };
}

function dynamicShape(tensor) {
  return Array.isArray(tensor?.shape) && tensor.shape.some((dimension) => Number(dimension) < 0);
}

function shapeDeclared(tensor) {
  return tensor?.shape_declared === true || tensor?.shapeDeclared === true;
}

function symbolRegistry() {
  const rows = [];
  const byKey = new Map();
  const occurrenceKeys = new Set();
  return {
    rows,
    get(key, source, declaredName, occurrence, bounds = null) {
      let row = byKey.get(key);
      if (!row) {
        const expressionIr = source === "onnx_derived_dimension_expression" ? parseOnnxDimensionExpression(declaredName) : null;
        row = {
          symbol_id: `D${rows.length}`,
          source,
          declared_name: declaredName || "",
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
      mergeSymbolBounds(row, bounds);
      const occurrenceKey = `${row.symbol_id}:${occurrence.tensor_index}:${occurrence.axis}`;
      if (!occurrenceKeys.has(occurrenceKey)) {
        occurrenceKeys.add(occurrenceKey);
        row.occurrences.push(occurrence);
      }
      return row.symbol_id;
    },
  };
}

function runtimeDimensionBounds(tensor, axis, parameter) {
  const rows = tensor?.runtime_dimension_bounds || tensor?.runtimeDimensionBounds || [];
  const row = rows.find((item) => Number(item?.axis) === axis
    && (!item?.symbol || String(item.symbol) === parameter));
  if (!row) return null;
  const lowerText = String(row.lower_bound_decimal ?? "");
  const upperText = String(row.upper_bound_decimal ?? "");
  const lower = /^(0|[1-9][0-9]*)$/.test(lowerText) && BigInt(lowerText) <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(lowerText) : null;
  const upper = /^(0|[1-9][0-9]*)$/.test(upperText) && BigInt(upperText) <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(upperText) : null;
  return {
    lower_bound: lower,
    upper_bound: upper,
    upper_bound_expression: String(row.upper_bound_expression || "") || null,
  };
}

function mergeSymbolBounds(row, bounds) {
  if (!bounds) return;
  for (const key of ["lower_bound", "upper_bound", "upper_bound_expression"]) {
    const value = bounds[key] ?? null;
    if (value == null) continue;
    if (row[key] != null && row[key] !== value) {
      row.bounds_status = "artifact_contract_bounds_conflict";
      return;
    }
    row[key] = value;
  }
  if (row.bounds_status !== "artifact_contract_bounds_conflict") {
    row.bounds_status = row.lower_bound != null && (row.upper_bound != null || row.upper_bound_expression)
      ? "artifact_derived_lower_and_upper" : "artifact_derived_partial";
  }
}

function dimensionDescriptor(tensor, axis, registry) {
  const raw = Number(tensor?.shape?.[axis]);
  if (Number.isSafeInteger(raw) && raw >= 0) return { kind: "constant", value: BigInt(raw) };
  const dimension = tensor?.type_proto?.shapeDimensions?.[axis] || tensor?.typeProto?.shapeDimensions?.[axis] || null;
  const index = tensorIndex(tensor);
  const occurrence = { tensor_index: index, tensor_name: tensor?.name || `T${index}`, axis };
  if (dimension?.kind === "symbolic" && String(dimension.parameter || "")) {
    const parameter = String(dimension.parameter);
    const derived = parameter.startsWith("deepbom_expr:");
    const runtimeValue = parameter.startsWith("deepbom_runtime:");
    const source = derived ? "onnx_derived_dimension_expression"
      : runtimeValue ? "onnx_runtime_value_dimension" : "onnx_dim_param";
    const bounds = runtimeDimensionBounds(tensor, axis, parameter);
    return {
      kind: "symbol",
      symbol_id: registry.get(
        `${derived ? "onnx:derived" : runtimeValue ? "onnx:runtime" : "onnx:param"}:${parameter}`,
        source,
        parameter,
        occurrence,
        bounds,
      ),
    };
  }
  return {
    kind: "symbol",
    symbol_id: registry.get(`onnx:anonymous:${index}:${axis}`, "onnx_anonymous_dimension", "", occurrence),
  };
}

function factorKey(factors) {
  return [...factors.entries()]
    .filter(([, exponent]) => exponent > 0)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([symbol, exponent]) => `${symbol}^${exponent}`)
    .join("*");
}

function normalizePolynomial(terms) {
  const coefficients = new Map();
  for (const term of terms || []) {
    const coefficient = BigInt(term.coefficient);
    if (coefficient === 0n) continue;
    const factors = new Map(term.factors || []);
    const key = factorKey(factors);
    coefficients.set(key, (coefficients.get(key) || 0n) + coefficient);
  }
  return [...coefficients.entries()]
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

function addPolynomials(...polynomials) {
  return normalizePolynomial(polynomials.flatMap((polynomial) => polynomial || []));
}

function scalePolynomial(polynomial, numerator, denominator = 1n) {
  const scaled = [];
  for (const term of polynomial || []) {
    const product = term.coefficient * BigInt(numerator);
    if (product % BigInt(denominator) !== 0n) return null;
    scaled.push({ coefficient: product / BigInt(denominator), factors: term.factors });
  }
  return normalizePolynomial(scaled);
}

function monomialForDimensions(tensor, axes, registry) {
  if (!Array.isArray(tensor?.shape) || !shapeDeclared(tensor)) return null;
  let coefficient = 1n;
  const factors = new Map();
  for (const axis of axes) {
    if (!Number.isSafeInteger(axis) || axis < 0 || axis >= tensor.shape.length) return null;
    const dimension = dimensionDescriptor(tensor, axis, registry);
    if (dimension.kind === "constant") coefficient *= dimension.value;
    else factors.set(dimension.symbol_id, (factors.get(dimension.symbol_id) || 0) + 1);
  }
  return normalizePolynomial([{ coefficient, factors }]);
}

function monomialForTensor(tensor, registry) {
  if (!Array.isArray(tensor?.shape) || !shapeDeclared(tensor)) return null;
  return monomialForDimensions(tensor, tensor.shape.map((_, axis) => axis), registry);
}

function termExpression(term) {
  const factors = term.factors.map(([symbol, exponent]) => exponent === 1 ? symbol : `${symbol}^${exponent}`);
  if (!factors.length) return term.coefficient.toString();
  if (term.coefficient === 1n) return factors.join("*");
  return `${term.coefficient.toString()}*${factors.join("*")}`;
}

function polynomialExpression(polynomial) {
  return polynomial?.length ? polynomial.map(termExpression).join(" + ") : "0";
}

function serializeFormula(polynomial, unit, method) {
  const normalized = normalizePolynomial(polynomial || []);
  return {
    status: "exact_symbolic_integer_polynomial",
    unit,
    expression: polynomialExpression(normalized),
    terms: normalized.map((term) => ({
      coefficient_decimal: term.coefficient.toString(),
      factors: term.factors.map(([symbol_id, exponent]) => ({ symbol_id, exponent })),
    })),
    symbol_ids: [...new Set(normalized.flatMap((term) => term.factors.map(([symbol]) => symbol)))].sort(),
    method,
  };
}

function formulaKey(polynomial) {
  return normalizePolynomial(polynomial || []).map((term) => `${term.coefficient}:${factorKey(new Map(term.factors))}`).join("|");
}

function coefficientMap(polynomial) {
  return new Map(normalizePolynomial(polynomial || []).map((term) => [factorKey(new Map(term.factors)), term.coefficient]));
}

function polynomialDominates(left, right) {
  const leftMap = coefficientMap(left);
  const rightMap = coefficientMap(right);
  for (const [key, coefficient] of rightMap) {
    if ((leftMap.get(key) || 0n) < coefficient) return false;
  }
  return true;
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

function tensorAt(tensorByIndex, index) {
  return Number.isSafeInteger(Number(index)) && Number(index) >= 0 ? tensorByIndex.get(Number(index)) || null : null;
}

function optionalInputTensor(op, position, tensorByIndex) {
  const index = op?.inputs?.[position];
  return index === "" || index == null ? null : tensorAt(tensorByIndex, index);
}

function staticDimension(shape, axis) {
  const value = Number(shape?.[axis]);
  return Number.isSafeInteger(value) && value >= 0 ? value : null;
}

function knownDimensionConflict(left, right) {
  return left != null && right != null && left !== right;
}

function multiplyMonomials(...polynomials) {
  let coefficient = 1n;
  const factors = new Map();
  for (const polynomial of polynomials) {
    if (!polynomial || polynomial.length !== 1) return null;
    const term = polynomial[0];
    coefficient *= term.coefficient;
    for (const [symbol, exponent] of term.factors) factors.set(symbol, (factors.get(symbol) || 0) + exponent);
  }
  return normalizePolynomial([{ coefficient, factors }]);
}

function polynomialExpressionIr(polynomial) {
  const terms = normalizePolynomial(polynomial || []).map((term) => integerCall("mul", [
    integerConstant(term.coefficient),
    ...term.factors.flatMap(([symbol, exponent]) => Array(exponent).fill(null).map(() => integerSymbol(symbol))),
  ]));
  return integerCall("add", terms);
}

function dimensionExpressionIr(tensor, axis, registry) {
  if (!Array.isArray(tensor?.shape) || !shapeDeclared(tensor) || axis < 0 || axis >= tensor.shape.length) return null;
  const dimension = dimensionDescriptor(tensor, axis, registry);
  return dimension.kind === "constant" ? integerConstant(dimension.value) : integerSymbol(dimension.symbol_id);
}

function exactSpatialValues(op, name, rank, fallback, allowZero = false) {
  const values = attributeInts(op, name);
  if (!values.length) return Array(rank).fill(fallback);
  return values.length === rank && values.every((value) => allowZero ? value >= 0 : value > 0) ? values : null;
}

function hasAttribute(op, name) {
  return (op?.onnx_attributes || []).some((attribute) => attribute.name === name);
}

function guardedConvTransposeFormula(op, input, weight, output, registry) {
  const group = attributeInt(op, "group", 1);
  const autoPad = attributeString(op, "auto_pad", "NOTSET");
  const inputDeclared = shapeDeclared(input);
  const rank = inputDeclared ? input.shape?.length || 0 : weight.shape?.length || 0;
  const weightRank = weight.shape?.length || 0;
  const outputRank = output.shape?.length || 0;
  if (rank < 3 || weightRank !== rank || outputRank !== rank || !Number.isSafeInteger(group) || group <= 0
    || !["NOTSET", "VALID"].includes(autoPad) || hasAttribute(op, "output_shape")) return null;
  if (autoPad === "VALID" && hasAttribute(op, "pads")) return null;
  if (weight.shape.some((value) => !Number.isSafeInteger(Number(value)) || Number(value) <= 0)) return null;
  const spatialRank = rank - 2;
  const strides = exactSpatialValues(op, "strides", spatialRank, 1);
  const dilations = exactSpatialValues(op, "dilations", spatialRank, 1);
  const outputPadding = exactSpatialValues(op, "output_padding", spatialRank, 0, true);
  const pads = autoPad === "VALID" ? Array(spatialRank * 2).fill(0) : exactSpatialValues(op, "pads", spatialRank * 2, 0, true);
  const kernelShape = hasAttribute(op, "kernel_shape") ? attributeInts(op, "kernel_shape") : weight.shape.slice(2).map(Number);
  if (![strides, dilations, outputPadding, pads].every(Boolean)
    || kernelShape.length !== spatialRank
    || kernelShape.some((value, axis) => value !== Number(weight.shape[axis + 2]))
    || outputPadding.some((value, axis) => value >= strides[axis] && value >= dilations[axis])) return null;

  const weightChannels = Number(weight.shape[0]);
  const weightOutputChannels = Number(weight.shape[1]);
  if (weightChannels % group !== 0) return null;
  const outputBatch = dimensionExpressionIr(output, 0, registry);
  const outputChannels = dimensionExpressionIr(output, 1, registry);
  if (!outputBatch || !outputChannels) return null;
  const preconditions = [equalityGuard(outputChannels, integerConstant(weightOutputChannels * group), "ConvTranspose output channels")];
  const factors = [outputBatch, integerConstant(weightChannels), integerConstant(weightOutputChannels)];

  if (!inputDeclared) {
    if ((input.shape?.length || 0) !== 0 || pads.some((value) => value !== 0)) return null;
    for (let axis = 0; axis < spatialRank; axis += 1) {
      const outputDimension = dimensionExpressionIr(output, axis + 2, registry);
      if (!outputDimension) return null;
      const effectiveKernel = (kernelShape[axis] - 1) * dilations[axis] + 1;
      const numerator = integerCall("add", [
        outputDimension,
        integerConstant(pads[axis] + pads[axis + spatialRank] - outputPadding[axis] - effectiveKernel + strides[axis]),
      ]);
      const stride = integerConstant(strides[axis]);
      preconditions.push(nonnegativeGuard(numerator, `ConvTranspose inverse input axis ${axis}`));
      preconditions.push(divisibleGuard(numerator, stride, `ConvTranspose inverse input axis ${axis}`));
      factors.push(integerCall("mul", integerCall("exact_div", numerator, stride), integerConstant(kernelShape[axis])));
    }
    return {
      expression_ir: integerCall("mul", factors),
      preconditions,
      method: "Exact ONNX ConvTranspose MAC count from N*C*(M/group) and the guarded inverse output-size relation for every uncropped spatial axis. Each inferred input extent is required to be non-negative and exactly divisible by stride.",
    };
  }

  if (!pads.some((value) => value !== 0)) return null;
  const inputBatch = dimensionExpressionIr(input, 0, registry);
  const inputChannels = dimensionExpressionIr(input, 1, registry);
  if (!inputBatch || !inputChannels) return null;
  preconditions.push(equalityGuard(inputBatch, outputBatch, "ConvTranspose batch"));
  preconditions.push(equalityGuard(inputChannels, integerConstant(weightChannels), "ConvTranspose input channels"));
  factors[0] = inputBatch;
  for (let axis = 0; axis < spatialRank; axis += 1) {
    const inputDimension = dimensionExpressionIr(input, axis + 2, registry);
    const outputDimension = dimensionExpressionIr(output, axis + 2, registry);
    if (!inputDimension || !outputDimension) return null;
    const effectiveKernel = (kernelShape[axis] - 1) * dilations[axis] + 1;
    const expectedOutput = integerCall("add", [
      integerCall("mul", integerCall("sub", inputDimension, integerConstant(1)), integerConstant(strides[axis])),
      integerConstant(outputPadding[axis] + effectiveKernel - pads[axis] - pads[axis + spatialRank]),
    ]);
    preconditions.push(equalityGuard(outputDimension, expectedOutput, `ConvTranspose output axis ${axis}`));
    factors.push(integerCall("conv_transpose_pairs", [
      inputDimension,
      integerConstant(kernelShape[axis]),
      integerConstant(strides[axis]),
      integerConstant(dilations[axis]),
      integerConstant(pads[axis]),
      outputDimension,
    ]));
  }
  return {
    expression_ir: integerCall("mul", factors),
    preconditions,
    method: "Exact contributing input/kernel pair count per ONNX ConvTranspose spatial axis, multiplied by N*C*(M/group). The emitted equality guards preserve the source output-size equation and prevent a cropped output-volume approximation.",
  };
}

function onnxOpMacFormula(op, tensorByIndex, registry) {
  const inputIndex = Number(op?.inputs?.[0]);
  const weightIndex = ["QLinearConv", "QLinearMatMul"].includes(op.name) ? Number(op?.inputs?.[3]) : Number(op?.inputs?.[1]);
  const input = tensorAt(tensorByIndex, inputIndex);
  const weight = tensorAt(tensorByIndex, weightIndex);
  const output = tensorAt(tensorByIndex, Number(op?.outputs?.[0]));
  const invalid = compactInvalidTensorContract(input, "input")
    || compactInvalidTensorContract(weight, "weight")
    || compactInvalidTensorContract(output, "output")
    || compactConditionalTensorConflict(input, "input")
    || compactConditionalTensorConflict(weight, "weight")
    || compactConditionalTensorConflict(output, "output");
  if (invalid) return { polynomial: null, reason: "blocked by an invalid serialized tensor contract", ...invalid };
  if (![input, weight, output].every(denseTensor)) return {
    polynomial: null,
    reason: "dense input, weight, and output tensors are required",
    resolution_class: "analyzer_or_contract_residual",
  };
  if (["Conv", "QLinearConv", "ConvInteger"].includes(op.name)) {
    const rank = weight.shape?.length || 0, inputRankDeclared = shapeDeclared(input), inputRank = input.shape?.length || 0;
    const group = attributeInt(op, "group", 1);
    const weightInputChannels = staticDimension(weight.shape, 1), weightOutputChannels = staticDimension(weight.shape, 0);
    const inputChannels = staticDimension(input.shape, 1), outputChannels = staticDimension(output.shape, 1);
    const inputBatch = staticDimension(input.shape, 0), outputBatch = staticDimension(output.shape, 0);
    if (rank < 3 || output.shape?.length !== rank || inputRankDeclared && inputRank !== rank || !Number.isSafeInteger(group) || group <= 0
      || weightInputChannels == null || weightInputChannels <= 0 || weightOutputChannels == null || weightOutputChannels <= 0
      || weight.shape.slice(2).some((value) => !Number.isSafeInteger(Number(value)) || Number(value) <= 0)
      || weightOutputChannels % group !== 0
      || knownDimensionConflict(inputChannels, weightInputChannels * group)
      || knownDimensionConflict(outputChannels, weightOutputChannels)
      || knownDimensionConflict(inputBatch, outputBatch)) {
      return { polynomial: null, reason: "rank-N (N>=3) output/weight tensors, any declared compatible input rank/channels, and a positive group are required" };
    }
    const outputWithoutChannels = monomialForDimensions(output, output.shape.map((_, axis) => axis).filter((axis) => axis !== 1), registry);
    const weightVolume = monomialForTensor(weight, registry);
    return {
      polynomial: multiplyMonomials(outputWithoutChannels, weightVolume),
      reason: "product(output_shape excluding C)*product(weight_shape), with source-compatible declared batch/channel/group constraints for ONNX rank-N Conv; an omitted input rank remains a runtime validity condition and does not change the contraction cardinality",
    };
  }
  if (op.name === "ConvTranspose") {
    const rank = input.shape?.length || 0, group = attributeInt(op, "group", 1), autoPad = attributeString(op, "auto_pad", "NOTSET");
    const pads = attributeInts(op, "pads"), kernelShape = attributeInts(op, "kernel_shape"), outputShape = attributeInts(op, "output_shape");
    const channels = staticDimension(input.shape, 1), weightChannels = staticDimension(weight.shape, 0), weightOutputChannels = staticDimension(weight.shape, 1);
    const outputChannels = staticDimension(output.shape, 1), inputBatch = staticDimension(input.shape, 0), outputBatch = staticDimension(output.shape, 0);
    const weightRank = weight.shape?.length || 0, outputRank = output.shape?.length || 0;
    if (!shapeDeclared(input) && rank === 0 && weightRank >= 3 && outputRank === weightRank && group > 0
      && !outputShape.length && ["NOTSET", "VALID"].includes(autoPad) && pads.every((value) => value === 0)) {
      const guarded = guardedConvTransposeFormula(op, input, weight, output, registry);
      return {
        polynomial: null,
        guarded,
        reason: guarded
          ? guarded.method
          : "ConvTranspose input rank is not serialized; exact uncropped MACs require a guarded inverse output-size relation with lower-bound and divisibility preconditions",
      };
    }
    if (pads.some((value) => value !== 0)) {
      const guarded = guardedConvTransposeFormula(op, input, weight, output, registry);
      return {
        polynomial: null,
        guarded,
        reason: guarded
          ? guarded.method
          : "Cropped ConvTranspose MACs require an exact piecewise spatial-overlap expression; output volume times kernel volume would overcount discarded pairs",
      };
    }
    if (rank < 3 || weightRank !== rank || outputRank !== rank || group <= 0
      || weightChannels == null || weightChannels <= 0 || weightOutputChannels == null || weightOutputChannels <= 0
      || weightChannels % group !== 0 || knownDimensionConflict(channels, weightChannels)
      || knownDimensionConflict(outputChannels, weightOutputChannels * group) || knownDimensionConflict(inputBatch, outputBatch) || outputShape.length
      || !["NOTSET", "VALID"].includes(autoPad)
      || kernelShape.length && (kernelShape.length !== rank - 2 || kernelShape.some((value, axis) => value !== Number(weight.shape?.[axis + 2])))) {
      return { polynomial: null, guarded: null, reason: "ConvTranspose symbolic MACs require compatible static channels and an uncropped NOTSET/VALID spatial contract" };
    }
    return {
      polynomial: multiplyMonomials(
        monomialForDimensions(input, input.shape.map((_, axis) => axis).filter((axis) => axis !== 1), registry),
        monomialForTensor(weight, registry),
      ),
      guarded: null,
      reason: "product(input_shape excluding C)*product(weight_shape) for source-compatible uncropped ONNX rank-N ConvTranspose",
    };
  }
  if (op.name === "Attention") {
    const valueTensor = optionalInputTensor(op, 2, tensorByIndex);
    const pastKey = optionalInputTensor(op, 4, tensorByIndex);
    const pastValue = optionalInputTensor(op, 5, tensorByIndex);
    if (![input, weight, valueTensor, output].every(denseTensor)) {
      return { polynomial: null, reason: "Attention symbolic MACs require dense Q, K, V, and Y tensors" };
    }
    const rank = input.shape?.length || 0;
    if (![3, 4].includes(rank) || weight.shape?.length !== rank || valueTensor.shape?.length !== rank || output.shape?.length !== rank) {
      return { polynomial: null, reason: "Attention Q, K, V, and Y must share source-compatible rank 3 or 4" };
    }
    let base, incomingSequence, qHeads, qkHeadSize, valueHeadSize;
    if (rank === 4) {
      qHeads = staticDimension(input.shape, 1);
      const kvHeads = staticDimension(weight.shape, 1);
      qkHeadSize = staticDimension(input.shape, 3);
      valueHeadSize = staticDimension(valueTensor.shape, 3);
      if ([qHeads, kvHeads, qkHeadSize, valueHeadSize].some((value) => value == null || value <= 0)
        || qHeads < kvHeads || qHeads % kvHeads !== 0
        || knownDimensionConflict(staticDimension(weight.shape, 3), qkHeadSize)) {
        return { polynomial: null, reason: "Attention rank-4 head counts and head widths must be static and source-compatible" };
      }
      base = monomialForDimensions(input, [0, 2], registry);
      incomingSequence = monomialForDimensions(weight, [2], registry);
    } else {
      qHeads = attributeInt(op, "q_num_heads", 0);
      const kvHeads = attributeInt(op, "kv_num_heads", 0);
      const qHidden = staticDimension(input.shape, 2), kHidden = staticDimension(weight.shape, 2), vHidden = staticDimension(valueTensor.shape, 2);
      if (![qHeads, kvHeads, qHidden, kHidden, vHidden].every((value) => Number.isSafeInteger(value) && value > 0)
        || qHeads < kvHeads || qHeads % kvHeads !== 0
        || qHidden % qHeads !== 0 || kHidden % kvHeads !== 0 || vHidden % kvHeads !== 0
        || qHidden / qHeads !== kHidden / kvHeads) {
        return { polynomial: null, reason: "Attention rank-3 hidden widths and head attributes must be static, divisible, and source-compatible" };
      }
      qkHeadSize = qHidden / qHeads;
      valueHeadSize = vHidden / kvHeads;
      base = monomialForDimensions(input, [0, 1], registry);
      incomingSequence = monomialForDimensions(weight, [1], registry);
    }
    const coefficient = qHeads * (qkHeadSize + valueHeadSize);
    const sequenceTerms = [incomingSequence];
    if (pastKey || pastValue) {
      if (!(pastKey && pastValue && denseTensor(pastKey) && denseTensor(pastValue))
        || pastKey.shape?.length !== 4 || pastValue.shape?.length !== 4) {
        return { polynomial: null, reason: "Attention past key and value must be a complete rank-4 cache pair" };
      }
      sequenceTerms.push(monomialForDimensions(pastKey, [2], registry));
    }
    const terms = sequenceTerms.map((sequence) => scalePolynomial(multiplyMonomials(base, sequence), coefficient));
    if (!base || sequenceTerms.some((term) => !term) || terms.some((term) => !term)) {
      return { polynomial: null, reason: "Attention symbolic batch or sequence dimensions could not be represented as non-negative integer monomials" };
    }
    return {
      polynomial: addPolynomials(...terms),
      reason: "B*q_num_heads*q_sequence_length*total_sequence_length*(qk_head_size+v_head_size), exactly summing the two source-defined dense Attention contractions; runtime equality of distinct symbolic Q/K/V dimensions remains an operator-validity condition",
    };
  }
  if (op.name === "DeformConv") {
    const offset = optionalInputTensor(op, 2, tensorByIndex);
    const rank = input.shape?.length || 0;
    const group = attributeInt(op, "group", 1), offsetGroup = attributeInt(op, "offset_group", 1);
    const weightInputChannels = staticDimension(weight.shape, 1), weightOutputChannels = staticDimension(weight.shape, 0);
    if (![input, weight, offset, output].every(denseTensor) || rank < 3 || weight.shape?.length !== rank
      || offset.shape?.length !== rank || output.shape?.length !== rank || group <= 0 || offsetGroup <= 0
      || weightInputChannels == null || weightInputChannels <= 0 || weightOutputChannels == null || weightOutputChannels <= 0
      || weight.shape.slice(2).some((value) => !Number.isSafeInteger(Number(value)) || Number(value) <= 0)
      || weightOutputChannels % group !== 0) {
      return { polynomial: null, reason: "DeformConv symbolic MACs require compatible dense rank-N X/W/offset/Y contracts and static positive channel/kernel/group values" };
    }
    return {
      polynomial: multiplyMonomials(
        monomialForDimensions(output, output.shape.map((_, axis) => axis).filter((axis) => axis !== 1), registry),
        monomialForTensor(weight, registry),
      ),
      reason: "product(Y shape excluding output channels)*product(W shape), counting only the source-defined sampled-value/weight contraction after the offset/mask contract is validated",
    };
  }
  if (op.name === "Einsum") {
    const inputs = (op.inputs || []).map((index) => index === "" || index == null ? null : tensorAt(tensorByIndex, index)).filter(Boolean);
    if (!inputs.length || inputs.some((tensor) => !denseTensor(tensor) || !shapeDeclared(tensor))) {
      return { polynomial: null, reason: "Einsum symbolic MACs require every operand rank to be serialized" };
    }
    const parsed = parseOnnxEinsumEquation(attributeString(op, "equation", ""), inputs.map((tensor) => tensor.shape.length));
    if (parsed.status !== "assessed") return { polynomial: null, reason: `Einsum equation contract is invalid: ${parsed.reason}` };
    if (inputs.length === 1) return { polynomial: [], reason: "A one-input Einsum has no binary tensor contraction" };
    if (inputs.length !== 2) return { polynomial: null, reason: "Einsum with more than two operands has contraction-order-dependent work not serialized by ONNX" };
    const domains = new Map();
    for (let inputIndex = 0; inputIndex < inputs.length; inputIndex += 1) {
      const operand = inputs[inputIndex];
      for (let axis = 0; axis < parsed.operands[inputIndex].length; axis += 1) {
        const label = parsed.operands[inputIndex][axis];
        const candidate = monomialForDimensions(operand, [axis], registry);
        if (!candidate) return { polynomial: null, reason: `Einsum label ${label} has no representable dimension domain` };
        const previous = domains.get(label);
        if (!previous) {
          domains.set(label, candidate);
          continue;
        }
        const previousStatic = previous.length === 1 && previous[0].factors.length === 0 ? previous[0].coefficient : null;
        const candidateStatic = candidate.length === 1 && candidate[0].factors.length === 0 ? candidate[0].coefficient : null;
        if (label.startsWith("@ellipsis:")) {
          if (previousStatic === 1n) domains.set(label, candidate);
          else if (candidateStatic === 1n || formulaKey(previous) === formulaKey(candidate)) continue;
          else if (previousStatic != null && candidateStatic != null) return { polynomial: null, reason: "Einsum ellipsis dimensions are not broadcast-compatible" };
          else return { polynomial: null, reason: "Einsum symbolic ellipsis broadcast maximum is not determined by the artifact" };
        } else if (previousStatic != null && candidateStatic != null && previousStatic !== candidateStatic) {
          return { polynomial: null, reason: `Einsum label ${label} has incompatible static dimensions` };
        }
      }
    }
    const polynomial = multiplyMonomials(...parsed.all_labels.map((label) => domains.get(label)));
    return {
      polynomial,
      reason: "Product of every unique index-domain extent for a two-input Einsum; equality of distinct symbolic dimensions sharing a non-ellipsis label remains an operator-validity condition",
    };
  }
  if (["RNN", "GRU", "LSTM"].includes(op.name)) {
    const recurrent = tensorAt(tensorByIndex, Number(op?.inputs?.[2]));
    const gates = op.name === "LSTM" ? 4 : op.name === "GRU" ? 3 : 1;
    const hidden = attributeInt(op, "hidden_size", 0), layout = attributeInt(op, "layout", 0), direction = attributeString(op, "direction", "forward");
    const directions = direction === "bidirectional" ? 2 : ["forward", "reverse"].includes(direction) ? 1 : 0;
    if (![input, weight, recurrent].every(denseTensor) || input.shape?.length !== 3 || weight.shape?.length !== 3 || recurrent.shape?.length !== 3
      || ![0, 1].includes(layout) || !directions || hidden <= 0 || weight.shape[0] !== directions || recurrent.shape[0] !== directions
      || weight.shape[1] !== gates * hidden || recurrent.shape[1] !== gates * hidden || weight.shape[2] !== input.shape[2] || recurrent.shape[2] !== hidden) {
      return { polynomial: null, reason: `${op.name} symbolic MACs require compatible X/W/R, hidden_size, direction, layout, and gate dimensions` };
    }
    return {
      polynomial: scalePolynomial(monomialForDimensions(input, [layout ? 1 : 0, layout ? 0 : 1], registry), directions * gates * hidden * (input.shape[2] + hidden)),
      reason: `sequence*batch*directions*${gates}*hidden_size*(input_size+hidden_size)`,
    };
  }
  if (op.name === "Gemm") {
    if (input.shape?.length !== 2 || weight.shape?.length !== 2) return { polynomial: null, reason: "rank-2 Gemm A and B tensors are required" };
    const transAValue = attributeInt(op, "transA", 0);
    const transBValue = attributeInt(op, "transB", 0);
    if (![0, 1].includes(transAValue) || ![0, 1].includes(transBValue)) return { polynomial: null, reason: "Gemm transA/transB must be 0 or 1" };
    const transA = transAValue === 1;
    const transB = transBValue === 1;
    const m = monomialForDimensions(input, [transA ? 1 : 0], registry);
    const k = monomialForDimensions(input, [transA ? 0 : 1], registry);
    const n = monomialForDimensions(weight, [transB ? 0 : 1], registry);
    return { polynomial: multiplyMonomials(m, n, k), reason: "M*N*K after Gemm transA/transB" };
  }
  if (["MatMul", "QLinearMatMul", "MatMulInteger"].includes(op.name)) {
    if ((input.shape?.length || 0) < 1 || (weight.shape?.length || 0) < 1 || !Array.isArray(output.shape)) {
      const rankUnbound = !shapeDeclared(input) || !shapeDeclared(weight) || !shapeDeclared(output);
      return {
        polynomial: null,
        reason: rankUnbound
          ? "MatMul rank is not fully serialized; exact broadcast and contraction cardinality require a bound rank contract"
          : "rank-1-or-greater MatMul inputs and a declared output tensor are required",
        resolution_class: rankUnbound ? "external_binding_required" : "analyzer_or_contract_residual",
      };
    }
    const inputK = staticDimension(input.shape, input.shape.length - 1);
    const weightK = staticDimension(weight.shape, Math.max(0, weight.shape.length - 2));
    const expectedOutputRank = Math.max(input.shape.length === 1 ? 2 : input.shape.length, weight.shape.length === 1 ? 2 : weight.shape.length)
      - (input.shape.length === 1 ? 1 : 0) - (weight.shape.length === 1 ? 1 : 0);
    if (knownDimensionConflict(inputK, weightK) || output.shape.length !== expectedOutputRank) {
      return { polynomial: null, reason: "MatMul inner dimensions or ONNX rank-promotion output rank are incompatible" };
    }
    const outputVolume = monomialForTensor(output, registry);
    const reduction = monomialForDimensions(input, [input.shape.length - 1], registry);
    return { polynomial: multiplyMonomials(outputVolume, reduction), reason: "output_element_count*K after ONNX rank-1 promotion and batch broadcasting" };
  }
  return { polynomial: null, reason: `${op.name || "UNKNOWN"} has no dynamic MAC formula` };
}

function tensorPayloadFormula(tensor, registry) {
  if (!denseTensor(tensor)) return { status: "not_assessed_non_dense_value", element: null, bits: null, bytes: null, byteExpression: "" };
  const elements = monomialForTensor(tensor, registry);
  if (!elements) return { status: "not_assessed_shape_missing", element: null, bits: null, bytes: null, byteExpression: "" };
  const storageBits = DTYPE_STORAGE_BITS.get(String(tensor?.dtype || ""));
  if (!storageBits) return { status: "not_assessed_dtype_storage_width", element: elements, bits: null, bytes: null, byteExpression: "" };
  const bits = scalePolynomial(elements, BigInt(storageBits));
  const bytes = scalePolynomial(bits, 1n, 8n);
  return {
    status: bytes ? "exact_symbolic_integer_polynomial" : "exact_symbolic_ceil_expression",
    element: elements,
    bits,
    bytes,
    byteExpression: bytes ? polynomialExpression(bytes) : `ceil((${polynomialExpression(bits)})/8)`,
    storageBits,
  };
}

function buildTensorRows(tensors, registry) {
  return (tensors || []).filter((tensor) => dynamicShape(tensor)).map((tensor) => {
    const payload = tensorPayloadFormula(tensor, registry);
    return {
      tensor_index: tensorIndex(tensor),
      tensor_name: tensor?.name || `T${tensorIndex(tensor)}`,
      dtype: tensor?.dtype || "UNKNOWN",
      shape: [...(tensor?.shape || [])],
      value_kind: tensor?.value_kind || tensor?.valueKind || "tensor",
      constant_buffer: tensor?.constant_buffer === true,
      formula_status: payload.status,
      element_count_formula: payload.element ? serializeFormula(payload.element, "elements", "product of serialized tensor dimensions") : null,
      payload_bits_formula: payload.bits ? serializeFormula(payload.bits, "bits", `element count times ${payload.storageBits} storage bits`) : null,
      payload_bytes_formula: payload.bytes ? serializeFormula(payload.bytes, "bytes", "exact division of payload bits by 8") : null,
      payload_bytes_expression: payload.byteExpression,
      declared_shape_projection_bytes: null,
      declared_shape_projection_status: "not_available_for_onnx_unknown_dimension",
      reason: payload.status.startsWith("exact_")
        ? "The expression is exact after every symbol is bound to a non-negative runtime dimension; the artifact contains no numeric bounds."
        : "A complete dense tensor shape and fixed-width dtype are required.",
    };
  });
}

function buildOpRows(ops, tensorByIndex, registry) {
  const rows = [];
  const unresolved = [];
  for (const op of ops || []) {
    if (!isOnnxMacBearingOperation(op?.name, op?.standard_domain !== false)) continue;
    const referenced = [...(op.inputs || []), ...(op.outputs || [])].map((index) => tensorAt(tensorByIndex, Number(index))).filter(Boolean);
    if (!referenced.some(dynamicShape)) continue;
    const result = onnxOpMacFormula(op, tensorByIndex, registry);
    if (!result.polynomial && !result.guarded) {
      unresolved.push({ op_index: Number(op.index), op_name: op.name || "UNKNOWN", ...result, polynomial: undefined, guarded: undefined });
      continue;
    }
    if (result.guarded) {
      rows.push({
        op_index: Number(op.index),
        op_name: op.name || "UNKNOWN",
        formula_status: "exact_guarded_integer_expression",
        macs_formula: serializeGuardedIntegerFormula(result.guarded.expression_ir, "MACs", result.guarded.method, result.guarded.preconditions),
        declared_shape_projection_macs: null,
        declared_shape_projection_status: "not_available_for_onnx_unknown_dimension",
        reason: result.reason,
        _guarded: result.guarded,
      });
      continue;
    }
    rows.push({
      op_index: Number(op.index),
      op_name: op.name || "UNKNOWN",
      formula_status: "exact_symbolic_integer_polynomial",
      macs_formula: serializeFormula(result.polynomial, "MACs", result.reason),
      declared_shape_projection_macs: null,
      declared_shape_projection_status: "not_available_for_onnx_unknown_dimension",
      reason: result.reason,
      _polynomial: result.polynomial,
    });
  }
  return { rows, unresolved };
}

function buildTotalMacFormula(ops, opRows, tensorByIndex, registry) {
  const dynamicByIndex = new Map(opRows.map((row) => [row.op_index, row]));
  const terms = [];
  const expressionTerms = [];
  const preconditions = [];
  let hasGuardedExpression = false;
  const unresolved = [];
  for (const op of ops || []) {
    if (!isOnnxMacBearingOperation(op?.name, op?.standard_domain !== false)) continue;
    if (dynamicByIndex.has(Number(op.index))) {
      const row = dynamicByIndex.get(Number(op.index));
      if (row._polynomial) {
        terms.push(row._polynomial);
        expressionTerms.push(polynomialExpressionIr(row._polynomial));
      } else {
        hasGuardedExpression = true;
        expressionTerms.push(row._guarded.expression_ir);
        preconditions.push(...row._guarded.preconditions);
      }
      continue;
    }
    const exactMacs = String(op.macs_decimal ?? (Number.isSafeInteger(Number(op.macs)) ? op.macs : ""));
    if (op.macs_status !== "assessed" || !/^(?:0|[1-9]\d*)$/.test(exactMacs)) {
      const formulaAttempt = onnxOpMacFormula(op, tensorByIndex, registry);
      if (formulaAttempt.guarded) {
        hasGuardedExpression = true;
        expressionTerms.push(formulaAttempt.guarded.expression_ir);
        preconditions.push(...formulaAttempt.guarded.preconditions);
        continue;
      }
      unresolved.push({
        op_index: Number(op.index),
        op_name: op.name || "UNKNOWN",
        reason: formulaAttempt.polynomial
          ? op.macs_reason || "static MAC value is not an exact nonnegative integer"
          : formulaAttempt.reason || op.macs_reason || "static MAC value is not an exact nonnegative integer",
        resolution_class: formulaAttempt.resolution_class || (/rank is not fully serialized|require a bound rank contract/i.test(formulaAttempt.reason || "")
          ? "external_binding_required" : "analyzer_or_contract_residual"),
        blocking_tensor_role: formulaAttempt.blocking_tensor_role || null,
        blocking_tensor_index: formulaAttempt.blocking_tensor_index ?? null,
        blocking_tensor_name: formulaAttempt.blocking_tensor_name || "",
        root_conflict: formulaAttempt.root_conflict || null,
      });
      continue;
    }
    const exactTerm = normalizePolynomial([{ coefficient: BigInt(exactMacs), factors: [] }]);
    terms.push(exactTerm);
    expressionTerms.push(integerConstant(exactMacs));
  }
  if (unresolved.length) return { polynomial: null, guarded: null, unresolved };
  if (hasGuardedExpression) return {
    polynomial: null,
    guarded: {
      expression_ir: integerCall("add", expressionTerms),
      preconditions,
      method: "Sum of every exact static MAC count, symbolic polynomial, and guarded integer expression after all source-defined preconditions are satisfied.",
    },
    unresolved,
  };
  return { polynomial: addPolynomials(...terms), guarded: null, unresolved };
}

function groupOpIssues(issues) {
  const grouped = new Map();
  for (const issue of issues || []) {
    const opIndex = Number(issue?.op_index);
    const opName = issue?.op_name || "UNKNOWN";
    const key = `${opIndex}:${opName}`;
    const row = grouped.get(key) || {
      op_index: opIndex, op_name: opName, reasons: new Set(), resolution_classes: new Set(), blocking_tensors: new Map(), root_conflicts: new Map(),
    };
    if (issue?.reason) row.reasons.add(String(issue.reason));
    row.resolution_classes.add(String(issue?.resolution_class || "analyzer_or_contract_residual"));
    if (issue?.blocking_tensor_name) row.blocking_tensors.set(`${issue.blocking_tensor_role}:${issue.blocking_tensor_index}:${issue.blocking_tensor_name}`, {
      role: issue.blocking_tensor_role || "unknown",
      index: issue.blocking_tensor_index ?? null,
      name: issue.blocking_tensor_name,
    });
    if (issue?.root_conflict) row.root_conflicts.set(JSON.stringify(issue.root_conflict), issue.root_conflict);
    grouped.set(key, row);
  }
  return [...grouped.values()]
    .sort((left, right) => left.op_index - right.op_index || left.op_name.localeCompare(right.op_name))
    .map((row) => ({
      op_index: row.op_index,
      op_name: row.op_name,
      reason: [...row.reasons].sort().join("; ") || "exact symbolic formula unavailable",
      resolution_class: row.resolution_classes.size === 1 ? [...row.resolution_classes][0] : "mixed",
      blocking_tensors: [...row.blocking_tensors.values()],
      root_conflicts: [...row.root_conflicts.values()],
    }));
}

function buildLivenessContract(tensors, ops, registry) {
  const tensorByIndex = new Map((tensors || []).map((tensor, index) => [tensorIndex(tensor, index), tensor]));
  const producer = new Map();
  const lastUse = new Map();
  for (const tensor of tensors || []) if (tensor?.role === "input") producer.set(tensorIndex(tensor), -1);
  for (const op of ops || []) {
    for (const index of op.outputs || []) if (Number(index) >= 0) producer.set(Number(index), Number(op.index));
    for (const index of op.inputs || []) {
      const tensor = tensorAt(tensorByIndex, Number(index));
      if (tensor && tensor.constant_buffer !== true) lastUse.set(Number(index), Math.max(lastUse.get(Number(index)) ?? -1, Number(op.index)));
    }
  }
  for (const tensor of tensors || []) {
    if (tensor?.role === "output") lastUse.set(tensorIndex(tensor), Math.max(lastUse.get(tensorIndex(tensor)) ?? -1, (ops || []).length));
  }
  const payloadByTensor = new Map();
  for (const tensor of tensors || []) payloadByTensor.set(tensorIndex(tensor), tensorPayloadFormula(tensor, registry));
  const grouped = new Map();
  let unresolvedCandidateCount = 0;
  for (let point = 0; point <= (ops || []).length; point += 1) {
    const polynomials = [];
    const unresolved = [];
    for (const tensor of tensors || []) {
      const index = tensorIndex(tensor);
      if (tensor.constant_buffer === true || !producer.has(index) || !lastUse.has(index)) continue;
      if (producer.get(index) > point || lastUse.get(index) < point) continue;
      const payload = payloadByTensor.get(index);
      if (payload?.bytes) polynomials.push(payload.bytes);
      else unresolved.push({ tensor_index: index, tensor_name: tensor.name || `T${index}`, reason: payload?.status || "payload_formula_unavailable" });
    }
    if (unresolved.length) {
      unresolvedCandidateCount += 1;
      continue;
    }
    const polynomial = addPolynomials(...polynomials);
    const key = formulaKey(polynomial);
    const row = grouped.get(key) || { polynomial, first_program_point: point, last_program_point: point, occurrence_count: 0 };
    row.last_program_point = point;
    row.occurrence_count += 1;
    grouped.set(key, row);
  }
  const candidates = [...grouped.values()].sort((left, right) => {
    const leftKey = formulaKey(left.polynomial);
    const rightKey = formulaKey(right.polynomial);
    return leftKey < rightKey ? -1 : leftKey > rightKey ? 1 : 0;
  });
  let dominant = null;
  if (unresolvedCandidateCount === 0) dominant = candidates.find((candidate) => candidates.every((other) => polynomialDominates(candidate.polynomial, other.polynomial))) || null;
  return {
    status: unresolvedCandidateCount ? candidates.length ? "partial" : "not_assessed" : "assessed",
    candidate_program_point_count: (ops || []).length + 1,
    exact_candidate_program_point_count: (ops || []).length + 1 - unresolvedCandidateCount,
    unresolved_candidate_program_point_count: unresolvedCandidateCount,
    distinct_exact_formula_count: candidates.length,
    candidates: candidates.map((candidate) => ({
      first_program_point: candidate.first_program_point,
      last_program_point: candidate.last_program_point,
      occurrence_count: candidate.occurrence_count,
      live_payload_formula: serializeFormula(candidate.polynomial, "bytes", "sum of exact live dense-tensor payload polynomials at this program point"),
    })),
    peak_selection_status: unresolvedCandidateCount
      ? "not_assessed_incomplete_payload_formula_coverage"
      : dominant ? "exact_by_nonnegative_coefficient_dominance" : "requires_runtime_dimension_binding",
    peak_live_payload_formula: dominant
      ? serializeFormula(dominant.polynomial, "bytes", "one live-set polynomial coefficient-wise dominates every candidate for all non-negative symbol assignments")
      : null,
    peak_live_payload_max_formula: unresolvedCandidateCount || !candidates.length ? null : {
      status: "exact_symbolic_max_of_integer_polynomials",
      expression: `max(${candidates.map((candidate) => polynomialExpression(candidate.polynomial)).join(",")})`,
      candidates: candidates.map((candidate) => serializeFormula(candidate.polynomial, "bytes", "exact live-set candidate")),
    },
    interpretation_boundary: "Tensor lifetimes are graph-derived. Complete candidate coverage preserves the exact peak as max(P1,...,Pn). A single polynomial is emitted only when coefficient-wise dominance proves it for every non-negative symbol assignment; otherwise numeric peak bytes require symbol binding.",
  };
}

export function buildOnnxDynamicShapeCostContract(tensors, ops) {
  const dynamicTensors = (tensors || []).filter(dynamicShape);
  if (!dynamicTensors.length) {
    return {
      schema: DYNAMIC_SHAPE_COST_SCHEMA,
      status: "not_applicable_static_shapes",
      evidence_class: "DERIVED",
      format: "onnx",
      source: ONNX_DYNAMIC_COST_SOURCE,
      dynamic_tensor_count: 0,
      symbol_count: 0,
      symbols: [],
      tensor_formula_count: 0,
      tensor_formulas: [],
      dynamic_compute_op_count: 0,
      op_formula_count: 0,
      unresolved_dynamic_compute_op_count: 0,
      unresolved_dynamic_compute_ops: [],
      total_macs_unresolved_op_count: 0,
      total_macs_unresolved_ops: [],
      total_macs_artifact_contract_conflict_op_count: 0,
      total_macs_external_binding_required_op_count: 0,
      total_macs_analyzer_or_contract_residual_op_count: 0,
      op_formulas: [],
      total_macs_formula: null,
      total_macs_formula_status: "not_applicable_static_shapes",
      liveness: { status: "not_applicable_static_shapes", candidates: [], peak_selection_status: "not_applicable_static_shapes", peak_live_payload_formula: null, peak_live_payload_max_formula: null },
      arena_projection_status: "not_applicable_static_shapes",
      dimension_bounds_status: "not_applicable_static_shapes",
      method: "No tensor carries an unknown dimension after ONNX shape inference.",
      interpretation_boundary: "Static-shape costs remain in the ordinary MAC, payload, liveness, and memory sections.",
    };
  }
  const registry = symbolRegistry();
  const tensorByIndex = new Map((tensors || []).map((tensor, index) => [tensorIndex(tensor, index), tensor]));
  const tensorRows = buildTensorRows(tensors, registry);
  const opResult = buildOpRows(ops, tensorByIndex, registry);
  const totalResult = buildTotalMacFormula(ops, opResult.rows, tensorByIndex, registry);
  const unresolvedDynamicOps = groupOpIssues(opResult.unresolved);
  const totalMacUnresolvedOps = groupOpIssues([...opResult.unresolved, ...totalResult.unresolved]);
  const liveness = buildLivenessContract(tensors, ops, registry);
  const publicOpRows = opResult.rows.map(({ _polynomial, _guarded, ...row }) => row);
  const partial = unresolvedDynamicOps.length > 0
    || totalMacUnresolvedOps.length > 0
    || tensorRows.some((row) => !row.formula_status.startsWith("exact_"))
    || liveness.status !== "assessed";
  return {
    schema: DYNAMIC_SHAPE_COST_SCHEMA,
    status: partial ? "partial" : "assessed",
    evidence_class: "DERIVED",
    format: "onnx",
    source: ONNX_DYNAMIC_COST_SOURCE,
    dynamic_tensor_count: dynamicTensors.length,
    symbol_count: registry.rows.length,
    symbols: registry.rows,
    tensor_formula_count: tensorRows.length,
    tensor_formulas: tensorRows,
    dynamic_compute_op_count: publicOpRows.length + unresolvedDynamicOps.length,
    op_formula_count: publicOpRows.length,
    unresolved_dynamic_compute_op_count: unresolvedDynamicOps.length,
    unresolved_dynamic_compute_ops: unresolvedDynamicOps,
    total_macs_unresolved_op_count: totalMacUnresolvedOps.length,
    total_macs_unresolved_ops: totalMacUnresolvedOps,
    total_macs_artifact_contract_conflict_op_count: totalMacUnresolvedOps.filter((row) => row.resolution_class === "artifact_contract_conflict").length,
    total_macs_external_binding_required_op_count: totalMacUnresolvedOps.filter((row) => row.resolution_class === "external_binding_required").length,
    total_macs_analyzer_or_contract_residual_op_count: totalMacUnresolvedOps.filter((row) => !["artifact_contract_conflict", "external_binding_required"].includes(row.resolution_class)).length,
    op_formulas: publicOpRows,
    total_macs_formula: totalResult.polynomial
      ? serializeFormula(totalResult.polynomial, "MACs", "sum of every static exact MAC term and every exact dynamic compute-op polynomial")
      : totalResult.guarded
        ? serializeGuardedIntegerFormula(totalResult.guarded.expression_ir, "MACs", totalResult.guarded.method, totalResult.guarded.preconditions)
        : null,
    total_macs_formula_status: totalResult.polynomial
      ? "exact_symbolic_integer_polynomial"
      : totalResult.guarded ? "exact_guarded_integer_expression" : "not_assessed_incomplete_compute_formula_coverage",
    liveness,
    arena_projection_status: "not_assessed_requires_runtime_dimension_binding_and_allocator_execution",
    dimension_bounds_status: registry.rows.some((row) => row.bounds_status !== "not_embedded_in_artifact")
      ? registry.rows.every((row) => row.bounds_status !== "not_embedded_in_artifact") ? "artifact_derived_complete" : "artifact_derived_partial"
      : "not_embedded_in_artifact",
    method: "Unknown ONNX dimensions become explicit symbols; repeated artifact dim_param strings share one symbol, analyzer-derived deepbom_expr dimensions retain expression identity, and source-defined runtime cardinalities such as NonZero NNZ retain a separate runtime-value provenance plus any exact artifact-derived bounds. Tensor payload and source-backed Conv, Attention, DeformConv, two-input Einsum, Gemm, and MatMul-family MACs remain exact non-negative integer polynomials. Cropped or rank-omitted ConvTranspose uses source-defined guarded integer expressions with exact division and contributing-pair primitives; no output-volume approximation or numeric bound is invented.",
    interpretation_boundary: "The formulas become numeric only after every symbol is bound to a runtime dimension satisfying the graph contract. They do not prove ORT execution-provider assignment, fusion, allocation, copy materialization, latency, or task accuracy.",
  };
}

export function evaluateDynamicIntegerFormula(formula, assignments) {
  if (formula?.status === "exact_guarded_integer_expression") return evaluateGuardedIntegerFormula(formula, assignments);
  if (!formula || formula.status !== "exact_symbolic_integer_polynomial") return null;
  let total = 0n;
  for (const term of formula.terms || []) {
    let value = BigInt(term.coefficient_decimal);
    for (const factor of term.factors || []) {
      const assigned = assignments instanceof Map ? assignments.get(factor.symbol_id) : assignments?.[factor.symbol_id];
      const integer = typeof assigned === "bigint" ? assigned : Number.isSafeInteger(Number(assigned)) ? BigInt(Number(assigned)) : null;
      if (integer == null || integer < 0n) return null;
      value *= integer ** BigInt(Number(factor.exponent));
    }
    total += value;
  }
  return total;
}

function tfliteDynamicAxes(tensor) {
  const signature = Array.isArray(tensor?.shape_signature) && tensor.shape_signature.length
    ? tensor.shape_signature
    : tensor?.shape;
  return (Array.isArray(signature) ? signature : [])
    .map((dimension, axis) => (Number(dimension) < 0 ? axis : -1))
    .filter((axis) => axis >= 0);
}

function safeBigIntNumber(value) {
  if (typeof value !== "bigint" || value > BigInt(Number.MAX_SAFE_INTEGER)) return null;
  return Number(value);
}

export function deriveTfliteBatchOneProjection(analysis) {
  const contract = analysis?.dynamic_shape_cost_contract;
  if (String(analysis?.format || "tflite").toLowerCase() !== "tflite"
    || !contract
    || contract.status === "not_applicable_static_shapes") {
    return {
      status: "not_applicable",
      evidence_class: "NOT_APPLICABLE",
      reason: "No TFLite dynamic-shape contract is present.",
    };
  }

  const inputs = Array.isArray(analysis?.inputs) ? analysis.inputs : [];
  const externalAxes = inputs.flatMap((tensor, inputOrdinal) => (
    tfliteDynamicAxes(tensor).map((axis) => ({
      input_ordinal: inputOrdinal,
      tensor_index: Number(tensor?.index ?? inputOrdinal),
      axis,
      serialized_value: Number(tensor?.shape?.[axis]),
    }))
  ));
  const nonBatchExternalAxes = externalAxes.filter((row) => row.axis !== 0);
  if (!externalAxes.length || nonBatchExternalAxes.length) {
    return {
      status: "requires_explicit_shape_binding",
      evidence_class: "DERIVED",
      external_dynamic_axis_count: externalAxes.length,
      non_batch_dynamic_axis_count: nonBatchExternalAxes.length,
      reason: nonBatchExternalAxes.length
        ? "At least one graph-input spatial or feature axis is dynamic."
        : "No dynamic graph-input batch axis was identified.",
    };
  }
  if (externalAxes.some((row) => row.serialized_value !== 1)) {
    return {
      status: "requires_explicit_shape_binding",
      evidence_class: "DERIVED",
      external_dynamic_axis_count: externalAxes.length,
      non_batch_dynamic_axis_count: 0,
      reason: "The serialized graph-input projection does not bind every dynamic batch axis to 1.",
    };
  }

  const tensorByIndex = new Map((analysis?.tensors || []).map((tensor, index) => [
    Number.isInteger(Number(tensor?.index)) ? Number(tensor.index) : index,
    tensor,
  ]));
  const formulaByIndex = new Map((contract.tensor_formulas || []).map((row) => [
    Number(row.tensor_index),
    row,
  ]));
  const occurrences = (contract.symbols || []).flatMap((symbol) => (
    (symbol.occurrences || []).map((occurrence) => ({ symbol, occurrence }))
  ));
  const invalidOccurrence = occurrences.find(({ occurrence }) => {
    if (Number(occurrence.axis) !== 0) return true;
    const tensor = tensorByIndex.get(Number(occurrence.tensor_index));
    const formula = formulaByIndex.get(Number(occurrence.tensor_index));
    return Number(tensor?.shape?.[0] ?? formula?.shape?.[0]) !== 1;
  });
  if (invalidOccurrence || !occurrences.length) {
    return {
      status: "requires_explicit_shape_binding",
      evidence_class: "DERIVED",
      external_dynamic_axis_count: externalAxes.length,
      non_batch_dynamic_axis_count: invalidOccurrence && Number(invalidOccurrence.occurrence.axis) !== 0 ? 1 : 0,
      reason: invalidOccurrence
        ? "An internal dynamic occurrence is not a serialized batch=1 axis."
        : "The dynamic contract does not expose symbol occurrences needed to verify the projection.",
    };
  }

  const assignments = Object.fromEntries((contract.symbols || []).map((symbol) => [symbol.symbol_id, 1]));
  const projectedMacs = evaluateDynamicIntegerFormula(contract.total_macs_formula, assignments);
  const projectedPeakBytes = evaluateDynamicIntegerFormula(
    contract.liveness?.peak_live_payload_formula,
    assignments,
  );
  const numericMacs = safeBigIntNumber(projectedMacs);
  const numericPeakBytes = safeBigIntNumber(projectedPeakBytes);
  const serializedMacs = Number(analysis?.total_macs);
  const projectionMatchesSerializedMacs = numericMacs != null
    && Number.isFinite(serializedMacs)
    && numericMacs === serializedMacs;

  return {
    status: numericMacs == null
      ? "batch_one_projection_formula_incomplete"
      : "assumption_bound_batch_one",
    evidence_class: "ASSUMPTION_BOUND",
    assumption: "Every dynamic graph-input batch axis and every propagated internal batch symbol is bound to the serialized value N=1.",
    batch_size: 1,
    external_dynamic_axis_count: externalAxes.length,
    non_batch_dynamic_axis_count: 0,
    internal_symbol_count: Number(contract.symbol_count || 0),
    dynamic_tensor_count: Number(contract.dynamic_tensor_count || 0),
    projected_total_macs: numericMacs,
    projected_total_macs_decimal: projectedMacs?.toString() || null,
    projected_peak_live_payload_bytes: numericPeakBytes,
    projected_peak_live_payload_bytes_decimal: projectedPeakBytes?.toString() || null,
    projection_matches_serialized_total_macs: projectionMatchesSerializedMacs,
    interpretation_boundary: "This is an exact evaluation of emitted integer polynomials at the artifact's serialized N=1 projection. It is not an approved N>1 capacity bound, runtime allocation measurement, or latency guarantee.",
  };
}
