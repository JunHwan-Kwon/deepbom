const INTEGER_TEXT = /^-?(?:0|[1-9][0-9]*)$/;

export const GUARDED_INTEGER_EXPRESSION_SCHEMA = "deepbom.guarded_integer_expression.v2";

export function integerConstant(value) {
  return { kind: "constant", value_decimal: BigInt(value).toString() };
}

export function integerSymbol(symbolId) {
  return { kind: "symbol", symbol_id: String(symbolId) };
}

export function integerCall(operator, ...arguments_) {
  const argumentsFlat = arguments_.flat().filter(Boolean);
  if (["add", "mul"].includes(operator)) {
    const flattened = argumentsFlat.flatMap((item) => item?.kind === "call" && item.operator === operator ? item.arguments : [item]);
    if (operator === "add") {
      const kept = flattened.filter((item) => !(item.kind === "constant" && item.value_decimal === "0"));
      return kept.length === 0 ? integerConstant(0) : kept.length === 1 ? kept[0] : { kind: "call", operator, arguments: kept };
    }
    if (flattened.some((item) => item.kind === "constant" && item.value_decimal === "0")) return integerConstant(0);
    const kept = flattened.filter((item) => !(item.kind === "constant" && item.value_decimal === "1"));
    return kept.length === 0 ? integerConstant(1) : kept.length === 1 ? kept[0] : { kind: "call", operator, arguments: kept };
  }
  return { kind: "call", operator, arguments: argumentsFlat };
}

export function nonnegativeGuard(expressionIr, label) {
  return { kind: "nonnegative", expression_ir: expressionIr, label: String(label || "integer expression") };
}

export function divisibleGuard(numeratorIr, denominatorIr, label) {
  return { kind: "divisible", numerator_ir: numeratorIr, denominator_ir: denominatorIr, label: String(label || "integer quotient") };
}

export function equalityGuard(leftIr, rightIr, label) {
  return { kind: "equal", left_ir: leftIr, right_ir: rightIr, label: String(label || "integer equality") };
}

export function integerExpressionText(node) {
  if (node?.kind === "constant") return String(node.value_decimal);
  if (node?.kind === "symbol") return String(node.symbol_id);
  if (node?.kind !== "call") return "?";
  const values = (node.arguments || []).map(integerExpressionText);
  if (node.operator === "add") return `(${values.join(" + ")})`;
  if (node.operator === "sub") return `(${values.join(" - ")})`;
  if (node.operator === "mul") return values.join("*");
  if (node.operator === "exact_div") return `exact_div(${values.join(",")})`;
  if (node.operator === "conv_transpose_pairs") return `conv_transpose_pairs(${values.join(",")})`;
  return `${node.operator}(${values.join(",")})`;
}

export function integerExpressionSymbolIds(node) {
  const symbols = new Set();
  (function visit(item) {
    if (item?.kind === "symbol") symbols.add(String(item.symbol_id));
    for (const child of item?.arguments || []) visit(child);
  }(node));
  return [...symbols].sort();
}

export function serializeGuardedIntegerFormula(expressionIr, unit, method, preconditions = []) {
  return {
    status: "exact_guarded_integer_expression",
    ir_schema: GUARDED_INTEGER_EXPRESSION_SCHEMA,
    unit,
    expression: integerExpressionText(expressionIr),
    expression_ir: expressionIr,
    symbol_ids: integerExpressionSymbolIds(expressionIr),
    preconditions: preconditions.map(serializeGuard),
    method,
  };
}

function serializeGuard(guard) {
  if (guard?.kind === "nonnegative") return { ...guard, expression: integerExpressionText(guard.expression_ir) };
  if (guard?.kind === "divisible") return {
    ...guard,
    numerator: integerExpressionText(guard.numerator_ir),
    denominator: integerExpressionText(guard.denominator_ir),
  };
  if (guard?.kind === "equal") return {
    ...guard,
    left: integerExpressionText(guard.left_ir),
    right: integerExpressionText(guard.right_ir),
  };
  return guard;
}

export function evaluateGuardedIntegerFormula(formula, assignments) {
  if (!formula || formula.status !== "exact_guarded_integer_expression"
    || formula.ir_schema !== GUARDED_INTEGER_EXPRESSION_SCHEMA) return null;
  const values = assignments instanceof Map ? assignments : new Map(Object.entries(assignments || {}));
  for (const guard of formula.preconditions || []) if (!evaluateGuard(guard, values)) return null;
  const result = evaluateIntegerExpression(formula.expression_ir, values);
  return typeof result === "bigint" && result >= 0n ? result : null;
}

export function evaluateIntegerExpression(node, assignments) {
  const values = assignments instanceof Map ? assignments : new Map(Object.entries(assignments || {}));
  if (node?.kind === "constant") return INTEGER_TEXT.test(String(node.value_decimal || "")) ? BigInt(node.value_decimal) : null;
  if (node?.kind === "symbol") {
    const assigned = values.get(node.symbol_id);
    const integer = typeof assigned === "bigint" ? assigned : Number.isSafeInteger(Number(assigned)) ? BigInt(Number(assigned)) : null;
    return integer != null && integer >= 0n ? integer : null;
  }
  if (node?.kind !== "call") return null;
  const args = (node.arguments || []).map((item) => evaluateIntegerExpression(item, values));
  if (args.some((item) => item == null)) return null;
  if (node.operator === "add") return args.reduce((sum, item) => sum + item, 0n);
  if (node.operator === "sub") return args.length === 2 ? args[0] - args[1] : null;
  if (node.operator === "mul") return args.reduce((product, item) => product * item, 1n);
  if (node.operator === "exact_div") return args.length === 2 && args[1] > 0n && args[0] % args[1] === 0n ? args[0] / args[1] : null;
  if (node.operator === "conv_transpose_pairs") return convTransposeAxisPairs(...args);
  return null;
}

function evaluateGuard(guard, assignments) {
  if (guard?.kind === "nonnegative") {
    const value = evaluateIntegerExpression(guard.expression_ir, assignments);
    return value != null && value >= 0n;
  }
  if (guard?.kind === "divisible") {
    const numerator = evaluateIntegerExpression(guard.numerator_ir, assignments);
    const denominator = evaluateIntegerExpression(guard.denominator_ir, assignments);
    return numerator != null && denominator != null && numerator >= 0n && denominator > 0n && numerator % denominator === 0n;
  }
  if (guard?.kind === "equal") {
    const left = evaluateIntegerExpression(guard.left_ir, assignments);
    const right = evaluateIntegerExpression(guard.right_ir, assignments);
    return left != null && right != null && left === right;
  }
  return false;
}

function floorDiv(value, divisor) {
  let quotient = value / divisor;
  if (value % divisor < 0n) quotient -= 1n;
  return quotient;
}

export function convTransposeAxisPairs(input, kernel, stride, dilation, padStart, output) {
  if ([input, padStart, output].some((value) => value < 0n)
    || [kernel, stride, dilation].some((value) => value <= 0n)) return null;
  const loopCount = kernel < input ? kernel : input;
  if (loopCount > 1_000_000n) return null;
  let total = 0n;
  if (kernel <= input) {
    for (let k = 0n; k < kernel; k += 1n) {
      const low = maxBigInt(-1n, floorDiv(padStart - k * dilation - 1n, stride)) + 1n;
      const high = minBigInt(input - 1n, floorDiv(output - 1n + padStart - k * dilation, stride));
      if (high >= low) total += high - low + 1n;
    }
  } else {
    for (let i = 0n; i < input; i += 1n) {
      const low = maxBigInt(-1n, floorDiv(padStart - i * stride - 1n, dilation)) + 1n;
      const high = minBigInt(kernel - 1n, floorDiv(output - 1n + padStart - i * stride, dilation));
      if (high >= low) total += high - low + 1n;
    }
  }
  return total;
}

function maxBigInt(left, right) { return left > right ? left : right; }
function minBigInt(left, right) { return left < right ? left : right; }
