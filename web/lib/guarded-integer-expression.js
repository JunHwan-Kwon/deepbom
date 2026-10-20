import { portableContract } from "./portable-contracts.js";


export const GUARDED_INTEGER_EXPRESSION_SCHEMA = "deepbom.guarded_integer_expression.v2";

export function parseNonnegativeInteger(value) {
  if (typeof value === "bigint") return value >= 0n ? value : null;
  if (typeof value === "number") return Number.isSafeInteger(value) && value >= 0 ? BigInt(value) : null;
  if (typeof value === "string" && /^(?:0|[1-9][0-9]*)$/.test(value)) return BigInt(value);
  return null;
}

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

function normalizedAssignments(assignments) {
  const entries = assignments instanceof Map ? assignments.entries() : Object.entries(assignments || {});
  return Object.fromEntries([...entries].filter(([key]) => typeof key === "string").map(([key, value]) => [key, parseNonnegativeInteger(value)?.toString() ?? null]));
}
function evaluated(input) {
  const result = portableContract("integer_expression", input);
  return result === null ? null : BigInt(result);
}

export function evaluateGuardedIntegerFormula(formula, assignments) {
  if (!formula || formula.status !== "exact_guarded_integer_expression"
    || formula.ir_schema !== GUARDED_INTEGER_EXPRESSION_SCHEMA) return null;
  return evaluated({ formula, assignments: normalizedAssignments(assignments) });
}

export function evaluateIntegerExpression(node, assignments) {
  return evaluated({ expression: node ?? null, assignments: normalizedAssignments(assignments) });
}

export function convTransposeAxisPairs(input, kernel, stride, dilation, padStart, output) {
  const values = [input, kernel, stride, dilation, padStart, output];
  if (values.some(value => typeof value !== "bigint")) return null;
  return evaluated({ expression: { kind: "call", operator: "conv_transpose_pairs",
    arguments: values.map(integerConstant) }, assignments: {} });
}
