import { portableContract } from "./portable-contracts.js";
const OPERATORS = new Set(["add", "sub", "mul", "div", "floor_div", "ceil_div", "max", "min", "broadcast_dim", "reshape_quotient", "range_len", "slice_len"]);

export function parseOnnxDimensionExpression(value) {
  try { return parseToken(String(value || "").trim()); } catch { return null; }
}

function parseToken(token) {
  if (!token || token === "?") throw new Error("unknown dimension token");
  if (token === "neg_inf" || token === "pos_inf") return { kind: "infinity", sign: token === "neg_inf" ? -1 : 1 };
  if (/^(?:v:|i64:)?-?\d+$/.test(token)) return { kind: "constant", value_decimal: token.replace(/^(?:v:|i64:)/, "") };
  if (token.startsWith("s:")) {
    const name = token.slice(2);
    return name.startsWith("deepbom_expr:") ? parseToken(name) : { kind: "symbol", name };
  }
  const expression = token.replace(/^deepbom_expr:/, "");
  const open = expression.indexOf("(");
  if (open < 1 || !expression.endsWith(")")) return { kind: "symbol", name: expression };
  const operator = expression.slice(0, open);
  if (!OPERATORS.has(operator)) throw new Error("unsupported dimension operator");
  return { kind: "call", operator, arguments: splitArguments(expression.slice(open + 1, -1)).map(parseToken) };
}

function splitArguments(value) {
  const rows = []; let depth = 0; let start = 0;
  for (let index = 0; index < value.length; index += 1) {
    if (value[index] === "(") depth += 1;
    else if (value[index] === ")") depth -= 1;
    else if (value[index] === "," && depth === 0) { rows.push(value.slice(start, index).trim()); start = index + 1; }
    if (depth < 0) throw new Error("invalid dimension expression");
  }
  if (depth !== 0) throw new Error("invalid dimension expression");
  rows.push(value.slice(start).trim());
  if (rows.some((row) => !row)) throw new Error("empty dimension argument");
  return rows;
}

export function onnxDimensionExpressionDependencies(expression) {
  const names = new Set();
  (function visit(node) {
    if (node?.kind === "symbol") names.add(node.name);
    for (const child of node?.arguments || []) visit(child);
  }(expression));
  return [...names].sort();
}

export function evaluateOnnxDimensionExpression(expression, assignments) {
  const rows = assignments instanceof Map ? [...assignments] : Object.entries(assignments || {});
  const normalized = Object.fromEntries(rows.map(([key, value]) => [key,
    typeof value === "bigint" ? String(value) : value != null && typeof value !== "boolean" && Number.isSafeInteger(Number(value)) ? String(Number(value)) : null]));
  const result = portableContract("dimension_expression", { expression, assignments: normalized });
  return result == null ? null : BigInt(result);
}
