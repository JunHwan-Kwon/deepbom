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
  const value = evaluate(expression, assignments instanceof Map ? assignments : new Map(Object.entries(assignments || {})));
  return typeof value === "bigint" && value >= 0n ? value : null;
}

function evaluate(node, assignments) {
  if (node?.kind === "constant") return BigInt(node.value_decimal);
  if (node?.kind === "infinity") return node.sign < 0 ? "-inf" : "+inf";
  if (node?.kind === "symbol") {
    const value = assignments.get(node.name);
    return typeof value === "bigint" ? value : Number.isSafeInteger(Number(value)) ? BigInt(Number(value)) : null;
  }
  if (node?.kind !== "call") return null;
  const values = node.arguments.map((argument) => evaluate(argument, assignments));
  if (values.some((value) => value == null)) return null;
  const finite = values.every((value) => typeof value === "bigint");
  if (node.operator === "slice_len") return sliceLength(values);
  if (!finite) return null;
  const [left, right] = values;
  if (node.operator === "add") return values.reduce((sum, value) => sum + value, 0n);
  if (node.operator === "sub") return left - right;
  if (node.operator === "mul") return values.reduce((product, value) => product * value, 1n);
  if (["div", "reshape_quotient"].includes(node.operator)) return right === 0n || node.operator === "reshape_quotient" && left % right !== 0n ? null : left / right;
  if (node.operator === "floor_div") return right > 0n ? floorDiv(left, right) : null;
  if (node.operator === "ceil_div") return right > 0n ? -floorDiv(-left, right) : null;
  if (node.operator === "max") return values.reduce((value, item) => value > item ? value : item);
  if (node.operator === "min") return values.reduce((value, item) => value < item ? value : item);
  if (node.operator === "broadcast_dim") {
    if (values.some((value) => value < 0n)) return null;
    const nonOne = [...new Set(values.filter((value) => value !== 1n).map(String))].map(BigInt);
    return nonOne.length <= 1 ? nonOne[0] ?? 1n : null;
  }
  if (node.operator === "range_len") {
    const [start, limit, delta] = values;
    if (delta === 0n) return null;
    const count = delta > 0n ? -floorDiv(start - limit, delta) : -floorDiv(limit - start, -delta);
    return count > 0n ? count : 0n;
  }
  return null;
}

function floorDiv(value, divisor) { let q = value / divisor; if (value % divisor < 0n) q -= 1n; return q; }

function sliceLength(values) {
  if (values.length !== 4 || typeof values[0] !== "bigint" || typeof values[3] !== "bigint" || values[0] < 0n || values[3] === 0n) return null;
  const [size, rawStart, rawEnd, step] = values;
  let start = rawStart; let end = rawEnd;
  if (step > 0n) {
    start = start === "-inf" ? 0n : start === "+inf" ? size : start < 0n ? start + size : start;
    end = end === "+inf" ? size : end === "-inf" ? 0n : end < 0n ? end + size : end;
    start = start < 0n ? 0n : start > size ? size : start;
    end = end < 0n ? 0n : end > size ? size : end;
    return end <= start ? 0n : -floorDiv(start - end, step);
  }
  start = start === "+inf" ? size - 1n : start === "-inf" ? -1n : start < 0n ? start + size : start;
  end = end === "-inf" ? -1n : end === "+inf" ? size - 1n : end < 0n ? end + size : end;
  start = start < -1n ? -1n : start >= size ? size - 1n : start;
  end = end < -1n ? -1n : end >= size ? size - 1n : end;
  return start <= end ? 0n : -floorDiv(end - start, -step);
}
