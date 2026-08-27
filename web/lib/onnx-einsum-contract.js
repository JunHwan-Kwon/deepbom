const LABEL = /^[A-Za-z]$/;

export function parseOnnxEinsumEquation(equation, ranks) {
  const source = String(equation || "").replace(/\s+/g, "");
  if (!source) return fail("einsum_equation_missing");
  const arrowParts = source.split("->");
  if (arrowParts.length > 2) return fail("einsum_equation_multiple_output_arrows");
  const inputTexts = arrowParts[0].split(",");
  if (inputTexts.length !== ranks.length) return fail("einsum_equation_input_cardinality_mismatch");
  const parsedInputs = [];
  let maximumEllipsisRank = 0;
  for (let index = 0; index < inputTexts.length; index += 1) {
    const parsed = parseSubscript(inputTexts[index]);
    if (!parsed) return fail(`einsum_input_subscript_invalid:${index}`);
    const rank = Number(ranks[index]);
    if (!Number.isSafeInteger(rank) || rank < 0) return fail(`einsum_input_rank_invalid:${index}`);
    const ellipsisRank = rank - parsed.labels.length;
    if (ellipsisRank < 0 || !parsed.hasEllipsis && ellipsisRank !== 0) return fail(`einsum_input_rank_subscript_mismatch:${index}`);
    maximumEllipsisRank = Math.max(maximumEllipsisRank, ellipsisRank);
    parsedInputs.push({ ...parsed, rank, ellipsisRank });
  }
  const ellipsisLabels = Array.from({ length: maximumEllipsisRank }, (_, index) => `@ellipsis:${index}`);
  const operands = parsedInputs.map((parsed) => expandInput(parsed, ellipsisLabels));
  const allLabels = new Set(operands.flat());
  let output = null;
  if (arrowParts.length === 2) {
    const parsed = parseSubscript(arrowParts[1], true);
    if (!parsed) return fail("einsum_output_subscript_invalid");
    output = [];
    for (const token of parsed.tokens) {
      if (token === "...") output.push(...ellipsisLabels);
      else output.push(token);
    }
    if (new Set(output).size !== output.length) return fail("einsum_output_label_duplicate");
    if (output.some((label) => !allLabels.has(label))) return fail("einsum_output_label_absent_from_inputs");
  } else {
    const counts = new Map();
    for (const labels of operands) for (const label of labels) if (!label.startsWith("@ellipsis:")) counts.set(label, (counts.get(label) || 0) + 1);
    output = [...ellipsisLabels, ...[...counts.entries()].filter(([, count]) => count === 1).map(([label]) => label).sort()];
  }
  return {
    status: "assessed",
    operands,
    output,
    ellipsis_labels: ellipsisLabels,
    reduction_labels: [...allLabels].filter((label) => !output.includes(label)).sort(),
    all_labels: [...allLabels].sort(),
    explicit_output: arrowParts.length === 2,
  };
}

function parseSubscript(text, output = false) {
  const tokens = [];
  let hasEllipsis = false;
  for (let index = 0; index < text.length;) {
    if (text.slice(index, index + 3) === "...") {
      if (hasEllipsis) return null;
      hasEllipsis = true;
      tokens.push("...");
      index += 3;
      continue;
    }
    const token = text[index];
    if (!LABEL.test(token)) return null;
    tokens.push(token);
    index += 1;
  }
  if (output && tokens.filter((token) => token === "...").length > 1) return null;
  return { tokens, labels: tokens.filter((token) => token !== "..."), hasEllipsis };
}

function expandInput(parsed, ellipsisLabels) {
  const selectedEllipsis = ellipsisLabels.slice(ellipsisLabels.length - parsed.ellipsisRank);
  const output = [];
  for (const token of parsed.tokens) {
    if (token === "...") output.push(...selectedEllipsis);
    else output.push(token);
  }
  return output;
}

function fail(reason) {
  return { status: "invalid", reason, operands: [], output: [], ellipsis_labels: [], reduction_labels: [], all_labels: [] };
}
