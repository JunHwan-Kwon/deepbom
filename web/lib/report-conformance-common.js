export function compactMlBomEvidencePointerValid(document) {
  const properties = [...(document?.metadata?.component?.properties || []), ...(document?.properties || [])];
  const pointer = properties.find((item) => item?.name === "deepbom:compatibility:detailLocation")?.value;
  const reference = (document?.metadata?.component?.externalReferences || []).find((item) => item?.url === "engineering_evidence.json");
  return pointer === "engineering_evidence.json#/evidence/static_analysis" && reference?.type === "evidence";
}

export function formatIntegerForConformance(value) {
  return Number(value || 0).toLocaleString("en-US");
}

export function markdownCellForConformance(value) {
  return String(value ?? "-")
    .replace(/\r?\n/g, "<br>")
    .replace(/\|/g, "\\|");
}

export function validateStaticSignedZeroLedger(tensors = []) {
  return tensors.every((tensor) => {
    const values = tensor?.static_values;
    const indices = tensor?.static_values_negative_zero_indices;
    const count = tensor?.static_values_negative_zero_count;
    if (!Array.isArray(values) || !Array.isArray(indices) || !Number.isSafeInteger(count) || count < 0 || count !== indices.length) return false;
    if (values.some((value) => Object.is(value, -0))) return false;
    const unique = new Set();
    return indices.every((index) => Number.isSafeInteger(index) && index >= 0 && index < values.length
      && !unique.has(index) && unique.add(index) && values[index] === 0);
  });
}

export function validateStaticCanonicalTextLedger(tensors = []) {
  const canonicalNumber = /^(?:-?(?:0|[1-9]\d*)(?:\.\d+)?(?:e[+-]?\d+)?|NaN|Infinity|-Infinity|-0)$/i;
  return tensors.every((tensor) => {
    const complete = tensor?.static_values_canonical_text_complete === true;
    const texts = tensor?.static_values_canonical_texts;
    if (!Array.isArray(texts)) return false;
    if (!complete) return texts.length === 0;
    return tensor.static_values_complete === false
      && Array.isArray(tensor.static_values) && tensor.static_values.length === 0
      && Number.isSafeInteger(tensor.initializer_elements) && tensor.initializer_elements === texts.length
      && texts.every((value) => typeof value === "string" && canonicalNumber.test(value));
  });
}
