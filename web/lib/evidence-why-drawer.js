export const EVIDENCE_EXPLANATION_SCHEMA = "deepbom.evidence_explanation.v1";

const EVIDENCE_CLASSES = new Set(["OBSERVED", "DERIVED", "PREDICTED", "ESTIMATED", "MEASURED", "NOT_ASSESSABLE"]);

export function createEvidenceWhyDrawer({ root, closeButton, copyButton, body, title, subtitle } = {}) {
  let current = null;
  const close = () => {
    if (!root) return;
    root.hidden = true;
    root.setAttribute("aria-hidden", "true");
  };
  const open = (value) => {
    if (!root || !body) return;
    current = normalizeEvidenceExplanation(value);
    if (title) title.textContent = current.title;
    if (subtitle) subtitle.textContent = `${current.evidence_class} | ${current.value ?? "value not supplied"}`;
    body.replaceChildren(
      row("Evidence class", current.evidence_class),
      row("Value", current.value ?? "Not supplied"),
      listRow("Input fields / JSON Pointers", current.source_pointers, "Not supplied"),
      row("Calculation / method", current.method || "Not supplied"),
      row("Formula / denominator", current.formula || "Not supplied"),
      listRow("Source commits / rulepack hashes", current.source_pins, "Not applicable or not supplied"),
      listRow("Conditions", current.conditions, "None declared"),
      listRow("Does not establish", current.limitations, "No additional boundary supplied"),
      row("Report / CycloneDX pointer", current.report_pointer || "Not supplied"),
    );
    root.hidden = false;
    root.setAttribute("aria-hidden", "false");
    root.focus({ preventScroll: true });
  };
  closeButton?.addEventListener("click", close);
  root?.addEventListener("keydown", (event) => { if (event.key === "Escape") close(); });
  copyButton?.addEventListener("click", async () => {
    if (!current) return;
    await navigator.clipboard?.writeText(`${JSON.stringify(current, null, 2)}\n`);
    const previous = copyButton.textContent;
    copyButton.textContent = "Copied";
    setTimeout(() => { copyButton.textContent = previous; }, 900);
  });
  return { open, close, get: () => current };
}

export function normalizeEvidenceExplanation(value) {
  const source = value && typeof value === "object" ? value : {};
  let evidenceClass = String(source.evidence_class || "NOT_ASSESSABLE").toUpperCase().replaceAll("+", "_");
  if (!EVIDENCE_CLASSES.has(evidenceClass)) {
    evidenceClass = [...EVIDENCE_CLASSES].find((candidate) => evidenceClass.startsWith(candidate))
      || (evidenceClass.includes("DEVICE") || evidenceClass.includes("RUNTIME") ? "MEASURED"
        : evidenceClass.includes("STATIC") ? "DERIVED" : "NOT_ASSESSABLE");
  }
  return {
    schema: EVIDENCE_EXPLANATION_SCHEMA,
    title: text(source.title || "Evidence explanation", 240),
    value: scalar(source.value),
    evidence_class: evidenceClass,
    source_pointers: strings(source.source_pointers),
    method: optionalText(source.method, 4000),
    formula: optionalText(source.formula, 4000),
    source_pins: strings(source.source_pins),
    conditions: strings(source.conditions),
    limitations: strings(source.limitations),
    report_pointer: optionalText(source.report_pointer, 1000),
  };
}

function row(label, value) {
  const wrapper = document.createElement("section");
  const heading = document.createElement("h3");
  const content = document.createElement("p");
  heading.textContent = label;
  content.textContent = String(value);
  wrapper.append(heading, content);
  return wrapper;
}

function listRow(label, values, empty) {
  const wrapper = document.createElement("section");
  const heading = document.createElement("h3");
  heading.textContent = label;
  wrapper.append(heading);
  if (!values.length) {
    const content = document.createElement("p");
    content.textContent = empty;
    wrapper.append(content);
    return wrapper;
  }
  const list = document.createElement("ul");
  for (const value of values) {
    const item = document.createElement("li");
    item.textContent = value;
    list.append(item);
  }
  wrapper.append(list);
  return wrapper;
}

function scalar(value) {
  if (value == null) return null;
  if (["string", "number", "boolean"].includes(typeof value)) return value;
  return JSON.stringify(value);
}
function strings(value) { return (Array.isArray(value) ? value : value == null ? [] : [value]).map((item) => optionalText(item, 2000)).filter(Boolean); }
function optionalText(value, maximum) { return value == null ? null : text(value, maximum); }
function text(value, maximum) { const result = String(value).trim(); return result.slice(0, maximum); }
