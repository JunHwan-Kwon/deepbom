// Editor for custom target profiles: pick a built-in profile as the base, edit
// its numbers, and save. Only fields that actually differ from the base become
// overrides, so a saved profile never claims to have retuned a value it left
// alone. The dialog builds its own DOM so no markup has to be reserved for it.
import {
  EVIDENCE_CLASSES,
  TUNABLE_FIELDS,
  deleteCustomTarget,
  loadCustomTargets,
  overridesFromForm,
  saveCustomTarget,
  validateCustomTargetSpec,
} from "./custom-targets.js";

const GROUP_TITLES = {
  compute: "Compute model",
  memory: "Memory and cache",
  isa: "Instruction set",
  delegate: "Delegate behaviour",
  identity: "Descriptive text",
};

function element(tag, className, text) {
  const node = document.doc ? document.doc.createElement(tag) : document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function labelled(labelText, control, hint) {
  const wrap = element("label", "custom-target-field");
  wrap.append(element("span", "custom-target-field-label", labelText));
  wrap.append(control);
  if (hint) wrap.append(element("small", "custom-target-field-hint", hint));
  return wrap;
}

export function createCustomTargetEditor({ getBuiltInProfiles, onSaved }) {
  let dialog = null;
  let inputs = new Map();
  let baseSelect = null;
  let idInput = null;
  let labelInput = null;
  let evidenceSelect = null;
  let noteInput = null;
  let familyInput = null;
  let errorBox = null;
  let editingId = null;

  function baseProfile() {
    return getBuiltInProfiles().find((profile) => profile.id === baseSelect.value) || null;
  }

  function fillFromBase(overrides = {}) {
    const base = baseProfile();
    for (const [key, control] of inputs) {
      const value = key in overrides ? overrides[key] : base?.[key];
      if (control.type === "checkbox") control.checked = Boolean(value);
      else control.value = value == null ? "" : String(value);
    }
    familyInput.value = overrides.compute_utilization_by_kernel_class
      ? JSON.stringify(overrides.compute_utilization_by_kernel_class, null, 2)
      : "";
  }

  function build() {
    dialog = element("dialog", "custom-target-dialog");
    const form = element("form", "custom-target-form");
    form.method = "dialog";

    const head = element("div", "custom-target-head");
    head.append(element("h3", null, "Custom CPU cost profile"));
    head.append(element("p", "custom-target-intro",
      "Start from a built-in profile and retune its planning numbers. Only changed "
      + "fields are recorded, and the saved profile carries its own hash, its base, "
      + "and the evidence class you declare."));
    form.append(head);

    baseSelect = element("select", "custom-target-base");
    baseSelect.addEventListener("change", () => fillFromBase());
    idInput = element("input", "custom-target-id");
    idInput.placeholder = "custom:my-device";
    labelInput = element("input", "custom-target-label");
    labelInput.placeholder = "My device, 1 thread";
    evidenceSelect = element("select", "custom-target-evidence");
    for (const value of EVIDENCE_CLASSES) evidenceSelect.append(new Option(value, value));
    noteInput = element("input", "custom-target-note");
    noteInput.placeholder = "How these numbers were obtained";

    const identity = element("div", "custom-target-grid");
    identity.append(
      labelled("Base profile", baseSelect),
      labelled("Profile id", idInput, "Must start with custom:"),
      labelled("Label", labelInput),
      labelled("Evidence class", evidenceSelect, "MEASURED requires a note naming the measurement"),
      labelled("Evidence note", noteInput),
    );
    form.append(identity);

    const byGroup = new Map();
    for (const field of TUNABLE_FIELDS) {
      if (!byGroup.has(field.group)) byGroup.set(field.group, []);
      byGroup.get(field.group).push(field);
    }
    for (const [group, fields] of byGroup) {
      const section = element("section", "custom-target-section");
      section.append(element("h4", null, GROUP_TITLES[group] || group));
      const grid = element("div", "custom-target-grid");
      for (const field of fields) {
        const control = element("input", "custom-target-input");
        if (field.kind === "boolean") control.type = "checkbox";
        else if (field.kind === "text") control.type = "text";
        else {
          control.type = "number";
          control.step = field.kind === "integer" ? "1" : "any";
          control.min = String(field.min);
          control.max = String(field.max);
        }
        inputs.set(field.key, control);
        grid.append(labelled(field.unit ? `${field.label} (${field.unit})` : field.label, control, field.hint));
      }
      section.append(grid);
      form.append(section);
    }

    const familySection = element("section", "custom-target-section");
    familySection.append(element("h4", null, "Per-kernel utilization"));
    familySection.append(element("p", "custom-target-intro",
      "Optional. One utilization per compute kernel family, as JSON. A family listed "
      + "here overrides the scalar for its operators. Family keys come from the "
      + "compute_kernel_class field on each analyzed operator, for example "
      + "f32_gemm, qu8_dwconv, qu8_igemm."));
    familyInput = element("textarea", "custom-target-families");
    familyInput.rows = 5;
    familyInput.placeholder = '{\n  "f32_gemm": 0.036,\n  "qu8_dwconv": 0.014\n}';
    familySection.append(familyInput);
    form.append(familySection);

    errorBox = element("p", "custom-target-error");
    errorBox.hidden = true;
    form.append(errorBox);

    const actions = element("div", "custom-target-actions");
    const cancel = element("button", "secondary-action", "Cancel");
    cancel.type = "button";
    cancel.addEventListener("click", () => dialog.close());
    const remove = element("button", "secondary-action custom-target-delete", "Delete");
    remove.type = "button";
    remove.addEventListener("click", () => {
      if (!editingId) return;
      deleteCustomTarget(editingId);
      dialog.close();
      onSaved?.();
    });
    const save = element("button", "primary-action", "Save target");
    save.type = "button";
    save.addEventListener("click", () => submit());
    actions.append(cancel, remove, save);
    form.append(actions);

    dialog.append(form);
    document.body.append(dialog);
  }

  function readFamilies() {
    const text = familyInput.value.trim();
    if (!text) return null;
    let parsed;
    try { parsed = JSON.parse(text); }
    catch (error) { throw new Error(`Per-kernel utilization is not valid JSON: ${error.message}`); }
    return parsed;
  }

  function submit() {
    errorBox.hidden = true;
    try {
      const base = baseProfile();
      if (!base) throw new Error("Select a base profile");
      const values = {};
      for (const [key, control] of inputs) {
        values[key] = control.type === "checkbox" ? control.checked : control.value;
      }
      const families = readFamilies();
      if (families) values.compute_utilization_by_kernel_class = families;
      const spec = validateCustomTargetSpec({
        base: base.id,
        id: idInput.value.trim(),
        label: labelInput.value.trim(),
        evidence_class: evidenceSelect.value,
        evidence_note: noteInput.value.trim(),
        overrides: overridesFromForm(base, values),
      }, base);
      if (editingId && editingId !== spec.id) deleteCustomTarget(editingId);
      saveCustomTarget(spec, undefined, base);
      dialog.close();
      onSaved?.(spec);
    } catch (error) {
      errorBox.textContent = error?.message || String(error);
      errorBox.hidden = false;
    }
  }

  return {
    open(existingId = null) {
      if (!dialog) build();
      const profiles = getBuiltInProfiles();
      baseSelect.replaceChildren(...profiles.map((profile) => new Option(profile.label, profile.id)));
      const existing = existingId ? loadCustomTargets().find((entry) => entry.id === existingId) : null;
      editingId = existing?.id || null;
      baseSelect.value = existing?.base || profiles[0]?.id || "";
      idInput.value = existing?.id || "custom:";
      labelInput.value = existing?.label || "";
      evidenceSelect.value = existing?.evidence_class || "USER_DECLARED";
      noteInput.value = existing?.evidence_note || "";
      fillFromBase(existing?.overrides || {});
      errorBox.hidden = true;
      dialog.querySelector(".custom-target-delete").hidden = !editingId;
      dialog.showModal();
    },
  };
}
