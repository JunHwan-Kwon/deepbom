import { buildExecutionPlacementEvidence } from "./execution-placement-evidence.js";

export function acceleratorProfilesForAnalysis(analysis, runtimeEvidence = null) {
  if (!analysis) return [];
  return buildExecutionPlacementEvidence(analysis, runtimeEvidence).static_profiles
    .filter((profile) => profileClass(profile) === "accelerator");
}

export function renderAcceleratorProfileSwitcher(root, {
  analysis,
  runtimeEvidence = null,
  selectedProfileId = "",
  onSelect = null,
  onLoadSourceLedgers = null,
} = {}) {
  if (!root) return "";
  let profiles = [];
  try {
    profiles = acceleratorProfilesForAnalysis(analysis, runtimeEvidence);
  } catch (error) {
    console.warn("Accelerator profile projection unavailable", error);
  }
  if (!profiles.length) {
    if (String(analysis?.format || "").toLowerCase() !== "tflite") {
      root.hidden = true;
      root.replaceChildren();
      return "";
    }
    const unavailable = ["TFLite GPU", "NNAPI"].map((name) => pendingButton(name));
    const load = button("Load source ledgers", "target-pill target-pill-add");
    load.addEventListener("click", () => onLoadSourceLedgers?.());
    root.hidden = false;
    root.replaceChildren(label(), ...unavailable, load, boundary());
    return "";
  }
  const selected = profiles.some((profile) => profile.profile_id === selectedProfileId)
    ? selectedProfileId
    : preferredProfile(profiles)?.profile_id || "";
  const pills = profiles.map((profile) => {
    const control = button(profile.label || profile.profile_id, `target-pill${profile.profile_id === selected ? " active" : ""}`);
    control.dataset.acceleratorProfileId = profile.profile_id;
    control.title = `${profile.label}. Source-backed independent eligibility projection; not observed assignment or a GPU cost model.`;
    const state = document.createElement("span");
    state.className = "target-pill-ready loaded";
    control.prepend(state);
    control.addEventListener("click", () => onSelect?.(profile.profile_id));
    return control;
  });
  root.hidden = false;
  root.replaceChildren(label(), ...pills, boundary());
  return selected;
}

function profileClass(profile) {
  const identity = `${profile?.profile_id || ""} ${profile?.label || ""}`.toLowerCase();
  if (/(gpu|directml|webgpu|webnn|nnapi|qnn|coreml|tensorrt|cuda|rocm|metal|vulkan|opencl)/.test(identity)) return "accelerator";
  if (/(cpu|xnnpack|wasm)/.test(identity)) return "cpu";
  return "other";
}

function preferredProfile(profiles) {
  const rank = ["tflite_gpu", "webgpu", "directml", "tensorrt", "qnn", "coreml", "nnapi", "tflite_nnapi"];
  return [...profiles].sort((left, right) => {
    const leftRank = rank.indexOf(String(left.profile_id || "").toLowerCase());
    const rightRank = rank.indexOf(String(right.profile_id || "").toLowerCase());
    return (leftRank < 0 ? rank.length : leftRank) - (rightRank < 0 ? rank.length : rightRank)
      || String(left.label || left.profile_id).localeCompare(String(right.label || right.profile_id));
  })[0] || null;
}

function pendingButton(name) {
  const control = button(`${name} - source ledger required`, "target-pill accelerator-profile-pending");
  control.disabled = true;
  control.title = `${name} eligibility is not assessed until the pinned source ledger is loaded.`;
  return control;
}

function label() {
  const node = document.createElement("span");
  node.className = "target-switcher-label";
  node.textContent = "Accelerator eligibility:";
  return node;
}

function boundary() {
  const node = document.createElement("p");
  node.className = "accelerator-switcher-boundary";
  node.textContent = "Changes source eligibility, exclusion reasons, segment boundaries, and logical edge exposure. It does not change the selected CPU roofline, infer GPU kernels, or claim runtime assignment.";
  return node;
}

function button(text, className) {
  const node = document.createElement("button");
  node.type = "button";
  node.className = className;
  const content = document.createElement("strong");
  content.textContent = text;
  node.append(content);
  return node;
}
