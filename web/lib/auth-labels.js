import { humanStatusLabel } from "./format.js";
import { accessProfileForUser } from "./access-policy.js";

// Stable API access IDs map to the neutral module descriptions shown in the request form.
export const ACCESS_REQUEST_PROFILES = {
  research: {
    id: "research",
    label: "Advanced analysis modules",
    description: "Controlled, research-stage local analysis modules. No model data is uploaded.",
    type: "research_collab",
    capability: "research_access",
    features: [
      "Artifact Geometry: experimental deploy-artifact composites and their deterministic components",
      "Perturbation: local input/output drift and weight sensitivity",
      "Backend Consistency: backend availability and output drift",
      "Deployment Sensitivity: finite-difference deployment-function observations",
    ],
  },
  medical_ai: {
    id: "medical_ai",
    label: "Regulatory workspace",
    description: "Medical AI evidence reporting and controlled analysis modules.",
    type: "feature_access",
    capability: "medical_ai_access",
    features: ["Full regulatory evidence bundle", "All Research-stage modules"],
  },
};

// Map historical feature IDs to their current access-profile ID.
export function profileIdForCapability(capabilityId) {
  const map = {
    artifact_geometry: "research",
    perturbation_analysis: "research",
    runtime_basin: "research",
    deployment_sensitivity: "research",
    regulatory_report: "medical_ai",
    research_beta: "research",
    research_beta_access: "research",
  };
  return map[capabilityId] || null;
}

export function roleLabel(role) {
  if (role === "admin") return "Admin";
  return "Account";
}

export function providerLabel(provider) {
  if (provider === "google") return "Google";
  return "Email";
}

export function gatedExportLabel(base, locked, user) {
  if (!locked) return base;
  return `${base} / ${user ? "Access required" : "Sign in"}`;
}

export function applyGatedExportLabels(reportButton, rawButtons, reportLocked, rawLocked, user) {
  reportButton.textContent = gatedExportLabel("Engineering Report HTML", reportLocked, user);
  const labels = ["Raw Evidence Package", "Roofline CSV", "Mermaid Stage Graph", "Visual PNGs", "Engineering Bundle ZIP", "Public-redacted ZIP"];
  rawButtons.forEach((button, index) => { button.textContent = gatedExportLabel(labels[index], rawLocked, user); });
}

export function accessLabel(user, accessGrantState = null) {
  if (user?.test_access?.active) return "External test access";
  const profile = accessProfileForUser(user || { access_profile: accessGrantState?.access_profile || "verified" });
  const status = user?.access_status || accessGrantState?.status || "active";
  if (profile === "admin") return "Admin access";
  if (status !== "active") return `${status} access`;
  if (profile === "medical_ai") return "Regulatory workspace";
  if (profile === "research") return "Advanced modules";
  return "Verified account";
}

export function statusClass(value) {
  if (["available", "granted", "closed"].includes(value)) return "ok";
  if (["new", "reviewing", "planned", "preview", "spec_only", "request_feedback", "available_by_request"].includes(value)) return "pending";
  if (["declined", "revoked", "request", "sign_in_required", "email_verification_required"].includes(value)) return "warn";
  return "neutral";
}

export function requestDraftTitle(profileId) {
  const titles = {
    research: "Request advanced module access",
    research_beta: "Request advanced module access",
    artifact_geometry: "Request advanced module access",
    perturbation_analysis: "Request advanced module access",
    runtime_basin: "Request advanced module access",
    deployment_sensitivity: "Request advanced module access",
    medical_ai: "Request regulatory workspace access",
    regulatory_report: "Request regulatory workspace access",
  };
  return titles[profileId] || "Feedback for on-device audit workflow";
}

export function capabilitiesForUser(user, accessGrantState = null) {
  const signedIn = Boolean(user);
  const admin = user?.role === "admin";
  const allowed = accessGrantState?.allowed || {};
  return {
    signedIn,
    admin,
    report: true,
    export: true,
    raw_export: true,
    deepbom: Boolean(signedIn && (admin || allowed.deepbom === true)),
    perturbation: Boolean(signedIn && (admin || allowed.perturbation === true)),
    runtime_basin: Boolean(signedIn && (admin || allowed.runtime_basin === true)),
    deployment_sensitivity: Boolean(signedIn && (admin || allowed.deployment_sensitivity === true)),
    regulatory_report: true,
  };
}

export function bundleUserIdentityForUser(user) {
  if (!user) return { label: "anonymous", email: "", role: "" };
  return {
    label: `${user.name || user.email || "signed-in user"} (${user.email || "no email"})`,
    email: user.email || "",
    name: user.name || "",
    role: user.role || "",
  };
}

export function moduleAccessStatesFor(capabilities = {}, user = null) {
  return {
    static: { label: "Open", className: "available", locked: false },
    engineering_report: { label: "Report", className: "available", locked: false },
    regulatory_report: { label: "Report", className: "available", locked: false },
    deepbom: capabilities.deepbom
      ? { label: "Run", className: "available", locked: false }
      : user
        ? { label: "Module access", className: "locked", locked: true }
        : { label: "Sign in", className: "locked", locked: true },
    perturbation: capabilities.perturbation
      ? { label: "Run", className: "available", locked: false }
      : user
        ? { label: "Module access", className: "locked", locked: true }
        : { label: "Sign in", className: "locked", locked: true },
    runtime_basin: capabilities.runtime_basin
      ? { label: "Run", className: "available", locked: false }
      : user
        ? { label: "Module access", className: "locked", locked: true }
        : { label: "Sign in", className: "locked", locked: true },
    deployment_sensitivity: capabilities.deployment_sensitivity
      ? { label: "Run", className: "available", locked: false }
      : { label: user ? "Module access" : "Sign in", className: "locked", locked: true },
  };
}

export function selectableModuleIdFor(tabs, requestedId, fallbackIds = ["engineering_report", "export_contracts"]) {
  const selectable = (tab) => Boolean(
    tab
    && !tab.hidden
    && !tab.disabled
    && tab.getAttribute("aria-disabled") !== "true"
  );
  const byId = (id) => tabs.find((tab) => tab.dataset.moduleTab === id);
  const selected = [requestedId, ...fallbackIds]
    .map(byId)
    .find(selectable)
    || tabs.find(selectable);
  return selected?.dataset.moduleTab || "export_contracts";
}

export function moduleWorkflowDescription(moduleId, fallback = "") {
  return {
    deepbom: "Artifact descriptors",
    perturbation: "Input/output drift",
    runtime_basin: "Backend drift",
    deployment_sensitivity: "Finite-difference observations",
  }[moduleId] || fallback;
}

export function combineModuleResults(results = []) {
  const present = results.filter(Boolean);
  if (!present.length) return null;
  const statuses = present.map((result) => result.status || "complete");
  if (statuses.includes("failed")) return { status: "failed" };
  if (statuses.includes("blocked")) return { status: "blocked" };
  if (present.length < results.length || statuses.some((status) => status !== "complete")) {
    return { status: "running" };
  }
  return { status: "complete" };
}

export function moduleTabStatusTextFor({
  moduleId,
  accessState,
  result = null,
  capabilities = {},
  hasCurrent = false,
} = {}) {
  if (accessState?.locked) return accessState.label;
  if (moduleId === "engineering_report") {
    if (!hasCurrent) return "Run audit";
    return "Report";
  }
  if (moduleId === "regulatory_report") {
    if (!hasCurrent) return "Run audit";
    return "Report";
  }
  if (moduleId === "export_contracts") {
    return hasCurrent ? "7 docs + pack" : "Run audit";
  }
  if (!result) return "Not run";
  if (result.status === "complete") return "Complete";
  if (result.status === "failed") return "Failed";
  if (result.status === "blocked") return "Blocked";
  if (!result.status) return "Complete";
  return humanStatusLabel(result.status);
}
