import { formatDrift } from "./format.js";

const DEFAULT_DRIFT_PROFILE = {
  dtype: "FLOAT32",
  unit: "raw output value",
  rmsOk: 0.0005,
  rmsWarn: 0.005,
  maxOk: 0.001,
  maxWarn: 0.01,
};

export function statusForEntropy(value) {
  const entropy = Number(value || 0);
  if (entropy > 0 && entropy < 4.5) return { tone: "warn", label: "watch", criteria: "watch < 4.5 bits/B; otherwise info. Interpret with file format and sparsity context." };
  return { tone: "info", label: "info", criteria: "watch < 4.5 bits/B; otherwise info. Interpret with file format and sparsity context." };
}

export function statusForRmsDrift(value, profile = DEFAULT_DRIFT_PROFILE, baselineMagnitude = 0) {
  const rms = Number(value || 0);
  const base = Number(baselineMagnitude || 0);
  // Switch to relative comparison when baseline has large magnitude (unnormalized logits, >1.0)
  // and the profile provides relative thresholds (FP32 only — INT8 LSB profiles do not carry them)
  const useRelative = base > 1.0 && profile.relRmsOk != null;
  const effective = useRelative ? rms / base : rms;
  const ok   = useRelative ? profile.relRmsOk   : profile.rmsOk;
  const warn = useRelative ? profile.relRmsWarn  : profile.rmsWarn;
  const criteria = useRelative
    ? `ok <= ${(ok * 100).toFixed(1)}% relative; warn <= ${(warn * 100).toFixed(1)}% relative; risk > warn. (Normalized by baseline output magnitude ${formatDrift(base)}.)`
    : `ok <= ${formatDrift(ok)}; warn <= ${formatDrift(warn)} ${profile.unit}; risk > warn.`;
  if (effective <= ok) return { tone: "good", label: "ok", criteria };
  if (effective <= warn) return { tone: "warn", label: "warn", criteria };
  return { tone: "risk", label: "risk", criteria };
}

export function statusForMaxDrift(value, profile = DEFAULT_DRIFT_PROFILE, baselineMagnitude = 0) {
  const drift = Number(value || 0);
  const base = Number(baselineMagnitude || 0);
  const useRelative = base > 1.0 && profile.relMaxOk != null;
  const effective = useRelative ? drift / base : drift;
  const ok   = useRelative ? profile.relMaxOk   : profile.maxOk;
  const warn = useRelative ? profile.relMaxWarn  : profile.maxWarn;
  const criteria = useRelative
    ? `ok <= ${(ok * 100).toFixed(1)}% relative; warn <= ${(warn * 100).toFixed(1)}% relative; risk > warn. (Normalized by baseline output magnitude ${formatDrift(base)}.)`
    : `ok <= ${formatDrift(ok)}; warn <= ${formatDrift(warn)} ${profile.unit}; risk > warn.`;
  if (effective <= ok) return { tone: "good", label: "ok", criteria };
  if (effective <= warn) return { tone: "warn", label: "warn", criteria };
  return { tone: "risk", label: "risk", criteria };
}

export function driftSeverity(drift, profile = DEFAULT_DRIFT_PROFILE) {
  const leftRms = Number(drift?.leftRms || 0);
  const statuses = [
    statusForRmsDrift(drift?.rms || 0, profile, leftRms),
    statusForMaxDrift(drift?.maxAbs || 0, profile, leftRms),
    statusForCosineDistance(drift?.cosineDistance || 0),
    statusForTop1Flip(Boolean(drift?.top1Flip)),
  ];
  const worst = statuses.reduce((selected, status) =>
    severityRank(status.label) > severityRank(selected.label) ? status : selected,
  statuses[0]);
  return {
    tone: worst.tone,
    label: worst.label,
    criteria: `Worst of RMS, max-abs, cosine, and top-1 checks. ${worst.criteria}`,
  };
}

export function severityRank(label) {
  if (label === "risk") return 3;
  if (label === "warn" || label === "watch") return 2;
  if (label === "ok" || label === "info") return 1;
  return 0;
}

export function statusFromSeverityLabel(label) {
  if (label === "risk") {
    return { tone: "risk", label: "risk", criteria: "Worst layer drift exceeded at least one configured output-drift risk threshold." };
  }
  if (label === "warn" || label === "watch") {
    return { tone: "warn", label: "warn", criteria: "Worst layer drift reached a warning threshold; validate with representative calibration samples." };
  }
  return { tone: "good", label: "ok", criteria: "Worst sampled layer stayed within configured output-drift thresholds." };
}

export function statusForCosineDistance(value) {
  const distance = Number(value || 0);
  const criteria = "ok <= 1e-4; warn <= 1e-2; risk > 1e-2.";
  if (distance <= 1e-4) return { tone: "good", label: "ok", criteria };
  if (distance <= 1e-2) return { tone: "warn", label: "warn", criteria };
  return { tone: "risk", label: "risk", criteria };
}

export function statusForTop1Flip(flipped) {
  return flipped
    ? { tone: "risk", label: "risk", criteria: "ok = no top-1/rank flip; risk = top-1 changed." }
    : { tone: "good", label: "ok", criteria: "ok = no top-1/rank flip; risk = top-1 changed." };
}

export function statusForBackendCoverage(successful, total) {
  if (!total || !successful) return { tone: "risk", label: "risk", criteria: "ok = all attempted backends completed; warn = partial completion; risk = none completed." };
  if (successful === total) return { tone: "good", label: "ok", criteria: "ok = all attempted backends completed; warn = partial completion; risk = none completed." };
  return { tone: "warn", label: "warn", criteria: "ok = all attempted backends completed; warn = partial completion; risk = none completed." };
}

export function statusInfo(criteria = "Reference value; interpret with neighboring metrics.") {
  return { tone: "info", label: "info", criteria };
}

export function statusBlocked(criteria = "Blocked until the required artifact/runtime is supplied.") {
  return { tone: "blocked", label: "blocked", criteria };
}
