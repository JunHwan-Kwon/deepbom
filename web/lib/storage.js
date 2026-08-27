const AGREEMENT_KEY = "ondevice-audit-privacy-agreement-v3";
const AGREEMENT_VERSION = 3;
const RESEARCH_CONSENT_KEY = "ondevice-audit-structure-research-consent-v1";
const RESEARCH_CONSENT_AT_KEY = "ondevice-audit-structure-research-consent-at-v1";
const TARGET_KEY = "ondevice-audit-target-v1";
const AUDIT_TIMING_KEY = "deepbom-static-audit-timing-v1";

export const AGREEMENT_POLICY_VERSION = `agreement v${AGREEMENT_VERSION} / research-consent v1`;

export function readAgreementRecord() {
  try {
    const raw = sessionStorage.getItem(AGREEMENT_KEY);
    if (!raw) return null;
    const record = JSON.parse(raw);
    return record && typeof record === "object" ? record : null;
  } catch {
    return null;
  }
}

export function readAgreementAccepted() {
  const record = readAgreementRecord();
  return record?.accepted === true && record?.version === AGREEMENT_VERSION;
}

export function writeAgreementAccepted() {
  try {
    sessionStorage.setItem(AGREEMENT_KEY, JSON.stringify({
      accepted: true,
      version: AGREEMENT_VERSION,
      accepted_at: new Date().toISOString(),
    }));
  } catch {
    // Storage can be unavailable in hardened browser profiles; the modal still closes.
  }
}

export function readResearchConsent() {
  try {
    return localStorage.getItem(RESEARCH_CONSENT_KEY) === "accepted";
  } catch {
    return false;
  }
}

export function readResearchConsentRecord() {
  const consent = readResearchConsent();
  try {
    return {
      consent,
      updated_at: localStorage.getItem(RESEARCH_CONSENT_AT_KEY) || null,
    };
  } catch {
    return { consent, updated_at: null };
  }
}

export function writeResearchConsent(value) {
  try {
    localStorage.setItem(RESEARCH_CONSENT_KEY, value ? "accepted" : "declined");
    localStorage.setItem(RESEARCH_CONSENT_AT_KEY, new Date().toISOString());
  } catch {
    // Storage can be unavailable in hardened browser profiles.
  }
}

export function readSavedTarget() {
  return safeLocalStorageGet(TARGET_KEY);
}

export function writeSavedTarget(value) {
  try {
    localStorage.setItem(TARGET_KEY, value);
  } catch {
    // Local storage can be unavailable in private profiles.
  }
}

export function readAuditTimings() {
  try {
    const parsed = JSON.parse(localStorage.getItem(AUDIT_TIMING_KEY) || "[]");
    return Array.isArray(parsed) ? parsed.filter(validAuditTimingRecord).slice(-24) : [];
  } catch {
    return [];
  }
}

export function recordAuditTiming(record) {
  if (!validAuditTimingRecord(record)) return;
  try {
    const records = readAuditTimings();
    records.push({
      format: String(record.format).toUpperCase(),
      sizeBytes: Math.round(Number(record.sizeBytes)),
      comparisonTargetCount: Math.round(Number(record.comparisonTargetCount)),
      durationMs: Math.round(Number(record.durationMs)),
      recordedAt: new Date().toISOString(),
    });
    localStorage.setItem(AUDIT_TIMING_KEY, JSON.stringify(records.slice(-24)));
  } catch {
    // Audit timing calibration is best-effort when local storage is unavailable.
  }
}

function validAuditTimingRecord(record) {
  return record && ["TFLITE", "ONNX"].includes(String(record.format || "").toUpperCase())
    && Number.isFinite(Number(record.sizeBytes)) && Number(record.sizeBytes) > 0
    && Number.isInteger(Number(record.comparisonTargetCount)) && Number(record.comparisonTargetCount) > 0
    && Number.isFinite(Number(record.durationMs)) && Number(record.durationMs) >= 50
    && Number(record.durationMs) <= 10 * 60 * 1000;
}

export function safeLocalStorageGet(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export async function syncResearchConsent() {
  const consented = readResearchConsent();
  try {
    await fetch("/api/benchmark/consent", {
      method: "POST",
      headers: { "content-type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ consented, scope: "benchmark_research" }),
    });
  } catch {
    // Consent is still respected locally; server sync is best-effort.
  }
}
