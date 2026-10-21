// Merge the component's independent view/selection writers. Some hosts expose
// widgetState asynchronously, so retain the last successfully written snapshot.
const states = new WeakMap();
export function readWidgetState(openai) {
  return states.get(openai) || openai?.widgetState || {};
}
export function updateWidgetState(openai, artifactSha256, patch) {
  if (typeof openai?.setWidgetState !== "function") return false;
  const previous = readWidgetState(openai);
  const sameArtifact = previous.modelContent?.artifact_sha256 === artifactSha256;
  const base = sameArtifact ? previous : {};
  const next = {
    ...base, ...patch,
    modelContent: { ...(base.modelContent || {}), ...(patch.modelContent || {}), artifact_sha256: artifactSha256 },
    privateContent: { ...(base.privateContent || {}), ...(patch.privateContent || {}) },
  };
  try { openai.setWidgetState(next); states.set(openai, next); return true; }
  catch { return false; } // Optional host state must not prevent evidence inspection.
}
export function widgetLanguage(openai) {
  return /^ko(?:-|$)/i.test(openai?.locale || globalThis.navigator?.language || "en") ? "ko" : "en";
}
