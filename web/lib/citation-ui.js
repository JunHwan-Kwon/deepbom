import { ANALYZER_SEMANTIC_VERSION, DEEPBOM_CITATION } from "./app-config.js";
import {
  ANALYZER_BUILD_COMMIT,
  ANALYZER_BUILD_SOURCE_STATE,
  ANALYZER_BUNDLE_CONTENT_SHA256,
} from "./build-metadata.js";
import { copyTextToClipboard } from "./clipboard.js";

export function initializeCitationUi(doc = document) {
  const build = doc.getElementById("applicationBuild");
  if (build) {
    const commit = ANALYZER_BUILD_COMMIT?.slice(0, 8) || "unavailable";
    const content = ANALYZER_BUNDLE_CONTENT_SHA256?.slice(0, 10) || "unavailable";
    build.textContent = `Application ${ANALYZER_SEMANTIC_VERSION} · source ${commit} · content ${content} · ${ANALYZER_BUILD_SOURCE_STATE}`;
    build.title = "Application version, source commit, analyzer content digest, and build source state";
    build.hidden = false;
  }
  const button = doc.getElementById("copyCitationBtn");
  if (!button) return;

  const defaultText = "Copy citation";
  let resetTimer = null;
  button.dataset.citationReady = "true";
  button.addEventListener("click", async () => {
    if (resetTimer !== null) window.clearTimeout(resetTimer);
    try {
      await copyTextToClipboard(DEEPBOM_CITATION, doc);
      button.textContent = "Copied";
      button.classList.add("copied");
      button.setAttribute("aria-label", "DEEPBOM citation copied");
    } catch {
      button.textContent = "Copy failed";
      button.classList.remove("copied");
      button.setAttribute("aria-label", "Could not copy the DEEPBOM citation");
    }
    resetTimer = window.setTimeout(() => {
      button.textContent = defaultText;
      button.classList.remove("copied");
      button.setAttribute("aria-label", "Copy the recommended DEEPBOM citation");
      resetTimer = null;
    }, 1800);
  });
}

if (typeof document !== "undefined") initializeCitationUi(document);
