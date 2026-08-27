export const APP_SURFACES = Object.freeze({
  DEFAULT: "default",
  MEDICAL: "medical",
});

export function detectAppSurface(locationLike = window.location) {
  const params = new URLSearchParams(locationLike.search || "");
  if (params.get("surface") === APP_SURFACES.MEDICAL) return APP_SURFACES.MEDICAL;

  const path = (locationLike.pathname || "").replace(/\/+$/, "");
  if (path === "/medical" || path === "/web/medical.html") return APP_SURFACES.MEDICAL;
  return APP_SURFACES.DEFAULT;
}

export function isMedicalSurface(surface) {
  return surface === APP_SURFACES.MEDICAL;
}

export function prepareAppSurface(surface, doc = document) {
  if (!isMedicalSurface(surface)) return;

  doc.title = "Medical Evidence Workspace / DEEPBOM";
  setText(doc.querySelector(".topbar .eyebrow"), "ON-DEVICE Pipeline / Medical");
  setText(doc.querySelector(".topbar h1"), "Medical Evidence Workspace");
  setText(
    doc.querySelector("#outputModuleSelector .module-head p"),
    "Select a completed module result, open a login-free Engineering or Regulatory Support Report, or assemble a controlled full evidence bundle.",
  );
  setText(
    doc.querySelector('[data-module-run-panel="engineering_report"] .module-panel-head p'),
    "Includes model structure, quantization scope, predicted XNNPACK segments, roofline signals, benchmark evidence, and engineering action items. It does not include the Regulatory Report.",
  );
  setText(
    doc.querySelector('[data-module-panel="engineering_report"] .report-export-panel p'),
    "Download the login-free, watermarked Engineering Report. Raw reusable derivatives remain separately controlled. This report never includes the Regulatory Support Report.",
  );

  insertAfter(
    doc.querySelector('[data-module-tab="engineering_report"]'),
    regulatoryTabHtml(),
    '[data-module-tab="regulatory_report"]',
  );
  insertAfter(
    doc.querySelector('[data-module-run-panel="engineering_report"]'),
    regulatoryRunPanelHtml(),
    '[data-module-run-panel="regulatory_report"]',
  );
  insertAfter(
    doc.querySelector('[data-module-panel="engineering_report"]'),
    regulatoryResultPanelHtml(),
    '[data-module-panel="regulatory_report"]',
  );
}

function setText(node, value) {
  if (node) node.textContent = value;
}

function insertAfter(anchor, html, existingSelector) {
  if (!anchor || !html) return;
  if (existingSelector && anchor.ownerDocument.querySelector(existingSelector)) return;
  anchor.insertAdjacentHTML("afterend", html.trim());
}

function regulatoryTabHtml() {
  return `
    <button class="module-tab" type="button" data-module-tab="regulatory_report" data-feature-id="regulatory_report">
      <span>Medical AI</span>
      <strong>Regulatory Report</strong>
      <em>Open</em>
    </button>
  `;
}

function regulatoryRunPanelHtml() {
  return `
    <section class="module-run-panel" data-module-run-panel="regulatory_report">
      <div class="module-panel-head">
        <div>
          <span>Medical AI report</span>
          <h3>Regulatory Report: broader evidence package with Engineering Report included as an appendix.</h3>
          <p>The watermarked Regulatory Support Report is available without sign-in. The full bundle with controlled Research evidence requires regulatory workspace authorization.</p>
        </div>
        <button class="secondary-action" type="button" data-request-feature="regulatory_report">Request full bundle access</button>
      </div>
    </section>
  `;
}

function regulatoryResultPanelHtml() {
  return `
    <section class="module-panel" data-module-panel="regulatory_report">
      <div class="report-workspace">
        <div class="report-preview">
          <div class="report-preview-head">
            <div>
              <span>Generated report preview</span>
              <strong id="regulatoryReportPreviewTitle">Regulatory report</strong>
            </div>
            <em id="regulatoryReportPreviewStatus">Not generated</em>
          </div>
          <pre id="regulatoryReportPreview">Run a static audit to generate the regulatory report preview.</pre>
        </div>
        <div class="report-export-panel">
          <div>
            <span>Medical AI evidence package</span>
            <p>Login-free, watermarked support copy with the Engineering Report appendix, evidence classification, and traceability. It is not a regulatory submission or approval.</p>
          </div>
          <div class="export-buttons">
            <button id="downloadRegulatoryReport" type="button">Regulatory Report</button>
          </div>
        </div>
        <div class="bundle-panel">
          <div class="bundle-copy">
            <span>Regulatory evidence bundle</span>
            <h3>Starts from the full Engineering Bundle, then adds the Regulatory Report and enabled Research evidence.</h3>
            <ul class="bundle-scope" id="evidenceBundleScope"></ul>
          </div>
          <div class="bundle-action-wrap">
            <button id="downloadEvidenceBundle" class="bundle-action" type="button" disabled>Regulatory Bundle ZIP</button>
            <p class="evidence-bundle-note" id="evidenceBundleNote">Run a static audit first. The full bundle requires regulatory workspace authorization; the standalone report above does not.</p>
          </div>
        </div>
      </div>
    </section>
  `;
}
