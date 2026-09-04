import { downloadBlob, downloadTextArtifact } from "./download.js";
import { TEXT_EXPORT_ARTIFACTS } from "./export-artifacts.js";
import { buildCanonicalPackageDigest, jsonForDownload, zipTextFile } from "./report-utils.js";
import { createZipBlob } from "./zip.js";
import { compareInterfaceContracts, parseProductionInterfaceContract } from "./interface-contract.js";
import { buildInterfaceQuantizationContractLedger } from "./quantization-contract-summary.js";
import { signPackageDigest, verifyPackageSignature } from "./public-key-signature.js";

const EXPORTS = [
  ["downloadCycloneDxEvidence", "cyclonedxEvidence", "cyclonedx_evidence"],
  ["downloadObservedFormulation", "observedFormulation", "observed_formulation"],
  ["downloadRuntimeRequirements", "runtimeRequirementManifest", "runtime_requirement_manifest"],
  ["downloadMissingProvenance", "missingProvenanceFields", "missing_provenance_field_specification"],
];
const PUBLIC_EXPORT_BUTTONS = new Set([
  "downloadCycloneDxEvidence",
]);

export function createExportContractController({
  elements,
  getContext,
  getDocuments,
  getPublicDocuments,
  getFilename,
  ensureAllowed,
  ensureHash,
  getAccess,
  onStatus,
  onProductionContractChange,
  onProductionComparison,
}) {
  let busy = false;
  let productionContractSource = null;
  let comparisonRevision = 0;

  elements?.productionContractInput?.addEventListener("change", async () => {
    const file = elements.productionContractInput.files?.[0];
    if (!file) return;
    if (file.size > 4_194_304) {
      onStatus?.("Production contract exceeds the 4 MiB browser limit", "error");
      elements.productionContractInput.value = "";
      return;
    }
    try {
      productionContractSource = await file.text();
      onProductionContractChange?.(productionContractSource);
      const parsed = parseProductionInterfaceContract(productionContractSource);
      onStatus?.(parsed.valid ? "Production interface contract loaded" : "Production contract is invalid", parsed.valid ? "ok" : "error");
      render();
    } catch (error) {
      console.error("[production-interface-contract]", error);
      onStatus?.("Production contract import failed", "error");
    }
  });

  elements?.clearProductionContract?.addEventListener("click", () => {
    productionContractSource = null;
    if (elements.productionContractInput) elements.productionContractInput.value = "";
    onProductionContractChange?.(null);
    onStatus?.("Production interface contract cleared", "ok");
    render();
  });

  for (const [buttonName, artifactName, documentName] of EXPORTS) {
    const button = elements?.[buttonName];
    if (!button) continue;
    button.addEventListener("click", async () => {
      const artifact = TEXT_EXPORT_ARTIFACTS[artifactName];
      const publicExport = PUBLIC_EXPORT_BUTTONS.has(buttonName);
      const documentBuilder = publicExport ? getPublicDocuments || getDocuments : getDocuments;
      try {
        await downloadTextArtifact({
          artifact,
          buildText: async () => jsonForDownload((await documentBuilder()).documents[documentName]),
          getFilename,
          isReady: () => ready(),
          ensureAllowed: publicExport ? async () => true : () => ensureAllowed(artifact.permissionLabel),
          ensureHash,
        });
      } catch (error) {
        console.error("[export-contract]", error);
        onStatus?.("Contract export failed", "error");
      }
    });
  }

  elements?.downloadContractPack?.addEventListener("click", async () => {
    if (!ready() || busy || !(await ensureAllowed("Deployment Contract Pack"))) return;
    busy = true;
    const button = elements.downloadContractPack;
    const label = button.textContent;
    button.disabled = true;
    button.textContent = "Building contract pack";
    try {
      await ensureHash();
      const exportSet = await getDocuments();
      const files = [
        zipTextFile(exportSet.files.cyclonedx, jsonForDownload(exportSet.documents.cyclonedx_evidence)),
        zipTextFile(exportSet.files.artifactEnvelope, jsonForDownload(exportSet.documents.artifact_evidence_envelope)),
        zipTextFile(exportSet.files.artifactIr, jsonForDownload(exportSet.documents.artifact_ir)),
        zipTextFile(exportSet.files.interfaceContracts, jsonForDownload(exportSet.documents.interface_contract_ledger)),
        zipTextFile(exportSet.files.formulation, jsonForDownload(exportSet.documents.observed_formulation)),
        zipTextFile(exportSet.files.runtime, jsonForDownload(exportSet.documents.runtime_requirement_manifest)),
        zipTextFile(exportSet.files.missingFields, jsonForDownload(exportSet.documents.missing_provenance_field_specification)),
      ];
      const packageDigest = await buildCanonicalPackageDigest(files);
      files.push(zipTextFile("deepbom_contract_pack_manifest.json", jsonForDownload({
        schema: "deepbom.deployment_contract_pack_manifest.v1.5",
        generated_at: exportSet.generated_at,
        subject: exportSet.subject,
        contract_set_schema: exportSet.schema,
        contract_set_integrity: exportSet.integrity,
        analyzer_provenance: exportSet.documents.runtime_requirement_manifest.provenance,
        package_hash_method: packageDigest.package_hash_method,
        package_hash_sha256: packageDigest.package_hash_sha256,
        files: packageDigest.files,
      })));
      const signedDigest = await buildCanonicalPackageDigest(files);
      const signature = await signPackageDigest(signedDigest, { scope: "deployment_contract_pack" });
      if (!(await verifyPackageSignature(signature, signedDigest))) throw new Error("Local public-key signature verification failed");
      files.push(zipTextFile("deepbom_public_key_signature.json", jsonForDownload(signature)));
      downloadBlob(getFilename("deployment_contracts.zip"), createZipBlob(files));
      onStatus?.("Deployment Contract Pack downloaded", "ok");
    } catch (error) {
      console.error("[export-contract-pack]", error);
      onStatus?.("Contract pack export failed", "error");
    } finally {
      busy = false;
      button.textContent = label;
      render();
    }
  });

  function ready() {
    const context = getContext?.() || {};
    return Boolean(context.analysis && context.modelBytes);
  }

  function render() {
    const context = getContext?.() || {};
    const analysis = context.analysis;
    const access = getAccess?.() || {};
    const available = Boolean(analysis && context.modelBytes);
    const allowed = Boolean(access.rawExportAllowed);
    const sha = String(analysis?.model_sha256 || "");
    if (elements?.exportContractModel) {
      elements.exportContractModel.textContent = analysis
        ? `${analysis.filename || "model"}${sha ? ` / SHA-256 ${sha.slice(0, 16)}...` : ""}`
        : "Run a static audit";
    }
    if (elements?.exportContractTarget) {
      const target = analysis?.target_profile;
      elements.exportContractTarget.textContent = target
        ? `${target.label || target.id} / ${target.id || "target"}`
        : "No analyzed target";
    }
    if (elements?.exportContractStatus) {
      elements.exportContractStatus.textContent = !available
        ? "No analyzed artifact"
        : "CycloneDX 1.7 and six companion contracts are ready";
      elements.exportContractStatus.classList.toggle("ready", available);
    }
    for (const [buttonName] of EXPORTS) {
      const publicExport = PUBLIC_EXPORT_BUTTONS.has(buttonName);
      updateButton(elements?.[buttonName], available, publicExport || allowed, false, publicExport);
    }
    updateButton(elements?.downloadContractPack, available, allowed, true);
    if (elements?.clearProductionContract) elements.clearProductionContract.disabled = !productionContractSource;
    refreshComparison(available);
  }

  async function refreshComparison(available) {
    const revision = ++comparisonRevision;
    if (!available) {
      renderComparison(null);
      return;
    }
    try {
      const context = getContext?.() || {};
      const ledger = buildInterfaceQuantizationContractLedger(context.analysis);
      const comparison = compareInterfaceContracts(
        ledger,
        productionContractSource,
        context.analysis?.model_sha256,
      );
      if (revision !== comparisonRevision) return;
      renderComparison(comparison, { interface_contract_requirement: { ledger_sha256: ledger.ledger_sha256 } });
      onProductionComparison?.(comparison);
    } catch (error) {
      if (revision !== comparisonRevision) return;
      console.error("[production-interface-comparison]", error);
      renderComparison({ status: "invalid_declaration", mismatch_count: 1, mismatches: [{ field: "comparison", declared: error.message }] });
    }
  }

  function renderComparison(comparison, runtime = null) {
    const status = comparison?.status || "unbound";
    const statusLabel = ({
      bound_exact_contract: "Exact match",
      contradiction_interface_contract_mismatch: "Mismatch",
      contradiction_artifact_hash_mismatch: "Artifact mismatch",
      invalid_declaration: "Invalid declaration",
      partial_artifact_hash_missing: "Artifact hash missing",
      partial_artifact_and_implementation_hash_missing: "Artifact and implementation hashes missing",
      partial_implementation_hash_missing: "Implementation hash missing",
      unbound: "Unbound",
    })[status] || status.replaceAll("_", " ");
    if (elements?.productionContractStatus) {
      elements.productionContractStatus.textContent = statusLabel;
      elements.productionContractStatus.dataset.state = comparison?.gate_result || (status.startsWith("contradiction") ? "block" : "pending");
    }
    if (elements?.productionContractSummary) {
      elements.productionContractSummary.textContent = comparison
        ? `${comparison.declared_parameter_count || 0}/${comparison.expected_parameter_count || 0} parameter declarations · ${comparison.mismatch_count || 0} field differences`
        : "No production declaration loaded";
    }
    renderDiffRows(comparison?.mismatches || [], status);
    renderTrustBoundaries(comparison, runtime);
  }

  function renderDiffRows(rows, status) {
    const body = elements?.productionContractDiffBody;
    if (!body) return;
    body.replaceChildren();
    if (!rows.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 4;
      cell.textContent = status === "bound_exact_contract" ? "All declared fields match the artifact contract." : "Not assessed";
      row.append(cell);
      body.append(row);
      return;
    }
    for (const item of rows.slice(0, 100)) {
      const row = document.createElement("tr");
      for (const value of [item.parameter_id || "document", item.field, compactValue(item.expected), compactValue(item.declared)]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        cell.title = String(value || "");
        row.append(cell);
      }
      body.append(row);
    }
  }

  function renderTrustBoundaries(comparison, runtime) {
    const body = elements?.trustBoundaryBody;
    if (!body) return;
    body.replaceChildren();
    const context = getContext?.() || {};
    const preprocessing = runtime?.preprocessing_contract_requirement?.production_binding;
    const runtimeEvidence = context.runtimeEvidence;
    const boundaries = [
      ["Model artifact", comparison?.expected_artifact_sha256 ? "bound" : "unbound", comparison?.expected_artifact_sha256],
      ["External tensor ABI", comparison?.gate_result || "pending", runtime?.interface_contract_requirement?.ledger_sha256],
      ["Encoder / decoder", comparison?.implementation_sha256 ? "bound" : "unbound", comparison?.implementation_sha256],
      ["Preprocessing", preprocessing?.status === "bound_exact_contract" ? "bound" : "unbound", preprocessing?.implementation_sha256],
      ["Runtime binary", runtimeEvidence?.runtime?.binary_sha256 ? "bound" : "unbound", runtimeEvidence?.runtime?.binary_sha256],
      ["Target profile", context.analysis?.target_profile?.profile_sha256 ? "bound" : "unbound", context.analysis?.target_profile?.profile_sha256],
    ];
    for (const [label, state, digest] of boundaries) {
      const row = document.createElement("div");
      const name = document.createElement("span");
      const value = document.createElement("strong");
      name.textContent = label;
      value.textContent = digest ? `${state} · ${String(digest).slice(0, 12)}...` : state;
      value.dataset.state = state;
      row.append(name, value);
      body.append(row);
    }
  }

  function updateButton(button, available, allowed, pack = false, publicExport = false) {
    if (!button || busy && pack) return;
    button.disabled = !available || !allowed;
    button.classList.toggle("account-locked", available && !allowed);
    button.title = !available
      ? "Run a static audit first."
      : allowed
        ? pack
          ? "Download six contract documents with a member-digest manifest and independently verifiable ES256 signature."
          : publicExport
            ? "Download this standalone CycloneDX document without signing in."
            : "Download this machine-readable model contract."
        : "A verified, authorized account is required.";
  }

  render();
  return { render };
}

function compactValue(value) {
  if (value == null) return "null";
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text.length > 120 ? `${text.slice(0, 117)}...` : text;
}
