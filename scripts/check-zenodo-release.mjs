import { readFileSync } from "node:fs";
import {
  PROTECTED_FILES,
  PROTECTED_PREFIXES,
  analyzerIdentity,
  collectSoftwareEntries,
  citationRecordVersion,
  publicationBlockers,
  softwareVersion,
} from "./zenodo-release-lib.mjs";

function expect(condition, message) {
  if (!condition) throw new Error(message);
}

const version = softwareVersion();
const citationVersion = citationRecordVersion();
const identity = analyzerIdentity();
const citation = readFileSync("CITATION.cff", "utf8");
const zenodo = JSON.parse(readFileSync(".zenodo.json", "utf8"));
const datasetMetadata = JSON.parse(readFileSync("docs/zenodo-validation-metadata.json", "utf8"));
const releaseGuide = readFileSync("docs/ZENODO_RELEASE.md", "utf8");
const reproducibility = readFileSync("docs/REPRODUCIBILITY.md", "utf8");
const appConfig = readFileSync("web/lib/app-config.js", "utf8");
const indexHtml = readFileSync("web/index.html", "utf8");
const entries = collectSoftwareEntries();
const names = new Set(entries.map((item) => item.path));
const versionDoi = "10.5281/zenodo.21834509";
const conceptDoi = "10.5281/zenodo.21834508";
const recommendedCitation = `Kwon, J. (2026). DEEPBOM: Browser-Native Static Analysis of On-Device Neural Network Deployment Artifacts (Version ${citationVersion}) [Computer software]. Zenodo. https://doi.org/${versionDoi}`;
const browserCitation = `Kwon, J. (2026). DEEPBOM: Browser-Native Static Analysis of On-Device Neural Network Deployment Artifacts [Computer software]. Zenodo concept DOI. https://doi.org/${conceptDoi}`;

expect(citation.includes("0000-0002-6464-3895"), "CITATION.cff must bind the creator ORCID.");
expect(citation.includes(`doi: "${versionDoi}"`) && citation.includes(recommendedCitation), "CITATION.cff must bind the published version DOI and recommended citation.");
expect(appConfig.includes(browserCitation) && !appConfig.includes(`Version ${citationVersion}`), "The browser copy source must use the release-independent concept DOI without pinning the old citation-record version.");
expect(indexHtml.includes(`name="citation_doi" content="${conceptDoi}"`) && indexHtml.includes('id="copyCitationBtn"'), "The public entrypoint must expose the concept DOI and citation copy control.");
expect(indexHtml.includes(browserCitation.replace(`https://doi.org/${conceptDoi}`, `<a href="https://doi.org/${conceptDoi}" rel="noopener" target="_blank">https://doi.org/${conceptDoi}</a>`)), "The public citation section must render the release-independent citation.");
expect(indexHtml.includes("The analyzed release version is recorded separately"), "The public citation section must explain where the analyzed software version is bound.");
expect(!indexHtml.includes("an archival DOI is planned"), "The public entrypoint must not claim that the published DOI is still planned.");
expect(!/^license:/m.test(citation) && !/^repository-code:/m.test(citation), "The citation record must not advertise an open-source license or repository.");
for (const format of ["TFLite", "ONNX", "GGUF", "SafeTensors", "Core ML"]) {
  expect(citation.includes(format) && zenodo.description.includes(format), `Citation and Zenodo descriptions must include ${format}.`);
}
expect(
  zenodo.upload_type === "software"
    && zenodo.access_right === "restricted"
    && typeof zenodo.access_conditions === "string"
    && zenodo.access_conditions.length > 40
    && !Object.hasOwn(zenodo, "license"),
  "Zenodo metadata must describe a restricted citation dossier without implying that absent implementation components are distributed.",
);
expect(zenodo.creators?.[0]?.orcid === "0000-0002-6464-3895", "Zenodo metadata must bind the creator ORCID.");
expect(datasetMetadata.version === citationVersion && datasetMetadata.upload_type === "dataset" && datasetMetadata.license === "cc-by-4.0", "Validation metadata must describe the published citation-record version and CC BY 4.0 dataset.");
expect(releaseGuide.includes("source code") && releaseGuide.includes("WebAssembly"), "Zenodo release guide must disclose code and executable exclusions.");
expect(reproducibility.includes("OBSERVED") && reproducibility.includes("DERIVED") && reproducibility.includes("PREDICTED"), "Reproducibility guide must define evidence classes.");
expect(identity.semantic_version === version, "Analyzer semantic version must match the current package version.");
expect(
  JSON.stringify(publicationBlockers("validation", { dirty: true, commit: "release" }, {
    validation_git_worktree_dirty: true,
    validation_git_commit: "old",
  })) === JSON.stringify(["worktree_dirty", "validation_worktree_dirty", "validation_commit_mismatch"]),
  "Publication blockers must distinguish repository and validation provenance failures.",
);

for (const name of names) {
  expect(!PROTECTED_FILES.has(name), `Protected file entered the release: ${name}`);
  expect(!PROTECTED_PREFIXES.some((prefix) => name.startsWith(prefix)), `Protected path entered the release: ${name}`);
}
const expectedRecordFiles = new Set([
  ".zenodo.json",
  "CHANGELOG.md",
  "CITATION.cff",
  "docs/REPRODUCIBILITY.md",
  "docs/SOFTWARE_RECORD.md",
]);
expect(names.size === expectedRecordFiles.size, `Citation record has an unexpected file count: ${names.size}.`);
for (const required of expectedRecordFiles) {
  expect(names.has(required), `Citation record is missing ${required}.`);
}
for (const name of names) {
  expect(!/\.(?:js|mjs|rs|wasm|tflite|onnx|gguf|safetensors|mlmodel)$/i.test(name), `Code, executable, or model content entered the citation record: ${name}.`);
}

console.log(`Zenodo citation contract passed (published ${citationVersion}; current software ${version}; ${entries.length} restricted non-code dossier files).`);
