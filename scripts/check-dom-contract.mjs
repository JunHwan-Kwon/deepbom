import { readFileSync } from "node:fs";

const checks = [
  {
    label: "main app",
    htmlPath: "web/index.html",
    jsPaths: ["web/app.js", "web/lib/elements.js", "web/lib/app-surface.js"],
    optionalIds: new Set([
      "regulatoryReportPreview",
      "regulatoryReportPreviewTitle",
      "regulatoryReportPreviewStatus",
      "downloadRegulatoryReport",
      "downloadEvidenceBundle",
      "evidenceBundleNote",
      "evidenceBundleScope",
    ]),
  },
  {
    label: "admin app",
    htmlPath: "web/admin.html",
    jsPaths: ["web/admin.js", "web/lib/admin-elements.js"],
  },
];

const errors = [];
const optionalEmptySelectors = new Set([".module-panel [data-scroll-target]"]);
let checkedIds = 0;
let checkedSelectors = 0;
let checkedLinks = 0;
let checkedLayouts = 0;

function readText(path) {
  return readFileSync(path, "utf8");
}

function htmlIds(html) {
  return new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]));
}

function classNames(html) {
  const names = new Set();
  for (const match of html.matchAll(/\bclass="([^"]+)"/g)) {
    for (const name of match[1].split(/\s+/)) {
      if (name) {
        names.add(name);
      }
    }
  }
  return names;
}

function dataAttributes(html) {
  return new Set([...html.matchAll(/\s(data-[a-zA-Z0-9_-]+)(?:=|\s|>)/g)].map((match) => match[1]));
}

function dataValues(html, attr) {
  return new Set([...html.matchAll(new RegExp(`\\b${attr}="([^"]+)"`, "g"))].map((match) => match[1]));
}

function literalDocumentIds(js) {
  return [...js.matchAll(/(?:document|doc)\.getElementById\("([^"]+)"\)/g)].map((match) => match[1]);
}

function literalDocumentSelectors(js) {
  const selectors = [];
  for (const method of ["querySelector", "querySelectorAll"]) {
    const pattern = new RegExp(`(?:document|doc)\\.${method}\\("([^"]+)"\\)`, "g");
    for (const match of js.matchAll(pattern)) {
      selectors.push(match[1]);
    }
  }
  return selectors;
}

function selectorContract(selector) {
  const classOnly = selector.match(/^\.([a-zA-Z0-9_-]+)$/);
  if (classOnly) {
    return { classes: [classOnly[1]], dataAttrs: [] };
  }

  const dataOnly = selector.match(/^\[(data-[a-zA-Z0-9_-]+)\]$/);
  if (dataOnly) {
    return { classes: [], dataAttrs: [dataOnly[1]] };
  }

  const classWithData = selector.match(/^\.([a-zA-Z0-9_-]+)\s+\[(data-[a-zA-Z0-9_-]+)\]$/);
  if (classWithData) {
    return { classes: [classWithData[1]], dataAttrs: [classWithData[2]] };
  }

  return null;
}

function checkIds(label, jsPath, htmlPath, js, ids, optionalIds = new Set()) {
  for (const id of literalDocumentIds(js)) {
    checkedIds += 1;
    if (!ids.has(id) && !optionalIds.has(id)) {
      errors.push(`${label}: ${jsPath} references #${id}, but ${htmlPath} does not define it.`);
    }
  }
}

function checkSelectors(label, jsPath, htmlPath, js, classes, attrs) {
  for (const selector of literalDocumentSelectors(js)) {
    const contract = selectorContract(selector);
    if (!contract) {
      continue;
    }
    checkedSelectors += 1;
    const optionalEmpty = optionalEmptySelectors.has(selector);
    for (const name of contract.classes) {
      if (!classes.has(name)) {
        errors.push(`${label}: selector "${selector}" in ${jsPath} expects .${name}, but ${htmlPath} does not define that class.`);
      }
    }
    for (const attr of contract.dataAttrs) {
      if (!attrs.has(attr) && !optionalEmpty) {
        errors.push(`${label}: selector "${selector}" in ${jsPath} expects ${attr}, but ${htmlPath} does not define that attribute.`);
      }
    }
  }
}

function checkTargetLinks(label, htmlPath, html, ids) {
  for (const target of dataValues(html, "data-scroll-target")) {
    checkedLinks += 1;
    if (!ids.has(target)) {
      errors.push(`${label}: ${htmlPath} data-scroll-target="${target}" does not point to an existing id.`);
    }
  }
}

function checkModuleLinks(label, htmlPath, html) {
  const tabs = dataValues(html, "data-module-tab");
  const panels = dataValues(html, "data-module-panel");
  const runPanels = dataValues(html, "data-module-run-panel");

  for (const tab of tabs) {
    checkedLinks += 2;
    if (!panels.has(tab)) {
      errors.push(`${label}: ${htmlPath} data-module-tab="${tab}" has no matching data-module-panel.`);
    }
    if (!runPanels.has(tab)) {
      errors.push(`${label}: ${htmlPath} data-module-tab="${tab}" has no matching data-module-run-panel.`);
    }
  }
}

function checkFullWidthPanel(label, htmlPath, html, childId) {
  checkedLayouts += 1;
  const childOffset = html.indexOf(`id="${childId}"`);
  const articleOffset = childOffset >= 0 ? html.lastIndexOf("<article", childOffset) : -1;
  const articleEnd = articleOffset >= 0 ? html.indexOf(">", articleOffset) : -1;
  const openingTag = articleEnd >= 0 ? html.slice(articleOffset, articleEnd + 1) : "";
  const classes = openingTag.match(/\bclass="([^"]+)"/)?.[1]?.split(/\s+/) || [];
  if (childOffset < 0 || articleOffset < 0 || !classes.includes("perf-panel") || !classes.includes("wide")) {
    errors.push(`${label}: ${htmlPath} #${childId} must be inside a full-width perf panel.`);
  }
}

for (const { label, htmlPath, jsPaths, optionalIds = new Set() } of checks) {
  const html = readText(htmlPath);
  const ids = htmlIds(html);
  const classes = classNames(html);
  const attrs = dataAttributes(html);

  for (const jsPath of jsPaths) {
    const js = readText(jsPath);
    checkIds(label, jsPath, htmlPath, js, ids, optionalIds);
    checkSelectors(label, jsPath, htmlPath, js, classes, attrs);
  }
  checkTargetLinks(label, htmlPath, html, ids);
  checkModuleLinks(label, htmlPath, html);
  if (htmlPath === "web/index.html") {
    checkFullWidthPanel(label, htmlPath, html, "xnnpackFallbackMap");
  }
}

if (errors.length > 0) {
  console.error("DOM contract check failed:");
  for (const error of errors) {
    console.error(`- ${error}`);
  }
  process.exit(1);
}

console.log(
  `DOM contract check passed (${checkedIds} ids, ${checkedSelectors} selectors, ${checkedLinks} linked targets, ${checkedLayouts} layout contracts).`,
);
