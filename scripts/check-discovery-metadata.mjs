import { readFileSync } from "node:fs";

const html = readFileSync("web/index.html", "utf8");
const manifest = JSON.parse(readFileSync("web/manifest.webmanifest", "utf8"));
const buildPages = readFileSync("scripts/build-pages.mjs", "utf8");
const errors = [];

for (const [condition, message] of [
  [html.includes("<title>DEEPBOM | Deployment Artifact Evidence for On-Device AI</title>"), "canonical page title is current"],
  [html.includes('rel="canonical" href="https://deepbom.org/"'), "canonical URL is absolute"],
  [html.includes('property="og:title" content="DEEPBOM | Deployment Artifact Evidence for On-Device AI"'), "Open Graph title matches the page title"],
  [html.includes('name="twitter:title" content="DEEPBOM | Deployment Artifact Evidence for On-Device AI"'), "Twitter title matches the page title"],
  [buildPages.includes('title = "DEEPBOM | Deployment Artifact Evidence for On-Device AI"'), "root production shell preserves the canonical page title"],
  [!html.includes("TFLite &amp; ONNX Model Static Analyzer") && !html.includes("TFLite & ONNX Model Static Analyzer"), "retired search title is absent"],
  [manifest.name.includes("Deployment Artifact Evidence Analyzer"), "PWA name uses the artifact-evidence identity"],
  [!html.includes('"@type": "Offer"') && !html.includes('"price"') && !html.includes('"priceCurrency"'), "public metadata contains no commercial offer or price"],
  [!/(pricing|subscription|purchase|member exports|paid access|commercial tier)/i.test(html), "public shell contains no pricing or commercial access-tier language"],
  [html.includes("editable HTML engineering report") && !html.includes("Markdown engineering report"), "public metadata matches the editable HTML Engineering Report export"],
  [html.includes("p50/p90/p95/p99 statistics"), "public benchmark metadata lists every reported percentile"],
  [buildPages.includes('"Sitemap: https://deepbom.org/sitemap.xml"'), "generated robots file advertises the canonical sitemap"],
  [buildPages.includes('"    <loc>https://deepbom.org/</loc>"') && buildPages.includes('"    <loc>https://deepbom.org/verify</loc>"'), "generated sitemap lists the canonical app and report verifier"],
  [["regulatory", "quality", "engineering"].every((brief) => buildPages.includes(`"${brief}"`))
    && buildPages.includes("https://deepbom.org/evaluate/${brief}/"), "generated sitemap lists all evaluation briefs"],
  [!buildPages.includes('"    <loc>https://deepbom.org/web/</loc>"'), "generated sitemap does not index the duplicate /web/ shell"],
  [buildPages.includes("const today = new Date().toISOString().slice(0, 10)"), "generated sitemap receives the build date"],
]) {
  if (!condition) errors.push(message);
}

if (errors.length) {
  console.error("Discovery metadata check failed:");
  for (const error of errors) console.error(`- ${error}`);
  process.exit(1);
}

console.log("Discovery metadata check passed (title, social cards, PWA identity, and canonical build-generated discovery files).\n");
