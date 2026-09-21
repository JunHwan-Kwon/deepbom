import { readFileSync } from "node:fs";

const html = readFileSync("web/index.html", "utf8");
const manifest = JSON.parse(readFileSync("web/manifest.webmanifest", "utf8"));
const buildPages = readFileSync("scripts/build-pages.mjs", "utf8");
const styles = readFileSync("web/styles.css", "utf8");
const agentPage = readFileSync("web/for-agents/index.html", "utf8");
const guidePaths = [
  "web/guides/index.html",
  "web/guides/inspect-onnx-quantization/index.html",
  "web/guides/inspect-gguf-tensor-encodings/index.html",
  "web/guides/compare-model-artifacts/index.html",
];
const guidePages = guidePaths.map((file) => readFileSync(file, "utf8"));
const packageVersion = JSON.parse(readFileSync("package.json", "utf8")).version;
const errors = [];

// The linked-data graph is what lets an answer engine resolve "DEEPBOM" to
// this software, this author, and this record rather than guessing. Parse it
// rather than string-matching so a malformed block fails here.
const BRIEFS = ["regulatory", "quality", "engineering"];
const briefPages = Object.fromEntries(
  BRIEFS.map((brief) => [brief, readFileSync(`web/evaluate/${brief}/index.html`, "utf8")]),
);

function linkedData(source, label) {
  const blocks = [...source.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)];
  if (!blocks.length) return errors.push(`${label} carries no linked data`) && [];
  try {
    return blocks.flatMap((block) => {
      const parsed = JSON.parse(block[1]);
      return parsed["@graph"] || (Array.isArray(parsed) ? parsed : [parsed]);
    });
  } catch (error) {
    errors.push(`${label} linked data is not valid JSON: ${error.message}`);
    return [];
  }
}

const rootNodes = linkedData(html, "web/index.html");
const agentNodes = linkedData(agentPage, "web/for-agents/index.html");
const application = rootNodes.find((node) => node["@type"] === "SoftwareApplication") || {};
const briefNodes = Object.fromEntries(
  BRIEFS.map((brief) => [brief, linkedData(briefPages[brief], `evaluate/${brief}`)]),
);

for (const [condition, message] of [
  [html.includes("<title>DEEPBOM | Local AI Model Analyzer</title>"), "canonical page title is current"],
  [html.includes('rel="canonical" href="https://deepbom.org/"'), "canonical URL is absolute"],
  [html.includes('property="og:title" content="DEEPBOM | Local AI Model Analyzer"'), "Open Graph title matches the page title"],
  [html.includes('name="twitter:title" content="DEEPBOM | Local AI Model Analyzer"'), "Twitter title matches the page title"],
  [buildPages.includes('title = "DEEPBOM | Local AI Model Analyzer"'), "root production shell preserves the canonical page title"],
  [!html.includes("TFLite &amp; ONNX Model Static Analyzer") && !html.includes("TFLite & ONNX Model Static Analyzer"), "retired search title is absent"],
  [manifest.name.includes("Deployment Artifact Evidence Analyzer"), "PWA name uses the artifact-evidence identity"],
  [!html.includes('"@type": "Offer"') && !html.includes('"price"') && !html.includes('"priceCurrency"'), "public metadata contains no commercial offer or price"],
  [!/(pricing|subscription|purchase|member exports|paid access|commercial tier)/i.test(html), "public shell contains no pricing or commercial access-tier language"],
  [html.includes("editable HTML engineering report") && !html.includes("Markdown engineering report"), "public metadata matches the editable HTML Engineering Report export"],
  [html.includes("p50/p90/p95/p99 statistics"), "public benchmark metadata lists every reported percentile"],
  [html.includes("Optional browser-observed runtime benchmark") && html.includes("reported separately from CLI and artifact-only static evidence"),
    "browser-observed benchmark metadata is separated from static agent and CLI evidence"],
  [buildPages.includes('"Sitemap: https://deepbom.org/sitemap.xml"'), "generated robots file advertises the canonical sitemap"],
  [buildPages.includes('"    <loc>https://deepbom.org/</loc>"') && buildPages.includes('"    <loc>https://deepbom.org/verify</loc>"'), "generated sitemap lists the canonical app and report verifier"],
  [buildPages.includes('"    <loc>https://deepbom.org/for-agents/</loc>"'), "generated sitemap lists the local agent guide"],
  [buildPages.includes("https://deepbom.org/guides/${guide}")
    && ["inspect-onnx-quantization/", "inspect-gguf-tensor-encodings/", "compare-model-artifacts/"].every((slug) => buildPages.includes(slug)),
    "generated sitemap lists all problem-focused inspection guides"],
  [["regulatory", "quality", "engineering"].every((brief) => buildPages.includes(`"${brief}"`))
    && buildPages.includes("https://deepbom.org/evaluate/${brief}/"), "generated sitemap lists all evaluation briefs"],
  [!buildPages.includes('"    <loc>https://deepbom.org/web/</loc>"'), "generated sitemap does not index the duplicate /web/ shell"],
  [buildPages.includes('import { sitemapLastmod } from "./sitemap-lastmod.mjs"')
    && !buildPages.includes("<lastmod>${today}"), "sitemap modification dates come from source revisions, not deployment time"],

  // llms.txt is the plain-text place an assistant learns the execution and
  // transfer boundaries. The remote endpoint is a control plane; local and
  // browser paths perform analysis without uploading model bytes to DEEPBOM.
  [buildPages.includes('path.join(dist, "llms.txt")'), "build generates llms.txt at the domain root"],
  [buildPages.includes('path.join(dist, "agent-capabilities.json")'), "build generates the machine-readable agent capability contract"],
  [buildPages.includes('"# https://deepbom.org/llms.txt",'), "generated robots file points at llms.txt"],
  [buildPages.includes('"User-agent: OAI-SearchBot"'), "robots file explicitly permits OAI-SearchBot"],
  [buildPages.includes("The remote MCP control plane serves tool metadata and validates the bounded")
    && buildPages.includes("it does not fetch or retain model bytes"),
    "llms.txt states the remote MCP control-plane and model-byte boundary"],
  [buildPages.includes("Remote MCP endpoint: https://deepbom.org/mcp"),
    "llms.txt publishes the stable ChatGPT Streamable HTTP endpoint"],
  [["deepbom_capabilities", "deepbom_audit", "deepbom_diff", "deepbom_explain_rule"].every((tool) => buildPages.includes(`\`${tool}\``)),
    "llms.txt lists all four local MCP tools"],
  [["artifact_defect", "caution", "evidence_gap"].every((kind) => buildPages.includes(`\`${kind}\``)),
    "llms.txt keeps the three finding kinds distinct"],

  // Entity resolution: the application must point at the records that identify
  // it, not only the author node.
  [Array.isArray(application.sameAs) && application.sameAs.some((url) => url.includes("github.com/"))
    && application.sameAs.some((url) => url.includes("doi.org/")), "application entity links its repository and DOI through sameAs"],
  [typeof application.license === "string" && application.license.length > 0, "application entity declares a license"],
  [rootNodes.some((node) => node["@type"] === "FAQPage" && Array.isArray(node.mainEntity) && node.mainEntity.length >= 4),
    "FAQ linked data carries at least four answered questions"],
  [html.includes('class="product-about"') && !html.includes('class="seo-footer"') && !styles.includes(".seo-footer"),
    "product information is available to visitors rather than hidden in a crawler-only footer"],
  [agentPage.includes('rel="canonical" href="https://deepbom.org/for-agents/"'), "agent guide has a canonical URL"],
  [agentPage.includes("there is no hosted DEEPBOM analysis endpoint") && agentPage.includes("browser-sandbox integration"),
    "agent guide distinguishes local execution from the ChatGPT browser-sandbox path"],
  [agentPage.includes("browser-observed runtime benchmark") && agentPage.includes("must not describe static output as measured latency"),
    "agent guide separates optional browser measurement from static analysis"],
  [agentPage.includes("integrate codex") && agentPage.includes("integrate claude-code") && agentPage.includes("--apply"),
    "agent guide documents preview-before-apply installation"],
  [agentPage.includes(`deepbom@${packageVersion}`) && !agentPage.includes("verified-version"),
    "agent guide pins every zero-install command to the verified release"],
  [agentPage.includes(`deepbom-${packageVersion}.mcpb`) && agentPage.includes(`channels-v${packageVersion}`),
    "agent guide links the version-matched local Claude Desktop extension"],
  [agentNodes.some((node) => node["@type"] === "TechArticle" && node.url === "https://deepbom.org/for-agents/"),
    "agent guide carries a citable TechArticle identity"],
  [guidePages.every((page) => page.includes('rel="canonical" href="https://deepbom.org/guides/')),
    "every problem guide has a canonical URL"],
  [guidePages.every((page) => page.includes("application/ld+json")),
    "every problem guide has linked data"],
  [guidePages[1].includes("c19f7155f030dc396aad04909b7429b8a4d6d1cb51fcd53a84e3c97ae9678d0e")
    && guidePages[1].includes("Runtime graph fusion"),
    "ONNX guide binds its public fixture and preserves the runtime boundary"],
  [guidePages[2].includes("does not redistribute or claim measurements for a third-party GGUF"),
    "GGUF guide does not fabricate or redistribute third-party evidence"],
  [guidePages[3].includes("does not establish clinical equivalence"),
    "artifact-diff guide separates observed change from acceptance decisions"],

  // Each brief is a citable page in the sitemap, so it needs its own identity
  // and its own link back to the software and the author.
  ...BRIEFS.flatMap((brief) => {
    const nodes = briefNodes[brief];
    const article = nodes.find((node) => String(node["@type"]).endsWith("Article")) || {};
    const page = briefPages[brief];
    return [
      [Boolean(article.headline) && article.url === `https://deepbom.org/evaluate/${brief}/`,
        `${brief} brief declares an article node with its canonical URL`],
      [article.author?.["@id"] === "https://deepbom.org/#author" && article.about?.["@id"] === "https://deepbom.org/#app",
        `${brief} brief links its author and the software it describes`],
      [nodes.some((node) => node["@type"] === "BreadcrumbList"), `${brief} brief carries a breadcrumb trail`],
      [page.includes('property="og:title"') && page.includes('name="twitter:card"'),
        `${brief} brief carries social card metadata`],
    ];
  }),
]) {
  if (!condition) errors.push(message);
}

// Each structured answer must have the same readable HTML counterpart.
const readableText = (text) => text.replace(/<[^>]*>/g, "").replaceAll("&amp;", "&")
  .replaceAll("&#x27;", "'").replaceAll("&#39;", "'").replaceAll("&quot;", '"').replace(/\s+/g, " ").trim();
const visibleFaq = [...html.matchAll(/<details class="product-about-question"><summary>(.*?)<\/summary><p>(.*?)<\/p><\/details>/gs)]
  .map((match) => ({ question: readableText(match[1]), answer: readableText(match[2]) }));
const schemaFaq = rootNodes.find((node) => node["@type"] === "FAQPage")?.mainEntity || [];
if (visibleFaq.length !== schemaFaq.length || schemaFaq.some((entry, index) =>
  entry.name !== visibleFaq[index]?.question || entry.acceptedAnswer?.text !== visibleFaq[index]?.answer)) {
  errors.push("FAQ structured data must match the user-readable questions and answers");
}

for (const relative of [
  ...guidePaths.slice(1), "web/guides/cli/index.html", "web/guides/cli/reference/index.html",
  ...BRIEFS.map((brief) => `web/evaluate/${brief}/index.html`),
  "web/for-agents/index.html", "web/chatgpt/index.html", "web/claude/index.html",
]) {
  const source = readFileSync(relative, "utf8");
  const breadcrumb = linkedData(source, relative).find((node) => node["@type"] === "BreadcrumbList");
  const trail = source.match(/<nav class="breadcrumbs"[^>]*>(.*?)<\/nav>/s)?.[1] || "";
  const names = [...trail.matchAll(/<li>(.*?)<\/li>/gs)].map((match) => readableText(match[1]));
  if (JSON.stringify(names) !== JSON.stringify(breadcrumb?.itemListElement?.map((item) => item.name))) {
    errors.push(`${relative} structured breadcrumb differs from its visible reading path`);
  }
}

if (errors.length) {
  console.error("Discovery metadata check failed:");
  for (const error of errors) console.error(`- ${error}`);
  process.exit(1);
}

console.log("Discovery metadata check passed (title, social cards, PWA identity, and canonical build-generated discovery files).\n");
