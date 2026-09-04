import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { createRoutePreservingDeployConfig } from "./write-cloudflare-deploy-config.mjs";
import worker from "../worker/index.js";

const configPath = "wrangler.jsonc";
const workerPath = "worker/index.js";
const config = JSON.parse(stripJsonComments(readFileSync(configPath, "utf8")));
const deployConfig = createRoutePreservingDeployConfig(config);
const workerSource = readFileSync(workerPath, "utf8");
const swSource = readFileSync("web/sw.js", "utf8");
const appSource = readFileSync("web/app.js", "utf8");
const verifySource = readFileSync("web/verify.html", "utf8");
const buildSource = readFileSync("scripts/build-pages.mjs", "utf8");
const errors = [];

for (const [pathname, sourceType, expectedType] of [
  ["/web/app.js", "text/javascript", "text/javascript; charset=utf-8"],
  ["/web/styles.css", "text/css", "text/css; charset=utf-8"],
  ["/web/report-workspace.css", "text/css", "text/css; charset=utf-8"],
  ["/web/explorer-question.css", "text/css", "text/css; charset=utf-8"],
  ["/web/execution-placement.css", "text/css", "text/css; charset=utf-8"],
  ["/web/onnx-domain.css", "text/css", "text/css; charset=utf-8"],
  ["/web/lib/report-engineering-entry.js", "text/javascript", "text/javascript; charset=utf-8"],
  ["/web/lib/report-regulatory-entry.js", "text/javascript", "text/javascript; charset=utf-8"],
  ["/web/manifest.webmanifest", "application/manifest+json", "application/manifest+json; charset=utf-8"],
  ["/web/pkg/analyzer_bg.wasm", "application/wasm", "application/wasm"],
]) {
  const response = await worker.fetch(new Request(`https://deepbom.test${pathname}`), {
    ASSETS: { fetch: async () => new Response("fixture", { headers: { "content-type": sourceType } }) },
  });
  expect(
    response.headers.get("content-type") === expectedType,
    `${pathname} content type must be ${expectedType}, received ${response.headers.get("content-type")}.`,
  );
}

const anonymousRawFormatter = await worker.fetch(
  new Request("https://deepbom.test/web/lib/report-raw-entry.js"),
  { ASSETS: { fetch: async () => new Response("fixture", { headers: { "content-type": "text/javascript" } }) } },
);
expect(anonymousRawFormatter.status === 200, "Anonymous users must receive the local raw-export formatter entrypoint.");

expect(config.name, "wrangler.jsonc must define a Worker name.");
expect(config.workers_dev === false, "wrangler.jsonc must explicitly disable the workers.dev route.");
expect(
  JSON.stringify(deployConfig.routes) === JSON.stringify(config.routes),
  "CI deploy config must retain the validated production routes so Wrangler activates the uploaded version.",
);
for (const invalidRoute of [null, {}, { pattern: "", zone_name: "deepbom.org" }, { pattern: "deepbom.org/*", zone_name: "" }]) {
  let rejected = false;
  try {
    createRoutePreservingDeployConfig({ ...config, routes: [invalidRoute] });
  } catch {
    rejected = true;
  }
  expect(rejected, "CI deploy config must reject incomplete production route bindings.");
}
expect(deployConfig.name === config.name && deployConfig.main === config.main, "CI deploy config must preserve the Worker identity and entrypoint.");
expect(JSON.stringify(deployConfig.assets) === JSON.stringify(config.assets), "CI deploy config must preserve the static asset binding.");
expect(JSON.stringify(deployConfig.d1_databases) === JSON.stringify(config.d1_databases), "CI deploy config must preserve D1 bindings.");
expect(config.main === `./${workerPath}`, `wrangler.jsonc main should be ./${workerPath}.`);
expect(existsSync(workerPath), `Worker entrypoint is missing: ${workerPath}`);
expect(config.assets?.binding === "ASSETS", "wrangler.jsonc assets.binding must be ASSETS.");
expect(config.assets?.directory === "./dist/", "wrangler.jsonc assets.directory must be ./dist/.");
expect(workerSource.includes("env.ASSETS.fetch"), "worker/index.js must route static assets through env.ASSETS.fetch.");
expect(config.vars?.PASSWORD_AUTH_ENABLED === "false", "Production must expose Google OAuth only until password auth is explicitly enabled.");
expect(
  workerSource.includes("passwordAuthEnabled(env)")
    && workerSource.includes("requirePasswordAuthEnabled(env)")
    && workerSource.includes('error.code = "password_auth_disabled"'),
  "Password signup/login endpoints must fail closed when the production flag is disabled.",
);
expect(
  appSource.includes("passwordTabs: authPasswordTabs")
    && appSource.includes("passwordForm: authForm")
    && appSource.includes("divider: authDivider")
    && appSource.includes("authConfigState.password ? authFormEmail : googleLogin"),
  "The auth modal must remove password surfaces and focus Google when password auth is disabled.",
);

const runWorkerFirst = new Set(config.assets?.run_worker_first || []);
expect(runWorkerFirst.has("/*"), "wrangler.jsonc assets.run_worker_first must include /* so browser security headers cover static pages and assets.");
expect(runWorkerFirst.size === 1, "The global /* Worker-first rule must not retain redundant narrower routes rejected by Wrangler.");
expect(!runWorkerFirst.has("/medical") && !runWorkerFirst.has("/medical/*"), "/medical should be served as a generated static shell, not Worker-first.");
expect(!workerSource.includes("routeMedicalWorkspace"), "worker/index.js should not use Worker compute for the static /medical shell.");
expect(
  buildSource.includes('path.join(dist, "test.html")')
    && buildSource.includes('"Disallow: /test"'),
  "/test must be assembled as a no-index static gateway rather than a duplicate Worker-rendered application.",
);

const d1Bindings = new Set((config.d1_databases || []).map((database) => database.binding));
expect(d1Bindings.has("DB"), "wrangler.jsonc d1_databases must include binding DB.");
expect(workerSource.includes("env.DB"), "worker/index.js must reference env.DB when wrangler.jsonc declares DB.");

const deepBomModulePath = extractWorkerString("module_url");
const deepBomWasmPath = extractWorkerString("wasm_url");
for (const assetPath of [deepBomModulePath, deepBomWasmPath]) {
  expect(assetPath?.startsWith("/web/protected/deepbom/"), `Protected module asset must stay under /web/protected/deepbom/: ${assetPath || "(missing)"}`);
  expect(existsSync(path.join("web", assetPath.replace(/^\/web\//, ""))), `Protected module source asset is missing: ${assetPath}`);
}

expect(
  workerSource.includes('path.startsWith("/web/protected/deepbom/")') && workerSource.includes("requireDeepBomAccess"),
  "worker/index.js must require DEEPBOM authorization before serving /web/protected/deepbom assets.",
);
expect(
  workerSource.includes("isProtectedReportFormatterAsset") && workerSource.includes("requireReportFormatterAccess"),
  "worker/index.js must require report formatter authorization before serving controlled report modules.",
);
expect(
  workerSource.includes('path === "/api/test/activate"')
    && workerSource.includes("verifyExternalTestLink")
    && workerSource.includes("account_bound: accountBound")
    && workerSource.includes("admin_access: false"),
  "External testing must verify a 24-hour bearer link while excluding Admin and account-bound operations.",
);
expect(
  workerSource.includes("const PROTECTED_RAW_EXPORT_ASSETS = new Set([])")
    && workerSource.includes("const PROTECTED_SHARED_REPORT_ASSETS = new Set([])"),
  "Local report and evidence formatters must remain public static assets.",
);
const protectedReportAssets = new Set([
  ...extractStringSet(workerSource, "PROTECTED_RAW_EXPORT_ASSETS"),
  ...extractStringSet(workerSource, "PROTECTED_SHARED_REPORT_ASSETS"),
]);
const authenticatedSwAssets = new Set(extractStringSet(swSource, "AUTHENTICATED_ASSET_PATHS"));
for (const asset of new Set([...protectedReportAssets, ...authenticatedSwAssets])) {
  expect(
    protectedReportAssets.has(asset) === authenticatedSwAssets.has(asset),
    `Worker authorization and service-worker no-cache boundaries disagree for ${asset}.`,
  );
}
expect(
  workerSource.includes("withBrowserSecurityHeaders")
    && workerSource.includes("isUtf8TextContentType")
    && workerSource.includes('charset=utf-8')
    && workerSource.includes("content-security-policy")
    && workerSource.includes("x-content-type-options")
    && workerSource.includes("frame-ancestors 'none'")
    && workerSource.includes("strict-transport-security"),
  "static browser responses must declare UTF-8 text, CSP, anti-sniffing, anti-framing, and transport security headers.",
);
expect(
  workerSource.includes("connect-src 'self' https://storage.googleapis.com")
    && !workerSource.includes("connect-src *")
    && !workerSource.includes("connect-src https:"),
  "CSP must allow generation-pinned Google sample downloads without opening connect-src to arbitrary HTTPS origins.",
);
expect(
  workerSource.includes('includes("text/html")')
    && workerSource.includes('headers.set("cache-control", "no-store, no-cache, must-revalidate, no-transform")'),
  "HTML responses must opt out of edge transformation so Cloudflare cannot inject a beacon that the self-only CSP blocks.",
);
expect(
  workerSource.includes("return withBrowserSecurityHeaders(new Response(JSON.stringify(body)"),
  "Worker JSON and authorization-error responses must receive the same browser security headers.",
);
expect(
  workerSource.includes("if (!consented)")
    && workerSource.includes("UPDATE consent")
    && !workerSource.includes("consented ? now : null"),
  "Consent withdrawal must not insert NULL into the consented_at NOT NULL column.",
);
expect(
  workerSource.includes("hmacVerificationTagHex")
    && !workerSource.includes("ed25519SignHex")
    && workerSource.includes("authentication_tag")
    && workerSource.includes('authentication_algorithm: authenticationTag ? "HS256"'),
  "report fingerprint registration must label its HMAC authentication tag accurately.",
);
expect(
  !verifySource.includes("innerHTML")
    && verifySource.includes("does not prove that DEEPBOM generated or issued the report")
    && verifySource.includes("Fingerprint match"),
  "public fingerprint comparison must avoid stored XSS and must not claim report origin or authenticity.",
);
expect(
  workerSource.includes('url.pathname.startsWith("/api/report/")') && workerSource.includes('path === "/api/report/sign"') && workerSource.includes("signReportManifest"),
  "worker/index.js must expose the signed report manifest endpoint through /api/report/sign.",
);
expect(
  workerSource.includes("report_uploaded: false") && workerSource.includes("package_hash_sha256") && workerSource.includes("hmacSha256Base64Url"),
  "signed report manifests must sign hashes/metadata only and avoid report or model upload.",
);
expect(
  workerSource.includes("deepbom.package_member_digest_set.v1") && workerSource.includes("sanitizePackageMembers") && workerSource.includes("package_hash_mismatch"),
  "attestation endpoint must recompute the canonical package-member digest-set hash.",
);
expect(
  workerSource.includes('schema: "deepbom.attestation_payload.v2.2"') && workerSource.includes('canonicalization: "RFC8785-JCS"') && workerSource.includes("attestation_scope"),
  "attestation payload must declare its schema, canonicalization, and signed interpretation boundary.",
);
expect(
  workerSource.includes("reportSigningKeyMetadata") && workerSource.includes("deepbom-hs256-") && !workerSource.includes("deepbom-hmac-current"),
  "attestation signing must use an immutable configured or secret-derived key ID, not a mutable current/latest alias.",
);
expect(
  workerSource.includes("metadata_valid_from") && workerSource.includes("verification_status") && config.vars?.REPORT_SIGNING_KEY_VALID_FROM,
  "attestation must bind signing-key lifecycle metadata and return an explicit verification receipt.",
);
expect(
  workerSource.includes("account_identity_embedded: false") && workerSource.includes("capability_matrix_embedded: false") && workerSource.includes("filename_or_filename_hash_embedded: false"),
  "external attestation must explicitly exclude account, capability, and filename identifiers.",
);
expect(
  workerSource.includes('verification: "server-only; the HMAC secret is not distributed to package recipients"'),
  "HS256 attestation must be labeled as server-only rather than independently verifiable.",
);
expect(
  appSource.includes("buildCanonicalPackageDigest(files)") && appSource.includes("validatePackageAttestation(signature, packageDigest)") && appSource.includes('zipTextFile("attestation.json"') && !appSource.includes("filename_hash_sha256: filenameHash"),
  "client bundle signing must use canonical member digests, validate the receipt, emit attestation.json, and avoid filename hashes.",
);
expect(
  workerSource.includes('"raw_data"') && workerSource.includes('"engineering_bundle"') && workerSource.includes("payload.allowed.raw_export"),
  "worker/index.js must require raw_export authorization for raw_data and engineering_bundle signing scopes.",
);

if (errors.length) {
  throw new Error(`Worker deployment config check failed:\n${errors.map((error) => `  - ${error}`).join("\n")}`);
}

console.log("Worker deployment config check passed.");

function expect(condition, message) {
  if (!condition) errors.push(message);
}

function extractWorkerString(key) {
  return new RegExp(`${key}:\\s*"([^"]+)"`).exec(workerSource)?.[1] || "";
}

function extractStringSet(source, name) {
  const body = new RegExp(`const ${name} = new Set\\(\\[([\\s\\S]*?)\\]\\);`).exec(source)?.[1] || "";
  return [...body.matchAll(/"([^"]+)"/g)].map((match) => match[1]);
}

function stripJsonComments(source) {
  let output = "";
  let inString = false;
  let escaped = false;
  let lineComment = false;
  let blockComment = false;

  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    const next = source[index + 1];

    if (lineComment) {
      if (char === "\n") {
        lineComment = false;
        output += char;
      }
      continue;
    }

    if (blockComment) {
      if (char === "*" && next === "/") {
        blockComment = false;
        index += 1;
      }
      continue;
    }

    if (!inString && char === "/" && next === "/") {
      lineComment = true;
      index += 1;
      continue;
    }

    if (!inString && char === "/" && next === "*") {
      blockComment = true;
      index += 1;
      continue;
    }

    output += char;

    if (escaped) {
      escaped = false;
    } else if (char === "\\") {
      escaped = true;
    } else if (char === '"') {
      inString = !inString;
    }
  }

  return output;
}
