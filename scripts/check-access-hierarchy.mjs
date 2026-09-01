import { existsSync, readFileSync } from "node:fs";
import { accessProfileForUser, canonicalAccessProfile, hasAccessProfile } from "../web/lib/access-policy.js";
import { moduleAccessStatesFor, selectableModuleIdFor } from "../web/lib/auth-labels.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual } = createCheck("Access hierarchy contract check");
const worker = readFileSync("worker/index.js", "utf8");
const app = readFileSync("web/app.js", "utf8");
const html = readFileSync("web/index.html", "utf8");
const admin = readFileSync("web/admin.js", "utf8");
const testPage = readFileSync("web/test.html", "utf8");
const testClient = readFileSync("web/test.js", "utf8");

expectEqual(canonicalAccessProfile("verified"), "verified", "Verified must remain a canonical access profile.");
expectEqual(canonicalAccessProfile("research"), "research", "Research must remain a canonical access profile.");
expectEqual(canonicalAccessProfile("medical_ai"), "medical_ai", "Medical AI must remain a canonical access profile.");
expectEqual(canonicalAccessProfile("unrecognized-profile"), "verified", "Unknown profiles must fail closed to the verified-account scope.");
expectEqual(accessProfileForUser({ role: "admin", access_profile: "verified" }), "admin", "Admin role should dominate the stored access profile.");
expect(hasAccessProfile("medical_ai", "research"), "Regulatory workspace scope should include controlled analysis modules.");
expect(!hasAccessProfile("verified", "research"), "Verified accounts should not receive controlled Research modules.");

const verifiedStates = moduleAccessStatesFor({ report: true, export: true, raw_export: true }, { role: "user", email_verified: true });
expectEqual(verifiedStates.engineering_report.label, "Report", "Engineering reports should remain open regardless of account state.");
expectEqual(verifiedStates.deepbom.label, "Module access", "DEEPBOM experimental artifact descriptors should remain authorization-controlled.");
const publicStates = moduleAccessStatesFor({}, null);
expectEqual(publicStates.engineering_report.label, "Report", "Public users should be able to reach the complete watermarked report workspace.");
expectEqual(publicStates.engineering_report.locked, false, "Public report workspace and watermarked report export must remain selectable.");

const moduleTab = (id, ariaDisabled = false) => ({
  dataset: { moduleTab: id },
  hidden: false,
  disabled: false,
  getAttribute: (name) => name === "aria-disabled" && ariaDisabled ? "true" : null,
});
expectEqual(
  selectableModuleIdFor([moduleTab("engineering_report", true), moduleTab("export_contracts")], "engineering_report"),
  "export_contracts",
  "Locked Engineering Report selection should fall back to the open contract workspace.",
);
expectEqual(
  selectableModuleIdFor([moduleTab("engineering_report"), moduleTab("export_contracts")], "regulatory_report"),
  "engineering_report",
  "A locked specialized report should fall back to an entitled Engineering Report.",
);

expect(worker.includes("const rawExportAllowed = admin || active;"), "Worker should grant raw exports at verified-account level.");
expect(worker.includes('hasAccessProfile(accessProfile, "research")'), "Worker should derive all controlled analysis modules from one Research threshold.");
expect(!worker.includes("advancedPlan") && !worker.includes("researcherPlan"), "Worker should not retain parallel legacy authorization branches.");
expect(admin.includes('["verified", "research", "medical_ai", "admin"]'), "Admin access-profile editor should expose only the canonical hierarchy.");
expect(html.includes('<option value="research">Advanced analysis module</option>') && html.includes('<option value="medical_ai">Regulatory workspace</option>'), "Account request UI should expose only the two controlled access scopes.");
expect(!html.includes('value="advanced"') && !html.includes('value="research_beta"'), "Legacy access tiers should not remain selectable.");

expect(app.includes('window.open(url, "deepbom-google-auth"'), "Google sign-in should use a popup that leaves the audit tab mounted.");
expect(!app.includes("location.href = `/api/auth/google/start"), "Google sign-in must not replace the audit page.");
expect(app.includes('type: "deepbom:auth-complete"') || readFileSync("web/auth-complete.js", "utf8").includes('type: "deepbom:auth-complete"'), "OAuth completion should use an explicit same-origin message contract.");
expect(existsSync("web/auth-complete.html") && existsSync("web/auth-complete.js"), "OAuth popup completion assets should exist.");
expect(existsSync("web/test.html") && existsSync("web/test.js"), "Private-link external test gateway assets should exist.");
expect(testPage.includes('name="robots" content="noindex, nofollow, noarchive"'), "External test gateway must not be indexed.");
expect(testClient.includes("location.replace(result.redirect_to") && testClient.includes('history.replaceState(null, "", "/test")') && testClient.includes("await activateTestAccess()"), "Test gateway must remove the bearer fragment, activate automatically, and reuse the canonical Workbench.");
expect(admin.includes('apiFetch("/api/admin/test-links"') && admin.includes("no Admin access"), "Admin console must issue explicit 24-hour non-Admin test links.");
expect(app.includes("selectableModuleIdFor(moduleTabs, resolved)"), "Programmatic module selection must use the capability-aware resolver.");
expect(app.includes('const activeState = states[activeModule] || states.static;'), "Access refresh must move focus away from a module that became locked.");

done("Access hierarchy and audit-preserving OAuth contract passed.");
