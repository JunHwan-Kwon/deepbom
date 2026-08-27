import { readFileSync } from "node:fs";
import { createCheck } from "./check-assert.mjs";
import {
  createExternalTestGrant,
  createExternalTestLink,
  EXTERNAL_TEST_GRANT_TTL_SECONDS,
  EXTERNAL_TEST_LINK_TTL_SECONDS,
  verifyExternalTestGrant,
  verifyExternalTestLink,
} from "../worker/test-access.js";
import worker from "../worker/index.js";

const { done, expect, expectEqual } = createCheck("External test access contract check");
const secret = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const nowMs = Date.UTC(2026, 7, 8, 2, 0, 0);

const accessLink = await createExternalTestLink({
  secret,
  nowMs,
  nonce: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
});
const linkPayload = await verifyExternalTestLink(accessLink.token, { secret, nowMs });
expectEqual(linkPayload.scope, "external_test_link", "Access-link scope must be domain-specific.");
expectEqual(
  linkPayload.expires_at - linkPayload.issued_at,
  EXTERNAL_TEST_LINK_TTL_SECONDS,
  "Every external test link must have an exact 24-hour lifetime.",
);
expectEqual(accessLink.expiresAt, "2026-08-09T02:00:00.000Z", "Link expiry must be deterministic from issue time.");

await expectReject(
  () => verifyExternalTestLink(tamper(accessLink.token), { secret, nowMs }),
  "invalid_test_token_signature",
  "Access-link signature tampering must fail closed.",
);
await expectReject(
  () => verifyExternalTestLink(accessLink.token, { secret, nowMs: nowMs + EXTERNAL_TEST_LINK_TTL_SECONDS * 1000 }),
  "test_token_expired",
  "The exact expiration instant must be rejected.",
);

const futureLink = await createExternalTestLink({
  secret,
  nowMs: nowMs + 10 * 60 * 1000,
  nonce: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
});
await expectReject(
  () => verifyExternalTestLink(futureLink.token, { secret, nowMs }),
  "invalid_test_token_time",
  "Links issued too far in the future must be rejected.",
);

const grant = await createExternalTestGrant({
  accessLink: linkPayload,
  secret,
  nowMs,
  nonce: "cccccccccccccccccccccccccccccccc",
});
const grantPayload = await verifyExternalTestGrant(grant.token, { secret, nowMs });
expectEqual(grantPayload.scope, "external_test_grant", "Browser-grant scope must differ from access-link scope.");
expectEqual(grantPayload.access_id, linkPayload.nonce, "Browser grant must retain the issuing link identity.");
expectEqual(grant.maxAgeSeconds, EXTERNAL_TEST_GRANT_TTL_SECONDS, "Immediate activation must receive the full 24-hour cookie lifetime.");

const lateGrant = await createExternalTestGrant({
  accessLink: linkPayload,
  secret,
  nowMs: nowMs + 6 * 60 * 60 * 1000,
  nonce: "dddddddddddddddddddddddddddddddd",
});
expectEqual(lateGrant.maxAgeSeconds, 18 * 60 * 60, "Late activation must receive only the link's remaining lifetime.");
expectEqual(lateGrant.payload.expires_at, linkPayload.expires_at, "A browser grant must never outlive its access link.");

await expectReject(
  () => verifyExternalTestGrant(accessLink.token, { secret, nowMs }),
  "invalid_test_token_signature",
  "An access-link token must not be accepted directly as a browser grant.",
);
await expectReject(
  () => verifyExternalTestGrant(grant.token, { secret, nowMs: nowMs + EXTERNAL_TEST_GRANT_TTL_SECONDS * 1000 }),
  "test_token_expired",
  "Expired browser grants must be rejected.",
);

const workerSource = readFileSync("worker/index.js", "utf8");
const activationSource = workerSource.slice(
  workerSource.indexOf("async function activateExternalTest"),
  workerSource.indexOf("async function requireDeepBomAccess"),
);
expect(workerSource.includes('path === "/api/admin/test-links"'), "Worker must expose the Admin-only link issuance endpoint.");
expect(activationSource.includes("verifyExternalTestLink") && !activationSource.includes("requireVerifiedSession"), "Link activation must not depend on an account session.");
expect(workerSource.includes("account_bound: accountBound") && workerSource.includes('role: "user"'), "Guest identity must declare its non-account and non-Admin boundaries.");
expect(workerSource.includes("clearCookie(EXTERNAL_TEST_COOKIE)"), "Logout and explicit deactivation must clear external test access.");
expect(workerSource.includes("cookie(EXTERNAL_TEST_COOKIE, grant.token, grant.maxAgeSeconds, true)"), "The browser grant must be stored in an HttpOnly cookie.");

const adminEnv = fixtureEnv(fixtureUser({ id: "admin-user", email: "admin@example.org", role: "admin", accessProfile: "admin" }));
const issueResponse = await worker.fetch(new Request("https://deepbom.test/api/admin/test-links", {
  method: "POST",
  headers: { cookie: "audit_session=admin-session", "content-type": "application/json" },
  body: "{}",
}), adminEnv);
const issueBody = await issueResponse.json();
expectEqual(issueResponse.status, 201, "Admin should be able to issue a 24-hour external test link.");
expectEqual(typeof issueBody.access_url, "string", "Issuance receipt must include the access URL.");
expect(issueBody.access_url?.startsWith("https://deepbom.test/test#access=") === true, "Access token must remain in the /test URL fragment.");
expectEqual(issueBody.valid_for_seconds, 24 * 60 * 60, "Issuance receipt must state the exact validity period.");
expectEqual(issueBody.admin_access, false, "Issuance receipt must state that Admin access is excluded.");
const issuedToken = typeof issueBody.access_url === "string"
  ? new URLSearchParams(new URL(issueBody.access_url).hash.slice(1)).get("access")
  : "";

const guestEnv = fixtureEnv(null);
const activationResponse = await worker.fetch(new Request("https://deepbom.test/api/test/activate", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ access: issuedToken }),
}), guestEnv);
const activationBody = await activationResponse.json();
expectEqual(activationResponse.status, 200, "A valid link must activate without an account cookie.");
expectEqual(activationBody.access_level, "medical_ai", "Activated access must use the Medical AI evaluation tier.");
expectEqual(activationBody.user.test_access.account_bound, false, "Anonymous activation must remain explicitly non-account-bound.");
expectEqual(activationBody.admin_access, false, "Activation must not synthesize Admin access.");
const grantCookie = firstCookie(activationResponse.headers.get("set-cookie"), "audit_external_test");
expect(grantCookie.includes("HttpOnly") && grantCookie.includes("SameSite=Lax") && grantCookie.includes("Secure"), "Grant cookie must be HttpOnly, same-site, and secure.");
const cookieHeader = grantCookie.split(";", 1)[0];

const sessionResponse = await worker.fetch(new Request("https://deepbom.test/api/auth/me", {
  headers: { cookie: cookieHeader },
}), guestEnv);
const session = await sessionResponse.json();
expectEqual(sessionResponse.status, 200, "The browser grant alone must establish a current session.");
expectEqual(session.user.provider, "access_link", "Guest session identity must expose its bearer-link provenance.");
expectEqual(session.user.role, "user", "Guest session must never receive the Admin role.");

const accessResponse = await worker.fetch(new Request("https://deepbom.test/api/access/status", {
  headers: { cookie: cookieHeader },
}), guestEnv);
const access = await accessResponse.json();
expectEqual(access.access_profile, "medical_ai", "Access status must expose the temporary capability profile.");
expect(access.allowed.report && access.allowed.raw_export && access.allowed.regulatory_report
  && access.allowed.deepbom && access.allowed.runtime_basin && access.allowed.deployment_sensitivity,
"Temporary link access must enable every non-Admin analysis capability.");

const manifestResponse = await worker.fetch(new Request("https://deepbom.test/api/analysis-module/deepbom/manifest", {
  headers: { cookie: cookieHeader },
}), guestEnv);
expectEqual(manifestResponse.status, 200, "Temporary link access must authorize the protected DEEPBOM manifest.");
const formatterResponse = await worker.fetch(new Request("https://deepbom.test/web/lib/report-regulatory-entry.js", {
  headers: { cookie: cookieHeader },
}), guestEnv);
expectEqual(formatterResponse.status, 200, "Temporary link access must authorize the Medical AI report formatter.");

const adminResponse = await worker.fetch(new Request("https://deepbom.test/api/admin/users", {
  headers: { cookie: cookieHeader },
}), guestEnv);
const adminDenied = await adminResponse.json();
expectEqual(adminResponse.status, 403, "Temporary link access must remain blocked from Admin APIs.");
expectEqual(adminDenied.error, "admin_required", "Admin denial must retain the canonical reason code.");

const accountWriteResponse = await worker.fetch(new Request("https://deepbom.test/api/account/requests", {
  method: "POST",
  headers: { cookie: cookieHeader, "content-type": "application/json" },
  body: JSON.stringify({ type: "feedback", message: "external test feedback" }),
}), guestEnv);
const accountWriteDenied = await accountWriteResponse.json();
expectEqual(accountWriteResponse.status, 403, "A bearer session must not write account-bound request data.");
expectEqual(accountWriteDenied.error, "account_required", "Account-bound write denial must be explicit.");

const directTokenResponse = await worker.fetch(new Request("https://deepbom.test/api/access/status", {
  headers: { cookie: `audit_external_test=${issuedToken}` },
}), guestEnv);
const directTokenLicense = await directTokenResponse.json();
expectEqual(directTokenLicense.authenticated, false, "A URL token copied directly into the cookie slot must not establish access.");

const logoutResponse = await worker.fetch(new Request("https://deepbom.test/api/auth/logout", {
  method: "POST",
  headers: { cookie: cookieHeader },
}), guestEnv);
const logoutCookies = logoutResponse.headers.get("set-cookie") || "";
expect(logoutCookies.includes("audit_session=") && logoutCookies.includes("audit_external_test=")
  && (logoutCookies.match(/Max-Age=0/g) || []).length === 2,
"Logout must clear both possible account and temporary test cookies.");

done("External test access passed (24-hour bearer issuance, account-free activation, token separation, capability authorization, protected assets, Admin/account-write denial, expiry, and cookie revocation).");

async function expectReject(run, code, label) {
  try {
    await run();
    expect(false, `${label} Expected ${code}.`);
  } catch (error) {
    expectEqual(error?.code, code, label);
  }
}

function tamper(token) {
  const last = token.at(-1);
  return `${token.slice(0, -1)}${last === "A" ? "B" : "A"}`;
}

function fixtureUser({ id, email, role = "user", accessProfile = "verified" }) {
  return {
    id,
    email,
    name: role === "admin" ? "Admin" : "User",
    avatar_url: "",
    provider: "google",
    role,
    access_profile: accessProfile,
    access_status: "active",
    access_expires_at: "",
    email_verified_at: "2026-08-01T00:00:00.000Z",
    created_at: "2026-08-01T00:00:00.000Z",
    expires_at: new Date(Date.now() + 60 * 60 * 1000).toISOString(),
  };
}

function fixtureEnv(user) {
  return {
    SESSION_SECRET: secret,
    DB: {
      prepare() {
        const statement = {
          bind() { return statement; },
          async first() { return user; },
          async all() { return { results: [] }; },
          async run() { return { meta: { changes: 1 } }; },
        };
        return statement;
      },
    },
    ASSETS: {
      async fetch() {
        return new Response("export const fixture = true;", {
          headers: { "content-type": "text/javascript; charset=utf-8" },
        });
      },
    },
  };
}

function firstCookie(header, name) {
  const match = new RegExp(`(?:^|,\\s*)(${name}=[^,]+)`).exec(header || "");
  return match?.[1] || "";
}
