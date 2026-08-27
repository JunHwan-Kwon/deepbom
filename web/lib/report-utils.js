export function markdownTable(headers, rows) {
  const safeHeaders = headers.map(markdownCell);
  const lines = [
    `| ${safeHeaders.join(" | ")} |`,
    `| ${headers.map(() => "---").join(" | ")} |`,
  ];
  for (const row of rows || []) {
    lines.push(`| ${row.map(markdownCell).join(" | ")} |`);
  }
  return lines.join("\n");
}

function markdownCell(value) {
  return String(value ?? "-")
    .replace(/\r?\n/g, "<br>")
    .replace(/\|/g, "\\|");
}

export function bulletList(items) {
  return (items || []).map((item) => `- ${item}`).join("\n");
}

export function code(value) {
  return `\`${String(value ?? "").replace(/`/g, "'")}\``;
}

export function csvCell(value) {
  const source = String(value ?? "");
  const text = typeof value === "string" && /^[\t\r\n ]*[=+\-@]/.test(source) ? `'${source}` : source;
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function zipTextFile(name, text) {
  return { name, data: String(text ?? "") };
}

export function zipBinaryFile(name, bytes) {
  return { name, data: bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes || []) };
}

export function jsonForDownload(value) {
  return JSON.stringify(value, (_key, item) => {
    if (typeof item === "number" && !Number.isFinite(item)) return null;
    return item;
  }, 2);
}

export function normalizeJsonContractValue(value, path = "$") {
  if (value === undefined) return null;
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new TypeError(`JSON contract rejects non-finite number at ${path}`);
    return Object.is(value, -0) ? 0 : value;
  }
  if (Array.isArray(value)) {
    return value.map((item, index) => normalizeJsonContractValue(item, `${path}/${index}`));
  }
  if (typeof value !== "object") {
    throw new TypeError(`JSON contract rejects ${typeof value} at ${path}`);
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    throw new TypeError(`JSON contract requires a plain object at ${path}`);
  }
  const normalized = {};
  for (const [key, item] of Object.entries(value)) {
    normalized[key] = normalizeJsonContractValue(item, `${path}/${escapeJsonPointer(key)}`);
  }
  return normalized;
}

export function canonicalJson(value) {
  const ancestors = new Set();
  const serialize = (item, path) => {
    if (item === null || typeof item === "boolean" || typeof item === "string") {
      assertIJsonString(item, path);
      return JSON.stringify(item);
    }
    if (typeof item === "number") {
      if (!Number.isFinite(item)) throw new TypeError(`RFC 8785 JCS rejects non-finite number at ${path}`);
      return JSON.stringify(item);
    }
    if (typeof item !== "object") throw new TypeError(`RFC 8785 JCS rejects ${typeof item} at ${path}`);
    if (ancestors.has(item)) throw new TypeError(`RFC 8785 JCS rejects a cyclic value at ${path}`);
    const prototype = Object.getPrototypeOf(item);
    if (prototype !== Object.prototype && prototype !== null && !Array.isArray(item)) {
      throw new TypeError(`RFC 8785 JCS requires plain JSON objects at ${path}`);
    }
    ancestors.add(item);
    let result;
    if (Array.isArray(item)) {
      result = `[${item.map((entry, index) => serialize(entry, `${path}/${index}`)).join(",")}]`;
    } else {
      const keys = Object.keys(item).sort();
      for (const key of keys) assertIJsonString(key, `${path}/<key>`);
      result = `{${keys.map((key) => `${JSON.stringify(key)}:${serialize(item[key], `${path}/${escapeJsonPointer(key)}`)}`).join(",")}}`;
    }
    ancestors.delete(item);
    return result;
  };
  return serialize(value, "$");
}

function assertIJsonString(value, path) {
  if (typeof value !== "string") return;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1);
      if (!(next >= 0xdc00 && next <= 0xdfff)) throw new TypeError(`RFC 8785 JCS rejects an unpaired surrogate at ${path}`);
      index += 1;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      throw new TypeError(`RFC 8785 JCS rejects an unpaired surrogate at ${path}`);
    }
  }
}

function escapeJsonPointer(value) {
  return String(value).replaceAll("~", "~0").replaceAll("/", "~1");
}

export async function buildCanonicalPackageDigest(files = []) {
  const encoder = new TextEncoder();
  const seen = new Set();
  const seenCaseFolded = new Set();
  const members = [];
  for (const file of files) {
    const name = validatePackageMemberName(file?.name, seen, seenCaseFolded);
    seen.add(name);
    seenCaseFolded.add(name.toLocaleLowerCase("en-US"));
    const bytes = file?.data instanceof Uint8Array ? file.data : encoder.encode(String(file?.data ?? ""));
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    const sha256 = [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
    members.push({ name, size: bytes.byteLength, sha256 });
  }
  members.sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
  const digestSet = {
    schema: "deepbom.package_member_digest_set.v1",
    files: members,
  };
  const canonicalBytes = encoder.encode(canonicalJson(digestSet));
  const packageDigest = await crypto.subtle.digest("SHA-256", canonicalBytes);
  return {
    package_hash_method: "SHA-256 over RFC8785-JCS canonical deepbom.package_member_digest_set.v1",
    canonicalization: "RFC8785-JCS",
    package_hash_sha256: [...new Uint8Array(packageDigest)].map((byte) => byte.toString(16).padStart(2, "0")).join(""),
    files: members,
  };
}

export function validatePackageMemberName(rawName, seen = new Set(), seenCaseFolded = new Set()) {
  const original = String(rawName || "");
  const name = original.normalize("NFC");
  const segments = name.split("/");
  if (!name || name !== original || name.includes("\\") || name.startsWith("/") || /^[A-Za-z]:/.test(name)
    || segments.some((segment) => !segment || segment === "." || segment === "..")
    || /[\u0000-\u001f\u007f]/.test(name)) {
    throw new Error(`Unsafe package member path: ${original || "<empty>"}`);
  }
  const caseFolded = name.toLocaleLowerCase("en-US");
  if (seen.has(name) || seenCaseFolded.has(caseFolded)) throw new Error(`Package member name collides after NFC/case folding: ${name}`);
  return name;
}

export function validatePackageAttestation(attestation, expectedDigest) {
  const fail = (message) => {
    throw new Error(`Invalid package attestation response: ${message}`);
  };
  if (attestation?.schema !== "deepbom.attestation.v2.2") fail("unexpected schema");
  const payload = attestation?.signed_payload;
  if (payload?.schema !== "deepbom.attestation_payload.v2.2") fail("unexpected signed payload schema");
  if (payload?.canonicalization !== "RFC8785-JCS") fail("unexpected canonicalization");
  if (payload?.package?.package_hash_sha256 !== expectedDigest?.package_hash_sha256) fail("package hash mismatch");
  if (canonicalJson(payload?.package?.unsigned_package_members || []) !== canonicalJson(expectedDigest?.files || [])) fail("member digest set mismatch");
  if (payload?.package?.attestation_member_excluded_from_package_hash !== true) fail("attestation member exclusion is not declared");
  if (payload?.package?.path_normalization !== "relative_posix_nfc" || payload?.package?.duplicate_member_policy !== "reject_after_nfc_and_case_fold" || payload?.package?.undeclared_member_policy !== "reject") fail("package member safety policy mismatch");
  const kid = String(attestation?.signature?.kid || "");
  if (!kid || /(^|[-_.])(current|latest|active)([-_.]|$)/i.test(kid)) fail("signing key id is missing or mutable");
  if (payload?.signing_key?.kid !== kid || payload?.signing_key?.alg !== "HS256") fail("signed key metadata mismatch");
  if (!attestation?.signature?.value || attestation?.signature?.alg !== "HS256") fail("signature is missing");
  if (attestation?.verification_status?.package_hash_sha256 !== expectedDigest?.package_hash_sha256) fail("verification receipt hash mismatch");
  return attestation;
}

export function markdownWithModelSha256(markdown, hash) {
  if (typeof markdown !== "string") return markdown;
  const line = `Model SHA-256: \`${hash}\``;
  if (/Model SHA-256: `[^`]*`/.test(markdown)) {
    return markdown.replace(/Model SHA-256: `[^`]*`/, line);
  }
  if (markdown.includes("## Summary")) {
    return markdown.replace("## Summary", `## Summary\n- ${line}`);
  }
  return `- ${line}\n\n${markdown}`;
}
