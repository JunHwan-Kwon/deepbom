import { webcrypto } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import { signPackageDigest, verifyPackageSignatureWithTrust } from "../web/lib/public-key-signature.js";
import { OFFICIAL_SIGNING_KEY_REGISTRY_URL, SIGNER_CLASSES } from "../web/lib/signing-key-trust.js";

const options = Object.fromEntries(process.argv.slice(2).map((argument) => {
  const [key, ...rest] = argument.replace(/^--/, "").split("=");
  return [key, rest.join("=")];
}));
const digestPath = path.resolve(options.digest || "");
const privateKeyPath = path.resolve(options["private-key"] || ".local-validation/signing/deepbom-official-release-es256-v1.private.jwk.json");
const registryPath = path.resolve(options.registry || "web/.well-known/deepbom-signing-keys.json");
const outputPath = path.resolve(options.out || "deepbom_official_package_signature.json");
const keyId = options["key-id"] || "deepbom-official-release-es256-v1";
const scope = options.scope || "official_release_evidence";
if (!options.digest) throw new Error("Usage: node scripts/sign-official-package-digest.mjs --digest=<package-digest.json> [--out=<signature.json>]");

const [digest, privateJwk, registry] = await Promise.all([
  readJson(digestPath),
  readJson(privateKeyPath),
  readJson(registryPath),
]);
const publicJwk = { kty: privateJwk.kty, crv: privateJwk.crv, x: privateJwk.x, y: privateJwk.y, alg: "ES256", use: "sig" };
const [privateKey, publicKey] = await Promise.all([
  webcrypto.subtle.importKey("jwk", privateJwk, { name: "ECDSA", namedCurve: "P-256" }, false, ["sign"]),
  webcrypto.subtle.importKey("jwk", publicJwk, { name: "ECDSA", namedCurve: "P-256" }, false, ["verify"]),
]);
const signature = await signPackageDigest(digest, {
  scope,
  keyPair: { privateKey, publicKey, publicJwk },
  signerClass: SIGNER_CLASSES.official,
  keyId,
  authorityRegistryUrl: OFFICIAL_SIGNING_KEY_REGISTRY_URL,
  cryptoProvider: webcrypto,
});
const verification = await verifyPackageSignatureWithTrust(signature, digest, registry, { cryptoProvider: webcrypto });
if (!verification.cryptographically_valid || !verification.trusted) throw new Error(`Official signature trust verification failed: ${verification.status}`);
await writeFile(outputPath, `${JSON.stringify(signature, null, 2)}\n`);
console.log(`Official detached signature written: ${outputPath} (${keyId}; ${scope})`);

async function readJson(file) {
  return JSON.parse(await readFile(file, "utf8"));
}
