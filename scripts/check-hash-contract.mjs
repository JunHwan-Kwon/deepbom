import { createCheck } from "./check-assert.mjs";
import { sha256Hex, sha256TypedArrayListHex } from "../web/lib/hash.js";
import { canonicalJson } from "../web/lib/report-utils.js";

const { done, expectEqual, expectThrows } = createCheck("Hash contract check");
const encoded = new TextEncoder().encode("abc");

expectEqual(
  await sha256Hex(encoded),
  "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  "sha256Hex should match the standard SHA-256 digest for abc.",
);
expectEqual(
  await sha256TypedArrayListHex([new Int8Array([1, -1, 2]), new Float32Array([1.5])]),
  "fcfeba0fea8e4a108350f1bd652ba4fd1564452a67156da40dd149b8193e33d2",
  "sha256TypedArrayListHex should include dtype and length framing.",
);
expectEqual(
  await sha256TypedArrayListHex([new Uint8Array([1, 255, 2]), new Float32Array([1.5])]),
  "1257a43c0ce819c7a01085da450a2d0ef1851205b804bf2a6a8d40b1ac5ddc22",
  "sha256TypedArrayListHex should distinguish signed and unsigned output views.",
);

expectEqual(canonicalJson({ n: -0, b: 1, a: [true, null, "x"] }), '{"a":[true,null,"x"],"b":1,"n":0}', "JCS property order and negative-zero serialization");
expectEqual(canonicalJson({ numbers: [333333333.33333329, 1e30, 4.50, 2e-3, 1e-27] }), '{"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27]}', "JCS ECMAScript number serialization");
expectThrows(() => canonicalJson({ value: Number.NaN }), "non-finite", "JCS rejects NaN");
expectThrows(() => canonicalJson({ value: Number.POSITIVE_INFINITY }), "non-finite", "JCS rejects Infinity");
expectThrows(() => canonicalJson({ value: undefined }), "undefined", "JCS rejects undefined object values");
expectThrows(() => canonicalJson([undefined]), "undefined", "JCS rejects undefined array values");
expectThrows(() => canonicalJson({ value: 1n }), "bigint", "JCS rejects BigInt");
expectThrows(() => canonicalJson("\ud800"), "unpaired surrogate", "JCS rejects lone UTF-16 surrogates");
const cyclic = {};
cyclic.self = cyclic;
expectThrows(() => canonicalJson(cyclic), "cyclic", "JCS rejects cycles");

done("Hash contract passed (browser SHA-256 helpers and strict RFC 8785 JCS are stable).");
