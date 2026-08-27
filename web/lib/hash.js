import { Sha256Accumulator } from "./sha256-sync.js";

export async function sha256Hex(bytes) {
  return digestHex("SHA-256", bytes);
}

export async function sha256FileHex(file, chunkBytes = 4 * 1024 * 1024) {
  if (!file || typeof file.slice !== "function" || !Number.isSafeInteger(file.size)) throw new Error("A browser File is required for streaming SHA-256.");
  const accumulator = new Sha256Accumulator();
  for (let offset = 0; offset < file.size; offset += chunkBytes) {
    accumulator.update(new Uint8Array(await file.slice(offset, Math.min(file.size, offset + chunkBytes)).arrayBuffer()));
  }
  return accumulator.digestHex();
}

export async function sha1Hex(bytes) {
  return digestHex("SHA-1", bytes);
}

async function digestHex(algorithm, bytes) {
  const digest = await crypto.subtle.digest(algorithm, bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function sha256TypedArrayListHex(arrays) {
  const encoder = new TextEncoder();
  const parts = [];
  let total = 0;
  for (const array of arrays || []) {
    if (!ArrayBuffer.isView(array)) continue;
    const header = encoder.encode(`${array.constructor?.name || "TypedArray"}:${array.length}:${array.byteLength};`);
    const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
    parts.push(header, bytes);
    total += header.byteLength + bytes.byteLength;
  }
  const packed = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    packed.set(part, offset);
    offset += part.byteLength;
  }
  return sha256Hex(packed);
}
