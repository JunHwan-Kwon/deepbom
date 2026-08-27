const PNG_SIGNATURE = Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10);

export function encodeRgbPng(rgbBytes, width, height) {
  const pixels = byteArray(rgbBytes, "RGB fixture");
  const w = positiveInteger(width, "PNG width");
  const h = positiveInteger(height, "PNG height");
  assert(pixels.length === w * h * 3, "RGB fixture size does not match the PNG dimensions.");
  const scanlines = new Uint8Array(h * (1 + w * 3));
  for (let y = 0; y < h; y += 1) {
    const row = y * (1 + w * 3);
    scanlines[row] = 0;
    scanlines.set(pixels.subarray(y * w * 3, (y + 1) * w * 3), row + 1);
  }
  const ihdr = new Uint8Array(13);
  writeU32(ihdr, 0, w);
  writeU32(ihdr, 4, h);
  ihdr.set([8, 2, 0, 0, 0], 8);
  return concatenate([
    PNG_SIGNATURE,
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", storedZlib(scanlines)),
    pngChunk("IEND", new Uint8Array()),
  ]);
}

export function decodeStoredRgbPng(pngBytes) {
  const bytes = byteArray(pngBytes, "PNG fixture");
  assert(bytes.length >= PNG_SIGNATURE.length && equalBytes(bytes.subarray(0, 8), PNG_SIGNATURE), "PNG signature mismatch.");
  let offset = 8;
  let width = 0;
  let height = 0;
  const idat = [];
  let sawEnd = false;
  while (offset < bytes.length) {
    assert(offset + 12 <= bytes.length, "PNG chunk header is truncated.");
    const length = readU32(bytes, offset);
    const typeBytes = bytes.subarray(offset + 4, offset + 8);
    const type = String.fromCharCode(...typeBytes);
    const dataStart = offset + 8;
    const dataEnd = dataStart + length;
    assert(dataEnd + 4 <= bytes.length, `PNG ${type} chunk is truncated.`);
    const data = bytes.subarray(dataStart, dataEnd);
    const declaredCrc = readU32(bytes, dataEnd);
    assert(crc32(concatenate([typeBytes, data])) === declaredCrc, `PNG ${type} CRC mismatch.`);
    if (type === "IHDR") {
      assert(length === 13 && data[8] === 8 && data[9] === 2 && data[10] === 0 && data[11] === 0 && data[12] === 0, "PNG IHDR is not non-interlaced RGB8.");
      width = readU32(data, 0);
      height = readU32(data, 4);
    } else if (type === "IDAT") {
      idat.push(data);
    } else if (type === "IEND") {
      assert(length === 0, "PNG IEND must be empty.");
      sawEnd = true;
      offset = dataEnd + 4;
      break;
    }
    offset = dataEnd + 4;
  }
  assert(sawEnd && offset === bytes.length && width > 0 && height > 0 && idat.length > 0, "PNG structure is incomplete.");
  const scanlines = inflateStoredZlib(concatenate(idat));
  const rowBytes = width * 3;
  assert(scanlines.length === height * (rowBytes + 1), "PNG scanline size mismatch.");
  const rgb = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    const row = y * (rowBytes + 1);
    assert(scanlines[row] === 0, "PNG fixture uses an unsupported row filter.");
    rgb.set(scanlines.subarray(row + 1, row + 1 + rowBytes), y * rowBytes);
  }
  return { width, height, rgb };
}

function storedZlib(data) {
  const blocks = [];
  for (let offset = 0; offset < data.length; offset += 65535) {
    const length = Math.min(65535, data.length - offset);
    const header = new Uint8Array(5);
    header[0] = offset + length === data.length ? 1 : 0;
    header[1] = length & 0xff;
    header[2] = (length >>> 8) & 0xff;
    const inverse = (~length) & 0xffff;
    header[3] = inverse & 0xff;
    header[4] = (inverse >>> 8) & 0xff;
    blocks.push(header, data.subarray(offset, offset + length));
  }
  const checksum = new Uint8Array(4);
  writeU32(checksum, 0, adler32(data));
  return concatenate([Uint8Array.of(0x78, 0x01), ...blocks, checksum]);
}

function inflateStoredZlib(bytes) {
  assert(bytes.length >= 6 && ((bytes[0] << 8) | bytes[1]) % 31 === 0 && (bytes[1] & 0x20) === 0, "PNG zlib header is invalid.");
  let offset = 2;
  const blocks = [];
  let final = false;
  while (!final) {
    assert(offset + 5 <= bytes.length - 4, "PNG zlib block header is truncated.");
    const header = bytes[offset];
    final = (header & 1) === 1;
    assert((header & 0x06) === 0, "PNG zlib stream is not stored-block encoded.");
    const length = bytes[offset + 1] | (bytes[offset + 2] << 8);
    const inverse = bytes[offset + 3] | (bytes[offset + 4] << 8);
    assert(((length ^ inverse) & 0xffff) === 0xffff, "PNG zlib stored-block length complement mismatch.");
    offset += 5;
    assert(offset + length <= bytes.length - 4, "PNG zlib stored block is truncated.");
    blocks.push(bytes.subarray(offset, offset + length));
    offset += length;
  }
  assert(offset === bytes.length - 4, "PNG zlib stream has trailing data.");
  const output = concatenate(blocks);
  assert(adler32(output) === readU32(bytes, offset), "PNG zlib Adler-32 mismatch.");
  return output;
}

function pngChunk(type, data) {
  const typeBytes = new TextEncoder().encode(type);
  assert(typeBytes.length === 4, "PNG chunk type must be four bytes.");
  const chunk = new Uint8Array(12 + data.length);
  writeU32(chunk, 0, data.length);
  chunk.set(typeBytes, 4);
  chunk.set(data, 8);
  writeU32(chunk, 8 + data.length, crc32(concatenate([typeBytes, data])));
  return chunk;
}

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function adler32(bytes) {
  let a = 1;
  let b = 0;
  for (const byte of bytes) {
    a = (a + byte) % 65521;
    b = (b + a) % 65521;
  }
  return ((b << 16) | a) >>> 0;
}

function concatenate(parts) {
  const length = parts.reduce((sum, part) => sum + part.length, 0);
  const output = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.length;
  }
  return output;
}

function writeU32(bytes, offset, value) {
  bytes[offset] = (value >>> 24) & 0xff;
  bytes[offset + 1] = (value >>> 16) & 0xff;
  bytes[offset + 2] = (value >>> 8) & 0xff;
  bytes[offset + 3] = value & 0xff;
}

function readU32(bytes, offset) {
  return ((bytes[offset] * 0x1000000) + (bytes[offset + 1] << 16) + (bytes[offset + 2] << 8) + bytes[offset + 3]) >>> 0;
}

function byteArray(value, label) {
  assert(value instanceof Uint8Array, `${label} must be Uint8Array.`);
  return value;
}

function positiveInteger(value, label) {
  assert(Number.isInteger(value) && value > 0 && value <= 0xffffffff, `${label} is invalid.`);
  return value;
}

function equalBytes(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
