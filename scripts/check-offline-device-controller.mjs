import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import {
  formatOfflineMacs,
  relativeDeviceTime,
} from "../web/lib/offline-device-controller.js";

assert.equal(formatOfflineMacs(300_775_552), "300.8M");
assert.equal(formatOfflineMacs(1_250_000_000), "1.25G");
assert.equal(formatOfflineMacs(999), "1K");
assert.equal(formatOfflineMacs(-1), "?");
assert.equal(formatOfflineMacs(Number.NaN), "?");
assert.equal(relativeDeviceTime(59.9), "59s ago");
assert.equal(relativeDeviceTime(60), "1m ago");
assert.equal(relativeDeviceTime(7_200), "2h ago");
assert.equal(relativeDeviceTime(-1), "unknown");

const source = await readFile("web/lib/offline-device-controller.js", "utf8");
assert(!source.includes("innerHTML"), "Offline device UI must not interpolate server fields through innerHTML.");
assert(source.includes("if (!response.ok)"), "Device registry requests must reject non-success HTTP responses.");

console.log("Offline device controller check passed (format boundaries, fail-closed HTTP, DOM-only rendering). ");
