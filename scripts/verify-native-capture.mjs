import process from "node:process";
import { verifyNativeCapturePackage } from "./native-capture-lib.mjs";

const captureDir = process.argv[2];
if (!captureDir) throw new Error("usage: node scripts/verify-native-capture.mjs <capture-directory>");
const result = await verifyNativeCapturePackage(captureDir);
console.log(`Native capture package verified: ${result.root}`);
console.log(`Mode: ${result.index.capture_mode}; assignments: ${result.assignment.assignments.length}`);
