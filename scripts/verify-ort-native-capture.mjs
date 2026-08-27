import process from "node:process";
import { verifyOrtNativeCapturePackage } from "./ort-native-capture-lib.mjs";

const captureDir = process.argv[2];
if (!captureDir) throw new Error("usage: node scripts/verify-ort-native-capture.mjs <capture-directory> [model.onnx]");
const result = await verifyOrtNativeCapturePackage(captureDir, { artifactPath: process.argv[3] || null });
console.log(`Native ORT capture package verified: ${result.root}`);
console.log(`Capture ID: ${result.index.capture_id}; profiles: ${result.index.profiles.length}`);
