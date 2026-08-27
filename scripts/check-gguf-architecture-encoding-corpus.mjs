import assert from "node:assert/strict";

import { readGgufArchitectureEncodingCorpus, validateGgufArchitectureEncodingCorpus } from "./gguf-architecture-encoding-corpus-lib.mjs";

const manifest = await readGgufArchitectureEncodingCorpus();
validateGgufArchitectureEncodingCorpus(manifest, { requireBaselines: true });
const architectures = new Set(manifest.artifacts.map((row) => row.baseline.architecture));
const encodings = new Set(manifest.artifacts.flatMap((row) => row.baseline.encoding_histogram.map((entry) => entry.encoding)));
assert(architectures.size >= 4, `Expected at least four serialized architecture values, received ${[...architectures].join(", ")}`);
assert(encodings.size >= 5, `Expected at least five observed GGML encodings, received ${[...encodings].join(", ")}`);
assert(manifest.artifacts.every((row) => row.baseline.unsupported_encoding_tensor_count === 0), "Selected GGUF corpus contains an unsupported tensor encoding");
console.log(`${manifest.artifact_count} immutable GGUF artifacts cover ${architectures.size} serialized architectures and ${encodings.size} observed tensor encodings.`);
