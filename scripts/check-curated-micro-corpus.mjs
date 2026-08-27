import {
  curatedMicroArtifactPath,
  readCuratedMicroCorpus,
} from "./curated-micro-corpus-lib.mjs";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual } = createCheck("Curated micro corpus");
const manifest = await readCuratedMicroCorpus();
const mcunet = manifest.sources.find((source) => source.id === "mit-han-lab/mcunet");

expectEqual(manifest.artifact_count, 11, "MCUNet should preserve its complete eleven-model TFLite index.");
expectEqual(manifest.artifacts.reduce((total, artifact) => total + artifact.size_bytes, 0), 10_374_208, "Pinned MCUNet bytes should conserve.");
expectEqual(new Set(manifest.artifacts.map((artifact) => artifact.task)).size, 3, "The corpus should cover ImageNet, VWW, and person detection.");
expectEqual(manifest.artifacts.filter((artifact) => artifact.published_precision === "int8_ptq").length, 11, "All supplied MCUNet TFLite artifacts should retain their upstream INT8 PTQ classification.");
expectEqual(mcunet?.revision, "9c164f8483003a5c6445871d94d30720aecc5918", "MCUNet source evidence should remain commit-pinned.");
expectEqual(manifest.discovery_source.revision, "16e565e55e5385000bc259cf621a169bf13a1ff6", "awesome-tinyml discovery evidence should remain commit-pinned.");
expect(manifest.discovery_candidates.some((candidate) => candidate.repository === "aztc/EtinyNet"
  && candidate.ingestion_status === "blocked"), "EtinyNet should remain visible without being misrepresented as an analyzable artifact.");
for (const artifact of manifest.artifacts) {
  expect(artifact.url.startsWith(mcunet.release_base_url), `${artifact.id}: artifact URL should come from the pinned MCUNet model index base.`);
  expect(curatedMicroArtifactPath("cache", manifest, artifact).endsWith(artifact.filename), `${artifact.id}: local cache path should preserve the filename.`);
  const macs = Number(artifact.upstream_metrics?.macs);
  if (Number.isFinite(macs)) expect(macs > 0, `${artifact.id}: published MAC evidence should be positive.`);
}

done("11 hash-pinned MCUNet INT8 TFLite artifacts; EtinyNet retained as a blocked discovery candidate");
