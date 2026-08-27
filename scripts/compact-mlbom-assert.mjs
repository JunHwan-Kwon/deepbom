const PROFILE = "deepbom.compact_mlbom_compatibility.v2";
const DETAIL_POINTER = "engineering_evidence.json#/evidence/static_analysis";

export function assertCompactMlBomProjection(document, {
  expect,
  expectEqual,
  omittedProperties = [],
  label = "Compact ML-BOM",
} = {}) {
  const component = document?.metadata?.component || {};
  const componentProperties = component.properties || [];
  const documentProperties = document?.properties || [];
  const values = new Map(componentProperties.map((item) => [item.name, item.value]));

  expectEqual(values.get("deepbom:compatibility:profile"), PROFILE, `${label} should identify the compact compatibility profile.`);
  expectEqual(values.get("deepbom:compatibility:detailLocation"), DETAIL_POINTER, `${label} should bind omitted detail to the canonical evidence ledger.`);
  expect(componentProperties.length <= 120, `${label} component compatibility properties should remain bounded.`);
  expect(documentProperties.length <= 20, `${label} document compatibility properties should remain bounded.`);
  expect((component.externalReferences || []).some((item) => item.type === "evidence" && item.url === "engineering_evidence.json"), `${label} should link the structured detailed-evidence companion.`);
  expect(omittedProperties.every((name) => !values.has(name)), `${label} should not duplicate detailed evidence properties removed by the compact profile.`);
}
