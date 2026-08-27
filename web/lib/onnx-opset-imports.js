export function assessOnnxOpsetImports(opsets) {
  const rows = (opsets || []).map((entry, index) => {
    const version = Number(entry?.version);
    const reasonCodes = [];
    if (!Number.isSafeInteger(version) || version <= 0) reasonCodes.push("opset_version_not_positive_safe_integer");
    return {
      index,
      domain: normalizeOnnxDomain(entry?.domain),
      version,
      status: reasonCodes.length ? "fail" : "pass",
      reason_codes: reasonCodes,
      diagnostic_codes: [],
      selected_effective_import: false,
    };
  });
  const byDomain = new Map();
  for (const row of rows) {
    const matches = byDomain.get(row.domain) || [];
    matches.push(row);
    byDomain.set(row.domain, matches);
  }

  const effectiveImports = [];
  let duplicateIdenticalDomainCount = 0;
  let duplicateVersionVariantDomainCount = 0;
  for (const [domain, matches] of byDomain) {
    const validRows = matches.filter((row) => row.status === "pass");
    const versions = [...new Set(validRows.map((row) => row.version))].sort((a, b) => a - b);
    if (matches.length > 1) {
      const diagnostic = versions.length > 1
        ? "repeated_opset_import_domain_multiple_versions"
        : "repeated_opset_import_domain_identical_version";
      matches.forEach((row) => row.diagnostic_codes.push(diagnostic));
      if (versions.length > 1) duplicateVersionVariantDomainCount += 1;
      else if (versions.length === 1 && validRows.length === matches.length) duplicateIdenticalDomainCount += 1;
    }
    if (!validRows.length) continue;
    const version = Math.max(...validRows.map((row) => row.version));
    validRows.filter((row) => row.version === version).forEach((row) => { row.selected_effective_import = true; });
    effectiveImports.push({
      domain,
      version,
      source_indices: validRows.map((row) => row.index),
      distinct_source_versions: versions,
      resolution: versions.length > 1 ? "highest_referenced_version" : "single_effective_version",
    });
  }
  effectiveImports.sort((left, right) => left.domain.localeCompare(right.domain));

  const invalidRows = rows.filter((row) => row.status === "fail");
  const invalidDomains = [...new Set(invalidRows.map((row) => row.domain))].sort();
  const unresolvableDomains = [...byDomain]
    .filter(([, matches]) => !matches.some((row) => row.status === "pass"))
    .map(([domain]) => domain)
    .sort();
  return {
    schema: "deepbom.onnx_opset_import_contract.v1.1",
    status: invalidRows.length ? "fail" : "pass",
    import_count: rows.length,
    valid_import_count: rows.length - invalidRows.length,
    invalid_import_count: invalidRows.length,
    effective_domain_count: effectiveImports.length,
    duplicate_domain_count: [...byDomain.values()].filter((matches) => matches.length > 1).length,
    duplicate_identical_domain_count: duplicateIdenticalDomainCount,
    duplicate_version_variant_domain_count: duplicateVersionVariantDomainCount,
    invalid_version_count: invalidRows.filter((row) => row.reason_codes.includes("opset_version_not_positive_safe_integer")).length,
    invalid_domains: invalidDomains,
    unresolvable_domains: unresolvableDomains,
    effective_imports: effectiveImports,
    rows,
    resolution_rule: "highest_version_per_normalized_domain",
    method: "Preserve every OperatorSetIdProto in source order, require positive safe-integer versions, normalize the default domain aliases, and bind nodes to the highest valid referenced version for each domain as specified by ModelProto.opset_import. Repeated records remain explicit diagnostics rather than being discarded.",
  };
}

export function effectiveOnnxOpsetMap(opsets) {
  const contract = assessOnnxOpsetImports(opsets);
  return new Map(contract.effective_imports.map((row) => [row.domain, row.version]));
}

export function normalizeOnnxDomain(value) {
  const domain = String(value || "").trim();
  return !domain || domain === "ai.onnx" ? "ai.onnx" : domain;
}
