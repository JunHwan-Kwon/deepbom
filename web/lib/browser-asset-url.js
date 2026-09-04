export function browserAssetUrl(browserPath, moduleRelativePath, moduleUrl) {
  const documentBase = globalThis.document?.baseURI;
  if (documentBase) return new URL(browserPath, documentBase);
  return new URL(moduleRelativePath, moduleUrl);
}
