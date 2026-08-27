const UNKNOWN = /^(?:unknown|unbound|not[_ -]?(?:assessed|declared|embedded|available)|n\/a|none)$/i;

function boundedText(value, maximum) {
  const text = String(value ?? "").replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, " ").trim();
  return !text || UNKNOWN.test(text) ? "" : text.slice(0, maximum);
}

export function artifactComponentMetadata(analysis = {}) {
  const metadata = analysis.metadata_presence || {};
  const description = boundedText(
    metadata.metadata_model_description || metadata.description || metadata.model_doc_string,
    16_384,
  );
  const author = boundedText(metadata.metadata_author, 1_024);
  const license = boundedText(metadata.metadata_license, 4_096);
  return {
    ...(description ? { description } : {}),
    ...(author ? { authors: [{ name: author }] } : {}),
    ...(license ? { licenses: [{ license: { name: license } }] } : {}),
  };
}
