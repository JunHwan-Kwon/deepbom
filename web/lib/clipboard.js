export async function copyTextToClipboard(text, doc = document, nav = navigator) {
  try {
    await nav.clipboard.writeText(text);
  } catch (clipboardError) {
    const textArea = doc.createElement("textarea");
    textArea.value = text;
    doc.body.append(textArea);
    textArea.select();
    const copied = doc.execCommand("copy");
    textArea.remove();
    if (!copied) throw clipboardError;
  }
}
