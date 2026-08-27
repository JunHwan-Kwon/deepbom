export function stripRustTests(source) {
  let output = "";
  let cursor = 0;
  while (cursor < source.length) {
    const attr = source.indexOf("#[cfg(test)]", cursor);
    if (attr < 0) {
      output += source.slice(cursor);
      break;
    }
    const modIndex = source.indexOf("mod tests", attr);
    if (modIndex < 0) {
      output += source.slice(cursor);
      break;
    }
    const between = source.slice(attr + "#[cfg(test)]".length, modIndex);
    if (!/^\s*$/.test(between)) {
      output += source.slice(cursor, modIndex + "mod tests".length);
      cursor = modIndex + "mod tests".length;
      continue;
    }
    const openBrace = source.indexOf("{", modIndex);
    if (openBrace < 0) {
      output += source.slice(cursor);
      break;
    }
    const closeBrace = findMatchingBrace(source, openBrace);
    if (closeBrace < 0) {
      output += source.slice(cursor);
      break;
    }
    output += source.slice(cursor, attr);
    cursor = closeBrace + 1;
  }
  return output.trim();
}

function findMatchingBrace(source, openBrace) {
  let depth = 0;
  let mode = "code";
  for (let index = openBrace; index < source.length; index += 1) {
    const char = source[index];
    const next = source[index + 1];
    const prev = source[index - 1];

    if (mode === "line-comment") {
      if (char === "\n") mode = "code";
      continue;
    }
    if (mode === "block-comment") {
      if (char === "*" && next === "/") {
        mode = "code";
        index += 1;
      }
      continue;
    }
    if (mode === "string") {
      if (char === "\"" && prev !== "\\") mode = "code";
      continue;
    }
    if (mode === "char") {
      if (char === "'" && prev !== "\\") mode = "code";
      continue;
    }

    if (char === "/" && next === "/") {
      mode = "line-comment";
      index += 1;
      continue;
    }
    if (char === "/" && next === "*") {
      mode = "block-comment";
      index += 1;
      continue;
    }
    if (char === "\"") {
      mode = "string";
      continue;
    }
    if (char === "'") {
      mode = "char";
      continue;
    }

    if (char === "{") depth += 1;
    if (char === "}") {
      depth -= 1;
      if (depth === 0) return index;
    }
  }
  return -1;
}
