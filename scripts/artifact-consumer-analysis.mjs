import { parse } from "acorn";

// Conservative syntactic taint propagation, including aliases and destructuring.
// This is a regression guard, not a whole-program type proof.
export function directAnalysisLedgerReads(source) {
  const tree = parse(source, { ecmaVersion: "latest", sourceType: "module", allowHashBang: true });
  const nodes = [];
  const visit = node => {
    if (!node || typeof node !== "object") return;
    if (node.type) nodes.push(node);
    for (const [key, value] of Object.entries(node)) {
      if (["start", "end", "loc", "range"].includes(key)) continue;
      if (Array.isArray(value)) value.forEach(visit);
      else if (value && typeof value === "object") visit(value);
    }
  };
  visit(tree);
  const aliases = new Set(["analysis"]);
  const root = node => node?.type === "ChainExpression" ? root(node.expression)
    : node?.type === "Identifier" ? aliases.has(node.name)
    : node?.type === "LogicalExpression" ? root(node.left) || root(node.right) : false;
  let changed = true;
  while (changed) {
    changed = false;
    for (const node of nodes) {
      const left = node.type === "VariableDeclarator" ? node.id : node.type === "AssignmentExpression" ? node.left : null;
      const right = node.type === "VariableDeclarator" ? node.init : node.right;
      if (left?.type === "Identifier" && root(right) && !aliases.has(left.name)) { aliases.add(left.name); changed = true; }
    }
  }
  const property = (node, computed) => computed ? node?.type === "Literal" ? node.value : null : node?.name ?? node?.value;
  const hits = [];
  for (const node of nodes) {
    if (node.type === "MemberExpression" && root(node.object)) {
      const name = property(node.property, node.computed);
      if (["ops", "tensors"].includes(name)) hits.push({ property: name, start: node.start });
    }
    const pattern = node.type === "VariableDeclarator" ? node.id : node.type === "AssignmentExpression" ? node.left : null;
    const value = node.type === "VariableDeclarator" ? node.init : node.right;
    if (pattern?.type === "ObjectPattern" && root(value)) {
      for (const p of pattern.properties) {
        const name = property(p.key, p.computed);
        if (["ops", "tensors"].includes(name)) hits.push({ property: name, start: p.start });
      }
    }
  }
  return hits;
}
