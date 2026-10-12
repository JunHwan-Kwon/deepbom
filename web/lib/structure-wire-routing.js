// Display geometry only. Tensor bindings remain owned by the structure draft.
const distance = (a, b) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
const inside = (p, b) => p.x > b.x && p.x < b.x + b.width && p.y > b.y && p.y < b.y + b.height;
export function segmentHitsBox(a, b, box) {
  if (a.x === b.x) return a.x > box.x && a.x < box.x + box.width && Math.max(a.y, b.y) > box.y && Math.min(a.y, b.y) < box.y + box.height;
  if (a.y === b.y) return a.y > box.y && a.y < box.y + box.height && Math.max(a.x, b.x) > box.x && Math.min(a.x, b.x) < box.x + box.width;
  return true;
}
const clear = (points, boxes) => points.every((p, i) => !i || boxes.every(b => !segmentHitsBox(points[i - 1], p, b)));
const cost = points => points.reduce((n, p, i) => n + (i ? distance(points[i - 1], p) : 0), 0);
function simplify(points) {
  const out = [];
  for (const p of points) {
    if (out.length && !distance(out.at(-1), p)) continue;
    while (out.length > 1) {
      const a = out.at(-2), b = out.at(-1);
      if (!((a.x === b.x && b.x === p.x && (b.y - a.y) * (p.y - b.y) >= 0)
        || (a.y === b.y && b.y === p.y && (b.x - a.x) * (p.x - b.x) >= 0))) break;
      out.pop();
    }
    out.push(p);
  }
  return out;
}
export function roundedWirePath(points, radius = 7) {
  points = simplify(points);
  if (!points.length) return '';
  let d = `M${points[0].x},${points[0].y}`;
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1], b = points[i], c = points[i + 1];
    if (!c) { d += ` L${b.x},${b.y}`; continue; }
    const r = Math.min(radius, distance(a, b) / 2, distance(b, c) / 2);
    const before = { x: b.x - Math.sign(b.x - a.x) * r, y: b.y - Math.sign(b.y - a.y) * r };
    const after = { x: b.x + Math.sign(c.x - b.x) * r, y: b.y + Math.sign(c.y - b.y) * r };
    d += ` L${before.x},${before.y} Q${b.x},${b.y} ${after.x},${after.y}`;
  }
  return d;
}

export function routeStructureWires(edges, nodes) {
  const boxes = nodes.map(b => ({ x: b.x - 7, y: b.y - 7, width: b.width + 14, height: b.height + 14 }));
  const left = Math.min(...nodes.map(b => b.x), 80) - 25;
  const right = Math.max(...nodes.map(b => b.x + b.width), 300) + 25;
  const lanes = new Map();
  return edges.map(edge => {
    const { start: a, end: b } = edge;
    const start = { x: a.x, y: a.y + 15 }, end = { x: b.x, y: b.y - 15 };
    let points = null, side = false;
    // Prefer a straight trunk, then a short elbow in the inter-layer gutter.
    if (b.y > a.y && a.x === b.x && clear([a, b], boxes)) points = [a, b];
    if (!points && b.y - a.y > 30) {
      for (const y of [(a.y + b.y) / 2, start.y, end.y]) {
        const candidate = [a, { x: a.x, y }, { x: b.x, y }, b];
        if (clear(candidate, boxes)) { points = candidate; break; }
      }
    }
    if (!points) {
      const candidates = [], lo = Math.min(start.y, end.y), hi = Math.max(start.y, end.y);
      for (const direction of [-1, 1]) for (let lane = 0; lane < edges.length + 1; lane++) {
        const x = direction < 0 ? left - lane * 14 : right + lane * 14;
        if (x < 12) break;
        const occupied = lanes.get(x) || [];
        if (occupied.some(row => row.lo < hi + 12 && row.hi > lo - 12 && row.net !== edge.value_ref)) continue;
        const path = [a, start, { x, y: start.y }, { x, y: end.y }, end, b];
        if (clear(path, boxes)) { candidates.push({ path, x }); break; }
      }
      candidates.sort((a, b) => cost(a.path) - cost(b.path));
      if (candidates.length) {
        const chosen = candidates[0]; points = chosen.path; side = true;
        const occupied = lanes.get(chosen.x) || [];
        occupied.push({ lo, hi, net: edge.value_ref }); lanes.set(chosen.x, occupied);
      }
    }
    // Dragged nodes can block the simple gutters: search visible orthogonal
    // corridors. Overlapping endpoint cards are reported rather than hidden.
    if (!points && clear([a, start], boxes) && clear([end, b], boxes)) {
      const middle = corridorPath(start, end, boxes);
      if (middle) { points = [a, ...middle, b]; side = true; }
    }
    const blocked = !points;
    points = simplify(points || [a, start, { x: b.x, y: start.y }, b]);
    return { ...edge, points, side, blocked, path: roundedWirePath(points) };
  });
}

function corridorPath(start, end, boxes) {
  if (boxes.some(b => inside(start, b) || inside(end, b))) return null;
  const sorted = values => [...new Set(values)].sort((a, b) => a - b);
  const xs = sorted([start.x, end.x, ...boxes.flatMap(b => [b.x - 8, b.x + b.width + 8])].filter(x => x >= 12));
  const ys = sorted([start.y, end.y, ...boxes.flatMap(b => [b.y - 8, b.y + b.height + 8])].filter(y => y >= 12));
  if (xs.length * ys.length > 40000) return null;
  const key = (x, y) => y * xs.length + x, point = id => ({ x: xs[id % xs.length], y: ys[Math.floor(id / xs.length)] });
  const first = key(xs.indexOf(start.x), ys.indexOf(start.y)), last = key(xs.indexOf(end.x), ys.indexOf(end.y));
  const heap = [], best = new Map([[first, 0]]), previous = new Map();
  function push(row) { heap.push(row); let i = heap.length - 1; while (i) { const p = (i - 1) >> 1; if (heap[p].score <= row.score) break; heap[i] = heap[p]; i = p; } heap[i] = row; }
  function pop() { const row = heap[0], tail = heap.pop(); if (heap.length) { let i = 0; while (i * 2 + 1 < heap.length) { let child = i * 2 + 1; if (child + 1 < heap.length && heap[child + 1].score < heap[child].score) child++; if (heap[child].score >= tail.score) break; heap[i] = heap[child]; i = child; } heap[i] = tail; } return row; }
  push({ id: first, score: distance(start, end), cost: 0 });
  while (heap.length) {
    const row = pop(); if (row.cost !== best.get(row.id)) continue;
    if (row.id === last) { const path = [end]; let id = last; while (previous.has(id)) { id = previous.get(id); path.push(point(id)); } return path.reverse(); }
    const p = point(row.id), x = row.id % xs.length, y = Math.floor(row.id / xs.length);
    for (const [nx, ny] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) {
      if (nx < 0 || nx >= xs.length || ny < 0 || ny >= ys.length) continue;
      const id = key(nx, ny), next = point(id), cost = row.cost + distance(p, next);
      if (cost >= (best.get(id) ?? Infinity) || boxes.some(b => segmentHitsBox(p, next, b))) continue;
      best.set(id, cost); previous.set(id, row.id); push({ id, cost, score: cost + distance(next, end) });
    }
  }
  return null;
}
