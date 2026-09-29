import { staticTensorPayloadBytes } from "./tensor-size.js";
import { artifactIrOperators, artifactIrValues } from "./artifact-ir-selectors.js";
function tensorMeta(analysis, tensorId, tensors = tensorByIndex(analysis)) {
  const tensor = tensors.get(Number(tensorId));
  return { bytes: staticTensorPayloadBytes(tensor), dtype: tensor?.dtype ?? null };
}

export function collectFullGraph(analysis, graphIndex) {
  const ops = Array.isArray(artifactIrOperators(analysis)) ? artifactIrOperators(analysis) : [];
  const tensors = tensorByIndex(analysis);
  const nodeSet = new Set(ops.map((op) => op.index));
  const nodes = ops.map((op) => ({
    op, column: 0, distance: 0,
    outputDtype: tensors.get(Number(op.outputs?.[0]))?.dtype ?? null,
  }));
  const edges = [];
  for (const op of ops) {
    for (const tensorId of op.outputs) {
      for (const consumer of graphIndex.consumers.get(tensorId) || []) {
        if (nodeSet.has(consumer)) {
          const { bytes, dtype } = tensorMeta(analysis, tensorId, tensors);
          edges.push({ from: op.index, to: consumer, tensorId, bytes, dtype });
        }
      }
    }
  }
  return { nodes, edges };
}

export function collectNeighborhood(analysis, graphIndex, centerIndex, depth) { // analysis used for tensor bytes
  const opByIndex = new Map((Array.isArray(artifactIrOperators(analysis)) ? artifactIrOperators(analysis) : []).map((op) => [Number(op.index), op]));
  const tensors = tensorByIndex(analysis);
  const nodes = new Map();
  const queue = [{ index: centerIndex, column: 0, distance: 0 }];
  const seen = new Set([centerIndex]);

  while (queue.length) {
    const item = queue.shift();
    const op = opByIndex.get(Number(item.index));
    if (!op) continue;
    nodes.set(item.index, { op, column: item.column, distance: item.distance, outputDtype: tensors.get(Number(op.outputs?.[0]))?.dtype ?? null });
    if (item.distance >= depth || nodes.size > 120) continue;

    const prev = op.inputs
      .filter((tensorId) => graphIndex.producers.has(tensorId))
      .map((tensorId) => graphIndex.producers.get(tensorId));
    const next = op.outputs.flatMap((tensorId) => graphIndex.consumers.get(tensorId) || []);

    for (const index of prev) {
      if (!seen.has(index)) {
        seen.add(index);
        queue.push({ index, column: item.column - 1, distance: item.distance + 1 });
      }
    }
    for (const index of next) {
      if (!seen.has(index)) {
        seen.add(index);
        queue.push({ index, column: item.column + 1, distance: item.distance + 1 });
      }
    }
  }

  const edges = [];
  for (const item of nodes.values()) {
    for (const tensorId of item.op.outputs) {
      for (const consumer of graphIndex.consumers.get(tensorId) || []) {
        if (nodes.has(consumer)) {
          const { bytes, dtype } = tensorMeta(analysis, tensorId, tensors);
          edges.push({ from: item.op.index, to: consumer, tensorId, bytes, dtype });
        }
      }
    }
  }
  return { nodes: [...nodes.values()], edges };
}

function tensorByIndex(analysis) {
  return new Map((Array.isArray(artifactIrValues(analysis)) ? artifactIrValues(analysis) : []).map((tensor, position) => {
    const candidate = Number(tensor?.index);
    return [Number.isSafeInteger(candidate) && candidate >= 0 ? candidate : position, tensor];
  }));
}

export function layoutNeighborhood(nodes) {
  const columns = new Map();
  for (const item of nodes) {
    if (!columns.has(item.column)) columns.set(item.column, []);
    columns.get(item.column).push(item);
  }
  return layoutColumns(columns, { columnWidth: 238, rowHeight: 88, margin: 42 });
}

export function layoutFullGraph(nodes, edges, {
  minimumColumns = 1,
  columnWidth = 264,
  rankHeight = 138,
  marginX = 54,
  marginY = 116,
} = {}) {
  const COLUMN_WIDTH = columnWidth;
  const RANK_HEIGHT = rankHeight;
  const MARGIN_X = marginX;
  const MARGIN_Y = marginY;
  const nodeByIndex = new Map(nodes.map((item) => [item.op.index, item]));
  const incoming = new Map(nodes.map((item) => [item.op.index, []]));
  const outgoing = new Map(nodes.map((item) => [item.op.index, []]));
  const indegree = new Map(nodes.map((item) => [item.op.index, 0]));
  for (const edge of edges) {
    if (!nodeByIndex.has(edge.from) || !nodeByIndex.has(edge.to)) continue;
    incoming.get(edge.to).push(edge.from);
    outgoing.get(edge.from).push(edge.to);
    indegree.set(edge.to, indegree.get(edge.to) + 1);
  }

  // Stable Kahn traversal derives a real top-down DAG rank. The bounded
  // fallback keeps malformed cyclic input visible without claiming a DAG.
  const ready = [...nodeByIndex.keys()].filter((index) => indegree.get(index) === 0).sort((a, b) => a - b);
  const rankByIndex = new Map(ready.map((index) => [index, 0]));
  const visited = new Set();
  while (ready.length) {
    const index = ready.shift();
    visited.add(index);
    const nextRank = (rankByIndex.get(index) || 0) + 1;
    for (const next of outgoing.get(index) || []) {
      rankByIndex.set(next, Math.max(rankByIndex.get(next) || 0, nextRank));
      indegree.set(next, indegree.get(next) - 1);
      if (indegree.get(next) === 0) {
        ready.push(next);
        ready.sort((a, b) => a - b);
      }
    }
  }
  for (const index of [...nodeByIndex.keys()].filter((item) => !visited.has(item)).sort((a, b) => a - b)) {
    const predecessorRanks = (incoming.get(index) || []).map((item) => rankByIndex.get(item)).filter(Number.isFinite);
    rankByIndex.set(index, predecessorRanks.length ? Math.max(...predecessorRanks) + 1 : 0);
  }

  const ranks = new Map();
  for (const item of nodes) {
    const rank = rankByIndex.get(item.op.index) || 0;
    if (!ranks.has(rank)) ranks.set(rank, []);
    ranks.get(rank).push(item);
  }
  const rankIds = [...ranks.keys()].sort((a, b) => a - b);
  const widestRank = Math.max(1, ...[...ranks.values()].map((items) => items.length));
  const displayColumns = Math.max(widestRank, Number.isSafeInteger(minimumColumns) ? minimumColumns : 1);
  const positions = new Map();
  const laidOut = [];
  for (const rank of rankIds) {
    const group = ranks.get(rank);
    group.sort((left, right) => {
      const preferred = (item) => {
        const parents = (incoming.get(item.op.index) || []).map((index) => positions.get(index)?.x).filter(Number.isFinite);
        return parents.length ? parents.reduce((sum, value) => sum + value, 0) / parents.length : item.op.index;
      };
      return preferred(left) - preferred(right) || left.op.index - right.op.index;
    });
    const offset = (displayColumns - group.length) * COLUMN_WIDTH / 2;
    group.forEach((item, slot) => {
      const placed = {
        ...item,
        column: rank,
        x: MARGIN_X + offset + slot * COLUMN_WIDTH,
        y: MARGIN_Y + rank * RANK_HEIGHT,
      };
      positions.set(item.op.index, placed);
      laidOut.push(placed);
    });
  }
  return {
    nodes: laidOut,
    positions,
    orientation: "top-down",
    isDag: visited.size === nodes.length,
    bounds: {
      x: 0,
      y: 0,
      width: MARGIN_X * 2 + displayColumns * COLUMN_WIDTH,
      height: MARGIN_Y * 2 + Math.max(1, rankIds.length) * RANK_HEIGHT,
    },
  };
}

export function layoutFoldedGraph(nodes) {
  const COLUMN_WIDTH = 222;
  const ROW_HEIGHT = 86;
  const MARGIN = 42;
  const COLUMNS = 8;
  const positions = new Map();
  const laidOut = [];
  [...nodes].sort((left, right) => left.op.index - right.op.index).forEach((item, sequence) => {
    const row = Math.floor(sequence / COLUMNS);
    const slot = sequence % COLUMNS;
    const column = row % 2 === 0 ? slot : COLUMNS - 1 - slot;
    const placed = { ...item, x: MARGIN + column * COLUMN_WIDTH, y: MARGIN + row * ROW_HEIGHT };
    positions.set(item.op.index, placed);
    laidOut.push(placed);
  });
  return {
    nodes: laidOut,
    positions,
    orientation: "folded",
    bounds: {
      x: 0,
      y: 0,
      width: MARGIN * 2 + COLUMNS * COLUMN_WIDTH,
      height: MARGIN * 2 + Math.max(1, Math.ceil(nodes.length / COLUMNS)) * ROW_HEIGHT,
    },
  };
}

function layoutColumns(columns, { columnWidth, rowHeight, margin }) {
  const columnIds = [...columns.keys()].sort((a, b) => a - b);
  const positions = new Map();
  const laidOut = [];
  columnIds.forEach((column, columnIndex) => {
    const items = columns.get(column).sort((a, b) => a.op.index - b.op.index);
    items.forEach((item, rowIndex) => {
      const x = margin + columnIndex * columnWidth;
      const y = margin + rowIndex * rowHeight;
      const placed = { ...item, x, y };
      positions.set(item.op.index, placed);
      laidOut.push(placed);
    });
  });
  const maxRows = Math.max(1, ...[...columns.values()].map((items) => items.length));
  const bounds = {
    x: 0,
    y: 0,
    width: margin * 2 + Math.max(1, columnIds.length) * columnWidth,
    height: margin * 2 + maxRows * rowHeight,
  };
  return { nodes: laidOut, positions, bounds };
}
