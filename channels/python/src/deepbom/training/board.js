/* Read-only projection of validated native evidence; no model execution. */
const data = JSON.parse(document.getElementById("data").textContent),
  q = (s) => document.querySelector(s),
  ns = "http://www.w3.org/2000/svg";
let chosen = null;
function el(tag, attributes, text) {
  const n = document.createElementNS(ns, tag);
  for (const [k, v] of Object.entries(attributes || {})) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
}
function plot() {
  const svg = q("#hist");
  svg.replaceChildren();
  const type = q("#view").value,
    t = chosen?.tensor;
  q("#tensor").textContent = JSON.stringify(chosen || {}, null, 2);
  q("#summary").replaceChildren();
  if (!t) return;
  for (const [name, value] of Object.entries({
    Status: t.status,
    Dtype: t.identity.dtype,
    Shape: JSON.stringify(t.identity.shape),
    "Element count": t.identity.element_count,
    "Zero count": t.statistics?.zero_count,
    Mean: t.statistics?.mean,
    "Population stddev": t.statistics?.population_stddev,
    Minimum: t.statistics?.minimum,
    Maximum: t.statistics?.maximum,
  })) {
    const tr = document.createElement("tr"),
      a = document.createElement("th"),
      b = document.createElement("td");
    a.textContent = name;
    b.textContent =
      value === null || value === undefined ? "Not assessed" : String(value);
    tr.append(a, b);
    q("#summary").append(tr);
  }
  const title = `${t.identity.name} · ${type} · ${chosen.source.snapshot_sha256}`;
  svg.append(el("title", {}, title));
  q("#figure-title").textContent = title;
  const feature = t.features;
  let cells = null,
    rows = 0,
    columns = 0;
  if (type === "histogram") {
    const hist = t.statistics?.histogram,
      counts = hist?.counts || [],
      maximum = Math.max(1, ...counts.map(Number));
    counts.forEach((v, i) => {
      const r = el("rect", {
        x: 20 + (i * 760) / counts.length,
        y: 180 - (Number(v) / maximum) * 150,
        width: Math.max(1, 760 / counts.length - 1),
        height: (Number(v) / maximum) * 150,
        fill: "#388c78",
        tabindex: 0,
      });
      r.append(
        el(
          "title",
          {},
          `[${hist.edges[i]}, ${hist.edges[i + 1]}${i === counts.length - 1 ? "]" : ")"}: ${v} values`,
        ),
      );
      svg.append(r);
    });
    svg.append(
      el(
        "text",
        { x: 20, y: 215, fill: "currentColor", "font-size": 12 },
        "Fixed log2 intervals; counts are exact, horizontal spacing is categorical.",
      ),
    );
  } else if (type === "spectrum") {
    const s = feature?.spectrum;
    if (s?.status !== "assessed") {
      q("#figure-title").textContent = s?.reason || "Spectrum not collected";
      return;
    }
    const v = s.normalized_singular_values,
      m = Math.max(...v, 1),
      points = v
        .map(
          (x, i) =>
            `${25 + (i / Math.max(1, v.length - 1)) * 740},${180 - (x / m) * 150}`,
        )
        .join(" ");
    svg.append(
      el("polyline", {
        points,
        fill: "none",
        stroke: "#388c78",
        "stroke-width": 3,
      }),
    );
    svg.append(
      el(
        "text",
        { x: 20, y: 215, fill: "currentColor", "font-size": 12 },
        `Normalized singular spectrum · rank ${s.numerical_rank} · no task-quality claim`,
      ),
    );
  } else if (type === "channels") {
    const c = feature?.channels;
    if (c?.status !== "assessed") {
      q("#figure-title").textContent =
        c?.reason || "Channel slices not collected";
      return;
    }
    cells = c.channels.map((c) => ({
      v: c.statistics.rms,
      label: `Storage-axis slice ${c.index}: RMS ${c.statistics.rms}, zero count ${c.statistics.zero_count}`,
    }));
    columns = Math.min(32, c.count);
    rows = Math.ceil(c.count / columns);
  } else if (type === "similarity") {
    const s = feature?.similarity;
    if (s?.status !== "assessed") {
      q("#figure-title").textContent = s?.reason || "Similarity not collected";
      return;
    }
    rows = columns = s.size;
    cells = s.values.map((v, i) => ({
      v,
      label: `${Math.floor(i / columns)} × ${i % columns}: ${v === null ? "undefined zero norm" : v}`,
    }));
  } else {
    const p = feature?.projection;
    if (p?.status !== "assessed") {
      q("#figure-title").textContent = p?.reason || "Projection not collected";
      return;
    }
    rows = p.rows;
    columns = p.columns;
    cells = p.cells.map((c, i) => ({
      v: type === "sparsity" ? c.zero_count / c.count : c.mean,
      label: `Tile ${i}: ${c.count} values, mean ${c.mean}, zeros ${c.zero_count}`,
    }));
  }
  if (cells) {
    const max = Math.max(...cells.map((c) => Math.abs(c.v || 0)), 1e-300),
      w = 760 / columns,
      h = Math.min(20, 170 / rows);
    cells.forEach((c, i) => {
      const r = el("rect", {
        x: 20 + (i % columns) * w,
        y: 20 + Math.floor(i / columns) * h,
        width: Math.max(0.1, w - 0.4),
        height: Math.max(0.1, h - 0.4),
        fill: c.v === null ? "#aaa" : c.v < 0 ? "#b97149" : "#267663",
        "fill-opacity":
          c.v === null ? 0.5 : 0.12 + (0.88 * Math.abs(c.v)) / max,
        tabindex: 0,
      });
      r.append(el("title", {}, c.label));
      svg.append(r);
    });
  }
}
function render() {
  const i = Number(q("#step").value),
    e = data.events[i];
  q("#coordinate").textContent = `Observation ${i + 1} / ${data.events.length}`;
  q("#details").textContent = JSON.stringify(e || {}, null, 2);
  const key = q("#metric").value,
    series = data.events.flatMap((r, j) =>
      typeof r.payload.values?.[key] === "number" &&
      Number.isFinite(r.payload.values[key])
        ? [{ v: r.payload.values[key], j }]
        : [],
    ),
    a = Math.min(...series.map((x) => x.v)),
    b = Math.max(...series.map((x) => x.v));
  q("#curve").setAttribute(
    "points",
    series
      .map(
        (x) =>
          `${30 + (x.j / Math.max(1, data.events.length - 1)) * 740},${170 - ((x.v - a) / (b - a || 1)) * 140}`,
      )
      .join(" "),
  );
  q("#plot-label").textContent = key
    ? `${key}: min ${a}, max ${b}; observation sequence; original coordinates below`
    : "No scalar metrics recorded";
  const current = data.events
      .slice(0, i + 1)
      .filter((e) => e.kind === "model_snapshot")
      .at(-1),
    links = current?.payload.evidence;
  q("#weights").replaceChildren();
  q("#activations").replaceChildren();
  chosen = null;
  function add(container, obj, tensor, context) {
    const button = document.createElement("button");
    button.textContent = tensor.identity.name;
    button.onclick = () => {
      for (const b of document.querySelectorAll(
        "#weights button,#activations button",
      ))
        b.setAttribute("aria-pressed", "false");
      button.setAttribute("aria-pressed", "true");
      chosen = { source: obj.source, context: context || null, tensor };
      plot();
    };
    q(container).append(button);
    if (!chosen) {
      chosen = { source: obj.source, context: context || null, tensor };
    }
  }
  if (links) {
    const weight = data.objects[links.weight_ir.sha256];
    for (const tensor of weight.tensors) add("#weights", weight, tensor);
    q("#state-note").textContent =
      `Last explicitly captured state at observation ${Number(current.sequence) + 1}; not inferred to match later metrics. ${weight.source.snapshot_sha256}`;
  } else
    q("#state-note").textContent =
      "No model state captured before this observation.";
  for (const event of data.events
    .slice(0, i + 1)
    .filter((e) => e.kind === "activation_capture")) {
    const obj = data.objects[event.payload.evidence.sha256];
    for (const tensor of obj.tensors)
      add("#activations", obj, tensor, {
        ...obj.context,
        subject_ref: tensor.subject_ref,
        invocation: tensor.invocation,
      });
  }
  plot();
}
for (const k of new Set(
  data.events.flatMap((r) => Object.keys(r.payload.values || {})),
)) {
  const o = document.createElement("option");
  o.value = k;
  o.textContent = k;
  q("#metric").append(o);
}
q("#step").max = Math.max(0, data.events.length - 1);
q("#step").value = q("#step").max;
q("#step").oninput = render;
q("#metric").onchange = render;
q("#view").onchange = plot;
q("#refresh").onclick = () => location.reload();
function save(blob, name) {
  const a = document.createElement("a"),
    url = URL.createObjectURL(blob);
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
q("#svg").onclick = () =>
  save(
    new Blob([new XMLSerializer().serializeToString(q("#hist"))], {
      type: "image/svg+xml",
    }),
    "deepbom-evidence.svg",
  );
q("#png").onclick = async () => {
  const svg = q("#hist").cloneNode(true);
  svg.setAttribute("xmlns", ns);
  svg.setAttribute("width", "1600");
  svg.setAttribute("height", "440");
  const url = URL.createObjectURL(
    new Blob([new XMLSerializer().serializeToString(svg)], {
      type: "image/svg+xml",
    }),
  );
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    const c = document.createElement("canvas");
    c.width = 1600;
    c.height = 440;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.drawImage(img, 0, 0);
    const blob = await new Promise((r) => c.toBlob(r, "image/png"));
    if (blob) save(blob, "deepbom-evidence.png");
  } finally {
    URL.revokeObjectURL(url);
  }
};
render();
