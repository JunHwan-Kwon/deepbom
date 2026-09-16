import { canvasToPngBytes } from "./dom.js";
import { sha256Hex } from "./hash.js";
import { modelIrVisualizationFiles } from "./model-ir-visualization.js";
import { jsonForDownload, zipBinaryFile, zipTextFile } from "./report-utils.js";
import { createZipBlob } from "./zip.js";

export async function buildBrowserModelIrVisualizationArchive(modelIr, { orientation = "portrait" } = {}) {
  const generated = modelIrVisualizationFiles(modelIr, { orientation });
  const files = generated.files.map((file) => zipTextFile(file.name, file.data));
  const pngRecords = [];
  for (const page of generated.bundle.pages) {
    const bytes = await rasterizeMonochromeModelView(page.svg, page.render_model);
    const name = page.filename.replace(/\.svg$/, ".png");
    files.push(zipBinaryFile(name, bytes));
    pngRecords.push({
      name,
      sha256: await sha256Hex(bytes),
      source_svg_sha256: page.sha256,
      resolution_dpi: 300,
      color_contract: "black_white_only",
      derivation: "browser_svg_raster_threshold_v1",
    });
  }
  const pngManifest = {
    schema: "deepbom.model_ir_png_derivation_manifest.v1",
    model_ir_sha256: modelIr.model_ir_sha256,
    canonical_source: "hash_bound_svg_page",
    files: pngRecords,
    interpretation_boundary: "Browser PNG pages are thresholded black-and-white derivatives for document systems that cannot preserve SVG. Their hashes bind this local rendering; canonical SVG pages remain authoritative.",
  };
  files.push(zipTextFile("model-views/png-derivation-manifest.json", jsonForDownload(pngManifest)));
  return { blob: createZipBlob(files), bundle: generated.bundle, png_manifest: pngManifest };
}

export async function rasterizeMonochromeModelView(svg, renderModel, { dpi = 300 } = {}) {
  if (!Number.isFinite(dpi) || dpi < 72 || dpi > 300) {
    throw new Error("Model IR PNG resolution must be between 72 and 300 DPI.");
  }
  const page = renderModel?.page || renderModel;
  const width = Math.round((page.width_mm / 25.4) * dpi);
  const height = Math.round((page.height_mm / 25.4) * dpi);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas 2D is unavailable for PNG derivation.");
  context.fillStyle = "#fff";
  context.fillRect(0, 0, width, height);
  if (Array.isArray(renderModel?.rows)) {
    // MCP App hosts may prohibit decoding blob/data SVG sources even though
    // the same canonical SVG is safe to display as DOM. Draw the already
    // validated render model directly so PNG export never depends on an image
    // source or CSP exception. The canonical SVG remains authoritative.
    drawRenderModel(context, renderModel, width, height);
  } else {
    await drawDecodedSvg(context, svg, width, height);
  }
  const raster = context.getImageData(0, 0, width, height);
  for (let offset = 0; offset < raster.data.length; offset += 4) {
    const black = (raster.data[offset] * 299 + raster.data[offset + 1] * 587 + raster.data[offset + 2] * 114) < 224000;
    const value = black ? 0 : 255;
    raster.data[offset] = value;
    raster.data[offset + 1] = value;
    raster.data[offset + 2] = value;
    raster.data[offset + 3] = 255;
  }
  context.putImageData(raster, 0, 0);
  return canvasToPngBytes(canvas);
}

function drawRenderModel(context, renderModel, width, height) {
  const source = renderModel.page;
  const scaleX = width / source.width;
  const scaleY = height / source.height;
  context.save();
  context.scale(scaleX, scaleY);
  context.strokeStyle = "#000";
  context.fillStyle = "#000";
  context.lineWidth = 2;
  context.strokeRect(42, 42, source.width - 84, source.height - 84);
  context.font = "700 24px Arial, sans-serif";
  context.fillText(`${renderModel.projection.level} ${renderModel.projection.title}`, 60, 76);
  context.font = "15px Arial, sans-serif";
  const rows = renderModel.rows;
  const headerHeight = 130;
  const footerHeight = 82;
  const rowHeight = Math.max(42, Math.floor((source.height - headerHeight - footerHeight) / Math.max(1, rows.length)));
  for (const [index, row] of rows.entries()) {
    const y = headerHeight + index * rowHeight;
    context.setLineDash(dashPattern(row.evidence_class));
    context.strokeRect(60, y, source.width - 120, rowHeight - 6);
    context.setLineDash([]);
    context.font = "700 15px Arial, sans-serif";
    context.fillText(clipText(`${row.kind} | ${row.title}`, 112), 72, y + 20);
    context.font = "15px Arial, sans-serif";
    context.fillText(clipText(row.detail, 128), 72, y + 40);
    context.font = "12px ui-monospace, monospace";
    context.fillText(clipText(row.subject_ref, 150), 72, y + rowHeight - 14);
  }
  context.font = "12px Arial, sans-serif";
  context.fillText("DEEPBOM Model IR | deterministic monochrome derivative; canonical SVG authoritative", 60, source.height - 52);
  context.restore();
}

async function drawDecodedSvg(context, svg, width, height) {
  const svgBlob = new Blob([svg], { type: "image/svg+xml" });
  let decoded = false;
  if (typeof createImageBitmap === "function") {
    try {
      const bitmap = await createImageBitmap(svgBlob);
      try { context.drawImage(bitmap, 0, 0, width, height); decoded = true; } finally { bitmap.close?.(); }
    } catch { /* fall through to the object URL compatibility path */ }
  }
  if (decoded) return;
  const url = URL.createObjectURL(svgBlob);
  try {
    const image = new Image();
    image.src = url;
    if (typeof image.decode === "function") await image.decode();
    else await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = () => reject(new Error("SVG page decoding failed.")); });
    context.drawImage(image, 0, 0, width, height);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function dashPattern(value) {
  const text = String(value || "").toUpperCase();
  if (text.includes("NOT_ASSESSABLE")) return [2, 7];
  if (text.includes("DERIVED") || text.includes("PREDICTED") || text.includes("INFERRED")) return [12, 7];
  return [];
}
function clipText(value, maximum) { const text = String(value ?? "").replace(/[\r\n]+/g, " "); return text.length <= maximum ? text : `${text.slice(0, maximum - 1)}…`; }
