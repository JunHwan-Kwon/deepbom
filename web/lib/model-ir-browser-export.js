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
    const bytes = await rasterizeMonochromeModelView(page.svg, page.render_model.page);
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

async function rasterizeMonochromeModelView(svg, page) {
  const width = page.width_mm > page.height_mm ? 3508 : 2480;
  const height = page.width_mm > page.height_mm ? 2480 : 3508;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas 2D is unavailable for PNG derivation.");
  context.fillStyle = "#fff";
  context.fillRect(0, 0, width, height);
  const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
  try {
    const image = new Image();
    image.src = url;
    if (typeof image.decode === "function") await image.decode();
    else await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error("SVG page decoding failed."));
    });
    context.drawImage(image, 0, 0, width, height);
  } finally {
    URL.revokeObjectURL(url);
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
