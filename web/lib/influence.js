// Rendering only. Influence propagation is computed by the Rust/WASM analyzer.

function colormap(t, mode) {
  let r, g, b;
  if (mode === "output") {
    if (t < 0.33)      { const f=t/0.33;        r=Math.round(80+f*155);  g=Math.round(f*60);      b=0; }
    else if (t < 0.66) { const f=(t-0.33)/0.33; r=255;                   g=Math.round(60+f*160);  b=0; }
    else               { const f=(t-0.66)/0.34; r=255;                   g=Math.round(220+f*35);  b=Math.round(f*255); }
  } else {
    if (t < 0.33)      { const f=t/0.33;        r=0;                     g=Math.round(f*220);     b=Math.round(100+f*155); }
    else if (t < 0.66) { const f=(t-0.33)/0.33; r=Math.round(f*255);    g=Math.round(220+f*35);  b=Math.round(255-f*255); }
    else               { const f=(t-0.66)/0.34; r=255;                   g=255;                   b=Math.round(f*255); }
  }
  return [r, g, b];
}

function renderChannelBar(map, numChannels, maxDim, mode) {
  if (numChannels < 1) return null;                        // C1: guard zero-channel crash
  const bW = Math.max(2, Math.floor(maxDim / numChannels));
  const bH = 60;
  const canvas = document.createElement("canvas");
  canvas.width = numChannels * bW; canvas.height = bH;
  canvas.style.maxWidth = maxDim + "px";                   // K1: cap overflow on high-ch models
  canvas.style.width = "100%";
  let maxVal = 0;
  for (let i = 0; i < Math.min(map.length, numChannels); i++) if (map[i] > maxVal) maxVal = map[i];
  if (maxVal === 0) return null;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;                                   // C1: null context guard
  ctx.fillStyle = "#f6f7f4"; ctx.fillRect(0, 0, canvas.width, bH);
  for (let c = 0; c < numChannels; c++) {
    const v = c < map.length ? map[c] : 0;                // W1-adjacent: bounds check
    const [r, g, b] = colormap(v / maxVal, mode);
    const barH = Math.max(1, Math.round((v / maxVal) * bH));
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(c * bW, bH - barH, bW, barH);
  }
  return canvas;
}

export function renderInfluenceCanvas(result, maxDim = 300, mode = "input") {
  if (!result) return null;
  const { map, h, w } = result;
  if (!map?.length) return null;
  if (w === 1 && h > 4) return renderChannelBar(map, h, maxDim, mode);
  const scale = Math.max(1, Math.floor(maxDim / Math.max(h, w)));
  const canvas = document.createElement("canvas");
  canvas.width = w * scale; canvas.height = h * scale;
  let maxVal = result.max_val || 0;
  if (!maxVal) for (const v of map) if (Math.abs(v) > maxVal) maxVal = Math.abs(v); // B1: use |v|
  if (maxVal === 0) return null;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const img = ctx.createImageData(w * scale, h * scale);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const raw = (y * w + x) < map.length ? map[y * w + x] : 0; // W1: bounds guard
      const [r, g, b] = colormap(Math.min(1, Math.max(0, raw / maxVal)), mode);
      for (let py = 0; py < scale; py++)
        for (let px = 0; px < scale; px++) {
          const i = ((y * scale + py) * w * scale + x * scale + px) * 4;
          img.data[i]=r; img.data[i+1]=g; img.data[i+2]=b; img.data[i+3]=255;
        }
    }
  ctx.putImageData(img, 0, 0);
  return canvas;
}
