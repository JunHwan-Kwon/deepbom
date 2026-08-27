import { shapeText } from "./format.js";

export function inferImageInputSpec(analysis) {
  const inputs = Array.isArray(analysis?.inputs) ? analysis.inputs : [];
  const tensor = inputs.find((item) => isImageLikeTensor(item)) || inputs[0];
  if (!tensor) {
    return { ok: false, reason: "The model has no input tensors." };
  }
  const shape = Array.isArray(tensor.shape) ? tensor.shape.map((dim, index) => resolveStaticDim(dim, index)) : [];
  if (shape.some((dim) => dim <= 0)) {
    return {
      ok: false,
      reason: `Input ${tensor.name || `T${tensor.index}`} has unresolved dynamic shape ${shapeText(tensor.shape)}.`,
    };
  }
  let height;
  let width;
  let channels;
  if (shape.length === 4) {
    [, height, width, channels] = shape;
  } else if (shape.length === 3) {
    [height, width, channels] = shape;
  } else {
    return {
      ok: false,
      reason: `Input ${tensor.name || `T${tensor.index}`} is rank ${shape.length}; this local image preprocessor expects NHWC rank-3 or rank-4.`,
    };
  }
  if (![1, 3, 4].includes(channels) || width <= 0 || height <= 0) {
    return {
      ok: false,
      reason: `Input shape ${shapeText(shape)} is not a supported image tensor layout.`,
    };
  }
  if (!["FLOAT32", "INT8", "UINT8"].includes(tensor.dtype)) {
    return {
      ok: false,
      reason: `Input dtype ${tensor.dtype} is not supported by the browser image preprocessor yet.`,
    };
  }
  return {
    ok: true,
    tensor,
    dtype: tensor.dtype,
    shape,
    width,
    height,
    channels,
  };
}

export function isImageLikeTensor(tensor) {
  if (!Array.isArray(tensor?.shape)) return false;
  const shape = tensor.shape.map((dim, index) => resolveStaticDim(dim, index));
  const channels = shape.length === 4 ? shape[3] : shape.length === 3 ? shape[2] : 0;
  return (shape.length === 3 || shape.length === 4) && [1, 3, 4].includes(channels);
}

export function resolveStaticDim(dim, index) {
  if (dim > 0) return dim;
  return index === 0 ? 1 : dim;
}

export function parseTriple(value, fallback) {
  const parts = String(value || "")
    .split(",")
    .map((item) => Number(item.trim()));
  if (parts.length !== 3 || parts.some((item) => !Number.isFinite(item))) {
    return fallback;
  }
  return parts;
}

export function quantParams(tensor) {
  return {
    scale: Number(tensor?.scale_sample?.[0] || 1),
    zeroPoint: Number(tensor?.zero_point_sample?.[0] || 0),
  };
}

export function preprocessPresetValues(value) {
  if (value === "unit") return { mean: "0,0,0", std: "1,1,1" };
  if (value === "signed") return { mean: "0.5,0.5,0.5", std: "0.5,0.5,0.5" };
  if (value === "imagenet") return { mean: "0.485,0.456,0.406", std: "0.229,0.224,0.225" };
  return null;
}

export function drawImageToCanvas(ctx, image, targetWidth, targetHeight, mode) {
  ctx.clearRect(0, 0, targetWidth, targetHeight);
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, targetWidth, targetHeight);
  const sourceWidth = image.width;
  const sourceHeight = image.height;
  if (mode === "stretch") {
    ctx.drawImage(image, 0, 0, targetWidth, targetHeight);
    return;
  }
  const scale =
    mode === "cover"
      ? Math.max(targetWidth / sourceWidth, targetHeight / sourceHeight)
      : Math.min(targetWidth / sourceWidth, targetHeight / sourceHeight);
  const drawWidth = sourceWidth * scale;
  const drawHeight = sourceHeight * scale;
  const dx = (targetWidth - drawWidth) / 2;
  const dy = (targetHeight - drawHeight) / 2;
  ctx.drawImage(image, dx, dy, drawWidth, drawHeight);
}

export async function loadBrowserImageBitmap(file, windowLike = globalThis.window) {
  if ("createImageBitmap" in windowLike) {
    return windowLike.createImageBitmap(file);
  }
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.decoding = "async";
    image.src = url;
    await image.decode();
    return image;
  } finally {
    URL.revokeObjectURL(url);
  }
}
