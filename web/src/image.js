// Images are plain {data, width, height, channels} pixel buffers (uint8,
// row-major, interleaved). ImageData (RGBA) qualifies as-is; browser image
// sources are drawn to a canvas first.

export async function toPixels(source) {
  if (source?.data && source.width && source.height) {
    const channels = source.channels ?? source.data.length / (source.width * source.height);
    return { data: source.data, width: source.width, height: source.height, channels };
  }
  // HTMLImageElement / ImageBitmap / HTMLCanvasElement / OffscreenCanvas / Blob
  const bitmap = source instanceof Blob ? await createImageBitmap(source) : source;
  const width = bitmap.naturalWidth ?? bitmap.width;
  const height = bitmap.naturalHeight ?? bitmap.height;
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d");
  context.drawImage(bitmap, 0, 0);
  const { data } = context.getImageData(0, 0, width, height);
  return { data, width, height, channels: 4 };
}

/** Grayscale exactly like PIL's convert("L"): (R*19595 + G*38470 + B*7471 + 0x8000) >> 16. */
export function toGray({ data, width, height, channels }) {
  const out = new Uint8Array(width * height);
  if (channels === 1) {
    out.set(data.subarray ? data.subarray(0, out.length) : data.slice(0, out.length));
    return out;
  }
  for (let i = 0, p = 0; i < out.length; i++, p += channels) {
    out[i] = (data[p] * 19595 + data[p + 1] * 38470 + data[p + 2] * 7471 + 0x8000) >>> 16;
  }
  return out;
}

export function toRGB({ data, width, height, channels }) {
  if (channels === 3) return data;
  const out = new Uint8Array(width * height * 3);
  for (let i = 0, p = 0, q = 0; i < width * height; i++, p += channels, q += 3) {
    const g = channels === 1;
    out[q] = data[p];
    out[q + 1] = g ? data[p] : data[p + 1];
    out[q + 2] = g ? data[p] : data[p + 2];
  }
  return out;
}

// --- PIL Image.resize(..., BILINEAR) for 8-bit single-channel images -------
// A port of Pillow's Resample.c: separable triangle filter whose support
// widens when downscaling (antialiasing), fixed-point coefficients with 22
// fractional bits, horizontal pass then vertical pass. Bit-exact with PIL.

const PRECISION_BITS = 32 - 8 - 2;
const ONE = 2 ** PRECISION_BITS;

function coefficients(inSize, outSize) {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1.0);
  const support = 1.0 * filterScale;
  const kSize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const weights = new Int32Array(outSize * kSize);
  const k = new Float64Array(kSize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let total = 0;
    for (let x = 0; x < xmax; x++) {
      const w = Math.max(0, 1.0 - Math.abs((x + xmin - center + 0.5) / filterScale));
      k[x] = w;
      total += w;
    }
    for (let x = 0; x < kSize; x++) {
      const w = x < xmax && total !== 0 ? k[x] / total : 0;
      weights[xx * kSize + x] = w < 0 ? Math.trunc(-0.5 + w * ONE) : Math.trunc(0.5 + w * ONE);
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { bounds, weights, kSize };
}

function clip8(value) {
  if (value >= 255 * ONE) return 255;
  if (value <= 0) return 0;
  return Math.floor(value / ONE);
}

export function resizeGrayPIL(gray, width, height, outWidth, outHeight) {
  let src = gray;
  let w = width;
  if (outWidth !== width) {
    const { bounds, weights, kSize } = coefficients(width, outWidth);
    const tmp = new Uint8Array(outWidth * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < outWidth; x++) {
        const xmin = bounds[x * 2], n = bounds[x * 2 + 1];
        let sum = ONE / 2;
        for (let i = 0; i < n; i++) sum += src[y * w + xmin + i] * weights[x * kSize + i];
        tmp[y * outWidth + x] = clip8(sum);
      }
    }
    src = tmp;
    w = outWidth;
  }
  if (outHeight === height) return src;
  const { bounds, weights, kSize } = coefficients(height, outHeight);
  const out = new Uint8Array(outWidth * outHeight);
  for (let y = 0; y < outHeight; y++) {
    const ymin = bounds[y * 2], n = bounds[y * 2 + 1];
    for (let x = 0; x < outWidth; x++) {
      let sum = ONE / 2;
      for (let i = 0; i < n; i++) sum += src[(ymin + i) * w + x] * weights[y * kSize + i];
      out[y * outWidth + x] = clip8(sum);
    }
  }
  return out;
}

/** Bilinear, half-pixel centers, no antialiasing, float64 math, round half up --
 *  the exact arithmetic of netra_ocr/inference/detector.py resize_bilinear. */
export function resizeRGBLinear(rgb, width, height, outWidth, outHeight) {
  const axis = (dst, src) => {
    const lo = new Int32Array(dst), hi = new Int32Array(dst), frac = new Float64Array(dst);
    for (let i = 0; i < dst; i++) {
      const pos = Math.min(Math.max((i + 0.5) * (src / dst) - 0.5, 0), src - 1);
      lo[i] = Math.floor(pos);
      hi[i] = Math.min(lo[i] + 1, src - 1);
      frac[i] = pos - lo[i];
    }
    return { lo, hi, frac };
  };
  const ys = axis(outHeight, height), xs = axis(outWidth, width);
  const out = new Uint8Array(outWidth * outHeight * 3);
  for (let y = 0; y < outHeight; y++) {
    const r0 = ys.lo[y] * width * 3, r1 = ys.hi[y] * width * 3, fy = ys.frac[y];
    for (let x = 0; x < outWidth; x++) {
      const c0 = xs.lo[x] * 3, c1 = xs.hi[x] * 3, fx = xs.frac[x];
      for (let c = 0; c < 3; c++) {
        const top = rgb[r0 + c0 + c] * (1 - fx) + rgb[r0 + c1 + c] * fx;
        const bottom = rgb[r1 + c0 + c] * (1 - fx) + rgb[r1 + c1 + c] * fx;
        out[(y * outWidth + x) * 3 + c] = Math.floor(top * (1 - fy) + bottom * fy + 0.5);
      }
    }
  }
  return out;
}
