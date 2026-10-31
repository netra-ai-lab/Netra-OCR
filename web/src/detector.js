// YOLO-family detectors in the browser (line detector, DocLayout): the same
// model files and arithmetic as netra_ocr/inference/detector.py.

import { resizeRGBLinear, toPixels, toRGB } from "./image.js";
import { openModel } from "./model.js";
import { resolveDevice } from "./session.js";

// Python's round(): halves go to the even neighbour.
function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

function clamp(v, lo, hi) {
  return Math.min(Math.max(v, lo), hi);
}

/** Resize keeping the aspect ratio so the long side is `imgsz`, then pad with
 *  `pad_value` to a multiple of `stride` (`auto`) or to the full square.
 *  Returns {tensor: {data, dims: [1, 3, H, W]}, info} with RGB in [0, 1]. */
export function letterbox(pixels, { imgsz, stride = 32, auto = true, pad_value = 114, scaleup = true }) {
  const { width: w, height: h } = pixels;
  let r = Math.min(imgsz / h, imgsz / w);
  if (!scaleup) r = Math.min(r, 1.0);
  const newW = Math.max(roundHalfEven(w * r), 1), newH = Math.max(roundHalfEven(h * r), 1);
  let dw = imgsz - newW, dh = imgsz - newH;
  if (auto) { dw %= stride; dh %= stride; }
  dw /= 2; dh /= 2;
  const top = roundHalfEven(dh - 0.1), bottom = roundHalfEven(dh + 0.1);
  const left = roundHalfEven(dw - 0.1), right = roundHalfEven(dw + 0.1);

  let rgb = toRGB(pixels);
  if (newW !== w || newH !== h) rgb = resizeRGBLinear(rgb, w, h, newW, newH);
  const outW = newW + left + right, outH = newH + top + bottom;
  const plane = outW * outH;
  const data = new Float32Array(3 * plane).fill(pad_value / 255);
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      const p = (y * newW + x) * 3, q = (y + top) * outW + x + left;
      data[q] = rgb[p] / 255;
      data[plane + q] = rgb[p + 1] / 255;
      data[2 * plane + q] = rgb[p + 2] / 255;
    }
  }
  const info = { origW: w, origH: h, gainX: newW / w, gainY: newH / h, padX: left, padY: top };
  return { tensor: { data, dims: [1, 3, outH, outW] }, info };
}

export function restoreBoxes(boxes, info) {
  return boxes.map(([x1, y1, x2, y2]) => [
    clamp((x1 - info.padX) / info.gainX, 0, info.origW), clamp((y1 - info.padY) / info.gainY, 0, info.origH),
    clamp((x2 - info.padX) / info.gainX, 0, info.origW), clamp((y2 - info.padY) / info.gainY, 0, info.origH),
  ]);
}

function iou(a, b) {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = w * h;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
}

/** Raw YOLO output (1, 4 + classes, anchors) -> [{box, score, classId}] after class-aware NMS. */
export function decodeYoloRaw(output, { conf, iou: iouThreshold = 0.7, max_det = 300, max_nms = 30000,
                                        agnostic = false }) {
  const [, rows, anchors] = output.dims;
  const data = output.data;
  let candidates = [];
  for (let a = 0; a < anchors; a++) {
    let classId = 0, score = data[4 * anchors + a];
    for (let c = 1; c < rows - 4; c++) {
      const s = data[(4 + c) * anchors + a];
      if (s > score) { score = s; classId = c; }
    }
    if (score <= conf) continue;
    const cx = data[a], cy = data[anchors + a], w = data[2 * anchors + a], h = data[3 * anchors + a];
    candidates.push({ box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], score, classId });
  }
  candidates.sort((x, y) => y.score - x.score);   // stable: ties keep anchor order
  candidates = candidates.slice(0, max_nms);
  const kept = [];
  for (const cand of candidates) {
    if (kept.length >= max_det) break;
    if (kept.every((k) => (!agnostic && k.classId !== cand.classId) || iou(k.box, cand.box) <= iouThreshold)) {
      kept.push(cand);
    }
  }
  return kept;
}

/** NMS-free YOLO output (1, max_det, 6) -> [{box, score, classId}] above `conf`. */
export function decodeYoloEnd2End(output, { conf }) {
  const [, rows] = output.dims;
  const data = output.data;
  const found = [];
  for (let r = 0; r < rows; r++) {
    const o = r * 6;
    if (data[o + 4] > conf) {
      found.push({ box: [data[o], data[o + 1], data[o + 2], data[o + 3]], score: data[o + 4], classId: data[o + 5] });
    }
  }
  return found;
}

const HEADS = { "yolo-nms": decodeYoloRaw, "yolo-end2end": decodeYoloEnd2End };

/**
 * Load a detector: image -> [{box: [x1, y1, x2, y2], score, label, classId}].
 * Options as for loadRecognizer. On WebGPU the image is padded to the full
 * square: fixed input shapes avoid per-shape kernel setup.
 */
export async function loadDetector(source, { ort, device = "auto", variant, readFile } = {}) {
  const model = await openModel(source, { readFile });
  device = resolveDevice(device);
  variant = variant !== undefined ? variant : model.pickVariant(device);
  const resize = { ...model.settings.preprocess, auto: (model.settings.preprocess.auto ?? true) && device !== "webgpu" };
  const { type, labels, conf: defaultConf = 0.25, ...head } = model.settings.head;
  const decode = HEADS[type];
  const { model: session } = await model.sessions(ort, device, variant);

  return {
    name: model.name, device, variant, labels,
    async detect(image, { conf } = {}) {
      const { tensor, info } = letterbox(await toPixels(image), resize);
      const out = await session.run({ images: session.tensor("float32", tensor.data, tensor.dims) });
      const found = decode(out.output0, { ...head, conf: conf ?? defaultConf });
      const boxes = restoreBoxes(found.map((d) => d.box), info);
      return found.map((d, i) => ({ box: boxes[i], score: d.score, label: labels[d.classId], classId: d.classId }));
    },
  };
}
