// Model-free unit tests for the browser version:  node test/unit.mjs

import assert from "node:assert/strict";

import { decodeYoloEnd2End, decodeYoloRaw, letterbox, restoreBoxes } from "../src/detector.js";
import { resizeGrayPIL, toGray } from "../src/image.js";
import { resolveSource } from "../src/model.js";
import { chunkLine, cleanText, makeVocab } from "../src/recognizer.js";

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test("grayscale uses PIL's integer weights", () => {
  const gray = toGray({ data: new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30]), width: 4, height: 1, channels: 3 });
  assert.deepEqual([...gray], [76, 150, 29, 18]);
});

test("PIL resize keeps a flat image flat and handles identity", () => {
  const flat = new Uint8Array(20 * 10).fill(200);
  assert.ok(resizeGrayPIL(flat, 20, 10, 7, 48).every((v) => v === 200));
  const ramp = Uint8Array.from({ length: 12 }, (_, i) => i * 20);
  assert.deepEqual([...resizeGrayPIL(ramp, 12, 1, 12, 1)], [...ramp]);
});

test("line chunks: shape, and the last chunk padded white", () => {
  const { data, dims } = chunkLine({ data: new Uint8Array(200 * 24), width: 200, height: 24, channels: 1 },
                                   { height: 48, chunk_width: 100, chunk_overlap: 16, min_width: 50 });
  assert.deepEqual(dims, [5, 1, 48, 100]);
  assert.equal(data[0], -1);
  assert.equal(data[data.length - 1], 1);
});

test("letterbox: rectangle or square, boxes restored", () => {
  const image = { data: new Uint8Array(310 * 1000 * 3), width: 1000, height: 310, channels: 3 };
  const { tensor, info } = letterbox(image, { imgsz: 640, stride: 32 });
  assert.deepEqual(tensor.dims, [1, 3, 224, 640]);
  assert.equal(info.padY, 13);
  const box = [100, 50, 400, 300];
  const scaled = [box[0] * info.gainX + info.padX, box[1] * info.gainY + info.padY,
                  box[2] * info.gainX + info.padX, box[3] * info.gainY + info.padY];
  restoreBoxes([scaled], info)[0].forEach((v, i) => assert.ok(Math.abs(v - box[i]) < 1e-6));
  assert.deepEqual(letterbox(image, { imgsz: 640, stride: 32, auto: false }).tensor.dims, [1, 3, 640, 640]);
});

test("yolo outputs: class-aware NMS and end2end threshold", () => {
  // (1, 4 + 2 classes, 3 anchors): anchors 0 and 1 are the same box, different classes
  const raw = { dims: [1, 6, 3], data: Float32Array.from([5, 5.5, 25, 5, 5.5, 25, 10, 9, 10, 10, 9, 10,
                                                          0.9, 0.1, 0.1, 0.1, 0.8, 0.6]) };
  assert.deepEqual(decodeYoloRaw(raw, { conf: 0.25, iou: 0.5 }).map((d) => d.classId), [0, 1, 1]);
  assert.equal(decodeYoloRaw(raw, { conf: 0.25, iou: 0.5, agnostic: true }).length, 2);
  const e2e = { dims: [1, 2, 6], data: Float32Array.from([0, 0, 5, 5, 0.9, 1, 0, 0, 5, 5, 0.1, 0]) };
  assert.deepEqual(decodeYoloEnd2End(e2e, { conf: 0.25 }).map((d) => d.classId), [1]);
});

test("vocabulary and text cleanup", () => {
  const vocab = makeVocab({ "<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3, "ក": 4, "ា": 5, "<mask>": 6 });
  assert.equal(vocab.decode([1, 4, 5, 3, 6, 4, 2, 4]), "កា  ក");
  assert.equal(cleanText("kkkkk"), "k");
  assert.equal(cleanText("H H H H"), "H");
  assert.equal(cleanText("P<KHMNHEM<<DARAYUT<<<<<<<<<<<<"), "P<KHMNHEM<<DARAYUT<<<<<<<<<<<<");
});

test("hf:// sources map to huggingface.co resolve URLs", () => {
  assert.equal(resolveSource("hf://Netra-Lab/Netra-OCR/lines"), "https://huggingface.co/Netra-Lab/Netra-OCR/resolve/main/lines/");
  assert.equal(resolveSource("hf://Netra-Lab/Netra-OCR/lines@v1"), "https://huggingface.co/Netra-Lab/Netra-OCR/resolve/v1/lines/");
  assert.equal(resolveSource("/models/lines/"), "/models/lines/");
});

let failed = 0;
for (const [name, fn] of tests) {
  try { fn(); console.log(`ok   ${name}`); } catch (e) { failed++; console.log(`FAIL ${name}\n     ${e.message}`); }
}
process.exit(failed ? 1 : 0);
