// Netra OCR in the browser: the recognizer and detectors on onnxruntime-web
// (WebGPU or WebAssembly), reading the same model files as the Python package.
//
//   import { loadRecognizer, loadDetector } from "@netra-ai-lab/netra-ocr-web";
//   const recognizer = await loadRecognizer("hf://Netra-Lab/Netra-OCR/recognition");
//   const { text, confidence } = await recognizer.recognize(imageElement);
//
//   const lines = await loadDetector("hf://Netra-Lab/Netra-OCR/lines", { device: "wasm" });
//   const boxes = await lines.detect(imageData);

import * as defaultOrt from "onnxruntime-web";

import { loadDetector as detector } from "./detector.js";
import { loadRecognizer as recognizer } from "./recognizer.js";

export { resolveSource } from "./model.js";
export { resolveDevice } from "./session.js";

/** See recognizer.js. `options.ort` defaults to the "onnxruntime-web" import. */
export function loadRecognizer(source, options = {}) {
  return recognizer(source, { ort: defaultOrt, ...options });
}

/** See detector.js. `options.ort` defaults to the "onnxruntime-web" import. */
export function loadDetector(source, options = {}) {
  return detector(source, { ort: defaultOrt, ...options });
}
