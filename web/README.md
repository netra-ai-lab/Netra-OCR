# Netra OCR in the browser

The recognizer and detectors of `netra_ocr/inference/`, written in JavaScript on
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) (**WebGPU** or
**WebAssembly**). They read the **same model files** as the Python package
([Netra-Lab/Netra-OCR](https://huggingface.co/Netra-Lab/Netra-OCR)) and give the
same results. Plain ES modules, no build step.

```js
import { loadRecognizer, loadDetector } from "@netra-ai-lab/netra-ocr-web";

const recognizer = await loadRecognizer("hf://Netra-Lab/Netra-OCR/recognition");   // or a folder URL
const { text, confidence } = await recognizer.recognize(imgElement);  // <img>, canvas, ImageBitmap, Blob, ImageData

const lines = await loadDetector("hf://Netra-Lab/Netra-OCR/lines", { device: "wasm" });
const boxes = await lines.detect(imageData);   // [{box: [x1, y1, x2, y2], score, label, classId}]
```

**Options**
- **`device`:** `"auto"` (WebGPU when `navigator.gpu` exists, otherwise WebAssembly), `"webgpu"` or `"wasm"`.
- **`variant`:** forces a precision variant, e.g. `"fp16"`. By default the model's per-device preference applies; the current models prefer fp16 only on CUDA.
- **`readFile(name) → ArrayBuffer`:** reads model files from somewhere other than a URL (tests, user-dropped files).
- **`ort`:** the `onnxruntime-web` module to use. Defaults to the bare `onnxruntime-web` import, so a page without a bundler maps it with an import map (see `test/browser.html`).

Pixel buffers are `{data, width, height, channels}` (uint8, interleaved), so it also works off the main thread and in Node.

## Files

| File | Python counterpart |
| --- | --- |
| `src/recognizer.js`: line chunking, greedy and blockwise decoding, vocabulary, text cleanup | `netra_ocr/inference/recognizer.py` |
| `src/detector.js`: letterbox, YOLO box decoding and NMS | `netra_ocr/inference/detector.py` |
| `src/model.js`: model folders, `hf://` paths, precision variants | `netra_ocr/inference/hub.py` |
| `src/session.js`, `src/image.js`: onnxruntime-web sessions; PIL-exact grayscale and resize | `netra_ocr/inference/session.py` |

A change to the pre/post-processing on one side needs the same change on the other; `test/parity.mjs` catches a mismatch.

## Same results as Python

Pre-processing reproduces the Python arithmetic exactly:
- **Line chunks:** a port of Pillow's fixed-point bilinear resize, with every step rounded to float32.
- **Letterbox:** the same float64 bilinear resize as `resize_bilinear` in `detector.py`.

| Check | WebAssembly (Node) |
| --- | --- |
| Line chunks | 325/325 bit-identical |
| `recognition` | 325/325 identical text, ~64 ms/line |
| `recognition-blockwise` | 325/325 identical text, ~65 ms/line |
| `lines` | 270/270 boxes identical (IoU 1.0), 117 ms/page |
| `layout` | 113/113 boxes identical (IoU 1.0), 1.33 s/page |

On WebGPU (Chrome, RTX 3060) the two decoders take about the same time: greedy 85 ms/line, blockwise 82 ms/line, 100/100 lines identical to Python, and detector boxes match Python's CPU output at IoU > 0.9 for 267/270 line boxes and 111/113 layout boxes (GPU float rounding). **fp16 variants don't work on WebGPU** (onnxruntime-web 1.31), so use the fp32 graphs in the browser; the current models never pick fp16 there automatically.

Image *decoding* is the browser's job. JPEG decoders can differ slightly from Pillow's; PNGs decode identically.

## Notes

- Lines are decoded one at a time (`batch()` loops). Batched decoding is Python-only for now.
- On WebGPU, greedy decoding keeps the KV cache in GPU memory. Blockwise decoding trims the cache on the CPU every round, so it reads the cache back.
- Detectors pad to the full square on WebGPU (fixed shapes) and to the tighter rectangle on WebAssembly, the same as the Python side on CUDA vs CPU.

## Tests

```bash
cd web && npm install
npm test                                                         # model-free unit tests
python test/make_fixtures.py --models M --lines L --pages P --out F   # reference outputs from Python
node test/parity.mjs M F                                         # WebAssembly parity
# browser: serve a folder with web/, M and F, then open
#   web/test/browser.html?models=/M/&fixtures=/F/&device=webgpu
```

`M` is a folder with the four model folders (`recognition/`, `recognition-blockwise/`, `lines/`, `layout/`).
