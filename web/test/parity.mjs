// Compare the browser version (onnxruntime-web, WebAssembly, run in Node)
// against the Python side's reference outputs from make_fixtures.py.
//
//   node test/parity.mjs MODELS_DIR FIXTURES_DIR

import fs from "node:fs";
import path from "node:path";

import { loadDetector, loadRecognizer } from "../src/index.js";
import { chunkLine } from "../src/recognizer.js";

const [modelsDir, fixturesDir] = process.argv.slice(2);
const fixtures = JSON.parse(fs.readFileSync(path.join(fixturesDir, "fixtures.json"), "utf8"));
const readFrom = (dir) => async (name) => fs.readFileSync(path.join(dir, name));
const pixels = (entry) => ({
  data: new Uint8Array(fs.readFileSync(path.join(fixturesDir, entry.file))),
  width: entry.width, height: entry.height, channels: 3,
});
let failed = false;
const report = (ok, message) => { console.log(`${ok ? "PASS" : "FAIL"} ${message}`); failed ||= !ok; };

// 1. line chunks: bit-identical to Python (PIL resize + float32 math)
{
  const config = JSON.parse(fs.readFileSync(path.join(modelsDir, "recognition", "netra-model.json")));
  let identical = 0;
  for (const [i, entry] of fixtures.lines.entries()) {
    const want = new Float32Array(new Uint8Array(fs.readFileSync(path.join(fixturesDir, `line_${i}.chunks`))).buffer);
    const got = chunkLine(pixels(entry), config.components.preprocess);
    if (got.data.length === want.length && got.data.every((v, j) => v === want[j])) identical++;
  }
  report(identical === fixtures.lines.length, `line chunks: ${identical}/${fixtures.lines.length} inputs bit-identical`);
}

// 2. recognition: identical text, matching confidence
for (const name of ["recognition", "recognition-blockwise"]) {
  const dir = path.join(modelsDir, name);
  if (!fs.existsSync(dir)) continue;
  const recognizer = await loadRecognizer(name, { device: "wasm", readFile: readFrom(dir) });
  let same = 0, worst = 0;
  const start = performance.now();
  for (const entry of fixtures.lines) {
    const want = entry[name];
    const got = await recognizer.recognize(pixels(entry));
    if (got.text === want.text) same++;
    else if (same + 3 > fixtures.lines.length) console.log(`   ${want.text}\n   ${got.text}`);
    worst = Math.max(worst, Math.abs(got.confidence - want.confidence));
  }
  const ms = (performance.now() - start) / fixtures.lines.length;
  const blocks = recognizer.decoder.stats ? `, ${(recognizer.decoder.stats.tokens / recognizer.decoder.stats.rounds).toFixed(3)} tokens/pass` : "";
  report(same === fixtures.lines.length && worst < 1e-3,
    `${name}: ${same}/${fixtures.lines.length} identical text, max confidence diff ${worst.toExponential(1)}, ${ms.toFixed(1)} ms/line${blocks}`);
}

// 3. detection: every Python box found again (same class, IoU > 0.98), same count
const iou = (a, b) => {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  return (w * h) / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h);
};
for (const name of ["lines", "layout"]) {
  const dir = path.join(modelsDir, name);
  if (!fs.existsSync(dir)) continue;
  const detector = await loadDetector(name, { device: "wasm", readFile: readFrom(dir) });
  let want = 0, got = 0, matched = 0, worstIou = 1;
  const start = performance.now();
  for (const entry of fixtures.pages) {
    const found = await detector.detect(pixels(entry));
    want += entry[name].length;
    got += found.length;
    for (const ref of entry[name]) {
      const best = Math.max(0, ...found.filter((d) => d.classId === ref.classId).map((d) => iou(d.box, ref.box)));
      if (best > 0.98) matched++;
      worstIou = Math.min(worstIou, best);
    }
  }
  const ms = (performance.now() - start) / fixtures.pages.length;
  report(matched === want && got === want,
    `${name}: boxes python=${want} js=${got} matched=${matched}, worst IoU ${worstIou.toFixed(4)}, ${ms.toFixed(0)} ms/page`);
}

process.exit(failed ? 1 : 0);
