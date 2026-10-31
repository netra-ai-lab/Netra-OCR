// Text-line recognizer in the browser: the same model files and the same
// arithmetic as netra_ocr/inference/recognizer.py, so both give the same text.

import { resizeGrayPIL, toGray, toPixels } from "./image.js";
import { openModel } from "./model.js";
import { release, resolveDevice } from "./session.js";

const f32 = Math.fround;

// ── pre-processing ───────────────────────────────────────────────────────────

/** Line image -> chunks [N, 1, height, chunk_width], rounded to float32 after
 *  every operation like the Python side's numpy float32 math. */
export function chunkLine(pixels, { height, chunk_width, chunk_overlap, min_width,
                                    pad_value = 1.0, mean = 0.5, std = 0.5 }) {
  const width = Math.max(min_width, Math.trunc((height * pixels.width) / pixels.height));
  const gray = resizeGrayPIL(toGray(pixels), pixels.width, pixels.height, width, height);
  const stride = chunk_width - chunk_overlap;
  const count = Math.ceil(width / stride);
  const data = new Float32Array(count * height * chunk_width);
  const padded = f32(f32(f32(pad_value) - f32(mean)) / f32(std));
  for (let c = 0; c < count; c++) {
    const start = c * stride;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < chunk_width; x++) {
        const sx = start + x;
        data[(c * height + y) * chunk_width + x] = sx < width
          ? f32(f32(f32(gray[y * width + sx] / 255) - f32(mean)) / f32(std))
          : padded;
      }
    }
  }
  return { data, dims: [count, 1, height, chunk_width] };
}

// ── tokens -> text ───────────────────────────────────────────────────────────

export function makeVocab(tokenToId, { sos = "<sos>", eos = "<eos>", pad = "<pad>", unk = "<unk>" } = {}) {
  const idToToken = new Map(Object.entries(tokenToId).map(([token, id]) => [id, token]));
  const ids = { sos: tokenToId[sos], eos: tokenToId[eos], pad: tokenToId[pad], unk: tokenToId[unk] };
  return {
    sosId: ids.sos,
    eosId: ids.eos,
    decode(tokens) {
      let text = "";
      for (const id of tokens) {
        if (id === ids.sos || id === ids.pad) continue;
        if (id === ids.eos) break;
        const token = idToToken.get(id) ?? "";
        const special = [...token].length > 1 && token.startsWith("<") && token.endsWith(">");
        text += id === ids.unk || special ? " " : token;
      }
      return text;
    },
  };
}

const NOISE = /[�\x00-\x08\x0b\x0c\x0e-\x1f]/gu;
const REPEAT_CLUSTER = /(\S{2,20}?)\1{1,}/gu;
const REPEAT_TOKEN = /(\S+)(?:\s+\1)+/gu;
const SPACES = /\s{2,}/gu;

function isMrzLine(text) {
  const stripped = [...text.replaceAll(" ", "")];
  if (stripped.length < 10 || !stripped.join("").includes("<<")) return false;
  return stripped.filter((c) => /[A-Z0-9<]/.test(c)).length / stripped.length >= 0.85;
}

/** Remove decoder artifacts (see clean_text in recognizer.py). */
export function cleanText(input, maxRepeat = 3) {
  let text = input.replace(NOISE, "");
  if (isMrzLine(text)) return text.replace(SPACES, " ").trim();
  text = text.replace(new RegExp(`(.)\\1{${maxRepeat},}`, "gu"), "$1");
  for (let i = 0; i < 3; i++) {
    const collapsed = text.replace(REPEAT_CLUSTER, "$1");
    if (collapsed === text) break;
    text = collapsed;
  }
  return text.replace(REPEAT_TOKEN, "$1").replace(SPACES, " ").trim();
}

// ── decoding (one line at a time) ────────────────────────────────────────────

function argmaxRow(data, offset, size) {
  let best = 0;
  for (let i = 1; i < size; i++) if (data[offset + i] > data[offset + best]) best = i;
  return best;
}

function probability(data, offset, size, token) {
  let total = 0;
  const top = data[offset + token];
  for (let i = 0; i < size; i++) total += Math.exp(data[offset + i] - top);
  return 1 / total;
}

function int64(session, ids) {
  return session.tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]);
}

/** Greedy decoding with a KV cache: each step feeds only the newest token. */
export function greedyDecoder({ layers, heads, head_dim, max_len }) {
  return {
    async decode(step, cross, sosId, eosId) {
      const S = cross.cross_k.dims[3];
      const crossMask = step.tensor("float32", new Float32Array(S), [1, S]);
      let pastK = step.tensor("float32", new Float32Array(0), [layers, 1, heads, 0, head_dim]);
      let pastV = pastK;
      const ids = [sosId], probs = [];
      for (let i = 0; i < max_len; i++) {
        const out = await step.run({ tokens: int64(step, [ids.at(-1)]), past_k: pastK, past_v: pastV,
                                     cross_k: cross.cross_k, cross_v: cross.cross_v, cross_mask: crossMask });
        const vocab = out.logits.dims[2];
        const logits = out.logits.data;
        const next = argmaxRow(logits, 0, vocab);
        probs.push(probability(logits, 0, vocab, next));
        release(pastK); release(pastV);
        pastK = out.present_k; pastV = out.present_v;
        if (next === eosId) break;
        ids.push(next);
      }
      release(pastK); release(pastV);
      return { ids, probs };
    },
  };
}

// Keep the first `length` positions (axis 3) of a [layers, 1, heads, P, head_dim] cache.
function trimCache(session, tensor, length) {
  const [L, B, H, P, D] = tensor.dims;
  const out = new Float32Array(L * B * H * length * D);
  for (let block = 0; block < L * B * H; block++) {
    out.set(tensor.data.subarray(block * P * D, block * P * D + length * D), block * length * D);
  }
  return session.tensor("float32", out, [L, B, H, length, D]);
}

/** Blockwise parallel decoding (see BlockwiseDecoder in recognizer.py): same
 *  text as greedy, several tokens per forward pass. */
export function blockwiseDecoder({ layers, heads, head_dim, max_len, block_size }) {
  const stats = { rounds: 0, tokens: 0 };
  return {
    stats,
    async decode(step, cross, sosId, eosId) {
      const S = cross.cross_k.dims[3];
      const crossMask = step.tensor("float32", new Float32Array(S), [1, S]);
      let pastK = step.tensor("float32", new Float32Array(0), [layers, 1, heads, 0, head_dim]);
      let pastV = pastK;
      const ids = [sosId], probs = [];
      let proposals = [];
      while (ids.length - 1 < max_len) {
        proposals = proposals.slice(0, max_len - ids.length);
        const out = await step.run({ tokens: int64(step, [ids.at(-1), ...proposals]), past_k: pastK,
                                     past_v: pastV, cross_k: cross.cross_k, cross_v: cross.cross_v,
                                     cross_mask: crossMask });
        const vocab = out.logits.dims[2];
        const logits = out.logits.data;
        let accepted = 0;
        while (accepted < proposals.length && argmaxRow(logits, accepted * vocab, vocab) === proposals[accepted]) {
          accepted++;
        }
        const fresh = [...proposals.slice(0, accepted), argmaxRow(logits, accepted * vocab, vocab)];
        stats.rounds++;
        const cacheLength = pastK.dims[3] + accepted + 1;
        pastK = trimCache(step, out.present_k, cacheLength);
        pastV = trimCache(step, out.present_v, cacheLength);
        for (let i = 0; i < fresh.length; i++) {
          probs.push(probability(logits, i * vocab, vocab, fresh[i]));
          stats.tokens++;
          if (fresh[i] === eosId) return { ids, probs };
          ids.push(fresh[i]);
          if (ids.length - 1 >= max_len) return { ids, probs };
        }
        const n = block_size - 1;
        proposals = Array.from(out.proposal_ids.data.subarray(accepted * n, accepted * n + n), Number);
      }
      return { ids, probs };
    },
  };
}

// ── the recognizer ───────────────────────────────────────────────────────────

/**
 * Load a recognizer: image of one line -> {text, confidence}.
 * @param source  model folder URL or "hf://Netra-Lab/Netra-OCR/recognition"
 * @param options.ort      the onnxruntime-web module
 * @param options.device   "auto" (WebGPU if available, else WebAssembly), "webgpu" or "wasm"
 * @param options.variant  force a precision variant; default: the config's preference for the device
 * @param options.readFile (name) => ArrayBuffer, to read files from somewhere other than a URL
 */
export async function loadRecognizer(source, { ort, device = "auto", variant, readFile } = {}) {
  const model = await openModel(source, { readFile });
  device = resolveDevice(device);
  variant = variant !== undefined ? variant : model.pickVariant(device);
  const { preprocess, decoder: decoderSettings, tokenizer, postprocess = [] } = model.settings;
  const { type, ...shape } = decoderSettings;
  const decoder = type === "blockwise-kv" ? blockwiseDecoder(shape) : greedyDecoder(shape);
  const { vocab: vocabFile, type: _, ...specials } = tokenizer;
  const vocab = makeVocab(await model.json(vocabFile), specials);
  const cleanup = postprocess.find((p) => p.type === "ocr-text-cleanup");
  // Plain greedy keeps the KV cache in GPU memory between steps; blockwise
  // trims the cache on the CPU every round, so it reads it back anyway.
  const { encoder, decoder_step: step } = await model.sessions(ort, device, variant, {
    encoder: ["cross_k", "cross_v"],
    decoder_step: type === "kv-cache" ? ["present_k", "present_v"] : [],
  });

  async function recognize(image) {
    const { data, dims } = chunkLine(await toPixels(image), preprocess);
    const cross = await encoder.run({ chunks: encoder.tensor("float32", data, dims) });
    const { ids, probs } = await decoder.decode(step, cross, vocab.sosId, vocab.eosId);
    release(cross.cross_k);
    release(cross.cross_v);
    let text = vocab.decode(ids);
    if (cleanup) text = cleanText(text, cleanup.max_repeat ?? 3);
    return { text, confidence: probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : 0 };
  }

  return {
    name: model.name, device, variant, decoder,
    recognize,
    async batch(images) {
      const results = [];
      for (const image of images) results.push(await recognize(image));
      return results;
    },
  };
}
