"""Export a recognition checkpoint as a Netra Runtime bundle.

Produces a folder that ``netra_runtime.load()`` runs on CPU or GPU without
torch or this package:

    netra-model.json     manifest (task, components, file hashes)
    encoder.onnx         chunks -> per-layer cross-attention keys/values
    decoder_step.onnx    one decoding step with a KV cache
    vocab.json

The decoder is re-expressed with an explicit KV cache: the original
``TransformerDecoderWrapper`` re-runs the whole prefix every step, which is
exactly what makes plain greedy decoding O(n^2). Weights are shared with the
loaded model, so the math is the same; ``--check`` confirms it by comparing
the runtime's output against ``OCRPredictor``'s greedy decode on a test set.

Attention is written out as matmul/softmax rather than
``nn.MultiheadAttention`` / ``scaled_dot_product_attention`` so the graphs only
use basic ONNX ops that every ONNX Runtime build (including onnxruntime-web)
supports.

Usage:
    python -m netra_ocr.recognition.scripts.export_runtime_bundle \
        --model netra_ocr/recognition/weight/khmerocr_cluster_ar.pth \
        --out bundles/recognition --check test_set/khob_eval
"""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn

from ..cluster_tokenizer import ClusterTokenizer
from ..config import OCRConfig
from ..model.blockwise_decoder import BlockwiseParallelWrapper
from ..model.model import KhmerOCR
from ..predictor import OCRPredictor
from ..utils import autodetect_config

OPSET = 18
DEFAULT_VOCAB_PATH = Path(__file__).resolve().parents[1] / "char2idx_cluster.json"


def _attention(q, k, v, heads, mask=None):
    """q: (B, Tq, D); k, v: (B, heads, Tk, hd) already split. Returns (B, Tq, D)."""
    B, Tq, D = q.shape
    hd = D // heads
    q = q.view(B, Tq, heads, hd).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    if mask is not None:
        scores = scores + mask
    out = torch.softmax(scores, dim=-1) @ v
    return out.transpose(1, 2).reshape(B, Tq, D)


def _split_heads(x, heads):
    B, T, D = x.shape
    return x.view(B, T, heads, D // heads).transpose(1, 2)


def _in_proj(mha: nn.MultiheadAttention, x, which: int):
    """Apply the q (0), k (1) or v (2) slice of a MultiheadAttention's packed input projection."""
    D = mha.embed_dim
    w = mha.in_proj_weight[which * D:(which + 1) * D]
    b = mha.in_proj_bias[which * D:(which + 1) * D]
    return x @ w.t() + b


def _self_attention_block(mha: nn.MultiheadAttention, x):
    heads = mha.num_heads
    q = _in_proj(mha, x, 0)
    k = _split_heads(_in_proj(mha, x, 1), heads)
    v = _split_heads(_in_proj(mha, x, 2), heads)
    return mha.out_proj(_attention(q, k, v, heads))


class RecognizerEncoder(nn.Module):
    """chunks (N, 1, H, W) -> cross_k, cross_v (layers, 1, heads, S, head_dim).

    Everything up to the decoder's memory (CNN, patch embedding, Transformer
    encoder, chunk merge + global positions, BiLSTM), followed by each decoder
    layer's cross-attention key/value projections -- those depend only on the
    memory, so they're computed once per line instead of once per token.
    """

    def __init__(self, model: KhmerOCR, dec_layers):
        super().__init__()
        self.model = model
        self.dec_layers = dec_layers

    def forward(self, chunks):
        m = self.model
        x, _ = m.patch(m.cnn(chunks))                   # (N, L, D)
        for layer in m.enc.layers:                      # post-norm encoder layers
            x = layer.norm1(x + _self_attention_block(layer.self_attn, x))
            x = layer.norm2(x + layer.linear2(torch.relu(layer.linear1(x))))
        N, L, D = x.shape
        merged = x.reshape(1, N * L, D)[:, :m.global_pos.size(0)]
        merged = merged + m.global_pos[:merged.size(1)].unsqueeze(0)
        memory, _ = m.context_bilstm(merged)

        ks, vs = [], []
        for layer in self.dec_layers:
            mha = layer.multihead_attn
            ks.append(_split_heads(_in_proj(mha, memory, 1), mha.num_heads))
            vs.append(_split_heads(_in_proj(mha, memory, 2), mha.num_heads))
        return torch.stack(ks), torch.stack(vs)


class RecognizerDecoderStep(nn.Module):
    """One decoding step over ``n`` new tokens given a self-attention cache of
    ``P`` positions, for a batch of ``B`` lines (padded encoder outputs + mask)."""

    def __init__(self, dec, proposal_ffn=None):
        super().__init__()
        self.dec = dec
        self.proposal_ffn = proposal_ffn

    def forward(self, tokens, past_k, past_v, cross_k, cross_v, cross_mask):
        dec = self.dec
        P = past_k.size(3)
        n = tokens.size(1)
        positions = torch.arange(n, device=tokens.device) + P
        x = dec.tok_emb(tokens) + dec.pos_emb[positions].unsqueeze(0)

        # Query i (absolute position P + i) may attend to keys 0..P + i.
        key_pos = torch.arange(P + n, device=tokens.device)
        mask = torch.where(key_pos.unsqueeze(0) > positions.unsqueeze(1),
                           torch.tensor(float("-inf")), torch.tensor(0.0))

        # cross_mask (B, S): 0 for real encoder positions, -inf for the padding
        # added when lines of different lengths are decoded as one batch.
        memory_mask = cross_mask[:, None, None, :]

        present_k, present_v = [], []
        for i, layer in enumerate(dec.decoder.layers):   # post-norm decoder layers
            sa = layer.self_attn
            k = torch.cat([past_k[i], _split_heads(_in_proj(sa, x, 1), sa.num_heads)], dim=2)
            v = torch.cat([past_v[i], _split_heads(_in_proj(sa, x, 2), sa.num_heads)], dim=2)
            present_k.append(k)
            present_v.append(v)
            x = layer.norm1(x + sa.out_proj(_attention(_in_proj(sa, x, 0), k, v, sa.num_heads, mask)))

            ca = layer.multihead_attn
            x = layer.norm2(x + ca.out_proj(_attention(_in_proj(ca, x, 0), cross_k[i], cross_v[i], ca.num_heads,
                                                       memory_mask)))
            x = layer.norm3(x + layer.linear2(torch.relu(layer.linear1(x))))

        outputs = (dec.out_proj(x), torch.stack(present_k), torch.stack(present_v))
        if self.proposal_ffn is None:
            return outputs
        # Blockwise proposal heads (BlockwiseParallelWrapper): head j guesses
        # the token j + 2 positions after each input position. Only the ids
        # leave the graph -- the runtime compares them, it never scores them.
        B, n, D = x.shape
        aux = self.proposal_ffn(x).view(B, n, -1, D) + x.unsqueeze(2)
        return outputs + (dec.out_proj(aux).argmax(-1),)


def load_predictor(model_path: str, vocab_path: str, device: str) -> OCRPredictor:
    config = OCRConfig(**autodetect_config(model_path), device=device)
    return OCRPredictor(model_path=model_path, tokenizer=ClusterTokenizer(vocab_path),
                        config=config, model_class=KhmerOCR)


def export(model_path: str, vocab_path: str, out_dir: Path, name: str, version: str):
    predictor = load_predictor(model_path, vocab_path, "cpu")
    model, cfg = predictor.model.eval(), predictor.cfg
    # A blockwise checkpoint's frozen base decoder is a plain AR decoder; its
    # proposal head becomes an extra decoder_step output.
    blockwise = isinstance(model.dec, BlockwiseParallelWrapper)
    dec = model.dec.base if blockwise else model.dec
    dec_layers = dec.decoder.layers
    for layer in list(model.enc.layers) + list(dec_layers):
        assert not layer.norm_first, "exporter assumes post-norm Transformer layers"

    layers, D = len(dec_layers), dec.pos_emb.size(1)
    heads = dec_layers[0].self_attn.num_heads
    head_dim = D // heads

    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = RecognizerEncoder(model, dec_layers).eval()
    step = RecognizerDecoderStep(dec, model.dec.proposal_ffn if blockwise else None).eval()

    chunks = torch.randn(3, 1, cfg.img_height, cfg.chunk_width)
    with torch.no_grad():
        cross_k, cross_v = encoder(chunks)
    S, P = torch.export.Dim("S"), torch.export.Dim("P", max=dec.pos_emb.size(0))
    n = torch.export.Dim("n", max=dec.pos_emb.size(0))

    with torch.no_grad():
        torch.onnx.export(
            encoder, (chunks,), out_dir / "encoder.onnx",
            input_names=["chunks"], output_names=["cross_k", "cross_v"],
            dynamic_shapes={"chunks": {0: torch.export.Dim("chunks")}},
            opset_version=OPSET, dynamo=True, external_data=False,
        )
        B = torch.export.Dim("B")
        past = torch.randn(layers, 2, heads, 4, head_dim)
        cross_k2, cross_v2 = cross_k.repeat(1, 2, 1, 1, 1), cross_v.repeat(1, 2, 1, 1, 1)
        cross_mask = torch.zeros(2, cross_k.size(3))
        torch.onnx.export(
            step, (torch.tensor([[5, 6], [7, 8]]), past, past.clone(), cross_k2, cross_v2, cross_mask),
            out_dir / "decoder_step.onnx",
            input_names=["tokens", "past_k", "past_v", "cross_k", "cross_v", "cross_mask"],
            output_names=["logits", "present_k", "present_v"] + (["proposal_ids"] if blockwise else []),
            dynamic_shapes={"tokens": {0: B, 1: n}, "past_k": {1: B, 3: P}, "past_v": {1: B, 3: P},
                            "cross_k": {1: B, 3: S}, "cross_v": {1: B, 3: S},
                            "cross_mask": {0: B, 1: S}},
            opset_version=OPSET, dynamo=True, external_data=False,
        )

    shutil.copyfile(vocab_path, out_dir / "vocab.json")
    from netra_runtime.bundle import write_manifest
    write_manifest(out_dir, {
        "name": name,
        "version": version,
        "task": "text-recognition",
        "graphs": {"encoder": "encoder.onnx", "decoder_step": "decoder_step.onnx"},
        "components": {
            "preprocess": {"type": "line-chunker", "height": cfg.img_height,
                           "chunk_width": cfg.chunk_width, "chunk_overlap": cfg.chunk_overlap,
                           "min_width": cfg.chunk_width // 2, "pad_value": 1.0,
                           "mean": 0.5, "std": 0.5},
            "decoder": {"type": "blockwise-kv" if blockwise else "kv-cache", "layers": layers,
                        "heads": heads, "head_dim": head_dim, "max_len": cfg.decode_max_len,
                        **({"block_size": model.dec.block_size} if blockwise else {})},
            "tokenizer": {"type": "vocab", "vocab": "vocab.json"},
            "postprocess": [{"type": "ocr-text-cleanup", "max_repeat": 3}],
        },
        "metadata": {"source_checkpoint": Path(model_path).name, "exported_by": "netra-ocr",
                     "license": "MIT"},
    })


def check(model_path: str, vocab_path: str, bundle_dir: Path, test_dir: Path, devices: list[str],
          batch_size: int = 16):
    """Runtime output (one line at a time, and batched) vs. the PyTorch greedy decoder."""
    import netra_runtime
    from PIL import Image

    images = [Image.open(p) for p in sorted((test_dir / "images").iterdir())]
    for device in devices:
        recognizer = netra_runtime.load(bundle_dir, device=device)
        torch_predictor = load_predictor(model_path, vocab_path, "cpu" if device == "cpu" else "cuda")
        for warm in images[:3]:
            torch_predictor.predict(warm, beam_width=1)
            recognizer(warm)

        start = time.perf_counter()
        expected = [torch_predictor.predict(image, beam_width=1) for image in images]
        t_torch = time.perf_counter() - start
        start = time.perf_counter()
        single = [recognizer(image).text for image in images]
        t_single = time.perf_counter() - start
        start = time.perf_counter()
        batched = [r.text for r in recognizer.batch(images, batch_size=batch_size)]
        t_batch = time.perf_counter() - start

        n = len(images)
        print(f"[{recognizer}] identical to torch greedy: single {sum(a == b for a, b in zip(single, expected))}/{n}, "
              f"batch={batch_size} {sum(a == b for a, b in zip(batched, expected))}/{n} | ms/line: "
              f"torch {1000 * t_torch / n:.1f}, single {1000 * t_single / n:.1f}, batch {1000 * t_batch / n:.1f}")
        for name, want, got in [(i, e, b) for i, (e, b) in enumerate(zip(expected, batched)) if e != b][:5]:
            print(f"   line {name}\n     torch: {want}\n     batch: {got}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Recognition checkpoint (.pth)")
    parser.add_argument("--vocab", default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--out", required=True, type=Path, help="Bundle output directory")
    parser.add_argument("--name", default="netra-ocr-recognition")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--check", type=Path, help="Test set ({images/,labels/}) for a parity check")
    parser.add_argument("--devices", nargs="+", default=["cpu"], help="Devices for --check")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for --check")
    args = parser.parse_args()

    export(args.model, args.vocab, args.out, args.name, args.version)
    print(f"Bundle written to {args.out}: " + json.dumps(
        {p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in sorted(args.out.iterdir())}))
    if args.check:
        check(args.model, args.vocab, args.out, args.check, args.devices, args.batch_size)


if __name__ == "__main__":
    main()
