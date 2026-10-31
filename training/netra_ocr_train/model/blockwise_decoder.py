import torch
import torch.nn as nn


class BlockwiseParallelWrapper(nn.Module):
    """Wraps an already-trained, fully-autoregressive decoder with Stern et
    al. 2018's frozen-base blockwise parallel decoding scheme (see
    research/Blockwise Parallel Decoding for Deep Autoregressive Models.pdf).

    The wrapped ``base`` decoder (self-/cross-attention stack, tok_emb,
    pos_emb, out_proj) is frozen and never modified -- p1, the ordinary
    next-token prediction, is always exactly the base decoder's own
    unchanged output. A small trainable residual feedforward layer predicts
    ``block_size - 1`` further positions ahead (p2..p_block_size) from the
    same per-position hidden state, in parallel (Section 6 / Figure 3 of the
    paper: one feedforward layer with a residual connection to each of the
    new outputs, then the *same* out_proj applied to every output).

    Because p1 is untouched, greedy decoding with exact-match verification
    (see `predictor.py`'s `_blockwise_decode`) is *provably* identical to
    plain greedy AR decoding from the base model -- the auxiliary heads can
    only let decoding skip steps when they're right, never produce a
    different result when they're wrong. This is why training this needs
    far less time than SAT or Mask-Predict: nearly the whole network stays
    frozen (only `proposal_ffn` here has trainable parameters -- see
    `KhmerOCR.__init__`'s `decoder_type == "blockwise"` branch for how the
    rest of the pipeline is frozen too), and there's no risk of a quality
    regression to train around since p1 never changes.

    forward() returns logits of shape (B, T, block_size, vocab_size): index
    0 is p1 (frozen, from the base decoder), indices 1..block_size-1 are the
    trainable auxiliary proposals p2..p_block_size.
    """

    def __init__(self, base_decoder, block_size=4, hidden_mult=1):
        super().__init__()
        self.base = base_decoder
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.block_size = block_size
        self.pad_idx = base_decoder.pad_idx
        self.n_aux = block_size - 1
        emb_dim = base_decoder.tok_emb.embedding_dim
        hidden = emb_dim * hidden_mult

        self.proposal_ffn = nn.Sequential(
            nn.Linear(emb_dim, self.n_aux * hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_aux * hidden, self.n_aux * emb_dim),
        )

    def forward(self, tgt_tokens, memory, memory_key_padding_mask):
        with torch.no_grad():
            hidden = self.base.forward_hidden(tgt_tokens, memory, memory_key_padding_mask)
            p1_logits = self.base.out_proj(hidden)

        B, T, D = hidden.shape
        aux = self.proposal_ffn(hidden).view(B, T, self.n_aux, D)
        aux = aux + hidden.unsqueeze(2)  # residual connection to each auxiliary output
        aux_logits = self.base.out_proj(aux)

        return torch.cat([p1_logits.unsqueeze(2), aux_logits], dim=2)
