import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .blockwise_decoder import BlockwiseParallelWrapper

# ==========================================
# Squeeze-and-Excitation Block (1D-SE)
# ==========================================
class SequenceSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SequenceSE, self).__init__()
        # We process each column (W) independently
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, h, w = x.size()

        y = torch.mean(x, dim=2).view(b, c, w)

        y = self.fc(y)

        y = y.view(b, c, 1, w)

        # 4. Scale
        return x * y

# ==========================================
# VGG Backbone
# ==========================================
class ImprovedFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True))

        self.se3 = SequenceSE(256)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv5 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True))
        self.conv6 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True))
        self.se4 = SequenceSE(512)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU(True)
        self.se5 = SequenceSE(512)

        self.final_pool = nn.AdaptiveAvgPool2d((2, 32))

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))

        x = self.conv4(self.conv3(x))
        x = self.se3(x)
        x = self.pool3(x)

        x = self.conv6(self.conv5(x))
        x = self.se4(x)
        x = self.pool4(x)

        x = self.relu7(self.bn7(self.conv7(x)))
        x = self.se5(x) 

        x = self.final_pool(x)
        return x

class PatchEncoder(nn.Module):
    def __init__(self, in_channels, emb_dim, k1=2, k2=1, max_patches=256):
        """
        Implements PATCHENC(k1, k2).
        """
        super().__init__()

        self.k1 = k1
        self.k2 = k2

        # Linear projection via Conv2d (ViT-style)
        self.proj = nn.Conv2d(
            in_channels,
            emb_dim,
            kernel_size=(k1, k2),
            stride=(k1, k2)
        )

        self.pos_emb = nn.Parameter(
            torch.zeros(max_patches, emb_dim)
        )
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, F):
        """
        F: (B, C, H', W')
        """
        x = self.proj(F)  # (B, D, H'/k1, W'/k2)

        B, D, Hp, Wp = x.shape
        N = Hp * Wp

        x = x.flatten(2).transpose(1, 2)

        x = x + self.pos_emb[:N].unsqueeze(0)

        return x, N

def make_encoder(emb_dim=384, nhead=8, num_layers=3, dim_feedforward=1024, dropout=0.1):

    enc_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead,
                                           dim_feedforward=dim_feedforward,
                                           dropout=dropout, activation='relu')
    
    encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
    return encoder
class Merger(nn.Module):
    def __init__(self, emb_dim, max_total_len=4096):

        super().__init__()

        self.global_pos = nn.Parameter(torch.zeros(max_total_len, emb_dim))
        nn.init.trunc_normal_(self.global_pos, std=0.02)

    def merge_and_pad(self, enc_list):
        """
        enc_list: list of tensors (B,), where each tensor is (L_i, D)
        Returns:
          enc_batch: (B, Lmax, D)
          mask: (B, Lmax) Boolean (True=Padding)
        """

        device = enc_list[0].device
        B = len(enc_list)
        D = enc_list[0].size(1)

        lengths = [t.size(0) for t in enc_list]
        Lmax = max(lengths)

        enc_batch = torch.zeros(B, Lmax, D, device=device)
        mask = torch.ones(B, Lmax, dtype=torch.bool, device=device)

        for i, t in enumerate(enc_list):
            L_i = t.size(0)
            enc_batch[i, :L_i, :] = t
            mask[i, :L_i] = False

        enc_batch = enc_batch + self.global_pos[:Lmax, :].unsqueeze(0)

        return enc_batch, mask
    
class TransformerDecoderWrapper(nn.Module):
    def __init__(self, vocab_size, emb_dim, nhead=8, num_layers=3, pad_idx=0, max_len=512):

        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        dec_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=emb_dim*4, dropout=0.1)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.pos_emb = nn.Parameter(torch.zeros(max_len, emb_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.1)
        self.out_proj = nn.Linear(emb_dim, vocab_size)
        self.pad_idx = pad_idx

    def generate_square_subsequent_mask(self, sz):

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward_hidden(self, tgt_tokens, memory, memory_key_padding_mask):
        """Decoder stack output before the final vocab projection -- shape
        (B, T, emb_dim). Exposed separately from ``forward()`` so
        ``BlockwiseParallelWrapper`` can share this same hidden state between
        the frozen p1 projection and its own auxiliary proposal heads.
        """
        B, T = tgt_tokens.size()
        device = tgt_tokens.device

        tok = self.tok_emb(tgt_tokens)
        pos = self.pos_emb[:T,:].unsqueeze(0).expand(B,-1,-1)
        tgt = (tok + pos).transpose(0,1)

        tgt_key_padding_mask = (tgt_tokens == self.pad_idx)

        if memory_key_padding_mask is not None:
             memory_key_padding_mask = memory_key_padding_mask.bool()

        tgt_mask = self.generate_square_subsequent_mask(T).to(device)

        mem = memory.transpose(0,1)

        dec_out = self.decoder(
            tgt,
            mem,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        return dec_out.transpose(0,1)

    def forward(self, tgt_tokens, memory, memory_key_padding_mask):
        hidden = self.forward_hidden(tgt_tokens, memory, memory_key_padding_mask)
        return self.out_proj(hidden)

class KhmerOCR(nn.Module):
    def __init__(self, vocab_size, pad_idx=0, emb_dim=256, max_global_len=4096,
                 decoder_type="ar", block_size=4):

        super().__init__()

        self.cnn = ImprovedFeatureExtractor()

        self.patch = PatchEncoder(512, emb_dim=emb_dim, k1=2, k2=1)

        self.enc = make_encoder(emb_dim=emb_dim, nhead=8, num_layers=2)

        self.global_pos = nn.Parameter(torch.zeros(max_global_len, emb_dim))
        nn.init.trunc_normal_(self.global_pos, std=0.02)

        # ==========================================
        # BiLSTM Smoothing Layer
        # ==========================================
        # Helps stitch chunks together smoothly
        self.context_bilstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=emb_dim // 2, # Output dim will be (emb_dim//2) * 2 = emb_dim
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        base_dec = TransformerDecoderWrapper(vocab_size, emb_dim=emb_dim, nhead=8,
                                             num_layers=2, pad_idx=pad_idx)
        if decoder_type == "blockwise":
            self.dec = BlockwiseParallelWrapper(base_dec, block_size=block_size)
        else:
            self.dec = base_dec
        self.decoder_type = decoder_type
        self.pad_idx = pad_idx

    def forward(self, chunk_lists, tgt_tokens):

        chunk_sizes = [len(c) for c in chunk_lists]
        flat_input_list = [chunk for img_chunks in chunk_lists for chunk in img_chunks]
        flat_input = torch.stack(flat_input_list)

        f = self.cnn(flat_input)

        p, _ = self.patch(f)
        p = p.transpose(0, 1).contiguous()
        enc_out = self.enc(p)
        enc_out = enc_out.transpose(0, 1)

        batch_encoded_list = []
        cursor = 0
        feature_dim = enc_out.size(-1)

        for size in chunk_sizes:
            img_chunks = enc_out[cursor : cursor + size]
            merged_seq = img_chunks.reshape(-1, feature_dim)
            batch_encoded_list.append(merged_seq)
            cursor += size

        memory = pad_sequence(batch_encoded_list, batch_first=True, padding_value=0.0)

        B, T, _ = memory.shape
        limit = min(T, self.global_pos.size(0))
        pos_emb = self.global_pos[:limit, :].unsqueeze(0)

        if T > self.global_pos.size(0):
             memory = memory[:, :limit, :] + pos_emb
             T = limit
        else:
             memory = memory + pos_emb

        # ==========================================
        # Apply BiLSTM Smoothing
        # ==========================================
        self.context_bilstm.flatten_parameters()
        memory, _ = self.context_bilstm(memory)

        # Create Mask & Decode
        memory_key_padding_mask = torch.ones((B, T), dtype=torch.bool, device=memory.device)
        for i, seq in enumerate(batch_encoded_list):
            valid_len = min(seq.shape[0], T)
            memory_key_padding_mask[i, :valid_len] = False

        logits = self.dec(tgt_tokens, memory, memory_key_padding_mask)

        return logits