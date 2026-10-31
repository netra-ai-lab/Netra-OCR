import torch
import torch.nn.functional as F
import logging
from pathlib import Path
from .config import OCRConfig
from .cluster_tokenizer import ClusterTokenizer
from .preprocessor import ImagePreprocessor
from .model.blockwise_decoder import BlockwiseParallelWrapper
from tqdm import tqdm

logger = logging.getLogger(__name__)

class OCRPredictor:
    def __init__(self,
                 model_path: str | Path,
                 tokenizer: ClusterTokenizer,
                 config: OCRConfig,
                 model_class):
        
        self.cfg = config
        self.tokenizer = tokenizer
        self.device = torch.device(self.cfg.device)
        self.preprocessor = ImagePreprocessor(config)

        # Initialize Architecture
        logger.info(f"Init Model: dim={self.cfg.emb_dim}, max_seq={self.cfg.max_seq_len}, decoder={self.cfg.decoder_type}")
        self.model = model_class(
            vocab_size=len(tokenizer),
            pad_idx=tokenizer.pad_idx,
            emb_dim=self.cfg.emb_dim,
            max_global_len=self.cfg.max_seq_len,
            decoder_type=self.cfg.decoder_type,
            block_size=self.cfg.block_size,
        )

        # Load Weights
        self._load_weights(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, path: str | Path):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            logger.warning("Strict load failed. Retrying with strict=False")
            self.model.load_state_dict(state_dict, strict=False)

    def predict(self, image_input, beam_width: int = 3) -> str:
        chunks = self.preprocessor.process(image_input).to(self.device)
        
        with torch.no_grad():

            f = self.model.cnn(chunks)
            
            p_out = self.model.patch(f)
            p = p_out[0] if isinstance(p_out, tuple) else p_out
            p = p.transpose(0, 1).contiguous()
            
            enc_out = self.model.enc(p).transpose(0, 1)

            N, L, D = enc_out.shape
            merged = enc_out.reshape(1, N * L, D)

            B, T, _ = merged.shape
            limit = min(T, self.model.global_pos.size(0))
            pos_emb = self.model.global_pos[:limit, :].unsqueeze(0)
            
            if T > limit: 
                merged = merged[:, :limit, :] + pos_emb
            else: 
                merged = merged + pos_emb

            # BiLSTM Smoothing Check
            if hasattr(self.model, 'context_bilstm'):
                self.model.context_bilstm.flatten_parameters()
                memory, _ = self.model.context_bilstm(merged)
            else:
                memory = merged

            return self._decode(memory, beam_width)

    def _decode(self, memory, beam_width):
        if isinstance(self.model.dec, BlockwiseParallelWrapper):
            # Blockwise decoding is a drop-in accelerated greedy decode (see
            # BlockwiseParallelWrapper's docstring: p1 is always exactly the
            # frozen base decoder's own prediction, so the result is provably
            # identical to plain greedy AR). There's no beam-search variant.
            return self._blockwise_decode(memory)
        if beam_width <= 1:
            return self._greedy_decode(memory)
        return self._beam_search(memory, beam_width)

    def _blockwise_decode(self, memory):
        """Greedy decode with Stern et al. 2018 blockwise-parallel verification.

        Each step proposes ``block_size`` tokens ahead from the current hidden
        state, then verifies them in one forward pass through the frozen base
        decoder (the same one plain greedy decoding would use): the proposal
        is accepted token-by-token up to the first mismatch, and the base
        decoder's own (always-correct) prediction at that point is appended
        too, so at least one new token is produced every iteration -- exactly
        like ordinary greedy decoding, just usually several tokens at a time.
        """
        wrapper: BlockwiseParallelWrapper = self.model.dec
        base = wrapper.base
        block_size = wrapper.block_size

        B, T, _ = memory.shape
        mask = torch.zeros((B, T), dtype=torch.bool, device=self.device)
        generated = [self.tokenizer.sos_idx]

        for _ in range(self.cfg.decode_max_len):
            tgt = torch.LongTensor([generated]).to(self.device)
            block_logits = wrapper(tgt, memory, mask)  # (1, len(generated), block_size, vocab)
            proposals = torch.argmax(block_logits[0, -1], dim=-1).tolist()

            candidate = generated + proposals
            cand_tgt = torch.LongTensor([candidate]).to(self.device)
            p1_logits = base(cand_tgt, memory, mask)  # (1, len(candidate), vocab) -- frozen p1 only

            # t0 - 1 is the position whose prediction is proposals[0]; walk
            # forward while the base decoder's own greedy choice agrees.
            t0 = len(generated)
            accept = 0
            while accept < block_size and torch.argmax(p1_logits[0, t0 - 1 + accept]).item() == proposals[accept]:
                accept += 1
            # The base decoder's prediction at the point verification stopped
            # is guaranteed correct (it's exactly what plain greedy would
            # produce next), so it's always appended too.
            next_forced = torch.argmax(p1_logits[0, t0 - 1 + accept]).item()

            stop = False
            for tok in proposals[:accept] + [next_forced]:
                if tok == self.tokenizer.eos_idx:
                    stop = True
                    break
                generated.append(tok)
            if stop:
                break

        return self.tokenizer.decode(generated)

    def _greedy_decode(self, memory):
        B, T, _ = memory.shape
        mask = torch.zeros((B, T), dtype=torch.bool, device=self.device)
        generated = [self.tokenizer.sos_idx]

        for _ in range(self.cfg.decode_max_len):
            tgt = torch.LongTensor([generated]).to(self.device)
            logits = self.model.dec(tgt, memory, mask)
            next_token = torch.argmax(logits[0, -1, :]).item()
            
            if next_token == self.tokenizer.eos_idx:
                break
            generated.append(next_token)
            
        return self.tokenizer.decode(generated)

    def _beam_search(self, memory, beam_width):
        B, T, D = memory.shape
        memory = memory.expand(beam_width, -1, -1)
        mask = torch.zeros((beam_width, T), dtype=torch.bool, device=self.device)
        
        beams = [(0.0, [self.tokenizer.sos_idx])]
        completed = []

        for _ in range(self.cfg.decode_max_len):
            k_curr = len(beams)
            current_seqs = [b[1] for b in beams]
            tgt = torch.tensor(current_seqs, dtype=torch.long, device=self.device)
            
            logits = self.model.dec(tgt, memory[:k_curr], mask[:k_curr])
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

            candidates = []
            for i in range(k_curr):
                score, seq = beams[i]
                top_probs, top_idxs = log_probs[i].topk(beam_width)
                for k in range(beam_width):
                    candidates.append((score + top_probs[k].item(), seq + [top_idxs[k].item()]))

            candidates.sort(key=lambda x: x[0], reverse=True)
            next_beams = []
            for s, seq in candidates:
                if seq[-1] == self.tokenizer.eos_idx:
                    completed.append((s / len(seq), seq))
                elif len(next_beams) < beam_width:
                    next_beams.append((s, seq))
            
            beams = next_beams
            if not beams: break

        best_seq = sorted(completed, key=lambda x: x[0], reverse=True)[0][1] if completed else beams[0][1]
        return self.tokenizer.decode(best_seq)
    
    def predict_batch(self, image_list: list, beam_width: int = 1, batch_size: int = 8) -> list:
        """
        Processes a list of images in mini-batches with a progress bar.
        """
        if not image_list:
            return []

        all_results = []
        
        # Wrap the range in tqdm for a progress bar
        pbar = tqdm(total=len(image_list), desc="OCR Recognition", unit="line")

        for i in range(0, len(image_list), batch_size):
            mini_batch = image_list[i : i + batch_size]
            
            all_chunks = []
            chunk_counts = []

            # 1. Preprocess mini-batch
            for img in mini_batch:
                chunks = self.preprocessor.process(img)
                all_chunks.append(chunks)
                chunk_counts.append(chunks.size(0))
            
            batch_tensor = torch.cat(all_chunks, dim=0).to(self.device)

            with torch.no_grad():
                # 2. Parallel Visual Extraction
                f = self.model.cnn(batch_tensor)
                p_out = self.model.patch(f)
                p = p_out[0] if isinstance(p_out, tuple) else p_out
                p = p.transpose(0, 1).contiguous()
                enc_out = self.model.enc(p).transpose(0, 1)

                # 3. Decode each image in the mini-batch
                current_idx = 0
                for count in chunk_counts:
                    img_enc = enc_out[current_idx : current_idx + count]
                    current_idx += count

                    N, L, D = img_enc.shape
                    merged = img_enc.reshape(1, N * L, D)

                    limit = min(N * L, self.model.global_pos.size(0))
                    pos_emb = self.model.global_pos[:limit, :].unsqueeze(0)
                    merged = merged[:, :limit, :] + pos_emb if (N * L) > limit else merged + pos_emb

                    if hasattr(self.model, 'context_bilstm'):
                        memory, _ = self.model.context_bilstm(merged)
                    else:
                        memory = merged

                    all_results.append(self._decode(memory, beam_width))
            
            # Update progress bar by the number of images processed in this mini-batch
            pbar.update(len(mini_batch))

        pbar.close()
        return all_results