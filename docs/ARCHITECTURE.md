# Architecture

Netra-OCR reads one text line at a time. A text-line image goes in, and a sequence of Khmer character clusters and English characters comes out. This page walks through each module; the README has the overview figure.

<p align="center">
  <img src="../assets/ocr-architecture.png" width="1000" alt="Netra-OCR architecture: chunked text line, SE network, patch embedding, Transformer encoder, merger, BiLSTM, Transformer decoder" />
</p>

## 1. Preprocessing: chunk and merge

To handle variable-length text lines without aggressive resizing, the line is cut into chunks:
- **Resize.** The image is resized to a height of 48 pixels, keeping its aspect ratio.
- **Chunk.** The image is split into overlapping chunks of 48 × 100 px, with 16 px of overlap.

Each chunk is encoded on its own, in parallel, and the chunks are merged again before decoding.

## 2. Squeeze-and-Excitation network

The backbone has 5 VGG-style convolution blocks that extract 64, 128, 256 and 512 channel feature maps. The later stages pool only over height, so the horizontal resolution, where character order lives, is preserved.

After the deeper blocks, a **1D Squeeze-and-Excitation** (SE) module recalibrates the feature maps:
- **Squeeze.** The original SE block squeezes both spatial axes into one descriptor per channel. This variant averages over height only, so it yields one channel descriptor *per horizontal column*.
- **Excitation.** A small bottleneck reduces then restores the channel dimension, with a ReLU and a sigmoid, giving a gate between 0 and 1 for each channel at each column.
- **Scale.** The feature map is multiplied by these gates column by column. Each part of the text line gets its own channel weighting, so the network can suppress noisy or background channels at each position rather than applying one correction to the whole chunk.

The backbone ends with adaptive average pooling, giving a fixed-size feature map per chunk.

<p align="center">
  <img src="../assets/Sequence%20Attention%20CNN.png" width="1000" alt="The Squeeze-and-Excitation network: 5 convolution blocks, SE modules in blocks 3 to 5" />
</p>

<p><em>Figure 1: The Squeeze-and-Excitation network. Blocks 3, 4 and 5 carry an SE module, which squeezes over height only, excites per column and rescales the feature map column by column.</em></p>

## 3. Patch module

The CNN's 2D feature map becomes a sequence of patch embeddings, ViT-style. A small convolutional projection collapses the height to one while keeping the width, and maps each column to the model's embedding dimension (384). A learnable positional embedding is added to each token, so the encoder knows each patch's position within the chunk.

## 4. Transformer encoder

Each chunk's patch sequence goes through a Transformer encoder (2 layers, 8 heads). Self-attention lets patches in the same chunk attend to each other, which resolves local ambiguities such as visually similar sub-consonant stacks. The output is a sequence of "vision tokens" per chunk.

## 5. Merging module

The vision tokens of all chunks from one line are concatenated into one sequence. Sequences in a batch are padded, with a padding mask so the decoder ignores the padding. A second, *global* positional embedding is added across the merged sequence, so the model knows each token's position in the whole line, not only within its chunk.

## 6. BiLSTM context smoother

Encoding chunks independently can leave the seams inconsistent: a character split across two chunks may be represented differently by each. After merging, a single-layer bidirectional LSTM sweeps the full sequence:
- the forward pass carries context from earlier chunks into later ones;
- the backward pass carries context from later chunks back into earlier ones.

The two directions are concatenated at every position, so each token sees both sides of every seam.

<p align="center">
  <img src="../assets/BiLSTM-Module.png" width="1000" alt="The BiLSTM context smoother across merged vision tokens" />
</p>

<p><em>Figure 2: The BiLSTM context smoother. Forward and backward passes over the merged sequence give every token context from both sides of each chunk seam.</em></p>

## 7. Transformer decoder

An autoregressive Transformer decoder (2 layers, 8 heads) generates the output one token at a time:
- target tokens are embedded and given a learned positional embedding;
- a causal mask stops each step from seeing later tokens;
- cross-attention reads the smoothed encoder memory, respecting the padding mask;
- a final linear layer maps to the 2,101-token vocabulary of Khmer character clusters and characters.

**Blockwise decoding.** The optional `blockwise` decoder ([Stern et al., 2018](https://arxiv.org/abs/1811.03115)) adds a small proposal head to the frozen decoder. Each step proposes up to 4 tokens and verifies them against the same decoder, so the output is identical to greedy decoding with fewer decoder passes: on KHOB it accepts 2.7 tokens per step on average ([RESULTS.md](RESULTS.md)).

## 8. Tokenisation

English text is tokenised by character. Khmer text uses a mix of single characters and **character clusters** (a consonant with its subscripts and vowels), which makes Khmer sequences about 43% shorter than one token per code point.
