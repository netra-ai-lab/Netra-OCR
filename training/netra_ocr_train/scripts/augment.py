"""Training-time image augmentation for line-image OCR.

Operates on a single-channel grayscale tensor in the 0-1 range (white
background = 1.0), applied before chunking/normalization -- see the
preprocessing contract in `preprocessor.py`. Written fresh against
torchvision/torch ops, covering the same categories of distortion real
scanned/photographed text exhibits (as opposed to clean synthetic renders):
pixelation, thinning, blur, noise, shift. Deliberately excludes any
rotation/shear/perspective-style geometric distortion, which would corrupt
Khmer glyph shapes rather than just degrade image quality.
"""

import random

import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur


class OCRAugmenter:
    """Callable: tensor[1,H,W] in [0,1] (white=1.0) -> tensor[1,H,W] in [0,1].

    Each transform is applied independently at its own probability, so a
    given call may apply zero, one, or several of them in sequence.
    """

    def __init__(self,
                 p_pixelate: float = 0.25,
                 p_erode: float = 0.25,
                 p_blur: float = 0.35,
                 p_noise: float = 0.35,
                 p_shift: float = 0.3):
        self.p_pixelate = p_pixelate
        self.p_erode = p_erode
        self.p_blur = p_blur
        self.p_noise = p_noise
        self.p_shift = p_shift

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_pixelate:
            img = self._pixelate(img)
        if random.random() < self.p_erode:
            img = self._erode(img)
        if random.random() < self.p_blur:
            img = self._blur(img)
        if random.random() < self.p_shift:
            img = self._shift(img)
        if random.random() < self.p_noise:
            img = self._noise(img)
        return img.clamp(0.0, 1.0)

    def _pixelate(self, img: torch.Tensor) -> torch.Tensor:
        """Downsample then upsample with nearest-neighbor to simulate a
        low-resolution capture."""
        _, h, w = img.shape
        factor = random.uniform(0.3, 0.6)
        small_h, small_w = max(1, int(h * factor)), max(1, int(w * factor))
        small = F.interpolate(img.unsqueeze(0), size=(small_h, small_w), mode="nearest")
        return F.interpolate(small, size=(h, w), mode="nearest").squeeze(0)

    def _erode(self, img: torch.Tensor) -> torch.Tensor:
        """Thins dark ink strokes: ink is low-value on a white(1.0)
        background, so eroding the ink is a max-pool on the inverted image."""
        inverted = 1.0 - img
        eroded = F.max_pool2d(inverted.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
        return 1.0 - eroded

    def _blur(self, img: torch.Tensor) -> torch.Tensor:
        sigma = random.uniform(0.3, 1.2)
        return GaussianBlur(kernel_size=3, sigma=sigma)(img)

    def _noise(self, img: torch.Tensor) -> torch.Tensor:
        sigma = random.uniform(0.02, 0.08)
        return img + torch.randn_like(img) * sigma

    def _shift(self, img: torch.Tensor) -> torch.Tensor:
        """Horizontal shift, refilling the exposed edge with white."""
        _, h, w = img.shape
        max_shift = max(1, int(w * 0.05))
        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return img
        shifted = torch.roll(img, shifts=shift, dims=2)
        if shift > 0:
            shifted[:, :, :shift] = 1.0
        else:
            shifted[:, :, shift:] = 1.0
        return shifted
