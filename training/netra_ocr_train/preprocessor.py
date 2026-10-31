import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torchvision import transforms
from .config import OCRConfig

class ImagePreprocessor:
    def __init__(self, config: OCRConfig):
        self.cfg = config
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def _chunk_tensor(self, img_tensor: torch.Tensor) -> list[torch.Tensor]:
        _, _, W = img_tensor.shape
        chunks = []
        start = 0
        
        while start < W:
            end = min(start + self.cfg.chunk_width, W)
            chunk = img_tensor[:, :, start:end]

            # (Pad with 1.0 = White)
            if chunk.shape[2] < self.cfg.chunk_width:
                pad_size = self.cfg.chunk_width - chunk.shape[2]
                chunk = F.pad(chunk, (0, pad_size, 0, 0), value=1.0)

            chunks.append(chunk)
            start += self.cfg.chunk_width - self.cfg.chunk_overlap
            
        return chunks

    def process(self, image_source: str | Path | Image.Image) -> torch.Tensor:
        if isinstance(image_source, (str, Path)):
            if not Path(image_source).exists():
                raise FileNotFoundError(f"Image not found: {image_source}")
            image = Image.open(image_source).convert('L')
        elif isinstance(image_source, Image.Image):
            image = image_source.convert('L')
        else:
            raise ValueError("Input must be a path or PIL Image")

        aspect_ratio = image.width / image.height
        new_width = int(self.cfg.img_height * aspect_ratio)
        new_width = max(self.cfg.chunk_width // 2, new_width)
        
        image = image.resize((new_width, self.cfg.img_height), Image.Resampling.BILINEAR)
        img_tensor = self.transform(image)
        
        chunks = self._chunk_tensor(img_tensor)
        
        # Normalization (0.5 mean/std)
        chunks_norm = [(c - 0.5) / 0.5 for c in chunks]
        
        # Return stacked tensor [Batch, Channel, H, W]
        return torch.stack(chunks_norm)