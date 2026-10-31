import os
import random
import re
import numpy as np
import cv2
import multiprocessing as mp
from PIL import Image, ImageFont, ImageDraw, features
from datasets import Dataset, Features, Value, Image as HFDImage
from huggingface_hub import login

# ============================================================
# 1. CONFIGURATION
# ============================================================
CONFIG = {
    "paths": {
        "corpus": "/content/drive/MyDrive/CNN-Transformer-OCR/texts/khmer_corpus.txt",
        "fonts": "/content/drive/MyDrive/CNN-Transformer-OCR/fonts",
        "output_dir": "/content/drive/MyDrive/CNN-Transformer-OCR/khmer_synthetic_dataset",
    },
    "generation": {
        "num_samples": 100_000,
        "num_workers": mp.cpu_count(),
        "image_size": None, # Dynamic size based on text
        "font_size": 14,
        "min_words": 3,
        "max_words": 5,
        "augment": False, # Set to False for clean text
    },
    "upload": {
        "enabled": True,
        "repo_id": "your username/khmer_document_text_dataset",
        "hf_token": os.environ.get("HF_TOKEN") # Set env var or use huggingface-cli login
    }
}

# Check Text Rendering Engine
LAYOUT_ENGINE = ImageFont.Layout.RAQM if features.check("raqm") else ImageFont.Layout.BASIC

# ============================================================
# 2. AUGMENTATION UTILS
# ============================================================
class Augmentor:
    @staticmethod
    def add_noise(img):
        """Adds salt-and-pepper style noise blobs."""
        if random.random() > 0.5: return img
        h, w = img.shape[:2]
        num_blobs = random.randint(1, 5)
        for _ in range(num_blobs):
            x, y = random.randint(0, w-1), random.randint(0, h-1)
            cv2.circle(img, (x, y), random.randint(1, 2), 0, -1) # Black dots
        return img

    @staticmethod
    def blur(img):
        """Applies slight Gaussian blur."""
        if random.random() > 0.3: return img
        return cv2.GaussianBlur(img, (3, 3), 0)

    @staticmethod
    def rotate(img):
        """Slight rotation (-3 to +3 degrees)."""
        if random.random() > 0.5: return img
        h, w = img.shape[:2]
        angle = random.uniform(-2.5, 2.5)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(img, M, (w, h), borderValue=255) # White border

    @staticmethod
    def apply_all(pil_img):
        """Converts PIL to CV2, applies effects, converts back."""
        # Convert PIL (RGB) to CV2 (Grayscale for processing)
        img_np = np.array(pil_img.convert("L"))
        
        # Apply chain
        img_np = Augmentor.add_noise(img_np)
        img_np = Augmentor.blur(img_np)
        img_np = Augmentor.rotate(img_np)
        
        # Convert back to RGB
        return cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

# ============================================================
# 3. GENERATOR LOGIC
# ============================================================
class DocumentTextGenerator:
    def __init__(self, words, fonts, config):
        self.words = words
        self.fonts = fonts
        self.cfg = config["generation"]

    def generate(self):
        """Generates a single image sample."""
        try:
            # 1. Select Text
            n_words = random.randint(self.cfg["min_words"], self.cfg["max_words"])
            text = " ".join(random.choices(self.words, k=n_words))

            # 2. Select Font
            font_path = random.choice(self.fonts)
            try:
                font = ImageFont.truetype(font_path, self.cfg["font_size"], layout_engine=LAYOUT_ENGINE)
            except:
                font = ImageFont.truetype(font_path, self.cfg["font_size"])

            # 3. Determine Size
            dummy = Image.new("L", (1, 1))
            draw_d = ImageDraw.Draw(dummy)
            bbox = draw_d.textbbox((0, 0), text, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            # 4. Draw (Black text on White BG)
            # Add padding
            w, h = text_w + 10, text_h + 10
            img = Image.new("RGB", (w, h), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.text((5, 5), text, fill=(0, 0, 0), font=font)

            # 5. Augment
            if self.cfg["augment"]:
                img_np = Augmentor.apply_all(img)
                img = Image.fromarray(img_np)

            return {"image": img, "label": text}

        except Exception:
            return None

# ============================================================
# 4. MULTIPROCESSING WORKERS
# ============================================================
worker_gen = None

def init_worker(words, fonts, config):
    global worker_gen
    worker_gen = DocumentTextGenerator(words, fonts, config)

def process_batch(_):
    return worker_gen.generate()

# ============================================================
# 5. MAIN EXECUTION
# ============================================================
def load_assets(config):
    print("⏳ Loading assets...")
    
    # Load Fonts
    f_dir = config["paths"]["fonts"]
    fonts = [os.path.join(f_dir, f) for f in os.listdir(f_dir) if f.lower().endswith(".ttf")]
    if not fonts: raise FileNotFoundError(f"No fonts in {f_dir}")

    # Load Corpus
    c_path = config["paths"]["corpus"]
    with open(c_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    
    cleaned_words = []
    for line in raw_lines:
        # Clean control characters
        clean = re.sub(r"[\x00-\x1F\x7F]", "", line).strip()
        # Normalize quotes
        clean = clean.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        if clean:
            cleaned_words.extend(clean.split())
            
    print(f"Loaded {len(cleaned_words):,} words and {len(fonts)} fonts.")
    return cleaned_words, fonts

def main():
    # 1. Setup
    words, fonts = load_assets(CONFIG)
    num_samples = CONFIG["generation"]["num_samples"]
    
    # 2. Multiprocess Generation
    print(f"Generating {num_samples} samples with {CONFIG['generation']['num_workers']} workers...")
    
    results = []
    with mp.Pool(
        processes=CONFIG["generation"]["num_workers"], 
        initializer=init_worker, 
        initargs=(words, fonts, CONFIG)
    ) as pool:
        from tqdm.auto import tqdm
        
        # We use a simple range to trigger the workers
        iterator = pool.imap_unordered(process_batch, range(num_samples), chunksize=250)
        
        for res in tqdm(iterator, total=num_samples, desc="Generating"):
            if res:
                results.append(res)

    print(f"Generated {len(results)} valid samples.")

    # 3. Save Dataset
    save_path = CONFIG["paths"]["output_dir"]
    print(f"Converting to Dataset and saving to: {save_path}")
    
    # Optimization: Dict of lists is faster to convert than list of dicts
    data_dict = {
        "image": [r["image"] for r in results],
        "label": [r["label"] for r in results]
    }
    
    features_schema = Features({'image': HFDImage(), 'label': Value('string')})
    dataset = Dataset.from_dict(data_dict, features=features_schema)
    dataset.save_to_disk(save_path)

    # 4. Upload
    if CONFIG["upload"]["enabled"]:
        repo_id = CONFIG["upload"]["repo_id"]
        print(f"Uploading to {repo_id}...")
        
        token = CONFIG["upload"]["hf_token"]
        if not token:
            print("HF_TOKEN not set. Attempting interactive login...")
            login()
        else:
            login(token=token)
            
        try:
            dataset.push_to_hub(repo_id, private=False)
            print(f"Success! https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()