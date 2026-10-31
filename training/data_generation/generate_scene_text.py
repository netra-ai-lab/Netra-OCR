import os
import re
import random
import numpy as np
import multiprocessing as mp
from PIL import Image, ImageDraw, ImageFont, features
from datasets import Dataset, Features, Value, Image as HFDImage
from huggingface_hub import login

# ============================================================
# 1. CONFIGURATION
# ============================================================
CONFIG = {
    "paths": {
        "corpus": "/content/drive/MyDrive/CNN-Transformer-OCR/texts/khmer_corpus.txt",
        "fonts": "/content/drive/MyDrive/CNN-Transformer-OCR/fonts/",
        "backgrounds": "/content/drive/MyDrive/CNN-Transformer-OCR/background",
        "output_dir": "/content/drive/MyDrive/CNN-Transformer-OCR/khmer_synthetic_scene_text_dataset",
    },
    "generation": {
        "num_samples": 100_000,
        "batch_size": 2500, # Number of items to process before saving a checkpoint (optional)
        "num_workers": mp.cpu_count(),
        "min_words": 3,
        "max_words": 5,
        "fontsize": {"min": 16, "max": 28},
    },
    "upload": {
        "enabled": True,
        "repo_id": "your username/khmer_scene_text_dataset",
        "hf_token": os.environ.get("HF_TOKEN") 
    }
}

# Layout engine check
LAYOUT_ENGINE = ImageFont.Layout.RAQM if features.check("raqm") else ImageFont.Layout.BASIC

# ============================================================
# 2. GENERATOR CLASS
# ============================================================
class KhmerSceneTextGenerator:
    def __init__(self, words, fonts, backgrounds, config):
        self.words = words
        self.fonts = fonts
        self.backgrounds = backgrounds
        self.cfg = config["generation"]

    def _get_random_background(self, target_w, target_h):
        """Fetches a background crop or generates noise if failed."""
        if self.backgrounds:
            try:
                bg_path = random.choice(self.backgrounds)
                bg = Image.open(bg_path).convert("L") # Ensure Grayscale
                bg_w, bg_h = bg.size

                # Resize if background is smaller than required text area
                if bg_w < target_w or bg_h < target_h:
                    scale = max(target_w / bg_w, target_h / bg_h)
                    new_w, new_h = int(bg_w * scale) + 1, int(bg_h * scale) + 1
                    bg = bg.resize((new_w, new_h))

                # Random crop
                max_x = max(0, bg.width - target_w)
                max_y = max(0, bg.height - target_h)
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                
                return bg.crop((x, y, x + target_w, y + target_h))
            except Exception:
                pass

        # Fallback: Gray noise
        return Image.fromarray(np.random.randint(100, 200, (target_h, target_w), dtype=np.uint8))

    def _get_contrast_color(self, image):
        """Determines if text should be Black or White based on background brightness."""
        # Calculate mean brightness (0-255)
        avg_brightness = np.mean(np.array(image))
        # If background is bright (>127), text is Black (0). Else White (255).
        return 0 if avg_brightness > 127 else 255

    def generate(self):
        """Generates a single image-text pair."""
        # 1. Select Text
        n_words = random.randint(self.cfg["min_words"], self.cfg["max_words"])
        text = " ".join(random.choices(self.words, k=n_words))

        # 2. Select Font
        font_path = random.choice(self.fonts)
        font_size = random.randint(self.cfg['fontsize']['min'], self.cfg['fontsize']['max'])
        try:
            font = ImageFont.truetype(font_path, font_size, layout_engine=LAYOUT_ENGINE)
        except:
            font = ImageFont.truetype(font_path, font_size)

        # 3. Calculate Text Dimensions
        dummy = Image.new("L", (1, 1))
        draw_dummy = ImageDraw.Draw(dummy)
        bbox = draw_dummy.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 4. Prepare Background (with padding)
        pad = 20
        img = self._get_random_background(text_w + pad, text_h + pad)
        
        # 5. Determine Color
        text_color = self._get_contrast_color(img)

        # 6. Draw Text
        draw = ImageDraw.Draw(img)
        offset_x, offset_y = random.randint(0, 5), random.randint(0, 5)
        draw.text(
            (10 + offset_x, 10 + offset_y),
            text,
            fill=text_color,
            font=font
        )

        return {"image": img, "label": text}

# ============================================================
# 3. WORKER FUNCTIONS (Multiprocessing)
# ============================================================
worker_instance = None

def init_worker(words, fonts, bgs, config):
    """Initializes the generator instance in each subprocess."""
    global worker_instance
    worker_instance = KhmerSceneTextGenerator(words, fonts, bgs, config)

def process_batch(batch_idx):
    """Worker function to generate one sample."""
    try:
        return worker_instance.generate()
    except Exception as e:
        # Return None on failure so we can filter it out later
        return None

# ============================================================
# 4. MAIN EXECUTION
# ============================================================
def load_assets(config):
    """Loads fonts, backgrounds, and text corpus."""
    print("Loading assets...")
    
    # Fonts
    font_dir = config['paths']['fonts']
    fonts = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.lower().endswith(".ttf")]
    if not fonts: raise FileNotFoundError(f"No fonts found in {font_dir}")

    # Backgrounds
    bg_dir = config['paths']['backgrounds']
    bgs = []
    if os.path.exists(bg_dir):
        bgs = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png'))]
    
    # Corpus
    with open(config['paths']['corpus'], "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Clean hidden chars
    cleaned_lines = [re.sub(r"[\x00-\x1F\x7F]", "", line) for line in lines]
    all_words = []
    for line in cleaned_lines:
        all_words.extend(line.split())
    
    print(f"Assets Loaded: {len(fonts)} Fonts, {len(bgs)} Backgrounds, {len(all_words)} Words.")
    return all_words, fonts, bgs

def main():
    # 1. Load Assets
    words, fonts, bgs = load_assets(CONFIG)
    
    # 2. Generate Data
    total_samples = CONFIG["generation"]["num_samples"]
    print(f"🚀 Generating {total_samples} samples with {CONFIG['generation']['num_workers']} workers...")

    results = []
    with mp.Pool(
        processes=CONFIG['generation']['num_workers'],
        initializer=init_worker,
        initargs=(words, fonts, bgs, CONFIG)
    ) as pool:
        # Use tqdm to show progress bar
        from tqdm.auto import tqdm
        
        # We pass a range just to trigger the mapping
        iterator = pool.imap_unordered(process_batch, range(total_samples), chunksize=100)
        
        for res in tqdm(iterator, total=total_samples, desc="Generating"):
            if res is not None:
                results.append(res)

    print(f"Generation Complete. Valid samples: {len(results)}")

    # 3. Create Hugging Face Dataset
    print("converting to HF Dataset...")
    features_schema = Features({'image': HFDImage(), 'label': Value('string')})
    
    # Convert list of dicts to Dictionary of lists for efficient conversion
    data_dict = {
        "image": [r["image"] for r in results],
        "label": [r["label"] for r in results]
    }
    
    dataset = Dataset.from_dict(data_dict, features=features_schema)

    # 4. Save Locally
    save_path = CONFIG['paths']['output_dir']
    print(f"Saving dataset to: {save_path}")
    dataset.save_to_disk(save_path)

    # 5. Upload to Hub (Optional)
    if CONFIG["upload"]["enabled"]:
        repo_id = CONFIG["upload"]["repo_id"]
        print(f"Uploading to Hugging Face Hub: {repo_id}")
        
        token = CONFIG["upload"]["hf_token"]
        if not token:
            print("HF_TOKEN not found in config or env. Trying interactive login...")
            login()
        else:
            login(token=token)

        try:
            dataset.push_to_hub(repo_id, private=True)
            print(f"Successfully uploaded to https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()