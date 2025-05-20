import os
import shutil
import random
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = list(Path(image_dir).glob("*.*"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def split_dataset(src_dir, out_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"

    src_dir = Path(src_dir)
    out_dir = Path(out_dir)

    images = list(src_dir.glob("*.*"))
    random.seed(seed)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, img_paths in splits.items():
        split_dir = out_dir / split
        print(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        for img_path in img_paths:
            shutil.copy(img_path, split_dir / img_path.name)
