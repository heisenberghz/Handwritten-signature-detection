"""
src/signature_dataset.py - PyTorch Dataset for preprocessed signature images
"""

import sys
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SPLIT_DIR, IMAGE_SIZE


class SignatureDataset(Dataset):
    """
    PyTorch Dataset for PREPROCESSED signature images.
    
    Loads from data/split/train/, val/, test/
    Images are 16-bit PNGs saved by preprocessing pipeline.
    """

    def __init__(
        self,
        split: str,
        split_dir: Path = SPLIT_DIR,
        transform: Optional[Callable] = None
    ):
        self.split = split
        self.split_path = split_dir / split
        self.transform = transform
        
        self.samples: list = []
        
        for label_name, label in [("genuine", 1), ("forged", 0)]:
            class_dir = self.split_path / label_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Directory not found: {class_dir}")
            
            for img_path in sorted(class_dir.glob("*.png")):
                if img_path.name.endswith("_viz.png"):
                    continue
                self.samples.append((img_path, label))
        
        print(f"✓ {split.upper()}: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, label = self.samples[idx]
        
        # Load 16-bit PNG
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ValueError(f"Could not read: {img_path}")
        
        # Convert 16-bit to float32 [0, 1]
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        
        # Ensure grayscale shape (H, W)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Add channel: (1, H, W)
        img = np.expand_dims(img, axis=0)
        
        image_tensor = torch.from_numpy(img).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, label_tensor

    def get_class_distribution(self) -> dict:
        labels = [label for _, label in self.samples]
        genuine = sum(1 for l in labels if l == 1)
        forged = sum(1 for l in labels if l == 0)
        return {"genuine": genuine, "forged": forged, "total": len(labels)}


# ── Standalone test ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("🔍 Testing SignatureDataset")
    print("=" * 55)
    
    for split_name in ["train", "val", "test"]:
        print(f"\n--- {split_name.upper()} ---")
        dataset = SignatureDataset(split=split_name)
        
        dist = dataset.get_class_distribution()
        print(f"   Distribution: {dist}")
        
        if len(dataset) > 0:
            img, label = dataset[0]
            print(f"   Sample: shape={img.shape}, dtype={img.dtype}")
            print(f"   Label: {label.item()} ({'genuine' if label.item() == 1 else 'forged'})")
            print(f"   Range: [{img.min():.3f}, {img.max():.3f}]")