"""
src/data_split.py - Writer-independent train/val/test split for CEDAR dataset
"""

import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SPLIT_DIR,
    RAW_DATA_DIR,
    CEDAR_GENUINE_DIR,
    CEDAR_FORGED_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED
)
from src.dataset_loader import CEDARDatasetLoader


class WriterIndependentSplit:
    """
    Splits CEDAR dataset by person (writer-independent).
    
    Training, validation, and test sets contain DIFFERENT people.
    This ensures the model learns to detect forgeries, not memorize writers.
    """

    def __init__(
        self,
        output_dir: Path = SPLIT_DIR,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO,
        seed: int = RANDOM_SEED
    ):
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        self.loader = CEDARDatasetLoader()
        
        # Will hold person IDs for each split
        self.train_persons: List[int] = []
        self.val_persons: List[int] = []
        self.test_persons: List[int] = []
        
        # Statistics
        self.stats: Dict[str, dict] = {}

    def _split_persons(self, all_persons: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """Split person IDs into train/val/test sets."""
        np.random.seed(self.seed)
        shuffled = np.random.permutation(all_persons).tolist()
        
        n = len(shuffled)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        # Test gets remainder to avoid rounding issues
        
        train = shuffled[:n_train]
        val = shuffled[n_train:n_train + n_val]
        test = shuffled[n_train + n_val:]
        
        return train, val, test

    def _copy_samples(self, samples, split_name: str) -> Tuple[int, int]:
        """
        Copy samples to split directory.
        Returns (genuine_count, forged_count).
        """
        split_dir = self.output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        genuine_count = 0
        forged_count = 0
        
        for sample in samples:
            # Create subdirectory: split/genuine/ or split/forged/
            class_dir = split_dir / sample.label_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            dest = class_dir / sample.file_path.name
            shutil.copy2(str(sample.file_path), str(dest))
            
            if sample.label == 1:
                genuine_count += 1
            else:
                forged_count += 1
        
        return genuine_count, forged_count

    def create_split(self) -> Dict[str, dict]:
        """
        Execute the full writer-independent split.
        Returns statistics dictionary.
        """
        print("=" * 55)
        print("📂 Writer-Independent Train/Val/Test Split")
        print("=" * 55)
        
        # Load dataset
        self.loader.discover()
        
        # Get unique person IDs
        all_persons = sorted(self.loader.persons)
        print(f"\nTotal persons found: {len(all_persons)}")
        
        # Split persons
        self.train_persons, self.val_persons, self.test_persons = self._split_persons(all_persons)
        
        print(f"\n🎲 Random split (seed={self.seed}):")
        print(f"   Train persons:   {len(self.train_persons)} → {self.train_persons}")
        print(f"   Val persons:     {len(self.val_persons)} → {self.val_persons}")
        print(f"   Test persons:    {len(self.test_persons)} → {self.test_persons}")
        
        # Verify no overlap
        assert len(set(self.train_persons) & set(self.val_persons)) == 0, "Train/Val overlap!"
        assert len(set(self.train_persons) & set(self.test_persons)) == 0, "Train/Test overlap!"
        assert len(set(self.val_persons) & set(self.test_persons)) == 0, "Val/Test overlap!"
        print("\n✓ No person overlap between splits")
        
        # Filter samples by split
        train_samples = [s for s in self.loader.samples if s.person_id in self.train_persons]
        val_samples = [s for s in self.loader.samples if s.person_id in self.val_persons]
        test_samples = [s for s in self.loader.samples if s.person_id in self.test_persons]
        
        # Copy files
        print(f"\n📥 Copying samples to {self.output_dir}...")
        
        train_g, train_f = self._copy_samples(train_samples, "train")
        val_g, val_f = self._copy_samples(val_samples, "val")
        test_g, test_f = self._copy_samples(test_samples, "test")
        
        # Compile statistics
        self.stats = {
            "train": {
                "persons": self.train_persons,
                "num_persons": len(self.train_persons),
                "genuine": train_g,
                "forged": train_f,
                "total": train_g + train_f
            },
            "val": {
                "persons": self.val_persons,
                "num_persons": len(self.val_persons),
                "genuine": val_g,
                "forged": val_f,
                "total": val_g + val_f
            },
            "test": {
                "persons": self.test_persons,
                "num_persons": len(self.test_persons),
                "genuine": test_g,
                "forged": test_f,
                "total": test_g + test_f
            }
        }
        
        return self.stats

    def print_summary(self):
        """Print split statistics."""
        if not self.stats:
            print("Run .create_split() first!")
            return
        
        print("\n" + "=" * 55)
        print("📊 Split Summary")
        print("=" * 55)
        
        for split_name, split_stats in self.stats.items():
            print(f"\n{split_name.upper()}:")
            print(f"   Persons:  {split_stats['num_persons']} ({split_stats['persons']})")
            print(f"   Genuine:  {split_stats['genuine']}")
            print(f"   Forged:   {split_stats['forged']}")
            print(f"   Total:    {split_stats['total']}")
        
        print("\n" + "=" * 55)
        print("✅ Writer-independent split complete!")
        print(f"   Data saved to: {self.output_dir}")
        print("=" * 55)


# ── Standalone run ──────────────────────────────────────────
if __name__ == "__main__":
    splitter = WriterIndependentSplit()
    splitter.create_split()
    splitter.print_summary()