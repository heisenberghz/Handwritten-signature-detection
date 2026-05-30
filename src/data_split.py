"""
src/data_split.py - Writer-independent train/val/test split from PREPROCESSED images
"""

import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SPLIT_DIR,
    PROCESSED_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED
)


class WriterIndependentSplit:
    """
    Splits PREPROCESSED CEDAR dataset by person (writer-independent).
    
    Training, validation, and test sets contain DIFFERENT people.
    Source: data/processed/genuine/ and data/processed/forged/
    Output: data/split/train/, val/, test/
    """

    def __init__(
        self,
        source_dir: Path = PROCESSED_DIR,
        output_dir: Path = SPLIT_DIR,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO,
        seed: int = RANDOM_SEED
    ):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        self.train_persons: List[int] = []
        self.val_persons: List[int] = []
        self.test_persons: List[int] = []
        
        self.stats: Dict[str, dict] = {}

    def _extract_person_id(self, filename: str) -> int:
        """Extract person ID from preprocessed filename (original_X_Y.png or forgeries_X_Y.png)."""
        # Remove _viz suffix if present
        stem = Path(filename).stem.replace("_viz", "")
        # Parse: original_1_1 or forgeries_1_1
        parts = stem.split("_")
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                pass
        return -1

    def _get_all_samples(self) -> List[Tuple[Path, int, int, str]]:
        """
        Scan processed/ directory and collect all samples.
        Returns list of (file_path, person_id, label, label_name).
        """
        samples = []
        
        for label_name, label in [("genuine", 1), ("forged", 0)]:
            class_dir = self.source_dir / label_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Processed directory not found: {class_dir}")
            
            # Only get non-viz files (the float32 versions)
            for f in sorted(class_dir.glob("*.png")):
                if f.name.endswith("_viz.png"):
                    continue
                
                person_id = self._extract_person_id(f.name)
                if person_id > 0:
                    samples.append((f, person_id, label, label_name))
                else:
                    print(f"   ⚠️  Could not parse person ID from: {f.name}")
        
        return samples

    def _split_persons(self, all_persons: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """Split person IDs into train/val/test sets."""
        np.random.seed(self.seed)
        shuffled = np.random.permutation(all_persons).tolist()
        
        n = len(shuffled)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        
        train = shuffled[:n_train]
        val = shuffled[n_train:n_train + n_val]
        test = shuffled[n_train + n_val:]
        
        return train, val, test

    def _copy_samples(self, samples: List[Tuple], split_name: str) -> Tuple[int, int]:
        """
        Copy samples to split directory.
        Returns (genuine_count, forged_count).
        """
        split_dir = self.output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        genuine_count = 0
        forged_count = 0
        
        for file_path, person_id, label, label_name in samples:
            class_dir = split_dir / label_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            dest = class_dir / file_path.name
            shutil.copy2(str(file_path), str(dest))
            
            if label == 1:
                genuine_count += 1
            else:
                forged_count += 1
        
        return genuine_count, forged_count

    def create_split(self) -> Dict[str, dict]:
        """Execute the full writer-independent split from processed images."""
        print("=" * 55)
        print("📂 Writer-Independent Split (from PREPROCESSED images)")
        print("=" * 55)
        
        # Collect all preprocessed samples
        print(f"\n🔍 Scanning preprocessed data at: {self.source_dir}")
        all_samples = self._get_all_samples()
        
        # Group by person
        person_to_samples: Dict[int, List[Tuple]] = {}
        for sample in all_samples:
            _, person_id, _, _ = sample
            if person_id not in person_to_samples:
                person_to_samples[person_id] = []
            person_to_samples[person_id].append(sample)
        
        all_persons = sorted(person_to_samples.keys())
        print(f"Found {len(all_persons)} persons with {len(all_samples)} total samples")
        
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
        train_samples = [s for s in all_samples if s[1] in self.train_persons]
        val_samples = [s for s in all_samples if s[1] in self.val_persons]
        test_samples = [s for s in all_samples if s[1] in self.test_persons]
        
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