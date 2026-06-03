"""
src/dataset_loader.py - Discovers and loads CEDAR signature dataset
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RAW_DATA_DIR,
    CEDAR_GENUINE_DIR,
    CEDAR_FORGED_DIR,
    GENUINE_PREFIX,
    FORGED_PREFIX,
    NUM_PERSONS,
    SAMPLES_PER_PERSON
)


@dataclass
class SignatureSample:
    """Represents a single signature image with metadata."""
    file_path: Path
    person_id: int          # 1 to 55
    sample_num: int         # 1 to 24
    label: int              # 1 = genuine, 0 = forged
    label_name: str         # "genuine" or "forged"
    
    def __repr__(self):
        return (f"SignatureSample(person={self.person_id:02d}, sample={self.sample_num:02d}, "
                f"{self.label_name}, {self.file_path.name})")


class CEDARDatasetLoader:
    """
    Loads and organizes the CEDAR signature dataset.
    
    Expected structure:
        data/raw/CEDAR/
        ├── full_org/
        │   ├── original_1_1.png
        │   ├── original_1_2.png
        │   └── ...
        └── full_forg/
            ├── forgeries_1_1.png
            ├── forgeries_1_2.png
            └── ...
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or RAW_DATA_DIR
        self.samples: List[SignatureSample] = []
        self.persons: set = set()
        self.df: Optional[pd.DataFrame] = None
        
    def _parse_filename(self, filename: str, label: int, label_name: str) -> Optional[SignatureSample]:
        """
        Parse CEDAR filename to extract person_id and sample_num.
        
        Patterns:
            original_1_1.png   -> person=1, sample=1
            forgeries_1_1.png  -> person=1, sample=1
        """
        # Remove extension
        stem = Path(filename).stem
        
        # Match pattern: prefix_person_sample
        # e.g., original_1_1 or forgeries_12_24
        prefix = GENUINE_PREFIX if label_name == 'genuine' else FORGED_PREFIX
        pattern = rf"^{re.escape(prefix)}_(\d+)_(\d+)$"
        match = re.match(pattern, stem)
        
        if match:
            person_id = int(match.group(1))
            sample_num = int(match.group(2))
            
            sub_dir = CEDAR_GENUINE_DIR if label_name == 'genuine' else CEDAR_FORGED_DIR
            return SignatureSample(
                file_path=self.data_dir / sub_dir / filename,
                person_id=person_id,
                sample_num=sample_num,
                label=label,
                label_name=label_name
            )
        
        return None
    
    def discover(self) -> "CEDARDatasetLoader":
        """Scan the data directory and collect all signature files."""
        print(f"🔍 Scanning dataset at: {self.data_dir}")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # ── Load Genuine Signatures ───────────────────────────
        genuine_dir = self.data_dir / CEDAR_GENUINE_DIR
        if genuine_dir.exists():
            genuine_files = sorted(genuine_dir.glob("*.png"))
            print(f"   Found {len(genuine_files)} files in '{CEDAR_GENUINE_DIR}/'")
            
            for f in genuine_files:
                sample = self._parse_filename(f.name, label=1, label_name="genuine")
                if sample:
                    self.samples.append(sample)
                    self.persons.add(sample.person_id)
                else:
                    print(f"   ⚠️  Could not parse: {f.name}")
        else:
            raise FileNotFoundError(f"Genuine directory not found: {genuine_dir}")
        
        # ── Load Forged Signatures ────────────────────────────
        forged_dir = self.data_dir / CEDAR_FORGED_DIR
        if forged_dir.exists():
            forged_files = sorted(forged_dir.glob("*.png"))
            print(f"   Found {len(forged_files)} files in '{CEDAR_FORGED_DIR}/'")
            
            for f in forged_files:
                sample = self._parse_filename(f.name, label=0, label_name="forged")
                if sample:
                    self.samples.append(sample)
                    self.persons.add(sample.person_id)
                else:
                    print(f"   ⚠️  Could not parse: {f.name}")
        else:
            raise FileNotFoundError(f"Forged directory not found: {forged_dir}")
        
        # Build DataFrame
        self.df = pd.DataFrame([
            {
                "file_path": str(s.file_path),
                "person_id": s.person_id,
                "sample_num": s.sample_num,
                "label": s.label,
                "label_name": s.label_name,
                "filename": s.file_path.name
            }
            for s in self.samples
        ])
        
        print(f"✓ Discovery complete!")
        return self
    
    def get_summary(self) -> dict:
        """Return dataset statistics."""
        if not self.samples:
            return {"error": "Run .discover() first"}
        
        total = len(self.samples)
        genuine_count = sum(1 for s in self.samples if s.label == 1)
        forged_count = sum(1 for s in self.samples if s.label == 0)
        
        # Verify counts per person
        person_genuine = self.df[self.df['label'] == 1].groupby('person_id').size()
        person_forged = self.df[self.df['label'] == 0].groupby('person_id').size()
        
        return {
            "total_samples": total,
            "total_persons": len(self.persons),
            "expected_persons": NUM_PERSONS,
            "genuine_samples": genuine_count,
            "forged_samples": forged_count,
            "expected_per_person": SAMPLES_PER_PERSON,
            "class_balance": f"{genuine_count}:{forged_count}",
            "person_ids_found": sorted(self.persons),
            "persons_with_wrong_count": [
                p for p in self.persons
                if (person_genuine.get(p, 0) != SAMPLES_PER_PERSON or 
                    person_forged.get(p, 0) != SAMPLES_PER_PERSON)
            ]
        }
    
    def print_summary(self):
        """Pretty-print dataset summary."""
        stats = self.get_summary()
        print("\n" + "=" * 55)
        print("📊 CEDAR Dataset Summary")
        print("=" * 55)
        print(f"   Total samples:      {stats['total_samples']}")
        print(f"   Total persons:      {stats['total_persons']} (expected: {stats['expected_persons']})")
        print(f"   Genuine samples:    {stats['genuine_samples']}")
        print(f"   Forged samples:     {stats['forged_samples']}")
        print(f"   Expected per person: {stats['expected_per_person']} each class")
        print(f"   Class balance:      {stats['class_balance']}")
        print(f"   Person IDs:         {stats['person_ids_found'][:10]}... (showing first 10)")
        
        if stats['persons_with_wrong_count']:
            print(f"\n⚠️  Persons with wrong sample count: {stats['persons_with_wrong_count']}")
        else:
            print(f"\n✓ All persons have correct sample count")
        print("=" * 55)


# ── Standalone test ─────────────────────────────────────────
if __name__ == "__main__":
    loader = CEDARDatasetLoader()
    loader.discover().print_summary()
    
    print("\n📋 First 5 samples:")
    for s in loader.samples[:5]:
        print(f"   {s}")
    
    print("\n📋 Last 5 samples:")
    for s in loader.samples[-5:]:
        print(f"   {s}")