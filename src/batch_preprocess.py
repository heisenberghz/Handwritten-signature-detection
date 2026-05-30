"""
src/batch_preprocess.py - Batch preprocessing for entire CEDAR dataset
"""

import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PROCESSED_DIR, RAW_DATA_DIR, CEDAR_GENUINE_DIR, CEDAR_FORGED_DIR
from src.dataset_loader import CEDARDatasetLoader
from src.preprocessing import SignaturePreprocessor


class BatchPreprocessor:
    """
    Preprocesses the entire CEDAR dataset and saves results.
    Maintains folder structure: processed/genuine/ and processed/forged/
    """

    def __init__(self, output_dir: Path = PROCESSED_DIR):
        self.output_dir = output_dir
        self.preprocessor = SignaturePreprocessor()
        self.loader = CEDARDatasetLoader()

    def process_all(self, save_visualizations: bool = True) -> dict:
        """
        Process all signatures in the dataset.

        Args:
            save_visualizations: If True, also save uint8 copies for viewing

        Returns:
            Statistics dictionary
        """
        self.loader.discover()
        stats = self.loader.get_summary()
        print(f"\n📦 Processing {stats['total_samples']} signatures...")

        processed_count = 0
        failed_count = 0

        for sample in tqdm(self.loader.samples, desc="Preprocessing"):
            class_dir = self.output_dir / sample.label_name
            class_dir.mkdir(parents=True, exist_ok=True)

            output_name = sample.file_path.stem + ".png"
            output_path = class_dir / output_name
            viz_path = class_dir / (sample.file_path.stem + "_viz.png")

            try:
                processed = self.preprocessor.preprocess(sample.file_path)

                if processed is None:
                    failed_count += 1
                    continue

                # Save float32 version for model training
                self.preprocessor.save_preprocessed(processed, output_path, as_uint8=False)

                # Save visualization copy
                if save_visualizations:
                    self.preprocessor.save_preprocessed(processed, viz_path, as_uint8=True)

                processed_count += 1

            except Exception as e:
                print(f"\n⚠️  Error processing {sample.file_path}: {e}")
                failed_count += 1

        result = {
            "total": stats['total_samples'],
            "processed": processed_count,
            "failed": failed_count,
            "output_dir": str(self.output_dir)
        }

        print(f"\n✅ Batch preprocessing complete!")
        print(f"   Processed: {processed_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Output: {self.output_dir}")

        return result

    def verify_output(self, num_samples: int = 3) -> None:
        """Verify preprocessed output by checking a few samples."""
        print(f"\n🔍 Verifying preprocessed data...")

        for class_name in ["genuine", "forged"]:
            class_dir = self.output_dir / class_name
            if not class_dir.exists():
                print(f"   ⚠️  Directory not found: {class_dir}")
                continue

            files = sorted([f for f in class_dir.glob("*.png") if not f.name.endswith("_viz.png")])

            checked = 0
            for test_file in files[:num_samples]:
                try:
                    import cv2
                    img = cv2.imread(str(test_file), cv2.IMREAD_UNCHANGED)

                    if img is not None:
                        # Convert from 16-bit back to float
                        if img.dtype == np.uint16:
                            img = img.astype(np.float32) / 65535.0
                        elif img.max() > 1.0:
                            img = img.astype(np.float32) / 255.0

                        print(f"   ✓ {class_name}/{test_file.name}: "
                              f"shape={img.shape}, dtype={img.dtype}, "
                              f"range=[{img.min():.3f}, {img.max():.3f}]")
                        checked += 1
                except Exception as e:
                    print(f"   ⚠️  Error reading {test_file}: {e}")

            print(f"   Verified {checked} {class_name} samples")


# ── Standalone run ──────────────────────────────────────────
if __name__ == "__main__":
    batch = BatchPreprocessor()
    results = batch.process_all(save_visualizations=True)
    batch.verify_output(num_samples=3)