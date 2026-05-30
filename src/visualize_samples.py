"""
src/visualize_samples.py - Visualize original vs preprocessed signatures
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_DATA_DIR, PROCESSED_DIR, CEDAR_GENUINE_DIR, CEDAR_FORGED_DIR
from src.preprocessing import SignaturePreprocessor


def show_comparison(person_id: int = None, num_pairs: int = 3):
    """
    Display original vs preprocessed signatures side by side.
    
    Args:
        person_id: Specific person to show (1-55), or None for random
        num_pairs: Number of signature pairs to show
    """
    preprocessor = SignaturePreprocessor()
    
    genuine_dir = RAW_DATA_DIR / CEDAR_GENUINE_DIR
    forged_dir = RAW_DATA_DIR / CEDAR_FORGED_DIR
    
    if not genuine_dir.exists() or not forged_dir.exists():
        print("CEDAR directories not found!")
        return
    
    # Get files
    genuine_files = sorted([f for f in genuine_dir.glob("*.png")])
    forged_files = sorted([f for f in forged_dir.glob("*.png")])
    
    if person_id:
        pattern_g = f"original_{person_id}_"
        pattern_f = f"forgeries_{person_id}_"
        genuine_files = [f for f in genuine_files if f.name.startswith(pattern_g)]
        forged_files = [f for f in forged_files if f.name.startswith(pattern_f)]
    
    # Select samples
    num_show = min(num_pairs, len(genuine_files), len(forged_files))
    selected_genuine = genuine_files[:num_show]
    selected_forged = forged_files[:num_show]
    
    if num_show == 0:
        print("No matching files found!")
        return
    
    fig, axes = plt.subplots(num_show, 4, figsize=(16, 4 * num_show))
    if num_show == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_show):
        # Genuine - Original
        orig_g = cv2.imread(str(selected_genuine[idx]), cv2.IMREAD_GRAYSCALE)
        axes[idx, 0].imshow(orig_g, cmap='gray')
        axes[idx, 0].set_title(f"Genuine\n{selected_genuine[idx].name}")
        axes[idx, 0].axis('off')
        
        # Genuine - Preprocessed
        proc_g = preprocessor.preprocess(selected_genuine[idx])
        axes[idx, 1].imshow(proc_g, cmap='gray')
        axes[idx, 1].set_title("Genuine - Preprocessed")
        axes[idx, 1].axis('off')
        
        # Forged - Original
        orig_f = cv2.imread(str(selected_forged[idx]), cv2.IMREAD_GRAYSCALE)
        axes[idx, 2].imshow(orig_f, cmap='gray')
        axes[idx, 2].set_title(f"Forged\n{selected_forged[idx].name}")
        axes[idx, 2].axis('off')
        
        # Forged - Preprocessed
        proc_f = preprocessor.preprocess(selected_forged[idx])
        axes[idx, 3].imshow(proc_f, cmap='gray')
        axes[idx, 3].set_title("Forged - Preprocessed")
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    
    from config import FIGURES_DIR
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURES_DIR / "preprocessing_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    show_comparison(person_id=1, num_pairs=3)