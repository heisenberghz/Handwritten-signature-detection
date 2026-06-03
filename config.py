"""
config.py - Central configuration for signature forgery detection
"""

from pathlib import Path

# ── Base Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "CEDAR"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "split"

# Output directories
CHECKPOINT_DIR = BASE_DIR / "experiments" / "checkpoints"
LOG_DIR = BASE_DIR / "experiments" / "logs"
FIGURES_DIR = BASE_DIR / "experiments" / "figures"

# ── Create directories if they don't exist ──────────────────
for directory in [PROCESSED_DIR, SPLIT_DIR, CHECKPOINT_DIR, LOG_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ── CEDAR Dataset Specific ──────────────────────────────────
CEDAR_GENUINE_DIR = "full_org"
CEDAR_FORGED_DIR = "full_forg"

GENUINE_PREFIX = "original"
FORGED_PREFIX = "forgeries"

NUM_PERSONS = 55
SAMPLES_PER_PERSON = 24

# ── Image Processing Parameters ─────────────────────────────
IMAGE_SIZE = (256, 256)
GRAYSCALE = True
NORMALIZE = True

# ── Preprocessing Parameters ────────────────────────────────
GAUSSIAN_BLUR_KERNEL = (5, 5)
MORPH_KERNEL_SIZE = (2, 2)

# ── Train/Val/Test Split ────────────────────────────────────
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ── Split Source ────────────────────────────────────────────
# Split is created FROM preprocessed images, not raw
SPLIT_SOURCE_DIR = PROCESSED_DIR  # data/processed/

# ── Model ────────────────────────────────────────────────────
MODEL_FILENAME = "cnn_best_v2.pth"

# ── Training Hyperparameters ─────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
PATIENCE = 5

# ── Verification ─────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.7

if __name__ == "__main__":
    print(f"Config loaded. Project root: {BASE_DIR}")
    print(f"Raw data: {RAW_DATA_DIR}")
    print(f"Preprocessed data: {PROCESSED_DIR}")
    print(f"Split output: {SPLIT_DIR}")
    print(f"Split source: {SPLIT_SOURCE_DIR}")