# Signature Forensics AI

Deep learning–based forensic analysis for handwritten signature forgery detection. Trained on the [CEDAR](http://www.cedar.buffalo.edu/) signature dataset (55 writers, 2,640 images).

## Features

- **Single Signature Analysis** — Classify a signature as genuine or forged using a CNN classifier (76.39% test accuracy)
- **Pairwise Verification** — Compare a questioned signature against a known genuine reference using deep feature cosine similarity (32,768-dim feature vectors)
- **Web UI** — Streamlit app with interactive analysis and visualization

## Project Structure

```
├── app.py                          # Streamlit web app
├── config.py                       # Central configuration
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── raw/CEDAR/                  # Original dataset
│   ├── processed/                  # Preprocessed images (16-bit PNG)
│   └── split/                      # Writer-independent train/val/test
├── src/
│   ├── model.py                    # SignatureCNN architecture
│   ├── preprocessing.py            # Image preprocessing pipeline
│   ├── batch_preprocess.py         # Batch preprocessing
│   ├── data_split.py               # Writer-independent split
│   ├── dataset_loader.py           # CEDAR dataset loader
│   ├── signature_dataset.py        # PyTorch Dataset wrapper
│   ├── feature_extraction.py       # Handcrafted features
│   ├── traditional_ml.py           # SVM / Random Forest baselines
│   ├── train_cnn.py                # CNN training pipeline
│   ├── predict.py                  # Single image prediction
│   ├── verify_signature.py         # Pairwise verification
│   └── visualize_samples.py        # Preprocessing visualization
├── experiments/
│   ├── checkpoints/                # Trained models
│   ├── figures/                    # Generated figures
│   └── logs/                       # Training logs
└── tests/                          # Test suite
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Web App

```bash
streamlit run app.py
```

### Training

```bash
python src/train_cnn.py
```

### Prediction

```bash
python src/predict.py --image path/to/signature.png
```

### Pairwise Verification

```bash
python src/verify_signature.py reference.png questioned.png
```

### Tests

```bash
python -m pytest tests/
```

## Model

| Metric | Value |
|--------|-------|
| Architecture | SignatureCNN (4 conv + 3 FC, ~4M params) |
| Test Accuracy | 76.39% |
| Precision | 71.43% |
| Recall | 87.96% |
| F1-Score | 78.84% |
| ROC-AUC | 87.78% |

## Baselines

| Model | Accuracy |
|-------|----------|
| SVM (handcrafted features) | 61.34% |
| Random Forest | 63.19% |
| CNN v2 (production) | **76.39%** |

## Dataset

The CEDAR dataset contains 1,320 genuine and 1,320 forged signatures from 55 writers (24 genuine + 24 forged per writer). A writer-independent split ensures no writer appears in more than one split.

## Project status

All core features are implemented. See the issues tracker for planned improvements.
