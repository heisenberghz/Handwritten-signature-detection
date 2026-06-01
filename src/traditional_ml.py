"""
src/traditional_ml.py - Traditional ML models for signature forgery detection
Trains SVM and Random Forest on handcrafted features from PREPROCESSED images.
"""

import sys
import pickle
from pathlib import Path
from typing import Tuple, List

import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SPLIT_DIR, CHECKPOINT_DIR
from src.feature_extraction import FeatureExtractor


def load_split_features(split_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract features from all PREPROCESSED images in a split.
    
    Returns:
        X: feature matrix (n_samples, n_features)
        y: labels (n_samples,)
        filenames: list of image filenames
    """
    print(f"\n📥 Loading {split_name} features (from PREPROCESSED images)...")
    
    extractor = FeatureExtractor()
    features_list = []
    labels_list = []
    filenames_list = []
    
    for label_name, label in [("genuine", 1), ("forged", 0)]:
        class_dir = SPLIT_DIR / split_name / label_name
        
        if not class_dir.exists():
            raise FileNotFoundError(f"Directory not found: {class_dir}")
        
        for img_path in sorted(class_dir.glob("*.png")):
            if img_path.name.endswith("_viz.png"):
                continue
            
            # Load preprocessed image (16-bit PNG)
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"   ⚠️  Could not read: {img_path.name}")
                continue
            
            # Convert 16-bit to float32 [0, 1]
            if img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            
            # Parse person_id and sample_num from filename
            # original_X_Y.png or forgeries_X_Y.png
            stem = img_path.stem
            parts = stem.split("_")
            person_id = int(parts[1]) if len(parts) > 1 else 0
            sample_num = int(parts[2]) if len(parts) > 2 else 0
            
            # Extract features
            features = extractor.extract_all(
                image=img,
                person_id=person_id,
                sample_num=sample_num,
                label=label,
                label_name=label_name
            )
            
            features_list.append(features.to_vector())
            labels_list.append(label)
            filenames_list.append(img_path.name)
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"   ✓ Loaded {len(X)} samples, {X.shape[1]} features each")
    
    return X, y, filenames_list


def train_and_evaluate():
    """Train SVM and Random Forest, evaluate on validation set."""
    print("=" * 60)
    print("🤖 Traditional ML: SVM + Random Forest (PREPROCESSED features)")
    print("=" * 60)
    
    # ── Load features ─────────────────────────────────────
    X_train, y_train, _ = load_split_features("train")
    X_val, y_val, _ = load_split_features("val")
    X_test, y_test, _ = load_split_features("test")
    
    # ── Feature scaling ───────────────────────────────────
    print("\n⚖️  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = CHECKPOINT_DIR / "feature_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"   ✓ Scaler saved: {scaler_path}")
    
    # ── Train SVM ─────────────────────────────────────────
    print("\n" + "-" * 40)
    print("📌 Training SVM...")
    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42
    )
    svm.fit(X_train_scaled, y_train)
    
    # Evaluate SVM on validation
    svm_val_pred = svm.predict(X_val_scaled)
    print("\n   📊 SVM Validation Results:")
    print(f"      Accuracy:  {accuracy_score(y_val, svm_val_pred):.4f}")
    print(f"      Precision: {precision_score(y_val, svm_val_pred):.4f}")
    print(f"      Recall:    {recall_score(y_val, svm_val_pred):.4f}")
    print(f"      F1-Score:  {f1_score(y_val, svm_val_pred):.4f}")
    print(f"\n      Confusion Matrix:\n{confusion_matrix(y_val, svm_val_pred)}")
    
    # Save SVM
    svm_path = CHECKPOINT_DIR / "svm_model.pkl"
    with open(svm_path, "wb") as f:
        pickle.dump(svm, f)
    print(f"   ✓ SVM saved: {svm_path}")
    
    # ── Train Random Forest ───────────────────────────────
    print("\n" + "-" * 40)
    print("🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate RF on validation
    rf_val_pred = rf.predict(X_val_scaled)
    print("\n   📊 Random Forest Validation Results:")
    print(f"      Accuracy:  {accuracy_score(y_val, rf_val_pred):.4f}")
    print(f"      Precision: {precision_score(y_val, rf_val_pred):.4f}")
    print(f"      Recall:    {recall_score(y_val, rf_val_pred):.4f}")
    print(f"      F1-Score:  {f1_score(y_val, rf_val_pred):.4f}")
    print(f"\n      Confusion Matrix:\n{confusion_matrix(y_val, rf_val_pred)}")
    
    # Feature importance
    feature_names = FeatureExtractor().extract_all(
        np.zeros((256, 256)), 0, 0, 0, ""
    ).get_feature_names()
    
    importances = rf.feature_importances_
    print(f"\n   🔝 Top 5 Important Features:")
    indices = np.argsort(importances)[::-1][:5]
    for i in indices:
        print(f"      {feature_names[i]:25s}: {importances[i]:.4f}")
    
    # Save RF
    rf_path = CHECKPOINT_DIR / "rf_model.pkl"
    with open(rf_path, "wb") as f:
        pickle.dump(rf, f)
    print(f"   ✓ Random Forest saved: {rf_path}")
    
    # ── Final Test Evaluation ─────────────────────────────
    print("\n" + "=" * 60)
    print("🧪 Final Test Set Evaluation")
    print("=" * 60)
    
    svm_test_pred = svm.predict(X_test_scaled)
    rf_test_pred = rf.predict(X_test_scaled)
    
    print("\n📌 SVM Test Results:")
    print(f"   Accuracy:  {accuracy_score(y_test, svm_test_pred):.4f}")
    print(f"   Precision: {precision_score(y_test, svm_test_pred):.4f}")
    print(f"   Recall:    {recall_score(y_test, svm_test_pred):.4f}")
    print(f"   F1-Score:  {f1_score(y_test, svm_test_pred):.4f}")
    
    print("\n🌲 Random Forest Test Results:")
    print(f"   Accuracy:  {accuracy_score(y_test, rf_test_pred):.4f}")
    print(f"   Precision: {precision_score(y_test, rf_test_pred):.4f}")
    print(f"   Recall:    {recall_score(y_test, rf_test_pred):.4f}")
    print(f"   F1-Score:  {f1_score(y_test, rf_test_pred):.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Traditional ML training complete!")
    print(f"   Models saved to: {CHECKPOINT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    train_and_evaluate()