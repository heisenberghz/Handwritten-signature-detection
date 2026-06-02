"""
src/predict.py - Predict if a signature is genuine or forged
"""

import sys
import importlib.util
from pathlib import Path

import torch
import numpy as np
import cv2

# Load modules
project_root = Path(__file__).parent.parent.resolve()

# Load config
config_path = project_root / "config.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

IMAGE_SIZE = config.IMAGE_SIZE
CHECKPOINT_DIR = config.CHECKPOINT_DIR

# Load model
model_path = project_root / "src" / "model.py"
spec = importlib.util.spec_from_file_location("model", model_path)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
SignatureCNN = model_module.SignatureCNN


class SignaturePredictor:
    """Load trained CNN and predict on new signature images."""
    
    def __init__(self, model_path: Path = None, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"💻 Using device: {self.device}")
        
        # Load model architecture
        self.model = SignatureCNN(num_classes=2, dropout=0.5).to(self.device)
        
        # Load trained weights
        if model_path is None:
            model_path = CHECKPOINT_DIR / "cnn_best_v2.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        print(f"✅ Loaded model from epoch {checkpoint.get('epoch', '?')}")
        print(f"   Validation accuracy: {checkpoint.get('val_accuracy', '?'):.4f}")
    
    def preprocess_image(self, image_path: Path) -> torch.Tensor:
        """
        Preprocess a raw signature image for prediction.
        Same pipeline as training: grayscale → denoise → binarize → center → normalize
        """
        # Read image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Denoise
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Binarize
        mean_val = np.mean(img)
        if mean_val > 127:
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove noise
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Center signature
        coords = cv2.findNonZero(binary)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cropped = binary[y:y+h, x:x+w]
            
            # Resize to target with padding
            canvas = np.zeros(IMAGE_SIZE, dtype=np.uint8)
            start_y = max(0, (IMAGE_SIZE[0] - h) // 2)
            start_x = max(0, (IMAGE_SIZE[1] - w) // 2)
            end_y = min(start_y + h, IMAGE_SIZE[0])
            end_x = min(start_x + w, IMAGE_SIZE[1])
            crop_h = end_y - start_y
            crop_w = end_x - start_x
            
            canvas[start_y:end_y, start_x:end_x] = cropped[:crop_h, :crop_w]
            binary = canvas
        
        # Normalize
        img_tensor = torch.from_numpy(binary.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        
        return img_tensor
    
    def predict(self, image_path: Path) -> dict:
        """
        Predict if signature is genuine or forged.
        
        Returns:
            dict with prediction, confidence, and probabilities
        """
        # Preprocess
        img_tensor = self.preprocess_image(image_path)
        img_tensor = img_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        # Get prediction
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
        
        genuine_prob = probs[0][1].item()
        forged_prob = probs[0][0].item()
        
        return {
            "prediction": "Genuine" if predicted_class == 1 else "Forged",
            "confidence": confidence,
            "genuine_probability": genuine_prob,
            "forged_probability": forged_prob,
            "predicted_class": predicted_class
        }


def test_on_split(split_name: str = "test"):
    """Test model on entire train/val/test split and show metrics."""
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    print(f"\n{'='*60}")
    print(f"🧪 Testing on {split_name.upper()} split")
    print(f"{'='*60}")
    
    predictor = SignaturePredictor()
    
    split_dir = CHECKPOINT_DIR.parent.parent / "data" / "split" / split_name
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for label_name, label in [("genuine", 1), ("forged", 0)]:
        class_dir = split_dir / label_name
        
        if not class_dir.exists():
            continue
        
        files = list(class_dir.glob("*.png"))
        files = [f for f in files if not f.name.endswith("_viz.png")]
        
        print(f"\n📂 {label_name.upper()}: {len(files)} images")
        
        correct = 0
        
        for img_path in files[:10]:  # Show first 10
            result = predictor.predict(img_path)
            
            pred_label = 1 if result["prediction"] == "Genuine" else 0
            all_preds.append(pred_label)
            all_labels.append(label)
            all_probs.append(result["genuine_probability"])
            
            is_correct = pred_label == label
            if is_correct:
                correct += 1
            
            status = "✅" if is_correct else "❌"
            print(f"   {status} {img_path.name:20s} → {result['prediction']:8s} "
                  f"(confidence: {result['confidence']:.3f})")
        
        print(f"   ... and {max(0, len(files)-10)} more")
        print(f"   Accuracy on shown samples: {correct}/10 = {correct/10:.1%}")
    
    # Overall metrics
    if len(all_preds) > 0:
        print(f"\n📊 Overall Metrics:")
        print(f"   Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
        print(f"   Confusion Matrix:\n{confusion_matrix(all_labels, all_preds)}")


def predict_single(image_path: str):
    """Predict on a single image."""
    predictor = SignaturePredictor()
    
    path = Path(image_path)
    if not path.exists():
        print(f"❌ File not found: {path}")
        return
    
    result = predictor.predict(path)
    
    print(f"\n{'='*50}")
    print(f"🖼️  Image: {path.name}")
    print(f"{'='*50}")
    print(f"   Prediction: {result['prediction']}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"")
    print(f"   Genuine probability: {result['genuine_probability']:.3f}")
    print(f"   Forged probability:  {result['forged_probability']:.3f}")
    print(f"{'='*50}")


# ── Standalone test ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict signature authenticity")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], 
                       help="Test on entire split")
    
    args = parser.parse_args()
    
    if args.image:
        predict_single(args.image)
    elif args.split:
        test_on_split(args.split)
    else:
        # Default: test on a few test images
        print("No arguments provided. Testing on first 10 test images...")
        test_on_split("test")