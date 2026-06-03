"""
src/verify_signature.py - Pairwise signature verification using CNN feature vectors
"""

import sys
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# Load modules directly from files (Windows-compatible)
project_root = Path(__file__).parent.parent.resolve()

# Load config
config_path = project_root / "config.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

IMAGE_SIZE = config.IMAGE_SIZE
CHECKPOINT_DIR = config.CHECKPOINT_DIR
MODEL_FILENAME = config.MODEL_FILENAME
SIMILARITY_THRESHOLD = config.SIMILARITY_THRESHOLD

# Load model
model_path = project_root / "src" / "model.py"
spec = importlib.util.spec_from_file_location("model", model_path)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
SignatureCNN = model_module.SignatureCNN


class _FeatureExtractor(nn.Module):
    """Shares conv weights with trained CNN, returns 32768-dim feature vector."""
    
    def __init__(self, trained_model: nn.Module):
        super().__init__()
        self.conv1 = trained_model.conv1
        self.bn1 = trained_model.bn1
        self.conv2 = trained_model.conv2
        self.bn2 = trained_model.bn2
        self.conv3 = trained_model.conv3
        self.bn3 = trained_model.bn3
        self.conv4 = trained_model.conv4
        self.bn4 = trained_model.bn4
        self.pool = trained_model.pool
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        return x.view(x.size(0), -1)


class SignatureVerifier:
    """Compare two signatures using cosine similarity of deep features."""
    
    def __init__(self, model_path: Path = None, threshold: float = SIMILARITY_THRESHOLD, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = threshold
        
        # Load model architecture
        self.model = SignatureCNN(num_classes=2, dropout=0.5).to(self.device)
        
        if model_path is None:
            model_path = CHECKPOINT_DIR / MODEL_FILENAME
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        # Create feature extractor from trained model
        self.feature_extractor = _FeatureExtractor(self.model).to(self.device)
        self.feature_extractor.eval()
    
    def preprocess_image(self, image_path: Path) -> torch.Tensor:
        """Same preprocessing pipeline as predict.py."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        mean_val = np.mean(img)
        if mean_val > 127:
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Center signature
        coords = cv2.findNonZero(binary)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cropped = binary[y:y + h, x:x + w]
            
            canvas = np.zeros(IMAGE_SIZE, dtype=np.uint8)
            start_y = max(0, (IMAGE_SIZE[0] - h) // 2)
            start_x = max(0, (IMAGE_SIZE[1] - w) // 2)
            end_y = min(start_y + h, IMAGE_SIZE[0])
            end_x = min(start_x + w, IMAGE_SIZE[1])
            crop_h = end_y - start_y
            crop_w = end_x - start_x
            
            canvas[start_y:end_y, start_x:end_x] = cropped[:crop_h, :crop_w]
            binary = canvas
        
        tensor = torch.from_numpy(binary.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        return tensor
    
    def extract_features(self, image_path: Path) -> np.ndarray:
        """Return 32768-dim feature vector for a signature image."""
        tensor = self.preprocess_image(image_path).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(tensor)
        return features.cpu().numpy().flatten()
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        return float(dot / (norm1 * norm2))
    
    def verify(self, reference_path: Path, questioned_path: Path) -> dict:
        ref_feats = self.extract_features(reference_path)
        q_feats = self.extract_features(questioned_path)
        similarity = self.cosine_similarity(ref_feats, q_feats)
        
        is_match = similarity >= self.threshold
        
        # Normalize confidence based on distance from threshold
        if is_match:
            normalized = (similarity - self.threshold) / (1.0 - self.threshold)
            confidence = min(0.5 + 0.5 * normalized, 0.99)
        else:
            normalized = (self.threshold - similarity) / self.threshold
            confidence = min(0.5 + 0.5 * normalized, 0.99)
        
        return {
            "similarity_score": round(similarity, 4),
            "is_match": is_match,
            "confidence": round(confidence, 4),
            "threshold": self.threshold,
            "prediction": "Match" if is_match else "Mismatch",
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify two signatures")
    parser.add_argument("reference", type=str, help="Path to reference (genuine) signature")
    parser.add_argument("questioned", type=str, help="Path to questioned signature")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold")
    
    args = parser.parse_args()
    
    verifier = SignatureVerifier(threshold=args.threshold)
    result = verifier.verify(Path(args.reference), Path(args.questioned))
    
    print(f"\n{'='*50}")
    print(f"Signature Verification Result")
    print(f"{'='*50}")
    print(f"  Similarity Score: {result['similarity_score']:.4f}")
    print(f"  Verdict:          {result['prediction']}")
    print(f"  Confidence:       {result['confidence']:.2%}")
    print(f"  Threshold:        {result['threshold']}")
    print(f"{'='*50}")