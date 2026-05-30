"""
src/feature_extraction.py - Handcrafted features for signature forgery detection
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import IMAGE_SIZE


@dataclass
class SignatureFeatures:
    """Container for all extracted features."""
    person_id: int
    sample_num: int
    label: int
    label_name: str
    
    # Geometric features
    aspect_ratio: float
    signature_area_ratio: float
    center_of_mass_x: float
    center_of_mass_y: float
    eccentricity: float
    solidity: float
    orientation: float
    
    # Stroke features
    num_strokes: int
    stroke_density: float
    avg_stroke_width: float
    
    # Texture features
    lbp_uniformity: float
    lbp_entropy: float
    
    # Shape features (Hu moments - rotation/scale invariant)
    hu_moment_1: float
    hu_moment_2: float
    hu_moment_3: float
    hu_moment_4: float
    hu_moment_5: float
    hu_moment_6: float
    hu_moment_7: float
    
    # GLCM features
    contrast: float
    correlation: float
    energy: float
    homogeneity: float
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy feature vector."""
        return np.array([
            self.aspect_ratio,
            self.signature_area_ratio,
            self.center_of_mass_x,
            self.center_of_mass_y,
            self.eccentricity,
            self.solidity,
            self.orientation,
            self.num_strokes,
            self.stroke_density,
            self.avg_stroke_width,
            self.lbp_uniformity,
            self.lbp_entropy,
            self.hu_moment_1,
            self.hu_moment_2,
            self.hu_moment_3,
            self.hu_moment_4,
            self.hu_moment_5,
            self.hu_moment_6,
            self.hu_moment_7,
            self.contrast,
            self.correlation,
            self.energy,
            self.homogeneity
        ], dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [
            "aspect_ratio",
            "signature_area_ratio",
            "center_of_mass_x",
            "center_of_mass_y",
            "eccentricity",
            "solidity",
            "orientation",
            "num_strokes",
            "stroke_density",
            "avg_stroke_width",
            "lbp_uniformity",
            "lbp_entropy",
            "hu_moment_1",
            "hu_moment_2",
            "hu_moment_3",
            "hu_moment_4",
            "hu_moment_5",
            "hu_moment_6",
            "hu_moment_7",
            "contrast",
            "correlation",
            "energy",
            "homogeneity"
        ]


class FeatureExtractor:
    """
    Extracts handcrafted features from preprocessed signature images.
    
    Features based on graphology + computer vision for forgery detection.
    """

    def __init__(self, image_size: Tuple[int, int] = IMAGE_SIZE):
        self.image_size = image_size

    def _get_signature_mask(self, image: np.ndarray) -> np.ndarray:
        """Get binary mask of signature (white pixels on black)."""
        return (image > 0.1).astype(np.uint8)

    def extract_geometric(self, image: np.ndarray) -> Dict[str, float]:
        """Extract geometric features."""
        mask = self._get_signature_mask(image)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                "aspect_ratio": 1.0,
                "signature_area_ratio": 0.0,
                "center_of_mass_x": 0.5,
                "center_of_mass_y": 0.5,
                "eccentricity": 0.0,
                "solidity": 0.0,
                "orientation": 0.0
            }
        
        # Use largest contour (main signature body)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Area ratio
        total_pixels = self.image_size[0] * self.image_size[1]
        signature_pixels = np.sum(mask)
        area_ratio = signature_pixels / total_pixels
        
        # Center of mass
        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            com_x = moments["m10"] / moments["m00"] / self.image_size[1]
            com_y = moments["m01"] / moments["m00"] / self.image_size[0]
        else:
            com_x, com_y = 0.5, 0.5
        
        # Fit ellipse for eccentricity and orientation
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            # Eccentricity from ellipse axes
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0
            orientation = ellipse[2]  # Angle in degrees
        else:
            eccentricity = 0.0
            orientation = 0.0
        
        # Solidity = contour area / convex hull area
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(largest_contour)
        solidity = contour_area / hull_area if hull_area > 0 else 0.0
        
        return {
            "aspect_ratio": aspect_ratio,
            "signature_area_ratio": area_ratio,
            "center_of_mass_x": com_x,
            "center_of_mass_y": com_y,
            "eccentricity": eccentricity,
            "solidity": solidity,
            "orientation": orientation
        }

    def extract_stroke_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract stroke-based features."""
        mask = self._get_signature_mask(image)
        
        # Skeletonize
        from skimage.morphology import skeletonize
        skeleton = skeletonize(mask.astype(bool)).astype(np.uint8)
        
        # Count skeleton pixels as proxy for stroke complexity
        num_strokes = max(1, int(np.sum(skeleton) / 50))
        
        # Stroke density
        signature_pixels = np.sum(mask)
        stroke_density = signature_pixels / (self.image_size[0] * self.image_size[1])
        
        # Average stroke width
        from scipy.ndimage import distance_transform_edt
        dist_transform = distance_transform_edt(mask)
        avg_width = np.mean(dist_transform[mask > 0]) * 2 if np.sum(mask) > 0 else 0
        
        return {
            "num_strokes": num_strokes,
            "stroke_density": stroke_density,
            "avg_stroke_width": avg_width
        }

    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract texture features using Local Binary Patterns."""
        img_uint8 = (image * 255).astype(np.uint8)
        
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(img_uint8, n_points, radius, method='uniform')
        
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        
        uniformity = np.sum(hist ** 2)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return {
            "lbp_uniformity": uniformity,
            "lbp_entropy": entropy
        }

    def extract_hu_moments(self, image: np.ndarray) -> Dict[str, float]:
        """Extract Hu moments (rotation/scale/translation invariant shape descriptors)."""
        mask = self._get_signature_mask(image)
        
        moments = cv2.moments(mask)
        hu = cv2.HuMoments(moments).flatten()
        
        # Log transform for better numerical stability
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        
        return {
            "hu_moment_1": float(hu_log[0]),
            "hu_moment_2": float(hu_log[1]),
            "hu_moment_3": float(hu_log[2]),
            "hu_moment_4": float(hu_log[3]),
            "hu_moment_5": float(hu_log[4]),
            "hu_moment_6": float(hu_log[5]),
            "hu_moment_7": float(hu_log[6])
        }

    def extract_glcm_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract GLCM texture features."""
        img_uint8 = (image * 255).astype(np.uint8)
        
        glcm = graycomatrix(
            img_uint8,
            distances=[1, 2, 3],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )
        
        return {
            "contrast": float(np.mean(graycoprops(glcm, 'contrast'))),
            "correlation": float(np.mean(graycoprops(glcm, 'correlation'))),
            "energy": float(np.mean(graycoprops(glcm, 'energy'))),
            "homogeneity": float(np.mean(graycoprops(glcm, 'homogeneity')))
        }

    def extract_all(self, image: np.ndarray, person_id: int, sample_num: int, label: int, label_name: str) -> SignatureFeatures:
        """Extract all features from a single image."""
        geo = self.extract_geometric(image)
        stroke = self.extract_stroke_features(image)
        texture = self.extract_texture_features(image)
        hu = self.extract_hu_moments(image)
        glcm = self.extract_glcm_features(image)
        
        return SignatureFeatures(
            person_id=person_id,
            sample_num=sample_num,
            label=label,
            label_name=label_name,
            **geo,
            **stroke,
            **texture,
            **hu,
            **glcm
        )


# ── Standalone test ─────────────────────────────────────────
if __name__ == "__main__":
    from config import SPLIT_DIR
    
    print("=" * 55)
    print("🔍 Testing FeatureExtractor")
    print("=" * 55)
    
    extractor = FeatureExtractor()
    
    # Load one sample from train split
    train_genuine_dir = SPLIT_DIR / "train" / "genuine"
    test_files = sorted(list(train_genuine_dir.glob("*.png")))
    
    if not test_files:
        print("No files found in train/genuine/")
        sys.exit(1)
    
    # Load first image
    img_path = test_files[0]
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"Could not read: {img_path}")
        sys.exit(1)
    
    # Convert 16-bit to float
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    
    print(f"\n🖼️  Testing with: {img_path.name}")
    print(f"   Image shape: {img.shape}, dtype: {img.dtype}")
    
    # Extract features
    features = extractor.extract_all(
        image=img,
        person_id=1,
        sample_num=1,
        label=1,
        label_name="genuine"
    )
    
    print(f"\n📊 Extracted Features:")
    for name, value in zip(features.get_feature_names(), features.to_vector()):
        print(f"   {name:25s}: {value:.6f}")
    
    print(f"\n✅ Feature vector shape: {features.to_vector().shape}")
    print(f"✅ Total features: {len(features.to_vector())}")