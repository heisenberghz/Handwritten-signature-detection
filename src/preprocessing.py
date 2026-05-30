"""
src/preprocessing.py - Image preprocessing pipeline for signature images
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    IMAGE_SIZE,
    GRAYSCALE,
    NORMALIZE,
    GAUSSIAN_BLUR_KERNEL,
    MORPH_KERNEL_SIZE,
    PROCESSED_DIR
)


class SignaturePreprocessor:
    """
    Preprocesses signature images for forgery detection.
    
    Pipeline:
    1. Read image
    2. Convert to grayscale
    3. Apply Gaussian blur for noise reduction
    4. Binarize using Otsu's thresholding
    5. Remove small noise with morphological operations
    6. Center the signature in the canvas
    7. Normalize pixel values
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = IMAGE_SIZE,
        grayscale: bool = GRAYSCALE,
        normalize: bool = NORMALIZE,
        blur_kernel: Tuple[int, int] = GAUSSIAN_BLUR_KERNEL,
        morph_kernel: Tuple[int, int] = MORPH_KERNEL_SIZE
    ):
        self.target_size = target_size
        self.grayscale = grayscale
        self.normalize = normalize
        self.blur_kernel = self._ensure_odd(blur_kernel)
        self.morph_kernel = morph_kernel

    @staticmethod
    def _ensure_odd(kernel: Tuple[int, int]) -> Tuple[int, int]:
        """Ensure kernel dimensions are odd numbers."""
        return tuple(k + 1 if k % 2 == 0 else k for k in kernel)

    def read_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Read an image from disk. Returns None if file cannot be read."""
        path = str(image_path)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        
        if image is None:
            print(f"⚠️  Could not read image: {path}")
            return None
        
        return image

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to grayscale."""
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to reduce noise."""
        return cv2.GaussianBlur(image, self.blur_kernel, 0)

    def binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Convert grayscale image to binary.
        Black ink on white background -> white signature on black background.
        """
        # Otsu's thresholding
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def remove_noise(self, binary_image: np.ndarray) -> np.ndarray:
        """Remove small noise dots using morphological opening."""
        kernel = np.ones(self.morph_kernel, np.uint8)
        cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
        return cleaned

    def center_signature(self, binary_image: np.ndarray) -> np.ndarray:
        """Center the signature within the image canvas."""
        coords = cv2.findNonZero(binary_image)
        
        if coords is None:
            return binary_image
        
        x, y, w, h = cv2.boundingRect(coords)
        cropped = binary_image[y:y+h, x:x+w]
        
        canvas = np.zeros(self.target_size, dtype=np.uint8)
        
        start_y = max(0, (self.target_size[0] - h) // 2)
        start_x = max(0, (self.target_size[1] - w) // 2)
        
        end_y = min(start_y + h, self.target_size[0])
        end_x = min(start_x + w, self.target_size[1])
        crop_h = end_y - start_y
        crop_w = end_x - start_x
        
        canvas[start_y:end_y, start_x:end_x] = cropped[:crop_h, :crop_w]
        
        return canvas

    def normalize_pixels(self, image: np.ndarray) -> np.ndarray:
        """Normalize pixel values to [0, 1] range."""
        if self.normalize:
            return image.astype(np.float32) / 255.0
        return image.astype(np.float32)

    def preprocess(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Run the complete preprocessing pipeline.
        
        Returns:
            Preprocessed image as float32 numpy array, shape (H, W)
        """
        image = self.read_image(image_path)
        if image is None:
            return None
        
        if self.grayscale:
            image = self.to_grayscale(image)
        
        image = self.denoise(image)
        image = self.binarize(image)
        image = self.remove_noise(image)
        image = self.center_signature(image)
        image = self.normalize_pixels(image)
        
        return image

    def save_preprocessed(
        self,
        image: np.ndarray,
        output_path: Union[str, Path],
        as_uint8: bool = False
    ) -> None:
        """
        Save preprocessed image to disk.
        
        Args:
            image: Preprocessed image array
            output_path: Where to save
            as_uint8: If True, convert back to 0-255 for visualization
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if as_uint8:
            save_img = (image * 255).astype(np.uint8)
        else:
            save_img = image
        
        if save_img.dtype == np.float32 or save_img.dtype == np.float64:
            save_img_16bit = (save_img * 65535).astype(np.uint16)
            cv2.imwrite(str(output_path), save_img_16bit)
        else:
            cv2.imwrite(str(output_path), save_img)

    def get_pipeline_steps(self) -> list:
        """Return list of preprocessing steps for documentation."""
        return [
            "1. Read image from disk",
            "2. Convert to grayscale" if self.grayscale else "2. Keep color channels",
            "3. Gaussian blur (noise reduction)",
            "4. Otsu's thresholding (binarization)",
            "5. Morphological opening (noise removal)",
            "6. Center signature in canvas",
            f"7. Canvas size: {self.target_size}",
            "8. Normalize pixels to [0, 1]" if self.normalize else "8. Keep raw pixel values"
        ]


# ── Standalone test ─────────────────────────────────────────
if __name__ == "__main__":
    from config import RAW_DATA_DIR, CEDAR_GENUINE_DIR, CEDAR_FORGED_DIR

    preprocessor = SignaturePreprocessor()
    
    print("🔧 Preprocessing Pipeline Steps:")
    for step in preprocessor.get_pipeline_steps():
        print(f"   {step}")
    
    # Test with one genuine and one forged sample
    genuine_dir = RAW_DATA_DIR / CEDAR_GENUINE_DIR
    forged_dir = RAW_DATA_DIR / CEDAR_FORGED_DIR
    
    # Test genuine
    if genuine_dir.exists():
        test_files = sorted(list(genuine_dir.glob("*.png")))
        if test_files:
            test_path = test_files[0]
            print(f"\n🖼️  Testing genuine: {test_path.name}")
            
            processed = preprocessor.preprocess(test_path)
            if processed is not None:
                print(f"   ✓ Shape: {processed.shape}")
                print(f"   ✓ Range: [{processed.min():.3f}, {processed.max():.3f}]")
                print(f"   ✓ Dtype: {processed.dtype}")
                
                out_path = PROCESSED_DIR / "test_genuine.png"
                preprocessor.save_preprocessed(processed, out_path, as_uint8=True)
                print(f"   ✓ Saved preview: {out_path}")
    
    # Test forged
    if forged_dir.exists():
        test_files = sorted(list(forged_dir.glob("*.png")))
        if test_files:
            test_path = test_files[0]
            print(f"\n🖼️  Testing forged: {test_path.name}")
            
            processed = preprocessor.preprocess(test_path)
            if processed is not None:
                print(f"   ✓ Shape: {processed.shape}")
                print(f"   ✓ Range: [{processed.min():.3f}, {processed.max():.3f}]")
                print(f"   ✓ Dtype: {processed.dtype}")
                
                out_path = PROCESSED_DIR / "test_forged.png"
                preprocessor.save_preprocessed(processed, out_path, as_uint8=True)
                print(f"   ✓ Saved preview: {out_path}")