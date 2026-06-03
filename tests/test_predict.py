import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.predict import SignaturePredictor


def test_predictor_initialization_without_model():
    try:
        SignaturePredictor(model_path=Path("nonexistent.pth"))
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


def test_preprocess_image_creates_tensor(tmp_path):
    import cv2
    img_path = tmp_path / "test.png"
    cv2.imwrite(str(img_path), np.ones((100, 100), dtype=np.uint8) * 200)

    try:
        predictor = SignaturePredictor()
    except FileNotFoundError:
        return

    from torch import Tensor
    tensor = predictor.preprocess_image(img_path)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == (1, 1, 256, 256)
