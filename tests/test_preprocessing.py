import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.preprocessing import SignaturePreprocessor


def test_preprocessor_initialization():
    p = SignaturePreprocessor()
    assert p.target_size == (256, 256)
    assert p.grayscale is True
    assert p.normalize is True


def test_grayscale_conversion():
    p = SignaturePreprocessor()
    color_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    gray = p.to_grayscale(color_img)
    assert len(gray.shape) == 2


def test_grayscale_passthrough():
    p = SignaturePreprocessor()
    gray_input = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    result = p.to_grayscale(gray_input)
    assert result.shape == (100, 100)


def test_denoise_output_shape():
    p = SignaturePreprocessor()
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    denoised = p.denoise(img)
    assert denoised.shape == (100, 100)


def test_binarize_output():
    p = SignaturePreprocessor()
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    binary = p.binarize(img)
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 255})


def test_center_signature_empty():
    p = SignaturePreprocessor()
    empty = np.zeros((100, 100), dtype=np.uint8)
    result = p.center_signature(empty)
    assert result.shape == p.target_size


def test_center_signature_with_content():
    p = SignaturePreprocessor()
    img = np.zeros((100, 100), dtype=np.uint8)
    img[30:70, 30:70] = 255
    result = p.center_signature(img)
    assert result.shape == p.target_size
    assert result.sum() > 0


def test_normalize_pixels():
    p = SignaturePreprocessor()
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    normalized = p.normalize_pixels(img)
    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_save_preprocessed_uint8(tmp_path):
    p = SignaturePreprocessor()
    img = np.random.rand(100, 100).astype(np.float32)
    out = tmp_path / "test.png"
    p.save_preprocessed(img, out, as_uint8=True)
    assert out.exists()


def test_save_preprocessed_float(tmp_path):
    p = SignaturePreprocessor()
    img = np.random.rand(100, 100).astype(np.float32)
    out = tmp_path / "test_float.png"
    p.save_preprocessed(img, out, as_uint8=False)
    assert out.exists()


def test_ensure_odd():
    assert SignaturePreprocessor._ensure_odd((4, 4)) == (5, 5)
    assert SignaturePreprocessor._ensure_odd((5, 5)) == (5, 5)
    assert SignaturePreprocessor._ensure_odd((4, 5)) == (5, 5)


def test_get_pipeline_steps():
    p = SignaturePreprocessor()
    steps = p.get_pipeline_steps()
    assert len(steps) == 8
