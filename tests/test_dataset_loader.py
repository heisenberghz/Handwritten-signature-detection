import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_loader import CEDARDatasetLoader


def test_loader_initialization():
    loader = CEDARDatasetLoader()
    assert loader.samples == []
    assert loader.persons == set()
    assert loader.df is None
    assert loader.data_dir is not None


def test_loader_discover_nonexistent_dir():
    loader = CEDARDatasetLoader(data_dir=Path("/nonexistent/path"))
    try:
        loader.discover()
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
