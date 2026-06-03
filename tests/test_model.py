import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model import SignatureCNN


def test_model_initialization():
    model = SignatureCNN(num_classes=2, dropout=0.5)
    assert model.count_parameters() > 0


def test_model_output_shape():
    model = SignatureCNN(num_classes=2, dropout=0.5)
    model.eval()
    batch_size = 4
    dummy = torch.randn(batch_size, 1, 256, 256)
    with torch.no_grad():
        output = model(dummy)
    assert output.shape == (batch_size, 2)


def test_model_logits():
    model = SignatureCNN(num_classes=2, dropout=0.5)
    model.eval()
    dummy = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        output = model(dummy)
    probs = torch.softmax(output, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]))


def test_model_dropout_config():
    model_high = SignatureCNN(num_classes=2, dropout=0.8)
    model_low = SignatureCNN(num_classes=2, dropout=0.1)
    assert model_high.dropout.p == 0.8
    assert model_low.dropout.p == 0.1


def test_model_train_mode():
    model = SignatureCNN(num_classes=2, dropout=0.5)
    model.train()
    assert model.training is True


def test_model_eval_mode():
    model = SignatureCNN(num_classes=2, dropout=0.5)
    model.eval()
    assert model.training is False
