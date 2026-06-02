"""
src/model.py - Smaller CNN architecture to reduce overfitting
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from config import IMAGE_SIZE


class SignatureCNN(nn.Module):
    """
    Smaller CNN to prevent overfitting on small dataset.
    ~4M parameters instead of 34M.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super(SignatureCNN, self).__init__()
        
        # Smaller channels: 16 → 32 → 64 → 128
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16, momentum=0.1)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32, momentum=0.1)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64, momentum=0.1)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128, momentum=0.1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        # 256 → 128 → 64 → 32 → 16
        self.flattened_size = 128 * 16 * 16
        
        # Smaller FC layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Standalone test ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("🔍 Testing SignatureCNN (Smaller)")
    print("=" * 55)
    
    model = SignatureCNN(num_classes=2, dropout=0.5)
    
    total_params = model.count_parameters()
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 256, 256)
    
    print(f"\n🧪 Forward pass test:")
    print(f"   Input shape:  {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Output (logits):\n{output}")
    
    probs = F.softmax(output, dim=1)
    print(f"\n   Probabilities (genuine, forged):")
    for i in range(batch_size):
        print(f"      Sample {i}: [{probs[i][0]:.4f}, {probs[i][1]:.4f}]")
    
    print("\n✅ Model test passed!")