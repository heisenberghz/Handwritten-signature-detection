"""
src/train_cnn.py - CNN training with data augmentation and better regularization
"""

import sys
import time
import json
import importlib.util
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# Load modules directly from files
project_root = Path(__file__).parent.parent.resolve()

# Load config
config_path = project_root / "config.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

SPLIT_DIR = config.SPLIT_DIR
CHECKPOINT_DIR = config.CHECKPOINT_DIR
LOG_DIR = config.LOG_DIR
IMAGE_SIZE = config.IMAGE_SIZE
RANDOM_SEED = config.RANDOM_SEED
BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE
WEIGHT_DECAY = config.WEIGHT_DECAY
NUM_EPOCHS = config.NUM_EPOCHS
PATIENCE = config.PATIENCE

# Load model
model_path = project_root / "src" / "model.py"
spec = importlib.util.spec_from_file_location("model", model_path)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
SignatureCNN = model_module.SignatureCNN

# Load dataset
dataset_path = project_root / "src" / "signature_dataset.py"
spec = importlib.util.spec_from_file_location("signature_dataset", dataset_path)
dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_module)
SignatureDataset = dataset_module.SignatureDataset


def _format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def set_seed(seed: int = RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AugmentedSignatureDataset(torch.utils.data.Dataset):
    """Wraps SignatureDataset with data augmentation for training."""
    
    def __init__(self, base_dataset, augment=True):
        self.base_dataset = base_dataset
        self.augment = augment
        
        # Augmentation: small rotations and scaling (signatures are sensitive)
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-5, 5)),
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        ])
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        if self.augment:
            # image is [1, 256, 256], need to add batch dim for transforms
            img = image.unsqueeze(0)  # [1, 1, 256, 256]
            img = self.transform(img)
            image = img.squeeze(0)  # [1, 256, 256]
        
        return image, label


def train_epoch(model, dataloader, criterion, optimizer, device, epoch: int, total_epochs: int) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"   Epoch {epoch:02d}/{total_epochs} [Train]", unit="batch", ncols=80)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        current_acc = correct / total
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.4f}'
        })
    
    pbar.close()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return {"loss": epoch_loss, "accuracy": epoch_acc}


def validate(model, dataloader, criterion, device, epoch: int, total_epochs: int) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"   Epoch {epoch:02d}/{total_epochs} [Val  ]", unit="batch", ncols=80)
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            current_acc = correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
    
    pbar.close()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return {"loss": epoch_loss, "accuracy": epoch_acc}


def train():
    total_start = time.time()
    
    print("=" * 70)
    print("🧠 CNN Training v2 — Smaller Model + Augmentation")
    print("=" * 70)
    
    set_seed()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # ── Datasets ──────────────────────────────────────────
    print("\n📂 Loading datasets...")
    load_start = time.time()
    
    train_base = SignatureDataset(split="train")
    val_dataset = SignatureDataset(split="val")
    test_dataset = SignatureDataset(split="test")
    
    # Wrap train with augmentation
    train_dataset = AugmentedSignatureDataset(train_base, augment=True)
    
    load_time = time.time() - load_start
    print(f"   ⏱️  Dataset loading time: {_format_time(load_time)}")
    
    # ── DataLoaders ───────────────────────────────────────
    num_workers = 0
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True if device.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")
    
    # ── Model ─────────────────────────────────────────────
    print("\n🏗️  Building model...")
    model = SignatureCNN(num_classes=2, dropout=0.5).to(device)
    print(f"   Parameters: {model.count_parameters():,}")
    
    # ── Loss & Optimizer ──────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3  # Faster LR reduction
    )
    
    # ── Training Loop ─────────────────────────────────────
    best_val_acc = 0.0
    patience_counter = 0
    history: List[Dict] = []
    
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = CHECKPOINT_DIR / "cnn_best_v2.pth"
    
    print(f"\n🚀 Training for up to {NUM_EPOCHS} epochs...")
    print("-" * 70)
    
    train_start = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS)
        val_metrics = validate(model, val_loader, criterion, device, epoch, NUM_EPOCHS)
        
        scheduler.step(val_metrics["loss"])
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n   📊 Epoch {epoch:02d} Summary:")
        print(f"      Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"      Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.4f}")
        print(f"      ⏱️  Epoch time: {_format_time(epoch_time)}")
        
        history.append({
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "epoch_time_seconds": round(epoch_time, 2),
            "learning_rate": round(optimizer.param_groups[0]["lr"], 8),
        })
        
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "train_accuracy": train_metrics["accuracy"],
            }, best_model_path)
            print(f"      💾 New best model! (Val Acc: {best_val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"      Patience: {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping triggered after {epoch} epochs")
            break
    
    train_time = time.time() - train_start
    print(f"\n⏱️  Total training time: {_format_time(train_time)}")
    
    log_file = LOG_DIR / "training_history.json"
    with open(log_file, "w") as f:
        json.dump({
            "model": "SignatureCNN v2",
            "hyperparameters": {
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "num_epochs": NUM_EPOCHS,
                "patience": PATIENCE,
                "dropout": 0.5,
                "image_size": list(IMAGE_SIZE),
            },
            "best_val_accuracy": best_val_acc,
            "total_training_time_seconds": round(train_time, 2),
            "history": history,
        }, f, indent=2)
    print(f"   📝 Training log saved: {log_file}")
    
    # ── Test Evaluation ───────────────────────────────────
    print("\n" + "=" * 70)
    print("🧪 Test Set Evaluation")
    print("=" * 70)
    
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\n📂 Loaded best model from epoch {checkpoint['epoch']} (Val Acc: {checkpoint['val_accuracy']:.4f})")
    
    test_start = time.time()
    test_metrics = validate(model, test_loader, criterion, device, 0, 0)
    test_time = time.time() - test_start
    
    print(f"\n📊 Test Results:")
    print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"   Loss:      {test_metrics['loss']:.4f}")
    print(f"   ⏱️  Test inference time: {_format_time(test_time)}")
    
    # Detailed evaluation
    print("\n📋 Detailed Test Metrics:")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="   Computing metrics", unit="batch", ncols=70):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score
    )
    
    print(f"\n   Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
    print(f"   Precision: {precision_score(all_labels, all_preds):.4f}")
    print(f"   Recall:    {recall_score(all_labels, all_preds):.4f}")
    print(f"   F1-Score:  {f1_score(all_labels, all_preds):.4f}")
    print(f"   ROC-AUC:   {roc_auc_score(all_labels, all_probs):.4f}")
    print(f"\n   Confusion Matrix:\n{confusion_matrix(all_labels, all_preds)}")
    
    # ── Total Time ────────────────────────────────────────
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("✅ CNN training v2 complete!")
    print(f"   Best val accuracy: {best_val_acc:.4f}")
    print(f"   Total execution time: {_format_time(total_time)}")
    print(f"   Best model saved: {best_model_path}")
    print("=" * 70)


if __name__ == "__main__":
    train()