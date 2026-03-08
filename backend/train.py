import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import json
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
import time

# -----------------------
# CONFIG  (CPU-optimised for 7h budget @ ~2 it/s)
# -----------------------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 32        # larger batch → fewer iterations, better grad estimates
EPOCHS      = 30        # more epochs to use the full 7 h window
LR          = 5e-4      # slightly higher start for OneCycleLR
DATA_DIR    = "data"
NUM_WORKERS = 4         # parallel data loading (set 0 if you get errors on Windows)
PIN_MEMORY  = False     # only useful on CUDA

# -----------------------
# AUGMENTATION
# -----------------------
train_transform = transforms.Compose([
    # ① Efficient crop – avoids redundant resize
    transforms.RandomResizedCrop(224, scale=(0.55, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    # ② TrivialAugmentWide = state-of-art single-policy augmentation, free
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    # ③ Random Erasing (Cutout) – effective against overfitting on leaf datasets
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────────────────────
#  IMPORTANT: On Windows, ALL code that uses multiprocessing
#  (including PyTorch DataLoader with num_workers > 0) MUST
#  live inside  `if __name__ == '__main__':`.
#  Without this guard, each spawned worker re-executes the
#  module-level code and crashes with a RuntimeError.
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # Boost CPU thread count for faster forward/backward passes
    torch.set_num_threads(os.cpu_count() or 4)
    torch.set_num_interop_threads(2)

    print(f"Using Device : {DEVICE}")
    print(f"CPU threads  : {torch.get_num_threads()}")

    # -----------------------
    # DATASET
    # -----------------------
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"),
                                         transform=train_transform)
    val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),
                                         transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    class_names  = train_dataset.classes
    num_classes  = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    with open("class_names.json", "w") as f:
        json.dump(class_names, f)

    # -----------------------
    # MODEL  — EfficientNet-B1
    #   Better accuracy than MobileNetV2 at only ~2× cost, still CPU-friendly.
    #   B0 is also fine if you want even more speed.
    # -----------------------
    model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)

    # Freeze stem + first 4 MBConv blocks, unfreeze the rest
    frozen_layers = list(model.features.children())[:5]
    for layer in frozen_layers:
        for p in layer.parameters():
            p.requires_grad = False

    # Improved classifier head with two FC layers
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.SiLU(),               # smooth activation used in EfficientNet body
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )

    model = model.to(DEVICE)

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    # -----------------------
    # CLASS WEIGHTS
    # -----------------------
    labels = train_dataset.targets
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # -----------------------
    # OPTIMISER — AdamW is strictly better than Adam for finetuning
    # -----------------------
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=2e-4,
    )

    # -----------------------
    # SCHEDULER — OneCycleLR
    #   Warms up LR then anneals; proven best scheduler for transfer learning.
    #   steps_per_epoch must match actual loader length.
    # -----------------------
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        pct_start=0.15,          # 15% of steps = warm-up
        anneal_strategy="cos",
        div_factor=10,           # start lr = max_lr / 10
        final_div_factor=100,    # end lr = start_lr / 100
    )

    # -----------------------
    # EARLY STOPPING
    # -----------------------
    PATIENCE    = 7      # stop if val_acc doesn't improve for 7 epochs
    patience_ct = 0
    best_acc    = 0.0

    # -----------------------
    # TRAINING LOOP
    # -----------------------
    print("\n" + "="*52)
    print(f"  Training EfficientNet-B1  |  {EPOCHS} epochs  |  {DEVICE}")
    print("="*52 + "\n")

    for epoch in range(EPOCHS):

        start_time = time.time()

        # -------- TRAIN --------
        model.train()
        train_correct = train_total = 0
        train_loss = 0.0

        train_bar = tqdm(train_loader,
                         desc=f"Epoch {epoch+1:>2}/{EPOCHS} [Train]",
                         ncols=90)

        for images, batch_labels in train_bar:
            images, batch_labels = images.to(DEVICE), batch_labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, batch_labels)
            loss.backward()

            # Gradient clipping — prevents occasional loss spikes on CPU
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()        # OneCycleLR steps every BATCH

            train_loss += loss.item()
            _, predicted  = torch.max(outputs, 1)
            train_total  += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100*train_correct/train_total:.1f}%",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        train_acc      = 100 * train_correct / train_total
        avg_train_loss = train_loss / steps_per_epoch

        # -------- VALIDATION --------
        model.eval()
        val_correct = val_total = 0
        val_loss_total = 0.0

        val_bar = tqdm(val_loader,
                       desc=f"Epoch {epoch+1:>2}/{EPOCHS} [Val  ]",
                       ncols=90)

        with torch.no_grad():
            for images, batch_labels in val_bar:
                images, batch_labels = images.to(DEVICE), batch_labels.to(DEVICE)
                outputs = model(images)
                v_loss  = criterion(outputs, batch_labels)

                val_loss_total += v_loss.item()
                _, predicted  = torch.max(outputs, 1)
                val_total    += batch_labels.size(0)
                val_correct  += (predicted == batch_labels).sum().item()

                val_bar.set_postfix(acc=f"{100*val_correct/val_total:.1f}%")

        val_acc      = 100 * val_correct / val_total
        avg_val_loss = val_loss_total / len(val_loader)
        epoch_time   = time.time() - start_time

        print(f"\n{'─'*52}")
        print(f"  Epoch {epoch+1}/{EPOCHS}  |  {epoch_time:.0f}s")
        print(f"  Train  → loss={avg_train_loss:.4f}  acc={train_acc:.2f}%")
        print(f"  Val    → loss={avg_val_loss:.4f}  acc={val_acc:.2f}%")
        print(f"{'─'*52}")

        # ---- Save best ----
        if val_acc > best_acc:
            best_acc    = val_acc
            patience_ct = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✅  New best saved  ({best_acc:.2f}%)")
        else:
            patience_ct += 1
            print(f"  ⏳  No improvement  ({patience_ct}/{PATIENCE})")

        # ---- Early stop ----
        if patience_ct >= PATIENCE:
            print(f"\n🛑  Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"\n{'='*52}")
    print(f"  Training Complete 🔥")
    print(f"  Best Validation Accuracy : {best_acc:.2f}%")
    print(f"{'='*52}\n")