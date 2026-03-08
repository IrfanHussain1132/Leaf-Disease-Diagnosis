import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR   = "data"
MODEL_PATH = "best_model.pth"

# Transform (same as val_transform in train.py)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Load dataset
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),
                                   transform=transform)
val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False,
                         num_workers=4, persistent_workers=True)

class_names = val_dataset.classes
num_classes = len(class_names)

# Load model — must match train.py architecture
model = models.efficientnet_b1(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features, 512),
    nn.SiLU(),
    nn.Dropout(p=0.2),
    nn.Linear(512, num_classes),
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ---- Report ----
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ---- Confusion Matrix ----
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=(num_classes <= 20), fmt="d",
            xticklabels=class_names, yticklabels=class_names,
            cmap="Blues", linewidths=0.3)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved → confusion_matrix.png")
