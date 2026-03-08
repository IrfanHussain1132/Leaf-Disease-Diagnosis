import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)

# -----------------------
# MODEL — must match train.py architecture exactly
# -----------------------
model = models.efficientnet_b1(weights=None)

in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features, 512),
    nn.SiLU(),
    nn.Dropout(p=0.2),
    nn.Linear(512, num_classes),
)

model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# -----------------------
# Inference transform (no augmentation, centre-cropped)
# -----------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def predict(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs      = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top3_conf, top3_pred = torch.topk(probabilities, 3)

    top3_conf = top3_conf.squeeze().cpu().numpy()
    top3_pred = top3_pred.squeeze().cpu().numpy()

    top_predictions = [
        {
            "disease":    class_names[top3_pred[i]],
            "confidence": round(float(top3_conf[i] * 100), 2),
        }
        for i in range(3)
    ]

    return {
        "disease":         top_predictions[0]["disease"],
        "confidence":      top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
    }
