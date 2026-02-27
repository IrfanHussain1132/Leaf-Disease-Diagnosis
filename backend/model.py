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

# Load model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def predict(image: Image.Image):
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return {
        "disease": class_names[predicted.item()],
        "confidence": round(confidence.item() * 100, 2)
    }
