🌿 AgriAI – AI Powered Plant Disease Detection

AgriAI is a full-stack AI system for plant disease detection using Deep Learning techniques and treatment recommendation using an LLM.

The system uses Computer Vision + LLM + Web API to assist farmers in plant disease detection and treatment.

🚀 Features

✔ Detection of plant diseases from images of leaves
✔ High accuracy CNN model with ~98.9% validation accuracy
✔ LLM-based treatment recommendation
✔ Organic and chemical treatment options
✔ Prevention tips for farmers
✔ Real-time API prediction
✔ Web interface for uploading images

🧠 Model (Level 2)

The Level-2 model has been improved for better generalization and real-world performance.

**Architecture**

**Model:** MobileNetV2 (Transfer Learning)
**Input Size:** 224 x 224
**Framework:** PyTorch

**Training Improvements**

Improvements to the Level 2 model:

Stronger image augmentation
Cosine learning rate scheduler
Label smoothing
Dropout regularization
Weight decay
Balanced class weights
Fine-tuning deeper layers

**Final Performance**
**Metric**  **Score**
**Training Accuracy**  98.63%
**Validation Accuracy**  98.97%
