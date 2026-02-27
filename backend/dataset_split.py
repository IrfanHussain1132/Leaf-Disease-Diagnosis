import os
import shutil
import random

SOURCE_DIR = "color"
DEST_DIR = "data"
SPLIT_RATIO = 0.8  # 80% train, 20% val

os.makedirs(os.path.join(DEST_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(DEST_DIR, "val"), exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        random.shuffle(images)

        split_index = int(len(images) * SPLIT_RATIO)

        train_images = images[:split_index]
        val_images = images[split_index:]

        os.makedirs(os.path.join(DEST_DIR, "train", class_name), exist_ok=True)
        os.makedirs(os.path.join(DEST_DIR, "val", class_name), exist_ok=True)

        for img in train_images:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(DEST_DIR, "train", class_name, img)
            )

        for img in val_images:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(DEST_DIR, "val", class_name, img)
            )

print("Dataset split completed ✅")
