import os
import random
import shutil

image_train_dir = "juice_dataset/data/images/train"
label_train_dir = "juice_dataset/data/labels/train"

image_val_dir = "juice_dataset/data/images/val"
label_val_dir = "juice_dataset/data/labels/val"

os.makedirs(image_val_dir, exist_ok=True)
os.makedirs(label_val_dir, exist_ok=True)

images = [f for f in os.listdir(image_train_dir) if f.endswith('.jpg')]

random.shuffle(images)
val_size = int(len(images) * 0.2)
val_images = images[:val_size]

print(f"Total images: {len(images)}")
print(f"Val images: {len(val_images)}")

for img in val_images:
    shutil.move(os.path.join(image_train_dir, img), os.path.join(image_val_dir, img))
    label_file = img.replace('.jpg', '.txt')
    label_src = os.path.join(label_train_dir, label_file)
    label_dst = os.path.join(label_val_dir, label_file)

    if os.path.exists(label_src):
        shutil.move(label_src, label_dst)

print("Split complete ✅")
