import os

image_dir = "juice_dataset/data/images/unlabeled_new"
label_dir = "juice_dataset/data/labels/fine_tuning_labels"

prefix = "new_"

images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

for img in images:
    name = os.path.splitext(img)[0]
    
    old_img_path = os.path.join(image_dir, img)
    old_label_path = os.path.join(label_dir, f"{name}.txt")

    new_img_name = prefix + img
    new_label_name = prefix + f"{name}.txt"

    new_img_path = os.path.join(image_dir, new_img_name)
    new_label_path = os.path.join(label_dir, new_label_name)

    os.rename(old_img_path, new_img_path)

    if os.path.exists(old_label_path):
        os.rename(old_label_path, new_label_path)

print("✅ 重命名完成")