import os
import shutil
import random
from pathlib import Path

# 原始路径
image_dir = Path("data/images")
label_dir = Path("data/labels")

# 输出路径
out_base = Path("dataset")
train_img = out_base / "images/train"
val_img = out_base / "images/val"
train_lbl = out_base / "labels/train"
val_lbl = out_base / "labels/val"

# 创建目录
for d in [train_img, val_img, train_lbl, val_lbl]:
    d.mkdir(parents=True, exist_ok=True)

# 找所有有标注的文件 - 排除 classes.txt
label_files = [f for f in label_dir.glob("*.txt") if f.name != "classes.txt"]
names = [f.stem for f in label_files]

print(f"✅ 找到有标注图片: {len(names)}张")

# 打乱
random.shuffle(names)

# 划分 80% train, 20% val
split = int(len(names) * 0.8)
train_names = names[:split]
val_names = names[split:]

print(f" 训练集: {len(train_names)}张")
print(f" 验证集: {len(val_names)}张")

def copy_data(name_list, img_out, lbl_out):
    for name in name_list:
        # 图片可能是jpg或png
        for ext in [".jpg", ".png", ".jpeg"]:
            img_path = image_dir / f"{name}{ext}"
            if img_path.exists():
                shutil.copy(img_path, img_out / img_path.name)
                break
        
        lbl_path = label_dir / f"{name}.txt"
        if lbl_path.exists():
            shutil.copy(lbl_path, lbl_out / lbl_path.name)

# 执行复制
copy_data(train_names, train_img, train_lbl)
copy_data(val_names, val_img, val_lbl)

print("✅ train/val 划分完成")