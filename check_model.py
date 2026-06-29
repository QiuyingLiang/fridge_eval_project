from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

# validate model
results = model.val(data='juice_dataset/juice_dataset.yaml')
print(results)
