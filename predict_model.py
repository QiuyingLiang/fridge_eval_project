from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

# predict on validation images
results = model.predict(source='juice_dataset/data/images/val', save=True, conf=0.25)
print('Prediction complete. Output saved in runs/detect/predict/')
