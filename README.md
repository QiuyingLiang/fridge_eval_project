# Fridge Product Detection with YOLOv8, yolov10

This project focuses on detecting and classifying beverage products inside a fridge using YOLOv8, yolov10, with a primary goal of identifying **Maidong (脉动)** vs **other products**, and supporting downstream purity analysis.

## Project Goals

- Detect products inside fridge scenes
- Classify:
  -  Maidong (脉动)
  -  Other products
  -  Maidong fridge region
- Build a scalable annotation pipeline
- Reduce manual labeling effort using auto-labeling
- Enable iterative model improvement

 ---
##  Workflow

### 1️ Manual Annotation
- Annotate ~100 images using `labelImg`
- Labels stored in `data/labels`

---

### 2️ Create Train/Val Split

```bash
python split_dataset.py
```

### 3 Train initial Model for auto-labelling
```bash
yolo detect train data=dataset.yaml model=yolov8n.pt epochs=80 imgsz=768
```
- the initial model was trained using 100 manually labeled images and serves as a bootstrap model for auto-labeling the remaining dataset.
- Model: YOLOv8n
- Training epochs: ~80
- Input size:768
- Output checkpoint: runs/detect/train-6/weights/best.pt

![](runs/detect/train-6/confusion_matrix_normalized.png)

### Observations:
- The initial model is able to distinguish general object regions
- classification between maidong and other is still imperfect
- misclassification are expected due to limited training data
- performance will improve after iterative retraining

### Performance:
![](img_0118.jpg)
![](img_0221.jpg)
![](img_0498.jpg)
### 4 Auto label remaining 400 images
```bash
yolo detect predict model=runs/detect/train6/weights/best.pt source=data/images conf=0.01 save_txt=True save=True
```

- Low conf ensures maximum box recall, with conf=0.01
- Accuracy is not critical at this stage

### 5 Manual correction
- correct the auto-labelled images


### 6 retrain model with yolov10
