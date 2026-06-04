from ultralytics import YOLO
from config import MODEL_PATH, CONF_THRESHOLD
import cv2
import numpy as np

class Detector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH if MODEL_PATH else 'yolov8n.pt')
        
    def infer(self, img):
        """只返回计数（保持向后兼容）"""
        results = self.model(img, conf=CONF_THRESHOLD)
        bottles, fridges = 0, 0
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:
                    fridges += 1
                elif cls_id == 1:
                    bottles += 1
        return fridges, bottles
    
    def detect_bottles_with_position(self, img):
        """
        检测所有瓶子的位置和品牌
        返回: [{'brand': '脉动', 'y': y_position}, ...]
        """
        results = self.model(img, conf=CONF_THRESHOLD)
        bottles = []
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 1:  # 瓶子类别
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    y_center = (y1 + y2) / 2
                    
                    # 识别品牌
                    brand = self.identify_brand(img, int(x1), int(y1), int(x2), int(y2))
                    
                    bottles.append({
                        'brand': brand,
                        'y': y_center
                    })
        
        return bottles
    
    def identify_brand(self, img, x1, y1, x2, y2):
        """临时方案：默认所有瓶子都是脉动"""
        return "脉动"  # 直接返回脉动，不做实际检测
    