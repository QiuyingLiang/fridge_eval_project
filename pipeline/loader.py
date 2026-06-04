import requests
import numpy as np
import cv2
from config import TIMEOUT

def load_image(url):
    try:
        r = requests.get(url, timeout=TIMEOUT)
        img_array = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except:
        return None
