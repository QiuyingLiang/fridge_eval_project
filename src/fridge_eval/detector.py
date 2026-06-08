from ultralytics import YOLO

class Detector:
    def __init__(self,model_path,class_names):
        self.model=YOLO(model_path)
        self.class_names=class_names

    def infer_batch(self,images):
        return self.model(images,verbose=False)

    def analyze(self,result):
        if result.boxes is None:
            return None

        boxes=result.boxes

        fridge=[b for b in boxes if self.class_names[int(b.cls[0])]=='maidong_fridge']
        if not fridge:
            return None

        f=max(fridge,key=lambda b:(b.xyxy[0][3]-b.xyxy[0][1]))
        x1,y1,x2,y2=f.xyxy[0]

        items=[]
        for b in boxes:
            name=self.class_names[int(b.cls[0])]
            conf=float(b.conf[0])
            if name not in ['maidong','other']:
                continue
            bx1,by1,bx2,by2=b.xyxy[0]
            cx,cy=(bx1+bx2)/2,(by1+by2)/2
            if x1<=cx<=x2 and y1<=cy<=y2:
                items.append((cy,name,conf))

        if not items:
            return None

        items=sorted(items,key=lambda x:x[0])

        layers=[]
        cur=[items[0]]
        TH=50
        for i in range(1,len(items)):
            if abs(items[i][0]-items[i-1][0])<TH:
                cur.append(items[i])
            else:
                layers.append(cur)
                cur=[items[i]]
        layers.append(cur)

        return layers
