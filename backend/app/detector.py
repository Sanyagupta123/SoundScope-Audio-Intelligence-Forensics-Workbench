from functools import lru_cache
from .config import MODEL_NAME

@lru_cache(maxsize=1)
def get_model():
    from ultralytics import YOLO
    return YOLO(MODEL_NAME)

def detect(image, threshold=.25):
    try:
        result=get_model().predict(source=image,conf=threshold,verbose=False)[0]
        output=[]
        if result.boxes is None: return output
        for i,box in enumerate(result.boxes):
            x1,y1,x2,y2=[round(float(x),2) for x in box.xyxy[0].tolist()]
            cls=int(box.cls[0]); output.append({"id":f"object-{i+1}","label":result.names[cls],"confidence":round(float(box.conf[0]),4),"bbox":[x1,y1,x2,y2]})
        return output
    except Exception as exc:
        raise RuntimeError(f"Object detection failed: {exc}") from exc
