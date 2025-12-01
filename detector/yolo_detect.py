from ultralytics import YOLO
import numpy as np

class CardDetector:
    def __init__(self,model_path):
        print(" YOLO 모델 로드 중:", model_path)
        self.model = YOLO(model_path)
        print(" 모델 로드 완료. 클래스 목록:", self.model.names)
    def detect_card(self,frame):
        results = self.model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []
        #print(" 감지된 박스 수:", len(boxes))
        if len(boxes) == 0:
            return None
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        idx = int(np.argmax(areas))
        return boxes[idx]