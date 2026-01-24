import cv2
from ultralytics import YOLO

MODEL_PATH = "models/yolov8l.pt"
_model = YOLO(MODEL_PATH)


def detect_people(frame):
    results = _model.predict(frame, conf=0.4, classes=[0], verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "crop": crop
            })

    return detections
