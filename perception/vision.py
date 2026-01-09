import cv2
from ultralytics import YOLO

# Load model once at startup to save time
try:
    model = YOLO("yolov8n.pt")
except:
    model = None

def count_people():
    if model is None:
        return None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    results = model(frame, verbose=False)
    count = 0

    for r in results:
        for box in r.boxes:
            # check the class name for 'person'
            if model.names[int(box.cls[0])] == "person":
                count += 1
    return count