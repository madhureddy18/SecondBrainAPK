import cv2
import time
from ultralytics import YOLO

# Load model once at startup
try:
    # Upgraded confidence threshold to 0.6 for robustness
    model = YOLO("yolo11n.pt") 
except Exception as e:
    print(f"Error loading YOLO: {e}")
    model = None

def get_vision_data():
    """
    Captures a high-quality frame, saves it, and returns detections.
    Includes a warm-up period to ensure the image isn't dark or blurry.
    """
    if model is None:
        return None, None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, None

    # Robustness: Discard first 5 frames to let the camera sensor auto-adjust
    for _ in range(5):
        cap.grab()
    
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, None

    image_path = "capture.jpg"
    cv2.imwrite(image_path, frame)

    # Robustness: Set 'conf=0.6' to ignore low-probability 'wrong' objects
    results = model(frame, conf=0.6, verbose=False)
    detections = {}

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            detections[label] = detections.get(label, 0) + 1
            
    return detections, image_path

def count_people():
    detections, _ = get_vision_data()
    if detections:
        return detections.get("person", 0)
    return 0