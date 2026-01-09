import cv2
from ultralytics import YOLO

# Load model once at startup to save time
try:
    model = YOLO("yolov8n.pt")
except:
    model = None

def get_vision_data():
    """
    Captures a frame from the camera, saves it as 'capture.jpg',
    and returns a dictionary of detected objects and the image path.
    """
    if model is None:
        return None, None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, None

    # Save image for LLM scene analysis (Llama 3.2 Vision)
    image_path = "capture.jpg"
    cv2.imwrite(image_path, frame)

    # Run YOLO for local object counting (people, bottles, etc.)
    results = model(frame, verbose=False)
    detections = {}

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            detections[label] = detections.get(label, 0) + 1
            
    return detections, image_path

def count_people():
    """Legacy function support: returns just the person count."""
    detections, _ = get_vision_data()
    if detections:
        return detections.get("person", 0)
    return 0