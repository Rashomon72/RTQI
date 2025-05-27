import cv2
import numpy as np
from ultralytics import YOLO

def pothole_detection(video_path, model_path, confidence_threshold=0.5, threshold=30):
    """
    Counts unique objects in a video using YOLO model detections.

    Args:
        video_path (str): Path to the video file.
        model_path (str): Path to the trained YOLO model.
        classes_to_count (list of str): List of classes to count.
        confidence_threshold (float): Confidence threshold for detections.
        threshold (int): Pixel distance threshold to consider detections as the same object.

    Returns:
        dict: Dictionary with counts of unique objects for each specified class.
    """
    # Load the YOLO model
    model = YOLO("C:/Users/mohammad asfraf/OneDrive/Desktop/BTP-Backend/server/weights/Potholes.pt")
    classes_to_count = ['Drain Hole', 'Pothole', 'Sewer Cover']

    # Initialize counts and tracking
    class_counts = {cls: 0 for cls in classes_to_count}
    tracked_objects = []

    # Helper function to check if two bounding boxes are close enough to be considered the same object
    def is_nearby(box1, box2, threshold=30):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        center1 = ((x1 + w1) / 2, (y1 + h1) / 2)
        center2 = ((x2 + w2) / 2, (y2 + h2) / 2)
        return np.linalg.norm(np.array(center1) - np.array(center2)) < threshold

    # Open the video
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the frame
        results = model(frame)

        # Process detections
        for det in results[0].boxes:  # Adjust based on the structure of results
            confidence = float(det.conf[0])
            class_id = int(det.cls[0])
            class_name = model.names[class_id]
            
            # Only consider detections with sufficient confidence and specified class names
            if class_name in classes_to_count and confidence >= confidence_threshold:
                x1, y1, x2, y2 = map(int, det.xywh[0])
                bbox = [x1, y1, x2, y2]

                # Check if the object is already tracked
                is_tracked = False
                for obj in tracked_objects:
                    if obj['class_name'] == class_name and is_nearby(obj['bbox'], bbox, threshold):
                        is_tracked = True
                        obj['bbox'] = bbox  # Update position for continuity
                        break

                # Count new objects and track them
                if not is_tracked:
                    class_counts[class_name] += 1
                    tracked_objects.append({'class_name': class_name, 'bbox': bbox})

    # Release video capture and return counts
    cap.release()
    return class_counts

