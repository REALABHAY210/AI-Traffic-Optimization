import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # or yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Open the webcam
cap = cv2.VideoCapture(0)  # Use 1 if you’re using an external cam

# Target class IDs (COCO dataset): car=2, motorcycle=3, bus=5, truck=7
vehicle_ids = [2, 3, 5, 7]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame, verbose=False)[0]

    # Reset car count per frame
    car_count = 0  # <-- Added

    # Loop through detections
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in vehicle_ids:
            conf = float(box.conf[0])
            label = f"{model.names[cls_id]} {conf:.2f}"
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Count cars
            if cls_id == 2:  # car
                car_count += 1  # <-- Added

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display car count on screen
    cv2.putText(frame, f"Cars detected: {car_count}", (10, 30),  # <-- Added
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("YOLOv8 Car Detection", frame)

    # ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
