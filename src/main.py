from tracker import ObjectTracker
import cv2
from ultralytics import YOLO
import time  # For FPS calculation

# Initialize the object tracker
tracker = ObjectTracker()
frame_count = 0

# Load YOLO model
model = YOLO("yolov8n.pt")
model.to("cpu")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize video writer for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 360))

# Start time for FPS calculation
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))

    # Run YOLO detection
    results = model(frame)[0]

    # Prepare detections for tracker
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        detections.append({
            'bbox': (x1, y1, x2, y2),
            'label': label
        })
        # Draw boxes for detections
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Update the tracker with new detections
    tracked_objects = tracker.update(detections, frame_count)
    frame_count += 1

    # Draw tracked objects
    for obj_id, data in tracked_objects.items():
        x, y = data['centroid']
        label = data['label']
        cv2.putText(frame, f"ID {obj_id}: {label}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Calculate and display FPS
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Write the frame to output video
    out.write(frame)

    # Show the frame
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()