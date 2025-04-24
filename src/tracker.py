import time
import numpy as np

class ObjectTracker:
    def __init__(self, max_missing_frames=30):
        self.next_id = 0
        self.objects = {}  # {id: {'centroid': (x, y), 'last_seen': frame_id, 'label': str}}
        self.max_missing_frames = max_missing_frames

    def update(self, detections, current_frame):
        updated_ids = []

        for box in detections:
            x1, y1, x2, y2 = box['bbox']
            label = box['label']
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

            matched_id = None
            for obj_id, info in self.objects.items():
                dist = np.linalg.norm(np.array(info['centroid']) - np.array(centroid))
                if dist < 50 and label == info['label']:
                    matched_id = obj_id
                    break

            if matched_id is not None:
                self.objects[matched_id] = {
                    'centroid': centroid,
                    'last_seen': current_frame,
                    'label': label
                }
                updated_ids.append(matched_id)
            else:
                self.objects[self.next_id] = {
                    'centroid': centroid,
                    'last_seen': current_frame,
                    'label': label
                }
                print(f"[NEW] Object {self.next_id} added: {label}")
                updated_ids.append(self.next_id)
                self.next_id += 1

        # Check for missing objects
        for obj_id in list(self.objects.keys()):
            if obj_id not in updated_ids:
                if current_frame - self.objects[obj_id]['last_seen'] > self.max_missing_frames:
                    print(f"[MISSING] Object {obj_id} disappeared: {self.objects[obj_id]['label']}")
                    del self.objects[obj_id]

        return self.objects
