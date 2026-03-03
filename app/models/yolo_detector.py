import random

class YOLODetector:

    def detect(self, frame):
        # Placeholder detections
        return {
            "player_bbox": [100, 100, 300, 500],
            "ball_bbox": [250, 200, 270, 220],
            "court_lines": [(0,0), (640,0), (640,480), (0,480)]
        }