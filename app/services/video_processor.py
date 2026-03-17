import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except ImportError:
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing
import torch

class ObjectDetectionService:
    def __init__(self, model_name="yolov8n.pt"):
        # Usamos YOLOv8 nano para eficiencia en el MVP
        # El modelo se descargará automáticamente la primera vez
        self.model = YOLO(model_name)
    
    def detect_objects(self, frame):
        # Clases de interés en COCO (yolov8 pre-entrenado):
        # 0: person, 32: tie (incorrecto), 34: frisbee, 36: skateboard, 
        # 37: surfboard, 38: tennis racket, 39: bottle, 40: wine glass, 
        # 41: cup, 42: fork, 43: knife, 44: spoon, 45: bowl, 
        # 32: sports ball (Pelota de tenis suele ser esta)
        results = self.model(frame, verbose=False)
        return results

class PoseEstimationService:
    def __init__(self):
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
    
    def estimate_pose(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        return results

    def draw_pose(self, frame, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        return frame

class VideoProcessor:
    def __init__(self):
        self.detector = ObjectDetectionService()
        self.pose_estimator = PoseEstimationService()
        
    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"No se pudo abrir el video: {video_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Escribir video de salida con anotaciones
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frames_data = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 1. Detección de objetos
            detections = self.detector.detect_objects(frame)
            
            # 2. Estimación de Pose
            pose_results = self.pose_estimator.estimate_pose(frame)
            
            # 3. Dibujar resultados en el frame
            # Dibujar detecciones de YOLO (personas y pelotas)
            if detections and len(detections) > 0:
                for box in detections[0].boxes:
                    # Obtener coordenadas y clase
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Dibujar solo personas (0) y pelotas (32) o raquetas (38)
                    if cls in [0, 32, 38]:
                        label = f"{self.detector.model.names[cls]} {conf:.2f}"
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Dibujar esqueleto de MediaPipe
            if pose_results and hasattr(pose_results, 'pose_landmarks') and pose_results.pose_landmarks:
                self.pose_estimator.draw_pose(frame, pose_results)
            
            # Escribir el frame anotado
            out.write(frame)

            # Recolectar datos por frame
            pose_detected = False
            if pose_results and hasattr(pose_results, 'pose_landmarks') and pose_results.pose_landmarks:
                pose_detected = True

            frames_data.append({
                "objects": len(detections[0].boxes) if detections and len(detections) > 0 else 0,
                "pose_detected": pose_detected
            })
            
        cap.release()
        out.release()
        
        return {
            "total_frames": len(frames_data),
            "output_video": output_path,
            "summary": "Procesamiento completado exitosamente"
        }
