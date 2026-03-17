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
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import os

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

    def calculate_angle(self, a, b, c):
        """Calcula el ángulo entre tres puntos (a, b, c). b es el vértice."""
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

class ShotClassificationService:
    def __init__(self):
        # En un MVP real, aquí cargaríamos un modelo de clasificación (ej. LSTM o Video Transformer)
        # Para esta versión, usaremos una lógica heurística basada en la trayectoria de los keypoints
        pass

    def classify_shot(self, pose_history):
        """
        pose_history: lista de diccionarios con landmarks por frame
        Devuelve el tipo de golpe detectado: "Forehand", "Backhand", "Serve", "None"
        """
        if len(pose_history) < 10:
            return "None"
        
        # Ejemplo de lógica simplificada:
        # Si la mano derecha sube por encima de la cabeza -> Serve
        # Si la mano derecha cruza el cuerpo hacia la izquierda -> Backhand (si es diestro)
        # Si la mano derecha se aleja del cuerpo y luego impacta -> Forehand
        
        # Tomamos los últimos frames para analizar el movimiento
        hand_y = [p.get('RIGHT_WRIST', {}).get('y', 1.0) for p in pose_history]
        head_y = [p.get('NOSE', {}).get('y', 0.0) for p in pose_history]
        
        if min(hand_y) < min(head_y):
             return "Serve"
             
        # Lógica de desplazamiento horizontal para Forehand/Backhand
        hand_x = [p.get('RIGHT_WRIST', {}).get('x', 0.5) for p in pose_history]
        if max(hand_x) - min(hand_x) > 0.2:
            return "Drive/Volley"
            
        return "Waiting"

class MetricsService:
    @staticmethod
    def calculate_rmse(player_pose, reference_pose):
        """Calcula el RMSE entre dos poses (keypoints normalizados)."""
        # player_pose y reference_pose son arrays de (N, 2) o (N, 3)
        diff = np.array(player_pose) - np.array(reference_pose)
        return np.sqrt(np.mean(np.square(diff)))

    @staticmethod
    def calculate_com(landmarks):
        """Estima el centro de masa (CoM) promediando cadera y hombros."""
        # Media de hombros (11, 12) y caderas (23, 24)
        points = []
        for idx in [11, 12, 23, 24]:
            points.append([landmarks[idx].x, landmarks[idx].y])
        return np.mean(points, axis=0)

    @staticmethod
    def generate_individual_graphs(frames_data, task_id, graphs_dir):
        """Genera gráficas individuales para cada métrica en data/graphs."""
        if not frames_data:
            return []
        
        df = pd.DataFrame(frames_data)
        df_pose = df[df['pose_detected']].copy()
        if df_pose.empty:
            return []

        generated_files = []
        metrics_to_plot = {
            "elbow_angle": ("Ángulo del Codo", "Grados"),
            "knee_angle": ("Ángulo de la Rodilla", "Grados"),
            "rmse_vs_ref": ("Desviación (RMSE)", "Error"),
            "com_stability": ("Estabilidad del CoM (Variación)", "Desplazamiento"),
            "wrist_speed": ("Velocidad de la Muñeca", "Píxeles/Frame (Relativo)"),
            "base_width": ("Ancho de Base (Pies)", "Distancia Normalizada")
        }

        for metric, (title, ylabel) in metrics_to_plot.items():
            if metric not in df_pose.columns:
                continue
            
            plt.figure(figsize=(8, 4))
            plt.plot(df_pose['time'], df_pose[metric], label=title)
            plt.title(f"{title} - {task_id}")
            plt.xlabel("Tiempo (s)")
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            filename = f"{task_id}_{metric}.png"
            filepath = os.path.join(graphs_dir, filename)
            plt.savefig(filepath, dpi=300)
            plt.close()
            generated_files.append(filepath)
            
        return generated_files

    @staticmethod
    def generate_report(frames_data, output_dir, task_id):
        """Genera gráficas de métricas y las guarda como imagen."""
        if not frames_data:
            return None
        
        df = pd.DataFrame(frames_data)
        # Filtrar solo frames donde se detectó pose
        df_pose = df[df['pose_detected']].copy()
        
        if df_pose.empty:
            return None
        
        plt.figure(figsize=(10, 6))
        
        # Gráfica de Ángulo del Codo
        plt.subplot(2, 1, 1)
        plt.plot(df_pose['time'], df_pose['elbow_angle'], label='Ángulo Codo (deg)', color='blue')
        plt.title(f'Análisis Técnico - Tarea {task_id}')
        plt.ylabel('Grados')
        plt.legend()
        
        # Gráfica de RMSE
        plt.subplot(2, 1, 2)
        plt.plot(df_pose['time'], df_pose['rmse_vs_ref'], label='Desviación vs Referencia (RMSE)', color='red')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Error')
        plt.legend()
        
        plt.tight_layout()
        report_path = os.path.join(output_dir, f"{task_id}_report.png")
        plt.savefig(report_path, dpi=300)
        plt.close()
        
        return report_path

class VideoProcessor:
    def __init__(self):
        self.detector = ObjectDetectionService()
        self.pose_estimator = PoseEstimationService()
        self.classifier = ShotClassificationService()
        self.metrics = MetricsService()
        
    def process_video(self, video_path, output_path, task_id="unknown"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"No se pudo abrir el video: {video_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frames_data = []
        frame_count = 0
        pose_history = []
        
        # Pose de referencia "ideal" simplificada (ej. para un Serve o un Drive)
        # En un sistema real, esto se cargaría de una base de datos de pro-players
        reference_pose_mock = np.zeros((33, 2)) 
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            detections = self.detector.detect_objects(frame)
            pose_results = self.pose_estimator.estimate_pose(frame)
            
            current_frame_metrics = {
                "time": frame_count / fps if fps > 0 else 0,
                "objects": 0,
                "pose_detected": False,
                "elbow_angle": 0.0,
                "knee_angle": 0.0,
                "rmse_vs_ref": 0.0
            }

            if detections and len(detections) > 0:
                current_frame_metrics["objects"] = len(detections[0].boxes)
                for box in detections[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls in [0, 32, 38]:
                        label = f"{self.detector.model.names[cls]} {conf:.2f}"
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if pose_results and pose_results.pose_landmarks:
                current_frame_metrics["pose_detected"] = True
                self.pose_estimator.draw_pose(frame, pose_results)
                
                # Extraer landmarks para análisis
                landmarks = pose_results.pose_landmarks.landmark
                
                # 1. Calcular Ángulos (Brazo derecho)
                # RIGHT_SHOULDER(12), RIGHT_ELBOW(14), RIGHT_WRIST(16)
                shoulder = [landmarks[12].x, landmarks[12].y]
                elbow = [landmarks[14].x, landmarks[14].y]
                wrist = [landmarks[16].x, landmarks[16].y]
                elbow_angle = self.pose_estimator.calculate_angle(shoulder, elbow, wrist)
                current_frame_metrics["elbow_angle"] = elbow_angle
                
                # 2. Calcular Ángulos (Pierna derecha)
                # RIGHT_HIP(24), RIGHT_KNEE(26), RIGHT_ANKLE(28)
                hip = [landmarks[24].x, landmarks[24].y]
                knee = [landmarks[26].x, landmarks[26].y]
                ankle = [landmarks[28].x, landmarks[28].y]
                knee_angle = self.pose_estimator.calculate_angle(hip, knee, ankle)
                current_frame_metrics["knee_angle"] = knee_angle

                # 3. Guardar en historial para clasificación
                current_pose_dict = {
                    "RIGHT_WRIST": {"x": landmarks[16].x, "y": landmarks[16].y},
                    "NOSE": {"x": landmarks[0].x, "y": landmarks[0].y}
                }
                pose_history.append(current_pose_dict)
                
                # 4. Calcular RMSE vs Referencia Mock
                player_points = np.array([[l.x, l.y] for l in landmarks])
                rmse = self.metrics.calculate_rmse(player_points, reference_pose_mock)
                current_frame_metrics["rmse_vs_ref"] = float(rmse)

                # 5. Nuevas métricas avanzadas
                # Centro de Masa
                com = self.metrics.calculate_com(landmarks)
                current_frame_metrics["com_x"] = float(com[0])
                current_frame_metrics["com_y"] = float(com[1])
                
                # Estabilidad del CoM (distancia al CoM anterior)
                if len(frames_data) > 0 and frames_data[-1]["pose_detected"]:
                    prev_com = [frames_data[-1]["com_x"], frames_data[-1]["com_y"]]
                    stability = np.linalg.norm(com - np.array(prev_com))
                    current_frame_metrics["com_stability"] = float(stability)
                else:
                    current_frame_metrics["com_stability"] = 0.0

                # Velocidad de la muñeca (distancia entre frames)
                if len(frames_data) > 0 and frames_data[-1]["pose_detected"]:
                    prev_wrist = [frames_data[-1]["wrist_x"], frames_data[-1]["wrist_y"]]
                    wrist_curr = [landmarks[16].x, landmarks[16].y]
                    speed = np.linalg.norm(np.array(wrist_curr) - np.array(prev_wrist))
                    current_frame_metrics["wrist_speed"] = float(speed)
                else:
                    current_frame_metrics["wrist_speed"] = 0.0
                
                # Guardar posición actual de la muñeca para el siguiente frame
                current_frame_metrics["wrist_x"] = landmarks[16].x
                current_frame_metrics["wrist_y"] = landmarks[16].y

                # Ancho de la base (distancia entre tobillos 27, 28)
                left_ankle = [landmarks[27].x, landmarks[27].y]
                right_ankle = [landmarks[28].x, landmarks[28].y]
                base_width = np.linalg.norm(np.array(left_ankle) - np.array(right_ankle))
                current_frame_metrics["base_width"] = float(base_width)

                # Mostrar métricas en tiempo real en el video
                cv2.putText(frame, f"Codo: {int(elbow_angle)} deg", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"RMSE: {rmse:.4f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out.write(frame)
            frames_data.append(current_frame_metrics)
            frame_count += 1
            
        cap.release()
        out.release()
        
        # Clasificar el golpe predominante en el video
        final_shot = self.classifier.classify_shot(pose_history)
        
        # Generar Reporte Visual
        output_dir = os.path.dirname(output_path)
        report_file = self.metrics.generate_report(frames_data, output_dir, task_id)
        
        # Generar Gráficas Individuales
        graphs_dir = os.path.join(os.path.dirname(os.path.dirname(output_path)), "graphs")
        os.makedirs(graphs_dir, exist_ok=True)
        individual_graphs = self.metrics.generate_individual_graphs(frames_data, task_id, graphs_dir)
        
        # Preparar métricas finales asegurando que sean JSON compliant
        avg_elbow = np.mean([f["elbow_angle"] for f in frames_data if f["pose_detected"]]) if any(f["pose_detected"] for f in frames_data) else 0.0
        avg_rmse = np.mean([f["rmse_vs_ref"] for f in frames_data if f["pose_detected"]]) if any(f["pose_detected"] for f in frames_data) else 0.0
        
        # Manejar posibles NaNs de numpy
        if np.isnan(avg_elbow): avg_elbow = 0.0
        if np.isnan(avg_rmse): avg_rmse = 0.0

        return {
            "total_frames": len(frames_data),
            "output_video": output_path,
            "report_image": report_file,
            "individual_graphs": individual_graphs,
            "detected_shot": final_shot,
            "metrics_summary": {
                "avg_elbow_angle": float(avg_elbow),
                "avg_rmse": float(avg_rmse)
            },
            "summary": "Análisis técnico completado"
        }
