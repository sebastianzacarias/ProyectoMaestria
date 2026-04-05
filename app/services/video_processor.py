import cv2
import numpy as np
from ultralytics import YOLO
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except ImportError:
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from scipy.signal import savgol_filter
import os
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
from app.config import (
    MODEL_PATH, POSE_MODEL_COMPLEXITY, POSE_MIN_DETECTION_CONFIDENCE,
    CONFIDENCE_THRESHOLD, BALL_PROXIMITY_THRESHOLD, MAIN_PLAYER_Y_THRESHOLD,
    POSE_HISTORY_BUFFER_SIZE, SMOOTHING_WINDOW_LENGTH, SMOOTHING_POLYORDER,
    METRICS_CONFIG, GRAPH_DPI, GRAPH_FIGSIZE, MOVEMENT_SIGNATURES,
    SERVE_Y_THRESHOLD, SHOT_DELTA_X_THRESHOLD, DISPLACEMENT_DELTA_X_THRESHOLD,
    FOREHAND_SIGNATURE_MAX_DISTANCE, COCO_PERSON_CLASS, COCO_SPORTS_BALL_CLASS,
    COCO_TENNIS_RACKET_CLASS, MIN_WRIST_SPEED_FOR_SHOT, HIGH_WRIST_SPEED_THRESHOLD,
    HIGH_ACCELERATION_THRESHOLD, FOREHAND_SHOULDER_ANGLE_MIN, FOREHAND_SHOULDER_ANGLE_MAX,
    BACKHAND_BODY_CROSS_THRESHOLD, SERVE_ELBOW_ANGLE_MIN, BODY_ROTATION_THRESHOLD,
    SHOT_STATES, FRAMES_IN_STATE_MIN, MIN_RACKET_SPEED_FOR_SHOT, DYNAMIC_REFERENCE_RANGES
)
from app.services.llm_service import OllamaLLMService

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ObjectDetectionService:
    def __init__(self, model_name: str = MODEL_PATH):
        """Servicio de detección de objetos usando YOLO."""
        try:
            self.model = YOLO(model_name)
            logger.info(f"Modelo YOLO cargado: {model_name}")
        except Exception as e:
            logger.error(f"Error al cargar modelo YOLO: {e}")
            raise
    
    def detect_objects(self, frame):
        # Clases de interés en COCO (yolo11 pre-entrenado):
        # 0: person, 32: sports ball (Pelota de tenis suele ser esta)
        # 38: tennis racket
        results = self.model(frame, verbose=False)
        return results

class PoseEstimationService:
    def __init__(self):
        """Servicio de estimación de pose usando MediaPipe."""
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        try:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=POSE_MODEL_COMPLEXITY,
                enable_segmentation=False,
                min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE
            )
            logger.info("MediaPipe Pose inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar MediaPipe Pose: {e}")
            raise

    def __del__(self):
        """Libera recursos de MediaPipe al destruir el objeto."""
        if hasattr(self, 'pose') and self.pose:
            self.pose.close()
            logger.debug("Recursos de MediaPipe Pose liberados")
    
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
        """Servicio de clasificación de golpes basado en heurísticas."""
        self.ideal_patterns = {
            "forehand_signature": np.array(MOVEMENT_SIGNATURES["forehand_signature"]),
            "serve_signature": np.array(MOVEMENT_SIGNATURES["serve_signature"])
        }
        logger.info("ShotClassificationService inicializado")

    def _compare_signature(self, signal, pattern):
        """Compara una señal con un patrón usando una distancia simple (MSE) como proxy de DTW."""
        if len(signal) < len(pattern):
            return 1.0  # Distancia máxima si no hay suficientes datos

        # Tomar la ventana más reciente
        recent_signal = np.array(signal[-len(pattern):])
        # Normalizar para comparar forma, no posición absoluta
        recent_signal = recent_signal - recent_signal[0]
        pattern = pattern - pattern[0]

        mse = np.mean((recent_signal - pattern)**2)
        return mse

    def _calculate_kinematics(self, pose_history: List[Dict], joint_name: str = 'RIGHT_WRIST') -> Dict[str, float]:
        """
        Calcula velocidad y aceleración de un joint específico.

        Returns:
            Dict con 'velocity', 'acceleration', 'speed' (magnitud de velocidad)
        """
        if len(pose_history) < 3:
            return {'velocity_x': 0.0, 'velocity_y': 0.0, 'speed': 0.0, 'acceleration': 0.0}

        # Posiciones en los últimos 3 frames
        curr = pose_history[-1].get(joint_name, {'x': 0.5, 'y': 0.5})
        prev = pose_history[-2].get(joint_name, {'x': 0.5, 'y': 0.5})
        prev2 = pose_history[-3].get(joint_name, {'x': 0.5, 'y': 0.5})

        # Velocidad (frame t-1 a t)
        vel_x = curr['x'] - prev['x']
        vel_y = curr['y'] - prev['y']
        speed = np.sqrt(vel_x**2 + vel_y**2)

        # Velocidad anterior (frame t-2 a t-1)
        prev_vel_x = prev['x'] - prev2['x']
        prev_vel_y = prev['y'] - prev2['y']
        prev_speed = np.sqrt(prev_vel_x**2 + prev_vel_y**2)

        # Aceleración (cambio de velocidad)
        acceleration = speed - prev_speed

        return {
            'velocity_x': float(vel_x),
            'velocity_y': float(vel_y),
            'speed': float(speed),
            'acceleration': float(acceleration)
        }

    def _calculate_body_rotation(self, pose: Dict) -> float:
        """
        Calcula la rotación del cuerpo basándose en la diferencia X entre hombros.
        Positivo = rotación hacia la derecha (forehand típico)
        Negativo = rotación hacia la izquierda (backhand típico)
        """
        r_shoulder = pose.get('RIGHT_SHOULDER', {'x': 0.5, 'y': 0.5})
        l_shoulder = pose.get('LEFT_SHOULDER', {'x': 0.5, 'y': 0.5})
        return float(r_shoulder['x'] - l_shoulder['x'])

    def _calculate_arm_extension(self, pose: Dict) -> float:
        """
        Calcula qué tan extendido está el brazo (0=plegado, 1=extendido).
        """
        r_shoulder = pose.get('RIGHT_SHOULDER', {'x': 0.5, 'y': 0.5})
        r_elbow = pose.get('RIGHT_ELBOW', {'x': 0.5, 'y': 0.5})
        r_wrist = pose.get('RIGHT_WRIST', {'x': 0.5, 'y': 0.5})

        # Distancia hombro-muñeca (directa)
        direct_dist = np.sqrt((r_wrist['x'] - r_shoulder['x'])**2 +
                             (r_wrist['y'] - r_shoulder['y'])**2)

        # Distancia hombro-codo + codo-muñeca
        shoulder_elbow = np.sqrt((r_elbow['x'] - r_shoulder['x'])**2 +
                                (r_elbow['y'] - r_shoulder['y'])**2)
        elbow_wrist = np.sqrt((r_wrist['x'] - r_elbow['x'])**2 +
                             (r_wrist['y'] - r_elbow['y'])**2)
        bent_dist = shoulder_elbow + elbow_wrist

        # Ratio: si direct_dist ≈ bent_dist → brazo extendido
        if bent_dist < 0.01:
            return 0.0
        extension = direct_dist / bent_dist
        return float(extension)

    def extract_features(self, pose_history, ball_pos=None, racket_pos=None):
        """
        Prepara un vector EXPANDIDO de características para clasificación mejorada.
        Incluye: posiciones, velocidades, ángulos, rotación corporal.
        """
        if not pose_history or len(pose_history) < 2:
            return None

        last_pose = pose_history[-1]
        prev_pose = pose_history[-2] if len(pose_history) >= 2 else last_pose

        # Extraer puntos clave
        r_wrist = last_pose.get('RIGHT_WRIST', {'x': 0.5, 'y': 0.5})
        r_elbow = last_pose.get('RIGHT_ELBOW', {'x': 0.5, 'y': 0.5})
        r_shoulder = last_pose.get('RIGHT_SHOULDER', {'x': 0.5, 'y': 0.5})
        l_shoulder = last_pose.get('LEFT_SHOULDER', {'x': 0.5, 'y': 0.5})
        r_hip = last_pose.get('RIGHT_HIP', {'x': 0.5, 'y': 0.5})
        nose = last_pose.get('NOSE', {'x': 0.5, 'y': 0.3})

        # 1. Posiciones absolutas
        features = [
            r_wrist['x'], r_wrist['y'],
            r_elbow['x'], r_elbow['y'],
            r_shoulder['x'], r_shoulder['y'],
        ]

        # 2. Posiciones relativas (respecto al centro del cuerpo)
        body_center_x = (r_shoulder['x'] + l_shoulder['x']) / 2
        body_center_y = (r_shoulder['y'] + r_hip['y']) / 2
        features.extend([
            r_wrist['x'] - body_center_x, r_wrist['y'] - body_center_y,
            r_elbow['x'] - body_center_x, r_elbow['y'] - body_center_y,
        ])

        # 3. Velocidades (cambio de posición entre frames)
        prev_r_wrist = prev_pose.get('RIGHT_WRIST', r_wrist)
        wrist_vel_x = r_wrist['x'] - prev_r_wrist['x']
        wrist_vel_y = r_wrist['y'] - prev_r_wrist['y']
        wrist_speed = np.sqrt(wrist_vel_x**2 + wrist_vel_y**2)
        features.extend([wrist_vel_x, wrist_vel_y, wrist_speed])

        # 4. Rotación de hombros (indica giro del cuerpo)
        shoulder_rotation = r_shoulder['x'] - l_shoulder['x']
        features.append(shoulder_rotation)

        # 5. Ángulo del brazo (aproximado)
        arm_angle = np.arctan2(r_wrist['y'] - r_elbow['y'],
                               r_wrist['x'] - r_elbow['x']) * 180 / np.pi
        features.append(arm_angle)

        # 6. Bola (si está disponible)
        if ball_pos:
            features.extend([
                ball_pos['x'], ball_pos['y'],
                ball_pos['x'] - r_wrist['x'], ball_pos['y'] - r_wrist['y']
            ])
        else:
            features.extend([0, 0, 0, 0])

        # 7. Raqueta (si está disponible)
        if racket_pos:
            features.extend([
                racket_pos['x'], racket_pos['y'],
                racket_pos['x'] - r_wrist['x'], racket_pos['y'] - r_wrist['y']
            ])
        else:
            features.extend([0, 0, 0, 0])

        return np.array(features)

    def classify_shot_frame(self, pose_history: List[Dict], ball_pos: Optional[Dict] = None,
                           racket_pos: Optional[Dict] = None) -> Tuple[str, float]:
        """
        CLASIFICADOR MEJORADO con biomecánica, cinemática y contexto de raqueta.

        Args:
            pose_history: Lista de diccionarios con landmarks por frame
            ball_pos: Diccionario {'x': float, 'y': float} o None
            racket_pos: Diccionario {'x': float, 'y': float} o None

        Returns:
            Tupla (Categoría, Probabilidad)
        """
        if len(pose_history) < 5:
            return "Iniciando...", 0.5

        curr_pose = pose_history[-1]

        # ============ MÉTRICAS CINEMÁTICAS ============
        wrist_kinematics = self._calculate_kinematics(pose_history, 'RIGHT_WRIST')
        wrist_speed = wrist_kinematics['speed']
        wrist_accel = wrist_kinematics['acceleration']

        # Velocidad de la raqueta (si está disponible)
        racket_speed = 0.0
        if racket_pos and len(pose_history) >= 2:
            # Calcular velocidad de raqueta manualmente
            if hasattr(self, '_prev_racket_pos') and self._prev_racket_pos:
                rx_vel = racket_pos['x'] - self._prev_racket_pos['x']
                ry_vel = racket_pos['y'] - self._prev_racket_pos['y']
                racket_speed = np.sqrt(rx_vel**2 + ry_vel**2)
            self._prev_racket_pos = racket_pos

        # ============ MÉTRICAS BIOMECÁNICAS ============
        body_rotation = self._calculate_body_rotation(curr_pose)
        arm_extension = self._calculate_arm_extension(curr_pose)

        # Posiciones
        r_wrist = curr_pose.get('RIGHT_WRIST', {'x': 0.5, 'y': 0.5})
        r_elbow = curr_pose.get('RIGHT_ELBOW', {'x': 0.5, 'y': 0.5})
        r_shoulder = curr_pose.get('RIGHT_SHOULDER', {'x': 0.5, 'y': 0.5})
        l_shoulder = curr_pose.get('LEFT_SHOULDER', {'x': 0.5, 'y': 0.5})
        nose = curr_pose.get('NOSE', {'x': 0.5, 'y': 0.3})

        # Centro del cuerpo
        body_center_x = (r_shoulder['x'] + l_shoulder['x']) / 2

        # Cruce de muñeca respecto al cuerpo (para backhand)
        wrist_cross = r_wrist['x'] - body_center_x

        # ============ DETECCIÓN DE IMPACTO CON BOLA ============
        is_impact = False
        if ball_pos:
            bx, by = ball_pos.get('x', 0.5), ball_pos.get('y', 0.5)
            # Distancia a muñeca o raqueta (lo más cercano)
            dist_to_wrist = np.sqrt((r_wrist['x'] - bx)**2 + (r_wrist['y'] - by)**2)
            dist_to_racket = 999
            if racket_pos:
                dist_to_racket = np.sqrt((racket_pos['x'] - bx)**2 + (racket_pos['y'] - by)**2)

            min_dist = min(dist_to_wrist, dist_to_racket)
            if min_dist < BALL_PROXIMITY_THRESHOLD:
                is_impact = True

        # ============ REGLAS DE CLASIFICACIÓN ============

        # 1️⃣ SAQUE: Mano muy alta + brazo extendido + alta velocidad
        if r_wrist['y'] < nose['y'] - SERVE_Y_THRESHOLD:
            prob = 0.65
            # Confirmar con extensión de brazo
            if arm_extension > 0.85:
                prob += 0.15
            # Confirmar con velocidad
            if wrist_speed > MIN_WRIST_SPEED_FOR_SHOT or racket_speed > MIN_RACKET_SPEED_FOR_SHOT:
                prob += 0.1
            # Impacto con bola
            if is_impact:
                prob = min(0.98, prob + 0.2)

            return "Saque", min(1.0, prob)

        # 2️⃣ FOREHAND: Rotación positiva + muñeca del lado derecho + alta velocidad
        is_high_speed = wrist_speed > MIN_WRIST_SPEED_FOR_SHOT or racket_speed > MIN_RACKET_SPEED_FOR_SHOT

        if body_rotation > BODY_ROTATION_THRESHOLD and wrist_cross > 0 and is_high_speed:
            prob = 0.70
            # Aumentar probabilidad si hay aceleración alta (swing activo)
            if abs(wrist_accel) > HIGH_ACCELERATION_THRESHOLD:
                prob += 0.10
            # Impacto confirmado
            if is_impact:
                return "Forehand (Impacto)", 0.96

            return "Forehand", min(1.0, prob)

        # 3️⃣ BACKHAND: Muñeca cruza el cuerpo + rotación negativa + alta velocidad
        if wrist_cross < BACKHAND_BODY_CROSS_THRESHOLD and is_high_speed:
            prob = 0.70
            # Confirmar con rotación corporal hacia el lado opuesto
            if body_rotation < -0.05:
                prob += 0.10
            # Aceleración
            if abs(wrist_accel) > HIGH_ACCELERATION_THRESHOLD:
                prob += 0.10
            # Impacto
            if is_impact:
                return "Backhand (Impacto)", 0.96

            return "Backhand", min(1.0, prob)

        # 4️⃣ VOLEA: Velocidad media-baja + impacto + brazo semi-extendido
        if is_impact and 0.02 < wrist_speed < 0.06 and 0.6 < arm_extension < 0.85:
            return "Volea", 0.80

        # 5️⃣ PREPARACIÓN: Velocidad baja + sin impacto
        if wrist_speed < 0.015 and not is_impact:
            return "Preparacion", 0.70

        # 6️⃣ DESPLAZAMIENTO: Velocidad moderada sin impacto
        if 0.02 < wrist_speed < 0.05 and not is_impact:
            return "Desplazamiento", 0.65

        # 7️⃣ FOLLOW-THROUGH: Alta velocidad pero sin bola cerca (después del impacto)
        if wrist_speed > 0.05 and not is_impact:
            return "Follow-Through", 0.60

        # Default
        return "En Espera", 0.50

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
    def calculate_com(landmarks) -> np.ndarray:
        """Estima el centro de masa (CoM) promediando cadera y hombros."""
        points = []
        for idx in [11, 12, 23, 24]:
            points.append([landmarks[idx].x, landmarks[idx].y])
        return np.mean(points, axis=0)

    @staticmethod
    def smooth_signal(data: np.ndarray, window_length: int = SMOOTHING_WINDOW_LENGTH,
                     polyorder: int = SMOOTHING_POLYORDER) -> np.ndarray:
        """Aplica un filtro de Savitzky-Golay para suavizar la señal."""
        if len(data) < window_length:
            return data
        return savgol_filter(data, window_length, polyorder)

    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        """Convierte un valor a float manejando NaN e infinitos."""
        try:
            f = float(value)
            if np.isnan(f) or np.isinf(f):
                return default
            return f
        except (TypeError, ValueError):
            return default

    @staticmethod
    def generate_individual_graphs(frames_data: List[Dict], task_id: str, graphs_dir: str) -> List[str]:
        """Genera gráficas individuales para cada métrica en data/graphs."""
        if not frames_data:
            logger.warning(f"No hay datos para generar gráficas - task_id: {task_id}")
            return []

        try:
            df = pd.DataFrame(frames_data)
            df_pose = df[df['pose_detected']].copy()
            if df_pose.empty:
                logger.warning(f"No hay frames con pose detectada - task_id: {task_id}")
                return []

            generated_files = []

            for metric, config in METRICS_CONFIG.items():
                if metric not in df_pose.columns:
                    continue

                try:
                    y_data = df_pose[metric].values
                    if len(y_data) >= SMOOTHING_WINDOW_LENGTH:
                        y_smooth = MetricsService.smooth_signal(y_data)
                    else:
                        y_smooth = y_data

                    plt.figure(figsize=GRAPH_FIGSIZE)
                    
                    # 1. Calcular banda de referencia dinámica
                    ref_range = config["range"]
                    y_min_vec = []
                    y_max_vec = []
                    
                    for _, row in df_pose.iterrows():
                        phase = row.get("movement", "IDLE")
                        # Limpiar nombre de fase si tiene "(Impacto)"
                        clean_phase = phase.split(" (")[0] if phase else "IDLE"
                        # Mapear nombres a los que tenemos en DYNAMIC_REFERENCE_RANGES si es necesario
                        if clean_phase == "En Espera": clean_phase = "IDLE"
                        if clean_phase == "Preparación": clean_phase = "PREPARATION"
                        if clean_phase == "Impacto": clean_phase = "IMPACT"
                        if clean_phase == "Seguimiento": clean_phase = "FOLLOW_THROUGH"
                        if clean_phase == "Iniciando...": clean_phase = "IDLE"
                        if clean_phase == "Volea": clean_phase = "IMPACT" # O asignar a Volea si existe en config
                        
                        phase_range = DYNAMIC_REFERENCE_RANGES.get(metric, {}).get(clean_phase, ref_range)
                        y_min_vec.append(phase_range[0])
                        y_max_vec.append(phase_range[1])
                    
                    # 2. Suavizar la banda si hay suficientes datos
                    if len(y_min_vec) >= SMOOTHING_WINDOW_LENGTH:
                        y_min_smooth = MetricsService.smooth_signal(np.array(y_min_vec))
                        y_max_smooth = MetricsService.smooth_signal(np.array(y_max_vec))
                    else:
                        y_min_smooth = np.array(y_min_vec)
                        y_max_smooth = np.array(y_max_vec)

                    # 3. Dibujar la banda variable
                    plt.fill_between(df_pose['time'], y_min_smooth, y_max_smooth, 
                                    color='green', alpha=0.2, label='Rango Ideal Dinámico')
                    
                    plt.plot(df_pose['time'], y_smooth, label=f"{config['title']} (Jugador Suavizado)",
                            color='blue', linewidth=2)
                    plt.title(f"{config['title']} (Suavizado) - {task_id}")
                    plt.xlabel("Tiempo (s)")
                    plt.ylabel(config['ylabel'])
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)

                    filename = f"{task_id}_{metric}.png"
                    filepath = os.path.join(graphs_dir, filename)
                    plt.savefig(filepath, dpi=GRAPH_DPI)
                    plt.close()
                    generated_files.append(filepath)
                    logger.info(f"Gráfica generada: {filepath}")
                except Exception as e:
                    logger.error(f"Error generando gráfica para {metric}: {e}")
                    plt.close()

            return generated_files
        except Exception as e:
            logger.error(f"Error general en generate_individual_graphs: {e}")
            return []

class VideoProcessor:
    def __init__(self):
        """Procesador principal de videos de tenis."""
        try:
            self.detector = ObjectDetectionService()
            self.pose_estimator = PoseEstimationService()
            self.classifier = ShotClassificationService()
            self.metrics = MetricsService()
            self.llm_service = OllamaLLMService()
            self.last_ball_pos: Optional[Dict[str, float]] = None
            logger.info("VideoProcessor inicializado correctamente")
        except Exception as e:
            logger.error(f"Error inicializando VideoProcessor: {e}")
            raise
        
    def _process_detections(self, detections, frame, width: int, height: int) -> Dict[str, Any]:
        """Procesa las detecciones de objetos y retorna métricas del frame."""
        metrics = {
            "objects": 0,
            "prob_jugador": 0.0,
            "prob_oponente": 0.0,
            "probs_raquetas": [],
            "probs_bolas": [],
            "ball_pos": None,
            "racket_pos": None  # NUEVO: posición de la raqueta
        }

        if not detections or len(detections) == 0:
            return metrics

        metrics["objects"] = len(detections[0].boxes)

        # Filtrar personas, raquetas y bolas
        persons, rackets, balls = [], [], []

        for box in detections[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            if cls == COCO_PERSON_CLASS:
                persons.append(box)
            elif cls == COCO_TENNIS_RACKET_CLASS:
                rackets.append(box)
            elif cls == COCO_SPORTS_BALL_CLASS:
                balls.append(box)

        # Persistencia de la bola
        if not balls and self.last_ball_pos:
            metrics["ball_pos"] = self.last_ball_pos
        elif balls:
            best_ball = sorted(balls, key=lambda b: float(b.conf[0]), reverse=True)[0]
            bx1, by1, bx2, by2 = best_ball.xyxy[0].tolist()
            self.last_ball_pos = {'x': (bx1 + bx2) / (2 * width), 'y': (by1 + by2) / (2 * height)}
            metrics["ball_pos"] = self.last_ball_pos

        # Seleccionar jugadores (máximo 2)
        persons = sorted(persons, key=lambda b: float(b.conf[0]), reverse=True)[:2]
        p_main, p_opp = self._identify_players(persons, height)

        # Guardar posición de la raqueta más probable (la del jugador principal)
        if rackets:
            # Usar la raqueta con mayor confianza
            best_racket = sorted(rackets, key=lambda r: float(r.conf[0]), reverse=True)[0]
            rx1, ry1, rx2, ry2 = best_racket.xyxy[0].tolist()
            metrics["racket_pos"] = {
                'x': (rx1 + rx2) / (2 * width),
                'y': (ry1 + ry2) / (2 * height)
            }

        # Dibujar anotaciones
        self._draw_players(frame, p_main, p_opp, metrics)
        self._draw_rackets(frame, rackets, metrics)
        self._draw_balls(frame, balls, metrics)

        return metrics

    def _identify_players(self, persons: List, height: int) -> Tuple[Optional[Any], Optional[Any]]:
        """Identifica jugador principal y oponente basándose en posición."""
        p_main, p_opp = None, None

        if len(persons) == 2:
            y_pos_0 = float(persons[0].xyxy[0][3])
            y_pos_1 = float(persons[1].xyxy[0][3])

            if y_pos_0 > y_pos_1:
                cand_main, cand_opp = persons[0], persons[1]
            else:
                cand_main, cand_opp = persons[1], persons[0]

            # Aplicar restricciones de posición
            if float(cand_main.xyxy[0][3]) > height * MAIN_PLAYER_Y_THRESHOLD:
                p_main = cand_main

            if float(cand_opp.xyxy[0][1]) < height * MAIN_PLAYER_Y_THRESHOLD:
                p_opp = cand_opp

        elif len(persons) == 1:
            p = persons[0]
            y1, y2 = float(p.xyxy[0][1]), float(p.xyxy[0][3])

            if y2 > height * MAIN_PLAYER_Y_THRESHOLD:
                p_main = p
            elif y1 < height * MAIN_PLAYER_Y_THRESHOLD:
                p_opp = p

        return p_main, p_opp

    def _draw_players(self, frame, p_main, p_opp, metrics: Dict):
        """Dibuja bounding boxes de jugadores."""
        if p_main:
            metrics["prob_jugador"] = float(p_main.conf[0])
            x1, y1, x2, y2 = p_main.xyxy[0].tolist()
            conf = float(p_main.conf[0])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Jugador {conf:.2f}", (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if p_opp:
            metrics["prob_oponente"] = float(p_opp.conf[0])
            x1, y1, x2, y2 = p_opp.xyxy[0].tolist()
            conf = float(p_opp.conf[0])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"Oponente {conf:.2f}", (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def _draw_rackets(self, frame, rackets: List, metrics: Dict):
        """Dibuja bounding boxes de raquetas."""
        for r in rackets:
            conf = float(r.conf[0])
            metrics["probs_raquetas"].append(conf)
            x1, y1, x2, y2 = r.xyxy[0].tolist()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            cv2.putText(frame, f"Raqueta {conf:.2f}", (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    def _draw_balls(self, frame, balls: List, metrics: Dict):
        """Dibuja bounding boxes de bolas."""
        for b in balls:
            conf = float(b.conf[0])
            metrics["probs_bolas"].append(conf)
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(frame, f"Bola {conf:.2f}", (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    def _process_pose(self, pose_results, frame, pose_history: deque) -> Optional[Dict[str, Any]]:
        """Procesa pose landmarks y calcula métricas biomecánicas."""
        if not pose_results or not pose_results.pose_landmarks:
            return None

        self.pose_estimator.draw_pose(frame, pose_results)
        landmarks = pose_results.pose_landmarks.landmark

        # Calcular ángulos
        shoulder = [landmarks[12].x, landmarks[12].y]
        elbow = [landmarks[14].x, landmarks[14].y]
        wrist = [landmarks[16].x, landmarks[16].y]
        elbow_angle = self.pose_estimator.calculate_angle(shoulder, elbow, wrist)

        hip = [landmarks[24].x, landmarks[24].y]
        knee = [landmarks[26].x, landmarks[26].y]
        ankle = [landmarks[28].x, landmarks[28].y]
        knee_angle = self.pose_estimator.calculate_angle(hip, knee, ankle)

        # Guardar en historial (buffer circular) - EXPANDIDO con más landmarks
        current_pose_dict = {
            # Brazos
            "RIGHT_SHOULDER": {"x": landmarks[12].x, "y": landmarks[12].y},
            "RIGHT_ELBOW": {"x": landmarks[14].x, "y": landmarks[14].y},
            "RIGHT_WRIST": {"x": landmarks[16].x, "y": landmarks[16].y},
            "LEFT_SHOULDER": {"x": landmarks[11].x, "y": landmarks[11].y},
            "LEFT_ELBOW": {"x": landmarks[13].x, "y": landmarks[13].y},
            "LEFT_WRIST": {"x": landmarks[15].x, "y": landmarks[15].y},
            # Torso
            "NOSE": {"x": landmarks[0].x, "y": landmarks[0].y},
            # Caderas
            "RIGHT_HIP": {"x": landmarks[24].x, "y": landmarks[24].y},
            "LEFT_HIP": {"x": landmarks[23].x, "y": landmarks[23].y},
            # Piernas
            "RIGHT_KNEE": {"x": landmarks[26].x, "y": landmarks[26].y},
            "LEFT_KNEE": {"x": landmarks[25].x, "y": landmarks[25].y},
            "RIGHT_ANKLE": {"x": landmarks[28].x, "y": landmarks[28].y},
            "LEFT_ANKLE": {"x": landmarks[27].x, "y": landmarks[27].y},
        }
        pose_history.append(current_pose_dict)

        # Métricas avanzadas
        com = self.metrics.calculate_com(landmarks)
        left_ankle = [landmarks[27].x, landmarks[27].y]
        right_ankle = [landmarks[28].x, landmarks[28].y]
        base_width = np.linalg.norm(np.array(left_ankle) - np.array(right_ankle))

        pose_metrics = {
            "elbow_angle": float(elbow_angle),
            "knee_angle": float(knee_angle),
            "com_x": float(com[0]),
            "com_y": float(com[1]),
            "wrist_x": float(landmarks[16].x),
            "wrist_y": float(landmarks[16].y),
            "base_width": float(base_width),
            "com_stability": 0.0,
            "wrist_speed": 0.0
        }

        # Mostrar métricas en el frame
        cv2.putText(frame, f"Codo: {int(elbow_angle)} deg", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return pose_metrics

    def _calculate_temporal_metrics(self, current_metrics: Dict, prev_frame: Optional[Dict]) -> Dict:
        """Calcula métricas que requieren comparación temporal."""
        if not prev_frame or not prev_frame.get("pose_detected"):
            return current_metrics

        # Estabilidad del CoM
        prev_com = [prev_frame["com_x"], prev_frame["com_y"]]
        curr_com = [current_metrics["com_x"], current_metrics["com_y"]]
        stability = np.linalg.norm(np.array(curr_com) - np.array(prev_com))
        current_metrics["com_stability"] = float(stability)

        # Velocidad de la muñeca
        prev_wrist = [prev_frame["wrist_x"], prev_frame["wrist_y"]]
        curr_wrist = [current_metrics["wrist_x"], current_metrics["wrist_y"]]
        speed = np.linalg.norm(np.array(curr_wrist) - np.array(prev_wrist))
        current_metrics["wrist_speed"] = float(speed)

        return current_metrics

    def process_video(self, video_path: str, output_path: str, task_id: str = "unknown") -> Dict[str, Any]:
        """
        Procesa un video de tenis detectando objetos, poses y clasificando movimientos.

        Args:
            video_path: Ruta al video de entrada
            output_path: Ruta donde guardar el video procesado
            task_id: Identificador de la tarea

        Returns:
            Diccionario con resultados del análisis
        """
        logger.info(f"Iniciando procesamiento de video - task_id: {task_id}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if fps <= 0:
            logger.warning("FPS inválido, usando default 30")
            fps = 30

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frames_data = []
        frame_count = 0
        pose_history = deque(maxlen=POSE_HISTORY_BUFFER_SIZE)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                current_frame_metrics = {
                    "time": frame_count / fps,
                    "objects": 0,
                    "pose_detected": False,
                    "elbow_angle": 0.0,
                    "knee_angle": 0.0,
                    "prob_jugador": 0.0,
                    "prob_oponente": 0.0,
                    "probs_raquetas": [],
                    "probs_bolas": []
                }

                # Detección de objetos
                detections = self.detector.detect_objects(frame)
                detection_metrics = self._process_detections(detections, frame, width, height)
                current_frame_metrics.update(detection_metrics)

                # Estimación de pose
                pose_results = self.pose_estimator.estimate_pose(frame)
                pose_metrics = self._process_pose(pose_results, frame, pose_history)

                if pose_metrics:
                    current_frame_metrics["pose_detected"] = True
                    current_frame_metrics.update(pose_metrics)

                    # Calcular métricas temporales
                    prev_frame = frames_data[-1] if frames_data else None
                    current_frame_metrics = self._calculate_temporal_metrics(current_frame_metrics, prev_frame)

                    # Clasificación de movimiento (CON RAQUETA INTEGRADA)
                    ball_pos = current_frame_metrics.get("ball_pos")
                    racket_pos = current_frame_metrics.get("racket_pos")
                    movement, move_prob = self.classifier.classify_shot_frame(
                        list(pose_history), ball_pos, racket_pos
                    )
                    current_frame_metrics["movement"] = movement
                    current_frame_metrics["movement_prob"] = move_prob

                    # Dibujar etiqueta de movimiento en el jugador principal
                    if detection_metrics.get("prob_jugador", 0.0) > 0:
                        # Extraer info del jugador desde detections
                        persons = [box for box in (detections[0].boxes if detections and len(detections) > 0 else [])
                                 if int(box.cls[0]) == COCO_PERSON_CLASS and float(box.conf[0]) >= CONFIDENCE_THRESHOLD]
                        persons = sorted(persons, key=lambda b: float(b.conf[0]), reverse=True)[:2]
                        p_main, _ = self._identify_players(persons, height)
                        if p_main:
                            x1, y1, x2, y2 = p_main.xyxy[0].tolist()
                            cv2.putText(frame, f"{movement} {move_prob:.2f}", (int(x1), int(y2) + 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                out.write(frame)
                frames_data.append(current_frame_metrics)
                frame_count += 1

        except Exception as e:
            logger.error(f"Error durante el procesamiento del video: {e}")
            raise
        finally:
            cap.release()
            out.release()
            logger.info(f"Video procesado: {frame_count} frames")

        # Clasificar el golpe predominante
        final_shot = self.classifier.classify_shot(list(pose_history))

        # Generar gráficas
        graphs_dir = os.path.join(os.path.dirname(os.path.dirname(output_path)), "graphs")
        os.makedirs(graphs_dir, exist_ok=True)
        individual_graphs = self.metrics.generate_individual_graphs(frames_data, task_id, graphs_dir)

        # Preparar métricas finales con validación de NaN
        frames_with_pose = [f for f in frames_data if f["pose_detected"]]
        avg_elbow = self.metrics.safe_float(
            np.mean([f["elbow_angle"] for f in frames_with_pose]) if frames_with_pose else 0.0
        )
        avg_knee = self.metrics.safe_float(
            np.mean([f["knee_angle"] for f in frames_with_pose]) if frames_with_pose else 0.0
        )

        logger.info(f"Análisis completado - task_id: {task_id}, golpe detectado: {final_shot}")

        # Generar resumen con LLM
        metrics_dict = {
            "avg_elbow_angle": avg_elbow,
            "avg_knee_angle": avg_knee,
            "frames_with_pose": len(frames_with_pose)
        }

        logger.info("Generando resumen interpretativo con LLM...")
        llm_result = self.llm_service.generate_summary(frames_data, metrics_dict, final_shot)

        return {
            "total_frames": len(frames_data),
            "output_video": output_path,
            "individual_graphs": individual_graphs,
            "detected_shot": final_shot,
            "metrics_summary": metrics_dict,
            "summary": "Análisis técnico completado",
            "llm_summary": llm_result
        }
