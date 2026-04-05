"""
Configuración centralizada para la aplicación de análisis de tenis.
"""

# Configuración de Modelos
MODEL_PATH = "yolo26m.pt"
POSE_MODEL_COMPLEXITY = 1
POSE_MIN_DETECTION_CONFIDENCE = 0.5

# Umbrales de Detección
CONFIDENCE_THRESHOLD = 0.3
BALL_PROXIMITY_THRESHOLD = 0.15  # Distancia normalizada para detección de impacto
MAIN_PLAYER_Y_THRESHOLD = 0.5  # Porcentaje de altura para identificar jugador principal

# Buffer y Ventanas
POSE_HISTORY_BUFFER_SIZE = 30  # Frames
SMOOTHING_WINDOW_LENGTH = 11
SMOOTHING_POLYORDER = 3

# Rangos de Referencia de Jugadores Top (para gráficas)
REFERENCE_RANGES = {
    "elbow_angle": (70, 110),
    "knee_angle": (90, 140),
    "com_stability": (0.0, 0.02),
    "wrist_speed": (0.0, 0.05),
    "base_width": (0.4, 0.7)
}

# Nuevos Rangos Dinámicos por Fase Detectada
DYNAMIC_REFERENCE_RANGES = {
    "elbow_angle": {
        "IDLE": (70, 110),
        "PREPARATION": (80, 100),
        "BACKSWING": (60, 90),
        "FORWARD_SWING": (90, 120),
        "IMPACT": (100, 140),
        "FOLLOW_THROUGH": (80, 130),
        "Saque": (140, 180),
        "Forehand": (90, 130),
        "Backhand": (80, 120),
        "Volea": (80, 110)
    },
    "knee_angle": {
        "IDLE": (100, 140),
        "PREPARATION": (100, 130),
        "BACKSWING": (90, 120),
        "FORWARD_SWING": (110, 140),
        "IMPACT": (120, 150),
        "FOLLOW_THROUGH": (100, 140)
    }
}

# Títulos y Etiquetas de Métricas
METRICS_CONFIG = {
    "elbow_angle": {
        "title": "Ángulo del Codo",
        "ylabel": "Grados",
        "range": REFERENCE_RANGES["elbow_angle"]
    },
    "knee_angle": {
        "title": "Ángulo de la Rodilla",
        "ylabel": "Grados",
        "range": REFERENCE_RANGES["knee_angle"]
    },
    "com_stability": {
        "title": "Estabilidad del CoM (Variación)",
        "ylabel": "Desplazamiento",
        "range": REFERENCE_RANGES["com_stability"]
    },
    "wrist_speed": {
        "title": "Velocidad de la Muñeca",
        "ylabel": "Píxeles/Frame (Relativo)",
        "range": REFERENCE_RANGES["wrist_speed"]
    },
    "base_width": {
        "title": "Ancho de Base (Pies)",
        "ylabel": "Distancia Normalizada",
        "range": REFERENCE_RANGES["base_width"]
    }
}

# Configuración de Gráficas
GRAPH_DPI = 300
GRAPH_FIGSIZE = (8, 4)

# Firmas de Movimientos (para clasificación heurística)
MOVEMENT_SIGNATURES = {
    "forehand_signature": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.3, 0.3, 0.3],
    "serve_signature": [0.8, 0.82, 0.85, 0.88, 0.9, 0.93, 0.95, 0.97, 0.99, 1.0]
}

# Umbrales de Clasificación de Movimientos
SERVE_Y_THRESHOLD = 0.05  # Mano debe estar este valor por encima de la cabeza
SHOT_DELTA_X_THRESHOLD = 0.15  # Movimiento horizontal mínimo para golpe
DISPLACEMENT_DELTA_X_THRESHOLD = 0.05  # Movimiento mínimo para desplazamiento
FOREHAND_SIGNATURE_MAX_DISTANCE = 0.01  # Distancia MSE máxima para reconocer patrón

# Umbrales de Velocidad (movimiento entre frames, normalizado)
MIN_WRIST_SPEED_FOR_SHOT = 0.02  # Velocidad mínima de muñeca para golpe
HIGH_WRIST_SPEED_THRESHOLD = 0.08  # Velocidad alta indica swing activo
MIN_RACKET_SPEED_FOR_SHOT = 0.03  # Velocidad mínima de raqueta

# Umbrales de Aceleración
HIGH_ACCELERATION_THRESHOLD = 0.005  # Cambio brusco de velocidad

# Umbrales Angulares (grados)
FOREHAND_SHOULDER_ANGLE_MIN = 80  # Ángulo mínimo hombro-codo-muñeca para forehand
FOREHAND_SHOULDER_ANGLE_MAX = 170
BACKHAND_BODY_CROSS_THRESHOLD = -0.1  # Muñeca cruza línea central del cuerpo (X negativo)
SERVE_ELBOW_ANGLE_MIN = 140  # Extensión de brazo en saque

# Umbrales de Rotación del Cuerpo
BODY_ROTATION_THRESHOLD = 0.15  # Diferencia X entre hombros para detectar rotación

# Máquina de Estados
SHOT_STATES = ["IDLE", "PREPARATION", "BACKSWING", "FORWARD_SWING", "IMPACT", "FOLLOW_THROUGH"]
FRAMES_IN_STATE_MIN = 3  # Frames mínimos para confirmar cambio de estado

# Optimización de Procesamiento
PROCESS_EVERY_N_FRAMES = 1  # Procesar 1 de cada N frames (1 = todos)
YOLO_BATCH_SIZE = 1  # Número de frames a procesar en batch

# Clases COCO de Interés
COCO_PERSON_CLASS = 0
COCO_SPORTS_BALL_CLASS = 32
COCO_TENNIS_RACKET_CLASS = 38
