# Tennis AI Analyzer 🎾

Plataforma avanzada de análisis biomecánico para tenis basada en Inteligencia Artificial. Este sistema utiliza visión por computadora (**YOLOv8**) y modelos de visión-lenguaje (**qwen2-vl:8b**) para proporcionar un análisis detallado de la técnica del jugador, identificando golpes, midiendo ángulos articulares y ofreciendo recomendaciones profesionales de entrenamiento.

## 🚀 Funcionalidades Principales

### 1. Detección y Seguimiento Multiobjeto
*   **Identificación de Jugadores:** Diferenciación automática entre el jugador principal (en primer plano) y el oponente.
*   **Seguimiento de Implementos:** Detección y rastreo en tiempo real de la raqueta y la pelota de tenis.
*   **Persistencia de Trayectoria:** Lógica avanzada para mantener la última posición conocida de la pelota incluso en momentos de alta velocidad o desenfoque.

### 2. Análisis Biomecánico y Cinemático
*   **Estimación de Pose:** Extracción de puntos clave (landmarks) del cuerpo humano utilizando **YOLOv8-Pose** (modelo `yolo26s-pose.pt`).
*   **Medición de Ángulos:** Cálculo dinámico de ángulos críticos como el codo (extensión/flexión) y la rodilla.
*   **Estabilidad del Centro de Masa (CoM):** Evaluación del equilibrio del jugador durante la ejecución del golpe.
*   **Métricas de Velocidad:** Medición de la velocidad de la muñeca y la raqueta para evaluar la potencia del swing.
*   **Ancho de Base:** Análisis de la separación de los pies para evaluar la estabilidad del apoyo.

### 3. Clasificación de Golpes con Inteligencia Artificial
*   **Reconocimiento Automático:** Clasificación de golpes en categorías como **Saque**, **Forehand (Derecha)**, **Backhand (Revés)** y **Volea**.
*   **Detección de Impacto:** Identificación precisa del momento exacto del contacto con la bola mediante proximidad espacial y cinemática de la muñeca.
*   **Análisis de Fases:** Desglose del movimiento en fases: Preparación, Backswing, Forward Swing, Impacto y Follow Through.

### 4. Resumen Inteligente con LLM servido por Ollama
*   **Dictamen Profesional:** Generación de un informe técnico basado en el framework **Observación -> Métricas -> Impacto Técnico -> Ejercicio Sugerido**.

### 5. Visualización y Reportes
*   **Video Procesado:** Exportación de video con anotaciones en tiempo real (esqueleto, bounding boxes y etiquetas de movimiento).
*   **Dashboard de Métricas:** Interfaz web interactiva que muestra estadísticas clave, resumen ejecutivoy feedback.
*   **Gráficas Biomecánicas:** Generación de gráficos detallados que comparan el desempeño del usuario con rangos de referencia de jugadores profesionales.

## 🛠 Herramientas y Tecnologías

*   **Backend:** FastAPI (Python 3.10+)
*   **Visión por Computadora:** Ultralytics YOLOv8 (Detección de Objetos: `yolo26m.pt`, Pose-Estimation: `yolo26s-pose.pt`)
*   **IA Generativa:** Ollama (qwen2-vl:8b)
*   **Procesamiento de Datos:** NumPy, Pandas, SciPy (Filtro Savitzky-Golay para suavizado de señales)
*   **Visualización:** Matplotlib, OpenCV
*   **Frontend:** HTML5, CSS3, JavaScript (Vanilla)

## 📋 Requisitos Previos

1.  **Python 3.10 o superior**
2.  **Ollama instalado** y ejecutando el modelo VLM:
    ```bash
    ollama run qwen2-vl:8b
    ```
3.  **Dependencias del proyecto:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Cómo Ejecutar

1.  Inicia el servidor backend:
    ```bash
    python -m uvicorn app.main:app --reload
    ```
2.  Accede a la aplicación en tu navegador:
    `http://127.0.0.1:8000`
3.  Sube un video de tenis (MP4, AVI, MOV o MKV) y espera a que la IA complete el análisis.

## 📈 Consideraciones Técnicas
*   **Suavizado de Datos:** Se aplica un filtro Savitzky-Golay a las coordenadas de los puntos clave para eliminar el ruido de la detección y obtener curvas de movimiento fluidas.
*   **Rangos Dinámicos:** Los ángulos de referencia cambian automáticamente según la fase del golpe detectada (ej. el ángulo ideal del codo es diferente en la preparación que en el impacto).
*   **Optimización VLM:** Los frames enviados al modelo VL se redimensionan a 448x448 para garantizar tiempos de respuesta rápidos sin perder calidad analítica.
