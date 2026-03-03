# Tennis AI MVP

Aplicación backend para análisis técnico de Tennis usando Computer Vision y ML.

## Objetivo

Reducir la brecha de acceso a herramientas de análisis técnico profesional mediante una aplicación gratuita.

## Stack Tecnológico

- FastAPI (Backend)
- OpenCV (Procesamiento de video)
- YOLO (detección objetos - stub en MVP)
- MoveNet (pose estimation - stub en MVP)
- Custom CNN (clasificación golpes - stub en MVP)
- Pandas + Matplotlib (métricas y gráficas)

## Flujo del Sistema

1. Usuario sube video
2. Extracción de frames
3. Detección objetos (YOLO)
4. Pose estimation (MoveNet)
5. Clasificación golpe (Custom CNN)
6. Cálculo RMSE vs jugador referencia
7. Generación gráfica y KPIs

## Ejecutar localmente

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload