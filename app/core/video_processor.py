import os
import shutil
import json
import numpy as np

from app.utils.frame_extractor import extract_frames
from app.models.yolo_detector import YOLODetector
from app.models.pose_estimator import PoseEstimator
from app.models.stroke_classifier import StrokeClassifier
from app.core.metrics import compute_rmse
from app.core.report_generator import generate_bar_chart
from app.config import UPLOAD_DIR, REFERENCE_DATA_PATH

async def process_video(file):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    frames = extract_frames(file_path)

    yolo = YOLODetector()
    pose = PoseEstimator()
    classifier = StrokeClassifier()

    keypoints_sequence = []
    strokes_metrics = []

    for frame in frames[:30]:  # limit for MVP
        yolo.detect(frame)
        kp = pose.estimate(frame)
        keypoints_sequence.append(kp)

    predicted_stroke = classifier.classify(keypoints_sequence)

    reference = np.random.rand(17, 2)
    rmse = compute_rmse(keypoints_sequence[0], reference)

    strokes_metrics.append({
        "stroke": predicted_stroke,
        "rmse": float(rmse)
    })

    chart_path = generate_bar_chart(strokes_metrics)

    return {
        "stroke_analysis": strokes_metrics,
        "report_chart": chart_path
    }