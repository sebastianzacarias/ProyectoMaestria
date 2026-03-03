from pydantic import BaseModel
from typing import List

class StrokeMetric(BaseModel):
    stroke_type: str
    rmse: float
    recommendation: str

class AnalysisResponse(BaseModel):
    strokes: List[StrokeMetric]
    max_ball_speed: float
    coverage_score: float