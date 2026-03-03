from fastapi import APIRouter, UploadFile, File
from app.core.video_processor import process_video

router = APIRouter()

@router.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    result = await process_video(file)
    return result