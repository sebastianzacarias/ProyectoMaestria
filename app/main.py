from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from app.services.video_processor import VideoProcessor
import os
import shutil
import uuid

app = FastAPI(title="Tennis AI MVP API")

# Definir la raíz del proyecto para evitar que se creen carpetas en /app/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Asegurarse de que los directorios existan en la raíz
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "graphs"), exist_ok=True)

# Inicializamos el procesador de video
# Nota: La carga del modelo puede tomar tiempo al inicio
video_processor = VideoProcessor()

# Diccionario global para simular persistencia de estados de tareas (en un sistema real usaríamos Redis o DB)
tasks_status = {}

@app.get("/")
def read_root():
    return {"message": "Welcome to Tennis AI MVP API"}

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    status = tasks_status.get(task_id, {"status": "Not Found"})
    return status

@app.get("/download-video/{task_id}")
async def download_video(task_id: str):
    if task_id not in tasks_status:
        return {"error": "Tarea no encontrada"}
    
    task = tasks_status[task_id]
    if task["status"] != "completed":
        return {"error": "El video aún no ha terminado de procesarse o falló"}
    
    output_path = task["results"].get("output_video")
    if output_path and os.path.exists(output_path):
        return FileResponse(output_path, media_type="video/mp4", filename=f"processed_{task_id}.mp4")
    
    return {"error": "Archivo de video no encontrado"}

@app.get("/download-report/{task_id}")
async def download_report(task_id: str):
    if task_id not in tasks_status:
        return {"error": "Tarea no encontrada"}
    
    task = tasks_status[task_id]
    if task["status"] != "completed":
        return {"error": "El reporte aún no está listo o la tarea falló"}
    
    report_path = task["results"].get("report_image")
    if report_path and os.path.exists(report_path):
        return FileResponse(report_path, media_type="image/png", filename=f"report_{task_id}.png")
    
    return {"error": "Archivo de reporte no encontrado"}

@app.post("/upload-video/")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Crear un ID único para la tarea
    task_id = str(uuid.uuid4())
    
    # Asegurarse de que los directorios existan
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    
    # Rutas para el video
    file_extension = file.filename.split(".")[-1]
    input_path = os.path.join(DATA_RAW_DIR, f"{task_id}.{file_extension}")
    output_path = os.path.join(DATA_PROCESSED_DIR, f"{task_id}_processed.{file_extension}")
    
    # Guardar archivo localmente
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Registrar estado inicial
    tasks_status[task_id] = {"status": "processing", "input_path": input_path}
    
    # Lanzar procesamiento en segundo plano
    background_tasks.add_task(process_video_task, input_path, output_path, task_id)
    
    return {
        "task_id": task_id,
        "message": "El video está siendo procesado en segundo plano.",
        "input_path": input_path,
        "status_url": f"/task-status/{task_id}"
    }

def process_video_task(input_path: str, output_path: str, task_id: str):
    # Aquí es donde llamamos al servicio de procesamiento real
    try:
        results = video_processor.process_video(input_path, output_path, task_id)
        # Guardar resultados en el diccionario de estados
        tasks_status[task_id] = {
            "status": "completed",
            "results": results
        }
        print(f"Tarea {task_id} finalizada: {results['summary']}")
    except Exception as e:
        tasks_status[task_id] = {
            "status": "failed",
            "error": str(e)
        }
        print(f"Error procesando la tarea {task_id}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
