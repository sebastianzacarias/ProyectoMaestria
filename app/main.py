from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.services.video_processor import VideoProcessor
from app.services.llm_service import OllamaLLMService
import os
import shutil
import uuid
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tennis AI MVP API")

# Definir la raíz del proyecto para evitar que se creen carpetas en /app/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Asegurarse de que los directorios existan en la raíz
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "graphs"), exist_ok=True)

# Inicializamos el procesador de video y servicio LLM
# Nota: La carga del modelo puede tomar tiempo al inicio
try:
    video_processor = VideoProcessor()
    llm_service = OllamaLLMService()
    logger.info("VideoProcessor y LLM Service inicializados correctamente")
except Exception as e:
    logger.error(f"Error al inicializar servicios: {e}")
    raise

# Diccionario global para simular persistencia de estados de tareas (en un sistema real usaríamos Redis o DB)
tasks_status = {}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Sirve el frontend HTML."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, encoding="utf-8") as f:
        return f.read()

@app.get("/graphs/{task_id}", response_class=JSONResponse)
async def list_graphs(task_id: str):
    """Lista las URLs de las gráficas individuales de una tarea completada."""
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    task = tasks_status[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="La tarea aún no está completada")
    graphs = task["results"].get("individual_graphs", [])
    urls = [f"/graphs/{task_id}/{i}" for i in range(len(graphs))]
    return {"graphs": urls, "count": len(urls)}

@app.get("/graphs/{task_id}/{index}")
async def get_graph(task_id: str, index: int):
    """Sirve una gráfica individual por índice."""
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    task = tasks_status[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="La tarea aún no está completada")
    graphs = task["results"].get("individual_graphs", [])
    if index < 0 or index >= len(graphs):
        raise HTTPException(status_code=404, detail=f"Gráfica {index} no encontrada")
    graph_path = graphs[index]
    if not os.path.exists(graph_path):
        raise HTTPException(status_code=404, detail="Archivo de gráfica no encontrado en disco")
    return FileResponse(graph_path, media_type="image/png")

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Obtiene el estado de una tarea de procesamiento."""
    status = tasks_status.get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Tarea {task_id} no encontrada")
    return status

@app.get("/download-video/{task_id}")
async def download_video(task_id: str):
    """Descarga el video procesado de una tarea."""
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")

    task = tasks_status[task_id]
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"El video no está disponible. Estado actual: {task['status']}"
        )

    output_path = task["results"].get("output_video")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Archivo de video no encontrado")

    return FileResponse(output_path, media_type="video/mp4", filename=f"processed_{task_id}.mp4")

@app.get("/download-report/{task_id}")
async def download_report(task_id: str):
    """Descarga el reporte (gráficas) de una tarea."""
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")

    task = tasks_status[task_id]
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"El reporte no está disponible. Estado actual: {task['status']}"
        )

    report_path = task["results"].get("report_image")
    if not report_path or not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Archivo de reporte no encontrado")

    return FileResponse(report_path, media_type="image/png", filename=f"report_{task_id}.png")

@app.get("/summary/{task_id}")
async def get_summary(task_id: str):
    """
    Obtiene el resumen generado por LLM para una tarea completada.

    Returns:
        JSON con el resumen interpretativo del análisis
    """
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")

    task = tasks_status[task_id]
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"El resumen no está disponible. Estado actual: {task['status']}"
        )

    llm_summary = task["results"].get("llm_summary")
    if not llm_summary:
        raise HTTPException(status_code=404, detail="Resumen no encontrado")

    return JSONResponse(content={
        "task_id": task_id,
        "llm_summary": llm_summary,
        "detected_shot": task["results"].get("detected_shot"),
        "metrics_summary": task["results"].get("metrics_summary")
    })

@app.get("/ollama-status")
async def check_ollama_status():
    """
    Verifica el estado de la conexión con Ollama y disponibilidad del modelo.

    Returns:
        JSON con el estado de Ollama
    """
    status = llm_service.check_connection()
    return JSONResponse(content=status)

@app.post("/upload-video")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Sube un video para su procesamiento. El análisis se ejecuta en background.
    """
    # Validar tipo de archivo
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nombre de archivo inválido")

    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in ["mp4", "avi", "mov", "mkv"]:
        raise HTTPException(
            status_code=400,
            detail=f"Formato de video no soportado: {file_extension}. Use mp4, avi, mov o mkv"
        )

    task_id = str(uuid.uuid4())
    logger.info(f"Nueva tarea de upload: {task_id}, archivo: {file.filename}")

    try:
        # Asegurarse de que los directorios existan
        os.makedirs(DATA_RAW_DIR, exist_ok=True)
        os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

        # Rutas para el video
        input_path = os.path.join(DATA_RAW_DIR, f"{task_id}.{file_extension}")
        output_path = os.path.join(DATA_PROCESSED_DIR, f"{task_id}_processed.{file_extension}")

        # Guardar archivo localmente
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Video guardado: {input_path}")

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
    except Exception as e:
        logger.error(f"Error al subir video {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")

def process_video_task(input_path: str, output_path: str, task_id: str):
    """Procesa un video en background."""
    logger.info(f"Iniciando procesamiento de tarea: {task_id}")
    try:
        results = video_processor.process_video(input_path, output_path, task_id)
        tasks_status[task_id] = {
            "status": "completed",
            "results": results
        }
        logger.info(f"Tarea {task_id} completada: {results['summary']}")
    except ValueError as e:
        logger.error(f"Error de validación en tarea {task_id}: {e}")
        tasks_status[task_id] = {
            "status": "failed",
            "error": f"Error de validación: {str(e)}"
        }
    except Exception as e:
        logger.exception(f"Error inesperado en tarea {task_id}")
        tasks_status[task_id] = {
            "status": "failed",
            "error": f"Error al procesar el video: {str(e)}"
        }

# Montar archivos estáticos DESPUÉS de todas las rutas.
# En Starlette, un mount registrado antes de las rutas puede
# interferir con el routing y provocar 404 en los endpoints de la API.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
