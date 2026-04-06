"""Tests de integración para los endpoints usados por el frontend HTML.

Se mockean VideoProcessor y OllamaLLMService a nivel de módulo
ANTES de crear el TestClient, para evitar descargar modelos YOLO.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# ── Crear mocks que se inyectan ANTES de importar app.main ──────────────
_mock_vp_class = MagicMock()
_mock_vp_instance = MagicMock()
_mock_vp_class.return_value = _mock_vp_instance

_mock_llm_class = MagicMock()
_mock_llm_instance = MagicMock()
_mock_llm_instance.check_connection.return_value = {
    "connected": True,
    "model_available": True,
    "available_models": ["llama3.2:3b"],
}
_mock_llm_class.return_value = _mock_llm_instance

# Parchear ANTES de que app.main se importe
with patch("app.services.video_processor.ObjectDetectionService"), \
     patch("app.services.video_processor.PoseEstimationService"), \
     patch("app.services.video_processor.ShotClassificationService"), \
     patch("app.services.video_processor.MetricsService"), \
     patch("app.services.video_processor.OllamaLLMService"):
    with patch.dict("sys.modules", {}):
        pass
    # Ahora parchear las clases que app.main instancia
    with patch("app.services.video_processor.VideoProcessor.__init__", return_value=None), \
         patch("app.services.llm_service.OllamaLLMService.__init__", return_value=None), \
         patch("app.services.llm_service.OllamaLLMService.check_connection",
               return_value={"connected": True, "model_available": True, "available_models": ["llama3.2:3b"]}):

        # Ahora importar app.main — las instancias de VideoProcessor y LLM
        # se crean con __init__ mockeado (no descargan nada)
        from app.main import app, tasks_status

from fastapi.testclient import TestClient
client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_tasks():
    """Limpia las tareas entre tests."""
    tasks_status.clear()
    yield
    tasks_status.clear()


# ---------------------------------------------------------------------------
# GET / — HTML frontend
# ---------------------------------------------------------------------------

class TestFrontendServing:

    def test_root_returns_html(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "<title>Tennis AI Analyzer</title>" in r.text

    def test_html_has_upload_elements(self):
        r = client.get("/")
        assert 'id="file-input"' in r.text
        assert 'id="upload-btn"' in r.text
        assert 'id="drop-zone"' in r.text

    def test_html_uses_absolute_urls(self):
        """El JS debe usar window.location.origin para URLs absolutas."""
        r = client.get("/")
        assert "window.location.origin" in r.text
        assert "API + '/upload-video'" in r.text

    def test_html_no_template_literals_in_js(self):
        """El JS no usa backticks (template literals) — mayor compatibilidad WebKit."""
        r = client.get("/")
        script_start = r.text.find("<script>")
        script_end = r.text.find("</script>")
        js_code = r.text[script_start:script_end]
        assert "`" not in js_code

    def test_html_formdata_includes_filename(self):
        """El FormData debe incluir el nombre del archivo explícitamente."""
        r = client.get("/")
        assert "selectedFile.name" in r.text


# ---------------------------------------------------------------------------
# POST /upload-video/
# ---------------------------------------------------------------------------

class TestUploadEndpoint:

    def test_upload_valid_mp4_returns_task_id(self):
        r = client.post(
            "/upload-video/",
            files={"file": ("test.mp4", b"fake-video", "video/mp4")},
        )
        assert r.status_code == 200
        data = r.json()
        assert "task_id" in data
        assert len(data["task_id"]) > 0
        assert "status_url" in data

    def test_upload_valid_mov(self):
        r = client.post(
            "/upload-video/",
            files={"file": ("clip.mov", b"fake", "video/quicktime")},
        )
        assert r.status_code == 200
        assert "task_id" in r.json()

    def test_upload_rejects_txt(self):
        r = client.post(
            "/upload-video/",
            files={"file": ("notes.txt", b"text", "text/plain")},
        )
        assert r.status_code == 400
        assert "no soportado" in r.json()["detail"]

    def test_upload_rejects_no_file(self):
        r = client.post("/upload-video/")
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /task-status/{task_id}
# ---------------------------------------------------------------------------

class TestTaskStatus:

    def test_unknown_task_404(self):
        r = client.get("/task-status/no-existe")
        assert r.status_code == 404

    def test_processing_task(self):
        tasks_status["t1"] = {"status": "processing"}
        r = client.get("/task-status/t1")
        assert r.status_code == 200
        assert r.json()["status"] == "processing"

    def test_completed_task(self):
        tasks_status["t2"] = {
            "status": "completed",
            "results": {"detected_shot": "Serve"},
        }
        r = client.get("/task-status/t2")
        assert r.status_code == 200
        assert r.json()["status"] == "completed"

    def test_failed_task(self):
        tasks_status["t3"] = {"status": "failed", "error": "bad video"}
        r = client.get("/task-status/t3")
        assert r.status_code == 200
        assert r.json()["status"] == "failed"
        assert "bad video" in r.json()["error"]


# ---------------------------------------------------------------------------
# GET /summary/{task_id}
# ---------------------------------------------------------------------------

class TestSummaryEndpoint:

    def test_summary_returns_all_fields(self):
        tasks_status["s1"] = {
            "status": "completed",
            "results": {
                "llm_summary": {
                    "success": True,
                    "summary": "Buen saque, mejorar rodilla.",
                    "model": "llama3.2:3b",
                    "generation_time": 5.2,
                },
                "detected_shot": "Serve",
                "metrics_summary": {
                    "avg_elbow_angle": 95.0,
                    "avg_knee_angle": 143.0,
                    "frames_with_pose": 510,
                },
            },
        }
        r = client.get("/summary/s1")
        assert r.status_code == 200
        data = r.json()
        assert data["llm_summary"]["summary"] == "Buen saque, mejorar rodilla."
        assert data["detected_shot"] == "Serve"
        assert data["metrics_summary"]["avg_elbow_angle"] == 95.0
        assert data["metrics_summary"]["frames_with_pose"] == 510

    def test_summary_not_ready_400(self):
        tasks_status["s2"] = {"status": "processing"}
        r = client.get("/summary/s2")
        assert r.status_code == 400

    def test_summary_missing_404(self):
        r = client.get("/summary/no-existe")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /graphs/{task_id} y /graphs/{task_id}/{index}
# ---------------------------------------------------------------------------

class TestGraphsEndpoints:

    def test_list_graphs(self):
        tasks_status["g1"] = {
            "status": "completed",
            "results": {"individual_graphs": ["/tmp/a.png", "/tmp/b.png", "/tmp/c.png"]},
        }
        r = client.get("/graphs/g1")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 3
        assert "/graphs/g1/0" in data["graphs"][0]
        assert "/graphs/g1/2" in data["graphs"][2]

    def test_graph_index_out_of_bounds(self):
        tasks_status["g2"] = {
            "status": "completed",
            "results": {"individual_graphs": ["/tmp/a.png"]},
        }
        r = client.get("/graphs/g2/5")
        assert r.status_code == 404

    def test_graph_file_not_on_disk(self):
        tasks_status["g3"] = {
            "status": "completed",
            "results": {"individual_graphs": ["/tmp/nonexistent_graph_xyz.png"]},
        }
        r = client.get("/graphs/g3/0")
        assert r.status_code == 404
        assert "no encontrado en disco" in r.json()["detail"]

    def test_graphs_not_ready(self):
        tasks_status["g4"] = {"status": "processing"}
        r = client.get("/graphs/g4")
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# GET /ollama-status
# ---------------------------------------------------------------------------

class TestOllamaStatus:

    def test_returns_connection_info(self):
        with patch("app.main.llm_service") as mock_llm:
            mock_llm.check_connection.return_value = {
                "connected": True,
                "model_available": True,
                "available_models": ["llama3.2:3b"],
            }
            r = client.get("/ollama-status")
        assert r.status_code == 200
        data = r.json()
        assert data["connected"] is True
