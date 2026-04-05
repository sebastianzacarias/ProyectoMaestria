import httpx
import logging
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger(__name__)


class OllamaLLMService:
    """Servicio para generar resúmenes usando Ollama (llama3.2:3b) en local."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b", timeout: int = 60):
        """
        Inicializa el servicio de LLM con Ollama.

        Args:
            base_url: URL base de Ollama (default: http://localhost:11434)
            model: Modelo a usar (default: llama3.2:3b)
            timeout: Timeout en segundos para las requests (default: 60)
        """
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        logger.info(f"OllamaLLMService inicializado - URL: {base_url}, Modelo: {model}")

    def _build_prompt(self, frames_data: List[Dict], metrics_summary: Dict, detected_shot: str) -> str:
        """
        Construye el prompt para el LLM basado en las métricas del análisis.

        Args:
            frames_data: Lista de diccionarios con datos de cada frame
            metrics_summary: Resumen de métricas promedio
            detected_shot: Tipo de golpe detectado

        Returns:
            String con el prompt formateado
        """
        # Analizar distribución de movimientos
        movements = [f.get("movement", "En Espera") for f in frames_data if f.get("pose_detected")]
        movement_counts = {}
        for mov in movements:
            # Limpiar nombre de movimiento (remover "(Impacto)")
            clean_mov = mov.split(" (")[0] if mov else "En Espera"
            movement_counts[clean_mov] = movement_counts.get(clean_mov, 0) + 1

        # Calcular estadísticas de velocidad y aceleración
        frames_with_pose = [f for f in frames_data if f.get("pose_detected")]
        wrist_speeds = [f.get("wrist_speed", 0.0) for f in frames_with_pose]
        com_stabilities = [f.get("com_stability", 0.0) for f in frames_with_pose]

        avg_wrist_speed = sum(wrist_speeds) / len(wrist_speeds) if wrist_speeds else 0.0
        max_wrist_speed = max(wrist_speeds) if wrist_speeds else 0.0
        avg_com_stability = sum(com_stabilities) / len(com_stabilities) if com_stabilities else 0.0

        # Construir el prompt
        prompt = f"""Eres un entrenador experto de tenis. Analiza los siguientes datos de un video de tenis y proporciona un resumen técnico conciso y útil para el jugador.

DATOS DEL ANÁLISIS:
- Total de frames analizados: {len(frames_data)}
- Frames con pose detectada: {len(frames_with_pose)}
- Golpe predominante detectado: {detected_shot}

DISTRIBUCIÓN DE MOVIMIENTOS:
{json.dumps(movement_counts, indent=2, ensure_ascii=False)}

MÉTRICAS BIOMECÁNICAS PROMEDIO:
- Ángulo de codo: {metrics_summary.get('avg_elbow_angle', 0.0):.1f}° (ideal: 140-170°)
- Ángulo de rodilla: {metrics_summary.get('avg_knee_angle', 0.0):.1f}° (ideal: 120-160°)

MÉTRICAS CINEMÁTICAS:
- Velocidad promedio de muñeca: {avg_wrist_speed:.4f}
- Velocidad máxima de muñeca: {max_wrist_speed:.4f}
- Estabilidad promedio del centro de masa: {avg_com_stability:.4f} (menor = más estable)

TAREAS:
1. Resume el rendimiento técnico del jugador basándote en las métricas
2. Identifica los aspectos positivos de su técnica
3. Señala áreas de mejora específicas (ángulos, estabilidad, velocidad)
4. Proporciona 2-3 recomendaciones concretas para mejorar

Formato del resumen: Máximo 300 palabras, dividido en secciones claras (Resumen General, Puntos Fuertes, Áreas de Mejora, Recomendaciones)."""

        return prompt

    def generate_summary(self, frames_data: List[Dict], metrics_summary: Dict,
                        detected_shot: str) -> Dict[str, Any]:
        """
        Genera un resumen interpretativo usando Ollama.

        Args:
            frames_data: Lista de datos de cada frame
            metrics_summary: Resumen de métricas
            detected_shot: Tipo de golpe detectado

        Returns:
            Diccionario con el resumen generado y metadatos
        """
        try:
            # Construir prompt
            prompt = self._build_prompt(frames_data, metrics_summary, detected_shot)

            # Payload para Ollama API
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }

            logger.info(f"Enviando request a Ollama - Modelo: {self.model}")

            # Llamar a Ollama
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                response.raise_for_status()

                result = response.json()
                summary_text = result.get("response", "")

                logger.info("Resumen generado exitosamente")

                return {
                    "success": True,
                    "summary": summary_text,
                    "model": self.model,
                    "tokens_used": result.get("eval_count", 0),
                    "generation_time": result.get("total_duration", 0) / 1e9  # nanosegundos a segundos
                }

        except httpx.ConnectError:
            logger.error(f"No se pudo conectar a Ollama en {self.base_url}")
            return {
                "success": False,
                "error": f"No se pudo conectar a Ollama. Asegúrate de que Ollama esté corriendo en {self.base_url}",
                "summary": self._generate_fallback_summary(frames_data, metrics_summary, detected_shot)
            }
        except httpx.TimeoutException:
            logger.error("Timeout al generar resumen con Ollama")
            return {
                "success": False,
                "error": "Timeout al generar el resumen",
                "summary": self._generate_fallback_summary(frames_data, metrics_summary, detected_shot)
            }
        except Exception as e:
            logger.exception(f"Error generando resumen con Ollama: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": self._generate_fallback_summary(frames_data, metrics_summary, detected_shot)
            }

    def _generate_fallback_summary(self, frames_data: List[Dict],
                                   metrics_summary: Dict, detected_shot: str) -> str:
        """
        Genera un resumen básico sin LLM cuando Ollama no está disponible.

        Args:
            frames_data: Lista de datos de cada frame
            metrics_summary: Resumen de métricas
            detected_shot: Tipo de golpe detectado

        Returns:
            Resumen básico en texto
        """
        frames_with_pose = [f for f in frames_data if f.get("pose_detected")]

        # Analizar movimientos
        movements = [f.get("movement", "En Espera") for f in frames_with_pose]
        movement_counts = {}
        for mov in movements:
            clean_mov = mov.split(" (")[0] if mov else "En Espera"
            movement_counts[clean_mov] = movement_counts.get(clean_mov, 0) + 1

        top_movements = sorted(movement_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        summary = f"""RESUMEN TÉCNICO (Generado sin LLM)

RESUMEN GENERAL:
Se analizaron {len(frames_data)} frames, de los cuales {len(frames_with_pose)} tuvieron detección de pose exitosa.
Golpe predominante: {detected_shot}

MOVIMIENTOS DETECTADOS:
"""
        for mov, count in top_movements:
            percentage = (count / len(frames_with_pose) * 100) if frames_with_pose else 0
            summary += f"- {mov}: {count} frames ({percentage:.1f}%)\n"

        avg_elbow = metrics_summary.get('avg_elbow_angle', 0.0)
        avg_knee = metrics_summary.get('avg_knee_angle', 0.0)

        summary += f"""
MÉTRICAS BIOMECÁNICAS:
- Ángulo promedio de codo: {avg_elbow:.1f}° (rango ideal: 140-170°)
- Ángulo promedio de rodilla: {avg_knee:.1f}° (rango ideal: 120-160°)

RECOMENDACIONES:
- Revisar las gráficas individuales para análisis detallado de cada métrica
- Comparar los ángulos medidos con los rangos ideales
- Observar la estabilidad del centro de masa durante los golpes
- Analizar la velocidad de muñeca en los momentos de impacto

NOTA: Para obtener un análisis más detallado, asegúrate de que Ollama esté corriendo (ollama serve) con el modelo llama3.2:3b."""

        return summary

    def check_connection(self) -> Dict[str, Any]:
        """
        Verifica la conexión con Ollama y disponibilidad del modelo.

        Returns:
            Diccionario con estado de la conexión
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                # Verificar que Ollama esté corriendo
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()

                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]

                model_available = self.model in model_names

                return {
                    "connected": True,
                    "model_available": model_available,
                    "available_models": model_names,
                    "message": "Conectado a Ollama" if model_available else f"Modelo {self.model} no encontrado"
                }
        except Exception as e:
            return {
                "connected": False,
                "model_available": False,
                "error": str(e),
                "message": "No se pudo conectar a Ollama. Verifica que esté corriendo (ollama serve)"
            }
