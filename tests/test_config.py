"""Tests para el módulo de configuración."""
import pytest
from app.config import (
    CONFIDENCE_THRESHOLD,
    POSE_HISTORY_BUFFER_SIZE,
    METRICS_CONFIG,
    MOVEMENT_SIGNATURES
)


def test_confidence_threshold():
    """Verifica que el umbral de confianza esté en rango válido."""
    assert 0.0 <= CONFIDENCE_THRESHOLD <= 1.0


def test_pose_history_buffer_size():
    """Verifica que el tamaño del buffer sea positivo."""
    assert POSE_HISTORY_BUFFER_SIZE > 0
    assert isinstance(POSE_HISTORY_BUFFER_SIZE, int)


def test_metrics_config_structure():
    """Verifica que METRICS_CONFIG tenga la estructura correcta."""
    assert isinstance(METRICS_CONFIG, dict)
    for metric_name, config in METRICS_CONFIG.items():
        assert "title" in config
        assert "ylabel" in config
        assert "range" in config
        assert isinstance(config["range"], tuple)
        assert len(config["range"]) == 2


def test_movement_signatures():
    """Verifica que las firmas de movimiento existan."""
    assert "forehand_signature" in MOVEMENT_SIGNATURES
    assert "serve_signature" in MOVEMENT_SIGNATURES
    assert len(MOVEMENT_SIGNATURES["forehand_signature"]) > 0
    assert len(MOVEMENT_SIGNATURES["serve_signature"]) > 0
