"""Tests para MetricsService."""
import pytest
import numpy as np
from app.services.video_processor import MetricsService


def test_safe_float_valid():
    """Test de conversión de float válido."""
    assert MetricsService.safe_float(3.14) == 3.14
    assert MetricsService.safe_float(42) == 42.0


def test_safe_float_nan():
    """Test de conversión de NaN."""
    result = MetricsService.safe_float(np.nan, default=0.0)
    assert result == 0.0


def test_safe_float_inf():
    """Test de conversión de infinito."""
    result = MetricsService.safe_float(np.inf, default=0.0)
    assert result == 0.0
    result = MetricsService.safe_float(-np.inf, default=0.0)
    assert result == 0.0


def test_safe_float_invalid():
    """Test de conversión de valor inválido."""
    result = MetricsService.safe_float("not_a_number", default=5.0)
    assert result == 5.0


def test_smooth_signal_short():
    """Test de suavizado con pocos datos."""
    data = np.array([1.0, 2.0, 3.0])
    result = MetricsService.smooth_signal(data, window_length=11)
    # Con pocos datos debe devolver el original
    assert len(result) == len(data)


def test_smooth_signal_long():
    """Test de suavizado con suficientes datos."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    result = MetricsService.smooth_signal(data, window_length=11, polyorder=3)
    assert len(result) == len(data)
    assert isinstance(result, np.ndarray)
