"""Tests para ShotClassificationService."""
import pytest
from app.services.video_processor import ShotClassificationService


@pytest.fixture
def classifier():
    """Fixture para crear una instancia del clasificador."""
    return ShotClassificationService()


def test_classifier_initialization(classifier):
    """Test de inicialización del clasificador."""
    assert classifier is not None
    assert "forehand_signature" in classifier.ideal_patterns
    assert "serve_signature" in classifier.ideal_patterns


def test_classify_shot_frame_insufficient_history(classifier):
    """Test de clasificación con historial insuficiente."""
    pose_history = [
        {"RIGHT_WRIST": {"x": 0.5, "y": 0.5}, "NOSE": {"x": 0.5, "y": 0.3}}
    ]
    category, prob = classifier.classify_shot_frame(pose_history)
    assert category == "Iniciando..."
    assert prob == 0.5


def test_classify_shot_frame_with_sufficient_history(classifier):
    """Test de clasificación con historial suficiente."""
    pose_history = [
        {"RIGHT_WRIST": {"x": 0.5, "y": 0.5}, "NOSE": {"x": 0.5, "y": 0.3}}
        for _ in range(10)
    ]
    category, prob = classifier.classify_shot_frame(pose_history)
    assert isinstance(category, str)
    assert 0.0 <= prob <= 1.0


def test_classify_shot_frame_serve_detection(classifier):
    """Test de detección de saque."""
    # Simulamos mano por encima de la cabeza
    pose_history = [
        {"RIGHT_WRIST": {"x": 0.5, "y": 0.1}, "NOSE": {"x": 0.5, "y": 0.3}}
        for _ in range(10)
    ]
    category, prob = classifier.classify_shot_frame(pose_history)
    assert category == "Saque"
    assert prob > 0.5


def test_extract_features(classifier):
    """Test de extracción de características."""
    # Ahora requiere al menos 2 frames para calcular velocidades
    pose_history = [
        {
            "RIGHT_WRIST": {"x": 0.5, "y": 0.5},
            "RIGHT_ELBOW": {"x": 0.45, "y": 0.4},
            "RIGHT_SHOULDER": {"x": 0.4, "y": 0.3},
            "LEFT_SHOULDER": {"x": 0.6, "y": 0.3},
            "RIGHT_HIP": {"x": 0.45, "y": 0.6},
            "NOSE": {"x": 0.5, "y": 0.3}
        },
        {
            "RIGHT_WRIST": {"x": 0.52, "y": 0.52},
            "RIGHT_ELBOW": {"x": 0.47, "y": 0.42},
            "RIGHT_SHOULDER": {"x": 0.42, "y": 0.32},
            "LEFT_SHOULDER": {"x": 0.58, "y": 0.32},
            "RIGHT_HIP": {"x": 0.47, "y": 0.62},
            "NOSE": {"x": 0.5, "y": 0.3}
        }
    ]
    ball_pos = {"x": 0.6, "y": 0.6}
    features = classifier.extract_features(pose_history, ball_pos)
    assert features is not None
    assert len(features) > 10  # Más features ahora


def test_extract_features_no_ball(classifier):
    """Test de extracción de características sin bola."""
    pose_history = [
        {
            "RIGHT_WRIST": {"x": 0.5, "y": 0.5},
            "RIGHT_ELBOW": {"x": 0.45, "y": 0.4},
            "RIGHT_SHOULDER": {"x": 0.4, "y": 0.3},
            "LEFT_SHOULDER": {"x": 0.6, "y": 0.3},
            "RIGHT_HIP": {"x": 0.45, "y": 0.6},
            "NOSE": {"x": 0.5, "y": 0.3}
        },
        {
            "RIGHT_WRIST": {"x": 0.52, "y": 0.52},
            "RIGHT_ELBOW": {"x": 0.47, "y": 0.42},
            "RIGHT_SHOULDER": {"x": 0.42, "y": 0.32},
            "LEFT_SHOULDER": {"x": 0.58, "y": 0.32},
            "RIGHT_HIP": {"x": 0.47, "y": 0.62},
            "NOSE": {"x": 0.5, "y": 0.3}
        }
    ]
    features = classifier.extract_features(pose_history, ball_pos=None)
    assert features is not None
    assert len(features) > 10
