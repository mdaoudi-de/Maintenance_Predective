"""Tests unitaires de la prédiction par numéro de cycle.

Nécessitent un modèle entraîné ; ignorés proprement sinon.
"""
import pytest

from hydraulic_valve import config, predict

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="Modèle absent (lancer : python -m hydraulic_valve.train)",
)


def test_predict_cycle_structure():
    res = predict.predict_cycle(1)
    assert {"cycle", "prediction", "label", "probability_optimal"} <= set(res)
    assert res["prediction"] in (0, 1)
    assert 0.0 <= res["probability_optimal"] <= 1.0


def test_predict_out_of_range_raises():
    with pytest.raises(ValueError):
        predict.predict_cycle(0)
    with pytest.raises(ValueError):
        predict.predict_cycle(10**9)


def test_known_cycles():
    # Cycle 1 : valve = 100 -> optimale (1).
    # Cycle 212 : valve = 73 -> non optimale (0).
    assert predict.predict_cycle(1)["prediction"] == 1
    assert predict.predict_cycle(212)["prediction"] == 0
