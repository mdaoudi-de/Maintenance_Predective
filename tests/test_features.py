"""Tests unitaires de l'extraction de features (sur signaux synthétiques).

On vérifie la justesse des descripteurs sur des signaux dont on connaît le
résultat attendu, sans dépendre du jeu de données réel.
"""
import numpy as np

from hydraulic_valve import features


def test_time_features_constant_signal():
    """Un signal constant a une moyenne égale à sa valeur et un écart-type nul."""
    x = np.full((3, 100), 5.0)
    feats = features._time_features(x, fs=100.0, prefix="S")
    assert np.allclose(feats["S_mean"], 5.0)
    assert np.allclose(feats["S_std"], 0.0)
    assert np.allclose(feats["S_range"], 0.0)
    assert np.allclose(feats["S_rms"], 5.0)


def test_slope_positive_for_increasing_ramp():
    """La pente doit être positive pour une rampe croissante."""
    x = np.tile(np.arange(100, dtype=float), (2, 1))
    feats = features._time_features(x, fs=1.0, prefix="S")
    assert np.all(feats["S_slope"] > 0)


def test_dominant_frequency_of_sine():
    """La fréquence dominante d'une sinusoïde doit être proche de sa fréquence."""
    fs = 100.0
    t = np.arange(0, 2, 1.0 / fs)
    freq = 5.0
    x = np.sin(2 * np.pi * freq * t)[None, :]
    feats = features._freq_features(x, fs=fs, prefix="S")
    assert abs(feats["S_dominant_freq"][0] - freq) < 0.5


def test_build_feature_matrix_shape_and_no_nan():
    """48 features (24 par capteur x 2), aucune valeur manquante."""
    rng = np.random.default_rng(0)
    sensors = {
        "PS2": rng.standard_normal((10, 6000)).astype(np.float32),
        "FS1": rng.standard_normal((10, 600)).astype(np.float32),
    }
    X = features.build_feature_matrix(sensors)
    assert X.shape == (10, 48)
    assert not X.isna().any().any()
    assert any(c.startswith("PS2_") for c in X.columns)
    assert any(c.startswith("FS1_") for c in X.columns)


def test_features_are_deterministic():
    """Deux extractions sur la même entrée donnent le même résultat."""
    rng = np.random.default_rng(1)
    sensors = {"PS2": rng.standard_normal((5, 6000)).astype(np.float32),
               "FS1": rng.standard_normal((5, 600)).astype(np.float32)}
    X1 = features.build_feature_matrix(sensors)
    X2 = features.build_feature_matrix(sensors)
    assert X1.equals(X2)
