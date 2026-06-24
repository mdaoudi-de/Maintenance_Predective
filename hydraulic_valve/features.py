"""Extraction de features à partir des séries temporelles brutes.

Chaque cycle est une série temporelle (PS2 : 6000 points à 100 Hz, FS1 : 600
points à 10 Hz). Plutôt que d'alimenter un modèle avec des milliers de points
bruts (coûteux et bruité), on résume chaque cycle par un petit ensemble de
descripteurs statistiques et fréquentiels, calculés de façon **vectorisée** et
**identique à l'entraînement et à l'inférence** (gage de cohérence pour l'API).

Ces features sont interprétables, ce qui sert le volet « analyser les causes »
de l'énoncé (importance des variables).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config

# Nombre de bandes fréquentielles pour l'énergie spectrale.
N_FREQ_BANDS = 4


def _time_features(x: np.ndarray, fs: float, prefix: str) -> dict[str, np.ndarray]:
    """Descripteurs du domaine temporel, calculés ligne par ligne (axis=1)."""
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    q = np.percentile(x, [10, 25, 50, 75, 90], axis=1)

    # Pente de la tendance linéaire (régression de x sur le temps), vectorisée.
    n = x.shape[1]
    t = np.arange(n, dtype=np.float64)
    t_centered = t - t.mean()
    denom = (t_centered ** 2).sum()
    slope = ((x - mean[:, None]) @ t_centered) / denom

    feats = {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_min": x.min(axis=1),
        f"{prefix}_max": x.max(axis=1),
        f"{prefix}_range": x.max(axis=1) - x.min(axis=1),
        f"{prefix}_median": q[2],
        f"{prefix}_q10": q[0],
        f"{prefix}_q25": q[1],
        f"{prefix}_q75": q[3],
        f"{prefix}_q90": q[4],
        f"{prefix}_iqr": q[3] - q[1],
        f"{prefix}_rms": np.sqrt((x ** 2).mean(axis=1)),
        f"{prefix}_mad": np.abs(x - mean[:, None]).mean(axis=1),
        f"{prefix}_mean_abs_diff": np.abs(np.diff(x, axis=1)).mean(axis=1),
        f"{prefix}_skew": stats.skew(x, axis=1),
        f"{prefix}_kurtosis": stats.kurtosis(x, axis=1),
        f"{prefix}_slope": slope,
    }
    return feats


def _freq_features(x: np.ndarray, fs: float, prefix: str) -> dict[str, np.ndarray]:
    """Descripteurs du domaine fréquentiel via FFT (ligne par ligne)."""
    n = x.shape[1]
    # Retrait de la moyenne pour ne pas dominer le spectre par la composante DC.
    xc = x - x.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(xc, axis=1))
    power = spectrum ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    total_power = power.sum(axis=1) + 1e-12
    # Centroïde spectral (fréquence « moyenne » pondérée par l'énergie).
    centroid = (power * freqs[None, :]).sum(axis=1) / total_power
    # Fréquence dominante (hors composante 0).
    dominant = freqs[1:][np.argmax(power[:, 1:], axis=1)]

    feats = {
        f"{prefix}_spec_centroid": centroid,
        f"{prefix}_dominant_freq": dominant,
        f"{prefix}_spec_total_power": total_power,
    }
    # Ratio d'énergie par bande de fréquences.
    band_edges = np.linspace(0, len(freqs), N_FREQ_BANDS + 1, dtype=int)
    for b in range(N_FREQ_BANDS):
        lo, hi = band_edges[b], band_edges[b + 1]
        feats[f"{prefix}_band{b}_ratio"] = power[:, lo:hi].sum(axis=1) / total_power
    return feats


def extract_signal_features(x: np.ndarray, fs: float, prefix: str) -> dict[str, np.ndarray]:
    """Toutes les features (temps + fréquence) pour une matrice capteur."""
    feats = _time_features(x, fs, prefix)
    feats.update(_freq_features(x, fs, prefix))
    return feats


def build_feature_matrix(sensors: dict[str, np.ndarray]) -> pd.DataFrame:
    """Construit la matrice de features (n_cycles, n_features) pour tous les capteurs.

    Parameters
    ----------
    sensors : dict
        ``{nom_capteur: tableau (n_cycles, n_points)}``.
    """
    all_feats: dict[str, np.ndarray] = {}
    for name, arr in sensors.items():
        fs = config.SENSORS[name]
        all_feats.update(extract_signal_features(arr, fs, prefix=name))
    return pd.DataFrame(all_feats)
