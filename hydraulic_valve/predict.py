"""Prédiction de la condition de la valve pour un cycle donné.

Module réutilisé par l'application web : à partir d'un **numéro de cycle**, on
récupère les signaux PS2/FS1 correspondants, on calcule les mêmes features qu'à
l'entraînement, puis on applique le modèle sauvegardé.

Convention : le numéro de cycle est **1-based** (le cycle 1 = première ligne des
fichiers), comme dans `profile.txt`.
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from . import config, data, features

LABELS = {0: "Non optimale", 1: "Optimale"}


@lru_cache(maxsize=1)
def load_model() -> dict:
    """Charge le modèle entraîné et ses métadonnées (mis en cache)."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {config.MODEL_PATH}. "
            "Lancez d'abord l'entraînement : python -m hydraulic_valve.train"
        )
    return joblib.load(config.MODEL_PATH)


def _features_for_cycles(cycle_indices: np.ndarray) -> pd.DataFrame:
    """Construit la matrice de features pour des indices de cycle 0-based."""
    sensors = data.load_raw_sensors()
    subset = {name: arr[cycle_indices] for name, arr in sensors.items()}
    return features.build_feature_matrix(subset)


def predict_cycle(cycle_number: int) -> dict:
    """Prédit la condition de la valve pour un numéro de cycle (1-based).

    Returns
    -------
    dict avec ``cycle``, ``prediction`` (0/1), ``label`` et ``probability``
    (probabilité que la valve soit optimale).
    """
    bundle = load_model()
    model, meta = bundle["model"], bundle["metadata"]

    n_total = len(data.get_target())
    if not (1 <= cycle_number <= n_total):
        raise ValueError(
            f"Numéro de cycle hors limites : {cycle_number} "
            f"(doit être entre 1 et {n_total})."
        )

    idx = np.array([cycle_number - 1])  # passage en 0-based
    X = _features_for_cycles(idx)[meta["feature_names"]]

    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0, 1]) if hasattr(model, "predict_proba") else None
    return {
        "cycle": cycle_number,
        "prediction": pred,
        "label": LABELS[pred],
        "probability_optimal": proba,
    }


def predict_features(X: pd.DataFrame) -> np.ndarray:
    """Prédit à partir d'une matrice de features déjà construite."""
    bundle = load_model()
    model, meta = bundle["model"], bundle["metadata"]
    return model.predict(X[meta["feature_names"]])


if __name__ == "__main__":
    import sys
    cycle = int(sys.argv[1]) if len(sys.argv) > 1 else 2050
    print(predict_cycle(cycle))
