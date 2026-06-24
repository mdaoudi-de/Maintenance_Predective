"""Chargement des données brutes (capteurs + profil) et définition de la cible.

Chaque fichier capteur est une matrice tabulée : 1 ligne = 1 cycle de production,
les colonnes = les points de mesure du cycle (60 s).

Les fichiers texte étant volumineux (PS2 : 2205 x 6000), ils sont mis en cache
au format `.npy` après le premier chargement pour accélérer les exécutions
suivantes.
"""
from __future__ import annotations

import numpy as np

from . import config


def load_sensor(name: str, use_cache: bool = True) -> np.ndarray:
    """Charge la matrice d'un capteur sous forme de tableau (n_cycles, n_points).

    Parameters
    ----------
    name : str
        Nom du capteur (ex. ``"PS2"``), doit correspondre à un fichier
        ``<name>.txt`` dans le dossier des données brutes.
    use_cache : bool
        Si vrai, lit/écrit une version ``.npy`` mise en cache.
    """
    cache_path = config.PROCESSED_DIR / f"{name}.npy"
    if use_cache and cache_path.exists():
        return np.load(cache_path)

    raw_path = config.RAW_DATA_DIR / f"{name}.txt"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Fichier capteur introuvable : {raw_path}. "
            "Vérifiez config.RAW_DATA_DIR."
        )
    # Les fichiers sont tabulés (séparateur \t), uniquement numériques.
    arr = np.loadtxt(raw_path, dtype=np.float32)
    if use_cache:
        np.save(cache_path, arr)
    return arr


def load_profile(use_cache: bool = True) -> np.ndarray:
    """Charge le fichier des cibles ``profile.txt`` (n_cycles, 5)."""
    cache_path = config.PROCESSED_DIR / "profile.npy"
    if use_cache and cache_path.exists():
        return np.load(cache_path)

    raw_path = config.RAW_DATA_DIR / config.PROFILE_FILE
    if not raw_path.exists():
        raise FileNotFoundError(f"Fichier profil introuvable : {raw_path}")
    arr = np.loadtxt(raw_path, dtype=np.float32)
    if use_cache:
        np.save(cache_path, arr)
    return arr


def get_target(use_cache: bool = True) -> np.ndarray:
    """Retourne la cible binaire : 1 = valve optimale (100 %), 0 = non optimale."""
    profile = load_profile(use_cache=use_cache)
    valve = profile[:, config.VALVE_COLUMN]
    y = (valve == config.VALVE_OPTIMAL_VALUE).astype(int)
    return y


def load_raw_sensors(use_cache: bool = True) -> dict[str, np.ndarray]:
    """Charge tous les capteurs définis dans la config."""
    return {name: load_sensor(name, use_cache=use_cache) for name in config.SENSORS}


def train_test_indices(n_samples: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Retourne les indices d'entraînement/test selon le split imposé.

    Les ``config.N_TRAIN`` premiers cycles -> entraînement,
    les cycles restants -> test final.
    """
    if n_samples is None:
        n_samples = len(get_target())
    idx = np.arange(n_samples)
    train_idx = idx[: config.N_TRAIN]
    test_idx = idx[config.N_TRAIN:]
    return train_idx, test_idx
