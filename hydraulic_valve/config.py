"""Configuration centrale du projet.

Tous les chemins et constantes utilisés par le pipeline sont définis ici, afin
qu'un tiers puisse adapter le projet à son environnement en un seul endroit.
"""
from __future__ import annotations

from pathlib import Path

# --- Arborescence du projet -------------------------------------------------
# Racine = dossier parent du package `hydraulic_valve`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Données brutes fournies (dézippées depuis data_subset.zip).
# Le dossier d'origine du jeu de données UCI.
RAW_DATA_DIR = PROJECT_ROOT / "condition+monitoring+of+hydraulic+systems (1)"

# Données intermédiaires (cache .npy pour accélérer les rechargements).
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Sorties.
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Définition du problème -------------------------------------------------
# Capteurs utilisés (imposés par l'énoncé : data_subset = PS2, FS1, profile).
# nom -> fréquence d'échantillonnage en Hz.
SENSORS: dict[str, float] = {
    "PS2": 100.0,   # Pression (bar), 6000 points / cycle
    "FS1": 10.0,    # Débit volumique (L/min), 600 points / cycle
}

# Fichier des cibles. La valve est la 2e colonne (index 1).
PROFILE_FILE = "profile.txt"
VALVE_COLUMN = 1            # index 0-based de la colonne "valve condition"
VALVE_OPTIMAL_VALUE = 100   # 100 % = comportement de commutation optimal

# --- Protocole d'évaluation (imposé par l'énoncé) ---------------------------
# Les 2000 premiers cycles servent à l'entraînement, le reste au test final.
N_TRAIN = 2000

# Graine aléatoire pour la reproductibilité.
RANDOM_STATE = 42

# Chemins des artefacts produits.
MODEL_PATH = MODELS_DIR / "valve_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
