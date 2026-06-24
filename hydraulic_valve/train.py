"""Entraînement et sélection du modèle de prédiction de la condition de la valve.

Démarche :
1. Construire la matrice de features à partir des capteurs PS2 et FS1.
2. Comparer plusieurs modèles par validation croisée stratifiée, **uniquement
   sur les 2000 cycles d'entraînement** (le test final reste intouché).
3. Sélectionner le meilleur modèle (ROC-AUC moyen), le ré-entraîner sur la
   totalité des 2000 cycles et le sauvegarder avec ses métadonnées.
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, data, features


def build_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    """Charge les capteurs, construit les features et la cible."""
    sensors = data.load_raw_sensors()
    X = features.build_feature_matrix(sensors)
    y = data.get_target()
    return X, y


def get_models() -> dict[str, object]:
    """Modèles candidats. La régression logistique sert de référence (baseline)."""
    rs = config.RANDOM_STATE
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=rs)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=400, class_weight="balanced",
            random_state=rs, n_jobs=-1,
        ),
        "hist_gb": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, random_state=rs,
        ),
    }


def compare_models(X_train: pd.DataFrame, y_train: np.ndarray) -> dict[str, dict]:
    """Validation croisée stratifiée (5 plis) de chaque modèle sur l'entraînement."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    scoring = ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc"]
    results: dict[str, dict] = {}
    for name, model in get_models().items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring,
                                n_jobs=-1)
        results[name] = {m: float(scores[f"test_{m}"].mean()) for m in scoring}
        results[name].update({f"{m}_std": float(scores[f"test_{m}"].std())
                              for m in scoring})
    return results


def select_best(cv_results: dict[str, dict], metric: str = "roc_auc") -> str:
    """Sélectionne le modèle au meilleur score moyen pour la métrique donnée."""
    return max(cv_results, key=lambda n: cv_results[n][metric])


def train() -> dict:
    """Pipeline complet d'entraînement. Retourne un résumé sérialisable."""
    print(">> Construction du jeu de features...")
    X, y = build_dataset()
    train_idx, test_idx = data.train_test_indices(len(y))
    X_train, y_train = X.iloc[train_idx], y[train_idx]

    print(f"   Features : {X.shape[1]} | Train : {len(train_idx)} cycles | "
          f"Test : {len(test_idx)} cycles")
    print(">> Comparaison des modèles (validation croisée 5 plis)...")
    cv_results = compare_models(X_train, y_train)
    for name, sc in cv_results.items():
        print(f"   {name:14} acc={sc['accuracy']:.4f}  bal_acc="
              f"{sc['balanced_accuracy']:.4f}  f1={sc['f1_macro']:.4f}  "
              f"auc={sc['roc_auc']:.4f}")

    best_name = select_best(cv_results)
    print(f">> Meilleur modèle : {best_name}")

    best_model = get_models()[best_name]
    best_model.fit(X_train, y_train)

    metadata = {
        "model_name": best_name,
        "feature_names": list(X.columns),
        "sensors": config.SENSORS,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "cv_results": cv_results,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
    }
    joblib.dump({"model": best_model, "metadata": metadata}, config.MODEL_PATH)
    with open(config.MODELS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f">> Modèle sauvegardé : {config.MODEL_PATH}")
    return metadata


if __name__ == "__main__":
    train()
