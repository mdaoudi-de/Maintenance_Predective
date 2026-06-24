"""Évaluation du modèle sur l'échantillon de test final (205 derniers cycles).

Produit les métriques (JSON) et les figures :
- matrice de confusion,
- courbe ROC,
- importance des variables (permutation) — pour analyser les causes.
"""
from __future__ import annotations

import json

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")  # rendu sans affichage (compatible serveur/CI)
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from . import config, data
from .train import build_dataset

CLASS_NAMES = ["Non optimale", "Optimale"]


def evaluate() -> dict:
    """Évalue le modèle sauvegardé sur le test final et sauvegarde les sorties."""
    bundle = joblib.load(config.MODEL_PATH)
    model, meta = bundle["model"], bundle["metadata"]

    X, y = build_dataset()
    X = X[meta["feature_names"]]
    _, test_idx = data.train_test_indices(len(y))
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": meta["model_name"],
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "classification_report": classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f">> Test final ({len(y_test)} cycles) — modèle : {meta['model_name']}")
    print(f"   Accuracy           : {metrics['accuracy']:.4f}")
    print(f"   Balanced accuracy  : {metrics['balanced_accuracy']:.4f}")
    print(f"   ROC-AUC            : {metrics['roc_auc']:.4f}")

    _plot_confusion(y_test, y_pred)
    _plot_roc(model, X_test, y_test)
    _plot_importance(model, X_test, y_test)
    print(f">> Métriques et figures sauvegardées dans : {config.REPORTS_DIR}")
    return metrics


def _plot_confusion(y_test, y_pred) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred), display_labels=CLASS_NAMES
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matrice de confusion — test final")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "confusion_matrix.png", dpi=130)
    plt.close(fig)


def _plot_roc(model, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("Courbe ROC — test final")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "roc_curve.png", dpi=130)
    plt.close(fig)


def _plot_importance(model, X_test, y_test, top_n: int = 15) -> None:
    """Importance des variables — pour analyser les causes.

    Pour un modèle à base d'arbres, on utilise l'importance native (réduction
    d'impureté), bien plus informative ici que la permutation : le modèle étant
    parfait et les features redondantes, permuter une seule variable ne dégrade
    presque pas la performance (les autres compensent).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        xlabel = "Importance (réduction d'impureté)"
        method = "impureté"
    else:
        result = permutation_importance(
            model, X_test, y_test, n_repeats=10,
            random_state=config.RANDOM_STATE, n_jobs=-1, scoring="roc_auc",
        )
        importances = result.importances_mean
        xlabel = "Baisse de ROC-AUC si la variable est permutée"
        method = "permutation"

    order = importances.argsort()[::-1][:top_n]
    names = np.array(X_test.columns)[order]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(order)), importances[order][::-1], color="#4a90e2")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel(xlabel)
    ax.set_title(f"Top {top_n} variables — importance ({method})")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "feature_importance.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    evaluate()
