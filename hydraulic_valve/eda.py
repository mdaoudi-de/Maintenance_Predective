"""Analyse exploratoire des données (EDA).

Génère les figures décrivant le jeu de données et la séparabilité des classes :
- répartition de la cible (global + train/test),
- signaux bruts PS2/FS1 pour un cycle optimal vs non optimal,
- séparation des classes selon les variables les plus discriminantes.
"""
from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config, data
from .train import build_dataset


def _plot_target_distribution(y: np.ndarray) -> None:
    train_idx, test_idx = data.train_test_indices(len(y))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Global
    counts = [np.sum(y == 0), np.sum(y == 1)]
    axes[0].bar(["Non optimale", "Optimale"], counts, color=["#e07b39", "#4a90e2"])
    axes[0].set_title(f"Répartition globale (n={len(y)})")
    for i, c in enumerate(counts):
        axes[0].text(i, c, str(c), ha="center", va="bottom")
    # Train vs test
    width = 0.35
    x = np.arange(2)
    tr = [np.sum(y[train_idx] == 0), np.sum(y[train_idx] == 1)]
    te = [np.sum(y[test_idx] == 0), np.sum(y[test_idx] == 1)]
    axes[1].bar(x - width / 2, tr, width, label=f"Train ({len(train_idx)})", color="#4a90e2")
    axes[1].bar(x + width / 2, te, width, label=f"Test ({len(test_idx)})", color="#e07b39")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Non optimale", "Optimale"])
    axes[1].set_title("Répartition train vs test (split imposé)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_target_distribution.png", dpi=130)
    plt.close(fig)


def _plot_example_signals(y: np.ndarray) -> None:
    sensors = data.load_raw_sensors()
    idx_opt = int(np.where(y == 1)[0][0])     # premier cycle optimal
    idx_bad = int(np.where(y == 0)[0][0])     # premier cycle non optimal
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, arr) in zip(axes, sensors.items()):
        fs = config.SENSORS[name]
        t = np.arange(arr.shape[1]) / fs
        ax.plot(t, arr[idx_opt], label=f"Optimale (cycle {idx_opt + 1})",
                color="#4a90e2", lw=0.8)
        ax.plot(t, arr[idx_bad], label=f"Non optimale (cycle {idx_bad + 1})",
                color="#e07b39", lw=0.8, alpha=0.8)
        ax.set_title(f"Signal {name} ({fs:.0f} Hz)")
        ax.set_xlabel("Temps (s)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_example_signals.png", dpi=130)
    plt.close(fig)


def _plot_class_separation(X, y) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # Boxplot de la variable la plus discriminante
    feat = "PS2_skew"
    axes[0].boxplot([X[feat][y == 0], X[feat][y == 1]],
                    tick_labels=["Non optimale", "Optimale"])
    axes[0].set_title(f"Séparation par {feat}")
    axes[0].set_ylabel(feat)
    # Nuage 2D des deux variables les plus discriminantes
    fx, fy = "PS2_skew", "PS2_kurtosis"
    for cls, color, lab in [(0, "#e07b39", "Non optimale"), (1, "#4a90e2", "Optimale")]:
        m = y == cls
        axes[1].scatter(X[fx][m], X[fy][m], s=8, alpha=0.5, color=color, label=lab)
    axes[1].set_xlabel(fx)
    axes[1].set_ylabel(fy)
    axes[1].set_title("Séparabilité des classes (2 variables)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_class_separation.png", dpi=130)
    plt.close(fig)


def run_eda() -> None:
    X, y = build_dataset()
    _plot_target_distribution(y)
    _plot_example_signals(y)
    _plot_class_separation(X, y)
    print(f">> Figures EDA sauvegardées dans : {config.FIGURES_DIR}")


if __name__ == "__main__":
    run_eda()
