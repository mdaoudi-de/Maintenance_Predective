"""Validation de robustesse et contrôle d'absence de fuite de données.

Deux analyses qui renforcent la crédibilité du résultat (100 % d'accuracy) :

1. **Test anti-fuite (permutation test)** : on ré-entraîne le modèle sur des
   étiquettes *mélangées*. S'il subsiste une performance élevée, c'est qu'il y a
   fuite ; si la performance chute au niveau du hasard, le modèle apprend un
   signal réel. On compare aussi à une régression logistique et à la classe
   majoritaire.

2. **Robustesse au bruit** : on simule des capteurs bruités *à l'inférence*
   (le modèle reste entraîné sur données propres) et on mesure la dégradation
   de l'accuracy — pour illustrer la limite « banc d'essai contrôlé ».
"""
from __future__ import annotations

import json

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config, data, features


def _rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )


def leakage_probe(n_shuffles: int = 5) -> dict:
    """Test de permutation : labels mélangés -> doit chuter vers le hasard."""
    sensors = data.load_raw_sensors()
    X = features.build_feature_matrix(sensors)
    y = data.get_target()
    tr, te = data.train_test_indices(len(y))
    Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]

    real = accuracy_score(yte, _rf().fit(Xtr, ytr).predict(Xte))
    rng = np.random.default_rng(config.RANDOM_STATE)
    shuffled = [accuracy_score(yte, _rf().fit(Xtr, rng.permutation(ytr)).predict(Xte))
                for _ in range(n_shuffles)]
    logreg = accuracy_score(yte, make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced",
                           random_state=config.RANDOM_STATE),
    ).fit(Xtr, ytr).predict(Xte))

    result = {
        "real_rf": float(real),
        "logreg": float(logreg),
        "shuffled_mean": float(np.mean(shuffled)),
        "shuffled_all": [float(s) for s in shuffled],
        "majority_baseline": float(max(yte.mean(), 1 - yte.mean())),
    }
    _plot_leakage(result)
    return result


def _plot_leakage(r: dict) -> None:
    labels = ["RF\n(réel)", "Régression\nlogistique", "Classe\nmajoritaire",
              "RF\n(labels mélangés)"]
    vals = [r["real_rf"], r["logreg"], r["majority_baseline"], r["shuffled_mean"]]
    colors = ["#4a90e2", "#4a90e2", "#999999", "#e07b39"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals, color=colors)
    ax.axhline(0.5, ls="--", c="k", alpha=0.5, label="hasard (0.5)")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy (test final)")
    ax.set_title("Contrôle anti-fuite : avec des labels mélangés, le modèle retombe au hasard")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "leakage_probe.png", dpi=130)
    plt.close(fig)


def noise_robustness(levels=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3)) -> dict:
    """Accuracy sur le test final quand on bruite les capteurs à l'inférence."""
    bundle = joblib.load(config.MODEL_PATH)
    model, meta = bundle["model"], bundle["metadata"]
    sensors = data.load_raw_sensors()
    y = data.get_target()
    _, te = data.train_test_indices(len(y))
    yte = y[te]
    stds = {name: float(arr.std()) for name, arr in sensors.items()}
    rng = np.random.default_rng(config.RANDOM_STATE)

    accs = []
    for lvl in levels:
        noisy = {}
        for name, arr in sensors.items():
            sub = arr[te].astype(np.float64)
            if lvl > 0:
                sub = sub + rng.normal(0, lvl * stds[name], size=sub.shape)
            noisy[name] = sub
        Xte = features.build_feature_matrix(noisy)[meta["feature_names"]]
        accs.append(float(accuracy_score(yte, model.predict(Xte))))

    result = {"levels": list(levels), "accuracy": accs}
    _plot_noise(result)
    return result


def _plot_noise(r: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([lvl * 100 for lvl in r["levels"]], r["accuracy"], "o-", color="#4a90e2")
    ax.set_xlabel("Bruit injecté (% de l'écart-type du capteur)")
    ax.set_ylabel("Accuracy (test final)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Robustesse au bruit capteur (modèle entraîné sur données propres)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "noise_robustness.png", dpi=130)
    plt.close(fig)


def run_all() -> dict:
    print(">> Test anti-fuite (permutation)...")
    leak = leakage_probe()
    print(f"   RF réel={leak['real_rf']:.3f} | labels mélangés={leak['shuffled_mean']:.3f} "
          f"| logreg={leak['logreg']:.3f} | majorité={leak['majority_baseline']:.3f}")
    print(">> Robustesse au bruit...")
    noise = noise_robustness()
    for lvl, acc in zip(noise["levels"], noise["accuracy"]):
        print(f"   bruit {lvl * 100:4.0f}% -> accuracy {acc:.3f}")
    out = {"leakage": leak, "noise": noise}
    with open(config.REPORTS_DIR / "robustness.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f">> Résultats sauvegardés : {config.REPORTS_DIR / 'robustness.json'}")
    return out


if __name__ == "__main__":
    run_all()
