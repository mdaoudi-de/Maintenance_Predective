# Maintenance prédictive — Condition de la valve hydraulique

Prédiction, pour chaque cycle de production, de la **condition de la valve**
d'un système hydraulique : **optimale (100 %)** ou **non optimale** (90 / 80 / 73 %).
Projet basé sur le jeu de données *Condition monitoring of hydraulic systems*
(UCI / ZeMA).

## Problème
- **Tâche** : classification binaire (1 = valve optimale, 0 = non optimale).
- **Données** (1 ligne = 1 cycle de 60 s) :
  - `PS2.txt` — pression (bar), 100 Hz → 6000 points/cycle
  - `FS1.txt` — débit volumique (L/min), 10 Hz → 600 points/cycle
  - `profile.txt` — cibles ; la **2ᵉ colonne** est la condition de la valve
- **Protocole imposé** : 2000 premiers cycles = entraînement, 205 restants = test final.

## Démarche
1. **Chargement** des capteurs (avec cache `.npy`).
2. **Extraction de features** par cycle : descripteurs temporels (moyenne, écart-type,
   quantiles, RMS, pente, asymétrie, aplatissement…) et fréquentiels (FFT : centroïde
   spectral, fréquence dominante, énergie par bandes). → 48 features.
3. **Comparaison de modèles** (régression logistique, Random Forest, Histogram
   Gradient Boosting) par validation croisée stratifiée sur l'entraînement.
4. **Évaluation** du meilleur modèle sur le test final (jamais vu).

## Résultats (test final, 205 cycles)
| Métrique | Valeur |
|---|---|
| Accuracy | **1.000** |
| Balanced accuracy | **1.000** |
| ROC-AUC | **1.000** |

Classification parfaite, cohérente avec la littérature (la valve est une cible
« facile »). Les variables les plus discriminantes sont la **forme du signal de
pression PS2** (skew, kurtosis) et son contenu spectral.

## Installation
```bash
pip install -r requirements.txt
```

## Utilisation
Depuis la racine du projet :
```bash
# 1. Analyse exploratoire (figures dans reports/figures/)
python -m hydraulic_valve.eda

# 2. Entraînement + sélection de modèle (modèle dans models/)
python -m hydraulic_valve.train

# 3. Évaluation sur le test final (métriques + figures dans reports/)
python -m hydraulic_valve.evaluate

# 4. Prédiction pour un numéro de cycle (1-based)
python -m hydraulic_valve.predict 2050
```

## Rapport / restitution
Le support de restitution (démarche, EDA, résultats, limites) est un notebook :
- [`notebooks/rapport_valve.ipynb`](notebooks/rapport_valve.ipynb) — exécutable, figures intégrées ;
- [`reports/rapport_valve.html`](reports/rapport_valve.html) — version statique consultable sans Jupyter.

Pour le régénérer (nécessite `pip install -r requirements-dev.txt`) :
```bash
python notebooks/build_report.py                       # construit + exécute le notebook
python -m nbconvert --to html --output-dir reports notebooks/rapport_valve.ipynb
```

## Application web (API + interface)
Architecture découplée : un backend **FastAPI** sert les prédictions, un frontend
**Streamlit** les consomme.

```bash
# 1. Backend API (terminal 1) -> http://localhost:8000/docs
uvicorn api.main:app --reload

# 2. Interface Streamlit (terminal 2) -> http://localhost:8501
streamlit run streamlit_app.py
```
L'interface propose un mode **« Local »** (appel direct du modèle) qui fonctionne
sans démarrer l'API, pratique pour une démonstration rapide.

Principaux endpoints de l'API : `GET /predict/{n}`, `GET /health`,
`GET /model/info`, documentation interactive sur `/docs`.

## Tests
```bash
pip install -r requirements-dev.txt
pytest
```
Couvre le chargement des données, l'extraction de features (sur signaux
synthétiques), la prédiction et les endpoints de l'API.

## Structure
```
hydraulic_valve/      # package principal
  config.py           # chemins et constantes
  data.py             # chargement des données + split imposé
  features.py         # extraction de features (séries temporelles -> vecteur)
  train.py            # comparaison de modèles + entraînement
  evaluate.py         # évaluation sur le test final + figures
  predict.py          # prédiction à partir d'un numéro de cycle
  eda.py              # figures d'exploration
api/main.py           # API FastAPI
streamlit_app.py      # interface web Streamlit
tests/                # tests unitaires (pytest)
notebooks/            # rapport_valve.ipynb + build_report.py
data/processed/       # cache .npy
models/               # modèle entraîné (.joblib) + résumé
reports/figures/      # figures (EDA, confusion, ROC, importance)
```

## Feuille de route
- [x] Chargement & exploration (EDA)
- [x] Extraction de features
- [x] Modèle + évaluation sur le test final
- [x] Rapport / notebook de restitution
- [x] Tests unitaires
- [x] API FastAPI + interface Streamlit (prédiction par n° de cycle)
- [ ] Containerisation (Docker)
- [ ] Versionnage modèle + dataset (DVC)
- [ ] CI/CD (GitHub Actions)
- [ ] Monitoring (Prometheus / Grafana)
# Maintenance_Predective
