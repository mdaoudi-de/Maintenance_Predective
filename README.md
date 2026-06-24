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
5. **Validation** : contrôle anti-fuite (test de permutation) et robustesse au bruit.

## Résultats (test final, 205 cycles)
| Métrique | Valeur |
|---|---|
| Accuracy | **1.000** |
| Balanced accuracy | **1.000** |
| ROC-AUC | **1.000** |

Classification parfaite, cohérente avec la littérature (la valve est une cible
« facile »). Les variables les plus discriminantes sont la **forme du signal de
pression PS2** (skew, kurtosis) et son contenu spectral.

**Résultat validé** (section 6 du rapport) : **aucune fuite de données** — avec
des étiquettes mélangées, l'accuracy retombe au hasard (~0,5). La pression PS2
seule suffit déjà à atteindre 100 %, ce qui explique la performance. Le modèle
reste robuste jusqu'à ~5 % de bruit capteur puis se dégrade — d'où l'intérêt du
**monitoring** de la dérive en production.

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

# 3b. Validation : test anti-fuite (permutation) + robustesse au bruit
python -m hydraulic_valve.robustness

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
L'interface est **guidée** : aide intégrée sur le numéro de cycle, **boutons
d'exemples** (valve optimale / dégradée / cycle de test au hasard), affichage de
la **valeur réelle** du cycle (comparaison prédiction vs réalité) et **aperçu des
signaux** PS2/FS1. Elle propose aussi un mode **« Local »** (appel direct du
modèle) qui fonctionne sans démarrer l'API.

Principaux endpoints de l'API : `GET /predict/{n}`, `GET /health`,
`GET /model/info`, documentation interactive sur `/docs`.

## Tests
```bash
pip install -r requirements-dev.txt
pytest
```
Couvre le chargement des données, l'extraction de features (sur signaux
synthétiques), la prédiction et les endpoints de l'API.

## Conteneurisation (Docker)
Tout l'environnement (API + interface + monitoring) démarre en une commande :
```bash
docker compose up --build
```
| Service | URL | Rôle |
|---|---|---|
| API | http://localhost:8000/docs | prédictions (FastAPI) |
| Streamlit | http://localhost:8501 | interface web |
| Prometheus | http://localhost:9090 | collecte des métriques |
| Grafana | http://localhost:3001 | tableaux de bord (admin / admin) |

Le modèle et les données sont montés en **volumes** (non inclus dans l'image, qui
reste légère et reconstructible). Récupérez-les au préalable via `dvc pull` ou en
(ré)entraînant le modèle (`python -m hydraulic_valve.train`).

## Versionnage des données et du modèle (DVC)
Le sous-ensemble de données (`data/raw`) et le modèle (`models/valve_model.joblib`)
sont suivis par **DVC** : les pointeurs `*.dvc` sont versionnés dans git, les
fichiers volumineux restent hors git. Le jeu de données complet d'origine (531 Mo)
n'est pas versionné.
```bash
dvc remote add -d storage <url_du_stockage>   # S3, Google Drive, dossier local…
dvc push        # envoyer données + modèle vers le stockage distant
dvc pull        # les récupérer sur une autre machine
```

## Monitoring (Prometheus / Grafana)
L'API expose ses métriques sur `/metrics` : latence, nombre de requêtes, et un
compteur `valve_predictions_total` par classe (optimale / non optimale).
Prometheus les collecte ; Grafana les affiche via un tableau de bord
pré-provisionné (« Maintenance prédictive — Condition de la valve »). L'ensemble
est lancé par `docker compose up`.

## Intégration continue (CI/CD)
Workflow GitHub Actions [`.github/workflows/ci.yml`](.github/workflows/ci.yml) :
à chaque push / pull request sur `main`, exécution du lint (ruff) puis des tests
(pytest), suivie du build de l'image Docker. Les tests dépendant des données ou
du modèle sont ignorés proprement en CI (artefacts gérés par DVC).

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
  robustness.py       # contrôle anti-fuite + robustesse au bruit
api/main.py           # API FastAPI (+ métriques Prometheus)
streamlit_app.py      # interface web Streamlit
tests/                # tests unitaires (pytest)
notebooks/            # rapport_valve.ipynb + build_report.py
monitoring/           # prometheus.yml + provisioning & dashboard Grafana
.github/workflows/    # CI/CD (GitHub Actions)
Dockerfile            # image de service
docker-compose.yml    # API + Streamlit + Prometheus + Grafana
data/raw.dvc          # pointeur DVC du dataset (data/raw versionné par DVC)
data/processed/       # cache .npy (local)
models/               # modèle entraîné (.joblib, versionné DVC) + résumé
reports/figures/      # figures (EDA, confusion, ROC, importance, validation)
```

## Feuille de route
- [x] Chargement & exploration (EDA)
- [x] Extraction de features
- [x] Modèle + évaluation sur le test final
- [x] Rapport / notebook de restitution
- [x] Tests unitaires
- [x] API FastAPI + interface Streamlit (prédiction par n° de cycle)
- [x] Containerisation (Docker)
- [x] Versionnage modèle + dataset (DVC)
- [x] CI/CD (GitHub Actions)
- [x] Monitoring (Prometheus / Grafana)
# Maintenance_Predective
