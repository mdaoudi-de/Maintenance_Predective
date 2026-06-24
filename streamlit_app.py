"""Interface web (Streamlit) — prédiction de la condition de la valve.

Lancement (depuis la racine du projet) :
    streamlit run streamlit_app.py

Deux modes :
- « API » : interroge le service FastAPI (architecture découplée, défaut) ;
- « Local » : appelle directement le modèle (démo sans serveur).
"""
from __future__ import annotations

import os

import requests
import streamlit as st

from hydraulic_valve import config, data

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Condition de la valve", page_icon="🔧", layout="centered")
st.title("🔧 Maintenance prédictive — Condition de la valve")
st.caption(
    "Saisissez un numéro de cycle de production pour prédire si la condition "
    "de la valve est **optimale (100 %)** ou **non optimale**."
)

# Nombre total de cycles (pour borner la saisie).
try:
    n_total = int(len(data.get_target()))
except Exception:
    n_total = 2205

with st.sidebar:
    st.header("Paramètres")
    mode = st.radio(
        "Source de prédiction",
        ["API (FastAPI)", "Local (direct)"],
        help="« API » interroge le service FastAPI. « Local » appelle le modèle directement.",
    )
    if mode.startswith("API"):
        st.text_input("URL de l'API", value=API_URL, key="api_url")

cycle = st.number_input(
    "Numéro de cycle", min_value=1, max_value=n_total, value=min(2050, n_total), step=1
)


def _predict_via_api(cycle_number: int) -> dict:
    url = st.session_state.get("api_url", API_URL).rstrip("/")
    resp = requests.get(f"{url}/predict/{cycle_number}", timeout=30)
    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text)
        raise RuntimeError(f"Erreur API ({resp.status_code}) : {detail}")
    return resp.json()


def _predict_local(cycle_number: int) -> dict:
    from hydraulic_valve import predict  # import paresseux
    return predict.predict_cycle(cycle_number)


def _show_result(res: dict) -> None:
    if res["prediction"] == 1:
        st.success(f"✅ Valve **OPTIMALE** — cycle {res['cycle']}")
    else:
        st.error(f"⚠️ Valve **NON OPTIMALE** — cycle {res['cycle']}")
    prob = res.get("probability_optimal")
    if prob is not None:
        st.metric("Probabilité que la valve soit optimale", f"{prob * 100:.1f} %")
        st.progress(min(max(prob, 0.0), 1.0))


if st.button("Prédire", type="primary"):
    try:
        if mode.startswith("API"):
            result = _predict_via_api(int(cycle))
        else:
            result = _predict_local(int(cycle))
        _show_result(result)
    except requests.exceptions.RequestException:
        st.error(
            f"Impossible de joindre l'API ({st.session_state.get('api_url', API_URL)}). "
            "Démarrez-la (`uvicorn api.main:app`) ou choisissez le mode « Local »."
        )
    except FileNotFoundError as exc:
        st.error(f"Modèle introuvable : {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))

st.divider()
st.caption(
    "Modèle : Random Forest sur features (PS2 + FS1). "
    "Projet de maintenance prédictive — jeu de données UCI *Condition monitoring "
    "of hydraulic systems*."
)
