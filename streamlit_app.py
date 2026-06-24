"""Interface web (Streamlit) — prédiction de la condition de la valve.

Lancement (depuis la racine du projet) :
    streamlit run streamlit_app.py

Deux modes :
- « API » : interroge le service FastAPI (architecture découplée, défaut) ;
- « Local » : appelle directement le modèle (démo sans serveur).
"""
from __future__ import annotations

import os

import numpy as np
import requests
import streamlit as st

from hydraulic_valve import config, data

API_URL = os.environ.get("API_URL", "http://localhost:8000")
LABELS = {0: "Non optimale", 1: "Optimale"}

st.set_page_config(page_title="Condition de la valve", page_icon="🔧", layout="centered")


@st.cache_data(show_spinner=False)
def load_reference():
    """Valeurs réelles de la valve (depuis profile) + cible binaire."""
    valve = data.load_profile()[:, config.VALVE_COLUMN].astype(int)
    y = (valve == config.VALVE_OPTIMAL_VALUE).astype(int)
    return valve, y


@st.cache_data(show_spinner=False)
def example_cycles():
    """Quelques cycles d'exemple pris dans le jeu de TEST (prédiction réelle)."""
    valve, _ = load_reference()
    test = range(config.N_TRAIN, len(valve))
    optimal = [i + 1 for i in test if valve[i] == 100]
    degraded = [i + 1 for i in test if valve[i] != 100]
    return optimal, degraded


valve, y = load_reference()
N, N_TRAIN = len(y), config.N_TRAIN
opt_ex, deg_ex = example_cycles()

st.title("🔧 Maintenance prédictive — Condition de la valve")
st.caption(
    "Entrez un **numéro de cycle** de production : le modèle prédit si la valve "
    "de ce cycle est **optimale (100 %)** ou **non optimale**."
)

# --- Barre latérale : mode + aide ------------------------------------------
with st.sidebar:
    st.header("⚙️ Paramètres")
    mode = st.radio(
        "Source de prédiction",
        ["API (FastAPI)", "Local (direct)"],
        help="« API » interroge le service FastAPI. « Local » appelle le modèle directement.",
    )
    if mode.startswith("API"):
        st.text_input("URL de l'API", value=API_URL, key="api_url")
    st.divider()
    st.markdown(
        f"""
        **Quel numéro entrer ?**
        - Plage valide : **1 à {N}**
        - Cycles **1 – {N_TRAIN}** : données d'entraînement
        - Cycles **{N_TRAIN + 1} – {N}** : jeu de **test** (jamais vus par le
          modèle) → c'est le plus pertinent pour une démonstration
        """
    )

# --- Aide intégrée ----------------------------------------------------------
with st.expander("ℹ️ C'est quoi un « numéro de cycle » ?"):
    st.markdown(
        f"""
        Le banc d'essai hydraulique répète **{N} cycles** de production de 60 s.
        Chaque cycle (numéroté de 1 à {N}) est un enregistrement des capteurs
        **PS2** (pression) et **FS1** (débit). On ne saisit donc pas une mesure,
        mais le **numéro** d'un cycle déjà enregistré, et le modèle prédit l'état
        de sa valve à partir de ses signaux.
        """
    )

# --- Exemples cliquables ----------------------------------------------------
if "cycle_input" not in st.session_state:
    st.session_state.cycle_input = opt_ex[0] if opt_ex else min(2050, N)

st.write("**Exemples à tester** (issus du jeu de test) :")
c1, c2, c3 = st.columns(3)
if c1.button("✅ Valve optimale", use_container_width=True) and opt_ex:
    st.session_state.cycle_input = opt_ex[0]
if c2.button("⚠️ Valve dégradée", use_container_width=True) and deg_ex:
    st.session_state.cycle_input = deg_ex[0]
if c3.button("🎲 Cycle au hasard", use_container_width=True):
    st.session_state.cycle_input = int(np.random.default_rng().choice(opt_ex + deg_ex))

cycle = st.number_input(
    f"Numéro de cycle (1 – {N})", min_value=1, max_value=N, step=1, key="cycle_input"
)

# --- Contexte du cycle choisi ----------------------------------------------
real_v = int(valve[cycle - 1])
real_pred = 1 if real_v == 100 else 0
zone = "entraînement" if cycle <= N_TRAIN else "TEST (jamais vu par le modèle)"
st.info(
    f"**Cycle {cycle}** — échantillon : *{zone}*  ·  "
    f"valeur réelle de la valve : **{real_v} %** → {LABELS[real_pred]}"
)

with st.expander("📈 Voir les signaux capteurs de ce cycle"):
    try:
        ps2 = data.load_sensor("PS2")[cycle - 1]
        fs1 = data.load_sensor("FS1")[cycle - 1]
        st.caption("Pression PS2 (100 Hz)")
        st.line_chart(ps2[::10], height=150)
        st.caption("Débit FS1 (10 Hz)")
        st.line_chart(fs1, height=150)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Signaux indisponibles : {exc}")


# --- Prédiction -------------------------------------------------------------
def _predict_via_api(cycle_number: int) -> dict:
    url = st.session_state.get("api_url", API_URL).rstrip("/")
    resp = requests.get(f"{url}/predict/{cycle_number}", timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Erreur API ({resp.status_code}) : "
                           f"{resp.json().get('detail', resp.text)}")
    return resp.json()


def _predict_local(cycle_number: int) -> dict:
    from hydraulic_valve import predict  # import paresseux
    return predict.predict_cycle(cycle_number)


if st.button("🔮 Prédire", type="primary", use_container_width=True):
    try:
        res = (_predict_via_api if mode.startswith("API") else _predict_local)(int(cycle))
    except requests.exceptions.RequestException:
        st.error(
            f"Impossible de joindre l'API ({st.session_state.get('api_url', API_URL)}). "
            "Démarrez-la (`uvicorn api.main:app`) ou choisissez le mode « Local »."
        )
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        st.stop()

    pred, prob = res["prediction"], res.get("probability_optimal")
    col_pred, col_real = st.columns(2)
    with col_pred:
        st.markdown("**Prédiction du modèle**")
        (st.success if pred == 1 else st.error)(f"### {'✅' if pred == 1 else '⚠️'} {LABELS[pred]}")
        if prob is not None:
            st.metric("Probabilité d'être optimale", f"{prob * 100:.1f} %")
            st.progress(min(max(prob, 0.0), 1.0))
    with col_real:
        st.markdown("**Valeur réelle (vérité terrain)**")
        st.metric("Condition de la valve", f"{real_v} %", LABELS[real_pred])
        if pred == real_pred:
            st.success("Prédiction = réalité ✓")
        else:
            st.error("Prédiction ≠ réalité ✗")

st.divider()
st.caption(
    "Modèle : Random Forest sur features PS2 + FS1. "
    "Jeu de données UCI *Condition monitoring of hydraulic systems*."
)
