"""API FastAPI — prédiction de la condition de la valve à partir d'un n° de cycle.

Lancement (depuis la racine du projet) :
    uvicorn api.main:app --reload
Documentation interactive : http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from hydraulic_valve import __version__, config, data, predict

app = FastAPI(
    title="Maintenance prédictive — Condition de la valve",
    description="Prédit si la condition de la valve d'un cycle est optimale (100 %) ou non.",
    version=__version__,
)


class PredictionResponse(BaseModel):
    """Réponse de prédiction pour un cycle."""
    cycle: int = Field(..., description="Numéro de cycle (1-based)")
    prediction: int = Field(..., description="1 = optimale, 0 = non optimale")
    label: str = Field(..., description="Libellé lisible de la prédiction")
    probability_optimal: float | None = Field(
        None, description="Probabilité estimée que la valve soit optimale"
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/", tags=["info"])
def root() -> dict:
    """Point d'entrée : informations et lien vers la documentation."""
    return {
        "service": "Prédiction de la condition de la valve",
        "version": __version__,
        "endpoints": {
            "prédiction": "/predict/{cycle_number}",
            "santé": "/health",
            "infos modèle": "/model/info",
            "documentation": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["info"])
def health() -> HealthResponse:
    """Vérifie que le service tourne et que le modèle est disponible."""
    model_ok = config.MODEL_PATH.exists()
    return HealthResponse(
        status="ok" if model_ok else "model_missing",
        model_loaded=model_ok,
    )


@app.get("/predict/{cycle_number}", response_model=PredictionResponse, tags=["prédiction"])
def predict_cycle(cycle_number: int) -> PredictionResponse:
    """Prédit la condition de la valve pour un numéro de cycle (1-based)."""
    try:
        result = predict.predict_cycle(cycle_number)
    except ValueError as exc:  # cycle hors limites
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:  # modèle non entraîné
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(**result)


@app.get("/model/info", tags=["info"])
def model_info() -> dict:
    """Métadonnées du modèle entraîné (type, features, scores de validation)."""
    try:
        bundle = predict.load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    meta = dict(bundle["metadata"])
    meta["n_cycles_total"] = int(len(data.get_target()))
    return meta
