# Image de service (API FastAPI + interface Streamlit, même image).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dépendances Python (couche mise en cache tant que requirements ne change pas).
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Code de l'application.
COPY hydraulic_valve/ ./hydraulic_valve/
COPY api/ ./api/
COPY streamlit_app.py ./

# Le modèle (models/) et les données (data/) sont fournis via volumes
# (voir docker-compose.yml) ou par `dvc pull` — ils ne sont pas copiés dans
# l'image afin de la garder légère et reconstructible sans les artefacts.

EXPOSE 8000 8501

# Par défaut : l'API. Surchargé par docker-compose pour le service Streamlit.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
