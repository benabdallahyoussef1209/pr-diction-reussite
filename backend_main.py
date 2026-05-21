"""
API FastAPI — Prédiction de réussite des étudiants
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── Chargement des modèles ──────────────────────────────────────────────────
BASE = Path(__file__).parent
model_reussite = joblib.load(BASE / "models/model_reussite.pkl")
model_note     = joblib.load(BASE / "models/model_note.pkl")
model_abandon  = joblib.load(BASE / "models/model_abandon.pkl")
model_niveau   = joblib.load(BASE / "models/model_niveau.pkl")
features       = joblib.load(BASE / "models/features.pkl")

app = FastAPI(title="Student Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schéma de la requête ────────────────────────────────────────────────────
class StudentData(BaseModel):
    heures_etude:       float = Field(..., ge=0, le=12,  description="Heures d'étude par jour")
    absences:           float = Field(..., ge=0, le=30,  description="Nombre d'absences")
    notes_precedentes:  float = Field(..., ge=0, le=20,  description="Notes des années précédentes")
    heures_sommeil:     float = Field(..., ge=0, le=12,  description="Heures de sommeil par nuit")
    participation:      float = Field(..., ge=0, le=10,  description="Score de participation (0-10)")
    reseaux_sociaux:    float = Field(..., ge=0, le=10,  description="Heures sur les réseaux sociaux/jour")
    motivation:         float = Field(..., ge=1, le=10,  description="Niveau de motivation (1-10)")
    soutien_famille:    float = Field(..., ge=0, le=10,  description="Soutien familial (0-10)")
    activite_extra:     float = Field(..., ge=0, le=10,  description="Activités extra-scolaires (h/sem)")
    stress:             float = Field(..., ge=1, le=10,  description="Niveau de stress (1-10)")

# ── Endpoint principal ──────────────────────────────────────────────────────
@app.post("/predict")
def predict(data: StudentData):
    X = pd.DataFrame([data.dict()])[features]

    # Prédictions
    reussite_proba  = model_reussite.predict_proba(X)[0]
    reussite_pred   = int(model_reussite.predict(X)[0])
    note_pred       = float(np.clip(model_note.predict(X)[0], 0, 20))
    abandon_proba   = model_abandon.predict_proba(X)[0]
    abandon_pred    = int(model_abandon.predict(X)[0])
    niveau_pred     = str(model_niveau.predict(X)[0])

    # Importances (via step 'c' ou 'r' du Pipeline)
    step = model_reussite.named_steps.get("c") or model_reussite.named_steps.get("clf")
    importances = {}
    if hasattr(step, "feature_importances_"):
        imp = step.feature_importances_
        importances = dict(zip(features, [round(float(v)*100, 1) for v in imp]))

    return {
        "reussite": {
            "prediction":    reussite_pred,
            "label":         "Réussite" if reussite_pred == 1 else "Échec",
            "probabilite":   round(reussite_proba[reussite_pred] * 100, 1),
            "proba_reussite": round(reussite_proba[1] * 100, 1),
            "proba_echec":    round(reussite_proba[0] * 100, 1),
        },
        "note_finale": {
            "prediction": round(note_pred, 2),
            "sur_20":     f"{round(note_pred, 1)}/20",
        },
        "risque_abandon": {
            "prediction":  abandon_pred,
            "label":       "Risque élevé" if abandon_pred == 1 else "Risque faible",
            "probabilite": round(abandon_proba[abandon_pred] * 100, 1),
            "proba_risque": round(abandon_proba[1] * 100, 1),
        },
        "niveau_performance": {
            "prediction": niveau_pred,
        },
        "feature_importances": importances,
    }

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}

@app.get("/stats")
def stats():
    """Statistiques du dataset d'entraînement"""
    df = pd.read_csv(BASE / "dataset.csv")
    return {
        "total_etudiants":      len(df),
        "taux_reussite":        round(df["reussite"].mean() * 100, 1),
        "taux_abandon":         round(df["risque_abandon"].mean() * 100, 1),
        "note_moyenne":         round(df["note_finale"].mean(), 2),
        "note_mediane":         round(df["note_finale"].median(), 2),
        "distribution_niveau":  df["niveau"].value_counts().to_dict(),
        "features":             features,
    }
