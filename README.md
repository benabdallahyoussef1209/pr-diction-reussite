# 🎓 Student Predictor — Prédiction de réussite académique

Projet complet de Machine Learning qui prédit la réussite d'un étudiant à partir de ses habitudes et performances.

## Architecture

```
frontend/
  index.html          ← App React (SPA autonome)
backend/
  train_model.py      ← Génération dataset + entraînement
  main.py             ← API FastAPI
  requirements.txt
  dataset.csv         ← Dataset synthétique 3000 étudiants
  models/
    model_reussite.pkl    ← GradientBoosting (binaire)
    model_note.pkl        ← GradientBoosting (régression)
    model_abandon.pkl     ← RandomForest (binaire)
    model_niveau.pkl      ← GradientBoosting (multi-classes)
    features.pkl
```

## Prédictions fournies

| Cible | Type | Modèle | Performance |
|-------|------|--------|-------------|
| Réussite / Échec | Classification binaire | GradientBoosting | ~85% accuracy |
| Note finale /20 | Régression | GradientBoosting | MAE ~2.1 pts |
| Risque d'abandon | Classification binaire | RandomForest | ~83% accuracy |
| Niveau de performance | Multi-classes (5 niveaux) | GradientBoosting | ~54% accuracy |

## Features (variables d'entrée)

- `heures_etude` — Heures d'étude par jour (0–12)
- `absences` — Nombre d'absences (0–30)
- `notes_precedentes` — Notes des années précédentes /20
- `heures_sommeil` — Heures de sommeil par nuit (3–12)
- `participation` — Score de participation en classe /10
- `reseaux_sociaux` — Heures sur les réseaux sociaux/jour
- `motivation` — Niveau de motivation /10
- `soutien_famille` — Soutien familial /10
- `activite_extra` — Activités extra-scolaires en h/semaine
- `stress` — Niveau de stress /10

## Installation & lancement

### Backend

```bash
cd backend
pip install -r requirements.txt

# Entraîner les modèles (génère dataset.csv + models/)
python train_model.py

# Lancer l'API
uvicorn main:app --reload --port 8000
```

L'API sera disponible sur http://localhost:8000  
Documentation Swagger : http://localhost:8000/docs

### Frontend

```bash
# Ouvrir simplement le fichier dans un navigateur
open frontend/index.html
# ou
python -m http.server 3000 --directory frontend
```

## Endpoints API

### `POST /predict`
```json
{
  "heures_etude": 7,
  "absences": 2,
  "notes_precedentes": 14,
  "heures_sommeil": 8,
  "participation": 8,
  "reseaux_sociaux": 1,
  "motivation": 9,
  "soutien_famille": 8,
  "activite_extra": 3,
  "stress": 3
}
```

**Réponse :**
```json
{
  "reussite": {
    "prediction": 1,
    "label": "Réussite",
    "probabilite": 99.8,
    "proba_reussite": 99.8,
    "proba_echec": 0.2
  },
  "note_finale": { "prediction": 19.5, "sur_20": "19.5/20" },
  "risque_abandon": { "prediction": 0, "label": "Risque faible", "proba_risque": 3.2 },
  "niveau_performance": { "prediction": "Très Bien" },
  "feature_importances": { "heures_etude": 18.4, ... }
}
```

### `GET /stats` — Statistiques du dataset
### `GET /health` — Santé de l'API

## Améliorations possibles

- [ ] Remplacer le dataset synthétique par un vrai dataset (Kaggle Student Performance)
- [ ] Ajouter SHAP pour l'explicabilité des prédictions
- [ ] Déploiement sur Railway / Render / Hugging Face Spaces
- [ ] Base de données PostgreSQL pour persister les profils
- [ ] Authentification JWT pour les enseignants
- [ ] Dashboard comparatif multi-étudiants

## Tech stack

| Couche | Technologies |
|--------|-------------|
| ML | scikit-learn, pandas, numpy, joblib |
| Backend | FastAPI, Pydantic, uvicorn |
| Frontend | React 18, Vanilla CSS |
