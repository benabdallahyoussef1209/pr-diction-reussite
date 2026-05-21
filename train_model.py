"""
Génération du dataset et entraînement du modèle ML
Prédit : réussite/échec, note finale, risque d'abandon, niveau de performance
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
from sklearn.pipeline import Pipeline
import joblib
import os

np.random.seed(42)
N = 2000

def generate_dataset(n=N):
    heures_etude       = np.random.normal(5, 2, n).clip(0, 12)
    absences           = np.random.poisson(3, n).clip(0, 30)
    notes_precedentes  = np.random.normal(12, 3, n).clip(0, 20)
    heures_sommeil     = np.random.normal(7, 1.5, n).clip(3, 12)
    participation      = np.random.randint(0, 11, n).astype(float)   # /10
    reseaux_sociaux    = np.random.normal(3, 1.5, n).clip(0, 10)     # h/jour
    motivation         = np.random.randint(1, 11, n).astype(float)   # /10
    soutien_famille    = np.random.randint(0, 11, n).astype(float)
    activite_extra     = np.random.randint(0, 6, n).astype(float)    # h/sem
    stress             = np.random.randint(1, 11, n).astype(float)

    # Score composite → note finale
    score = (
        heures_etude       * 1.8
        + notes_precedentes * 0.5
        + participation     * 0.4
        + heures_sommeil    * 0.3
        + motivation        * 0.3
        + soutien_famille   * 0.2
        + activite_extra    * 0.1
        - absences          * 0.5
        - reseaux_sociaux   * 0.4
        - stress            * 0.15
        + np.random.normal(0, 1.5, n)
    )
    note_finale = np.clip(score * 0.9 + 5, 0, 20)

    reussite = (note_finale >= 10).astype(int)

    prob_abandon = (
        0.5
        - heures_etude    * 0.05
        + absences        * 0.04
        - notes_precedentes * 0.02
        - motivation      * 0.04
        + reseaux_sociaux * 0.03
        + stress          * 0.03
        + np.random.normal(0, 0.05, n)
    ).clip(0, 1)
    risque_abandon = (prob_abandon > 0.5).astype(int)

    niveau = pd.cut(
        note_finale,
        bins=[0, 8, 10, 12, 14, 20],
        labels=["Insuffisant", "Passable", "Assez Bien", "Bien", "Très Bien"]
    ).astype(str)

    df = pd.DataFrame({
        "heures_etude":      heures_etude,
        "absences":          absences,
        "notes_precedentes": notes_precedentes,
        "heures_sommeil":    heures_sommeil,
        "participation":     participation,
        "reseaux_sociaux":   reseaux_sociaux,
        "motivation":        motivation,
        "soutien_famille":   soutien_famille,
        "activite_extra":    activite_extra,
        "stress":            stress,
        "note_finale":       note_finale,
        "reussite":          reussite,
        "risque_abandon":    risque_abandon,
        "niveau":            niveau,
    })
    return df

print("Génération du dataset...")
df = generate_dataset()
df.to_csv("dataset.csv", index=False)
print(f"Dataset : {df.shape[0]} lignes, {df.shape[1]} colonnes")

features = [
    "heures_etude", "absences", "notes_precedentes",
    "heures_sommeil", "participation", "reseaux_sociaux",
    "motivation", "soutien_famille", "activite_extra", "stress"
]
X = df[features]

# ─── 1. Réussite (Classification binaire) ───
y_reussite = df["reussite"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y_reussite, test_size=0.2)
model_reussite = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=4))
])
model_reussite.fit(X_tr, y_tr)
acc = accuracy_score(y_te, model_reussite.predict(X_te))
print(f"[Réussite] Accuracy : {acc:.3f}")

# ─── 2. Note finale (Régression) ───
y_note = df["note_finale"]
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y_note, test_size=0.2)
model_note = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=4))
])
model_note.fit(X_tr2, y_tr2)
mae = mean_absolute_error(y_te2, model_note.predict(X_te2))
print(f"[Note finale] MAE : {mae:.2f} pts")

# ─── 3. Risque d'abandon (Classification binaire) ───
y_abandon = df["risque_abandon"]
X_tr3, X_te3, y_tr3, y_te3 = train_test_split(X, y_abandon, test_size=0.2)
model_abandon = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, max_depth=6))
])
model_abandon.fit(X_tr3, y_tr3)
acc3 = accuracy_score(y_te3, model_abandon.predict(X_te3))
print(f"[Abandon] Accuracy : {acc3:.3f}")

# ─── 4. Niveau de performance (Multi-classes) ───
y_niveau = df["niveau"]
X_tr4, X_te4, y_tr4, y_te4 = train_test_split(X, y_niveau, test_size=0.2)
model_niveau = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=4))
])
model_niveau.fit(X_tr4, y_tr4)
acc4 = accuracy_score(y_te4, model_niveau.predict(X_te4))
print(f"[Niveau] Accuracy : {acc4:.3f}")

# ─── Sauvegarde ───
os.makedirs("models", exist_ok=True)
joblib.dump(model_reussite, "models/model_reussite.pkl")
joblib.dump(model_note,     "models/model_note.pkl")
joblib.dump(model_abandon,  "models/model_abandon.pkl")
joblib.dump(model_niveau,   "models/model_niveau.pkl")
joblib.dump(features,       "models/features.pkl")
print("\nModèles sauvegardés dans models/")

# Stats dataset
print("\n── Statistiques du dataset ──")
print(f"Taux de réussite    : {df['reussite'].mean()*100:.1f}%")
print(f"Taux d'abandon      : {df['risque_abandon'].mean()*100:.1f}%")
print(f"Note moyenne        : {df['note_finale'].mean():.2f}/20")
print(df["niveau"].value_counts())
