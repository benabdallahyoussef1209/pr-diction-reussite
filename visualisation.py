import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataset
df = pd.read_csv("dataset_2000_students.csv")

print("Dataset chargé :", df.shape)
print(df.head())

# ─────────────────────────────
# 1. Distribution des notes
# ─────────────────────────────
plt.figure()
sns.histplot(df["note_finale"], bins=30, kde=True)
plt.title("Distribution des notes finales")
plt.xlabel("Note /20")
plt.ylabel("Nombre d'étudiants")
plt.savefig("note_distribution.png")
plt.show()
plt.close()

# ─────────────────────────────
# 2. Réussite vs Échec
# ─────────────────────────────
plt.figure()
sns.countplot(x="reussite", data=df)
plt.title("Réussite vs Échec")
plt.xticks([0,1], ["Échec", "Réussite"])
plt.savefig("2.png")
plt.show()
plt.close()

# ─────────────────────────────
# 3. Abandon
# ─────────────────────────────
plt.figure()
sns.countplot(x="risque_abandon", data=df)
plt.title("Risque d'abandon")
plt.xticks([0,1], ["Faible", "Élevé"])
plt.savefig("3.png")
plt.show()
plt.close()

# ─────────────────────────────
# 4. Étude vs note
# ─────────────────────────────
plt.figure()
sns.scatterplot(x="heures_etude", y="note_finale", data=df)
plt.title("Heures d'étude vs Note finale")
plt.savefig("4.png")
plt.show()
plt.close()

# ─────────────────────────────
# 5. Stress vs note
# ─────────────────────────────
plt.figure()
sns.boxplot(x="stress", y="note_finale", data=df)
plt.title("Stress vs Note finale")
plt.savefig("5.png")
plt.show()
plt.close()

# ─────────────────────────────
# 6. Corrélation
# ─────────────────────────────
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Matrice de corrélation")
plt.savefig("6.png")
plt.show()
plt.close()

# ─────────────────────────────
# 7. Réseaux sociaux vs note
# ─────────────────────────────
plt.figure()
sns.scatterplot(x="reseaux_sociaux", y="note_finale", data=df)
plt.title("Réseaux sociaux vs Note")
plt.savefig("7.png")
plt.show()
plt.close()

# ─────────────────────────────
# 8. Distribution niveaux
# ─────────────────────────────
plt.figure()
sns.countplot(x="niveau", data=df,
              order=df["niveau"].value_counts().index)
plt.title("Distribution des niveaux")
plt.xticks(rotation=45)
plt.savefig("8.png")
plt.show()
plt.close()