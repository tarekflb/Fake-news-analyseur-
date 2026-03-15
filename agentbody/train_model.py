import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les donnee

print("=" * 55)
print("  Fake News Detector - Entrainement des modeles")
print("=" * 55)

df = pd.read_csv("data/dataset_clean.csv")
X = df["content_clean"]
y = df["label"]

print(f"\nDataset charge : {len(df):,} articles")

# Split train / tes

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train : {len(X_train):,} articles")
print(f"Test  : {len(X_test):,} articles")

# Vectorisation TF-ID

print("\nVectorisation TF-IDF en cours...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print(f"Shape Train : {X_train_tfidf.shape}")
print(f"Shape Test  : {X_test_tfidf.shape}")

# Definition des 3 modele


# On met les 3 modeles dans un dictionnaire
# cle   = nom du modele
# valeur = l'objet modele avec ses parametres
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,      # nb max d'iterations pour converger
        C=1.0,              # force de regularisation (evite l'overfitting)
        solver="lbfgs",     # algorithme d'optimisation
        n_jobs=-1           # utiliser tous les coeurs du CPU
    ),
    "Passive Aggressive": PassiveAggressiveClassifier(
        max_iter=1000,
        C=0.1,              # plus petit = plus "passif" = moins agressif
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,   # nombre d'arbres dans la foret
        max_depth=None,     # les arbres peuvent grandir librement
        random_state=42,
        n_jobs=-1
    )
}

# Entrainement et evaluation de chaque model

print("\nEntrainement des modeles...\n")

results = {}  

for name, model in models.items():
    print(f"--- {name} ---")

    # ENTRAINEMENT
    # On donne au modele les textes vectorises + les labels corrects
    # Il ajuste ses parametres internes pour minimiser les erreurs
    model.fit(X_train_tfidf, y_train)
    print(f"  Entrainement termine")

    # PREDICTION
    
    y_pred = model.predict(X_test_tfidf)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print()

    # On sauvegarde les resultats pour comparer ensuite
    results[name] = {
        "model"    : model,
        "accuracy" : acc,
        "f1"       : f1,
        "precision": prec,
        "recall"   : rec,
        "y_pred"   : y_pred
    }
    
# Comparer les modele

print("\n" + "=" * 55)
print("  COMPARAISON DES MODELES")
print("=" * 55)
print(f"{'Modele':<25} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
print("-" * 55)

best_name = None
best_f1   = 0

for name, res in results.items():
    print(
        f"{name:<25}"
        f"{res['accuracy']:>10.4f}"
        f"{res['f1']:>10.4f}"
        f"{res['precision']:>10.4f}"
        f"{res['recall']:>10.4f}"
    )
    # On cherche le meilleur modele selon le F1-Score
    if res["f1"] > best_f1:
        best_f1   = res["f1"]
        best_name = name

print("-" * 55)
print(f"\nMeilleur modele : {best_name} (F1 = {best_f1:.4f})")

# Rapport detaille du meilleur model

print(f"\nRapport detaille - {best_name} :")
print(classification_report(
    y_test,
    results[best_name]["y_pred"],
    target_names=["FAKE", "REAL"]
))

#  Matrice de confusion du meilleur model

os.makedirs("data/plots", exist_ok=True)

cm = confusion_matrix(y_test, results[best_name]["y_pred"])

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["FAKE", "REAL"],
    yticklabels=["FAKE", "REAL"]
)
plt.title(f"Matrice de confusion - {best_name}", fontsize=14, fontweight="bold")
plt.ylabel("Vraie classe")
plt.xlabel("Classe predite")
plt.tight_layout()
plt.savefig("data/plots/confusion_matrix.png", dpi=150)
plt.show()
print("Matrice de confusion sauvegardee dans data/plots/")

# Sauvegarder le meilleur model

os.makedirs("model", exist_ok=True)

best_model = results[best_name]["model"]

# On sauvegarde les deux fichiers indispensables :
# - le modele : pour faire les predictions
# - le vectorizer : pour transformer le texte brut en vecteurs
joblib.dump(best_model,  "model/model.pkl")
joblib.dump(vectorizer,  "model/vectorizer.pkl")

print(f"\nModele sauvegarde    : model/model.pkl")
print(f"Vectorizer sauvegarde : model/vectorizer.pkl")
print("\nJour 3 + 4 termine ! Prochain : app.py (Streamlit)")
