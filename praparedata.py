import pandas as pd
import re
import os
from collections import Counter

 
# Configuration

DATA_DIR = "data"
TRUE_CSV  = os.path.join(DATA_DIR, "True.csv")
FAKE_CSV  = os.path.join(DATA_DIR, "Fake.csv")
OUTPUT    = os.path.join(DATA_DIR, "dataset_clean.csv")
MIN_WORDS = 20  # Ignorer les textes trop courts


 
# Nettoyage du texte
 
def clean_text(text: str) -> str:
    """
    Nettoie un texte pour le NLP :
    - Mise en minuscules
    - Suppression des URLs, mentions, chiffres isolés
    - Suppression de la ponctuation
    - Normalisation des espaces
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)   # URLs
    text = re.sub(r"@\w+|#\w+", "", text)                  # Mentions / hashtags
    text = re.sub(r"\b\d+\b", "", text)                    # Chiffres isolés
    text = re.sub(r"[^a-z\s]", " ", text)                  # Caractères spéciaux
    text = re.sub(r"\s+", " ", text).strip()               # Espaces multiples

    return text

# Chargement
 
def load_data() -> pd.DataFrame:
    print("📂 Chargement des données...")

    if not os.path.exists(TRUE_CSV) or not os.path.exists(FAKE_CSV):
        raise FileNotFoundError(
            "\n❌ Fichiers introuvables !\n"
            "   → Télécharge le dataset ISOT sur Kaggle :\n"
            "   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
            f"   → Place True.csv et Fake.csv dans le dossier '{DATA_DIR}/'"
        )

    df_true = pd.read_csv(TRUE_CSV)
    df_fake = pd.read_csv(FAKE_CSV)

    df_true["label"] = 1  # REAL
    df_fake["label"] = 0  # FAKE

    df = pd.concat([df_true, df_fake], ignore_index=True)

    print(f"   ✅ Articles réels  : {len(df_true):,}")
    print(f"   ✅ Articles fake   : {len(df_fake):,}")
    print(f"   ✅ Total chargé    : {len(df):,}")

    return df


 
# Nettoyage
 
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🧹 Nettoyage en cours...")

    # Combiner titre + texte
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["content_clean"] = df["content"].apply(clean_text)

    # Filtrer textes trop courts
    df["word_count"] = df["content_clean"].apply(lambda x: len(x.split()))
    before = len(df)
    df = df[df["word_count"] >= MIN_WORDS].copy()
    print(f"   Textes trop courts supprimés : {before - len(df)}")

    # Supprimer les doublons
    before = len(df)
    df = df.drop_duplicates(subset="content_clean").reset_index(drop=True)
    print(f"   Doublons supprimés           : {before - len(df)}")

    df["label_str"] = df["label"].map({0: "FAKE", 1: "REAL"})

    print(f"\n   ✅ Dataset final : {len(df):,} articles")
    print(f"      → FAKE : {(df['label'] == 0).sum():,}")
    print(f"      → REAL : {(df['label'] == 1).sum():,}")

    return df


 
# Statistiques rapides
 
def print_stats(df: pd.DataFrame):
    print("\n📊 Statistiques de longueur (en mots) :")
    stats = df.groupby("label_str")["word_count"].describe()[["mean", "min", "max", "50%"]]
    stats.columns = ["Moyenne", "Min", "Max", "Médiane"]
    print(stats.round(1).to_string())


 
# Sauvegarde
 
def save_clean(df: pd.DataFrame):
    os.makedirs(DATA_DIR, exist_ok=True)
    df_out = df[["content_clean", "label", "label_str"]]
    df_out.to_csv(OUTPUT, index=False)
    print(f"\n💾 Dataset sauvegardé → {OUTPUT}")


 
# Main
 
if __name__ == "__main__":
    print("=" * 50)
    print("  🕵️  Fake News Detector — Préparation des données")
    print("=" * 50)

    df = load_data()
    df = preprocess(df)
    print_stats(df)
    save_clean(df)