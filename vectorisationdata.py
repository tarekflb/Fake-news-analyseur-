import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Charger le dataset du Jour 1
df = pd.read_csv("data/dataset_clean.csv")

print(f"Dataset chargé : {len(df):,} articles")
print(df["label_str"].value_counts())


X = df["content_clean"]
y = df["label"]

print(f"X shape : {X.shape}")  # nombre d'articles
print(f"y shape : {y.shape}")  # même nombre


X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y,

)
print(f"Train : {len(X_train):,} articles")
print(f"Test  : {len(X_test):,} articles")

# Créer le vectoriseur
vectorizer = TfidfVectorizer(
    max_features=50000,  # garder les 50 000 mots les plus importants
    ngram_range=(1, 2),  # unigrammes ET bigrammes (expliqué ci-dessous)
    min_df=2,            # ignorer les mots qui apparaissent dans moins de 2 articles
    sublinear_tf=True    # atténuer l'effet des mots très fréquents
)

# IMPORTANT : fit UNIQUEMENT sur le train, jamais sur le test !
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print(f"Shape Train : {X_train_tfidf.shape}")
# → (nb_articles, nb_mots) ex: (35000, 50000)
print(f"Shape Test  : {X_test_tfidf.shape}")
