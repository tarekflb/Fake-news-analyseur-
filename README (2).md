# Fake News Detector

Un outil de détection de fake news basé sur le Machine Learning (NLP).  
Application web interactive construite avec Streamlit.

---

## Demo

> Lien Streamlit Cloud : *(à ajouter après déploiement)*

---

## Aperçu

L'application analyse un texte en anglais et prédit s'il s'agit d'une
fake news ou d'un article réel, avec un score de confiance.

---

## Dataset

**ISOT Fake News Dataset** — Université de Victoria  
- ~44 000 articles en anglais (2015–2018)  
- Articles réels : Reuters  
- Articles fake : sites fact-checkés comme faux  
- Source : https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

---

## Pipeline ML

```
Texte brut
    ↓
Nettoyage (lowercase, suppression URLs, ponctuation)
    ↓
Vectorisation TF-IDF (50 000 features, unigrammes + bigrammes)
    ↓
Classification (Passive Aggressive Classifier)
    ↓
Prédiction + score de confiance
```

---

## Résultats

| Modèle                  | Accuracy | F1-Score | Precision | Recall |
|-------------------------|----------|----------|-----------|--------|
| Logistic Regression     | 98.59%   | 98.72%   | 98.29%    | 99.14% |
| Random Forest           | 99.23%   | 99.30%   | 99.12%    | 99.47% |
| **Passive Aggressive**  | **99.56%** | **99.59%** | **99.45%** | **99.74%** |

Meilleur modèle : **Passive Aggressive Classifier** (F1 = 99.59%)

---

## Installation

```bash
# 1. Cloner le repo
git clone https://github.com/TON_USERNAME/fake-news-detector.git
cd fake-news-detector

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application
streamlit run app.py
```

---

## Structure du projet

```
fake-news-detector/
├── app.py                  # Application Streamlit
├── prepare_data.py         # Nettoyage des données (Jour 1)
├── train_model.py          # Entraînement des modèles (Jour 3)
├── model/
│   ├── model.pkl           # Modèle sauvegardé
│   └── vectorizer.pkl      # Vectorizer TF-IDF sauvegardé
├── data/
│   └── dataset_clean.csv   # Dataset nettoyé
├── requirements.txt
└── README.md
```

---

## Limites du projet

- Le modèle a été entraîné sur des articles en **anglais uniquement**
- Le dataset date de **2015–2018** — les fake news récentes peuvent ne pas être détectées
- Le modèle a potentiellement appris le **style Reuters** plutôt que la vraie notion de fake news (data leakage partiel)
- Un texte court (moins de 10 mots) ne peut pas être analysé correctement

---

## Technologies utilisées

- Python 3.10+
- scikit-learn
- pandas / numpy
- Streamlit
- joblib
- matplotlib / seaborn

---

## Auteur

**Ton Nom**  
Étudiant en Master Informatique — IA  
[LinkedIn](https://linkedin.com/in/TON_PROFIL) · [GitHub](https://github.com/TON_USERNAME)
