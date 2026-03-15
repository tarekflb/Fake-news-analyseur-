import streamlit as st
import joblib
import re

# -------------------------------------------------
# CONFIGURATION DE LA PAGE
# Doit etre la premiere instruction Streamlit
# -------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="detective",
    layout="centered"
)

# -------------------------------------------------
# FONCTION DE NETTOYAGE
# La meme qu'au Jour 1 - on reutilise exactement
# le meme traitement pour etre coherent avec
# ce que le modele a appris
# -------------------------------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\b\d+\b", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------------------------------
# CHARGEMENT DU MODELE ET DU VECTORIZER
# @st.cache_resource : charge une seule fois
# et garde en memoire pour tous les clics suivants
# -------------------------------------------------
@st.cache_resource
def load_model():
    model      = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

# FONCTION DE PREDICTION
# Applique le pipeline complet :
# texte brut → nettoyage → vecteur → prediction
def predict(text: str, model, vectorizer):
    # Etape 1 : nettoyage
    text_clean = clean_text(text)

    # Etape 2 : vectorisation
    text_vect = vectorizer.transform([text_clean])

    # Etape 3 : prediction du label
    label = model.predict(text_vect)[0]

    # Etape 4 : score de confiance
    # On vérifie si le modèle supporte predict_proba
    if hasattr(model, "predict_proba"):
        probas     = model.predict_proba(text_vect)[0]
        confidence = probas[label]
    else:
        # Passive Aggressive n'a pas predict_proba
        # On utilise decision_function comme score brut
        score      = model.decision_function(text_vect)[0]
        # On convertit en pseudo-probabilité entre 0 et 1
        import math
        proba_real = 1 / (1 + math.exp(-score))  # fonction sigmoïde
        probas     = [1 - proba_real, proba_real]
        confidence = probas[label]

    return label, confidence, probas

# INTERFACE STREAMLIT
# -------------------------------------------------

# Titre
st.title("Fake News Detector")
st.write("Colle un article ou un texte ci-dessous pour analyser s'il s'agit d'une fake news.")

st.divider()

# Zone de saisie du texte
user_input = st.text_area(
    label="Texte a analyser",
    placeholder="Colle ici le contenu de l'article...",
    height=250
)

# Bouton d'analyse
analyze_btn = st.button("Analyser", type="primary", use_container_width=True)

# -------------------------------------------------
# LOGIQUE DE PREDICTION
# Se declenche uniquement quand on clique
# -------------------------------------------------
if analyze_btn:

    # Verifier que l'utilisateur a bien saisi quelque chose
    if not user_input.strip():
        st.warning("Merci de saisir un texte avant d'analyser.")

    # Verifier que le texte est assez long pour etre analyse
    elif len(user_input.split()) < 10:
        st.warning("Le texte est trop court. Merci de saisir au moins 10 mots.")

    else:
        # Afficher une animation pendant le chargement
        with st.spinner("Analyse en cours..."):
            model, vectorizer = load_model()
            label, confidence, probas = predict(user_input, model, vectorizer)

        st.divider()
        st.subheader("Resultat de l'analyse")

        # Afficher le resultat selon le label predit
        if label == 0:
            st.error(f"FAKE NEWS detectee")
        else:
            st.success(f"Article REEL")

        # Afficher le score de confiance
        st.metric(
            label="Niveau de confiance",
            value=f"{confidence * 100:.1f}%"
        )

        # Afficher les deux probabilites cote a cote
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Proba FAKE",
                value=f"{probas[0] * 100:.1f}%"
            )
        with col2:
            st.metric(
                label="Proba REAL",
                value=f"{probas[1] * 100:.1f}%"
            )

        # Barre de confiance visuelle
        st.write("Repartition FAKE / REAL :")
        st.progress(float(probas[1]))  # barre qui se remplit vers REAL

        st.divider()

        # Section transparence
        with st.expander("Voir le texte nettoye envoye au modele"):
            st.code(clean_text(user_input))

        # Avertissement sur les limites
        st.caption(
            "Ce modele a ete entraine sur le dataset ISOT (articles en anglais, 2015-2018). "
            "Les resultats peuvent ne pas generaliser sur des articles recents ou dans d'autres langues."
        )