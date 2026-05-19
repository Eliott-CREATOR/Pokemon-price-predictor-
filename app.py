import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys

sys.path.append("src")
from preprocessing import preparer_donnees, FEATURES

st.title("Pokemon Price Predictor")


@st.cache_resource
def charger_modele():
    return joblib.load("models/xgboost.pkl")


@st.cache_data
def charger_catalogue():
    df = preparer_donnees()
    cols = ["id", "nom", "rarity", "set_name", "prix_market"] + FEATURES
    return df[[c for c in cols if c in df.columns]]


modele = charger_modele()
catalogue = charger_catalogue()

recherche = st.text_input("Nom ou ID de la carte", placeholder="ex: Charizard, base1-4")

if recherche:
    masque = (
        catalogue["nom"].str.contains(recherche, case=False, na=False)
        | catalogue["id"].str.contains(recherche, case=False, na=False)
    )
    resultats = catalogue[masque]

    if resultats.empty:
        st.error("Carte non trouvée")
    else:
        if len(resultats) > 1:
            options = (
                resultats["nom"] + " — "
                + resultats["set_name"].fillna("?") + " — "
                + resultats["rarity"].fillna("?") + " ("
                + resultats["id"] + ")"
            )
            choix = st.selectbox(f"{len(resultats)} cartes trouvées", options)
            carte = resultats[options == choix].iloc[0]
        else:
            carte = resultats.iloc[0]

        X = pd.DataFrame([carte[FEATURES]])
        prix_pred = np.expm1(modele.predict(X)[0])

        st.subheader(f"{carte['nom']}  —  {carte['id']}")
        col1, col2 = st.columns(2)
        col1.metric("Prix prédit", f"${prix_pred:.2f}")
        if pd.notna(carte.get("prix_market")):
            col2.metric("Prix réel (TCGPlayer)", f"${carte['prix_market']:.2f}")
