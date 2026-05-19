import streamlit as st
import pandas as pd
import numpy as np
import joblib

RARITY_MAP = {
    "Common": 1, "Uncommon": 2, "Rare": 3,
    "Rare Holo": 4, "Rare Holo EX": 5, "Rare Holo GX": 5, "Rare Holo V": 5,
    "Rare Ultra": 6, "Rare Holo VMAX": 6,
    "Rare Secret": 7, "Amazing Rare": 7,
    "Rare Rainbow": 8, "Rare Shining": 8, "Rare Shiny GX": 8, "LEGEND": 8,
}

FEATURES = [
    "hp", "retreat_cost", "has_evolution", "generation", "pokedex_number",
    "carte_age_ans", "set_total_cartes", "position_dans_set",
    "set_serie_encoded", "supertype_encoded",
    "rarity_encoded", "est_secret_rare", "est_holo_premium",
    "is_holo", "is_full_art", "is_v_card", "is_ex_gx",
    "is_basic", "is_stage1", "is_stage2", "est_full_art_v",
    "type_fire", "type_water", "type_grass", "type_lightning",
    "type_psychic", "type_fighting", "type_darkness", "type_metal",
    "type_dragon", "type_fairy", "type_colorless", "nb_types",
    "nb_attaques", "max_damage", "total_damage_attaques",
    "a_attaque_100plus", "a_attaque_200plus",
    "cout_energie_moyen", "nb_abilities", "has_ability", "efficacite_combat",
    "standard_encoded", "expanded_encoded", "a_faiblesse", "a_resistance",
]


@st.cache_resource
def charger_modele():
    return joblib.load("models/xgboost.pkl"), joblib.load("models/imputer.pkl")


@st.cache_data
def charger_catalogue():
    df = pd.read_csv("data/raw/cartes_pokemon.csv")
    df = df[df["prix_market"] > 0].copy()

    annee = pd.to_numeric(df["set_annee"], errors="coerce").fillna(2000)
    df["carte_age_ans"] = 2026 - annee
    numero_num = pd.to_numeric(df["numero"], errors="coerce")
    df["est_secret_rare"] = (numero_num > df["set_total_cartes"]).fillna(False).astype(int)
    df["efficacite_combat"] = df["max_damage"].fillna(0) / (df["cout_energie_moyen"].fillna(1) + 1)
    df["est_full_art_v"] = df["is_full_art"] * df["is_v_card"]
    cols_types = [c for c in df.columns if c.startswith("type_") and df[c].dtype != object]
    df["nb_types"] = df[cols_types].sum(axis=1)

    df["rarity_encoded"] = df["rarity"].map(RARITY_MAP).fillna(3)
    df["est_holo_premium"] = df["is_holo"] * df["rarity_encoded"]
    df["supertype_encoded"] = pd.Categorical(df["supertype"]).codes
    df["set_serie_encoded"] = pd.Categorical(df["set_serie"]).codes
    df["standard_encoded"] = (df["legal_standard"] == "Legal").astype(int)
    df["expanded_encoded"] = (df["legal_expanded"] == "Legal").astype(int)
    df["a_faiblesse"] = df["faiblesse"].notna().astype(int)
    df["a_resistance"] = df["resistance"].notna().astype(int)

    medians = {c: df[c].median() for c in ["hp", "position_dans_set", "set_total_cartes"]}
    df = df.fillna({**medians, "generation": -1, "pokedex_number": -1,
                    "max_damage": 0, "total_damage_attaques": 0,
                    "cout_energie_moyen": 0, "efficacite_combat": 0})

    cols = ["id", "nom", "rarity", "set_name", "prix_market"] + FEATURES
    return df[[c for c in cols if c in df.columns]]


modele, imputer = charger_modele()
catalogue = charger_catalogue()

st.title("Pokemon Price Predictor")

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
        X_imputed = imputer.transform(X)
        prix_pred = np.expm1(modele.predict(X_imputed)[0])

        st.subheader(f"{carte['nom']}  —  {carte['id']}")
        col1, col2 = st.columns(2)
        col1.metric("Prix prédit", f"${prix_pred:.2f}")
        if pd.notna(carte.get("prix_market")):
            col2.metric("Prix réel (TCGPlayer)", f"${carte['prix_market']:.2f}")
