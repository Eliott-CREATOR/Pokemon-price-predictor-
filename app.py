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
    "age_du_set", "set_total_cartes", "position_dans_set",
    "set_serie_encode", "supertype_Pokémon", "supertype_Trainer", "supertype_Energy",
    "score_rarete", "est_secret_rare", "holo_x_rarete",
    "is_holo", "is_full_art", "is_v_card", "is_ex_gx",
    "is_basic", "is_stage1", "is_stage2", "full_art_et_v",
    "type_fire", "type_water", "type_grass", "type_lightning",
    "type_psychic", "type_fighting", "type_darkness", "type_metal",
    "type_dragon", "type_fairy", "type_colorless", "nb_types_total",
    "nb_attaques", "max_damage", "total_damage_attaques",
    "a_attaque_100plus", "a_attaque_200plus",
    "cout_energie_moyen", "nb_abilities", "has_ability", "ratio_degats_energie",
    "legal_en_standard", "legal_en_expanded", "a_faiblesse", "a_resistance",
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
    df["est_secret_rare"] = (pd.to_numeric(df["numero"], errors="coerce") > df["set_total_cartes"]).fillna(False).astype(int)
    df["efficacite_combat"] = df["max_damage"].fillna(0) / (df["cout_energie_moyen"].fillna(1) + 1)
    df["est_full_art_v"] = df["is_full_art"] * df["is_v_card"]
    df["nb_types"] = df[[c for c in df.columns if c.startswith("type_") and df[c].dtype != object]].sum(axis=1)
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


def predire(features_dict, modele, imputer):
    X_base = pd.read_csv("data/processed/X_train.csv").median().to_dict()
    X_base.update(features_dict)
    X = pd.DataFrame([X_base])[FEATURES]
    return np.expm1(modele.predict(imputer.transform(X))[0])


def expliquer(f, prix):
    bullets = []
    rarity_labels = {1: "Common", 2: "Uncommon", 3: "Rare", 4: "Rare Holo",
                     5: "Rare Holo V/EX/GX", 6: "Rare Ultra/VMAX", 7: "Rare Secret", 8: "Rainbow/Shining"}
    bullets.append(f"Rareté **{rarity_labels.get(f['score_rarete'], '?')}** — {'segment premium, moins de 5% des cartes' if f['score_rarete'] >= 7 else 'rareté standard du marché' if f['score_rarete'] <= 3 else 'rareté intermédiaire'}")
    if f["full_art_et_v"]:
        bullets.append("Full Art + Carte V — visuellement unique, très recherché par les collectionneurs")
    elif f["is_full_art"]:
        bullets.append("Full Art — illustration premium, valorise la carte chez les collectionneurs")
    if f["est_secret_rare"]:
        bullets.append("Secret Rare — hors numérotation du set, tirage limité, forte demande")
    if f["is_holo"] and f["score_rarete"] >= 5:
        bullets.append(f"Holo × Rareté {f['score_rarete']}/8 — combinaison qui amplifie la valeur perçue")
    if f["age_du_set"] <= 3:
        bullets.append(f"Set de {2026 - int(f['age_du_set'])} — carte récente, marché actif et demande soutenue")
    elif f["age_du_set"] >= 20:
        bullets.append(f"Set de {2026 - int(f['age_du_set'])} — carte vintage, valeur nostalgique forte")
    if f["has_ability"]:
        bullets.append("Possède une Ability — cartes avec capacités passives souvent plus jouées et plus chères")
    return bullets


modele, imputer = charger_modele()

st.title("Pokemon Card Price Predictor")

onglet1, onglet2 = st.tabs(["Nouvelle carte", "Carte existante"])

with onglet1:
    st.subheader("Estimez le prix d'une carte avant sa sortie")
    st.caption("Renseignez les caractéristiques de la carte — notre modèle prédit sa valeur marché.")

    col1, col2 = st.columns(2)

    with col1:
        rarete = st.selectbox("Rareté", list(RARITY_MAP.keys()), index=3)
        hp = st.slider("HP", 0, 400, 120, step=10)
        max_dmg = st.slider("Dégâts max", 0, 350, 120, step=10)
        annee = st.selectbox("Année de sortie", list(range(2026, 1995, -1)), index=0)
        generation = st.selectbox("Génération", list(range(1, 9)), index=0)

    with col2:
        is_holo = st.checkbox("Holo")
        is_full_art = st.checkbox("Full Art")
        is_v = st.checkbox("Carte V / VMAX")
        is_ex_gx = st.checkbox("EX / GX")
        has_ability = st.checkbox("A une Ability")
        est_secret = st.checkbox("Secret Rare (numéro hors set)")

    if st.button("Prédire le prix", type="primary"):
        rarity_encoded = RARITY_MAP[rarete]
        features = {
            "score_rarete":      rarity_encoded,
            "hp":                hp,
            "max_damage":        max_dmg,
            "total_damage_attaques": max_dmg,
            "a_attaque_100plus": 1 if max_dmg >= 100 else 0,
            "a_attaque_200plus": 1 if max_dmg >= 200 else 0,
            "age_du_set":        2026 - annee,
            "generation":        generation,
            "is_holo":           int(is_holo),
            "is_full_art":       int(is_full_art),
            "is_v_card":         int(is_v),
            "is_ex_gx":          int(is_ex_gx),
            "has_ability":       int(has_ability),
            "nb_abilities":      int(has_ability),
            "est_secret_rare":   int(est_secret),
            "holo_x_rarete":     int(is_holo) * rarity_encoded,
            "full_art_et_v":     int(is_full_art and is_v),
            "ratio_degats_energie": max_dmg / 3,
            "supertype_Pokémon": 1,
            "supertype_Trainer": 0,
            "supertype_Energy":  0,
        }
        prix = predire(features, modele, imputer)

        st.metric("Prix estimé", f"${prix:.2f}")
        st.markdown("**Pourquoi ce prix ?**")
        for bullet in expliquer(features, prix):
            st.markdown(f"- {bullet}")

with onglet2:
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
                options = (resultats["nom"] + " — " + resultats["set_name"].fillna("?")
                           + " — " + resultats["rarity"].fillna("?") + " (" + resultats["id"] + ")")
                choix = st.selectbox(f"{len(resultats)} cartes trouvées", options)
                carte = resultats[options == choix].iloc[0]
            else:
                carte = resultats.iloc[0]

            X = pd.DataFrame([carte[FEATURES]])
            prix_pred = np.expm1(modele.predict(imputer.transform(X))[0])

            st.subheader(f"{carte['nom']}  —  {carte['id']}")
            c1, c2 = st.columns(2)
            c1.metric("Prix prédit", f"${prix_pred:.2f}")
            if pd.notna(carte.get("prix_market")):
                c2.metric("Prix réel (TCGPlayer)", f"${carte['prix_market']:.2f}")
