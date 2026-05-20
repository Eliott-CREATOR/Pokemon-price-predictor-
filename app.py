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

SERIES_SIGNIFICATIVES = ["NP", "E-Card", "POP", "Other", "Autre (récent)"]

# Basic/Stage1/Stage2/V/VMAX/EX-GX → colonnes correspondantes
TYPES_CARTE = {
    "Basic":   {"is_basic": 1},
    "Stage 1": {"is_stage1": 1},
    "Stage 2": {"is_stage2": 1},
    "Carte V": {"is_v_card": 1},
    "Carte VMAX": {"is_v_card": 1},
    "EX / GX": {"is_ex_gx": 1},
    "Trainer": {"supertype_Trainer": 1, "supertype_Pokémon": 0, "supertype_Energy": 0},
    "Énergie": {"supertype_Energy": 1, "supertype_Pokémon": 0, "supertype_Trainer": 0},
}

FEATURES = [
    "hp", "retreat_cost", "has_evolution", "generation", "pokedex_number",
    "age_du_set", "set_total_cartes", "position_dans_set",
    "serie_NP", "serie_E-Card", "serie_POP", "serie_Other",
    "supertype_Pokémon", "supertype_Trainer", "supertype_Energy",
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
def charger_medians():
    return pd.read_csv("data/processed/X_train.csv").median().to_dict()


def predire(features_dict, modele, imputer, medians):
    X_base = medians.copy()
    X_base.update(features_dict)
    X = pd.DataFrame([X_base])[FEATURES]
    log_pred = modele.predict(imputer.transform(X))[0]
    prix = np.expm1(log_pred)
    return prix, np.expm1(log_pred - 0.15), np.expm1(log_pred + 0.15)


def expliquer(f, prix, modele):
    imp = dict(zip(FEATURES, modele.feature_importances_))
    total = sum(imp.values())
    pct = lambda feat: round(float(imp.get(feat, 0)) / float(total) * 100, 1)

    rarity_labels = {1: "Common", 2: "Uncommon", 3: "Rare", 4: "Rare Holo",
                     5: "Rare Holo V/EX/GX", 6: "Rare Ultra/VMAX", 7: "Rare Secret", 8: "Rainbow/Shining"}

    bullets = []
    r = f["score_rarete"]
    if r >= 7:
        desc = "moins de 3% des cartes atteignent ce niveau — prime de rareté extrême"
    elif r >= 5:
        desc = "segment premium, cartes recherchées en compétitif et en collection"
    elif r >= 4:
        desc = "holo standard, demande stable mais pas exceptionnelle"
    else:
        desc = "rareté commune, le prix dépendra surtout des autres facteurs"
    bullets.append(f"**Rareté {rarity_labels.get(r, '?')}** — poids modèle : **{pct('score_rarete')}%** — {desc}")

    if f.get("full_art_et_v", 0):
        bullets.append(f"**Full Art + Carte V** — poids : **{pct('full_art_et_v')}%** — illustration pleine page sur une carte compétitive, les collectionneurs payent en moyenne 2-3× le prix d'une version standard")
    elif f.get("is_full_art", 0):
        bullets.append(f"**Full Art** — poids : **{pct('is_full_art')}%** — artwork premium, valorise la carte même sans usage en tournoi")

    if f.get("est_secret_rare", 0):
        bullets.append(f"**Secret Rare** — poids : **{pct('est_secret_rare')}%** — numéro hors set, tirage ~1 pack sur 72, forte demande des chasseurs de complétions")

    if f.get("holo_x_rarete", 0) >= 4:
        bullets.append(f"**Holo × Rareté {r}/8** — poids : **{pct('holo_x_rarete')}%** — double signal de valeur, prime perçue par les acheteurs")

    age = f.get("age_du_set", 5)
    annee_set = 2026 - int(age)
    if age <= 2:
        bullets.append(f"**Set {annee_set} (très récent)** — poids : **{pct('age_du_set')}%** — marché en phase de découverte, prix instables mais demande active des joueurs en tournoi")
    elif age >= 20:
        bullets.append(f"**Set {annee_set} (vintage)** — poids : **{pct('age_du_set')}%** — exemplaires en bon état de plus en plus rares, valeur nostalgique croissante")
    else:
        bullets.append(f"**Set {annee_set}** — poids : **{pct('age_du_set')}%** — set établi, prix stabilisé, marché liquide")

    if f.get("has_ability", 0):
        bullets.append(f"**Ability présente** — poids : **{pct('has_ability')}%** — cartes avec capacités passives souvent jouées en tournoi, soutient la demande long terme")

    if f.get("max_damage", 0) >= 200:
        bullets.append(f"**{int(f['max_damage'])} dégâts max** — poids : **{pct('max_damage')}%** — carte offensivement viable en compétitif")

    return bullets


modele, imputer = charger_modele()
medians = charger_medians()

st.title("Pokemon Card Price Predictor")
st.caption("Estimez la valeur marché d'une nouvelle carte avant sa sortie officielle.")

col1, col2 = st.columns(2)

with col1:
    rarete = st.selectbox("Rareté", list(RARITY_MAP.keys()), index=3)
    serie = st.selectbox("Série du set", SERIES_SIGNIFICATIVES, index=4)
    type_carte = st.selectbox("Type de carte", list(TYPES_CARTE.keys()), index=0)
    generation = st.selectbox("Génération", list(range(1, 9)), index=0)

with col2:
    hp = st.slider("HP", 0, 400, 120, step=10)
    max_dmg = st.slider("Dégâts max", 0, 350, 120, step=10)
    is_holo = st.checkbox("Holo")
    is_full_art = st.checkbox("Full Art")
    has_ability = st.checkbox("A une Ability")
    est_secret = st.checkbox("Secret Rare (numéro hors set)")
    legal_expanded = st.checkbox("Légal en Expanded")

if st.button("Prédire le prix", type="primary"):
    rarity_encoded = RARITY_MAP[rarete]
    type_overrides = TYPES_CARTE[type_carte]

    series_cols = {
        "serie_NP":     1 if serie == "NP" else 0,
        "serie_E-Card": 1 if serie == "E-Card" else 0,
        "serie_POP":    1 if serie == "POP" else 0,
        "serie_Other":  1 if serie == "Other" else 0,
    }

    features = {
        "score_rarete": rarity_encoded,
        **series_cols,
        "hp":           hp,
        "max_damage":        max_dmg,
        "total_damage_attaques": max_dmg,
        "a_attaque_100plus": 1 if max_dmg >= 100 else 0,
        "a_attaque_200plus": 1 if max_dmg >= 200 else 0,
        "generation":        generation,
        "is_holo":           int(is_holo),
        "is_full_art":       int(is_full_art),
        "has_ability":       int(has_ability),
        "nb_abilities":      int(has_ability),
        "est_secret_rare":   int(est_secret),
        "holo_x_rarete":     int(is_holo) * rarity_encoded,
        "full_art_et_v":     int(is_full_art and type_overrides.get("is_v_card", 0)),
        "ratio_degats_energie": max_dmg / 3,
        "supertype_Pokémon": 1,
        "supertype_Trainer": 0,
        "supertype_Energy":  0,
        "legal_en_expanded": int(legal_expanded),
        **type_overrides,
    }

    prix, prix_bas, prix_haut = predire(features, modele, imputer, medians)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Fourchette basse", f"${prix_bas:.2f}")
    c2.metric("Prix estimé", f"${prix:.2f}")
    c3.metric("Fourchette haute", f"${prix_haut:.2f}")

    st.markdown("**Pourquoi ce prix ?**")
    for bullet in expliquer(features, prix, modele):
        st.markdown(f"- {bullet}")
