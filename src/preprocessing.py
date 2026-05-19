import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

CHEMIN_RAW = "data/raw/cartes_pokemon.csv"
CHEMIN_PROCESSED = "data/processed/"

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


def preparer_donnees():
    df = pd.read_csv(CHEMIN_RAW)
    print("shape:", df.shape)
    print(df.head())

    # on garde uniquement les cartes avec un prix réel, sans plafond
    df = df[df["prix_market"] > 0].copy()
    print(f"apres filtrage : {len(df)} cartes")

    # features construites
    annee = pd.to_numeric(df["set_annee"], errors="coerce").fillna(2000)
    df["carte_age_ans"] = 2026 - annee

    # numéro > total du set = secret rare (hors numérotation classique)
    numero_num = pd.to_numeric(df["numero"], errors="coerce")
    df["est_secret_rare"] = (
        numero_num > df["set_total_cartes"]
    ).fillna(False).astype(int)

    df["efficacite_combat"] = (
        df["max_damage"].fillna(0) / (df["cout_energie_moyen"].fillna(1) + 1)
    )
    df["est_full_art_v"] = df["is_full_art"] * df["is_v_card"]
    cols_types = [c for c in df.columns if c.startswith("type_") and df[c].dtype != object]
    df["nb_types"] = df[cols_types].sum(axis=1)

    # encoding
    df["rarity_encoded"] = df["rarity"].map(RARITY_MAP).fillna(3)
    df["est_holo_premium"] = df["is_holo"] * df["rarity_encoded"]
    df["supertype_encoded"] = pd.Categorical(df["supertype"]).codes
    df["set_serie_encoded"] = pd.Categorical(df["set_serie"]).codes
    df["standard_encoded"] = (df["legal_standard"] == "Legal").astype(int)
    df["expanded_encoded"] = (df["legal_expanded"] == "Legal").astype(int)
    df["a_faiblesse"] = df["faiblesse"].notna().astype(int)
    df["a_resistance"] = df["resistance"].notna().astype(int)

    # valeurs manquantes
    medians = {c: df[c].median() for c in ["hp", "position_dans_set", "set_total_cartes"]}
    df = df.fillna({
        **medians,
        "generation": -1, "pokedex_number": -1,
        "max_damage": 0, "total_damage_attaques": 0,
        "cout_energie_moyen": 0, "efficacite_combat": 0,
    })

    return df


if __name__ == "__main__":
    df = preparer_donnees()

    # log1p pour corriger la distribution skewed du prix
    y = np.log1p(df["prix_market"])
    X = df[FEATURES]
    print(f"\nX : {X.shape}  |  y  min={y.min():.2f}  max={y.max():.2f}  mean={y.mean():.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs(CHEMIN_PROCESSED, exist_ok=True)
    for nom, data in [("X_train", X_train), ("X_test", X_test),
                      ("y_train", y_train), ("y_test", y_test)]:
        data.to_csv(f"{CHEMIN_PROCESSED}{nom}.csv", index=False)

    print(f"splits sauvegardes -> train : {len(X_train)}  test : {len(X_test)}")
