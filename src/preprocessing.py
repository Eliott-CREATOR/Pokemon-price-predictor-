import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

chemin_csv = "../data/raw/cartes_pokemon.csv"
dossier_output = "../data/processed/"

rarete_en_chiffre = {
    "Common": 1, "Uncommon": 2, "Rare": 3, "Rare Holo": 4,
    "Rare Holo EX": 5, "Rare Holo GX": 5, "Rare Holo V": 5,
    "Rare Ultra": 6, "Rare Holo VMAX": 6,
    "Rare Secret": 7, "Amazing Rare": 7,
    "Rare Rainbow": 8, "Rare Shining": 8, "Rare Shiny GX": 8, "LEGEND": 8,
}
colonnes_a_garder = [
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
def creation_features(df):
    
    annee = pd.to_numeric(df["set_annee"], errors="coerce").fillna(2000)
    df["age_du_set"] = 2026 - annee

    # carte secret rare = son numero depasse le total du set
    numero = pd.to_numeric(df["numero"], errors="coerce")
    df["est_secret_rare"] = (numero > df["set_total_cartes"]).fillna(False).astype(int)

    # features combat
    df["ratio_degats_energie"] = df["max_damage"].fillna(0) / (df["cout_energie_moyen"].fillna(1) + 1)
    df["full_art_et_v"] = df["is_full_art"] * df["is_v_card"]

    # compter le nombre de types
    cols_types = [c for c in df.columns if c.startswith("type_") and df[c].dtype != object]
    df["nb_types_total"] = df[cols_types].sum(axis=1)
    return df


def encoder(df):
    df["score_rarete"] = df["rarity"].map(rarete_en_chiffre).fillna(3)
    df["holo_x_rarete"] = df["is_holo"] * df["score_rarete"]

    # one-hot encoding pour supertype, pas de hierarchie entre pokemon trainer energy
    dummies = pd.get_dummies(df["supertype"], prefix="supertype")
    df = pd.concat([df, dummies], axis=1)

    # encodage des series de cartes
    df["set_serie_encode"] = pd.Categorical(df["set_serie"]).codes #demander aux profs pour ça
    df["legal_en_standard"] = (df["legal_standard"] == "Legal").astype(int)
    df["legal_en_expanded"] = (df["legal_expanded"] == "Legal").astype(int)
    df["a_faiblesse"] = df["faiblesse"].notna().astype(int)
    df["a_resistance"] = df["resistance"].notna().astype(int)
    return df


def gerer_manquants(df):
    df["hp"] = df["hp"].fillna(df["hp"].median())
    df["position_dans_set"] = df["position_dans_set"].fillna(df["position_dans_set"].median())
    df["set_total_cartes"] = df["set_total_cartes"].fillna(df["set_total_cartes"].median())
    df["generation"] = df["generation"].fillna(-1)
    df["pokedex_number"] = df["pokedex_number"].fillna(-1)
    df["max_damage"] = df["max_damage"].fillna(0)
    df["total_damage_attaques"] = df["total_damage_attaques"].fillna(0)
    df["cout_energie_moyen"] = df["cout_energie_moyen"].fillna(0)
    return df


def preparer_donnees():
    df = pd.read_csv(chemin_csv)
    print("shape:", df.shape)

    # je garde que les cartes avec un prix
    df = df[df["prix_market"] > 0].copy()
    print(f"{len(df)} cartes avec un prix")

    df = creation_features(df)
    df = encoder(df)
    df = gerer_manquants(df)
    return df


df = preparer_donnees()

# le prix est tres skewé donc je prends le log et on vérifie qu'il n'y a pas de bug 
y = np.log1p(df["prix_market"])
X = df[colonnes_a_garder]

print("X shape:", X.shape)
print("y min:", round(y.min(), 2), "max:", round(y.max(), 2))

# 80% train 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("train:", len(X_train), "cartes | test:", len(X_test), "cartes")

# sauvegarde des splits
os.makedirs(dossier_output, exist_ok=True)

X_train.to_csv(dossier_output + "X_train.csv", index=False)
X_test.to_csv(dossier_output + "X_test.csv", index=False)
y_train.to_csv(dossier_output + "y_train.csv", index=False)
y_test.to_csv(dossier_output + "y_test.csv", index=False)

print("parfait ! les données sont prêtes pour l'entraînement du modèle.")