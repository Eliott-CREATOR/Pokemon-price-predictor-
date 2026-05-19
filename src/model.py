import pandas as pd
import joblib
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

CHEMIN_PROCESSED = "data/processed/"
CHEMIN_MODELES = "models/"

X_train = pd.read_csv(f"{CHEMIN_PROCESSED}X_train.csv")
y_train = pd.read_csv(f"{CHEMIN_PROCESSED}y_train.csv").squeeze()

imputer = SimpleImputer(strategy="median")

modeles = {
    "ridge":         make_pipeline(imputer, StandardScaler(), Ridge()),
    "random_forest": make_pipeline(imputer, RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    "xgboost":       make_pipeline(imputer, XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=-1)),
}

os.makedirs(CHEMIN_MODELES, exist_ok=True)
for nom, modele in modeles.items():
    print(f"entrainement {nom}...")
    modele.fit(X_train, y_train)
    joblib.dump(modele, f"{CHEMIN_MODELES}{nom}.pkl")
    print(f"  -> sauvegarde : models/{nom}.pkl")
