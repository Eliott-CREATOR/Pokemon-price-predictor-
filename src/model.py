import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# chargement des donnees
X_train = pd.read_csv("../data/processed/X_train.csv")
y_train = pd.read_csv("../data/processed/y_train.csv").squeeze()

print("X_train shape:", X_train.shape)

os.makedirs("../models/", exist_ok=True)

# je remplis les valeurs manquantes avec la mediane avant d entrainer
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)

# modele 1 : Ridge (baseline simple)
print("entrainement Ridge...")
ridge = Ridge()
ridge.fit(X_train, y_train)
joblib.dump(ridge, "../models/ridge.pkl")
print("ok")

# modele 2 : Random Forest
print("entrainement Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, "../models/random_forest.pkl")
print("ok")

# modele 3 : XGBoost
print("entrainement XGBoost...")
xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
xgb.fit(X_train, y_train)
joblib.dump(xgb, "../models/xgboost.pkl")
print("ok")

# sauvegarder aussi l imputer pour l evaluation
joblib.dump(imputer, "../models/imputer.pkl")
print("tous les modeles sauvegardes !")
