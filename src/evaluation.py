import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

# chargement des donnees
X_train = pd.read_csv("../data/processed/X_train.csv")
y_train = pd.read_csv("../data/processed/y_train.csv").squeeze()
X_test = pd.read_csv("../data/processed/X_test.csv")
y_test = pd.read_csv("../data/processed/y_test.csv").squeeze()

# imputer pour gerer les valeurs manquantes
imputer = joblib.load("../models/imputer.pkl")
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

# evaluation Ridge
ridge = joblib.load("../models/ridge.pkl")
y_pred_ridge = ridge.predict(X_test)
scores_ridge = cross_val_score(ridge, X_train, y_train, cv=5, scoring="r2")
print("Ridge")
print("  R2  :", round(r2_score(y_test, y_pred_ridge), 4))
print("  R2 cross-val :", round(scores_ridge.mean(), 4), "(+/-", round(scores_ridge.std(), 4), ")")
print("  RMSE:", round(np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred_ridge))), 2), "euros")
print("  MAE :", round(mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_ridge)), 2), "euros")

# evaluation Random Forest
rf = joblib.load("../models/random_forest.pkl")
y_pred_rf = rf.predict(X_test)
scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring="r2")
print("\nRandom Forest")
print("  R2  :", round(r2_score(y_test, y_pred_rf), 4))
print("  R2 cross-val :", round(scores_rf.mean(), 4), "(+/-", round(scores_rf.std(), 4), ")")
print("  RMSE:", round(np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred_rf))), 2), "euros")
print("  MAE :", round(mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_rf)), 2), "euros")

# evaluation XGBoost
xgb = joblib.load("../models/xgboost.pkl")
y_pred_xgb = xgb.predict(X_test)
scores_xgb = cross_val_score(xgb, X_train, y_train, cv=5, scoring="r2")
print("\nXGBoost")
print("  R2  :", round(r2_score(y_test, y_pred_xgb), 4))
print("  R2 cross-val :", round(scores_xgb.mean(), 4), "(+/-", round(scores_xgb.std(), 4), ")")
print("  RMSE:", round(np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred_xgb))), 2), "euros")
print("  MAE :", round(mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_xgb)), 2), "euros")