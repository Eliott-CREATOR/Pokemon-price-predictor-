import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

CHEMIN_PROCESSED = "data/processed/"
CHEMIN_MODELES = "models/"

X_test = pd.read_csv(f"{CHEMIN_PROCESSED}X_test.csv")
y_test = pd.read_csv(f"{CHEMIN_PROCESSED}y_test.csv").squeeze()

resultats = []

for nom in ["ridge", "random_forest", "xgboost"]:
    modele = joblib.load(f"{CHEMIN_MODELES}{nom}.pkl")
    y_pred = modele.predict(X_test)

    # métriques en espace réel pour que ce soit lisible en dollars
    y_reel = np.expm1(y_test)
    y_pred_reel = np.expm1(y_pred)

    resultats.append({
        "modele":   nom,
        "R²":       round(r2_score(y_test, y_pred), 4),
        "RMSE ($)": round(np.sqrt(mean_squared_error(y_reel, y_pred_reel)), 2),
        "MAE ($)":  round(mean_absolute_error(y_reel, y_pred_reel), 2),
    })

df_resultats = pd.DataFrame(resultats).sort_values("R²", ascending=False)
print(df_resultats.to_string(index=False))
print(f"\nmeilleur modele : {df_resultats.iloc[0]['modele']}")
