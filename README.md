# Pokemon Card Price Predictor

Projet de ML pour prédire le prix des cartes Pokémon à partir de leurs métadonnées (rareté, set, HP, attaques, génération...).

**Données** : Pokémon TCG API — 20 000+ cartes  
**Stack** : Python, pandas, scikit-learn, XGBoost, Streamlit

---

## Lancer l'app

```bash
# 1. installer les dépendances
pip install -r requirements.txt

# 2. récupérer les données
python src/scraper.py

# 3. preprocessing + split train/test
python src/preprocessing.py

# 4. entraîner les modèles
python src/model.py

# 5. lancer le streamlit
streamlit run app.py
```

Ouvrir **http://localhost:8501**, taper le nom d'une carte en anglais (ex: `Charizard`, `Pikachu`) et sélectionner la version souhaitée.

---

## Structure

```
├── data/
│   ├── raw/          # données brutes du scraper
│   └── processed/    # splits train/test
├── models/           # modèles entraînés (.pkl)
├── notebooks/        # preprocessing, model, evaluation
├── src/
│   ├── scraper.py
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluation.py
└── app.py            # interface Streamlit
```

## Résultats

| Modèle | R² | MAE |
|---|---|---|
| XGBoost | 0.853 | 8.69$ |
| Random Forest | 0.835 | 9.22$ |
| Ridge | 0.547 | 12.99$ |
