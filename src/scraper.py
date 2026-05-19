import requests
import pandas as pd
import time
import os


URL_API = "https://api.pokemontcg.io/v2/cards"
CHEMIN_SAUVEGARDE = "data/raw/cartes_pokemon.csv"


# Isole le parsing pour ne pas polluer extraire_combat avec du nettoyage de str
def parser_degats(damage_brut):
    nettoye = (
        str(damage_brut)
        .replace("+", "").replace("x", "").replace("×", "")
    )
    return int(nettoye) if nettoye.isdigit() else 0


# La génération est une feature à part entière, mieux vaut l'isoler
def calculer_generation(pokedex_num):
    if not pokedex_num:
        return None
    seuils = [
        (151, 1), (251, 2), (386, 3), (493, 4),
        (649, 5), (721, 6), (809, 7),
    ]
    for seuil, gen in seuils:
        if pokedex_num <= seuil:
            return gen
    return 8


# Un appel par page pour pouvoir relancer proprement en cas d'erreur réseau
def recuperer_page(page):
    params = {"pageSize": 250, "page": page}
    reponse = requests.get(URL_API, params=params)
    if reponse.status_code != 200:
        print(f"erreur API page {page} :", reponse.status_code)
        return []
    return reponse.json().get("data", [])


# Boucle isolée pour pouvoir tester sans relancer tout le script
def recuperer_toutes_les_cartes():
    print("debut du scraping...")
    toutes_les_cartes = []
    page = 1

    while True:
        print(f"je recupere la page {page}...")
        cartes = recuperer_page(page)

        if not cartes:
            print("plus de cartes, on arrete")
            break

        toutes_les_cartes.extend(cartes)
        print(f"  -> {len(cartes)} cartes, total : {len(toutes_les_cartes)}")
        page += 1
        time.sleep(0.5)

    print(f"\ntotal cartes recuperees : {len(toutes_les_cartes)}")
    return toutes_les_cartes


# Les types en booléens sont les features catégorielles de base pour le modèle
def extraire_types(carte):
    types_list = carte.get("types", [])
    tous_les_types = [
        "Fire", "Water", "Grass", "Lightning", "Psychic",
        "Fighting", "Darkness", "Metal", "Dragon", "Fairy", "Colorless",
    ]
    features = {"types": ", ".join(types_list) if types_list else None}
    for nom_type in tous_les_types:
        cle = f"type_{nom_type.lower()}"
        features[cle] = 1 if nom_type in types_list else 0
    return features


# TCGPlayer : hiérarchie normal > holo > reverse pour avoir une seule cible
def extraire_prix_tcg(carte):
    prix_bruts = carte.get("tcgplayer", {}).get("prices", {})
    for variante in ["normal", "holofoil", "reverseHolofoil"]:
        if variante in prix_bruts:
            p = prix_bruts[variante]
            return {
                "type_prix":   variante,
                "prix_market": p.get("market"),
                "prix_mid":    p.get("mid"),
                "prix_low":    p.get("low"),
                "prix_high":   p.get("high"),
            }
    return {
        "type_prix": None, "prix_market": None,
        "prix_mid": None, "prix_low": None, "prix_high": None,
    }


# Cardmarket : référence europe, avg7 et trend sont les plus stables en ML
def extraire_prix_cardmarket(carte):
    cm = carte.get("cardmarket", {}).get("prices", {})
    return {
        "prix_cm_avg1":          cm.get("avg1"),
        "prix_cm_avg7":          cm.get("avg7"),
        "prix_cm_avg30":         cm.get("avg30"),
        "prix_cm_trend":         cm.get("trendPrice"),
        "prix_cm_low":           cm.get("lowPrice"),
        "prix_cm_reverse_trend": cm.get("reverseHoloTrend"),
    }


# Extrait les stats d'attaque pour garder extraire_combat sous 20 lignes
def calculer_stats_attaques(attacks):
    max_dmg = 0
    total_dmg = 0
    for atk in attacks:
        val = parser_degats(atk.get("damage", "0"))
        total_dmg += val
        max_dmg = max(max_dmg, val)
    couts = [atk.get("convertedEnergyCost", 0) for atk in attacks]
    return max_dmg, total_dmg, couts


# HP, dégâts et abilities sont corrélés au prix : on les regroupe en profil combat
def extraire_combat(carte):
    hp_brut = carte.get("hp", "0")
    hp = int(hp_brut) if str(hp_brut).isdigit() else None

    attacks = carte.get("attacks", [])
    max_dmg, total_dmg, couts = calculer_stats_attaques(attacks)
    abilities = carte.get("abilities", [])

    premier_cout = attacks[0].get("convertedEnergyCost", 0) if attacks else None
    cout_moyen = round(sum(couts) / len(couts), 2) if couts else None

    return {
        "hp":                       hp,
        "nb_attaques":              len(attacks),
        "max_damage":               max_dmg or None,
        "total_damage_attaques":    total_dmg or None,
        "a_attaque_100plus":        1 if max_dmg >= 100 else 0,
        "a_attaque_200plus":        1 if max_dmg >= 200 else 0,
        "cout_energie_moyen":       cout_moyen,
        "premier_atk_cout_energie": premier_cout,
        "nb_abilities":             len(abilities),
        "has_ability":              1 if abilities else 0,
    }


# Métadonnées du set pour contextualiser la rareté et la position de la carte
def extraire_infos_set(carte):
    set_info = carte.get("set", {})
    num_brut = carte.get("number", 0)
    set_total = set_info.get("total", 1)
    try:
        position = round(int(num_brut) / set_total, 3)
    except (ValueError, ZeroDivisionError):
        position = None
    return {
        "set_name":          set_info.get("name"),
        "set_serie":         set_info.get("series"),
        "set_annee":         set_info.get("releaseDate", "")[:4],
        "set_total_cartes":  set_info.get("total"),
        "set_printed_total": set_info.get("printedTotal"),
        "position_dans_set": position,
    }


# Subtypes en booléens : features discrètes importantes pour prédire les holo/V/ex
def extraire_subtypes(carte):
    subtypes_list = carte.get("subtypes", [])
    subtypes_str = " ".join(subtypes_list).lower()
    v_cards = ["V", "VMAX", "VSTAR"]
    ex_gx = ["EX", "GX", "ex"]
    return {
        "subtypes":    ", ".join(subtypes_list) if subtypes_list else None,
        "is_holo":     1 if "holo" in subtypes_str else 0,
        "is_full_art": 1 if "full art" in subtypes_str else 0,
        "is_v_card":   1 if any(s in subtypes_list for s in v_cards) else 0,
        "is_ex_gx":    1 if any(s in subtypes_list for s in ex_gx) else 0,
        "is_basic":    1 if "Basic" in subtypes_list else 0,
        "is_stage1":   1 if "Stage 1" in subtypes_list else 0,
        "is_stage2":   1 if "Stage 2" in subtypes_list else 0,
    }


# Identité de la carte : ce qui l'identifie dans la base, hors features ML
def extraire_identite_carte(carte):
    pokedex = carte.get("nationalPokedexNumbers", [])
    pokedex_num = pokedex[0] if pokedex else None
    weaknesses = carte.get("weaknesses", [])
    resistances = carte.get("resistances", [])
    legalities = carte.get("legalities", {})
    return {
        "id":             carte.get("id"),
        "nom":            carte.get("name"),
        "supertype":      carte.get("supertype"),
        "rarity":         carte.get("rarity"),
        "numero":         carte.get("number"),
        "artist":         carte.get("artist"),
        "pokedex_number": pokedex_num,
        "generation":     calculer_generation(pokedex_num),
        "evolves_from":   carte.get("evolvesFrom"),
        "has_evolution":  1 if carte.get("evolvesFrom") else 0,
        "faiblesse":      weaknesses[0].get("type") if weaknesses else None,
        "resistance":     resistances[0].get("type") if resistances else None,
        "retreat_cost":   len(carte.get("retreatCost", [])),
        "legal_standard": legalities.get("standard"),
        "legal_expanded": legalities.get("expanded"),
    }


# Point d'entrée pour une carte : appelle chaque extracteur et fusionne tout
def transformer_carte_en_ligne(carte):
    ligne = extraire_identite_carte(carte)
    ligne.update(extraire_infos_set(carte))
    ligne.update(extraire_subtypes(carte))
    ligne.update(extraire_types(carte))
    ligne.update(extraire_combat(carte))
    ligne.update(extraire_prix_tcg(carte))
    ligne.update(extraire_prix_cardmarket(carte))
    return ligne


# Sauvegarde en data/raw car c'est la donnée brute directement issue de l'API
def sauvegarder_dataset(lignes):
    df = pd.DataFrame(lignes)

    print("\nvoila ce qu'on a recupere :")
    print("shape:", df.shape)
    print(df.head())
    print("\nvaleurs manquantes :")
    print(df.isnull().sum())

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(CHEMIN_SAUVEGARDE, index=False)
    print(f"\nfichier sauvegarde ! {df.shape[0]} cartes, {df.shape[1]} colonnes")
    print("c'est bon !")


if __name__ == "__main__":
    toutes_les_cartes = recuperer_toutes_les_cartes()
    lignes = [transformer_carte_en_ligne(c) for c in toutes_les_cartes]
    sauvegarder_dataset(lignes)
