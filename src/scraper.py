
import requests
import pandas as pd
import time
import os

print("debut du scraping...")

toutes_les_cartes = []
page = 1

while True:
    print(f"je recupere la page {page}...")
    
    url = "https://api.pokemontcg.io/v2/cards"
    params = {
        "pageSize": 250,
        "page": page
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print("erreur avec l'api :", response.status_code)
        break
    
    data = response.json()
    cartes = data["data"]
    
    if len(cartes) == 0:
        print("plus de cartes, on arrete")
        break
    
    toutes_les_cartes.extend(cartes)
    print(f"  -> {len(cartes)} cartes recuperees, total : {len(toutes_les_cartes)}")
    
    page += 1
    time.sleep(0.5)

print(f"\ntotal cartes recuperees : {len(toutes_les_cartes)}")

lignes = []

for carte in toutes_les_cartes:

    ligne = {
        "id":         carte.get("id"),
        "nom":        carte.get("name"),
        "supertype":  carte.get("supertype"),
        "rarity":     carte.get("rarity"),
        "set_name":   carte.get("set", {}).get("name"),
        "set_serie":  carte.get("set", {}).get("series"),
        "set_annee":  carte.get("set", {}).get("releaseDate", "")[:4],
        "set_total_cartes":   carte.get("set", {}).get("total"),
        "set_printed_total":  carte.get("set", {}).get("printedTotal"),
        "numero":     carte.get("number"),
        "artist":     carte.get("artist"),
    }

    # --- HP ---
    try:
        ligne["hp"] = int(carte.get("hp", 0))
    except:
        ligne["hp"] = None

    # --- POKEDEX ET GENERATION ---
    pokedex = carte.get("nationalPokedexNumbers", [])
    ligne["pokedex_number"] = pokedex[0] if pokedex else None

    pokedex_num = pokedex[0] if pokedex else None
    if pokedex_num:
        if pokedex_num <= 151:   ligne["generation"] = 1
        elif pokedex_num <= 251: ligne["generation"] = 2
        elif pokedex_num <= 386: ligne["generation"] = 3
        elif pokedex_num <= 493: ligne["generation"] = 4
        elif pokedex_num <= 649: ligne["generation"] = 5
        elif pokedex_num <= 721: ligne["generation"] = 6
        elif pokedex_num <= 809: ligne["generation"] = 7
        else:                    ligne["generation"] = 8
    else:
        ligne["generation"] = None

    # --- TYPES EN BOOLEENS ---
    types_list = carte.get("types", [])
    ligne["types"]           = ", ".join(types_list) if types_list else None
    ligne["type_fire"]       = 1 if "Fire"      in types_list else 0
    ligne["type_water"]      = 1 if "Water"     in types_list else 0
    ligne["type_grass"]      = 1 if "Grass"     in types_list else 0
    ligne["type_lightning"]  = 1 if "Lightning" in types_list else 0
    ligne["type_psychic"]    = 1 if "Psychic"   in types_list else 0
    ligne["type_fighting"]   = 1 if "Fighting"  in types_list else 0
    ligne["type_darkness"]   = 1 if "Darkness"  in types_list else 0
    ligne["type_metal"]      = 1 if "Metal"     in types_list else 0
    ligne["type_dragon"]     = 1 if "Dragon"    in types_list else 0
    ligne["type_fairy"]      = 1 if "Fairy"     in types_list else 0
    ligne["type_colorless"]  = 1 if "Colorless" in types_list else 0

    # --- SUBTYPES ET BOOLEENS ---
    subtypes_list = carte.get("subtypes", [])
    ligne["subtypes"]     = ", ".join(subtypes_list) if subtypes_list else None
    subtypes_str          = " ".join(subtypes_list).lower()
    ligne["is_holo"]      = 1 if "holo"     in subtypes_str else 0
    ligne["is_full_art"]  = 1 if "full art" in subtypes_str else 0
    ligne["is_v_card"]    = 1 if any(s in subtypes_list for s in ["V", "VMAX", "VSTAR"]) else 0
    ligne["is_ex_gx"]     = 1 if any(s in subtypes_list for s in ["EX", "GX", "ex"])    else 0
    ligne["is_basic"]     = 1 if "Basic"   in subtypes_list else 0
    ligne["is_stage1"]    = 1 if "Stage 1" in subtypes_list else 0
    ligne["is_stage2"]    = 1 if "Stage 2" in subtypes_list else 0

    # --- EVOLUTION ---
    ligne["evolves_from"] = carte.get("evolvesFrom")
    ligne["has_evolution"] = 1 if carte.get("evolvesFrom") else 0

    # --- COMBAT ---
    attacks = carte.get("attacks", [])
    ligne["nb_attaques"] = len(attacks)

    max_dmg   = 0
    total_dmg = 0
    for atk in attacks:
        dmg = atk.get("damage", "0").replace("+","").replace("x","").replace("×","")
        try:
            val = int(dmg)
            total_dmg += val
            max_dmg = max(max_dmg, val)
        except:
            pass
    ligne["max_damage"]           = max_dmg if max_dmg > 0 else None
    ligne["total_damage_attaques"] = total_dmg if total_dmg > 0 else None
    ligne["a_attaque_100plus"]    = 1 if max_dmg >= 100 else 0
    ligne["a_attaque_200plus"]    = 1 if max_dmg >= 200 else 0

    if attacks:
        couts = [atk.get("convertedEnergyCost", 0) for atk in attacks]
        ligne["cout_energie_moyen"]        = round(sum(couts) / len(couts), 2)
        ligne["premier_atk_cout_energie"]  = attacks[0].get("convertedEnergyCost", 0)
    else:
        ligne["cout_energie_moyen"]       = None
        ligne["premier_atk_cout_energie"] = None

    # --- ABILITIES ---
    abilities = carte.get("abilities", [])
    ligne["nb_abilities"] = len(abilities)
    ligne["has_ability"]  = 1 if len(abilities) > 0 else 0

    # --- FAIBLESSES ET RESISTANCES ---
    weaknesses  = carte.get("weaknesses", [])
    resistances = carte.get("resistances", [])
    ligne["faiblesse"]   = weaknesses[0].get("type")  if weaknesses  else None
    ligne["resistance"]  = resistances[0].get("type") if resistances else None

    # --- RETREAT COST ---
    retreat = carte.get("retreatCost", [])
    ligne["retreat_cost"] = len(retreat)

    # --- POSITION DANS LE SET ---
    try:
        num   = int(carte.get("number", 0))
        total = carte.get("set", {}).get("total", 1)
        ligne["position_dans_set"] = round(num / total, 3)
    except:
        ligne["position_dans_set"] = None

    # --- LEGALITE TOURNOI ---
    legalities = carte.get("legalities", {})
    ligne["legal_standard"] = legalities.get("standard")
    ligne["legal_expanded"] = legalities.get("expanded")

    # --- PRIX TCGPLAYER ---
    prix = carte.get("tcgplayer", {}).get("prices", {})
    if "normal" in prix:
        p = prix["normal"]
        ligne["type_prix"] = "normal"
    elif "holofoil" in prix:
        p = prix["holofoil"]
        ligne["type_prix"] = "holofoil"
    elif "reverseHolofoil" in prix:
        p = prix["reverseHolofoil"]
        ligne["type_prix"] = "reverseHolofoil"
    else:
        p = {}
        ligne["type_prix"] = None

    ligne["prix_market"] = p.get("market")
    ligne["prix_mid"]    = p.get("mid")
    ligne["prix_low"]    = p.get("low")
    ligne["prix_high"]   = p.get("high")

    # --- PRIX CARDMARKET ---
    cardmarket = carte.get("cardmarket", {}).get("prices", {})
    ligne["prix_cm_avg1"]        = cardmarket.get("avg1")
    ligne["prix_cm_avg7"]        = cardmarket.get("avg7")
    ligne["prix_cm_avg30"]       = cardmarket.get("avg30")
    ligne["prix_cm_trend"]       = cardmarket.get("trendPrice")
    ligne["prix_cm_low"]         = cardmarket.get("lowPrice")
    ligne["prix_cm_reverse_trend"] = cardmarket.get("reverseHoloTrend")

    lignes.append(ligne)

# --- SAUVEGARDE ---
df = pd.DataFrame(lignes)

print("\nvoila ce qu'on a recupere :")
print("shape:", df.shape)
print(df.head())
print("\nvaleurs manquantes :")
print(df.isnull().sum())

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/cartes_pokemon.csv", index=False)
print(f"\nfichier sauvegarde ! {df.shape[0]} cartes, {df.shape[1]} colonnes")
print("c'est bon !")