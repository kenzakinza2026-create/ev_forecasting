"""
json_to_csv.py

Convertit les fichiers JSON du dossier "mqtt_data(copy)" (format
{"pattern": ..., "data": "<json string>"}) en un CSV agrégé, au format
attendu par DCForecaster : grille temporelle de créneaux de 15 min,
par zone et par type de chargeur (arrival_count, departure_count,
avg_energy_kWh, avg_duration_mins), avec grille complète (0 si rien
ne s'est passé sur un créneau).

CORRECTIONS APPORTÉES (par rapport à la version précédente) :
- Utilisation de la fin réelle (msg_timestamp) pour les sessions terminées
  (status "finished"/"stopped") au lieu de l'échéance (deadline).
- Les sessions en cours ne contribuent ni aux départs ni à la durée.
- La grille est bornée par les derniers créneaux réels observés.
- msg_timestamp est converti en datetime lisible.
"""

import json
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR  = SCRIPT_DIR / "mqtt_data(copy)"
OUTPUT_CSV = SCRIPT_DIR / "forecaster_input.csv"   # fichier de sortie
ERRORS_CSV = SCRIPT_DIR / "erreurs.csv"

# Si True : la grille est étendue pour couvrir chaque jour touché en entier (00:00 -> 23:45)
# Si False (recommandé) : la grille s'arrête exactement au dernier créneau réellement observé.
EXTEND_TO_FULL_DAYS = False


# ============================================================
# MAPPING EVSEID -> zone / type de chargeur
# ============================================================

ALIAS_TO_EVSEID = {
    "1-Links":  "E13-10160885",
    "1-Rechts": "E14-10160885",
    "2-Links":  "E11-10160885",
    "2-Rechts": "E12-10160885",
    "3-Links":  "E7-10160885",
    "3-Rechts": "E8-10160885",
    "4-Links":  "E5-10160885",
    "4-Rechts": "E6-10160885",
    "5-Links":  "E3-10160885",
    "5-Rechts": "E4-10160885",
    "6-Links":  "E1-10160885",
    "6-Rechts": "E2-10160885",
    "Garage-1": "E21-10160885",
    "Garage-2": "E20-10160885",
    "Garage-3": "E19-10160885",
    "Garage-4": "E18-10160885",
    "Garage-5": "E17-10160885",
    "Garage-6": "E16-10160885",
    "Garage-7": "E15-10160885",
}

EVSEID_INFO = {}

def _register(evseid, zone, charger_type, max_power_ref):
    EVSEID_INFO[evseid] = {
        "zone": zone,
        "chargerType": charger_type,
        "maxPower_ref": max_power_ref,
    }

# Enregistrement des chargeurs
for e in ["SC11", "SC12", "SC21", "SC22", "SC31"]:
    _register(e, "PublicParking", "DC", 280)

for e in ["DCW11", "DCW12", "DCW13", "DCW14",
          "DCW21", "DCW22", "DCW23", "DCW24",
          "DCGen21", "DCGen22", "DCGen23", "DCGen24"]:
    _register(e, "Non-public Underground garage", "DC", 140)

for e in ["E1-10160885", "E2-10160885", "E3-10160885", "E4-10160885",
          "E5-10160885", "E6-10160885", "E7-10160885", "E8-10160885",
          "E11-10160885", "E12-10160885", "E13-10160885", "E14-10160885"]:
    _register(e, "Employee Parking", "AC", 22)

for e in ["E15-10160885", "E16-10160885", "E17-10160885", "E18-10160885",
          "E19-10160885", "E20-10160885", "E21-10160885"]:
    _register(e, "Garage", "AC", 22)

# Toutes les combinaisons (zone, chargerType) possibles -> sert à construire la grille complète
COMBINAISONS_ZONE_TYPE = sorted({
    (info["zone"], info["chargerType"]) for info in EVSEID_INFO.values()
})

def resolve_zone_and_type(evse_id_raw):
    evse_id_raw = str(evse_id_raw)
    canonical = ALIAS_TO_EVSEID.get(evse_id_raw, evse_id_raw)
    info = EVSEID_INFO.get(canonical)
    if info is None:
        return None, None
    return info["zone"], info["chargerType"]


# ============================================================
# GESTION DES ERREURS
# ============================================================

erreurs_liste = []

def log_erreur(source, session_id, type_erreur, explication):
    erreurs_liste.append({
        "source"     : source,
        "session_id" : session_id,
        "type_erreur": type_erreur,
        "explication": explication,
    })
    print(f"[ERREUR] {type_erreur} | fichier={source} | session={session_id} | {explication}")


# ============================================================
# EXTRACTION DU PAYLOAD
# ============================================================

def extraire_charging_processes(fichier_json, source):
    """
    Extrait le timestamp global (msg_timestamp) et la liste des chargingProcesses
    à partir du format {"pattern": ..., "data": "<json string>"}.
    """
    data = fichier_json.get("data")

    # data peut être une chaîne JSON (ancien format) ou déjà un dict
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            raise KeyError(f"'data' n'est pas un JSON valide : {e}")

    if not isinstance(data, dict):
        raise KeyError("Champ 'data' introuvable ou de type inattendu.")

    timestamp_ms = data.get("timestamp")
    processes = data.get("chargingProcesses")

    # Conversion du timestamp global en datetime
    msg_timestamp = None
    if timestamp_ms is not None:
        try:
            msg_timestamp = pd.to_datetime(timestamp_ms, unit="ms")
        except Exception as e:
            log_erreur(source, "fichier entier", "TIMESTAMP_GLOBAL_INVALIDE",
                       f"Impossible de convertir 'timestamp'={timestamp_ms}. Détail : {e}")
            msg_timestamp = None

    if processes is None:
        raise KeyError("Champ 'chargingProcesses' introuvable dans 'data'.")

    return msg_timestamp, processes


# ============================================================
# VALIDATION D'UN PROCESS (MODIFIÉE)
# ============================================================

def valider_process(process, msg_timestamp, source):
    """
    Valide une session de recharge et retourne un dictionnaire avec les champs nécessaires.
    Ajoute les champs end_time et is_finished basés sur le statut réel.
    """
    session_id = process.get("_id", "inconnu")

    champs_obligatoires = [
        "_id", "evseId", "connectorId", "transactionId",
        "start", "deadline", "status", "power",
        "targetSoc", "carBatteryCapacity",
        "adjustmentOptions", "provideFlexibility",
    ]
    champs_manquants = [c for c in champs_obligatoires if c not in process]
    if champs_manquants:
        log_erreur(source, session_id, "CHAMPS_MANQUANTS",
                   f"Champs absents : {champs_manquants}")
        return None

    adj = process["adjustmentOptions"]
    if "minPower" not in adj or "maxPower" not in adj:
        log_erreur(source, session_id, "CHAMPS_MANQUANTS",
                   "adjustmentOptions doit contenir 'minPower' et 'maxPower'.")
        return None

    initial_soc = process.get("initialSoc")
    soc         = process.get("soc")

    # Vérification des champs numériques
    champs_numeriques = {
        "targetSoc"         : process["targetSoc"],
        "carBatteryCapacity": process["carBatteryCapacity"],
        "maxPower"          : adj["maxPower"],
        "minPower"          : adj["minPower"],
        "power"             : process["power"],
    }
    for nom, valeur in champs_numeriques.items():
        if not isinstance(valeur, (int, float)):
            log_erreur(source, session_id, "TYPE_INVALIDE",
                       f"'{nom}' doit être un nombre, reçu : {type(valeur).__name__} = '{valeur}'")
            return None

    if not (0 <= process["targetSoc"] <= 100):
        log_erreur(source, session_id, "SOC_HORS_LIMITES",
                   f"targetSoc={process['targetSoc']} doit être entre 0 et 100.")
        return None

    if initial_soc is not None:
        if not isinstance(initial_soc, (int, float)) or not (0 <= initial_soc <= 100):
            log_erreur(source, session_id, "SOC_HORS_LIMITES",
                       f"initialSoc={initial_soc} invalide.")
            return None
        if process["targetSoc"] <= initial_soc:
            log_erreur(source, session_id, "SOC_INCOHERENT",
                       f"targetSoc={process['targetSoc']} doit être > initialSoc={initial_soc}.")
            return None

    if process["carBatteryCapacity"] <= 0:
        log_erreur(source, session_id, "CAPACITE_INVALIDE",
                   f"carBatteryCapacity={process['carBatteryCapacity']} doit être > 0.")
        return None

    if adj["maxPower"] <= 0:
        log_erreur(source, session_id, "PUISSANCE_INVALIDE",
                   f"maxPower={adj['maxPower']} doit être > 0.")
        return None
    if adj["minPower"] < 0:
        log_erreur(source, session_id, "PUISSANCE_INVALIDE",
                   f"minPower={adj['minPower']} ne peut pas être négatif.")
        return None
    if adj["minPower"] > adj["maxPower"]:
        log_erreur(source, session_id, "PUISSANCE_INCOHERENTE",
                   f"minPower={adj['minPower']} > maxPower={adj['maxPower']}.")
        return None

    # Conversion des timestamps start / deadline (ce dernier est conservé mais non utilisé pour les départs)
    try:
        start    = pd.to_datetime(process["start"], unit="ms")
        deadline = pd.to_datetime(process["deadline"], unit="ms")
    except Exception as e:
        log_erreur(source, session_id, "TIMESTAMP_INVALIDE",
                   f"Impossible de convertir start ou deadline. Détail : {e}")
        return None

    if deadline <= start:
        log_erreur(source, session_id, "DUREE_INVALIDE",
                   f"deadline={deadline} doit être après start={start}.")
        return None

    zone, charger_type = resolve_zone_and_type(process["evseId"])
    if zone is None:
        log_erreur(source, session_id, "EVSEID_NON_RECONNU",
                   f"evseId='{process['evseId']}' introuvable.")
        return None

    # ---------- NOUVEAU : DÉTERMINATION DE LA FIN RÉELLE ----------
    status = process.get("status", "").lower()
    # Liste des statuts considérés comme terminaux (adaptez selon vos données)
    terminal_statuses = ("finished", "stopped", "completed", "done")
    is_finished = status in terminal_statuses

    end_time = None
    if is_finished:
        if msg_timestamp is not None:
            end_time = msg_timestamp   # on utilise l'instant de capture comme fin réelle
        else:
            # Si le timestamp global est manquant, on ne peut pas connaître la fin
            log_erreur(source, session_id, "FIN_SANS_TIMESTAMP",
                       "Session terminée mais msg_timestamp manquant, elle ne contribuera pas aux départs.")
            is_finished = False   # on la traite comme non terminée pour les départs
    # -------------------------------------------------------------

    return {
        "source_file"       : source,
        "msg_timestamp"     : msg_timestamp,
        "session_id"        : process["_id"],
        "evseId"            : process["evseId"],
        "zone"              : zone,
        "chargerType"       : charger_type,
        "connectorId"       : process["connectorId"],
        "transactionId"     : process["transactionId"],
        "start"             : start,
        "deadline"          : deadline,
        "end_time"          : end_time,          # NOUVEAU : fin réelle (ou None)
        "is_finished"       : is_finished,       # NOUVEAU : booléen
        "status"            : status,
        "power_kW"          : process["power"],
        "soc"               : soc,
        "initialSoc"        : initial_soc,
        "targetSoc"         : process["targetSoc"],
        "carBatteryCapacity": process["carBatteryCapacity"],
        "minPower"          : adj["minPower"],
        "maxPower"          : adj["maxPower"],
        "provideFlexibility": process["provideFlexibility"],
    }


# ============================================================
# AGRÉGATION EN GRILLE TEMPORELLE 15 MIN (MODIFIÉE)
# ============================================================

def agreger_sessions(records, extend_to_full_days=EXTEND_TO_FULL_DAYS):
    """
    Agrège les sessions en grille 15 min.
    - Arrivées : toujours basées sur start.
    - Départs et durées : uniquement pour les sessions terminées (is_finished=True),
      en utilisant end_time (qui est le msg_timestamp du fichier).
    """
    df = pd.DataFrame(records)

    # --- Arrivée : toujours à partir de start ---
    df["arrival_slot_15min"] = df["start"].dt.floor("15min")

    # --- Départs et durées : uniquement pour les terminées ---
    mask_finished = df["is_finished"] == True
    df["departure_slot_15min"] = None
    df["duration_mins"] = None

    if mask_finished.any():
        df.loc[mask_finished, "departure_slot_15min"] = df.loc[mask_finished, "end_time"].dt.floor("15min")
        df.loc[mask_finished, "duration_mins"] = (
            df.loc[mask_finished, "end_time"] - df.loc[mask_finished, "start"]
        ).dt.total_seconds() / 60

    # --- Agrégation des arrivées ---
    result_arrivals = []
    for (slot, zone, charger_type), group in df.groupby(["arrival_slot_15min", "zone", "chargerType"]):
        energies = (group["targetSoc"] - group["initialSoc"]) / 100 * group["carBatteryCapacity"]
        avg_energy = round(energies.mean(), 3) if energies.notna().any() else 0.0
        # Durée moyenne : on ne prend que les durées connues (terminées)
        durations = group["duration_mins"].dropna()
        avg_duration = round(durations.mean(), 1) if not durations.empty else 0.0
        result_arrivals.append({
            "timestamp"        : slot,
            "zone"             : zone,
            "chargerType"      : charger_type,
            "arrival_count"    : len(group),
            "avg_energy_kWh"   : avg_energy,
            "avg_duration_mins": avg_duration,
        })
    df_arrivals = pd.DataFrame(result_arrivals)

    # --- Agrégation des départs (uniquement terminées) ---
    df_departures = pd.DataFrame(columns=["timestamp", "zone", "chargerType", "departure_count"])
    if mask_finished.any():
        df_departures = (
            df[mask_finished]
              .groupby(["departure_slot_15min", "zone", "chargerType"])
              .size()
              .reset_index(name="departure_count")
              .rename(columns={"departure_slot_15min": "timestamp"})
        )

    # --- Fusion ---
    df_final = pd.merge(
        df_arrivals, df_departures,
        on=["timestamp", "zone", "chargerType"], how="outer"
    )

    # --- Bornes de la grille ---
    # On prend tous les slots d'arrivée et de départ (pour les terminées)
    slots_arr = df["arrival_slot_15min"].dropna()
    slots_dep = df.loc[mask_finished, "departure_slot_15min"].dropna()
    all_slots = pd.concat([slots_arr, slots_dep])
    if all_slots.empty:
        # Pas de données -> grille vide
        return pd.DataFrame(columns=["timestamp","zone","chargerType","arrival_count","departure_count","avg_energy_kWh","avg_duration_mins"])

    slot_min = all_slots.min()
    slot_max = all_slots.max()

    if extend_to_full_days:
        date_min = slot_min.floor("D")
        date_max = slot_max.floor("D") + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    else:
        date_min = slot_min
        date_max = slot_max

    toutes_les_slots = pd.date_range(start=date_min, end=date_max, freq="15min")

    # Grille complète pour toutes les combinaisons (zone, chargerType)
    grille = pd.MultiIndex.from_product(
        [toutes_les_slots, COMBINAISONS_ZONE_TYPE],
        names=["timestamp", "zone_type"]
    ).to_frame(index=False)
    grille[["zone", "chargerType"]] = pd.DataFrame(grille["zone_type"].tolist(), index=grille.index)
    grille = grille.drop(columns="zone_type")

    df_final = grille.merge(df_final, on=["timestamp", "zone", "chargerType"], how="left")
    df_final["arrival_count"]     = df_final["arrival_count"].fillna(0).astype(int)
    df_final["departure_count"]   = df_final["departure_count"].fillna(0).astype(int)
    df_final["avg_energy_kWh"]    = df_final["avg_energy_kWh"].fillna(0)
    df_final["avg_duration_mins"] = df_final["avg_duration_mins"].fillna(0)

    df_final = df_final.sort_values(["timestamp", "zone", "chargerType"]).reset_index(drop=True)
    return df_final


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_DIR.exists():
        print(f"[ERREUR] Dossier introuvable : {INPUT_DIR}")
        print("Crée un dossier 'mqtt_data(copy)' à côté du script et mets-y tes fichiers .json.")
        return

    fichiers = sorted(INPUT_DIR.glob("*.json"))
    if not fichiers:
        print(f"[WARN] Aucun fichier .json trouvé dans {INPUT_DIR}")
        return

    print(f"→ {len(fichiers)} fichier(s) JSON trouvé(s) dans {INPUT_DIR.name}")
    print(f"→ Mode grille temporelle : "
          f"{'jours complets (00:00-23:45)' if EXTEND_TO_FULL_DAYS else 'arrêt exact au dernier créneau réel'}")
    print("→ Utilisation de la fin réelle (msg_timestamp) pour les sessions terminées.")

    tous_les_records = []

    for fichier in fichiers:
        try:
            with open(fichier, "r", encoding="utf-8") as f:
                contenu = json.load(f)
        except json.JSONDecodeError as e:
            log_erreur(fichier.name, "fichier entier", "JSON_INVALIDE", str(e))
            continue
        except Exception as e:
            log_erreur(fichier.name, "fichier entier", "ERREUR_LECTURE", str(e))
            continue

        try:
            msg_timestamp, processes = extraire_charging_processes(contenu, fichier.name)
        except KeyError as e:
            log_erreur(fichier.name, "fichier entier", "STRUCTURE_MANQUANTE", str(e))
            continue

        if not isinstance(processes, list) or len(processes) == 0:
            log_erreur(fichier.name, "fichier entier", "LISTE_VIDE",
                       "chargingProcesses est vide ou n'est pas une liste.")
            continue

        nb_valides = 0
        for process in processes:
            record = valider_process(process, msg_timestamp, fichier.name)
            if record is not None:
                tous_les_records.append(record)
                nb_valides += 1

        print(f"  - {fichier.name} : {nb_valides}/{len(processes)} session(s) valide(s)")

    # --- Agrégation et écriture du CSV final ---
    if tous_les_records:
        df_final = agreger_sessions(tous_les_records)
        df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\n✔ Grille agrégée écrite dans {OUTPUT_CSV.name} "
              f"({len(df_final)} lignes, {len(tous_les_records)} sessions sources)")
    else:
        print("\n[WARN] Aucune session valide, CSV de sortie non créé.")

    # --- Écriture du CSV d'erreurs ---
    if erreurs_liste:
        df_err = pd.DataFrame(erreurs_liste)
        df_err.to_csv(ERRORS_CSV, index=False, encoding="utf-8-sig")
        print(f"⚠ {len(df_err)} erreur(s) écrite(s) dans {ERRORS_CSV.name}")


if __name__ == "__main__":
    main()