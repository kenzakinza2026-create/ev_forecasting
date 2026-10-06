import csv
import json
from pathlib import Path

import pandas as pd


###########################################################################################################################
# CONFIGURATION
###########################################################################################################################

# Dossier contenant les fichiers .json (un message par fichier), placé à côté de ce script
INPUT_DIR = Path(__file__).resolve().parent / "mqtt_data(copy)"

OUTPUT_DIR      = Path(__file__).resolve().parent
OUTPUT_CSV      = OUTPUT_DIR / "forecaster_input.csv"
ERRORS_CSV      = OUTPUT_DIR / "erreurs.csv"
SESSIONS_BUFFER = OUTPUT_DIR / "sessions_buffer.csv"

DUREE_MAX_MIN = 1440

SESSION_RETENTION_DAYS = 30

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


###########################################################################################################################
# MAPPING — EVSEID canonique
###########################################################################################################################

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

# Filtre de périmètre : ne garder que les sessions DC en charge publique
# (les 5 bornes SC11/SC12/SC21/SC22/SC31, zone "PublicParking", chargerType "DC").
# Pour inclure aussi les DC non-publiques (garage souterrain), ajouter :
#   ("Non-public Underground garage", "DC")
ZONES_TYPES_CIBLES = {
    ("PublicParking", "DC"),
}

COMBINAISONS_ZONE_TYPE = sorted(ZONES_TYPES_CIBLES)


def resolve_zone_and_type(evse_id_raw):
    evse_id_raw = str(evse_id_raw)
    canonical = ALIAS_TO_EVSEID.get(evse_id_raw, evse_id_raw)
    info = EVSEID_INFO.get(canonical)
    if info is None:
        return None, None
    return info["zone"], info["chargerType"]


###########################################################################################################################
# ETAT PARTAGE (traitement séquentiel, plus besoin de threading/lock)
###########################################################################################################################

sessions_dict = {}
erreurs_counts = {}
_erreurs_header_ecrit = ERRORS_CSV.exists() and ERRORS_CSV.stat().st_size > 0


def log_erreur(source, session_id, type_erreur, explication):
    global _erreurs_header_ecrit
    entry = {
        "horodatage" : pd.Timestamp.utcnow().tz_localize(None).isoformat(),
        "source"     : source,
        "session_id" : session_id,
        "type_erreur": type_erreur,
        "explication": explication,
    }
    erreurs_counts[type_erreur] = erreurs_counts.get(type_erreur, 0) + 1
    try:
        with open(ERRORS_CSV, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(entry.keys()))
            if not _erreurs_header_ecrit:
                writer.writeheader()
                _erreurs_header_ecrit = True
            writer.writerow(entry)
    except Exception as e:
        print(f"[WARN] impossible d'écrire dans erreurs.csv : {e}")
    print(f"[ERREUR] {type_erreur} | session={session_id} | {explication}")


###########################################################################################################################
# DECODAGE JSON IMBRIQUE
#
# Les fichiers observés contiennent parfois plusieurs niveaux de JSON encodés en chaîne
# (ex: {"pattern":..., "data": "{\"timestamp\":..., \"chargingProcesses\":[...] }"})
# On dé-sérialise récursivement toute valeur de type string qui ressemble à du JSON.
###########################################################################################################################

def _dejsonifier(valeur, profondeur=0, profondeur_max=5):
    if profondeur > profondeur_max:
        return valeur
    if isinstance(valeur, str):
        s = valeur.strip()
        if s[:1] in "{[":
            try:
                return _dejsonifier(json.loads(s), profondeur + 1, profondeur_max)
            except json.JSONDecodeError:
                return valeur
        return valeur
    if isinstance(valeur, dict):
        return {k: _dejsonifier(v, profondeur + 1, profondeur_max) for k, v in valeur.items()}
    if isinstance(valeur, list):
        return [_dejsonifier(v, profondeur + 1, profondeur_max) for v in valeur]
    return valeur


def extraire_charging_processes(data):
    timestamp = data.get("timestamp")
    if timestamp is None and "payload" in data:
        timestamp = data["payload"].get("timestamp") if isinstance(data["payload"], dict) else None
    if timestamp is None and "data" in data and isinstance(data["data"], dict):
        timestamp = data["data"].get("timestamp")

    for getter in (
        lambda d: d["payload"]["data"]["chargingProcesses"],
        lambda d: d["payload"]["chargingProcesses"],
        lambda d: d["data"]["chargingProcesses"],
        lambda d: d["chargingProcesses"],
    ):
        try:
            return timestamp, getter(data)
        except (KeyError, TypeError):
            continue

    def find_list(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "chargingProcesses" and isinstance(value, list):
                    return value
                if isinstance(value, (dict, list)):
                    result = find_list(value)
                    if result is not None:
                        return result
        elif isinstance(obj, list):
            for item in obj:
                result = find_list(item)
                if result is not None:
                    return result
        return None

    processes = find_list(data)
    if processes is not None:
        return timestamp, processes

    raise KeyError("Impossible de localiser 'chargingProcesses' dans le message.")


def valider_process(process, mqtt_timestamp, source):
    session_id = process.get("_id", "inconnu")

    champs_obligatoires = [
        "_id", "evseId", "connectorId", "transactionId",
        "start", "deadline", "status", "power",
        "soc", "initialSoc", "targetSoc",
        "carBatteryCapacity", "adjustmentOptions", "provideFlexibility",
    ]
    champs_manquants = [c for c in champs_obligatoires if c not in process]
    if champs_manquants:
        log_erreur(source, session_id, "CHAMPS_MANQUANTS",
                   f"Champs absents dans la session : {champs_manquants}")
        return None

    adj = process["adjustmentOptions"]
    if "minPower" not in adj or "maxPower" not in adj:
        log_erreur(source, session_id, "CHAMPS_MANQUANTS",
                   "adjustmentOptions doit contenir 'minPower' et 'maxPower'.")
        return None

    champs_numeriques = {
        "initialSoc"        : process["initialSoc"],
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

    if not (0 <= process["initialSoc"] <= 100):
        log_erreur(source, session_id, "SOC_HORS_LIMITES",
                   f"initialSoc={process['initialSoc']} doit être entre 0 et 100.")
        return None

    if not (0 <= process["targetSoc"] <= 100):
        log_erreur(source, session_id, "SOC_HORS_LIMITES",
                   f"targetSoc={process['targetSoc']} doit être entre 0 et 100.")
        return None

    if process["targetSoc"] <= process["initialSoc"]:
        log_erreur(source, session_id, "SOC_INCOHERENT",
                   f"targetSoc={process['targetSoc']} doit être > initialSoc={process['initialSoc']}.")
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

    duree_minutes = (deadline - start).total_seconds() / 60
    if duree_minutes > DUREE_MAX_MIN:
        log_erreur(source, session_id, "DUREE_SUSPECTE",
                   f"Durée = {duree_minutes:.0f} min (> {DUREE_MAX_MIN} min).")
        return None

    zone, charger_type = resolve_zone_and_type(process["evseId"])
    if zone is None:
        log_erreur(source, session_id, "EVSEID_NON_RECONNU",
                   f"evseId='{process['evseId']}' introuvable.")
        return None

    if (zone, charger_type) not in ZONES_TYPES_CIBLES:
        # Hors périmètre (pas DC/PublicParking) : ignoré silencieusement, ce n'est pas une erreur.
        return None

    return {
        "mqtt_timestamp"    : mqtt_timestamp,
        "session_id"        : process["_id"],
        "evseId"            : process["evseId"],
        "zone"              : zone,
        "chargerType"       : charger_type,
        "connectorId"       : process["connectorId"],
        "transactionId"     : process["transactionId"],
        "start"             : start.isoformat(),
        "deadline"          : deadline.isoformat(),
        "status"            : process["status"],
        "power_kW"          : process["power"],
        "soc"               : process["soc"],
        "initialSoc"        : process["initialSoc"],
        "targetSoc"         : process["targetSoc"],
        "carBatteryCapacity": process["carBatteryCapacity"],
        "minPower"          : adj["minPower"],
        "maxPower"          : adj["maxPower"],
        "provideFlexibility": process["provideFlexibility"],
    }


###########################################################################################################################
# PERSISTANCE DES SESSIONS
###########################################################################################################################

def charger_sessions_existantes():
    if SESSIONS_BUFFER.exists():
        try:
            df = pd.read_csv(SESSIONS_BUFFER, encoding="utf-8-sig")
            for _, row in df.iterrows():
                sessions_dict[row["session_id"]] = row.to_dict()
            print(f"→ {len(sessions_dict)} session(s) rechargée(s) depuis {SESSIONS_BUFFER.name}")
        except Exception as e:
            print(f"[WARN] impossible de recharger le buffer de sessions : {e}")


def purger_sessions_anciennes():
    if SESSION_RETENTION_DAYS is None:
        return
    seuil = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=SESSION_RETENTION_DAYS)
    a_supprimer = [
        sid for sid, rec in sessions_dict.items()
        if pd.to_datetime(rec["deadline"]) < seuil
    ]
    for sid in a_supprimer:
        del sessions_dict[sid]
    if a_supprimer:
        print(f"→ {len(a_supprimer)} session(s) de plus de {SESSION_RETENTION_DAYS}j purgée(s) de la mémoire")
        sauvegarder_sessions_buffer()


def sauvegarder_sessions_buffer():
    if not sessions_dict:
        return
    df = pd.DataFrame(list(sessions_dict.values()))
    try:
        df.to_csv(SESSIONS_BUFFER, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[WARN] impossible d'écrire {SESSIONS_BUFFER.name} : {e}")


###########################################################################################################################
# TRAITEMENT D'UN MESSAGE (fichier JSON)
###########################################################################################################################

def traiter_payload(data, source):
    try:
        mqtt_timestamp, charging_processes = extraire_charging_processes(data)
    except KeyError as e:
        log_erreur(source, "message entier", "STRUCTURE_MANQUANTE", str(e))
        return

    if not isinstance(charging_processes, list) or len(charging_processes) == 0:
        log_erreur(source, "message entier", "LISTE_VIDE",
                   "chargingProcesses est vide ou n'est pas une liste.")
        return

    nb_valides = 0
    for process in charging_processes:
        record = valider_process(process, mqtt_timestamp, source)
        if record is not None:
            sessions_dict[record["session_id"]] = record  # upsert
            nb_valides += 1

    if nb_valides:
        print(f"{source} : {nb_valides} session(s) valide(s) mise(s) à jour "
              f"(total en mémoire : {len(sessions_dict)})")


def traiter_dossier_entree():
    if not INPUT_DIR.exists():
        print(f"[ERREUR] Dossier introuvable : {INPUT_DIR}")
        return

    fichiers = sorted(INPUT_DIR.glob("*.json"))
    if not fichiers:
        print(f"[INFO] Aucun fichier .json trouvé dans {INPUT_DIR}")
        return

    print(f"→ {len(fichiers)} fichier(s) trouvé(s) dans {INPUT_DIR.name}")
    for chemin in fichiers:
        try:
            with open(chemin, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log_erreur(chemin.name, "message entier", "JSON_INVALIDE", str(e))
            continue

        data = _dejsonifier(data)
        traiter_payload(data, source=chemin.name)


###########################################################################################################################
# RECALCUL DE L'AGREGATION
###########################################################################################################################

def recalculer_et_sauvegarder():
    if not sessions_dict:
        print("[INFO] Aucune session valide en mémoire, rien à agréger.")
        return
    df = pd.DataFrame(list(sessions_dict.values()))

    df["start"]    = pd.to_datetime(df["start"])
    df["deadline"] = pd.to_datetime(df["deadline"])

    df["arrival_slot_15min"]   = df["start"].dt.floor("15min")
    df["departure_slot_15min"] = df["deadline"].dt.floor("15min")

    # arrivées --
    result_arrivals = []
    for (slot, zone, charger_type), group in df.groupby(["arrival_slot_15min", "zone", "chargerType"]):
        energies = (group["targetSoc"] - group["initialSoc"]) / 100 * group["carBatteryCapacity"]
        avg_energy = round(energies.mean(), 3)
        durations = (group["deadline"] - group["start"]).dt.total_seconds() / 60
        avg_duration = round(durations.mean(), 1)
        result_arrivals.append({
            "timestamp"        : slot,
            "zone"             : zone,
            "chargerType"      : charger_type,
            "arrival_count"    : len(group),
            "avg_energy_kWh"   : avg_energy,
            "avg_duration_mins": avg_duration,
        })
    df_arrivals = pd.DataFrame(result_arrivals)

    # départs --
    df_departures = (
        df.groupby(["departure_slot_15min", "zone", "chargerType"])
          .size()
          .reset_index(name="departure_count")
          .rename(columns={"departure_slot_15min": "timestamp"})
    )

    # fusion --
    df_final = pd.merge(
        df_arrivals, df_departures,
        on=["timestamp", "zone", "chargerType"], how="outer"
    )

    # grille complète 96 slots/jour --
    date_min = min(df["arrival_slot_15min"].min(), df["departure_slot_15min"].min()).floor("D")
    date_max = max(df["arrival_slot_15min"].max(), df["departure_slot_15min"].max()).floor("D") \
               + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    toutes_les_slots = pd.date_range(start=date_min, end=date_max, freq="15min")

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

    tmp_path = OUTPUT_CSV.with_suffix(".tmp")
    df_final.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    tmp_path.replace(OUTPUT_CSV)

    print(f"CSV mis à jour → {OUTPUT_CSV.name} "
          f"({len(df_final)} lignes, {len(sessions_dict)} sessions en mémoire)")


###########################################################################################################################
# MAIN
###########################################################################################################################

if __name__ == "__main__":
    charger_sessions_existantes()
    traiter_dossier_entree()
    sauvegarder_sessions_buffer()
    purger_sessions_anciennes()
    recalculer_et_sauvegarder()

    if erreurs_counts:
        print("\nRésumé des erreurs :")
        for type_err, n in sorted(erreurs_counts.items(), key=lambda x: -x[1]):
            print(f"   {type_err} : {n}")

    print("Terminé.")