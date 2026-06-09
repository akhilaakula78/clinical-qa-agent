"""Generate synthetic oncology EHR data into DuckDB.

Schema mirrors Cerner/Epic-style EHR with oncology bent for MD Anderson context.
All data is synthetic — no real PHI.
"""
from __future__ import annotations

import argparse
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

random.seed(42)

CANCER_DX = [
    ("C50.911", "Malignant neoplasm of breast, unspecified site, female"),
    ("C34.90", "Malignant neoplasm of unspecified part of unspecified bronchus or lung"),
    ("C18.9", "Malignant neoplasm of colon, unspecified"),
    ("C61", "Malignant neoplasm of prostate"),
    ("C92.00", "Acute myeloblastic leukemia, not having achieved remission"),
    ("C71.9", "Malignant neoplasm of brain, unspecified"),
    ("C25.9", "Malignant neoplasm of pancreas, unspecified"),
    ("C56.9", "Malignant neoplasm of unspecified ovary"),
    ("C73", "Malignant neoplasm of thyroid gland"),
    ("C82.90", "Follicular lymphoma, unspecified, unspecified site"),
]
COMORBID_DX = [
    ("I50.9", "Heart failure, unspecified"),
    ("E11.9", "Type 2 diabetes mellitus without complications"),
    ("N18.3", "Chronic kidney disease, stage 3"),
    ("J44.9", "Chronic obstructive pulmonary disease, unspecified"),
    ("I10", "Essential (primary) hypertension"),
]
CHEMO_RX = [
    "Cisplatin", "Carboplatin", "Paclitaxel", "Doxorubicin",
    "Cyclophosphamide", "Pembrolizumab", "Trastuzumab", "Rituximab",
    "Bevacizumab", "Fluorouracil",
]
SUPPORTIVE_RX = ["Ondansetron", "Filgrastim", "Dexamethasone", "Lorazepam"]
LAB_PANEL = [
    ("WBC", "10^3/uL", 4.0, 11.0),
    ("HGB", "g/dL", 12.0, 16.0),
    ("PLT", "10^3/uL", 150, 400),
    ("ANC", "10^3/uL", 1.5, 8.0),
    ("CREATININE", "mg/dL", 0.6, 1.2),
    ("ALT", "U/L", 7, 56),
]
DEPTS = ["Oncology", "Hematology", "Radiation Oncology", "Surgical Oncology", "Palliative Care"]
RACES = ["White", "Black or African American", "Asian", "Other", "Unknown"]
ETHNICITIES = ["Hispanic or Latino", "Not Hispanic or Latino", "Unknown"]
SEXES = ["M", "F"]


def gen_patients(n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        pid = f"PT{i:06d}"
        sex = random.choice(SEXES)
        dob = date(1940, 1, 1) + timedelta(days=random.randint(0, 365 * 70))
        rows.append({
            "patient_id": pid,
            "mrn": f"MRN{random.randint(10_000_000, 99_999_999)}",
            "first_name": f"First{i}",
            "last_name": f"Last{i}",
            "date_of_birth": dob,
            "sex": sex,
            "race": random.choice(RACES),
            "ethnicity": random.choice(ETHNICITIES),
            "zip_code": f"{random.randint(77000, 77099):05d}",
            "deceased": random.random() < 0.15,
            "death_date": None,
        })
    df = pd.DataFrame(rows)
    df.loc[df["deceased"], "death_date"] = [
        date(2024, 1, 1) + timedelta(days=random.randint(0, 700))
        for _ in range(df["deceased"].sum())
    ]
    return df


def gen_providers(n: int = 25) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "provider_id": f"PR{i:04d}",
            "npi": f"{random.randint(1_000_000_000, 9_999_999_999)}",
            "name": f"Dr. Provider{i}",
            "department": random.choice(DEPTS),
            "specialty": random.choice(["Medical Oncology", "Hematology", "Radiation Oncology"]),
        }
        for i in range(n)
    ])


def gen_encounters(patients: pd.DataFrame, providers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    enc_id = 0
    for _, pt in patients.iterrows():
        n_enc = random.randint(1, 8)
        for _ in range(n_enc):
            admit = datetime(2024, 1, 1) + timedelta(
                days=random.randint(0, 700), hours=random.randint(0, 23)
            )
            los_days = random.choices([0, 1, 2, 3, 5, 8, 14], weights=[3, 4, 3, 2, 2, 1, 1])[0]
            discharge = admit + timedelta(days=los_days, hours=random.randint(0, 23))
            enc_type = random.choices(
                ["Inpatient", "Outpatient", "Emergency", "Observation"],
                weights=[2, 6, 1, 1],
            )[0]
            rows.append({
                "encounter_id": f"EN{enc_id:08d}",
                "patient_id": pt["patient_id"],
                "provider_id": providers.sample(1).iloc[0]["provider_id"],
                "encounter_type": enc_type,
                "department": random.choice(DEPTS),
                "admit_datetime": admit,
                "discharge_datetime": discharge,
                "length_of_stay_days": los_days,
                "discharge_disposition": random.choices(
                    ["Home", "SNF", "Hospice", "Expired", "AMA"],
                    weights=[7, 1, 1, 0.5, 0.5],
                )[0],
            })
            enc_id += 1
    return pd.DataFrame(rows)


def gen_diagnoses(encounters: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dx_id = 0
    for _, enc in encounters.iterrows():
        primary = random.choice(CANCER_DX)
        rows.append({
            "diagnosis_id": f"DX{dx_id:09d}",
            "encounter_id": enc["encounter_id"],
            "patient_id": enc["patient_id"],
            "icd10_code": primary[0],
            "description": primary[1],
            "is_primary": True,
            "diagnosed_date": enc["admit_datetime"].date(),
        })
        dx_id += 1
        for _ in range(random.randint(0, 3)):
            sec = random.choice(COMORBID_DX)
            rows.append({
                "diagnosis_id": f"DX{dx_id:09d}",
                "encounter_id": enc["encounter_id"],
                "patient_id": enc["patient_id"],
                "icd10_code": sec[0],
                "description": sec[1],
                "is_primary": False,
                "diagnosed_date": enc["admit_datetime"].date(),
            })
            dx_id += 1
    return pd.DataFrame(rows)


def gen_medications(encounters: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rx_id = 0
    for _, enc in encounters.iterrows():
        for _ in range(random.randint(0, 4)):
            is_chemo = random.random() < 0.6
            drug = random.choice(CHEMO_RX if is_chemo else SUPPORTIVE_RX)
            rows.append({
                "medication_id": f"RX{rx_id:09d}",
                "encounter_id": enc["encounter_id"],
                "patient_id": enc["patient_id"],
                "drug_name": drug,
                "drug_class": "Chemotherapy" if is_chemo else "Supportive Care",
                "dose_mg": round(random.uniform(10, 500), 1),
                "route": random.choice(["IV", "PO", "SC"]),
                "start_date": enc["admit_datetime"].date(),
                "end_date": enc["discharge_datetime"].date(),
            })
            rx_id += 1
    return pd.DataFrame(rows)


def gen_labs(encounters: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lab_id = 0
    for _, enc in encounters.iterrows():
        for _ in range(random.randint(1, 6)):
            name, unit, lo, hi = random.choice(LAB_PANEL)
            val = round(random.gauss((lo + hi) / 2, (hi - lo) / 3), 2)
            rows.append({
                "lab_result_id": f"LB{lab_id:010d}",
                "encounter_id": enc["encounter_id"],
                "patient_id": enc["patient_id"],
                "test_name": name,
                "value_numeric": val,
                "unit": unit,
                "reference_low": lo,
                "reference_high": hi,
                "abnormal_flag": "L" if val < lo else ("H" if val > hi else "N"),
                "collected_datetime": enc["admit_datetime"] + timedelta(hours=random.randint(0, 48)),
            })
            lab_id += 1
    return pd.DataFrame(rows)


def gen_readmissions(encounters: pd.DataFrame) -> pd.DataFrame:
    """Compute 30-day readmission flag per inpatient encounter."""
    enc = encounters[encounters["encounter_type"] == "Inpatient"].sort_values(
        ["patient_id", "admit_datetime"]
    ).reset_index(drop=True)
    enc["next_admit"] = enc.groupby("patient_id")["admit_datetime"].shift(-1)
    enc["days_to_readmit"] = (enc["next_admit"] - enc["discharge_datetime"]).dt.days
    enc["readmitted_30d"] = (enc["days_to_readmit"] >= 0) & (enc["days_to_readmit"] <= 30)
    return enc[["encounter_id", "patient_id", "discharge_datetime",
                "days_to_readmit", "readmitted_30d"]]


def load(db_path: Path, n_patients: int):
    print(f"Generating {n_patients} patients...")
    patients = gen_patients(n_patients)
    providers = gen_providers()
    print("Generating encounters...")
    encounters = gen_encounters(patients, providers)
    print("Generating diagnoses...")
    diagnoses = gen_diagnoses(encounters)
    print("Generating medications...")
    medications = gen_medications(encounters)
    print("Generating labs...")
    labs = gen_labs(encounters)
    print("Computing readmissions...")
    readmissions = gen_readmissions(encounters)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    for name, df in [
        ("patients", patients),
        ("providers", providers),
        ("encounters", encounters),
        ("diagnoses", diagnoses),
        ("medications", medications),
        ("lab_results", labs),
        ("readmissions", readmissions),
    ]:
        con.register("df_tmp", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df_tmp")
        con.unregister("df_tmp")
        n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"  {name}: {n:,} rows")
    con.close()
    print(f"\nLoaded → {db_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/clinical.duckdb")
    ap.add_argument("--patients", type=int, default=500)
    args = ap.parse_args()
    load(Path(args.db), args.patients)


if __name__ == "__main__":
    main()
