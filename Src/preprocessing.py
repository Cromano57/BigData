"""
preprocessing.py
================
Pipeline de prétraitement pour le dataset BRFSS 2015 (Diabetes Health Indicators).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parent.parent
RAW_PATH  = ROOT / "data" / "raw" / "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
PROC_DIR  = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Diabetes_binary"
BINARY_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]
ORDINAL_FEATURES    = ["GenHlth", "Age", "Education", "Income"]
CONTINUOUS_FEATURES = ["BMI", "MentHlth", "PhysHlth"]


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]  {df.shape[0]:,} lignes · {df.shape[1]} colonnes chargées depuis {path.name}")
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        print(f"[validate]  ⚠️  {n_dup} doublons supprimés")
        df = df.drop_duplicates()
    else:
        print(f"[validate]  ✅ Aucun doublon")
    miss = df.isnull().sum().sum()
    print(f"[validate]  {'✅ Aucune' if miss == 0 else f'⚠️  {miss}'} valeur(s) manquante(s)")
    bmi_out = ((df["BMI"] < 10) | (df["BMI"] > 100)).sum()
    if bmi_out > 0:
        df.loc[(df["BMI"] < 10) | (df["BMI"] > 100), "BMI"] = df["BMI"].median()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Obese"]             = (df["BMI"] >= 30).astype(int)
    df["CardioRisk"]        = df["HighBP"] + df["HighChol"] + df["HeartDiseaseorAttack"] + df["Stroke"]
    df["UnhealthyLifestyle"]= (df["PhysActivity"] == 0).astype(int) + df["Smoker"] + df["HvyAlcoholConsump"]
    df["PoorHealth"]        = ((df["MentHlth"] > 14) | (df["PhysHlth"] > 14)).astype(int)
    print("[engineer]  4 nouvelles features créées : Obese, CardioRisk, UnhealthyLifestyle, PoorHealth")
    return df


def scale_continuous(train: pd.DataFrame, test: pd.DataFrame, cols: list = None):
    if cols is None:
        cols = CONTINUOUS_FEATURES + ["CardioRisk", "UnhealthyLifestyle"]
    scaler = StandardScaler()
    train[cols] = scaler.fit_transform(train[cols])
    test[cols]  = scaler.transform(test[cols])
    print(f"[scale]  StandardScaler appliqué sur : {cols}")
    return train, test


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[TARGET])
    print(f"[split]  Train : {len(train):,} | Test : {len(test):,}")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def save(train: pd.DataFrame, test: pd.DataFrame) -> None:
    train.to_csv(PROC_DIR / "train.csv", index=False)
    test.to_csv(PROC_DIR  / "test.csv",  index=False)
    print(f"[save]  ✅ train.csv et test.csv sauvegardés dans {PROC_DIR}")


def run_pipeline(scale: bool = True):
    print("=" * 55)
    print("  PIPELINE DE PRÉTRAITEMENT — BRFSS 2015")
    print("=" * 55)
    df = load_data()
    df = validate(df)
    df = engineer_features(df)
    train, test = split_data(df)
    if scale:
        train, test = scale_continuous(train, test)
    save(train, test)
    print("=" * 55)
    print(f"  Pipeline terminé. Features finales : {train.shape[1] - 1}")
    print("=" * 55)
    return train, test


if __name__ == "__main__":
    run_pipeline(scale=True)