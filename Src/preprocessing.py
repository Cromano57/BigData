"""
Pipeline de prétraitement pour le dataset BRFSS 2015 (Diabetes Health Indicators).

Usage :
    python src/preprocessing.py

Produit :
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─── Chemins ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
RAW_PATH  = ROOT / "data" / "raw" / "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
PROC_DIR  = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ─── Métadonnées ──────────────────────────────────────────────────────────────
TARGET = "Diabetes_binary"

BINARY_FEATURES     = ["HighBP","HighChol","CholCheck","Smoker","Stroke",
                        "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies",
                        "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","DiffWalk","Sex"]
ORDINAL_FEATURES    = ["GenHlth","Age","Education","Income"]
CONTINUOUS_FEATURES = ["BMI","MentHlth","PhysHlth"]
SCALE_COLS          = CONTINUOUS_FEATURES + ["CardioRisk","UnhealthyLifestyle"]


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]      {df.shape[0]:,} lignes · {df.shape[1]} colonnes — {path.name}")
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        print(f"[validate]    {n_dup} doublons supprimés")
        df = df.drop_duplicates()
    else:
        print(f"[validate]   Aucun doublon")
    miss = df.isnull().sum().sum()
    print(f"[validate]  {' Aucune' if miss == 0 else f'  {miss}'} valeur(s) manquante(s)")
    bmi_out = ((df["BMI"] < 10) | (df["BMI"] > 100)).sum()
    if bmi_out > 0:
        df.loc[(df["BMI"] < 10) | (df["BMI"] > 100), "BMI"] = df["BMI"].median()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Obese"]              = (df["BMI"] >= 30).astype(int)
    df["CardioRisk"]         = df["HighBP"] + df["HighChol"] + df["HeartDiseaseorAttack"] + df["Stroke"]
    df["UnhealthyLifestyle"] = (df["PhysActivity"] == 0).astype(int) + df["Smoker"] + df["HvyAlcoholConsump"]
    df["PoorHealth"]         = ((df["MentHlth"] > 14) | (df["PhysHlth"] > 14)).astype(int)
    print("[engineer]  4 features : Obese · CardioRisk · UnhealthyLifestyle · PoorHealth")
    return df


def split_data(df, test_size=0.20, val_size=0.20, random_state=42):
    """Split stratifié : 64% train / 16% val / 20% test."""
    train_val, test = train_test_split(df, test_size=test_size,
                                       random_state=random_state, stratify=df[TARGET])
    train, val = train_test_split(train_val, test_size=val_size,
                                  random_state=random_state, stratify=train_val[TARGET])
    n = len(df)
    print(f"[split]     Train {len(train):,} ({len(train)/n*100:.0f}%) · "
          f"Val {len(val):,} ({len(val)/n*100:.0f}%) · "
          f"Test {len(test):,} ({len(test)/n*100:.0f}%)")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def scale_continuous(train, val, test, cols=None, save_scaler=True):
    """StandardScaler fit sur train, transform sur val et test."""
    if cols is None:
        cols = SCALE_COLS
    scaler = StandardScaler()
    train = train.copy(); val = val.copy(); test = test.copy()
    train[cols] = scaler.fit_transform(train[cols])
    val[cols]   = scaler.transform(val[cols])
    test[cols]  = scaler.transform(test[cols])
    if save_scaler:
        joblib.dump(scaler, PROC_DIR / "scaler.joblib")
        print(f"[scale]     Scaler sauvegardé → scaler.joblib")
    print(f"[scale]     StandardScaler sur : {cols}")
    return train, val, test


def save(train, val, test):
    for name, df in [("train", train), ("val", val), ("test", test)]:
        df.to_csv(PROC_DIR / f"{name}.csv", index=False)
    print(f"[save]       train / val / test sauvegardés dans {PROC_DIR}")


def run_pipeline(scale=True, test_size=0.20, val_size=0.20, random_state=42):
    print("=" * 58)
    print("  PIPELINE DE PRÉTRAITEMENT — BRFSS 2015")
    print("=" * 58)
    df = load_data()
    df = validate(df)
    df = engineer_features(df)
    train, val, test = split_data(df, test_size, val_size, random_state)
    if scale:
        train, val, test = scale_continuous(train, val, test)
    save(train, val, test)
    print("=" * 58)
    print(f"  Features : {train.shape[1]-1}  |  Cible : {TARGET}")
    print("=" * 58)
    return train, val, test


if __name__ == "__main__":
    run_pipeline()
