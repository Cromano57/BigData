"""
preprocess_imbalanced.py
========================
Sprint 3 — Prétraitement du dataset non-équilibré BRFSS 2015.

Produit Data/Processed/imbalanced/train.csv, val.csv, test.csv
pour illustrer la gestion du déséquilibre de classes.

Dataset : diabetes_binary_health_indicators_BRFSS2015.csv
  - 253,680 lignes
  - ~13.9% de positifs (diabétiques/pré-diabétiques)
  - ratio 1:6.2 (minoritaire:majoritaire)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "Data" / "Raw" / "diabetes_binary_health_indicators_BRFSS2015.csv"
PROC_DIR = ROOT / "Data" / "Processed" / "imbalanced"
PROC_DIR.mkdir(parents=True, exist_ok=True)

TARGET       = "Diabetes_binary"
SCALE_COLS   = ["BMI", "MentHlth", "PhysHlth", "CardioRisk", "UnhealthyLifestyle"]
SEED         = 42


def run():
    print("=" * 55)
    print("  PRÉTRAITEMENT — DATASET NON-ÉQUILIBRÉ")
    print("=" * 55)

    df = pd.read_csv(RAW_PATH)
    print(f"[load]    {df.shape[0]:,} lignes · {df.shape[1]} colonnes")

    # Suppression des doublons
    n_dup = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[clean]   {n_dup:,} doublons supprimés → {len(df):,} lignes")

    # Feature engineering (identique au Sprint 1)
    df["Obese"]              = (df["BMI"] >= 30).astype(int)
    df["CardioRisk"]         = df["HighBP"] + df["HighChol"] + df["HeartDiseaseorAttack"] + df["Stroke"]
    df["UnhealthyLifestyle"] = (df["PhysActivity"] == 0).astype(int) + df["Smoker"] + df["HvyAlcoholConsump"]
    df["PoorHealth"]         = ((df["MentHlth"] > 14) | (df["PhysHlth"] > 14)).astype(int)
    print(f"[engineer] 4 features créées")

    # Distribution des classes
    pos = df[TARGET].sum()
    neg = len(df) - pos
    print(f"[balance]  Négatifs: {int(neg):,} ({neg/len(df)*100:.1f}%) | "
          f"Positifs: {int(pos):,} ({pos/len(df)*100:.1f}%) | "
          f"Ratio: 1:{neg/pos:.1f}")

    # Split stratifié
    train_val, test = train_test_split(df, test_size=0.20, random_state=SEED,
                                       stratify=df[TARGET])
    train, val = train_test_split(train_val, test_size=0.20, random_state=SEED,
                                  stratify=train_val[TARGET])
    print(f"[split]    Train {len(train):,} | Val {len(val):,} | Test {len(test):,}")

    # Normalisation
    scaler = StandardScaler()
    for split in [train, val, test]:
        split = split.copy()
    train_c, val_c, test_c = train.copy(), val.copy(), test.copy()
    train_c[SCALE_COLS] = scaler.fit_transform(train_c[SCALE_COLS])
    val_c[SCALE_COLS]   = scaler.transform(val_c[SCALE_COLS])
    test_c[SCALE_COLS]  = scaler.transform(test_c[SCALE_COLS])

    joblib.dump(scaler, PROC_DIR / "scaler.joblib")

    # Sauvegarde
    train_c.to_csv(PROC_DIR / "train.csv", index=False)
    val_c.to_csv(  PROC_DIR / "val.csv",   index=False)
    test_c.to_csv( PROC_DIR / "test.csv",  index=False)
    print(f"[save]     Données → {PROC_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    run()
