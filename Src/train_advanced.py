"""
train_advanced.py
=================
Sprint 3 — Architecture enrichie + Gestion déséquilibre + MLOps + Carbon tracking.

Architectures comparées :
  1. Baseline (64-32)       : réplique Sprint 2
  2. Deep     (256-128-64)  : réseau plus profond et plus large
  3. Residual (256-128-64)  : connexions résiduelles (skip connections)

Gestion du déséquilibre des classes :
  - class_weight auto-calculé (inverse de la fréquence)
  - SMOTE oversampling (via imbalanced-learn)

Tracking MLflow avec registre de modèles.
Carbon footprint via CodeCarbon.

Usage :
    python Src/train_advanced.py
    python Src/train_advanced.py --arch all --imbalance both
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, classification_report
)
from imblearn.over_sampling import SMOTE
from codecarbon import EmissionsTracker

warnings.filterwarnings("ignore")

# ─── Chemins ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PROC_DIR  = ROOT / "Data" / "Processed"
RAW_DIR   = ROOT / "Data" / "Raw"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "Diabetes_binary"
SEED   = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# Chargement des données
# ══════════════════════════════════════════════════════════════════════════════

def load_splits(proc_dir: Path = PROC_DIR):
    """Charge train / val / test depuis Data/Processed/."""
    train = pd.read_csv(proc_dir / "train.csv")
    val   = pd.read_csv(proc_dir / "val.csv")
    test  = pd.read_csv(proc_dir / "test.csv")

    X_train = train.drop(columns=[TARGET]).values.astype("float32")
    y_train = train[TARGET].values.astype("float32")
    X_val   = val.drop(columns=[TARGET]).values.astype("float32")
    y_val   = val[TARGET].values.astype("float32")
    X_test  = test.drop(columns=[TARGET]).values.astype("float32")
    y_test  = test[TARGET].values.astype("float32")

    print(f"[data]  Train {X_train.shape} · Val {X_val.shape} · Test {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 1. Architectures
# ══════════════════════════════════════════════════════════════════════════════

def build_baseline(input_dim: int, l2: float = 1e-4, lr: float = 1e-3) -> keras.Model:
    """
    Architecture Sprint 2 (référence).
    Input → Dense(64) → BN → Drop(0.3) → Dense(32) → BN → Drop(0.2) → Sigmoid
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name="input"),
        layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(l2), name="dense_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.Dropout(0.3, name="drop_1"),
        layers.Dense(32, activation="relu",
                     kernel_regularizer=regularizers.l2(l2), name="dense_2"),
        layers.BatchNormalization(name="bn_2"),
        layers.Dropout(0.2, name="drop_2"),
        layers.Dense(1, activation="sigmoid", name="output"),
    ], name="baseline_64_32")
    _compile(model, lr)
    return model


def build_deep(input_dim: int, l2: float = 1e-4, lr: float = 1e-3) -> keras.Model:
    """
    Architecture enrichie — plus large et plus profonde.
    Input → Dense(256) → BN → Drop(0.4)
          → Dense(128) → BN → Drop(0.3)
          → Dense(64)  → BN → Drop(0.2)
          → Sigmoid
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name="input"),
        layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(l2), name="dense_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.Dropout(0.4, name="drop_1"),
        layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(l2), name="dense_2"),
        layers.BatchNormalization(name="bn_2"),
        layers.Dropout(0.3, name="drop_2"),
        layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(l2), name="dense_3"),
        layers.BatchNormalization(name="bn_3"),
        layers.Dropout(0.2, name="drop_3"),
        layers.Dense(1, activation="sigmoid", name="output"),
    ], name="deep_256_128_64")
    _compile(model, lr)
    return model


def build_residual(input_dim: int, l2: float = 1e-4, lr: float = 1e-3) -> keras.Model:
    """
    Architecture résiduelle (skip connections).
    Les connexions résiduelles permettent au gradient de circuler plus facilement.
    Adapté au traitement de données tabulaires avec des features corrélées.
    """
    inputs = layers.Input(shape=(input_dim,), name="input")

    # Projection initiale vers l'espace latent 128
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(l2), name="proj")(inputs)
    x = layers.BatchNormalization(name="bn_proj")(x)

    # Bloc résiduel 1 : 128 → 128
    skip = x
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(l2), name="res1_fc1")(x)
    x = layers.BatchNormalization(name="res1_bn1")(x)
    x = layers.Dropout(0.3, name="res1_drop1")(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(l2), name="res1_fc2")(x)
    x = layers.BatchNormalization(name="res1_bn2")(x)
    x = layers.Add(name="res1_add")([x, skip])
    x = layers.Activation("relu", name="res1_relu")(x)

    # Bloc résiduel 2 : 128 → 64
    skip2 = layers.Dense(64, name="res2_proj")(x)   # projection pour aligner les dims
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(l2), name="res2_fc1")(x)
    x = layers.BatchNormalization(name="res2_bn1")(x)
    x = layers.Dropout(0.2, name="res2_drop1")(x)
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(l2), name="res2_fc2")(x)
    x = layers.BatchNormalization(name="res2_bn2")(x)
    x = layers.Add(name="res2_add")([x, skip2])
    x = layers.Activation("relu", name="res2_relu")(x)

    # Sortie
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="residual_128_64")
    _compile(model, lr)
    return model


def _compile(model: keras.Model, lr: float):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. Gestion du déséquilibre
# ══════════════════════════════════════════════════════════════════════════════

def compute_weights(y: np.ndarray) -> dict:
    """Calcule les poids de classes inversement proportionnels à leur fréquence."""
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    cw = dict(zip(classes.astype(int), weights))
    print(f"[weights]  class_weight = {cw}")
    return cw


def apply_smote(X: np.ndarray, y: np.ndarray, seed: int = SEED):
    """Rééchantillonnage SMOTE pour équilibrer les classes minoritaires."""
    print(f"[smote]    Avant : {int(y.sum())} positifs / {len(y)} total "
          f"({y.mean()*100:.1f}%)")
    sm = SMOTE(random_state=seed)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"[smote]    Après : {int(y_res.sum())} positifs / {len(y_res)} total "
          f"({y_res.mean()*100:.1f}%)")
    return X_res.astype("float32"), y_res.astype("float32")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Entraînement avec MLflow
# ══════════════════════════════════════════════════════════════════════════════

def get_callbacks(model_name: str, patience: int = 7) -> list:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=0,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / f"{model_name}_best.keras"),
            monitor="val_auc", save_best_only=True, verbose=0,
        ),
    ]


def evaluate_model(model, X_test, y_test, threshold: float = 0.45):
    """Calcule les métriques de classification sur le jeu de test."""
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "auc_roc":   roc_auc_score(y_test, y_prob),
        "f1":        f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
    }


def train_model(
    build_fn,
    arch_name:  str,
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    epochs:     int   = 50,
    batch_size: int   = 256,
    lr:         float = 1e-3,
    l2:         float = 1e-4,
    patience:   int   = 10,
    class_weight: dict = None,
    track_carbon: bool = True,
    experiment_name: str = "diabetes_sprint3",
):
    """Entraîne un modèle et logue les résultats dans MLflow."""
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}")
    mlflow.set_experiment(experiment_name)

    input_dim = X_train.shape[1]
    model = build_fn(input_dim, l2, lr)

    # Carbon tracking
    tracker = None
    if track_carbon:
        tracker = EmissionsTracker(
            project_name=f"diabetes_{arch_name}",
            output_dir=str(ROOT / "Reports"),
            output_file="emissions.csv",
            log_level="error",
        )
        tracker.start()

    with mlflow.start_run(run_name=arch_name):
        mlflow.log_params({
            "architecture": arch_name,
            "input_dim": input_dim,
            "epochs_max": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "l2_lambda": l2,
            "patience": patience,
            "class_weight": str(class_weight) if class_weight else "None",
            "smote_applied": "yes" if X_train.shape[0] > 44196 else "no",
        })
        mlflow.set_tags({
            "framework": "TensorFlow/Keras",
            "sprint": "3",
            "dataset": "BRFSS 2015",
        })

        t0 = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=get_callbacks(arch_name, patience),
            class_weight=class_weight,
            verbose=0,
        )
        train_time = time.time() - t0
        n_epochs = len(history.history["loss"])

        # Carbon tracking stop
        if tracker is not None:
            emissions = tracker.stop()
            co2_kg = emissions if emissions else 0.0
            mlflow.log_metric("co2_kg", co2_kg)
        else:
            co2_kg = 0.0

        # Métriques test
        metrics = evaluate_model(model, X_test, y_test)
        metrics["train_time_s"] = round(train_time, 2)
        metrics["n_epochs"] = n_epochs
        metrics["co2_kg"] = co2_kg

        mlflow.log_metrics(metrics)

        # Log courbes
        for i, (loss, val_loss) in enumerate(zip(
            history.history["loss"], history.history["val_loss"]
        )):
            mlflow.log_metrics({"epoch_train_loss": loss, "epoch_val_loss": val_loss}, step=i)

        # Sauvegarde
        model.save(MODEL_DIR / f"{arch_name}.keras")
        history_path = MODEL_DIR / f"history_{arch_name}.json"
        with open(history_path, "w") as f:
            json.dump({k: [float(v) for v in vals]
                       for k, vals in history.history.items()}, f, indent=2)
        mlflow.log_artifact(str(history_path))

        run_id = mlflow.active_run().info.run_id
        print(f"  [{arch_name}]  AUC={metrics['auc_roc']:.4f} | "
              f"F1={metrics['f1']:.4f} | "
              f"Recall={metrics['recall']:.4f} | "
              f"Epochs={n_epochs} | "
              f"CO2={co2_kg*1e6:.2f} µgCO2eq | "
              f"RunID={run_id[:8]}…")

    return model, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def run(arch: str = "all", imbalance_strategy: str = "none", epochs: int = 50):
    """
    Lance la comparaison d'architectures et de stratégies de déséquilibre.

    arch              : "baseline" | "deep" | "residual" | "all"
    imbalance_strategy: "none" | "weights" | "smote" | "both"
    """
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()

    builders = {
        "baseline": build_baseline,
        "deep":     build_deep,
        "residual": build_residual,
    }
    if arch == "all":
        selected = builders
    else:
        selected = {arch: builders[arch]}

    results = {}

    print("\n" + "="*65)
    print("  SPRINT 3 — COMPARAISON D'ARCHITECTURES")
    print("="*65)

    # ── Architectures sur données équilibrées (Strategy: none) ────────────────
    if imbalance_strategy in ("none", "both"):
        print("\n[Phase 1] Données équilibrées (50/50) — pas de correction")
        for name, fn in selected.items():
            _, m = train_model(fn, name, X_train, y_train,
                               X_val, y_val, X_test, y_test,
                               epochs=epochs,
                               experiment_name="diabetes_sprint3_balanced")
            results[name] = m

    # ── Class weights ─────────────────────────────────────────────────────────
    if imbalance_strategy in ("weights", "both"):
        print("\n[Phase 2] Class weights (données équilibrées, poids inversés)")
        cw = compute_weights(y_train)
        for name, fn in selected.items():
            _, m = train_model(fn, f"{name}_weighted", X_train, y_train,
                               X_val, y_val, X_test, y_test,
                               epochs=epochs,
                               class_weight=cw,
                               experiment_name="diabetes_sprint3_weighted")
            results[f"{name}_weighted"] = m

    # ── SMOTE ─────────────────────────────────────────────────────────────────
    if imbalance_strategy in ("smote", "both"):
        print("\n[Phase 3] SMOTE oversampling")
        X_sm, y_sm = apply_smote(X_train, y_train)
        for name, fn in selected.items():
            _, m = train_model(fn, f"{name}_smote", X_sm, y_sm,
                               X_val, y_val, X_test, y_test,
                               epochs=epochs,
                               experiment_name="diabetes_sprint3_smote")
            results[f"{name}_smote"] = m

    # ── Résumé ────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  RÉSUMÉ DES RÉSULTATS")
    print("="*65)
    print(f"  {'Modèle':<30} {'AUC':>6} {'F1':>6} {'Recall':>8} {'CO2 µg':>10}")
    print("  " + "-"*60)
    for name, m in results.items():
        print(f"  {name:<30} {m['auc_roc']:>6.4f} {m['f1']:>6.4f} "
              f"{m['recall']:>8.4f} {m.get('co2_kg',0)*1e6:>10.2f}")

    # Sauvegarde du résumé
    summary_path = ROOT / "Reports" / "sprint3_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Résultats → {summary_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",      default="all",
                        choices=["baseline", "deep", "residual", "all"])
    parser.add_argument("--imbalance", default="none",
                        choices=["none", "weights", "smote", "both"])
    parser.add_argument("--epochs",    type=int, default=50)
    args = parser.parse_args()

    run(arch=args.arch, imbalance_strategy=args.imbalance, epochs=args.epochs)
