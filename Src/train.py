"""
train.py
========
Construction, entraînement et sauvegarde du réseau de neurones (Keras + MLflow).

Usage :
    python src/train.py                        # config par défaut
    python src/train.py --epochs 50 --lr 1e-3  # config personnalisée

Architecture :
    Input → Dense(64, ReLU) → Dropout(0.3)
          → Dense(32, ReLU) → Dropout(0.2)
          → Dense(1,  Sigmoid)

Sorties :
    models/nn_baseline.keras   modèle sauvegardé
    models/history.json        historique loss/accuracy
    mlruns/                    expériences MLflow
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ─── Chemins ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PROC_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "Diabetes_binary"

# ─── Reproductibilité ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
def load_splits():
    """Charge train / val / test depuis data/processed/."""
    train = pd.read_csv(PROC_DIR / "train.csv")
    val   = pd.read_csv(PROC_DIR / "val.csv")
    test  = pd.read_csv(PROC_DIR / "test.csv")

    X_train = train.drop(columns=[TARGET]).values.astype("float32")
    y_train = train[TARGET].values.astype("float32")
    X_val   = val.drop(columns=[TARGET]).values.astype("float32")
    y_val   = val[TARGET].values.astype("float32")
    X_test  = test.drop(columns=[TARGET]).values.astype("float32")
    y_test  = test[TARGET].values.astype("float32")

    print(f"[data]  Train {X_train.shape} · Val {X_val.shape} · Test {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model(
    input_dim: int,
    hidden_units: list  = [64, 32],
    dropout_rates: list = [0.3, 0.2],
    l2_lambda: float    = 1e-4,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Réseau de neurones binaire : Input → [Dense → Dropout] × N → Dense(1, sigmoid)

    Régularisation : L2 + Dropout sur chaque couche cachée.
    Optimiseur     : Adam avec lr configurable.
    Loss           : BinaryCrossentropy.
    """
    model = keras.Sequential(name="nn_baseline")

    # Couche d'entrée
    model.add(layers.Input(shape=(input_dim,), name="input"))

    # Couches cachées
    for i, (units, drop) in enumerate(zip(hidden_units, dropout_rates)):
        model.add(layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_lambda),
            name=f"dense_{i+1}",
        ))
        model.add(layers.BatchNormalization(name=f"bn_{i+1}"))
        model.add(layers.Dropout(drop, name=f"dropout_{i+1}"))

    # Couche de sortie
    model.add(layers.Dense(1, activation="sigmoid", name="output"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def train(
    epochs:        int   = 30,
    batch_size:    int   = 256,
    learning_rate: float = 1e-3,
    hidden_units:  list  = None,
    dropout_rates: list  = None,
    l2_lambda:     float = 1e-4,
    patience:      int   = 7,
    experiment_name: str = "diabetes_nn",
):
    """
    Entraîne le modèle avec suivi MLflow.

    Callbacks :
      - EarlyStopping   (val_loss, patience=patience, restore_best_weights)
      - ReduceLROnPlateau (val_loss, factor=0.5, patience=3)
    """
    if hidden_units  is None: hidden_units  = [64, 32]
    if dropout_rates is None: dropout_rates = [0.3, 0.2]

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    input_dim = X_train.shape[1]

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"nn_baseline_{int(time.time())}"):

        # Log des hyperparamètres
        params = {
            "epochs": epochs, "batch_size": batch_size,
            "learning_rate": learning_rate, "hidden_units": str(hidden_units),
            "dropout_rates": str(dropout_rates), "l2_lambda": l2_lambda,
            "patience": patience, "input_dim": input_dim,
            "optimizer": "Adam", "loss": "BinaryCrossentropy",
        }
        mlflow.log_params(params)
        mlflow.set_tag("framework", "TensorFlow/Keras")
        mlflow.set_tag("dataset",   "BRFSS 2015 50/50 split")

        # ── Modèle ────────────────────────────────────────────────────────────
        model = build_model(input_dim, hidden_units, dropout_rates, l2_lambda, learning_rate)
        model.summary()

        # ── Callbacks ─────────────────────────────────────────────────────────
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience,
                restore_best_weights=True, verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3,
                min_lr=1e-6, verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(MODEL_DIR / "nn_best.keras"),
                monitor="val_auc", save_best_only=True, verbose=0,
            ),
        ]

        # ── Entraînement ──────────────────────────────────────────────────────
        t0 = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        train_time = time.time() - t0
        print(f"\n[train]  Durée : {train_time:.1f}s · Epochs réels : {len(history.history['loss'])}")

        # ── Évaluation sur le test ─────────────────────────────────────────────
        test_results = model.evaluate(X_test, y_test, verbose=0)
        metric_names = ["test_" + m.name for m in model.metrics]
        test_metrics = dict(zip(metric_names, test_results))

        print("\n[eval]   Métriques sur le jeu de test :")
        for k, v in test_metrics.items():
            print(f"         {k:25s} = {v:.4f}")

        # ── Log des métriques dans MLflow ────────────────────────────────────
        mlflow.log_metrics(test_metrics)
        mlflow.log_metric("train_time_s", round(train_time, 2))

        # Log de chaque epoch
        for epoch_i, (loss, val_loss, acc, val_acc) in enumerate(zip(
            history.history["loss"],        history.history["val_loss"],
            history.history["accuracy"],    history.history["val_accuracy"],
        )):
            mlflow.log_metrics({
                "epoch_train_loss": loss, "epoch_val_loss": val_loss,
                "epoch_train_acc": acc,  "epoch_val_acc": val_acc,
            }, step=epoch_i)

        # ── Sauvegarde du modèle ───────────────────────────────────────────────
        model_path = MODEL_DIR / "nn_baseline.keras"
        model.save(model_path)
        mlflow.keras.log_model(model, artifact_path="model")
        print(f"\n[save]   Modèle sauvegardé → {model_path}")

        # ── Sauvegarde de l'historique ─────────────────────────────────────────
        hist_path = MODEL_DIR / "history.json"
        with open(hist_path, "w") as f:
            json.dump({k: [float(v) for v in vals]
                       for k, vals in history.history.items()}, f, indent=2)
        mlflow.log_artifact(str(hist_path))
        print(f"[save]   Historique → {hist_path}")

        run_id = mlflow.active_run().info.run_id
        print(f"\n[mlflow] Run ID : {run_id}")
        print(f"[mlflow] UI     : python -m mlflow ui --backend-store-uri sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}")

    return model, history, test_metrics


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement du réseau de neurones")
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--batch",     type=int,   default=256)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--l2",        type=float, default=1e-4)
    parser.add_argument("--patience",  type=int,   default=7)
    parser.add_argument("--exp",       type=str,   default="diabetes_nn")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        l2_lambda=args.l2,
        patience=args.patience,
        experiment_name=args.exp,
    )
