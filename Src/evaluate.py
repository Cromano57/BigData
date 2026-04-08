"""
evaluate.py
===========
Évaluation complète du modèle entraîné :
  - Métriques : Accuracy, AUC-ROC, F1, Précision, Rappel
  - Analyse du seuil de décision (courbe ROC + Precision-Recall)
  - Matrice de confusion
  - Courbes loss / accuracy

Usage :
    python src/evaluate.py                          # évalue nn_baseline.keras sur test.csv
    python src/evaluate.py --threshold 0.4          # seuil personnalisé
    python src/evaluate.py --model models/nn_best.keras
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, f1_score,
)
import tensorflow as tf
from tensorflow import keras

# ─── Chemins ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PROC_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

TARGET = "Diabetes_binary"

# ─── Style ────────────────────────────────────────────────────────────────────
PALETTE = ["#4C72B0", "#DD8452"]
BG      = "#F8F9FA"
DARK    = "#1a1a2e"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#cccccc", "axes.titleweight": "bold",
    "axes.titlesize": 12, "figure.dpi": 130,
})


# ══════════════════════════════════════════════════════════════════════════════
def load_model_and_data(model_path: Path):
    model   = keras.models.load_model(model_path)
    test    = pd.read_csv(PROC_DIR / "test.csv")
    X_test  = test.drop(columns=[TARGET]).values.astype("float32")
    y_test  = test[TARGET].values.astype("float32")

    train   = pd.read_csv(PROC_DIR / "train.csv")
    val     = pd.read_csv(PROC_DIR / "val.csv")

    print(f"[load]  Modèle : {model_path.name}  |  Test : {X_test.shape}")
    return model, X_test, y_test, train, val


def find_best_threshold(y_true, y_prob):
    """Cherche le seuil maximisant le F1-score."""
    thresholds = np.arange(0.1, 0.91, 0.01)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    best_t = thresholds[np.argmax(f1s)]
    return best_t, max(f1s)


# ── Figure 1 : Courbes d'entraînement ─────────────────────────────────────────
def plot_training_curves(history_path: Path, out_dir: Path):
    if not history_path.exists():
        print("[curves]  history.json introuvable — figure ignorée")
        return

    with open(history_path) as f:
        h = json.load(f)

    epochs = range(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.suptitle("Courbes d'entraînement", fontsize=14, fontweight="bold", color=DARK)

    # Loss
    axes[0].plot(epochs, h["loss"],     color=PALETTE[0], label="Train loss", linewidth=2)
    axes[0].plot(epochs, h["val_loss"], color=PALETTE[1], label="Val loss",   linewidth=2, linestyle="--")
    axes[0].set_title("Loss (BinaryCrossentropy)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.4)

    # Accuracy
    axes[1].plot(epochs, h["accuracy"],     color=PALETTE[0], label="Train acc", linewidth=2)
    axes[1].plot(epochs, h["val_accuracy"], color=PALETTE[1], label="Val acc",   linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    path = out_dir / "eval_1_training_curves.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"[fig 1]  Courbes entraînement → {path.name}")


# ── Figure 2 : ROC + Precision-Recall ─────────────────────────────────────────
def plot_roc_pr(y_test, y_prob, threshold, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig.suptitle("Analyse du seuil de décision", fontsize=14, fontweight="bold", color=DARK)

    # ROC
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=PALETTE[0], linewidth=2.5,
                 label=f"AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    # Marquer le seuil choisi
    idx = np.argmin(np.abs(roc_thresholds - threshold))
    axes[0].scatter(fpr[idx], tpr[idx], color=PALETTE[1], s=120, zorder=5,
                    label=f"Seuil = {threshold:.2f}")
    axes[0].set_title("Courbe ROC")
    axes[0].set_xlabel("Taux de faux positifs (FPR)")
    axes[0].set_ylabel("Taux de vrais positifs (TPR)")
    axes[0].legend(); axes[0].grid(True, alpha=0.4)

    # Precision-Recall
    prec, rec, pr_thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(rec, prec)
    axes[1].plot(rec, prec, color=PALETTE[0], linewidth=2.5,
                 label=f"PR-AUC = {pr_auc:.4f}")
    # Marquer le seuil choisi
    if len(pr_thresholds) > 0:
        idx_pr = np.argmin(np.abs(pr_thresholds - threshold))
        axes[1].scatter(rec[idx_pr], prec[idx_pr], color=PALETTE[1], s=120, zorder=5,
                        label=f"Seuil = {threshold:.2f}")
    axes[1].set_title("Courbe Precision-Recall")
    axes[1].set_xlabel("Rappel (Recall)")
    axes[1].set_ylabel("Précision")
    axes[1].legend(); axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    path = out_dir / "eval_2_roc_pr.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"[fig 2]  ROC + PR → {path.name}")
    return roc_auc, pr_auc


# ── Figure 3 : Matrice de confusion ───────────────────────────────────────────
def plot_confusion(y_test, y_pred, threshold, out_dir: Path):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.suptitle(f"Matrice de confusion (seuil = {threshold:.2f})",
                 fontsize=14, fontweight="bold", color=DARK)

    # Matrice absolue
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Prédit 0", "Prédit 1"],
                yticklabels=["Réel 0", "Réel 1"],
                linewidths=0.5, cbar=False)
    axes[0].set_title("Valeurs absolues")

    # Matrice normalisée
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", ax=axes[1],
                xticklabels=["Prédit 0", "Prédit 1"],
                yticklabels=["Réel 0", "Réel 1"],
                linewidths=0.5, cbar=False)
    axes[1].set_title("Normalisée par classe réelle")

    plt.tight_layout()
    path = out_dir / "eval_3_confusion.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"[fig 3]  Confusion → {path.name}")
    return tn, fp, fn, tp


# ── Figure 4 : F1 vs seuil ────────────────────────────────────────────────────
def plot_threshold_analysis(y_test, y_prob, best_t, out_dir: Path):
    thresholds = np.arange(0.05, 0.96, 0.01)
    f1s   = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
    accs  = [(y_test == (y_prob >= t).astype(int)).mean() for t in thresholds]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.plot(thresholds, f1s,  color=PALETTE[0], linewidth=2.5, label="F1-score")
    ax.plot(thresholds, accs, color=PALETTE[1], linewidth=2,   label="Accuracy", linestyle="--")
    ax.axvline(best_t, color="gray", linestyle=":", linewidth=1.5,
               label=f"Meilleur seuil F1 = {best_t:.2f}")
    ax.axvline(0.5,    color="silver", linestyle="--", linewidth=1, label="Seuil 0.5")
    ax.set_title("F1-score et Accuracy selon le seuil de décision",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Seuil"); ax.set_ylabel("Score")
    ax.set_xlim(0.05, 0.95); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    path = out_dir / "eval_4_threshold.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"[fig 4]  Analyse seuil → {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model_path: Path = None, threshold: float = None):
    if model_path is None:
        model_path = MODEL_DIR / "nn_baseline.keras"
        if not model_path.exists():
            model_path = MODEL_DIR / "nn_best.keras"

    model, X_test, y_test, train_df, val_df = load_model_and_data(model_path)
    y_prob = model.predict(X_test, verbose=0).flatten()

    # Seuil optimal si non spécifié
    best_t, best_f1 = find_best_threshold(y_test, y_prob)
    if threshold is None:
        threshold = best_t
        print(f"[thresh]  Meilleur seuil (max F1) = {threshold:.2f}  (F1 = {best_f1:.4f})")
    else:
        print(f"[thresh]  Seuil utilisateur = {threshold:.2f}")

    y_pred = (y_prob >= threshold).astype(int)

    # ── Rapport de classification ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RAPPORT D'ÉVALUATION")
    print("=" * 55)
    print(classification_report(y_test, y_pred,
          target_names=["Non-diabétique (0)", "Diabétique (1)"]))

    # ── Figures ───────────────────────────────────────────────────────────────
    hist_path = MODEL_DIR / "history.json"
    plot_training_curves(hist_path, REPORT_DIR)
    roc_auc, pr_auc = plot_roc_pr(y_test, y_prob, threshold, REPORT_DIR)
    tn, fp, fn, tp  = plot_confusion(y_test, y_pred, threshold, REPORT_DIR)
    plot_threshold_analysis(y_test, y_prob, best_t, REPORT_DIR)

    print(f"\n[résumé]  AUC-ROC = {roc_auc:.4f}  |  PR-AUC = {pr_auc:.4f}")
    print(f"[résumé]  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"[résumé]  Figures sauvegardées dans {REPORT_DIR}")
    print("=" * 55)

    return {"roc_auc": roc_auc, "pr_auc": pr_auc,
            "threshold": threshold, "best_f1": best_f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation du modèle entraîné")
    parser.add_argument("--model",     type=str,   default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else None
    evaluate(model_path=model_path, threshold=args.threshold)
