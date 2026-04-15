"""
explainability.py
=================
Sprint 3 — IA Explicable : SHAP + LIME + analyse de durabilité.

Analyses produites :
  1. SHAP Summary Plot (importance globale des features)
  2. SHAP Beeswarm Plot (distribution des SHAP values)
  3. SHAP Waterfall Plot (explication d'un individu)
  4. SHAP Dependence Plots (interactions entre features)
  5. LIME — explication locale d'une prédiction
  6. Carbon footprint summary (via emissions.csv si disponible)

Sorties (dans Reports/) :
  shap_summary.png
  shap_beeswarm.png
  shap_waterfall_pos.png
  shap_waterfall_neg.png
  shap_dependence_GenHlth.png
  shap_dependence_BMI.png
  lime_positive.png
  lime_negative.png

Usage :
    python Src/explainability.py
    python Src/explainability.py --model models/residual_128_64.keras --n_shap 500
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import tensorflow as tf
from tensorflow import keras
import joblib

warnings.filterwarnings("ignore")

# ─── Chemins ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PROC_DIR  = ROOT / "Data" / "Processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "Reports"
REPORT_DIR.mkdir(exist_ok=True)

TARGET = "Diabetes_binary"

# Noms lisibles des features (pour les graphiques)
FEATURE_NAMES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDisease", "PhysActivity", "Fruits", "Veggies", "HvyAlcohol",
    "AnyHealthcare", "NoDocCost", "GenHlth", "MentHlth", "PhysHlth",
    "DiffWalk", "Sex", "Age", "Education", "Income",
    "Obese", "CardioRisk", "UnhealthyLifestyle", "PoorHealth",
]


def load_data(proc_dir: Path = PROC_DIR):
    """Charge les données test et train."""
    train = pd.read_csv(proc_dir / "train.csv")
    test  = pd.read_csv(proc_dir / "test.csv")

    X_train = train.drop(columns=[TARGET]).values.astype("float32")
    y_train = train[TARGET].values

    X_test  = test.drop(columns=[TARGET]).values.astype("float32")
    y_test  = test[TARGET].values

    feature_names = [c for c in train.columns if c != TARGET]
    print(f"[data]  Test {X_test.shape} · Features: {len(feature_names)}")
    return X_train, y_train, X_test, y_test, feature_names


def load_model(model_path: Path):
    """Charge le modèle Keras."""
    model = keras.models.load_model(str(model_path))
    print(f"[model] Chargé : {model_path.name}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# SHAP — Analyse globale et locale
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap(model, X_train, X_test, n_background: int = 100, n_explain: int = 300):
    """
    Calcule les valeurs SHAP avec un explainer DeepExplainer (Keras).

    - n_background : échantillon d'arrière-plan (résumé de la distribution d'entrée)
    - n_explain    : nombre de samples sur lesquels calculer les SHAP values
    """
    print(f"\n[SHAP]  Calcul des SHAP values "
          f"(background={n_background}, explain={n_explain})…")

    # Échantillon d'arrière-plan : résumé de X_train
    np.random.seed(42)
    bg_idx = np.random.choice(len(X_train), n_background, replace=False)
    background = X_train[bg_idx]

    # Explainer
    explainer = shap.DeepExplainer(model, background)

    # Données à expliquer
    idx = np.random.choice(len(X_test), n_explain, replace=False)
    X_explain = X_test[idx]
    y_explain = None

    shap_values = explainer.shap_values(X_explain)

    # shap_values peut être une liste [neg, pos] ou un array 3D
    if isinstance(shap_values, list):
        sv = shap_values[0]  # valeurs pour la classe positive
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 0]
    else:
        sv = shap_values

    print(f"[SHAP]  Shape des SHAP values : {sv.shape}")
    return explainer, sv, X_explain, idx


def plot_shap_summary(sv, X_explain, feature_names, out_dir: Path):
    """Bar plot de l'importance globale (valeur absolue moyenne)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X_explain, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=20)
    plt.title("SHAP — Importance globale des features\n(valeur absolue moyenne)",
              fontsize=13, pad=12)
    plt.tight_layout()
    path = out_dir / "shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP]  → {path.name}")


def plot_shap_beeswarm(sv, X_explain, feature_names, out_dir: Path):
    """Beeswarm plot : distribution des SHAP values par feature."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_explain, feature_names=feature_names,
                      plot_type="violin", show=False, max_display=20)
    plt.title("SHAP — Distribution des valeurs d'impact\n(beeswarm / violin)",
              fontsize=13, pad=12)
    plt.tight_layout()
    path = out_dir / "shap_beeswarm.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP]  → {path.name}")


def plot_shap_waterfall(explainer, sv, X_explain, y_test, idx_in_test,
                        feature_names, out_dir: Path, which: str = "positive"):
    """
    Waterfall plot pour un individu donné.
    which : "positive" (diabétique) ou "negative" (non-diabétique)
    """
    # Reconstruit l'objet Explanation SHAP
    ev = explainer.expected_value
    if hasattr(ev, '__len__'):
        base_val = float(np.array(ev).ravel()[0])
    else:
        try:
            base_val = float(ev)
        except Exception:
            base_val = float(np.array(ev))

    # Trouve un exemple du bon type
    y_pred_prob = explainer.model.predict(X_explain).ravel()
    if which == "positive":
        candidates = np.where(y_pred_prob >= 0.5)[0]
    else:
        candidates = np.where(y_pred_prob < 0.5)[0]

    if len(candidates) == 0:
        print(f"[SHAP]  Waterfall {which}: aucun exemple trouvé")
        return

    sample_i = candidates[0]
    explanation = shap.Explanation(
        values=sv[sample_i],
        base_values=base_val,
        data=X_explain[sample_i],
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.waterfall_plot(explanation, show=False, max_display=15)
    label = "Cas Positif (Diabétique)" if which == "positive" else "Cas Négatif (Non-diabétique)"
    plt.title(f"SHAP Waterfall — {label}", fontsize=12, pad=10)
    plt.tight_layout()
    path = out_dir / f"shap_waterfall_{which}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP]  → {path.name}")


def plot_shap_dependence(sv, X_explain, feature_names, out_dir: Path,
                         feature: str = "GenHlth"):
    """Dependence plot : relation entre une feature et ses SHAP values."""
    if feature not in feature_names:
        print(f"[SHAP]  Feature '{feature}' introuvable")
        return
    feat_idx = feature_names.index(feature)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(feat_idx, sv, X_explain,
                         feature_names=feature_names,
                         interaction_index="auto",
                         ax=ax, show=False)
    ax.set_title(f"SHAP Dependence — {feature}", fontsize=12)
    plt.tight_layout()
    path = out_dir / f"shap_dependence_{feature}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP]  → {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# LIME — Explications locales
# ══════════════════════════════════════════════════════════════════════════════

def explain_with_lime(model, X_train, X_test, y_test, feature_names, out_dir: Path,
                      n_features: int = 15, n_samples: int = 500):
    """
    Génère des explications LIME pour deux individus :
      - un vrai positif (prédit diabétique)
      - un vrai négatif (prédit non-diabétique)
    """
    print("\n[LIME]  Création de l'explainer…")

    # Predict function : retourne proba pour les 2 classes
    def predict_fn(X):
        proba_pos = model.predict(X.astype("float32"), verbose=0).ravel()
        return np.column_stack([1 - proba_pos, proba_pos])

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["Non-diabétique", "Diabétique"],
        mode="classification",
        random_state=42,
    )

    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred      = (y_pred_prob >= 0.45).astype(int)

    # Vrai positif (y=1, pred=1)
    tp_idx = np.where((y_test == 1) & (y_pred == 1))[0]
    # Vrai négatif (y=0, pred=0)
    tn_idx = np.where((y_test == 0) & (y_pred == 0))[0]

    for label, idx_arr, fname in [
        ("positive", tp_idx, "lime_positive.png"),
        ("negative", tn_idx, "lime_negative.png"),
    ]:
        if len(idx_arr) == 0:
            print(f"[LIME]  Aucun exemple {label}")
            continue

        i = idx_arr[0]
        exp = explainer.explain_instance(
            X_test[i],
            predict_fn,
            num_features=n_features,
            num_samples=n_samples,
        )

        fig = exp.as_pyplot_figure()
        lbl = "Cas Positif (Diabétique)" if label == "positive" else "Cas Négatif (Non-diabétique)"
        fig.suptitle(f"LIME — Explication locale : {lbl}\n"
                     f"P(diabète)={y_pred_prob[i]:.3f}", fontsize=11)
        fig.set_size_inches(10, 6)
        plt.tight_layout()
        path = out_dir / fname
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[LIME]  → {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Carbon footprint summary
# ══════════════════════════════════════════════════════════════════════════════

def print_carbon_summary(report_dir: Path):
    """Affiche un résumé des émissions de CO2 si le fichier emissions.csv existe."""
    csv_path = report_dir / "emissions.csv"
    if not csv_path.exists():
        print("\n[Carbon] Aucun fichier emissions.csv trouvé.")
        return

    df = pd.read_csv(csv_path)
    print("\n" + "="*55)
    print("  EMPREINTE CARBONE DES ENTRAÎNEMENTS")
    print("="*55)
    print(f"  Runs mesurés     : {len(df)}")
    print(f"  CO2 total (kg)   : {df['emissions'].sum():.6f}")
    print(f"  CO2 total (mg)   : {df['emissions'].sum()*1e6:.2f}")
    print(f"  Durée totale (s) : {df['duration'].sum():.1f}")
    print(f"  Énergie (kWh)    : {df['energy_consumed'].sum():.6f}")
    print(f"  Pays             : {df['country_name'].iloc[0] if 'country_name' in df.columns else 'N/A'}")
    print("="*55)

    # Comparaison équivalents
    co2_total_g = df["emissions"].sum() * 1000
    print(f"\n  Équivalences :")
    print(f"    = {co2_total_g:.3f} g CO2")
    print(f"    = {co2_total_g / 209:.4f} km en voiture (209g/km)")
    print(f"    = {co2_total_g / 0.185:.2f}s de streaming HD (0.185g/s)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def run(model_path: Path = None, n_shap: int = 300, n_background: int = 100):
    """Lance toutes les analyses d'explicabilité."""

    # Recherche du meilleur modèle disponible
    if model_path is None:
        for candidate in ["residual_128_64_best.keras", "deep_256_128_64_best.keras",
                          "nn_best.keras", "baseline_best.keras"]:
            p = MODEL_DIR / candidate
            if p.exists():
                model_path = p
                break

    if model_path is None or not model_path.exists():
        print(f"[ERROR] Aucun modèle trouvé dans {MODEL_DIR}")
        print("        Lancez d'abord : python Src/train_advanced.py")
        return

    print(f"\n{'='*55}")
    print(f"  XAI — SHAP + LIME")
    print(f"  Modèle : {model_path.name}")
    print(f"{'='*55}")

    X_train, y_train, X_test, y_test, feature_names = load_data()
    model = load_model(model_path)

    # ── SHAP ─────────────────────────────────────────────────────────────────
    explainer, sv, X_explain, idx = compute_shap(
        model, X_train, X_test,
        n_background=n_background, n_explain=n_shap
    )

    print("\n[SHAP]  Génération des graphiques…")
    plot_shap_summary(sv, X_explain, feature_names, REPORT_DIR)
    plot_shap_beeswarm(sv, X_explain, feature_names, REPORT_DIR)
    plot_shap_waterfall(explainer, sv, X_explain, y_test[idx],
                        idx, feature_names, REPORT_DIR, "positive")
    plot_shap_waterfall(explainer, sv, X_explain, y_test[idx],
                        idx, feature_names, REPORT_DIR, "negative")

    # Dependence plots pour les 2 features les plus importantes
    mean_abs = np.abs(sv).mean(axis=0)
    top2_idx = np.argsort(mean_abs)[::-1][:2]
    for fi in top2_idx:
        plot_shap_dependence(sv, X_explain, feature_names, REPORT_DIR, feature_names[fi])

    # ── LIME ──────────────────────────────────────────────────────────────────
    explain_with_lime(model, X_train, X_test, y_test, feature_names, REPORT_DIR)

    # ── Carbon summary ────────────────────────────────────────────────────────
    print_carbon_summary(REPORT_DIR)

    print(f"\n[done]  Tous les graphiques XAI sauvegardés dans {REPORT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default=None,
                        help="Chemin vers le modèle Keras")
    parser.add_argument("--n_shap",     type=int, default=300,
                        help="Nb de samples pour SHAP")
    parser.add_argument("--n_background", type=int, default=100,
                        help="Nb de samples pour le background SHAP")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else None
    run(model_path=model_path, n_shap=args.n_shap, n_background=args.n_background)
