# Projet BigData - Prédiction du Diabète - GROUPE 3
**FISE A4 INFO - Nancy**
**ROMANO Corentin - RIECKENBERG Bruno - BOUDEMAGH Isrâ - FRAOILI Enzo**

Prédiction du diabète à partir des données de santé BRFSS 2015 (CDC).  
Classification binaire : déterminer si un individu est diabétique ou non à partir de 21 indicateurs de santé déclaratifs.

---

## Dataset

**Source :** [CDC Diabetes Health Indicators — Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)  
**Origine :** Behavioral Risk Factor Surveillance System (BRFSS) 2015

| Fichier | Lignes | Description |
|---|---|---|
| `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` | 70 692 | Équilibré 50/50 — utilisé Sprint 1 & 2 |
| `diabetes_binary_health_indicators_BRFSS2015.csv` | 253 680 | Déséquilibre réel (~86/14%) — utilisé Sprint 3 |

**Variable cible :** `Diabetes_binary` (0 = non-diabétique, 1 = diabétique/pré-diabétique)

---

## Structure du projet

```
BigData/
├── data/
│   ├── raw/
│   │   ├── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
│   │   └── diabetes_binary_health_indicators_BRFSS2015.csv
│   └── processed/
│       ├── train.csv          # Sprint 1 & 2 (64%)
│       ├── val.csv            # Sprint 1 & 2 (16%)
│       ├── test.csv           # Sprint 1 & 2 (20%)
│       ├── scaler.joblib      # StandardScaler Sprint 1 & 2
│       ├── train_v2.csv       # Sprint 3 + SMOTE (64%)
│       ├── val_v2.csv         # Sprint 3 (16%)
│       ├── test_v2.csv        # Sprint 3 (20%)
│       └── scaler_v2.joblib   # StandardScaler Sprint 3
│
├── models/
│   ├── nn_best.keras           # Meilleur modèle Sprint 2
│   ├── history.json            # Historique entraînement Sprint 2
│   ├── nn_sprint3_best.keras   # Meilleur modèle Sprint 3
│   └── history_sprint3.json    # Historique entraînement Sprint 3
│
├── mlruns/
│   └── mlflow.db               # Base SQLite — toutes les expériences MLflow
│
├── notebooks/
│   ├── eda.ipynb               # Sprint 1 — EDA complète
│   ├── modeling.ipynb          # Sprint 2 — Réseau de neurones baseline
│   └── sprint3.ipynb           # Sprint 3 — Architecture avancée + SHAP
│
├── src/
│   ├── preprocessing.py        # Pipeline Sprint 1 & 2
│   ├── preprocessing_v2.py     # Pipeline Sprint 3 (SMOTE + features interaction)
│   ├── train.py                # Entraînement en ligne de commande
│   └── evaluate.py             # Évaluation & génération des figures
│
└── reports/
    ├── sprint1_report.md       # Rapport préparation des données
    ├── sprint2_report.md       # Rapport intermédiaire
    └── sprint3_report.md       # Rapport final
```

---

## Installation

```bash
pip install tensorflow mlflow scikit-learn imbalanced-learn shap \
            pandas numpy matplotlib seaborn joblib codecarbon
```

---

## Utilisation

### Sprint 1 & 2 — Pipeline de base

**1. Prétraitement**
```bash
python src/preprocessing.py
```
Génère `data/processed/train.csv`, `val.csv`, `test.csv`, `scaler.joblib`.

**2. EDA**  
Ouvrir `notebooks/eda.ipynb` et exécuter toutes les cellules.

**3. Entraînement**
```bash
python src/train.py
python src/train.py --epochs 50 --lr 5e-4 --batch 128
```

**4. Évaluation**
```bash
python src/evaluate.py
python src/evaluate.py --threshold 0.35
```

---

### Sprint 3 — Pipeline avancé

**1. Prétraitement v2** (dataset complet + SMOTE)
```bash
python src/preprocessing_v2.py
```
Génère `data/processed/train_v2.csv`, `val_v2.csv`, `test_v2.csv`, `scaler_v2.joblib`.

**2. Notebook Sprint 3**  
Ouvrir `notebooks/sprint3.ipynb` et exécuter toutes les cellules.

---

### MLflow — suivi des expériences

```bash
python -m mlflow ui --backend-store-uri "sqlite:///C:/BigData/mlruns/mlflow.db"
```
Puis ouvrir **http://localhost:5000**

---

## Architecture des modèles

### Sprint 2 — Réseau dense baseline
```
Input (25)
    → Dense(256, ReLU) + BatchNorm + Dropout(0.4)
    → Dense(128, ReLU) + BatchNorm + Dropout(0.3)
    → Dense(1, Sigmoid)
```

### Sprint 3 — Architecture résiduelle
```
Input (29)
    → Dense(256, ReLU) + BatchNorm + Dropout(0.4)
    → Dense(128, ReLU) + BatchNorm  ← + skip connection depuis Input
    → Add() → ReLU → Dropout(0.3)
    → Dense(64, ReLU) + BatchNorm + Dropout(0.2)
    → Dense(1, Sigmoid)
```

**Optimiseur :** Adam + Cosine Decay  
**Régularisation :** L2 (λ=1e-4) + Dropout + BatchNormalization + EarlyStopping  
**Équilibrage :** SMOTE sur le train uniquement (Sprint 3)

---

## Résultats

| Sprint | Modèle | Dataset | AUC-ROC | Seuil optimal |
|---|---|---|---|---|
| Sprint 2 | Dense 256→128 | 70k 50/50 | ~0.824 | ~0.45 |
| Sprint 3 | Résiduel 256→128→64 | 253k + SMOTE | — | 0.35 (médical) |

**Seuil recommandé en production :** `0.35`  
Dans un contexte de dépistage médical, minimiser les faux négatifs (diabétiques non détectés) est prioritaire.

---

## Sprints

| Sprint | Objectif | Notebook | Rapport |
|---|---|---|---|
| Sprint 1 | Prétraitement & EDA | `eda.ipynb` | `sprint1_report.md` |
| Sprint 2 | Réseau de neurones baseline + MLflow | `modeling.ipynb` | `sprint2_report.md` |
| Sprint 3 | Architecture avancée + SMOTE + SHAP + Bilan CO2 | `sprint3.ipynb` | `sprint3_report.md` |

---

## Technologies

| Catégorie | Outil |
|---|---|
| Deep Learning | TensorFlow 2.x / Keras |
| Tracking | MLflow (SQLite) |
| Équilibrage | imbalanced-learn (SMOTE) |
| Explicabilité | SHAP (DeepExplainer) |
| Prétraitement | scikit-learn, pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Empreinte carbone | codecarbon |

---

## Auteur

Projet réalisé dans le cadre du cours **BigData — FISE A4 INFO Nancy**
