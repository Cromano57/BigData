# Projet BigData — Prédiction du Diabète
**BOUDEMAGH Isrâ - FRAIOLI Enzo - RIECKENBERG Bruno - ROMANO Corentin**
**FISE A4 INFO — Nancy**

Prédiction du diabète à partir des données de santé BRFSS 2015 (CDC).  
Classification binaire : déterminer si un individu est diabétique ou non à partir de 21 indicateurs de santé déclaratifs.

---

## Dataset

**Source :** [CDC Diabetes Health Indicators — Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)  
**Origine :** Behavioral Risk Factor Surveillance System (BRFSS) 2015

| Fichier | Lignes | Description |
|---|---|---|
| `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` | 70 692 | Équilibré 50/50 |

**Variable cible :** `Diabetes_binary` (0 = non-diabétique, 1 = diabétique/pré-diabétique)

---

## Structure du projet

```
BigData/
├── data/
│   ├── raw/
│   │   └── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
│   └── processed/
│       ├── train.csv        # 64%
│       ├── val.csv          # 16%
│       ├── test.csv         # 20%
│       └── scaler.joblib    # StandardScaler sérialisé
│
├── notebooks/
│   └── eda.ipynb            # EDA complète
│
├── src/
│   └── preprocessing.py     # Pipeline de prétraitement
│
└── reports/
    └── sprint1_report.md
```

---

## Installation

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
```

---

## Utilisation

**1. Prétraitement** — génère train / val / test dans `data/processed/`
```bash
python src/preprocessing.py
```

**2. EDA** — ouvrir `notebooks/eda.ipynb` et exécuter toutes les cellules

---

## Pipeline de prétraitement (`src/preprocessing.py`)

| Étape | Fonction | Description |
|---|---|---|
| Chargement | `load_data()` | Lecture du CSV brut |
| Validation | `validate()` | Suppression doublons, vérification BMI |
| Feature engineering | `engineer_features()` | Création de 4 nouvelles variables |
| Split | `split_data()` | 64% train / 16% val / 20% test stratifié |
| Normalisation | `scale_continuous()` | StandardScaler sur les variables continues |
| Sauvegarde | `save()` | Export CSV + scaler.joblib |

**Features engineered :**

| Feature | Définition |
|---|---|
| `Obese` | BMI ≥ 30 |
| `CardioRisk` | HighBP + HighChol + HeartDiseaseorAttack + Stroke (0–4) |
| `UnhealthyLifestyle` | Pas d'activité + Smoker + HvyAlcoholConsump (0–3) |
| `PoorHealth` | MentHlth > 14 OU PhysHlth > 14 jours |

---

## Résultats EDA

- **1 635 doublons** supprimés (2.3%)
- **Aucune valeur manquante**
- **Classes équilibrées** : 50% / 50%
- **25 features** après engineering

**Top 5 features prédictives (Random Forest + Pearson) :**

| Feature | Importance (Gini) | Corrélation \|r\| |
|---|---|---|
| `HighBP` | 0.2417 | 0.382 |
| `GenHlth` | 0.2404 | 0.408 |
| `BMI` | 0.1300 | 0.293 |
| `HighChol` | 0.1038 | 0.289 |
| `Age` | 0.0855 | 0.279 |

---

## Auteur

Projet réalisé dans le cadre du cours **BigData — FISE A4 INFO Nancy**
