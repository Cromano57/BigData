# Projet BigData — Prédiction du Diabète
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
├── Data/
│   ├── Raw/
│   │   ├── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
│   │   ├── diabetes_binary_health_indicators_BRFSS2015.csv
│   │   └── diabetes_012_health_indicators_BRFSS2015.csv
│   └── Processed/
│       ├── train.csv        # 64%
│       ├── val.csv          # 16%
│       ├── test.csv         # 20%
│       └── scaler.joblib    # StandardScaler sérialisé
│
├── Notebooks/
│   ├── eda.ipynb            # Analyse exploratoire des données
│   └── modeling.ipynb       # Entraînement et évaluation du réseau de neurones
│
├── Src/
│   ├── preprocessing.py     # Pipeline de prétraitement
│   ├── train.py             # Entraînement du modèle
│   └── evaluate.py          # Évaluation des modèles
│
├── models/
│   ├── nn_best.keras        # Meilleur modèle sauvegardé
│   ├── final_256-128_lr1e-3.keras
│   └── history.json         # Historique d'entraînement
│
├── Reports/
│   ├── sprint1_report.md
│   └── sprint2_report.md
│
├── mlruns/                  # Tracking MLflow
└── requirements.txt
```

---

## Installation

**1. Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**2. Installer les dépendances**
```bash
pip install -r requirements.txt
```

---

## Utilisation

**Notebooks** — ouvrir et exécuter les notebooks dans le dossier `Notebooks/`
```bash
jupyter notebook Notebooks/
```

- `eda.ipynb` — analyse exploratoire des données
- `modeling.ipynb` — entraînement et évaluation du réseau de neurones

---

## Auteur

Projet réalisé dans le cadre du cours **BigData — FISE A4 INFO Nancy**
