# Sprint 1 — Rapport : Préparation des données
**Projet :** Prédiction du diabète — BRFSS 2015  
**Fichiers :** `eda.ipynb` · `preprocessing.py`  
**Dataset :** `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`

---

## Livrables

| # | Livrable | Statut | Détail |
|---|---|---|---|
| 1 | Chargement et compréhension initiale | `eda.ipynb` section 2 |
| 2 | Séparation cible / variables explicatives |  `preprocessing.py` — `TARGET = "Diabetes_binary"` |
| 3 | Scission train / val / test | 64% / 16% / 20% stratifié |
| 4 | Typage des variables |  `eda.ipynb` section 3 |
| 5 | Nettoyage doublons & valeurs manquantes | `preprocessing.py` — `validate()` |
| 6 | Analyse exploratoire quantitative| `eda.ipynb` sections 3, 6 |
| 7 | Analyse exploratoire qualitative (EDA visuelle) | `eda.ipynb` sections 4, 5 |
| 8 | Normalisation | `preprocessing.py` — `scale_continuous()` |
| 9 | Sauvegarde du jeu de données nettoyé | `data/processed/` |

---

## 1. Chargement et compréhension initiale

**Fichier source :** `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`  
**Origine :** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015

| Indicateur | Valeur |
|---|---|
| Lignes brutes | 70 692 |
| Colonnes | 22 |
| Variable cible | `Diabetes_binary` (0 = non-diabétique, 1 = diabétique/pré-diabétique) |
| Balance brute | 50% / 50% (dataset déjà sous-échantillonné) |

---

## 2. Séparation cible / variables explicatives

La variable cible `Diabetes_binary` est isolée dès le chargement dans `preprocessing.py`.  
Les 21 variables explicatives sont réparties en 3 types :

**Variables binaires (14)**  
`HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`, `Sex`

**Variables ordinales (4)**  
`GenHlth` (1–5), `Age` (1–13), `Education` (1–6), `Income` (1–8)

**Variables continues (3)**  
`BMI`, `MentHlth` (0–30 jours), `PhysHlth` (0–30 jours)

---

## 3. Scission du dataset

Implémentée dans `preprocessing.py` via la fonction `split_data()`.  
Split **stratifié** sur `Diabetes_binary` pour garantir l'équilibre des classes dans chaque jeu.

| Jeu | Proportion | Lignes (approx.) |
|---|---|---|
| Train | 64% | ~44 200 |
| Validation | 16% | ~11 000 |
| Test | 20% | ~13 800 |

```python
# Extrait de preprocessing.py — split_data()
train_val, test = train_test_split(df, test_size=0.20, stratify=df[TARGET])
train, val      = train_test_split(train_val, test_size=0.20, stratify=train_val[TARGET])
```

---

## 4. Typage des variables

Réalisé dans `eda.ipynb` (section 3). Tableau récapitulatif produit automatiquement :

| Type | Nombre | Variables |
|---|---|---|
| Binaire (0/1) | 14 | HighBP, HighChol, Smoker… |
| Ordinale | 4 | GenHlth, Age, Education, Income |
| Continue | 3 | BMI, MentHlth, PhysHlth |
| Engineered | 4 | Obese, CardioRisk, UnhealthyLifestyle, PoorHealth |

---

## 5. Nettoyage — doublons & valeurs manquantes

Implémenté dans `preprocessing.py` via `validate()` :

| Vérification | Résultat | Action |
|---|---|---|
| Doublons | **1 635 détectés** (2.3%) | Supprimés via `drop_duplicates()` |
| Valeurs manquantes | **0** | Aucune action nécessaire |
| BMI aberrant (< 10 ou > 100) | 0 | Aucune action nécessaire |

---

## 6. Analyse exploratoire quantitative

Réalisée dans `eda.ipynb` (sections 3 et 6).

**Stats descriptives des variables clés :**

| Variable | Observation principale |
|---|---|
| `BMI` | Médiane ~30 (diabétiques) vs ~27 (non-diabétiques) |
| `Age` | Risque croissant à partir de la tranche 7 (50–54 ans) |
| `GenHlth` | Diabétiques majoritairement "Passable" ou "Mauvais" |
| `MentHlth` / `PhysHlth` | Distribution similaire entre les deux classes |

**Top 5 corrélations avec `Diabetes_binary` (Pearson) :**

| Rang | Feature | \|r\| |
|---|---|---|
| 1 | `GenHlth` | 0.408 |
| 2 | `HighBP` | 0.382 |
| 3 | `BMI` | 0.293 |
| 4 | `HighChol` | 0.289 |
| 5 | `Age` | 0.279 |

**Top 5 Feature Importance (Random Forest, 150 arbres, max_depth=8) :**

| Rang | Feature | Importance (Gini) |
|---|---|---|
| 1 | `HighBP` | 0.2417 |
| 2 | `GenHlth` | 0.2404 |
| 3 | `BMI` | 0.1300 |
| 4 | `HighChol` | 0.1038 |
| 5 | `Age` | 0.0855 |

---

## 7. Analyse exploratoire qualitative (EDA visuelle)

Réalisée dans `eda.ipynb` (sections 4 et 5). 4 familles de visualisations produites :

**Distribution de la cible** (section 4) — bar chart + pie chart confirment l'équilibre 50/50.

**Variables binaires** (section 5.1) — 14 bar charts de taux de prévalence par classe. Observations notables :

| Feature | Non-diabétiques | Diabétiques | Écart |
|---|---|---|---|
| `HighBP` | ~42% | ~67% | +25 pts |
| `HeartDiseaseorAttack` | ~7% | ~17% | +10 pts |
| `DiffWalk` | ~14% | ~31% | +17 pts |

**BMI** (section 5.2) — histogramme superposé + boxplot par classe. Distribution clairement décalée vers des valeurs plus élevées chez les diabétiques.

**Variables ordinales** (section 5.3) — 6 bar charts par tranche. Gradient de risque net sur `Age` et `GenHlth`.

**Features engineered** (section 5.4) — `CardioRisk` et `Obese` montrent une séparation nette entre les deux classes.

**Heatmap de corrélations** (section 6) — matrice 22×22 de Pearson. Pas de multicolinéarité sévère. Les corrélations inter-features restent modérées (max ~0.5 entre `GenHlth` et `DiffWalk`).

---

## 8. Normalisation

Implémentée dans `preprocessing.py` via `scale_continuous()`.

**Méthode choisie : StandardScaler** (centrage-réduction : moyenne 0, écart-type 1)

**Colonnes normalisées :** `BMI`, `MentHlth`, `PhysHlth`, `CardioRisk`, `UnhealthyLifestyle`

**Règle de data leakage :** le scaler est ajusté **uniquement sur le train** puis appliqué sur val et test :

```python
scaler = StandardScaler()
train[cols] = scaler.fit_transform(train[cols])  # fit + transform
val[cols]   = scaler.transform(val[cols])         # transform seulement
test[cols]  = scaler.transform(test[cols])        # transform seulement
```

Le scaler est sauvegardé dans `data/processed/scaler.joblib` pour réutilisation en inférence.

**Pourquoi pas les variables binaires/ordinales ?**  
Les variables binaires (0/1) et ordinales (entières bornées) ont des plages déjà comparables et ne nécessitent pas de normalisation pour les réseaux denses.

---

## 9. Sauvegarde du jeu de données nettoyé

Implémentée dans `preprocessing.py` via `save()`. Fichiers produits dans `data/processed/` :

| Fichier | Lignes | Description |
|---|---|---|
| `train.csv` | ~44 200 | Jeu d'entraînement, normalisé |
| `val.csv` | ~11 000 | Jeu de validation, normalisé |
| `test.csv` | ~13 800 | Jeu de test, normalisé |
| `scaler.joblib` | — | StandardScaler sérialisé (scikit-learn) |

---

## Structure des fichiers Sprint 1

```
project/
├── data/
│   ├── raw/
│   │   └── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── scaler.joblib
├── notebooks/
│   └── eda.ipynb
├── src/
│   └── preprocessing.py
└── reports/
    └── sprint1_report.md
```

---

## Conclusions

Le dataset est propre et prêt pour la modélisation :

- Aucune valeur manquante dans le dataset source
- 1 635 doublons supprimés (2.3% du total)
- Classes parfaitement équilibrées (50/50) — pas de rééchantillonnage nécessaire pour le Sprint 2
- 4 features engineered ajoutées portant le total à **25 variables explicatives**
- Split 64/16/20 stratifié et reproductible (`random_state=42`)
- StandardScaler ajusté sur le train uniquement, sauvegardé pour l'inférence

Les features les plus prédictives identifiées sont `HighBP`, `GenHlth`, `BMI`, `HighChol` et `Age`, ce qui est cohérent avec la littérature médicale sur les facteurs de risque du diabète de type 2.

---

*Sprint 1 — Préparation des données*
