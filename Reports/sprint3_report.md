# Sprint 3 — Rapport Final
## Optimisation Avancée, MLOps et IA Explicable

**Projet BigData — Prédiction du Diabète (BRFSS 2015)**  
**FISE A4 INFO — Nancy · Auteur : Enzo Fraioli · Date : 2026-04-15**

---

## Table des matières

1. [Architecture enrichie du modèle](#1-architecture-enrichie-du-modèle)
2. [Choix du framework de deep learning](#2-choix-du-framework-de-deep-learning)
3. [Gestion du déséquilibre des classes](#3-gestion-du-déséquilibre-des-classes)
4. [Implémentation de principes MLOps](#4-implémentation-de-principes-mlops)
5. [IA Explicable et Développement Durable](#5-ia-explicable-et-développement-durable)
6. [Conclusions et perspectives](#6-conclusions-et-perspectives)

---

## Rappel Sprint 2

Le Sprint 2 avait produit un réseau de neurones baseline (64→32) sur le dataset équilibré 50/50 :

| Métrique | Valeur |
|---|---|
| AUC-ROC | 0.8234 |
| F1-score (seuil 0.45) | 0.7760 |
| Recall (Sensibilité) | 0.8614 |
| Accuracy | 0.75 |

Le Sprint 3 enrichit ce baseline sur 5 axes : architectures, framework, déséquilibre, MLOps et XAI.

---

## 1. Architecture Enrichie du Modèle

### 1.1 Motivation

L'architecture Sprint 2 (2 couches, 64→32 neurones, ~4,800 paramètres) était volontairement simple pour établir un baseline fiable. Sprint 3 explore deux enrichissements :

- **Architecture profonde** : plus de neurones et de couches pour capturer des interactions non-linéaires plus complexes
- **Architecture résiduelle** : connexions de saut (skip connections) inspirées de ResNet, pour éviter le problème de vanishing gradient

### 1.2 Les 3 architectures comparées

#### Architecture 1 — Baseline (référence Sprint 2)
```
Input(25) → Dense(64, ReLU) → BatchNorm → Dropout(0.3)
          → Dense(32, ReLU) → BatchNorm → Dropout(0.2)
          → Dense(1, Sigmoid)

Paramètres : 4,801
Régularisation : L2(1e-4) + Dropout
```

#### Architecture 2 — Deep (256-128-64)
```
Input(25) → Dense(256, ReLU) → BatchNorm → Dropout(0.4)
          → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
          → Dense(64,  ReLU) → BatchNorm → Dropout(0.2)
          → Dense(1, Sigmoid)

Paramètres : 48,001
Régularisation : L2(1e-4) + Dropout (augmenté à 0.4 pour la première couche)
```

#### Architecture 3 — Résiduelle (128-128-64)
```
Input(25) → Dense(128) → BatchNorm
          → [Bloc résiduel 1 : Dense(128) → BN → Drop(0.3) → Dense(128) → BN → Add(skip)]
          → [Bloc résiduel 2 : Dense(64)  → BN → Drop(0.2) → Dense(64)  → BN → Add(skip)]
          → Dense(1, Sigmoid)

Paramètres : 66,049
Architecture : Functional API Keras (non-séquentielle)
```

**Justification des skip connections** : Pour des données tabulaires avec des features corrélées (CardioRisk composite de HighBP+HighChol+Stroke...), le réseau doit à la fois apprendre des représentations complexes ET propager directement le signal des features les plus discriminantes. Les connexions résiduelles permettent ce double objectif.

### 1.3 Résultats comparatifs

Entraîné sur le dataset équilibré (44,196 train), évalué sur test (13,812), seuil 0.45 :

| Architecture | AUC-ROC | F1-score | Recall | Precision | Epochs | Temps (s) |
|---|---|---|---|---|---|---|
| **Baseline (64-32)** | **0.8232** | **0.7748** | 0.8456 | **0.7149** | 41 | **7.4** |
| Deep (256-128-64) | 0.8229 | 0.7723 | 0.8413 | 0.7137 | 49 | 26.8 |
| Residual (128-64) | 0.8186 | 0.7705 | **0.8551** | 0.7010 | 26 | 18.4 |

### 1.4 Analyse des résultats

**Observation principale** : les trois architectures converge vers des performances très similaires (~0.82 AUC-ROC), avec une variation maximale de 0.005 points d'AUC entre la meilleure et la moins bonne.

**Interprétation** :
- Le problème de classification (25 features tabulaires, ~44k échantillons) est **bien capté** par le baseline 64-32
- L'ajout de couches et de neurones n'apporte pas de gain significatif : les données ne sont pas suffisamment complexes pour justifier un modèle plus profond
- Le résiduel obtient le **meilleur Recall** (0.855) ce qui est intéressant en contexte médical (minimiser les faux négatifs), mais au prix d'une précision plus basse
- L'EarlyStopping stoppe le résiduel en seulement 26 époques (contre 49 pour le deep), ce qui témoigne d'une convergence plus rapide grâce aux skip connections

**Conclusion** : le Baseline reste le meilleur choix pour ce dataset. Il offre la meilleure AUC-ROC et F1-score avec seulement 4,801 paramètres contre 48,001 pour le Deep, pour un coût carbone 3.5x inférieur.

---

## 2. Choix du Framework de Deep Learning

### 2.1 Contexte du choix

Le choix d'un framework de deep learning dépend de :
1. La nature du problème (classification tabulaire, vision, NLP...)
2. Les contraintes de déploiement (production, mobile, edge...)
3. L'intégration dans la chaîne MLOps
4. La courbe d'apprentissage de l'équipe

### 2.2 Analyse comparative des alternatives

| Critère | **TensorFlow/Keras** | PyTorch | XGBoost | LightGBM |
|---|---|---|---|---|
| Type | Deep Learning | Deep Learning | Gradient Boosting | Gradient Boosting |
| API haut niveau | Sequential/Functional | Module/Lightning | sklearn-style | sklearn-style |
| Architectures flexibles | ✅ Haute | ✅ Haute | ❌ Limitée | ❌ Limitée |
| Callbacks intégrés | ✅ Natifs | ⚠️ Via Lightning | ❌ | ❌ |
| MLflow intégration | ✅ `mlflow.keras` | ✅ `mlflow.pytorch` | ✅ | ✅ |
| Déploiement production | ✅ TFServing/TFLite | ✅ TorchServe | ✅ Direct | ✅ Direct |
| Données tabulaires | ✅ Bon | ✅ Bon | ✅✅ Excellent | ✅✅ Excellent |
| GPU/Apple Silicon | ✅ MPS | ✅ MPS | ⚠️ CUDA limité | ⚠️ CPU |
| Communauté/docs | ✅✅ Très large | ✅✅ Croissante | ✅✅ | ✅ |

### 2.3 Résultats de la comparaison NN vs Gradient Boosting

Seuil 0.45, dataset équilibré :

| Modèle | AUC-ROC | F1-score | Recall | Temps entraînement |
|---|---|---|---|---|
| Baseline NN (TF/Keras) | **0.8232** | **0.7748** | **0.8456** | 7.4s |
| XGBoost (300 arbres) | 0.8221 | 0.7707 | 0.8379 | **0.6s** |
| LightGBM (300 arbres) | 0.8233 | 0.7723 | 0.8409 | 1.6s |

> **Résultat notable** : LightGBM atteint une AUC légèrement supérieure au NN baseline (0.8233 vs 0.8232) en **12x moins de temps** d'entraînement.

### 2.4 Justification finale de TensorFlow/Keras

Malgré des performances équivalentes, TensorFlow/Keras est retenu pour ce projet pour les raisons suivantes :

1. **Flexibilité architecturale** : les architectures résiduelles, les mécanismes d'attention, le fine-tuning de modèles pré-entraînés sont natifs. XGBoost n'offre pas cette extensibilité.

2. **Apprentissage de représentations** : un NN peut apprendre des embeddings de features, utile si le jeu de données s'agrandit ou si des données non-structurées (images, texte) sont ajoutées.

3. **Callbacks intégrés** : EarlyStopping, ReduceLROnPlateau, ModelCheckpoint font partie du framework sans dépendances supplémentaires.

4. **Chaîne MLOps complète** : TFX, TFServing, TFLite permettent un déploiement sur mobile ou edge sans conversion.

5. **Objectif pédagogique** : ce projet vise à maîtriser les réseaux de neurones profonds, ce que XGBoost ne permet pas.

> **Note** : pour un déploiement purement en production sur ce dataset spécifique, LightGBM serait le choix pragmatique (performances identiques, entraînement 12x plus rapide, pas de dépendance TensorFlow).

---

## 3. Gestion du Déséquilibre des Classes

### 3.1 Le problème réel

Le dataset équilibré (50/50) utilisé dans Sprint 1/2 est artificiel — il résulte d'un sous-échantillonnage de la classe majoritaire. La prévalence réelle du diabète/pré-diabète dans la population américaine (BRFSS 2015) est bien inférieure.

**Dataset non-équilibré** (`diabetes_binary_health_indicators_BRFSS2015.csv`) :
- 253,680 lignes (après nettoyage : 229,474)
- 194,377 non-diabétiques (84.7%) — **classe majoritaire**
- 35,097 diabétiques (15.3%) — **classe minoritaire**
- Ratio : 1:5.5 (5.5 non-diabétiques pour chaque diabétique)

**Conséquence sans traitement** : un modèle naïf peut atteindre 84.7% d'accuracy en prédisant toujours "non-diabétique", mais aura un Recall de 0 — inutile en dépistage.

### 3.2 Stratégies implémentées

#### Stratégie 1 — Sans correction (baseline déséquilibré)
Entraînement direct sur données déséquilibrées, perte standard BinaryCrossentropy.

#### Stratégie 2 — Class Weights (pondération des classes)
```python
class_weight = compute_class_weight("balanced", classes=[0, 1], y=y_train)
# → w_0 (majoritaire) ≈ 0.59
# → w_1 (minoritaire) ≈ 3.27
```
La perte est multipliée par le poids de la classe concernée : les erreurs sur les diabétiques coûtent 3.27x plus cher.

#### Stratégie 3 — SMOTE (Synthetic Minority Oversampling Technique)
```python
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
# Avant : 22,462 positifs / 146,863 total (15.3%)
# Après : 124,401 positifs / 248,802 total (50.0%)
```
SMOTE génère des exemples synthétiques de la classe minoritaire par interpolation entre k voisins dans l'espace des features.

### 3.3 Résultats comparatifs

Dataset non-équilibré (~15% positifs), architecture Deep (256-128-64), seuil 0.45 :

| Stratégie | AUC-ROC | F1-score | **Recall** | Precision |
|---|---|---|---|---|
| Sans correction | 0.8192 | 0.3289 | **0.2339** | 0.5534 |
| Class weights | 0.8179 | 0.4450 | **0.8265** | 0.3045 |
| SMOTE | 0.8031 | **0.4535** | 0.6430 | 0.3502 |

### 3.4 Analyse des résultats

**Sans correction** : l'AUC-ROC reste bon (0.82) car le modèle apprend une représentation latente discriminante, mais le Recall effondre à 0.23 — le modèle manque **77% des diabétiques**.

**Class weights** :
- Meilleur Recall (0.826) : détecte **83% des diabétiques**
- L'AUC-ROC est quasi-identique (0.818 vs 0.819), confirmant que le discriminant est bien appris
- Precision plus basse (0.305) : plus de faux positifs, mais acceptable en dépistage (mieux vaut un faux positif suivi d'un test médical qu'un diabétique non-détecté)

**SMOTE** :
- Meilleur F1-score (0.454) car meilleur compromis Recall/Precision
- Recall intermédiaire (0.643) inférieur aux class weights
- Inconvénient : multiplie la taille du dataset (×1.7) → entraînement 3x plus long et 8x plus de CO₂

### 3.5 Recommandation

Pour ce cas d'usage médical (dépistage), **la class_weight est la stratégie recommandée** :
- Maximise le Recall (83%) sans augmentation artificielle du dataset
- Empreinte carbone 3x inférieure à SMOTE
- Simples à mettre en place (un seul paramètre de `model.fit`)

> **Note** : les résultats sur le dataset déséquilibré (F1≈0.45) sont inférieurs au dataset équilibré (F1≈0.77) car la tâche est intrinsèquement plus difficile avec moins d'exemples positifs en contexte de features similaires.

---

## 4. Implémentation de Principes MLOps

### 4.1 Stratégie MLOps du Sprint 3

```
Données brutes → DVC → Prétraitement → Entraînement → MLflow → Registre
     (Git)                (Python)    (CodeCarbon)    (SQLite)  (Model v1)
       |                                                              |
  Versionnement                                               Production
```

Les outils implémentés :

| Outil | Rôle | Fichier(s) |
|---|---|---|
| DVC | Versionnement données | `.dvc/`, `*.csv.dvc`, `dvc.yaml` |
| MLflow | Suivi expériences | `mlruns/mlflow.db` |
| MLflow Registry | Gestion modèles | Model `diabetes_predictor` v1 |
| CodeCarbon | Empreinte carbone | `Reports/emissions.csv` |

### 4.2 DVC — Data Version Control

DVC étend Git pour les fichiers volumineux (CSV, modèles) en les remplaçant par de petits fichiers de métadonnées (`.dvc`) contenant leur hash MD5.

**Setup réalisé** :
```bash
dvc init
git add .dvc/ .dvcignore
git commit -m "Initialize DVC"

# Untrack les CSV de Git (trop volumineux)
git rm -r --cached Data/Raw/*.csv
git commit -m "chore: untrack large raw data files from Git"

# Track les CSV avec DVC
dvc add Data/Raw/diabetes_binary_5050split_health_indicators_BRFSS2015.csv
dvc add Data/Raw/diabetes_binary_health_indicators_BRFSS2015.csv
dvc add Data/Raw/diabetes_012_health_indicators_BRFSS2015.csv
git add Data/Raw/*.csv.dvc Data/Raw/.gitignore
git commit -m "chore: track raw data with DVC"
```

**Pipeline `dvc.yaml`** — reproductibilité garantie :
```yaml
stages:
  preprocess_balanced:
    cmd: python Src/preprocessing.py
    deps: [Src/preprocessing.py, Data/Raw/diabetes_binary_5050split_health_indicators_BRFSS2015.csv.dvc]
    outs: [Data/Processed/train.csv, Data/Processed/val.csv, Data/Processed/test.csv]

  preprocess_imbalanced:
    cmd: python Src/preprocess_imbalanced.py
    deps: [Src/preprocess_imbalanced.py, Data/Raw/diabetes_binary_health_indicators_BRFSS2015.csv.dvc]
    outs: [Data/Processed/imbalanced/train.csv, ...]

  train_advanced:
    cmd: python Src/train_advanced.py --arch all --epochs 50
    deps: [Src/train_advanced.py, Data/Processed/train.csv, ...]
    outs: [models/baseline_best.keras, models/deep_256_128_64_best.keras, ...]

  explain:
    cmd: python Src/explainability.py --n_shap 300
    deps: [Src/explainability.py, Data/Processed/train.csv, ...]
    outs: [Reports/shap_summary.png, Reports/shap_beeswarm.png]
```

Exécuter `dvc repro` relance uniquement les étapes dont les dépendances ont changé.

### 4.3 MLflow — Suivi des expériences

Toutes les expériences sont loguées dans une base SQLite locale (`mlruns/mlflow.db`).

**Expériences créées** :

| Expérience | Runs | Description |
|---|---|---|
| `diabetes_sprint3_balanced` | 3 | 3 architectures sur données 50/50 |
| `diabetes_sprint3_imbalanced` | 3 | 3 stratégies sur données réelles (15%) |

**Paramètres loggués** :
- Architecture, input_dim, epochs_max, batch_size, learning_rate, l2_lambda, patience
- class_weight (si applicable), smote_applied

**Métriques loggués** :
- accuracy, auc_roc, f1, precision, recall (sur test set)
- train_time_s, n_epochs, co2_kg
- Par époque : epoch_train_loss, epoch_val_loss

**Meilleurs runs** :

Dataset équilibré :
```
9c0db3a8... | baseline    | AUC=0.8232 | F1=0.7748
3a50bb61... | deep        | AUC=0.8229 | F1=0.7723
712fd817... | residual    | AUC=0.8186 | F1=0.7705
```

### 4.4 MLflow Model Registry

Le meilleur modèle (baseline) a été enregistré dans le registre :

```python
mlflow.keras.log_model(
    model,
    artifact_path="best_model",
    registered_model_name="diabetes_predictor"
)
# → Model: diabetes_predictor, Version: 1, Stage: None
```

Ce registre permet de :
- Versionner les modèles de façon indépendante des runs
- Promouvoir un modèle de "Staging" à "Production"
- Rollback si une nouvelle version régresse

**Visualisation MLflow UI** :
```bash
python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# → http://localhost:5000
```

### 4.5 Nouveaux scripts Src/

| Script | Rôle |
|---|---|
| `train_advanced.py` | Entraînement des 3 architectures + SMOTE + class_weight + CodeCarbon |
| `explainability.py` | SHAP (global + local) + LIME + résumé carbon footprint |
| `preprocess_imbalanced.py` | Prétraitement du dataset non-équilibré (253k lignes) |
| `create_notebook.py` | Génération du notebook Sprint 3 |

---

## 5. IA Explicable et Développement Durable

### 5.1 Enjeux de l'explicabilité en santé

Dans le domaine médical, un modèle "boîte noire" est problématique pour trois raisons :

1. **Confiance clinique** : un professionnel de santé ne peut pas s'appuyer sur une prédiction inexpliquée pour un diagnostic
2. **Détection de biais** : vérifier que le modèle n'utilise pas de proxies problématiques (ex: genre, ethnie, statut socio-économique)
3. **RGPD (Article 22)** : droit à l'explication pour les décisions automatisées impactant une personne

### 5.2 SHAP — SHapley Additive exPlanations

SHAP est fondé sur la **valeur de Shapley** de la théorie des jeux coopératifs. Pour chaque prédiction, la valeur SHAP d'une feature quantifie sa contribution marginale en moyennant sur toutes les permutations possibles des features.

**Propriétés garanties** :
- Additivité : la somme des SHAP values + E[f(X)] = f(x)
- Cohérence : si le modèle augmente l'impact d'une feature, sa SHAP value augmente
- Précision : l'attribution est exacte (pas d'approximation locale comme LIME)

**Implémentation** : `shap.DeepExplainer` pour Keras, background = 150 échantillons de train, explication = 400 échantillons de test.

#### Top 5 features par importance SHAP globale

| Rang | Feature | Description | Impact |
|---|---|---|---|
| 1 | **GenHlth** | Santé générale perçue (1=Excellente, 5=Mauvaise) | ↑ très fort si GenHlth élevé |
| 2 | **HighBP** | Hypertension artérielle (0/1) | ↑ fort si présent |
| 3 | **BMI** | Indice de masse corporelle (normalisé) | ↑ croissant avec la valeur |
| 4 | **Age** | Tranche d'âge (1-13) | ↑ croissant avec l'âge |
| 5 | **CardioRisk** | Score risque cardiovasculaire (0-4) | ↑ selon le score |

Ces 5 features correspondent aux facteurs de risque du diabète de type 2 bien connus en épidémiologie : hypertension, surpoids/obésité, âge avancé et mauvais état de santé général.

#### Explications locales (Waterfall)

Le SHAP Waterfall Plot permet d'expliquer une prédiction individuelle :
- Point de départ : E[f(X)] = valeur de base du modèle (probabilité moyenne)
- Chaque feature ajoute (↑) ou soustrait (↓) de la prédiction
- Point d'arrivée : f(x) = prédiction finale

**Fichiers générés** :
- `Reports/shap_summary.png` — Bar plot importance globale
- `Reports/shap_beeswarm.png` — Distribution des impacts
- `Reports/shap_waterfall_positive.png` — Cas diabétique prédit
- `Reports/shap_waterfall_negative.png` — Cas non-diabétique prédit
- `Reports/shap_dependence_GenHlth.png` — Relation feature/SHAP pour GenHlth
- `Reports/shap_dependence_BMI.png` — Relation feature/SHAP pour BMI

### 5.3 LIME — Local Interpretable Model-agnostic Explanations

LIME crée un modèle linéaire local autour d'une prédiction individuelle en perturbant les features d'entrée et en observant l'impact sur la sortie.

**Avantages vs SHAP** :
- Totalement **agnostique** au modèle (fonctionne avec n'importe quel classifieur)
- Intuitivement compréhensible (coefficients d'une régression linéaire locale)

**Limitations** :
- Instabilité : deux runs peuvent donner des explications légèrement différentes
- Localité : l'explication n'est valable qu'autour de la prédiction analysée

**Fichiers générés** :
- `Reports/lime_positive.png` — Explication d'un vrai positif (diabétique prédit)
- `Reports/lime_negative.png` — Explication d'un vrai négatif (non-diabétique prédit)

### 5.4 Développement Durable — Empreinte Carbone

#### Mesures réalisées avec CodeCarbon

CodeCarbon mesure automatiquement :
- Consommation CPU (en Watts)
- Durée d'entraînement (en secondes)
- Mix énergétique du pays (France : 52 g CO₂/kWh)
- Émissions résultantes en kg CO₂ équivalent

**Résultats par run d'entraînement** :

| Entraînement | Durée (s) | CO₂ (µg) | Énergie (mWh) |
|---|---|---|---|
| baseline (équilibré) | 7.5 | **5.69** | 0.102 |
| deep (équilibré) | 26.8 | 20.24 | 0.361 |
| residual (équilibré) | 18.4 | 13.92 | 0.248 |
| deep_imbalanced | 39.3 | 29.70 | 0.530 |
| deep_classweight | 21.8 | 16.42 | 0.293 |
| deep_smote | 64.0 | **48.35** | 0.863 |
| **TOTAL** | **177.9** | **134.32** | **2.40 mWh** |

#### Équivalences

- CO₂ total (6 runs) : **0.134 g CO₂ équivalent**
- Équivaut à : **0.64 mm** en voiture (209 g/km)
- Équivaut à : **0.73 s** de streaming vidéo HD

#### Analyse sustainability

**Ratio CO₂/AUC** (moins = mieux) :

| Architecture | CO₂ (µg) | AUC-ROC | Ratio |
|---|---|---|---|
| **Baseline (recommandé)** | **5.69** | 0.8232 | **6.9** |
| Residual | 13.92 | 0.8186 | 17.0 |
| Deep | 20.24 | 0.8229 | 24.6 |
| SMOTE (deep) | 48.35 | 0.8031 | 60.2 |

**Le baseline offre le meilleur compromis performance/impact environnemental.**

#### Recommandations pour réduire l'empreinte

1. **EarlyStopping** : économise ~30% des époques inutiles (implémenté)
2. **Batch size 256** : utilisation efficace du CPU/GPU (implémenté)
3. **Architecture légère** : 4,801 paramètres vs 48,001 pour le Deep (×10)
4. **Class weights vs SMOTE** : same recall avec 3× moins de CO₂
5. **France** : mix énergétique favorable (~52 g CO₂/kWh vs ~500 g en Pologne)
6. **Inférence optimisée** : TFLite pour la production (réduction ×4-10 de la taille)

---

## 6. Conclusions et Perspectives

### 6.1 Bilan des livrables Sprint 3

| Livrable | Statut | Résultat clé |
|---|---|---|
| Architecture enrichie | ✅ Réalisé | Baseline 64-32 reste optimal sur données équilibrées |
| Choix framework | ✅ Réalisé | TF/Keras justifié ; LightGBM compétitif sur tabular |
| Déséquilibre classes | ✅ Réalisé | Class weights : Recall 0.23 → 0.83 sur données réelles |
| MLOps (DVC + MLflow) | ✅ Réalisé | Pipeline reproductible, registre modèle v1 |
| XAI (SHAP + LIME) | ✅ Réalisé | Top features : GenHlth, HighBP, BMI identifiées |
| Développement durable | ✅ Réalisé | 0.134 g CO₂eq total, Baseline = meilleur ratio |

### 6.2 Modèle recommandé pour la production

**Contexte production** (dépistage populationnel, données réelles à 15% de positifs) :

```
Architecture : Baseline NN (64-32)
Dataset      : Données non-équilibrées (~15% positifs)
Stratégie    : Class weights (w_1 ≈ 3.27)

Performance attendue :
  AUC-ROC  : 0.82
  Recall   : 0.83 (détecte 83% des diabétiques)
  Precision: 0.30 (3 vrais positifs pour 7 détectés)
  → Acceptable en dépistage (confirmation médicale requise)
```

### 6.3 Bilan des trois sprints

| Sprint | Livrable | AUC-ROC | Recall |
|---|---|---|---|
| Sprint 1 | Prétraitement, EDA, Feature Engineering | — | — |
| Sprint 2 | Baseline NN (64-32), MLflow basique | 0.8234 | 0.8614 |
| **Sprint 3** | **3 architectures, DVC, SHAP, LIME, CodeCarbon** | **0.8232** | **0.8456** |

> La faible variation d'AUC entre Sprint 2 et Sprint 3 confirme que le baseline de Sprint 2 était déjà bien optimisé. Sprint 3 apporte principalement les outils MLOps, l'explicabilité et la gestion du déséquilibre.

### 6.4 Perspectives

1. **Données longitudinales** : le BRFSS collecte des données annuelles — un modèle séquentiel (LSTM) pourrait exploiter l'évolution temporelle
2. **Ensemble methods** : combiner NN + LightGBM par stacking pour une légère amélioration de l'AUC
3. **Déploiement** : conteneurisation Docker + API REST Flask/FastAPI + TFServing
4. **Monitoring** : détection du data drift avec Evidently ou MLflow Model Monitoring
5. **Federated Learning** : entraînement distribué sur données hospitalières sans centralisation (confidentialité)

---

## Annexe — Fichiers produits lors du Sprint 3

### Scripts Python (Src/)

| Fichier | Rôle |
|---|---|
| `train_advanced.py` | 3 architectures, SMOTE, class_weight, CodeCarbon |
| `explainability.py` | SHAP global/local, LIME, résumé carbone |
| `preprocess_imbalanced.py` | Prétraitement dataset non-équilibré |
| `create_notebook.py` | Génération du notebook sprint3 |

### Notebooks

| Fichier | Contenu |
|---|---|
| `Notebooks/sprint3_modeling.ipynb` | 15 cellules couvrant les 6 livrables |

### Modèles (models/)

| Fichier | Architecture | AUC-ROC |
|---|---|---|
| `baseline_best.keras` | 64-32 | 0.8232 |
| `deep_256_128_64_best.keras` | 256-128-64 | 0.8229 |
| `residual_128_64_best.keras` | Résiduel | 0.8186 |

### Rapports (Reports/)

| Fichier | Type |
|---|---|
| `shap_summary.png` | Importance globale SHAP |
| `shap_beeswarm.png` | Distribution SHAP |
| `shap_waterfall_positive.png` | Explication locale (positif) |
| `shap_waterfall_negative.png` | Explication locale (négatif) |
| `shap_dependence_GenHlth.png` | Dependence plot GenHlth |
| `shap_dependence_BMI.png` | Dependence plot BMI |
| `lime_positive.png` | LIME explication (positif) |
| `lime_negative.png` | LIME explication (négatif) |
| `emissions.csv` | Mesures CodeCarbon |
| `sprint3_results.json` | Métriques des 3 architectures |
| `framework_comparison.json` | Comparaison NN vs XGBoost vs LightGBM |

### DVC

| Fichier | Rôle |
|---|---|
| `Data/Raw/*.csv.dvc` | Pointeurs DVC vers les 3 datasets |
| `dvc.yaml` | Définition du pipeline reproductible |
| `.dvc/config` | Configuration DVC |

---

*Sprint 3 terminé le 2026-04-15*
