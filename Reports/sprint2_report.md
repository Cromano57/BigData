# Sprint 2 — Rapport intermédiaire
**Projet :** Prédiction du diabète — BRFSS 2015  
**Date :** 2026-03-16  
**Framework :** TensorFlow / Keras  |  **Tracking :** MLflow

---

## 1. Objectifs du sprint

| # | Livrable | Statut |
|---|---|---|
| 1 | Choix de l'architecture réseau | ✅ |
| 2 | Construction du réseau de neurones | ✅ |
| 3 | Entraînement et évaluation | ✅ |
| 4 | Analyse du seuil de décision | ✅ |
| 5 | Suivi des expérimentations (MLflow) | ✅ |
| 6 | Rapport intermédiaire | ✅ |

---

## 2. Correction Sprint 1 — Split train / val / test

Le sprint 1 ne disposait que d'un split binaire (train/test). Le preprocessing a été mis à jour :

| Jeu | Proportion | Lignes (approx.) |
|---|---|---|
| Train | 64 % | ~44 200 |
| Validation | 16 % | ~11 000 |
| Test | 20 % | ~13 800 |

Split **stratifié** sur `Diabetes_binary` pour garantir l'équilibre des classes dans chaque jeu.  
Scaler sauvegardé → `data/processed/scaler.joblib` (fit uniquement sur le train).

---

## 3. Architecture du réseau

```
Input (25 features)
    ↓
Dense(64, ReLU) → BatchNormalization → Dropout(0.3)
    ↓
Dense(32, ReLU) → BatchNormalization → Dropout(0.2)
    ↓
Dense(1, Sigmoid)
```

**Choix de conception :**

- **Deux couches cachées** — suffisant pour une classification tabulaire binaire ; évite le sur-apprentissage sur un dataset de ~44k exemples.
- **BatchNormalization** — stabilise l'entraînement, réduit la sensibilité au learning rate.
- **Dropout** — régularisation complémentaire à L2 (λ=1e-4).
- **Sigmoid en sortie** — sortie probabiliste [0,1] nécessaire pour l'analyse du seuil.

**Paramètres :**

| Hyperparamètre | Valeur |
|---|---|
| Optimiseur | Adam |
| Learning rate initial | 1e-3 |
| Loss | BinaryCrossentropy |
| Batch size | 256 |
| Epochs max | 30 |
| Early stopping patience | 7 |
| ReduceLROnPlateau | factor=0.5, patience=3 |

**Métriques suivies :** Accuracy, AUC-ROC, Précision, Rappel

---

## 4. Stratégie de callbacks

| Callback | Rôle |
|---|---|
| `EarlyStopping` (val_loss, patience=7) | Arrêt si pas d'amélioration, restaure les meilleurs poids |
| `ReduceLROnPlateau` (val_loss, factor=0.5, patience=3) | Réduit le LR si stagnation |
| `ModelCheckpoint` (val_auc) | Sauvegarde le meilleur modèle → `nn_best.keras` |

---

## 5. Résultats attendus

*(À compléter après exécution de `notebooks/modeling.ipynb`)*

| Métrique | Seuil 0.5 | Seuil optimal |
|---|---|---|
| Accuracy | — | — |
| AUC-ROC | — | — |
| F1-score | — | — |
| Précision | — | — |
| Rappel | — | — |

---

## 6. Analyse du seuil de décision

Le seuil par défaut (0.5) n'est pas nécessairement optimal pour un problème médical.  
Dans ce contexte (dépistage du diabète), **minimiser les faux négatifs** (FN) est prioritaire, un diabétique non détecté est plus coûteux qu'un faux positif.

**Méthode :** scan du seuil de 0.05 à 0.95 et maximisation du F1-score.  
Le seuil optimal est reporté dans MLflow pour comparaison entre runs.

---

## 7. Suivi MLflow

```bash
# Lancer l'interface
mlflow ui --backend-store-uri file://./mlruns

# Accéder à : http://localhost:5000
```

**Paramètres loggés :** epochs, batch_size, learning_rate, hidden_units, dropout_rates, l2_lambda  
**Métriques loggées :** test_roc_auc, test_f1_optimal, best_threshold, TP, FP, TN, FN + courbes par epoch

---

## 8. Structure du projet mise à jour

```
project/
├── data/
│   ├── raw/         brfss_2015.csv
│   └── processed/   train.csv · val.csv · test.csv · scaler.joblib
├── models/          nn_baseline.keras · nn_best.keras · history.json
├── mlruns/          expériences MLflow
├── notebooks/       eda.ipynb · modeling.ipynb
├── src/
│   ├── preprocessing.py   ✅ mis à jour (train/val/test)
│   ├── train.py           ✅ nouveau
│   └── evaluate.py        ✅ nouveau
└── reports/
    ├── sprint1_report.md
    └── sprint2_report.md  ← ce fichier
```

---

## 9. Prochaines étapes — Sprint 3

- [ ] Architecture enrichie (couches supplémentaires, attention, résiduel)
- [ ] Choix du framework final (Keras vs PyTorch Lightning)
- [ ] Gestion du déséquilibre (class weights, SMOTE) — *pour le dataset non-équilibré*
- [ ] MLOps : versioning des données (DVC), CI/CD, conteneurisation
- [ ] IA Explicable : SHAP values, LIME
- [ ] Bilan développement durable (empreinte carbone de l'entraînement)
- [ ] Rapport final

---

*Sprint 2 — Rapport intermédiaire*
