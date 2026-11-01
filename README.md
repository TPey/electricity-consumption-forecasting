# Prévision de la consommation électrique par séries temporelles

## Présentation

Ce projet  porte sur la prévision de la consommation électrique d’un bâtiment à partir de données temporelles mesurées toutes les 15 minutes.

L’objectif est de comparer plusieurs méthodes de prévision afin d’identifier les modèles les plus performants, en considérant deux approches :
- **Univariée** : prévision de la consommation à partir de son historique.
- **Multivariée** : prévision de la consommation à partir de son historique et de la température extérieure.

Le projet comprend également l’implémentation en R d’une méthode de prévision basée sur les plus proches voisins, appelée **Weighted Nearest Neighbours (WNN)**.

## Objectifs

- Explorer et analyser une série temporelle de consommation électrique.
- Identifier les tendances et la saisonnalité.
- Comparer plusieurs familles de modèles de prévision.
- Optimiser les hyperparamètres à l’aide de la validation croisée temporelle.
- Évaluer les modèles avec les métriques MAE, RMSE et MAPE.
- Étudier l’apport de la température extérieure dans les prévisions.

## Données

Le jeu de données contient **4 795 observations** et trois variables :

| Variable | Description |
|---|---|
| `Timestamp` | Date et heure de mesure |
| `Power (kW)` | Puissance électrique consommée — variable cible |
| `Temp (°C)` | Température extérieure |

Les mesures sont enregistrées toutes les 15 minutes, soit **96 observations par jour**. Les 96 valeurs manquantes de la consommation correspondent à la journée à prévoir.

## Méthodologie

### 1. Préparation et analyse exploratoire

- Conversion des timestamps au format date-heure.
- Vérification des valeurs manquantes et des types de données.
- Visualisation de la consommation au cours du temps.
- Analyse des fonctions d’autocorrélation (ACF) et d’autocorrélation partielle (PACF).
- Décomposition de la série temporelle pour étudier ses composantes.

L’analyse met en évidence une **saisonnalité journalière**, correspondant à une période de 96 observations, ainsi qu’une tendance et une autocorrélation marquée.

### 2. Modélisation et comparaison

Les modèles sont évalués avec une validation croisée temporelle à cinq folds, sur un horizon de prévision de 96 observations.

Les méthodes étudiées comprennent :

- Holt-Winters (additif et multiplicatif)
- ETS
- ARIMA et SARIMA
- Réseaux de neurones pour séries temporelles (NNAR)
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- Weighted Nearest Neighbours (WNN)

Les modèles multivariés exploitent également la température extérieure.

### 3. Évaluation

Les performances sont comparées à l’aide de trois métriques :

- **MAE** : erreur absolue moyenne, en kW.
- **RMSE** : racine de l’erreur quadratique moyenne, en kW.
- **MAPE** : erreur absolue moyenne en pourcentage.

## Résultats principaux

### Meilleurs modèles univariés

| Modèle | MAE validation | RMSE validation | MAPE validation | MAE test | RMSE test | MAPE test |
|---|---:|---:|---:|---:|---:|---:|
| SARIMA | 6,24 | 10,99 | 2,97 % | 11,26 | 20,26 | 4,24 % |
| NNAR | 6,71 | 11,98 | 3,19 % | 10,68 | 19,66 | 4,08 % |
| Random Forest | 7,27 | 11,85 | 3,43 % | 12,14 | 20,88 | 4,68 % |
| XGBoost | 7,62 | 11,58 | 3,68 % | 13,02 | 22,16 | 5,23 % |
| Holt-Winters multiplicatif | 9,28 | 13,40 | 4,42 % | 13,70 | 21,42 | 5,55 % |

### Meilleurs modèles multivariés

| Modèle | MAE validation | RMSE validation | MAPE validation | MAE test | RMSE test | MAPE test |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 7,63 | 12,48 | 3,61 % | 12,42 | 21,28 | 4,83 % |
| Neural Net | 6,31 | 11,51 | 3,05 % | 10,81 | 19,79 | 4,09 % |
| XGBoost | **4,02** | **6,17** | **1,89 %** | **8,02** | **16,72** | **2,94 %** |

Le modèle **XGBoost multivarié**, intégrant la température extérieure, obtient les meilleures performances globales parmi les modèles comparés.

Le modèle WNN avec `k = 12` présente des performances proches de SARIMA en validation, sans amélioration significative sur le jeu de test.

## Méthode WNN (Weighted Nearest Neighbours)

La méthode WNN implémentée en R est celle décrite dans la section 3.1 de l’article de Talavera-Llames et al. (2016), référencé ci-dessous. Cet article propose un algorithme de prévision de séries temporelles fondé sur les plus proches voisins, testé sur des données de consommation électrique d’une université espagnole.

Le principe général est le suivant :

- Identifier, dans l’historique de la série, les séquences les plus proches de la séquence la plus récente.
- Retenir les `k` voisins les plus proches (ici `k = 12`).
- Construire la prévision à partir des valeurs qui suivent ces voisins, en pondérant leur contribution.

La fonction WNN est fournie avec un exemple d’utilisation sur une série temporelle simple (`co2`), puis appliquée à la série de consommation électrique pour être comparée aux autres modèles.

## Technologies utilisées

- **Langage** : R
- **Analyse et manipulation** : `dplyr`, `ggplot2`, `readxl`
- **Prévision et séries temporelles** : `forecast`, `tseries`
- **Machine Learning** : `xgboost`, `randomForest`, `e1071`
- **Deep Learning** : `keras`, `tensorflow`

## Compétences mobilisées

- Analyse exploratoire de séries temporelles
- Détection de tendance et de saisonnalité
- Prévision univariée et multivariée
- Modélisation statistique et Machine Learning
- Optimisation d’hyperparamètres
- Validation croisée temporelle
- Évaluation et comparaison de modèles

J’ai réalisé ce projet dans le cadre de mon cursus, autour de la prévision de séries temporelles. C’était l’occasion de mettre en pratique mes connaissances en analyse de données, en statistiques et en machine learning pour prévoir la consommation électrique d’un bâtiment.

## Contexte académique et personnel

J’ai étudié et comparé plusieurs approches de modélisation, en évaluant leur précision et leur capacité de généralisation. Ce projet m’a permis de renforcer mes compétences en programmation R, en analyse de séries temporelles, en validation croisée et en sélection de modèles. J’y ai aussi approfondi ma compréhension de la méthode WNN (Weighted Nearest Neighbours), que j’ai implémentée moi-même en R.

## Référence

Talavera-Llames, R.L., Pérez-Chacón, R., Martínez-Ballesteros, M., Troncoso, A., Martínez-Álvarez, F. (2016). *A Nearest Neighbours-Based Algorithm for Big Time Series Data Forecasting*. In: Martínez-Álvarez, F., Troncoso, A., Quintián, H., Corchado, E. (eds) Hybrid Artificial Intelligent Systems. HAIS 2016. Lecture Notes in Computer Science, vol 9648, pp. 174-185. Springer, Cham.
DOI : [10.1007/978-3-319-32034-2_15](https://doi.org/10.1007/978-3-319-32034-2_15)