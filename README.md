# 🌿 Churn Client — Prédiction d'Attrition Botanic

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rafceray-churn-botanic-app.streamlit.app)

## Contexte
Botanic est un réseau de 75+ jardineries spécialistes du jardinage écologique en France. Ce projet construit un modèle prédictif pour anticiper le départ des clients avant qu'ils ne quittent l'enseigne.

## Objectif
Identifier les clients à risque d'attrition afin de permettre à Botanic de prendre des mesures proactives de rétention ciblées.

## Dataset
- 7 408 792 transactions clients
- 5 tables sources jointes : Clients, Entêtes tickets, Lignes tickets, Référentiel articles, Référentiel magasins
- Variable cible : ATTRITION (14.5% attritionnistes / 85.5% non-attritionnistes)

## Méthodologie
1. Pipeline de jointures (5 tables sources)
2. Rééquilibrage avec SMOTE
3. Sélection de variables avec RFE
4. Régression Logistique optimisée
5. GridSearchCV pour hyperparamètre tuning

## Variables les plus prédictives
- ANCIENNETE_DERNIERE_CDE_2016 — récence dernière commande (coef: +8.15)
- TEMPS_MOY_CDE_2016 — temps moyen entre commandes (coef: -8.11)
- FREQ_COMMANDE_2016 — fréquence des commandes (coef: -3.23)
- TYPE_UNIVERS — univers produit préféré (coef: +0.28)

## Stack technique
Python · Pandas · Scikit-learn · Imbalanced-learn · Statsmodels · Matplotlib · Seaborn · Streamlit · Plotly

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Auteur
Rafika Ayari Cervera — [LinkedIn](https://www.linkedin.com/in/rafikacervera/) · [GitHub](https://github.com/RAFCERAY)
