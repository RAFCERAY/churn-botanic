import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
print("Chargement des données...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "NOTEBOOK BOTANIC ANALYSES.pkl")

# On charge uniquement 10% des données pour travailler efficacement
df = pd.read_pickle(DATA_PATH)
df = df.sample(frac=0.1, random_state=42)

print(f"✅ Données chargées : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
print(f"\nVariable cible :")
print(df['ATTRITION'].value_counts())
print(df['ATTRITION'].value_counts(normalize=True).mul(100).round(1))

print(f"\nColonnes disponibles :")
print(list(df.columns))
# ─────────────────────────────────────────────
# KPIs FINANCIERS
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("KPIs FINANCIERS")
print("="*50)

# Statistiques par segment
stats = df.groupby('ATTRITION').agg(
    panier_moyen       = ('TIC_TOTALTTC', 'mean'),
    achat_total_moyen  = ('TOTAL_AMOUNT', 'mean'),
    freq_achat_moyenne = ('freq_achat', 'mean'),
    remise_moyenne     = ('MOY_REMISE', 'mean'),
    nb_articles_moyen  = ('NBRE_FAM_ARTICLE', 'mean')
).round(2)

print(stats)

print(f"\nPanier moyen global        : {df['TIC_TOTALTTC'].mean():.2f} €")
print(f"Achat total moyen annuel   : {df['TOTAL_AMOUNT'].mean():.2f} €")
print(f"Fréquence achat moyenne    : {df['freq_achat'].mean():.2f} fois/an")
print(f"Remise moyenne             : {df['MOY_REMISE'].mean():.2f} €")
print(f"Ancienneté dernière cde    : {df['ANCIENNETE_DER_CDE'].mean():.0f} jours")
# ─────────────────────────────────────────────
# IMPACT FINANCIER
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("IMPACT FINANCIER")
print("="*50)

# Chiffres réels dataset (extrapolés à 100%)
total_clients        = 7408792
pct_attritionnistes  = 0.145
n_attritionnistes    = int(total_clients * pct_attritionnistes)
n_non_attritionnistes = total_clients - n_attritionnistes

# KPIs financiers réels
panier_moyen_attr    = 54.35
achat_annuel_attr    = 2815.77
freq_achat           = 2.47
taux_retention       = 0.05   # si on retient 5% des attritionnistes

# Calculs
clients_retenus      = int(n_attritionnistes * taux_retention)
revenu_recupere      = clients_retenus * achat_annuel_attr
cout_campagne        = clients_retenus * 15   # coût moyen campagne rétention 15€/client
roi                  = ((revenu_recupere - cout_campagne) / cout_campagne) * 100

print(f"Clients attritionnistes totaux  : {n_attritionnistes:,}")
print(f"Panier moyen attritionniste     : {panier_moyen_attr:.2f} €")
print(f"Achat annuel moyen              : {achat_annuel_attr:.2f} €")
print(f"Fréquence d'achat               : {freq_achat:.2f} fois/an")
print(f"\n--- Scénario : rétention de 5% des attritionnistes ---")
print(f"Clients retenus                 : {clients_retenus:,}")
print(f"Revenu récupéré                 : {revenu_recupere:,.0f} €")
print(f"Coût campagne rétention         : {cout_campagne:,.0f} €")
print(f"ROI campagne                    : {roi:.0f}%")

print(f"\n--- Scénario : rétention de 10% des attritionnistes ---")
clients_retenus_10   = int(n_attritionnistes * 0.10)
revenu_recupere_10   = clients_retenus_10 * achat_annuel_attr
cout_campagne_10     = clients_retenus_10 * 15
roi_10               = ((revenu_recupere_10 - cout_campagne_10) / cout_campagne_10) * 100
print(f"Clients retenus                 : {clients_retenus_10:,}")
print(f"Revenu récupéré                 : {revenu_recupere_10:,.0f} €")
print(f"Coût campagne rétention         : {cout_campagne_10:,.0f} €")
print(f"ROI campagne                    : {roi_10:.0f}%")

print(f"\n--- Coût total de l'attrition ---")
revenu_perdu = n_attritionnistes * achat_annuel_attr
print(f"Revenu perdu si aucune action   : {revenu_perdu:,.0f} €/an")