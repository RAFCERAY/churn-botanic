"""
=============================================================
PIPELINE DE DONNÉES — Churn Client Botanic
=============================================================
Auteur  : Rafika Ayari Cervera
Date    : 2024
Projet  : Modélisation prédictive d'attrition clients Botanic

Description :
    Ce script documente et reproduit le pipeline complet de
    préparation des données depuis les sources brutes jusqu'à
    la base analytique finale utilisée pour la modélisation.

Sources de données :
    1. CLIENTS_Botanic.csv       — données clients (CRM)
    2. ENTETES_TICKET_V4.csv     — en-têtes des tickets d'achat
    3. LIGNES_TICKET_V4.csv      — lignes détail des tickets
    4. REF_ARTICLE.CSV           — référentiel articles
    5. REF_MAGASIN.CSV           — référentiel magasins

Résultat :
    NOTEBOOK BOTANIC ANALYSES.pkl — base analytique finale
    7 408 792 lignes · 33 variables · prête pour modélisation
=============================================================
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "..", "data")
OUTPUT_PKL = os.path.join(DATA_DIR, "NOTEBOOK BOTANIC ANALYSES.pkl")


# ─────────────────────────────────────────────────────────────
# ÉTAPE 1 — Chargement et nettoyage des données clients
# ─────────────────────────────────────────────────────────────

def charger_clients(path: str) -> pd.DataFrame:
    """
    Charge et nettoie la table clients.

    Variables clés :
        IDCLIENT              — identifiant unique client
        SEXE                  — genre du client
        AGE_GROUP             — tranche d'âge (18-30, 31-45, 46-60, 61-100)
        ANCIENNETE_ADHESION   — ancienneté d'adhésion au programme fidélité
        MAGASIN               — code du magasin de rattachement

    Traitements appliqués :
        - Conversion IDCLIENT en string
        - Regroupement modalités ANCIENNETE_ADHESION_CAT
          (0-1 an, 1-2 ans, 2-3 ans → 0-3 ans)
        - Suppression des doublons
    """
    print("📂 Chargement des données clients...")
    df = pd.read_csv(path, sep='|', low_memory=False)
    print(f"   Shape initiale : {df.shape}")

    # Conversion des types
    df['IDCLIENT'] = df['IDCLIENT'].astype(str)

    # Regroupement des modalités d'ancienneté
    df['ANCIENNETE_ADHESION_CAT'] = df['ANCIENNETE_ADHESION_CAT'].replace({
        '0-1 an':  '0-3 ans',
        '1-2 ans': '0-3 ans',
        '2-3 ans': '0-3 ans'
    })

    # Suppression des doublons
    df = df.drop_duplicates()
    print(f"   Shape après nettoyage : {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# ÉTAPE 2 — Chargement et nettoyage des en-têtes de tickets
# ─────────────────────────────────────────────────────────────

def charger_entetes_tickets(path: str) -> pd.DataFrame:
    """
    Charge et nettoie la table des en-têtes de tickets.

    Variables clés :
        IDTICKET      — identifiant unique du ticket
        IDCLIENT      — clé de jointure avec la table clients
        TIC_DATE      — date de l'achat
        TIC_TOTALTTC  — montant total TTC du ticket
        MAG_CODE      — code du magasin

    Traitements appliqués :
        - Conversion IDTICKET, IDCLIENT en string
        - Nettoyage TIC_TOTALTTC (virgule → point, conversion float)
        - Formatage de la date (datetime → dd/mm/yyyy)
        - Suppression des valeurs aberrantes (méthode IQR)
        - Suppression des doublons
    """
    print("📂 Chargement des en-têtes de tickets...")
    df = pd.read_csv(path, sep='|', low_memory=False)
    print(f"   Shape initiale : {df.shape}")

    # Conversion des types
    df['IDTICKET'] = df['IDTICKET'].astype(str)
    df['IDCLIENT'] = df['IDCLIENT'].astype(str)

    # Nettoyage du montant TTC
    df['TIC_TOTALTTC'] = (df['TIC_TOTALTTC']
                          .str.replace(',', '.')
                          .str.strip('"')
                          .pipe(pd.to_numeric, errors='coerce'))

    # Formatage de la date
    df['TIC_DATE'] = pd.to_datetime(
        df['TIC_DATE'], format="%Y-%m-%d %H:%M:%S"
    ).dt.strftime("%d/%m/%Y")

    # Suppression des valeurs aberrantes (IQR sur TIC_TOTALTTC)
    Q1 = df['TIC_TOTALTTC'].quantile(0.25)
    Q3 = df['TIC_TOTALTTC'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[
        (df['TIC_TOTALTTC'] >= Q1 - 1.5 * IQR) &
        (df['TIC_TOTALTTC'] <= Q3 + 1.5 * IQR)
    ]

    # Suppression des doublons
    df = df.drop_duplicates()
    print(f"   Shape après nettoyage : {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# ÉTAPE 3 — Chargement et nettoyage des lignes de tickets
# ─────────────────────────────────────────────────────────────

def charger_lignes_tickets(path: str) -> pd.DataFrame:
    """
    Charge et nettoie la table des lignes de tickets.

    Variables clés :
        IDTICKET        — clé de jointure avec en-têtes tickets
        NUMLIGNETICKET  — numéro de ligne du ticket
        CODEARTICLE     — clé de jointure avec REF_ARTICLE
        QUANTITE        — quantité achetée
        TOTAL           — montant total de la ligne
        MONTANTREMISE   — montant de la remise
        MARGESORTIE     — marge sur la ligne

    Traitements appliqués :
        - Conversion IDTICKET, NUMLIGNETICKET en string
        - Conversion QUANTITE, TOTAL, MONTANTREMISE, MARGESORTIE en float
        - Suppression des valeurs aberrantes (IQR sur chaque variable numérique)
        - Suppression des doublons
    """
    print("📂 Chargement des lignes de tickets...")
    df = pd.read_csv(path, sep='|', low_memory=False)
    print(f"   Shape initiale : {df.shape}")

    # Conversion des types
    df['IDTICKET']       = df['IDTICKET'].astype(str)
    df['NUMLIGNETICKET'] = df['NUMLIGNETICKET'].astype(str)

    # Nettoyage des variables numériques
    for col in ['QUANTITE', 'TOTAL', 'MONTANTREMISE', 'MARGESORTIE']:
        df[col] = df[col].str.replace(',', '.').astype(float)

    # Suppression des valeurs aberrantes (IQR) sur chaque variable numérique
    for col in ['QUANTITE', 'TOTAL', 'MONTANTREMISE', 'MARGESORTIE']:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df  = df[
            (df[col] >= Q1 - 1.5 * IQR) &
            (df[col] <= Q3 + 1.5 * IQR)
        ]

    # Suppression des doublons
    df = df.drop_duplicates()
    print(f"   Shape après nettoyage : {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# ÉTAPE 4 — Chargement des référentiels
# ─────────────────────────────────────────────────────────────

def charger_ref_article(path: str) -> pd.DataFrame:
    """
    Charge le référentiel articles.
    Clé de jointure : CODEARTICLE
    """
    print("📂 Chargement REF_ARTICLE...")
    df = pd.read_csv(path, sep='|', low_memory=False)
    print(f"   {df.shape[0]:,} articles · {df.shape[1]} colonnes")
    return df


def charger_ref_magasin(path: str) -> pd.DataFrame:
    """
    Charge le référentiel magasins.
    Clé de jointure : CODESOCIETE (= MAG_CODE dans entetes_tickets)
    """
    print("📂 Chargement REF_MAGASIN...")
    df = pd.read_csv(path, sep='|', low_memory=False)
    print(f"   {df.shape[0]} magasins · {df.shape[1]} colonnes")
    return df


# ─────────────────────────────────────────────────────────────
# ÉTAPE 5 — Jointures et construction de la base analytique
# ─────────────────────────────────────────────────────────────

def construire_base_analytique(
    df_clients, df_entetes, df_lignes, df_article, df_magasin
) -> pd.DataFrame:
    """
    Construit la base analytique finale par jointures successives.

    Schéma des jointures :
    ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
    │   CLIENTS    │────▶│  ENTETES_TICKET  │────▶│  LIGNES_TICKET   │
    │  (IDCLIENT)  │     │  (IDCLIENT)      │     │  (IDTICKET)      │
    └──────────────┘     └──────────────────┘     └──────────────────┘
                                  │                        │
                                  ▼                        ▼
                         ┌──────────────┐        ┌──────────────────┐
                         │ REF_MAGASIN  │        │   REF_ARTICLE    │
                         │ (CODESOCIETE)│        │  (CODEARTICLE)   │
                         └──────────────┘        └──────────────────┘

    Agrégations créées :
        - FREQ_COMMANDE_2016        : nombre de commandes en 2016
        - TOTAL_ACHAT_2016          : montant total des achats 2016
        - NBRE_ARTICLE              : nombre d'articles achetés
        - NBRE_FAM_ARTICLE          : nombre de familles d'articles
        - TOTAL_REMISE              : total des remises obtenues
        - Variete_Panier            : diversité du panier
        - ANCIENNETE_1ERE_CDE_2016  : ancienneté de la 1ère commande
        - ANCIENNETE_DERNIERE_CDE   : ancienneté de la dernière commande
        - TEMPS_MOY_CDE_2016        : temps moyen entre commandes
    """
    print("\n🔗 Construction de la base analytique...")

    # Jointure 1 : Entêtes tickets ← Clients
    print("   Jointure 1 : Entêtes ← Clients")
    df = df_entetes.merge(
        df_clients,
        on='IDCLIENT',
        how='left'
    )

    # Jointure 2 : + Lignes tickets
    print("   Jointure 2 : + Lignes tickets")
    df = df.merge(
        df_lignes,
        on='IDTICKET',
        how='left'
    )

    # Jointure 3 : + Référentiel articles
    print("   Jointure 3 : + Référentiel articles")
    df = df.merge(
        df_article[['CODEARTICLE', 'CODEUNIVERS', 'CODEFAMILLE']],
        on='CODEARTICLE',
        how='left'
    )

    # Jointure 4 : + Référentiel magasins
    print("   Jointure 4 : + Référentiel magasins")
    df = df.merge(
        df_magasin.rename(columns={'CODESOCIETE': 'MAG_CODE'}),
        on='MAG_CODE',
        how='left'
    )

    print(f"   Base après jointures : {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# ÉTAPE 6 — Construction de la variable cible ATTRITION
# ─────────────────────────────────────────────────────────────

def construire_label_attrition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit la variable cible ATTRITION.

    Définition d'un client ATTRITIONNISTE :
        Condition 1 : FREQ_COMMANDE_2016 > 1
            → Le client a commandé plus d'une fois en 2016
              (client engagé qui a ensuite décroché)
        Condition 2 : Pas de commande en 2017
            → Le client n'a pas renouvelé ses achats l'année suivante

    Logique métier :
        Un client qui n'a jamais commandé n'est pas attritionniste
        (on ne peut pas perdre un client qu'on n'avait pas).
        C'est pourquoi on filtre sur FREQ_COMMANDE_2016 > 1.

    Résultat :
        ATTRITION = 1 (attritionniste)  : 14.5% des clients
        ATTRITION = 0 (non-attritionniste) : 85.5% des clients
        → Dataset déséquilibré → nécessite SMOTE
    """
    print("\n🎯 Construction du label ATTRITION...")

    df['DERNIERE_COMMANDE_2016'] = pd.to_datetime(
        df['DERNIERE_COMMANDE_2016']
    )

    condition_1 = df['FREQ_COMMANDE_2016'] > 1
    condition_2 = df['DERNIERE_COMMANDE_2016'].dt.year != 2017

    df['ATTRITION'] = (condition_1 & condition_2).astype(int)
    df['ATTRITION'] = df['ATTRITION'].map({
        0: 'non_attritionniste',
        1: 'attritionniste'
    })

    print(f"   Distribution ATTRITION :")
    print(df['ATTRITION'].value_counts(normalize=True).mul(100).round(1))
    return df


# ─────────────────────────────────────────────────────────────
# ÉTAPE 7 — Sélection des variables finales
# ─────────────────────────────────────────────────────────────

def selectionner_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sélectionne les variables retenues après analyse de corrélation.

    Variables supprimées :
        - MONTANT_TOTAL, MARGE, Nb_Univers, NBRE_COMMANDE_2016
          → fortement corrélées (corrélation > 0.8)
        - VILLE, LIBELLEREGIONCOMMERCIALE
          → corrélation catégorielle élevée (V de Cramér > 0.8)
        - PREMIERE_COMMANDE_2016, TIC_DATE, DATEREADHESION
          → variables de date redondantes
        - IDCLIENT, IDTICKET, CODEUNIVERS
          → identifiants non informatifs

    Variables retenues (20 variables explicatives) :
        Catégorielles : MAGASIN, SEXE, AGE_GROUP,
                        ANCIENNETE_ADHESION_CAT, ANCIENNETE_READ,
                        LIBELLEDEPARTEMENT, TYPE_UNIVERS
        Numériques    : TIC_TOTALTTC, FREQ_MAG, Quantite,
                        NBRE_ARTICLE, NBRE_FAM_ARTICLE,
                        TOTAL_REMISE, Variete_Panier,
                        UNIVERS_PREFERE, TOTAL_ACHAT_2016,
                        ANCIENNETE_1ERE_CDE_2016,
                        ANCIENNETE_DERNIERE_CDE_2016,
                        TEMPS_MOY_CDE_2016, FREQ_COMMANDE_2016
    """
    cols_a_supprimer = [
        'PREMIERE_COMMANDE_2016', 'TIC_DATE', 'DATEREADHESION',
        'IDCLIENT', 'IDTICKET', 'CODEUNIVERS',
        'MONTANT_TOTAL', 'MARGE', 'Nb_Univers', 'NBRE_COMMANDE_2016',
        'VILLE', 'LIBELLEREGIONCOMMERCIALE'
    ]
    cols_existantes = [c for c in cols_a_supprimer if c in df.columns]
    df = df.drop(columns=cols_existantes)
    print(f"\n✅ Variables finales : {df.shape[1]} colonnes")
    return df


# ─────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────

def run_pipeline(data_dir: str) -> pd.DataFrame:
    """
    Exécute le pipeline complet de préparation des données.

    Args:
        data_dir: chemin vers le dossier contenant les fichiers sources

    Returns:
        DataFrame analytique final prêt pour la modélisation
    """
    print("=" * 55)
    print("PIPELINE BOTANIC — PRÉPARATION DES DONNÉES")
    print("=" * 55)

    # Chargement des tables
    df_clients = charger_clients(
        os.path.join(data_dir, "CLIENTS_Botanic.csv"))
    df_entetes = charger_entetes_tickets(
        os.path.join(data_dir, "ENTETES_TICKET_V4.csv"))
    df_lignes = charger_lignes_tickets(
        os.path.join(data_dir, "LIGNES_TICKET_V4.csv"))
    df_article = charger_ref_article(
        os.path.join(data_dir, "REF_ARTICLE.CSV"))
    df_magasin = charger_ref_magasin(
        os.path.join(data_dir, "REF_MAGASIN.CSV"))

    # Construction de la base analytique
    df = construire_base_analytique(
        df_clients, df_entetes, df_lignes, df_article, df_magasin)

    # Construction du label
    df = construire_label_attrition(df)

    # Sélection des variables
    df = selectionner_variables(df)

    # Sauvegarde
    output = os.path.join(data_dir, "NOTEBOOK BOTANIC ANALYSES.pkl")
    df.to_pickle(output)
    print(f"\n💾 Base sauvegardée : {output}")
    print(f"   {df.shape[0]:,} lignes · {df.shape[1]} colonnes")
    print("=" * 55)

    return df


if __name__ == "__main__":
    run_pipeline(DATA_DIR)
