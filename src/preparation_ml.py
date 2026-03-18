"""
=============================================================
ÉTAPE 3 — Nettoyage pandas + Construction label ATTRITION
=============================================================
"""
import sqlite3
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DB_PATH  = os.path.join(DATA_DIR, "botanic.db")

def charger_base_analytique(conn):
    """Charge la base analytique depuis SQLite."""
    query = """
    SELECT
        c.IDCLIENT,
        c.CIVILITE,
        c.DATENAISSANCE,
        c.DATEDEBUTADHESION,
        c.VIP,
        c.MAGASIN                           as MAGASIN_CLIENT,
        m.LIBELLEDEPARTEMENT,
        m.LIBELLEREGIONCOMMERCIALE,
        COUNT(DISTINCT e.IDTICKET)          as NB_TICKETS,
        ROUND(SUM(e.TIC_TOTALTTC), 2)       as TOTAL_ACHATS,
        ROUND(AVG(e.TIC_TOTALTTC), 2)       as PANIER_MOYEN,
        MIN(e.TIC_DATE)                     as PREMIERE_VISITE,
        MAX(e.TIC_DATE)                     as DERNIERE_VISITE,
        SUM(l.QUANTITE)                     as TOTAL_QUANTITE,
        COUNT(DISTINCT l.IDARTICLE)         as NB_ARTICLES_DISTINCTS,
        ROUND(SUM(l.MONTANTREMISE), 2)      as TOTAL_REMISE,
        a.CODEUNIVERS                       as UNIVERS_PREFERE
    FROM clients c
    LEFT JOIN entetes_ticket e  ON c.IDCLIENT  = e.IDCLIENT
    LEFT JOIN lignes_ticket  l  ON e.IDTICKET  = l.IDTICKET
    LEFT JOIN ref_article    a  ON l.IDARTICLE = a.CODEARTICLE
    LEFT JOIN ref_magasin    m  ON e.MAG_CODE  = m.CODESOCIETE
    GROUP BY c.IDCLIENT
    LIMIT 100000
    """
    print("⏳ Chargement depuis SQLite...")
    df = pd.read_sql(query, conn)
    print(f"✅ {df.shape[0]:,} clients · {df.shape[1]} colonnes")
    return df

def nettoyer_et_enrichir(df):
    """Nettoyage et création de nouvelles variables."""
    print("\n" + "="*55)
    print("NETTOYAGE ET ENRICHISSEMENT")
    print("="*55)

    # 1. Conversion des dates
    df['DATENAISSANCE']     = pd.to_datetime(df['DATENAISSANCE'],    errors='coerce', dayfirst=True)
    df['DATEDEBUTADHESION'] = pd.to_datetime(df['DATEDEBUTADHESION'], errors='coerce', dayfirst=True)
    df['PREMIERE_VISITE']   = pd.to_datetime(df['PREMIERE_VISITE'],  errors='coerce')
    df['DERNIERE_VISITE']   = pd.to_datetime(df['DERNIERE_VISITE'],  errors='coerce')

    # 2. Calcul de l'âge
    annee_ref = 2016
    df['AGE'] = annee_ref - df['DATENAISSANCE'].dt.year
    df['AGE_GROUP'] = pd.cut(df['AGE'],
        bins=[0, 30, 45, 60, 120],
        labels=['18-30 ans', '31-45 ans', '46-60 ans', '61+ ans']
    )
    print(f"✅ AGE_GROUP créé")
    print(df['AGE_GROUP'].value_counts())

    # 3. Ancienneté adhésion
    date_ref = pd.Timestamp('2016-12-31')
    df['ANCIENNETE_ADHESION_ANS'] = (
        (date_ref - df['DATEDEBUTADHESION']).dt.days / 365
    ).round(1)
    df['ANCIENNETE_ADHESION_CAT'] = pd.cut(df['ANCIENNETE_ADHESION_ANS'],
        bins=[0, 3, 5, 8, 100],
        labels=['0-3 ans', '3-5 ans', '5-8 ans', '8 ans+']
    )
    print(f"\n✅ ANCIENNETE_ADHESION_CAT créé")
    print(df['ANCIENNETE_ADHESION_CAT'].value_counts())

    # 4. Ancienneté dernière commande (en jours)
    df['ANCIENNETE_DERNIERE_VISITE'] = (
        date_ref - df['DERNIERE_VISITE']
    ).dt.days

    # 5. Fréquence d'achat annuelle
    df['FREQ_ACHAT_ANNUELLE'] = df['NB_TICKETS']

    # 6. Valeurs manquantes
    print(f"\n📊 Valeurs manquantes :")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    return df

def construire_label_attrition(df):
    """
    Construit la variable cible ATTRITION.

    Définition métier :
    - Client ATTRITIONNISTE : a acheté plusieurs fois en 2016
      mais n'a pas acheté en 2017
    - Logique : on ne perd que les clients qu'on avait !
    """
    print("\n" + "="*55)
    print("CONSTRUCTION LABEL ATTRITION")
    print("="*55)

    # Condition 1 : client actif (plusieurs tickets)
    condition_1 = df['NB_TICKETS'] > 1

    # Condition 2 : dernière visite avant 2017
    df['DERNIERE_VISITE'] = pd.to_datetime(df['DERNIERE_VISITE'], errors='coerce')
    condition_2 = df['DERNIERE_VISITE'].dt.year < 2017

    # Construction du label
    df['ATTRITION'] = (condition_1 & condition_2).map({
        True: 'attritionniste',
        False: 'non_attritionniste'
    })

    print(f"\n✅ Distribution ATTRITION :")
    print(df['ATTRITION'].value_counts())
    print(df['ATTRITION'].value_counts(normalize=True).mul(100).round(1))

    return df

def main():
    conn = sqlite3.connect(DB_PATH)
    df = charger_base_analytique(conn)
    conn.close()

    df = nettoyer_et_enrichir(df)
    df = construire_label_attrition(df)

    # Sauvegarde
    output = os.path.join(DATA_DIR, "base_analytique.pkl")
    df.to_pickle(output)
    print(f"\n💾 Base sauvegardée : {output}")
    print(f"   {df.shape[0]:,} clients · {df.shape[1]} colonnes")

    return df

if __name__ == "__main__":
    df = main()