"""
=============================================================
ÉTAPE 2 — Jointures SQL
=============================================================
"""
import sqlite3
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DB_PATH  = os.path.join(DATA_DIR, "botanic.db")

def explorer_tables(conn):
    print("="*55)
    print("EXPLORATION DES TABLES")
    print("="*55)
    tables = ['ref_magasin','ref_article','clients','entetes_ticket','lignes_ticket']
    for table in tables:
        cols = pd.read_sql(f"PRAGMA table_info({table})", conn)
        print(f"\n📋 {table.upper()}")
        print(f"   Colonnes : {list(cols['name'])}")

def construire_base_analytique(conn):
    print("\n" + "="*55)
    print("CONSTRUCTION BASE ANALYTIQUE")
    print("="*55)

    query = """
    SELECT
        c.IDCLIENT,
        c.CIVILITE,
        c.DATENAISSANCE,
        c.DATEDEBUTADHESION,
        c.VIP,
        c.MAGASIN                           as MAGASIN_CLIENT,
        m.VILLE                             as VILLE_MAGASIN,
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
        ROUND(SUM(l.MARGESORTIE), 2)        as TOTAL_MARGE,
        ROUND(SUM(l.TOTAL), 2)              as TOTAL_LIGNES,
        a.CODEUNIVERS                       as UNIVERS_PREFERE
    FROM clients c
    LEFT JOIN entetes_ticket e  ON c.IDCLIENT  = e.IDCLIENT
    LEFT JOIN lignes_ticket  l  ON e.IDTICKET  = l.IDTICKET
    LEFT JOIN ref_article    a  ON l.IDARTICLE = a.CODEARTICLE
    LEFT JOIN ref_magasin    m  ON e.MAG_CODE  = m.CODESOCIETE
    GROUP BY c.IDCLIENT
    LIMIT 100000
    """

    print("⏳ Exécution de la requête SQL...")
    df = pd.read_sql(query, conn)
    print(f"✅ Base analytique : {df.shape[0]:,} clients · {df.shape[1]} colonnes")
    print(f"\nAperçu :")
    print(df.head(3).to_string())
    print(f"\nColonnes : {list(df.columns)}")
    return df

def main():
    conn = sqlite3.connect(DB_PATH)
    explorer_tables(conn)
    df = construire_base_analytique(conn)
    conn.close()
    return df

if __name__ == "__main__":
    df = main()