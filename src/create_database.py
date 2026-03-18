"""
=============================================================
ÉTAPE 1 — Création de la base SQLite
=============================================================
Objectif : Charger les 5 tables CSV dans une base SQLite
Avantage : Les jointures SQL seront 10x plus rapides
           qu'avec pandas sur 7M de lignes
=============================================================
"""
import sqlite3
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DB_PATH  = os.path.join(DATA_DIR, "botanic.db")

def creer_table_depuis_csv(conn, csv_path, table_name, sep='|'):
    """Charge un CSV dans une table SQLite."""
    print(f"   Chargement {table_name}...")
    df = pd.read_csv(csv_path, sep=sep, low_memory=False)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"   ✅ {table_name} : {count:,} lignes")
    return df

def main():
    print("="*50)
    print("CRÉATION BASE SQLITE — BOTANIC")
    print("="*50)

    conn = sqlite3.connect(DB_PATH)

    # Table 1 — Référentiel magasins
    creer_table_depuis_csv(
        conn,
        os.path.join(DATA_DIR, "REF_MAGASIN.CSV"),
        "ref_magasin"
    )

    # Table 2 — Référentiel articles
    creer_table_depuis_csv(
        conn,
        os.path.join(DATA_DIR, "REF_ARTICLE.CSV"),
        "ref_article"
    )

    # Les 3 autres tables seront ajoutées
    # quand tu auras CLIENTS, ENTETES_TICKET, LIGNES_TICKET
    print("\n⚠️  Tables manquantes :")
    print("   → CLIENTS_Botanic.csv")
    print("   → ENTETES_TICKET_V4.csv")
    print("   → LIGNES_TICKET_V4.csv")

    conn.close()
    print(f"\n💾 Base créée : {DB_PATH}")
    print("="*50)

if __name__ == "__main__":
    main()
    """
    =============================================================
    ÉTAPE 1 — Création de la base SQLite
    =============================================================
    Objectif : Charger les 5 tables CSV dans une base SQLite
    Avantage : Les jointures SQL sont 10x plus rapides
               qu'avec pandas sur des millions de lignes
    =============================================================
    """
    import sqlite3
    import pandas as pd
    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")
    DB_PATH = os.path.join(DATA_DIR, "botanic.db")


    def creer_table_depuis_csv(conn, csv_path, table_name, sep='|', chunksize=100000):
        """
        Charge un CSV dans SQLite par chunks.
        Avantage : ne charge pas tout en mémoire d'un coup.
        """
        print(f"   Chargement {table_name}...")
        first_chunk = True
        total = 0
        for chunk in pd.read_csv(csv_path, sep=sep, low_memory=False, chunksize=chunksize):
            chunk.to_sql(
                table_name, conn,
                if_exists='replace' if first_chunk else 'append',
                index=False
            )
            first_chunk = False
            total += len(chunk)
            print(f"   ... {total:,} lignes chargées", end='\r')
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"   ✅ {table_name} : {count:,} lignes")


    def main():
        print("=" * 55)
        print("CRÉATION BASE SQLITE — BOTANIC")
        print("=" * 55)

        conn = sqlite3.connect(DB_PATH)

        # Table 1 — Référentiel magasins
        creer_table_depuis_csv(conn, os.path.join(DATA_DIR, "REF_MAGASIN.CSV"), "ref_magasin")

        # Table 2 — Référentiel articles
        creer_table_depuis_csv(conn, os.path.join(DATA_DIR, "REF_ARTICLE.CSV"), "ref_article")

        # Table 3 — Clients
        creer_table_depuis_csv(conn, os.path.join(DATA_DIR, "CLIENT.CSV"), "clients")

        # Table 4 — En-têtes tickets
        creer_table_depuis_csv(conn, os.path.join(DATA_DIR, "ENTETES_TICKET_V4.CSV"), "entetes_ticket")

        # Table 5 — Lignes tickets
        creer_table_depuis_csv(conn, os.path.join(DATA_DIR, "LIGNES_TICKET_V4.CSV"), "lignes_ticket")

        # Création des index pour accélérer les jointures
        print("\n🔧 Création des index...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entetes_client ON entetes_ticket(IDCLIENT)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entetes_ticket ON entetes_ticket(IDTICKET)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lignes_ticket  ON lignes_ticket(IDTICKET)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_article        ON ref_article(CODEARTICLE)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_magasin        ON ref_magasin(CODESOCIETE)")
        print("   ✅ Index créés")

        # Résumé final
        print("\n📊 Résumé de la base :")
        tables = ['ref_magasin', 'ref_article', 'clients', 'entetes_ticket', 'lignes_ticket']
        for t in tables:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"   {t:25s} : {n:>12,} lignes")

        conn.close()
        print(f"\n💾 Base SQLite : {DB_PATH}")
        print("=" * 55)


    if __name__ == "__main__":
        main()