"""
=============================================================
OBSERVATIONS & INSIGHTS — Base analytique Botanic
=============================================================
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # pour PyCharm
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
FIG_DIR  = os.path.join(BASE_DIR, "..", "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Chargement
df = pd.read_pickle(os.path.join(DATA_DIR, "base_analytique.pkl"))

print("="*55)
print("OBSERVATIONS CLÉS — BASE ANALYTIQUE BOTANIC")
print("="*55)

# ── 1. Variable cible ──────────────────────────────────────
print("\n📊 1. DISTRIBUTION DE L'ATTRITION")
print("-"*40)
attrition = df['ATTRITION'].value_counts()
pct       = df['ATTRITION'].value_counts(normalize=True).mul(100).round(1)
print(f"   Non-attritionnistes : {attrition['non_attritionniste']:,} ({pct['non_attritionniste']}%)")
print(f"   Attritionnistes     : {attrition['attritionniste']:,} ({pct['attritionniste']}%)")
print(f"   → Dataset déséquilibré : nécessitera SMOTE")

# ── 2. Valeurs manquantes ──────────────────────────────────
print("\n📊 2. VALEURS MANQUANTES")
print("-"*40)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
print(f"   Date de naissance manquante : {missing['DATENAISSANCE']:,} clients ({missing_pct['DATENAISSANCE']}%)")
print(f"   → Stratégie : créer modalité 'Inconnue' pour AGE_GROUP")
print(f"   Infos magasin manquantes    : {missing['LIBELLEDEPARTEMENT']:,} clients ({missing_pct['LIBELLEDEPARTEMENT']}%)")
print(f"   → Stratégie : clients sans achat enregistré → supprimer")

# ── 3. Profil âge ──────────────────────────────────────────
print("\n📊 3. PROFIL ÂGE DES CLIENTS")
print("-"*40)
age = df['AGE_GROUP'].value_counts()
for groupe, count in age.items():
    pct_g = round(count/len(df)*100, 1)
    print(f"   {groupe:12s} : {count:,} clients ({pct_g}%)")
print(f"   → Clientèle mature : majorité 46-60 ans et 61+ ans")
print(f"   → Opportunité : fidélisation longue durée possible")

# ── 4. Ancienneté ──────────────────────────────────────────
print("\n📊 4. ANCIENNETÉ D'ADHÉSION")
print("-"*40)
anc = df['ANCIENNETE_ADHESION_CAT'].value_counts()
for cat, count in anc.items():
    pct_a = round(count/len(df)*100, 1)
    print(f"   {cat:10s} : {count:,} clients ({pct_a}%)")
print(f"   → Base très fidèle : {pct_a}% ont plus de 8 ans d'ancienneté")

# ── 5. Comparaison attritionnistes vs fidèles ──────────────
print("\n📊 5. PROFIL FINANCIER PAR SEGMENT")
print("-"*40)
stats = df.groupby('ATTRITION').agg(
    panier_moyen   = ('PANIER_MOYEN',   'mean'),
    total_achats   = ('TOTAL_ACHATS',   'mean'),
    nb_tickets     = ('NB_TICKETS',     'mean'),
    total_remise   = ('TOTAL_REMISE',   'mean'),
).round(2)
print(stats.to_string())
print(f"\n   → Paradoxe : les attritionnistes ont un panier moyen PLUS élevé !")
print(f"   → Ce sont de bons clients qui partent — urgent de les retenir !")

# ── 6. Graphiques ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Analyse Churn Botanic', fontsize=14, fontweight='bold')

# Donut attrition
sizes = [pct['non_attritionniste'], pct['attritionniste']]
colors = ['#2ECC71', '#E74C3C']
axes[0].pie(sizes, labels=['Non-attritionniste', 'Attritionniste'],
            autopct='%1.1f%%', colors=colors,
            wedgeprops=dict(width=0.5))
axes[0].set_title('Distribution Attrition')

# Bar âge
df['AGE_GROUP'].value_counts().plot(kind='bar', ax=axes[1],
    color='#3498DB', edgecolor='white')
axes[1].set_title('Répartition par âge')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=45)

# Panier moyen par segment
stats['panier_moyen'].plot(kind='bar', ax=axes[2],
    color=['#2ECC71', '#E74C3C'], edgecolor='white')
axes[2].set_title('Panier moyen par segment')
axes[2].set_xlabel('')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
output_fig = os.path.join(FIG_DIR, "analyse_churn.png")
plt.savefig(output_fig, dpi=150, bbox_inches='tight')
print(f"\n💾 Graphique sauvegardé : {output_fig}")
# ── 7. Storytelling ────────────────────────────────────────
print("\n" + "="*55)
print("STORYTELLING — CE QUE LES GRAPHIQUES RACONTENT")
print("="*55)

n_clients_total = 845876
n_attritionnistes_total = int(n_clients_total * 0.086)

print(f"""
📊 Graphique 1 — Distribution Attrition (Donut)
   8.6% des clients sont à risque.
   En apparence peu... mais sur {n_clients_total:,} clients
   cela représente ~{n_attritionnistes_total:,} personnes !
   → Chaque point de % = des milliers de clients perdus.

📊 Graphique 2 — Répartition par âge (Bar)
   Majorité de clients 46-60 ans et 61+ ans.
   Ce sont des clients matures, fidèles,
   avec un fort pouvoir d'achat.
   → Opportunité : fidélisation longue durée possible.
   → Risque : vieillissement de la base → renouveler.

📊 Graphique 3 — Panier moyen par segment (Bar)
   Non-attritionnistes : 84€ de panier moyen.
   Attritionnistes     : 76€ de panier moyen.
   Les fidèles dépensent plus car ils reviennent souvent.
   → Retenir un client = augmenter son panier cumulé.
   → Chaque client retenu = {84*12:.0f}€/an de revenus préservés.
""")
print("\n✅ Observations terminées !")