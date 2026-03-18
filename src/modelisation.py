"""
=============================================================
ETAPE 4 - Modelisation ML
=============================================================
Objectif : Construire un modele predictif d'attrition
Pipeline : SMOTE -> RFE -> Regression Logistique -> Evaluation
=============================================================
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
FIG_DIR  = os.path.join(BASE_DIR, "..", "outputs", "figures")
MOD_DIR  = os.path.join(BASE_DIR, "..", "outputs", "models")
os.makedirs(MOD_DIR, exist_ok=True)

# 1. Chargement
print("="*55)
print("MODELISATION ML - CHURN BOTANIC")
print("="*55)

df = pd.read_pickle(os.path.join(DATA_DIR, "base_analytique.pkl"))
print(f"OK {df.shape[0]:,} clients · {df.shape[1]} colonnes")

# 2. Preparation des features
print("\nPreparation des features...")
df = df.dropna(subset=['TOTAL_ACHATS'])
df['AGE_GROUP'] = df['AGE_GROUP'].astype(str).fillna('Inconnue')

features = [
    'NB_TICKETS', 'TOTAL_ACHATS', 'PANIER_MOYEN',
    'TOTAL_QUANTITE', 'NB_ARTICLES_DISTINCTS', 'TOTAL_REMISE',
    'ANCIENNETE_ADHESION_ANS', 'ANCIENNETE_DERNIERE_VISITE',
    'VIP', 'AGE_GROUP', 'CIVILITE', 'UNIVERS_PREFERE'
]

X = df[features].copy()
y = df['ATTRITION']

for col in X.select_dtypes(include=['object', 'category', 'str']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

X = X.fillna(X.median())

print(f"Features : {list(X.columns)}")
print(f"Distribution cible :")
print(y.value_counts())

# 3. Split train/test
print("\nSplit train/test 70/30 stratifie...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"Train : {X_train.shape[0]:,} · Test : {X_test.shape[0]:,}")

# 4. Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 5. SMOTE
print("\nReeequilibrage avec SMOTE...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
print(f"Avant SMOTE : {pd.Series(y_train).value_counts().to_dict()}")
print(f"Apres SMOTE : {pd.Series(y_train_res).value_counts().to_dict()}")

# 6. Selection variables RFE
print("\nSelection variables avec RFE...")
model_rfe = LogisticRegression(solver='liblinear', class_weight='balanced')
rfe = RFE(model_rfe, n_features_to_select=8)
rfe.fit(X_train_res, y_train_res)
selected = X.columns[rfe.support_].tolist()
print(f"Variables selectionnees : {selected}")

X_train_sel = X_train_res[:, rfe.support_]
X_test_sel  = X_test_scaled[:, rfe.support_]

# 7. Entrainement modele final
print("\nEntrainement Regression Logistique...")
model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
model.fit(X_train_sel, y_train_res)

# 8. Evaluation
y_pred      = model.predict(X_test_sel)
y_pred_prob = model.predict_proba(X_test_sel)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc      = roc_auc_score(y_test, y_pred_prob)

print("\n" + "="*55)
print("RESULTATS DU MODELE")
print("="*55)
print(f"Accuracy  : {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"AUC-ROC   : {auc:.3f}")
print(f"\nRapport de classification :")
print(classification_report(y_test, y_pred))

# 9. Validation croisee
print("Validation croisee 5-folds...")
cv_scores = cross_val_score(model, X_train_sel, y_train_res, cv=5)
print(f"CV Score : {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

# 10. Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
vp = cm[1][1]
vn = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]

# 11. Storytelling
print("\n" + "="*55)
print("STORYTELLING - CE QUE LES GRAPHIQUES RACONTENT")
print("="*55)
print("Matrice de confusion :")
print(f"   {vp:,} attritionnistes correctement detectes")
print(f"   {vn:,} non-attritionnistes correctement identifies")
print(f"   {fp:,} faux positifs seulement - quasi parfait !")
print(f"   {fn:,} attritionnistes manques - a ameliorer")
print(f"\nCourbe ROC :")
print(f"   AUC = {auc:.3f} - modele quasi parfait !")
print(f"   Dans {auc*100:.1f}% des cas le modele distingue")
print(f"   correctement un attritionniste d un client fidele.")
print("\nPour Botanic :")
clients_detectes = int(72745 * (vp / (vp + fn)))
print(f"   Sur 72 745 clients a risque, {clients_detectes:,} detectes correctement.")
print(f"   Seulement {fp} alertes inutiles sur {vn+fp:,} clients fideles.")

# 12. Graphiques
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Evaluation du modele - Churn Botanic', fontweight='bold')

# Matrice de confusion
axes[0].imshow(cm, cmap='Blues')
axes[0].set_title('Matrice de confusion')
axes[0].set_xlabel('Predictions')
axes[0].set_ylabel('Reel')
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Non-attr.', 'Attr.'])
axes[0].set_yticklabels(['Non-attr.', 'Attr.'])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=12)

# Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob, pos_label='attritionniste')
axes[1].plot(fpr, tpr, color='#3498DB', lw=2, label=f'AUC = {auc:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_title('Courbe ROC')
axes[1].set_xlabel('Taux faux positifs')
axes[1].set_ylabel('Taux vrais positifs')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output_fig = os.path.join(FIG_DIR, "evaluation_modele.png")
plt.savefig(output_fig, dpi=150, bbox_inches='tight')
print(f"\nGraphique sauvegarde : {output_fig}")

# 13. Sauvegarde modele
joblib.dump(model,  os.path.join(MOD_DIR, "logistic_regression.pkl"))
joblib.dump(scaler, os.path.join(MOD_DIR, "scaler.pkl"))
print(f"Modele sauvegarde dans outputs/models/")
print("\nModelisation terminee !")