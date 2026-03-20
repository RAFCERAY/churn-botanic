import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Churn Botanic — Prédiction Attrition",
    page_icon="🌿",
    layout="wide"
)

@st.cache_data
def load_and_train():
    with open("data/base_analytique.pkl", "rb") as f:
        df = pickle.load(f)

    df_clean = df.dropna(subset=["ATTRITION"])
    df_clean["ATTRITION_NUM"] = (df_clean["ATTRITION"] == "attritionniste").astype(int)

    features = ["NB_TICKETS", "TOTAL_ACHATS", "PANIER_MOYEN",
                "TOTAL_QUANTITE", "NB_ARTICLES_DISTINCTS", "TOTAL_REMISE",
                "AGE", "ANCIENNETE_ADHESION_ANS", "ANCIENNETE_DERNIERE_VISITE",
                "FREQ_ACHAT_ANNUELLE", "VIP"]

    df_model = df_clean[features + ["ATTRITION_NUM"]].dropna()
    X = df_model[features]
    y = df_model["ATTRITION_NUM"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_s, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_s))
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_s)[:, 1])
    roc_auc = auc(fpr, tpr)

    importance = pd.DataFrame({
        "feature": features,
        "importance": abs(model.coef_[0])
    }).sort_values("importance", ascending=False)

    return df_clean, X, y, acc, fpr, tpr, roc_auc, importance, model, scaler, features

df, X, y, acc, fpr, tpr, roc_auc, importance, model, scaler, features = load_and_train()

# ── Header ──────────────────────────────────────────────────
st.markdown("# 🌿 Churn Botanic — Prédiction d'Attrition Client")
st.markdown("**Identification des clients à risque · Régression Logistique · SMOTE**")
st.markdown("---")

tabs = st.tabs(["📊 Vue globale", "🤖 Modèle ML", "🔍 Prédiction client", "🗺️ Analyse régionale"])

# ══ TAB 1 ═══════════════════════════════════════════════════
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clients total", f"{len(df):,}")
    col2.metric("Attritionnistes", f"{(df['ATTRITION']=='attritionniste').sum():,}")
    col3.metric("Taux attrition", f"{(df['ATTRITION']=='attritionniste').mean()*100:.1f}%")
    col4.metric("Variables", f"{len(features)}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Répartition Attrition**")
        vc = df["ATTRITION"].value_counts().reset_index()
        fig = px.pie(vc, values="count", names="ATTRITION",
                     hole=0.45, template="plotly_white",
                     color_discrete_sequence=["#2ecc71", "#e74c3c"])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Panier moyen — Attritionnistes vs Non-attritionnistes**")
        fig = px.box(df, x="ATTRITION", y="PANIER_MOYEN",
                     color="ATTRITION", template="plotly_white",
                     color_discrete_map={"attritionniste": "#e74c3c",
                                         "non_attritionniste": "#2ecc71"})
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Distribution du nombre de tickets par statut**")
    fig = px.histogram(df, x="NB_TICKETS", color="ATTRITION",
                       barmode="overlay", template="plotly_white",
                       color_discrete_map={"attritionniste": "#e74c3c",
                                           "non_attritionniste": "#2ecc71"},
                       nbins=50)
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# ══ TAB 2 ═══════════════════════════════════════════════════
with tabs[1]:
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{acc*100:.1f}%")
    col2.metric("AUC-ROC", f"{roc_auc:.3f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Courbe ROC**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"Régression Logistique (AUC={roc_auc:.3f})",
                                 line=dict(color="#2C5F2D", width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                 name="Aléatoire",
                                 line=dict(color="gray", dash="dash")))
        fig.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Importance des variables**")
        fig = px.bar(importance, x="importance", y="feature",
                     orientation="h", template="plotly_white",
                     color="importance", color_continuous_scale="Greens")
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ══ TAB 3 ═══════════════════════════════════════════════════
with tabs[2]:
    st.markdown("**Prédire le risque d'attrition d'un client**")
    st.info("Remplis les informations du client pour obtenir une prédiction.")

    col1, col2, col3 = st.columns(3)
    with col1:
        nb_tickets = st.number_input("Nb tickets", 0, 500, 10)
        total_achats = st.number_input("Total achats (€)", 0.0, 10000.0, 200.0)
        panier_moyen = st.number_input("Panier moyen (€)", 0.0, 500.0, 30.0)
        total_quantite = st.number_input("Total quantité", 0, 1000, 20)
    with col2:
        nb_articles = st.number_input("Nb articles distincts", 0, 500, 15)
        total_remise = st.number_input("Total remise (€)", 0.0, 1000.0, 10.0)
        age = st.number_input("Âge", 18, 100, 45)
    with col3:
        anciennete = st.number_input("Ancienneté adhésion (ans)", 0, 20, 3)
        anciennete_visite = st.number_input("Ancienneté dernière visite (jours)", 0, 1000, 90)
        freq_achat = st.number_input("Fréquence achat annuelle", 0.0, 50.0, 3.0)
        vip = st.selectbox("VIP", [0, 1])

    if st.button("🔮 Prédire le risque", type="primary"):
        input_data = [[nb_tickets, total_achats, panier_moyen, total_quantite,
                       nb_articles, total_remise, age, anciennete,
                       anciennete_visite, freq_achat, vip]]
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        if pred == 1:
            st.error(f"⚠️ Client à RISQUE d'attrition — Probabilité : {proba[1]*100:.1f}%")
        else:
            st.success(f"✅ Client FIDÈLE — Probabilité de rester : {proba[0]*100:.1f}%")

# ══ TAB 4 ═══════════════════════════════════════════════════
with tabs[3]:
    st.markdown("**Taux d'attrition par région commerciale**")
    region_churn = df.groupby("LIBELLEREGIONCOMMERCIALE").apply(
        lambda x: (x["ATTRITION"] == "attritionniste").mean() * 100
    ).reset_index()
    region_churn.columns = ["region", "taux_attrition"]
    region_churn = region_churn.sort_values("taux_attrition", ascending=False)

    fig = px.bar(region_churn, x="region", y="taux_attrition",
                 template="plotly_white", color="taux_attrition",
                 color_continuous_scale="RdYlGn_r")
    fig.update_layout(height=400, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Attrition par univers préféré**")
    univers_churn = df.groupby("UNIVERS_PREFERE").apply(
        lambda x: (x["ATTRITION"] == "attritionniste").mean() * 100
    ).reset_index()
    univers_churn.columns = ["univers", "taux_attrition"]
    fig = px.bar(univers_churn.sort_values("taux_attrition", ascending=False),
                 x="univers", y="taux_attrition", template="plotly_white",
                 color="taux_attrition", color_continuous_scale="RdYlGn_r")
    fig.update_layout(height=350, xaxis_tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("*🌿 Churn Botanic · Prédiction Attrition · Rafika Cervera*")