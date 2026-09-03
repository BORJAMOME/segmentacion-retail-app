"""Carga de datos y artefactos del modelo, con cache de Streamlit."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "model" / "artifacts"

FEATURES = [
    "Age", "Annual_Income", "Total_Purchases", "Average_Ticket", "Total_Spending",
    "Loyalty_Points", "Website_Visits", "Days_Since_Last_Purchase", "Online_Purchases",
]

FEATURE_LABELS = {
    "Age": "Edad",
    "Annual_Income": "Ingreso anual (€)",
    "Total_Purchases": "Compras totales",
    "Average_Ticket": "Ticket medio (€)",
    "Total_Spending": "Gasto total (€)",
    "Loyalty_Points": "Puntos de fidelidad",
    "Website_Visits": "Visitas a la web",
    "Days_Since_Last_Purchase": "Días desde la última compra",
    "Online_Purchases": "Compras online",
}

CLUSTER_META = {
    0: {"name": "Bajo valor y en riesgo", "color": "#C2412E"},
    1: {"name": "Digital, joven y frecuente", "color": "#4A628E"},
    2: {"name": "Premium", "color": "#6E7F5B"},
    3: {"name": "Valor medio-alto, de tienda", "color": "#273A5F"},
}


@st.cache_data(show_spinner=False)
def load_json(name: str):
    with open(ARTIFACTS / name, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / name, **kwargs)


def artifacts_ready() -> bool:
    return (ARTIFACTS / "centroids.json").exists() and (ARTIFACTS / "dataset_stats.json").exists()
