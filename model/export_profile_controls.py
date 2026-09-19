"""Exporta la media, por Customer_Profile del negocio, de variables que NO se usaron para segmentar.

Sirve para contrastar con datos (y no con intuición) por qué K-Means no separa los perfiles 1 y 5:
la app cita estas medias en la sección de limitaciones. Es un cálculo descriptivo con pandas; no toca
el modelo, ni los centroides, ni las etiquetas de cluster (por eso se ejecuta aparte de train.py:
volver a entrenar podría renumerar los clusters).

    py -3.10 model/export_profile_controls.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CONTROL_VARS = ["Coupons_Used", "Customer_Tenure", "Video_Games", "Marketing_Clicks", "Satisfaction", "Returns"]

df = pd.read_excel(ROOT / "data" / "clientes_mediamarkt.xlsx")
out = df.groupby("Customer_Profile")[CONTROL_VARS].mean().round(2)
out.to_csv(ROOT / "model" / "artifacts" / "profile_control_vars.csv")
print(out.to_string())
