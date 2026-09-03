"""
Replica la metodologia del notebook original (t-SNE + K-Means, k=4) y
guarda los artefactos que consume la app de Streamlit: coordenadas
t-SNE, perfiles de cluster, curva de codo/silhouette, crosstab de
validacion contra el perfil ya asignado, y los parametros del scaler
+ centroides (para poder predecir el cluster de un cliente hipotetico
en vivo en el Playground, sin depender de un pickle de sklearn).

Ejecutar una sola vez:
    py -3.10 model/train.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "clientes_mediamarkt.xlsx"
ARTIFACTS = ROOT / "model" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Age", "Annual_Income", "Total_Purchases", "Average_Ticket", "Total_Spending",
    "Loyalty_Points", "Website_Visits", "Days_Since_Last_Purchase", "Online_Purchases",
]
N_CLUSTERS = 4
RANDOM_STATE = 42


def main():
    print("Cargando dataset...")
    df = pd.read_excel(DATA_PATH)
    n_columns_original = df.shape[1]
    X = df[FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -- Correlacion entre features (multicolinealidad) -----------------------
    corr = df[FEATURES].corr().round(4)
    corr.to_csv(ARTIFACTS / "correlation.csv")

    # -- t-SNE (proyeccion 2D) --------------------------------------------------
    print("Ajustando t-SNE (puede tardar 1-3 minutos con ~6.500 filas)...")
    t0 = time.time()
    tsne = TSNE(n_components=2, perplexity=35, learning_rate="auto", init="pca",
                random_state=RANDOM_STATE)
    Xt = tsne.fit_transform(X_scaled)
    print(f"  t-SNE listo en {time.time() - t0:.1f}s")

    # Correlacion de cada eje t-SNE con las variables originales (para interpretarlos)
    tsne_df = pd.DataFrame(Xt, columns=["tSNE_1", "tSNE_2"], index=df.index)
    combined = pd.concat([X, tsne_df], axis=1)
    axis_corr = combined.corr()[["tSNE_1", "tSNE_2"]].drop(index=["tSNE_1", "tSNE_2"])
    axis_corr["importancia"] = axis_corr.abs().max(axis=1)
    axis_corr = axis_corr.sort_values("importancia", ascending=False).round(3)
    axis_corr.to_csv(ARTIFACTS / "tsne_axis_correlation.csv")

    # -- Eleccion de k: codo + silhouette ----------------------------------------
    print("Calculando codo y silhouette para k=2..8...")
    k_range = range(2, 9)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(silhouette_score(X_scaled, labels))})
        print(f"  k={k} inertia={rows[-1]['inertia']:.0f} silhouette={rows[-1]['silhouette']:.4f}")
    pd.DataFrame(rows).to_csv(ARTIFACTS / "elbow_silhouette.csv", index=False)

    # -- Modelo final K-Means (k=4) ------------------------------------------------
    print(f"Entrenando K-Means final (k={N_CLUSTERS})...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    df["KMeans_Profile"] = kmeans.fit_predict(X_scaled)
    silhouette_final = float(silhouette_score(X_scaled, df["KMeans_Profile"]))
    print(f"  Silhouette final: {silhouette_final:.4f}")

    # -- Guardar coordenadas t-SNE + labels para el mapa interactivo --------------
    plot_df = pd.DataFrame({
        "tSNE_1": Xt[:, 0], "tSNE_2": Xt[:, 1],
        "KMeans_Profile": df["KMeans_Profile"].values,
        "Customer_Profile": df["Customer_Profile"].values,
    })
    plot_df.to_csv(ARTIFACTS / "tsne_coords.csv", index=False)

    # -- Perfiles de cluster (valores reales) --------------------------------------
    profile = df.groupby("KMeans_Profile")[FEATURES].mean().round(2)
    profile.to_csv(ARTIFACTS / "cluster_profiles.csv")

    sizes = df["KMeans_Profile"].value_counts().sort_index()
    sizes_df = pd.DataFrame({"cluster": sizes.index, "n": sizes.values,
                              "pct": (sizes.values / len(df) * 100).round(1)})
    sizes_df.to_csv(ARTIFACTS / "cluster_sizes.csv", index=False)

    # -- Crosstab de validacion contra el perfil ya asignado -----------------------
    crosstab = pd.crosstab(df["KMeans_Profile"], df["Customer_Profile"])
    crosstab.to_csv(ARTIFACTS / "crosstab.csv")

    # -- Scaler + centroides (para predecir en vivo sin pickles de sklearn) --------
    with open(ARTIFACTS / "scaler_params.json", "w", encoding="utf-8") as f:
        json.dump({
            "features": FEATURES,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        }, f, ensure_ascii=False, indent=2)

    with open(ARTIFACTS / "centroids.json", "w", encoding="utf-8") as f:
        json.dump({
            "features": FEATURES,
            "centroids": kmeans.cluster_centers_.tolist(),  # k x 9, en espacio escalado
        }, f, ensure_ascii=False, indent=2)

    with open(ARTIFACTS / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_customers": int(len(df)),
            "n_columns_original": int(n_columns_original),
            "n_features_used": len(FEATURES),
            "silhouette_final": silhouette_final,
            "n_clusters": N_CLUSTERS,
            "feature_ranges": {
                feat: {"min": float(X[feat].min()), "max": float(X[feat].max()),
                       "mean": float(X[feat].mean()), "median": float(X[feat].median())}
                for feat in FEATURES
            },
        }, f, ensure_ascii=False, indent=2)

    # Histogramas: guardamos los datos crudos de las 9 features (dataset es
    # pequeno, 6.457 filas x 9 columnas ~ nada que optimizar) para que la app
    # pueda dibujar distribuciones sin tener que releer el Excel completo.
    df[FEATURES].to_csv(ARTIFACTS / "features_raw.csv", index=False)

    print("\nListo. Resumen:")
    print(f"  {len(df)} clientes, {len(FEATURES)} variables de comportamiento")
    print(f"  k=4, silhouette={silhouette_final:.4f}")
    for cluster, cantidad in sizes.items():
        print(f"  Cluster {cluster}: {cantidad} ({cantidad/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()
