"""Figuras Plotly. Mismo sistema de color que el resto del portfolio:
azules marino para lo estructural, verde/rojo solo para lo semántico.
Los 4 clusters se colorean con significado, no arbitrariamente: rojo
para el segmento en riesgo, verde para el premium, dos tonos de azul
marino para los dos intermedios (ninguno es "malo" ni "bueno" en sí)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

INK = "#1D2638"
NAVY2 = "#273A5F"
NAVY3 = "#4A628E"
NAVY4 = "#B9C5D6"
MUTED = "#6B7280"
LINE = "#E3DFD5"
POSITIVE = "#6E7F5B"
NEGATIVE = "#C2412E"
SUPPORT = "#B8783C"
FONT = "Arial, Helvetica, sans-serif"

CLUSTER_COLOR = {0: NEGATIVE, 1: NAVY3, 2: POSITIVE, 3: NAVY2}


def _base_layout(fig, height=420, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=12.5),
        hovermode="closest",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                     font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, linecolor=LINE, tickfont=dict(color=MUTED)),
        yaxis=dict(showgrid=True, gridcolor=LINE, zeroline=False, tickfont=dict(color=MUTED)),
    )
    return fig


def histogram(series: pd.Series, title_x: str) -> go.Figure:
    fig = go.Figure(go.Histogram(x=series, marker_color=NAVY2, opacity=0.85, nbinsx=30))
    fig.update_xaxes(title_text=title_x)
    fig.update_yaxes(title_text="Clientes")
    return _base_layout(fig, height=300, legend=False)


def correlation_heatmap(corr: pd.DataFrame, labels: dict) -> go.Figure:
    cols = list(corr.columns)
    nice = [labels.get(c, c) for c in cols]
    z = corr.values
    fig = go.Figure(go.Heatmap(
        z=z, x=nice, y=nice, zmin=-1, zmax=1,
        colorscale=[[0, NEGATIVE], [0.5, "#FBFBFB"], [1, NAVY2]],
        text=np.round(z, 2), texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(thickness=12, outlinewidth=0),
    ))
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(family=FONT, color=INK, size=11))
    fig.update_xaxes(tickangle=-35)
    return fig


def tsne_scatter(df: pd.DataFrame, color_col: str, color_map: dict, label_map: dict,
                  title_prefix: str = "") -> go.Figure:
    fig = go.Figure()
    for val in sorted(df[color_col].unique()):
        mask = df[color_col] == val
        name = label_map.get(val, f"{title_prefix}{val}")
        fig.add_trace(go.Scatter(
            x=df.loc[mask, "tSNE_1"], y=df.loc[mask, "tSNE_2"], mode="markers",
            name=name, marker=dict(size=5, color=color_map.get(val, MUTED), opacity=0.6),
        ))
    fig.update_xaxes(title_text="tSNE_1 · valor económico")
    fig.update_yaxes(title_text="tSNE_2 · digital / recencia")
    return _base_layout(fig, height=460)


def elbow_silhouette(df: pd.DataFrame, chosen_k: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["k"], y=df["inertia"], name="Inercia (codo)",
                              line=dict(color=NAVY2, width=2.4), mode="lines+markers",
                              yaxis="y1"))
    fig.add_trace(go.Scatter(x=df["k"], y=df["silhouette"], name="Silhouette",
                              line=dict(color=SUPPORT, width=2.4, dash="dash"), mode="lines+markers",
                              yaxis="y2"))
    fig.add_vline(x=chosen_k, line_dash="dot", line_color=INK,
                  annotation_text=f"k elegido = {chosen_k}", annotation_font_color=INK)
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=12.5),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Número de clusters (k)", showgrid=False, linecolor=LINE, tickfont=dict(color=MUTED)),
        yaxis=dict(title="Inercia", showgrid=True, gridcolor=LINE, tickfont=dict(color=NAVY2)),
        yaxis2=dict(title="Silhouette", overlaying="y", side="right", showgrid=False, tickfont=dict(color=SUPPORT)),
    )
    return fig


def cluster_sizes(sizes_df: pd.DataFrame, names: dict) -> go.Figure:
    labels = [names.get(int(c), str(c)) for c in sizes_df["cluster"]]
    colors = [CLUSTER_COLOR.get(int(c), MUTED) for c in sizes_df["cluster"]]
    fig = go.Figure(go.Bar(
        x=sizes_df["n"], y=labels, orientation="h", marker_color=colors,
        text=[f"{n:,} ({p}%)".replace(",", ".") for n, p in zip(sizes_df["n"], sizes_df["pct"])],
        textposition="outside",
    ))
    fig.update_xaxes(title_text="Clientes")
    return _base_layout(fig, height=260, legend=False)


def cluster_profile_bars(profile: pd.DataFrame, feature: str, names: dict, labels: dict) -> go.Figure:
    """Una variable, un valor por cluster — comparación directa."""
    clusters = list(profile.index)
    values = profile[feature].values
    colors = [CLUSTER_COLOR.get(int(c), MUTED) for c in clusters]
    labs = [names.get(int(c), str(c)) for c in clusters]
    fig = go.Figure(go.Bar(x=labs, y=values, marker_color=colors,
                            text=[f"{v:,.0f}".replace(",", ".") for v in values], textposition="outside"))
    fig.update_yaxes(title_text=labels.get(feature, feature))
    return _base_layout(fig, height=300, legend=False)


def crosstab_heatmap(ct: pd.DataFrame, names: dict) -> go.Figure:
    y_labels = [f"Cluster {i} — {names.get(i, '')}" for i in ct.index]
    x_labels = [f"Perfil {c}" for c in ct.columns]
    fig = go.Figure(go.Heatmap(
        z=ct.values, x=x_labels, y=y_labels,
        colorscale=[[0, "#FBFBFB"], [1, NAVY2]],
        text=ct.values, texttemplate="%{text}", textfont=dict(size=11),
        colorbar=dict(thickness=12, outlinewidth=0, title="clientes"),
    ))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(family=FONT, color=INK, size=11))
    return fig


def playground_radar(user_scaled: dict, centroid_scaled: dict, features: list, labels: dict) -> go.Figure:
    """Compara el perfil escalado del cliente hipotético con el centroide
    del cluster asignado — un radar es la forma más directa de leer
    "en qué se parece y en qué no" sobre varias variables a la vez."""
    theta = [labels.get(f, f) for f in features] + [labels.get(features[0], features[0])]
    user_vals = [user_scaled[f] for f in features] + [user_scaled[features[0]]]
    cen_vals = [centroid_scaled[f] for f in features] + [centroid_scaled[features[0]]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=cen_vals, theta=theta, name="Centroide del cluster",
                                   line=dict(color=NAVY3, width=2), fill="toself",
                                   fillcolor="rgba(74,98,142,0.12)"))
    fig.add_trace(go.Scatterpolar(r=user_vals, theta=theta, name="Tu cliente",
                                   line=dict(color=SUPPORT, width=2.4), fill="toself",
                                   fillcolor="rgba(184,120,60,0.14)"))
    fig.update_layout(
        height=420, margin=dict(l=40, r=40, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(showticklabels=False, gridcolor=LINE),
                   angularaxis=dict(gridcolor=LINE, tickfont=dict(size=10, color=MUTED))),
    )
    return fig
