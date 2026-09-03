"""
Segmentación de Clientes Retail — K-Means + t-SNE
Case study interactivo en Streamlit: de la pregunta de negocio a la
decisión, pasando por los datos, el modelo y su explicabilidad.

Autor: Borja Mora Méndez
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from components import charts, ui
from utils.clustering import predict_cluster, scale_input
from utils.data_loader import (CLUSTER_META, FEATURES, FEATURE_LABELS, artifacts_ready,
                                load_csv, load_json)

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Segmentación de Clientes Retail · K-Means + t-SNE",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

with open(ROOT / "assets" / "style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if not artifacts_ready():
    st.error(
        "Los artefactos del modelo todavía no se han generado. "
        "Ejecuta `py -3.10 model/train.py` desde la raíz del proyecto y recarga esta página."
    )
    st.stop()

stats = load_json("dataset_stats.json")
scaler_params = load_json("scaler_params.json")
centroids_data = load_json("centroids.json")
correlation = load_csv("correlation.csv", index_col=0)
tsne_axis_corr = load_csv("tsne_axis_correlation.csv", index_col=0)
elbow_df = load_csv("elbow_silhouette.csv")
sizes_df = load_csv("cluster_sizes.csv")
profile_df = load_csv("cluster_profiles.csv", index_col=0)
crosstab_df = load_csv("crosstab.csv", index_col=0)
crosstab_df.columns = crosstab_df.columns.astype(int)
tsne_coords = load_csv("tsne_coords.csv")
features_raw = load_csv("features_raw.csv")

CLUSTER_NAMES = {k: v["name"] for k, v in CLUSTER_META.items()}
CLUSTER_COLORS = {k: v["color"] for k, v in CLUSTER_META.items()}
n_customers = stats["n_customers"]


def eur(value: float, sign: bool = False) -> str:
    spec = f"{value:+,.0f}" if sign else f"{value:,.0f}"
    return spec.replace(",", ".") + " €"


ui.nav()
ui.install_smooth_scroll()

# ============================================================ HERO ==
n_fmt = f"{n_customers:,}".replace(",", ".")
st.markdown(
    f"""
    <div id="top" class="hero-wrap">
      <p class="hero-kicker">Machine Learning Case Study · Clustering</p>
      <h1 class="hero-title">Tu equipo de marketing ya sabe quiénes son tus mejores clientes. ¿Puede un algoritmo llegar a la misma conclusión <em>sin que nadie se lo diga</em>?</h1>
      <p class="hero-sub">Un modelo de segmentación no supervisada agrupó a {n_fmt} clientes de una cadena
      de electrónica de consumo usando solo su comportamiento de compra — sin ver nunca la etiqueta de
      perfil que el negocio ya les tenía asignada. El resultado: casi recupera él solo al mismo cliente
      premium que marketing ya conocía.</p>
      <div class="hero-meta">
        <span class="hero-pill">Borja Mora Méndez</span>
        <span class="hero-pill">Python · scikit-learn (K-Means + t-SNE)</span>
        <span class="hero-pill">Streamlit</span>
        <span class="hero-pill">{n_fmt} clientes</span>
      </div>
      <div class="hero-scroll-row">
        <a href="#contexto" class="hero-scroll">explorar el caso &#8595;</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================ CONTEXTO ==
ui.section_open("contexto")
ui.eyebrow("Contexto")
ui.h2("El problema")
ui.lead(
    "Una cadena de electrónica de consumo trata a toda su base de clientes por igual: las mismas ofertas, "
    "los mismos emails, el mismo descuento genérico. El equipo de marketing sabe que no todos compran "
    "igual — pero con miles de clientes y decenas de variables, no hay forma humana de separarlos a mano."
)
ui.kpi_grid([
    {"num": n_fmt, "label": "clientes"},
    {"num": f"{stats['n_columns_original']}", "label": "variables originales"},
    {"num": f"{stats['n_features_used']}", "label": "variables de comportamiento usadas"},
    {"num": f"{stats['n_clusters']}", "label": "segmentos encontrados"},
])
st.write("")
ui.question_block(
    "La pregunta de negocio",
    '¿Existen grupos naturales de clientes según su comportamiento de compra, '
    '<span class="accent">y se pueden explicar de un vistazo</span>?',
    "No se trata de inventar categorías de marketing sobre el papel: se trata de dejar que el propio "
    "comportamiento de compra — gasto, frecuencia, canal, antigüedad — revele si esos grupos ya existen.",
)
ui.section_close()

# ============================================================ DATOS ==
ui.section_open("datos")
ui.eyebrow("Materia prima")
ui.h2("Los datos")
ui.lead(
    f"{n_fmt} clientes reales, {stats['n_columns_original']} variables por cliente: demografía, gasto, "
    "canal, categorías de producto e interacción con marketing. Para no arrastrar ruido al modelo, se usan solo "
    "9 variables de comportamiento y valor — el mismo criterio RFM (Recencia, Frecuencia, valor Monetario) "
    "ampliado con canal digital."
)
ui.kpi_grid([
    {"num": n_fmt, "label": "clientes"},
    {"num": "9", "label": "variables de comportamiento"},
    {"num": "1", "label": "perfil ya asignado, reservado como control"},
    {"num": "0", "label": "valores nulos"},
])
ui.pipeline(["Datos crudos", "Escalado (StandardScaler)", "t-SNE (verificar estructura)",
             "Elegir k (codo + silhouette)", "K-Means", "Validar contra el perfil dado"])

ui.eyebrow("Las 9 variables", muted=True)
cols = st.columns(3)
feature_groups = [
    ("Valor económico", ["Total_Spending", "Average_Ticket", "Annual_Income"]),
    ("Frecuencia y fidelidad", ["Total_Purchases", "Loyalty_Points", "Age"]),
    ("Canal y actividad", ["Online_Purchases", "Website_Visits", "Days_Since_Last_Purchase"]),
]
for c, (group_name, feats) in zip(cols, feature_groups):
    with c:
        items = "".join(f"<li>{FEATURE_LABELS[f]}</li>" for f in feats)
        st.markdown(
            f'<div class="co-card" style="height:100%;">'
            f'<div class="kpi-label" style="text-transform:uppercase; letter-spacing:.08em; font-size:.7rem; '
            f'font-weight:700; color:var(--label); margin-bottom:.6rem;">{group_name}</div>'
            f'<ul class="limit-list" style="margin:0;">{items}</ul></div>',
            unsafe_allow_html=True,
        )
st.write("")
ui.finding(
    "El dataset trae un campo <b>Customer_Profile</b> (1 a 5) ya asignado por el negocio. No se le da al "
    "modelo — se reserva aparte y se usa solo al final, como control externo, para comprobar si K-Means "
    "llega a conclusiones parecidas sin haberlo visto nunca."
)
ui.section_close()

# ============================================================ EXPLORACIÓN ==
ui.section_open("exploracion")
ui.eyebrow("Antes de modelar")
ui.h2("¿Qué nos dicen los datos?")
ui.lead("Dos preguntas antes de tocar ningún algoritmo: ¿cómo se distribuye el gasto?, "
        "¿hay variables que en el fondo miden lo mismo?")

ui.h3("Distribución del gasto total")
st.plotly_chart(charts.histogram(features_raw["Total_Spending"], "Gasto total (€)"),
                 use_container_width=True, config={"displayModeBar": False})
mean_spend = features_raw["Total_Spending"].mean()
median_spend = features_raw["Total_Spending"].median()
ui.finding(
    f"La media de gasto ({eur(mean_spend)}) supera claramente a la mediana ({eur(median_spend)}): "
    "hay un grupo pequeño de clientes que gasta mucho más que el resto, estirando la distribución hacia "
    "la derecha. Esa cola es, como se verá más abajo, el segmento premium."
)

ui.h3("¿Hay variables que miden lo mismo dos veces?")
st.plotly_chart(charts.correlation_heatmap(correlation, FEATURE_LABELS), use_container_width=True,
                 config={"displayModeBar": False})
ui.finding(
    "Hay multicolinealidad fuerte: <b>Gasto total</b> y <b>Puntos de fidelidad</b> correlacionan al 0.99 "
    "(los puntos se acumulan proporcionalmente al gasto — son casi la misma información dos veces), y "
    "<b>Ticket medio</b> con <b>Ingreso anual</b> al 0.95. K-Means no exige variables independientes, pero "
    "explica por qué varias métricas se moverán juntas al segmentar."
)
ui.section_close()

# ============================================================ METODOLOGÍA ==
ui.section_open("metodologia")
ui.eyebrow("Cómo se llegó al modelo")
ui.h2("El camino hasta el modelo")
ui.lead(
    "Segmentar sin supervisión tiene una trampa: el algoritmo siempre encuentra grupos, aunque no exista "
    "ninguna estructura real. Este es el camino para no caer en ella."
)
ui.story_steps([
    ("Escalamos los datos",
     "K-Means mide distancias. Sin estandarizar, el ingreso anual (rango 18.000–114.000) aplastaría a "
     "variables como la edad, que aporta información real en una escala mucho más pequeña."),
    ("Comprobamos que había estructura real",
     "Antes de forzar ningún número de grupos, proyectamos las 9 variables en 2D con t-SNE. Si no hay "
     "regiones separables en ese mapa, no tiene sentido segmentar — solo estaríamos troceando ruido."),
    ("Probamos varios números de grupos",
     "Con el método del codo y el silhouette score comparamos k=2 a k=8. El mejor silhouette aislado no "
     "siempre es el más útil para el negocio — hubo que decidir, no solo leer un gráfico."),
    ("Entrenamos K-Means y contrastamos con la realidad",
     "Con k=4 ya elegido, entrenamos el modelo final y lo comparamos contra el perfil que el negocio ya "
     "tenía asignado — sin haberlo visto nunca durante el entrenamiento."),
])

with st.expander("Para quien quiera el detalle técnico — t-SNE, perplexity, y por qué k=4"):
    st.markdown(
        "**t-SNE** (*t-distributed Stochastic Neighbor Embedding*) proyecta las 9 variables en 2D "
        "conservando las distancias *locales* entre puntos cercanos, no la varianza global (a diferencia de "
        "PCA) — suele producir mapas más legibles para detectar agrupaciones. Se usó `perplexity=35` "
        "(dentro del rango recomendado 5–50, escalado a los ~6.500 clientes) e `init=\"pca\"`, que estabiliza "
        "el resultado frente a una inicialización aleatoria.\n\n"
        f"El silhouette más alto se da en k=2 ({elbow_df.loc[elbow_df['k']==2,'silhouette'].values[0]:.2f}), "
        "pero ese resultado es demasiado grueso para uso comercial — separaría solo \"gasta poco\" de "
        f"\"gasta mucho\". A partir de k=3 el silhouette se estabiliza "
        f"(k=3: {elbow_df.loc[elbow_df['k']==3,'silhouette'].values[0]:.3f}, "
        f"k=4: {elbow_df.loc[elbow_df['k']==4,'silhouette'].values[0]:.3f}, prácticamente empatados) y baja "
        "progresivamente después. Se eligió k=4 porque, sin perder apenas calidad de separación frente a "
        "k=3, da al equipo de marketing cuatro perfiles accionables en vez de tres."
    )

st.write("")
ui.h3("t-SNE — ¿hay estructura real antes de segmentar?")
profile_values = sorted(tsne_coords["Customer_Profile"].unique())
profile_palette = ["#B9C5D6", "#4A628E", "#6E7F5B", "#B8783C", "#C2412E"]
profile_colors = dict(zip(profile_values, profile_palette))
profile_labels = {p: f"Perfil {p} (dado)" for p in profile_values}
st.plotly_chart(
    charts.tsne_scatter(tsne_coords, "Customer_Profile", profile_colors, profile_labels),
    use_container_width=True, config={"displayModeBar": False},
)
ui.finding(
    "Al colorear por el perfil que ya venía en los datos (sin usarlo para entrenar nada), se ven zonas del "
    "mapa claramente dominadas por un único color. Es la señal que buscábamos: hay grupos reales que "
    "descubrir, no solo ruido — el paso siguiente (K-Means) tiene sentido."
)

ui.h3("¿Qué representa cada eje del mapa?")
top_axis1 = tsne_axis_corr["tSNE_1"].abs().sort_values(ascending=False).index[0]
ui.body(
    f"Las variables que más correlacionan con el eje horizontal (tSNE_1) son "
    f"<b>{FEATURE_LABELS['Total_Spending']}</b> ({tsne_axis_corr.loc['Total_Spending','tSNE_1']:.2f}), "
    f"<b>{FEATURE_LABELS['Loyalty_Points']}</b> ({tsne_axis_corr.loc['Loyalty_Points','tSNE_1']:.2f}) y "
    f"<b>{FEATURE_LABELS['Average_Ticket']}</b> ({tsne_axis_corr.loc['Average_Ticket','tSNE_1']:.2f}): "
    "ese eje ordena a los clientes de menor a mayor <b>valor económico</b>. El eje vertical (tSNE_2) está "
    f"dominado por <b>{FEATURE_LABELS['Online_Purchases']}</b> "
    f"({tsne_axis_corr.loc['Online_Purchases','tSNE_2']:.2f}) y "
    f"<b>{FEATURE_LABELS['Days_Since_Last_Purchase']}</b> "
    f"({tsne_axis_corr.loc['Days_Since_Last_Purchase','tSNE_2']:.2f}): separa al cliente digital y activo del "
    "que compra poco por internet y lleva tiempo sin volver. El mapa resume 9 variables en dos preguntas: "
    "<i>¿cuánto vale este cliente?</i> y <i>¿qué tan digital y activo es?</i>"
)

st.write("")
ui.h3("Eligiendo k: codo y silhouette")
st.plotly_chart(charts.elbow_silhouette(elbow_df, stats["n_clusters"]), use_container_width=True,
                 config={"displayModeBar": False})
ui.finding(
    f"El silhouette score más alto se da en k=2, pero es demasiado grueso para el negocio. Con k=3 y k=4 "
    "prácticamente empatados, se eligió <b>k=4</b>: misma calidad de separación, un perfil accionable más."
)
ui.section_close()

# ============================================================ MODELO ==
ui.section_open("modelo")
ui.eyebrow("¿Cómo intenta resolverlo?")
ui.h2("K-Means (k=4)")
ui.lead(
    f"Con la estructura confirmada y k=4 decidido, el modelo final agrupa a los {n_fmt} clientes en "
    "4 segmentos según sus 9 variables de comportamiento — sin ver nunca el perfil que el negocio ya tenía "
    "asignado."
)
m1, m2 = st.columns(2)
with m1:
    st.metric("Silhouette score (k=4)", f"{stats['silhouette_final']:.3f}")
with m2:
    st.metric("Clientes segmentados", n_fmt)

with st.expander("¿Qué es el silhouette score?"):
    st.markdown(
        "Mide, para cada cliente, si está más cerca de los compañeros de su propio grupo que de los del "
        "grupo más próximo — va de -1 (mal asignado) a 1 (perfectamente separado). Un valor alrededor de "
        "0.3 en datos de comportamiento real (no sintético) es razonable: los clientes no caen en cajas "
        "perfectamente separadas, se mueven en un continuo, y el modelo lo refleja honestamente."
    )

st.write("")
ui.h3("Tamaño de cada segmento")
st.plotly_chart(charts.cluster_sizes(sizes_df, CLUSTER_NAMES), use_container_width=True,
                 config={"displayModeBar": False})

ui.h3("Los 4 clusters, proyectados en el mapa t-SNE")
st.plotly_chart(
    charts.tsne_scatter(tsne_coords, "KMeans_Profile", CLUSTER_COLORS,
                         {k: f"Cluster {k} — {v}" for k, v in CLUSTER_NAMES.items()}),
    use_container_width=True, config={"displayModeBar": False},
)
ui.finding(
    "Los cuatro clusters ocupan regiones bien diferenciadas del mapa, coherentes con los ejes ya "
    "interpretados: el premium se concentra en la zona de mayor valor económico, y los otros dos se separan "
    "sobre todo en el eje digital/recencia. Que K-Means y t-SNE — dos algoritmos con lógicas distintas — "
    "lleguen a fronteras parecidas confirma que la segmentación responde a una estructura real, no es un "
    "artefacto del método."
)
ui.section_close()

# ============================================================ EXPLICABILIDAD ==
ui.section_open("explicabilidad")
ui.eyebrow("¿Por qué estos 4 grupos?")
ui.h2("Los 4 perfiles")
ui.lead("Cada cluster tiene un comportamiento de compra distinto y reconocible — así es como lo vería "
        "un responsable de marketing, en lenguaje llano, no en coordenadas.")

CLUSTER_TEXT = {
    2: ("El más pequeño y, con diferencia, el de mayor valor. Compran online y en tienda casi por igual, "
        "y son los más activos: la recencia media más baja de los cuatro grupos."),
    3: ("Gasto medio-alto, pero el grupo menos digital: apenas la mitad de sus compras son online. "
        "Compran en tienda física más que ningún otro segmento."),
    1: ("El más numeroso. Los más jóvenes, casi todas sus compras son online, pero el ticket medio es "
        "el más bajo de los cuatro — compran mucho, pero barato."),
    0: ("El gasto más bajo, la actividad web más baja y, sobre todo, el doble de días desde la última "
        "compra que el segmento premium — la señal de alarma más clara del análisis."),
}
order = [2, 3, 1, 0]
cols_p = st.columns(4)
for c, cid in zip(cols_p, order):
    with c:
        row = profile_df.loc[cid]
        size_row = sizes_df[sizes_df["cluster"] == cid].iloc[0]
        size_fmt = f'{int(size_row["n"]):,}'.replace(",", ".")
        st.markdown(
            f'<div class="co-card" style="height:100%;">'
            f'<div class="kpi-num" style="font-size:1.5rem; color:{CLUSTER_COLORS[cid]};">Cluster {cid}</div>'
            f'<div class="kpi-label" style="font-weight:700; color:var(--ink); margin:.2rem 0 .6rem;">{CLUSTER_NAMES[cid]}</div>'
            f'<div class="kpi-label">{size_fmt} clientes ({size_row["pct"]}%)</div>'
            f'<div class="kpi-label" style="margin-top:.6rem;">{CLUSTER_TEXT[cid]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.write("")
ui.h3("Comparación directa: gasto total por cluster")
st.plotly_chart(charts.cluster_profile_bars(profile_df, "Total_Spending", CLUSTER_NAMES, FEATURE_LABELS),
                 use_container_width=True, config={"displayModeBar": False})
ui.finding(
    f"El cluster premium gasta de media {eur(profile_df.loc[2,'Total_Spending'])} — "
    f"{profile_df.loc[2,'Total_Spending']/profile_df.loc[0,'Total_Spending']:.1f} veces más que el cluster "
    "en riesgo. No es un matiz: son negocios distintos dentro del mismo negocio."
)
ui.body(
    "Importante: los perfiles describen promedios de grupo, no reglas fijas — dentro de cada cluster hay "
    "variación individual. Sirven para dirigir estrategia, no para juzgar a un cliente concreto."
)
ui.section_close()

# ============================================================ PLAYGROUND ==
ui.section_open("playground")
ui.eyebrow("Pruébalo tú mismo")
ui.h2("Playground — ¿en qué segmento caería este cliente?")
ui.lead(
    "Ajusta el comportamiento de un cliente hipotético y el modelo predice, en vivo, a qué cluster "
    "pertenecería — usando el mismo cálculo que K-Means: escalar y buscar el centroide más cercano."
)

pg_left, pg_right = st.columns([1, 1.2], gap="large")
with pg_left:
    st.markdown("**Valor económico**")
    total_spending = st.slider("Gasto total (€)", 50, 27000, 3400, step=50)
    average_ticket = st.slider("Ticket medio (€)", 30, 830, 210, step=5)
    annual_income = st.slider("Ingreso anual (€)", 18000, 115000, 40000, step=500)
    st.markdown("**Frecuencia y fidelidad**")
    total_purchases = st.slider("Compras totales", 1, 42, 17)
    loyalty_points = st.slider("Puntos de fidelidad", 0, 69600, 9000, step=100)
    age = st.slider("Edad", 18, 75, 39)
    st.markdown("**Canal y actividad**")
    online_purchases = st.slider("Compras online", 0, 31, 12)
    website_visits = st.slider("Visitas a la web", 5, 296, 122)
    days_since = st.slider("Días desde la última compra", 1, 466, 190)

user_values = {
    "Total_Spending": total_spending, "Average_Ticket": average_ticket, "Annual_Income": annual_income,
    "Total_Purchases": total_purchases, "Loyalty_Points": loyalty_points, "Age": age,
    "Online_Purchases": online_purchases, "Website_Visits": website_visits,
    "Days_Since_Last_Purchase": days_since,
}
cluster_id, dists = predict_cluster(user_values, FEATURES, scaler_params["mean"], scaler_params["scale"],
                                     centroids_data["centroids"])
user_scaled_arr = scale_input(user_values, FEATURES, scaler_params["mean"], scaler_params["scale"])
user_scaled = dict(zip(FEATURES, user_scaled_arr))
centroid_scaled = dict(zip(FEATURES, centroids_data["centroids"][cluster_id]))

with pg_right:
    st.markdown(
        f'<div class="co-card" style="text-align:center; padding:1.8rem 1rem;">'
        f'<div class="kpi-label">Este cliente encaja en</div>'
        f'<div class="kpi-num" style="font-size:2rem; color:{CLUSTER_COLORS[cluster_id]}; margin:.3rem 0;">'
        f'Cluster {cluster_id} — {CLUSTER_NAMES[cluster_id]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.plotly_chart(
        charts.playground_radar(user_scaled, centroid_scaled, FEATURES, FEATURE_LABELS),
        use_container_width=True, config={"displayModeBar": False},
    )

ui.h3("¿Por qué este cluster?")
sorted_dists = sorted(zip(CLUSTER_NAMES.keys(), dists), key=lambda x: x[1])
runner_up_id, runner_up_dist = sorted_dists[1]
ui.finding(
    f"De los 4 centroides, el más cercano en el espacio escalado es el del <b>Cluster {cluster_id} "
    f"({CLUSTER_NAMES[cluster_id]})</b>. El segundo más próximo es el Cluster {runner_up_id} "
    f"({CLUSTER_NAMES[runner_up_id]}) — cuanto más parecidas sean las dos distancias, más "
    "\"fronterizo\" es este cliente entre ambos perfiles, igual que ocurre con clientes reales."
)
ui.section_close()

# ============================================================ RESULTADOS ==
ui.section_open("resultados")
ui.eyebrow("¿Funciona de verdad?")
ui.h2("Resultados")
ui.h3("¿Coincide con el perfil que el negocio ya tenía?")
ui.body(
    "El dataset traía un campo <b>Customer_Profile</b> (1 a 5) que nunca se usó para entrenar. Se compara "
    "ahora, solo como control, si los 4 clusters de K-Means reconstruyen esa segmentación original."
)
st.plotly_chart(charts.crosstab_heatmap(crosstab_df, CLUSTER_NAMES), use_container_width=True,
                 config={"displayModeBar": False})

premium_match = crosstab_df.loc[2, 2]
premium_total = int(sizes_df.loc[sizes_df["cluster"] == 2, "n"].values[0])
premium_pct = premium_match / premium_total * 100
ui.finding(
    f"El cluster premium recupera casi exactamente el perfil 2 original: <b>{int(premium_match)} de sus "
    f"{premium_total} clientes ({premium_pct:.1f}%)</b> pertenecen a ese perfil. K-Means, sin conocer la "
    "etiqueta, aísla solo con comportamiento el mismo grupo que el negocio ya identificaba como distinto."
)
ui.finding(
    "El cluster 1, en cambio, no distingue entre los perfiles 1 y 5 originales: los reparte casi al 50%. "
    "Esto no es un fallo del modelo — es una señal honesta de que la diferencia entre esos dos perfiles no "
    "está en las 9 variables de comportamiento usadas, sino probablemente en otras (categoría de producto, "
    "satisfacción, devoluciones) que se dejaron fuera a propósito."
)
ui.section_close()

# ============================================================ IMPACTO ==
ui.section_open("impacto", tight=True)
ui.impact_banner(
    f'El <span class="accent-pos">15% de clientes premium</span> genera el gasto medio '
    f'<span class="accent-neg">{profile_df.loc[2,"Total_Spending"]/profile_df.loc[0,"Total_Spending"]:.0f} veces mayor</span> '
    'que el segmento en riesgo — sin haberlos etiquetado nunca a mano.',
    quote='"El modelo no inventó los grupos: encontró, solo con comportamiento, casi el mismo segmento premium que el negocio ya sabía que existía."',
)
ui.section_close()

# ============================================================ INSIGHTS ==
ui.section_open("insights")
ui.eyebrow("De los resultados a las conclusiones")
ui.h2("Insights")
ui.insight_card("01", "El comportamiento por sí solo ya explica el segmento de mayor valor",
                 f"Con solo 9 variables (gasto, frecuencia, canal, recencia), K-Means recupera el "
                 f"{premium_pct:.1f}% del segmento premium que el negocio ya tenía identificado.",
                 "Una segmentación por comportamiento es viable incluso sin las variables más ricas del CRM.")
cluster0_n_fmt = f"{int(sizes_df.loc[sizes_df['cluster']==0,'n'].values[0]):,}".replace(",", ".")
cluster0_pct = sizes_df.loc[sizes_df['cluster']==0, 'pct'].values[0]
cluster0_recency = profile_df.loc[0, 'Days_Since_Last_Purchase']
ui.insight_card("02", "El segmento en riesgo es el más numeroso de los dos extremos",
                 f"El cluster 0 (bajo valor y en riesgo) son {cluster0_n_fmt} clientes ({cluster0_pct}%), "
                 f"con una media de {cluster0_recency:.0f} días sin comprar.",
                 "El tamaño de este grupo lo convierte en la prioridad de reactivación, no solo una nota al pie.")
ui.insight_card("03", "El grupo más grande no es el más rentable",
                 f"El cluster 1 (digital y joven) es el {sizes_df.loc[sizes_df['cluster']==1,'pct'].values[0]}% "
                 f"de la base, pero su ticket medio ({eur(profile_df.loc[1,'Average_Ticket'])}) es el más bajo de los cuatro.",
                 "El volumen no sustituye al valor — la estrategia para este grupo no es el mismo playbook que para el premium.")
ui.insight_card("04", "Hay multicolinealidad, y eso es información, no ruido",
                 "Gasto total y puntos de fidelidad correlacionan al 0.99 — el modelo ve, en la práctica, "
                 "una sola dimensión de valor económico repetida varias veces.",
                 "Simplificar el set de variables en el futuro no perdería casi señal, y sí velocidad de cálculo.")
ui.insight_card("05", "El modelo es honesto sobre lo que no sabe",
                 "El cluster 1 no distingue dos de los cinco perfiles originales del negocio — los reparte "
                 "casi al 50%.",
                 "Señala con precisión qué información falta (producto, satisfacción, devoluciones) para completar el mapa.")
ui.section_close()

# ============================================================ DECISIONES ==
ui.section_open("decisiones")
ui.eyebrow("¿Qué haríamos con esto?")
ui.h2("Decisiones que habilita")
ui.decision_flow(
    f"Cluster 0 ({cluster0_n_fmt} clientes) lleva {cluster0_recency:.0f} días sin comprar de media",
    "Lanzar campaña de reactivación antes de que termine de desconectarse",
    "Recuperar actividad en el segmento de mayor riesgo, no el más numeroso",
    "% de clientes reactivados",
)
st.write("")
ui.decision_flow(
    f"Cluster 2 (premium, {sizes_df.loc[sizes_df['cluster']==2,'pct'].values[0]}%) genera el gasto medio "
    "más alto y la mayor frecuencia de visita",
    "Programa de fidelización dirigido, no genérico",
    "Proteger el segmento de mayor valor, no el más numeroso",
    "Retención del segmento premium",
)
st.write("")
ui.decision_flow(
    f"Cluster 1 ({sizes_df.loc[sizes_df['cluster']==1,'pct'].values[0]}%, el más grande) es joven, digital "
    "y de ticket bajo",
    "Cross-selling y financiación para subir el ticket medio, no más frecuencia — ya es alta",
    "Aumentar el valor del segmento más numeroso sin cambiar su comportamiento digital",
    "Ticket medio del segmento",
)
ui.section_close()

# ============================================================ LIMITACIONES ==
ui.section_open("limitaciones")
ui.eyebrow("Honestidad ante todo")
ui.h2("Limitaciones")
lc1, lc2 = st.columns(2, gap="large")
with lc1:
    st.markdown('<p class="limit-col-title">Lo que el modelo SÍ puede hacer</p>', unsafe_allow_html=True)
    st.markdown(
        """<ul class="limit-list">
        <li>Separar a los clientes en grupos accionables usando solo comportamiento de compra.</li>
        <li>Recuperar, sin supervisión, el segmento de mayor valor que el negocio ya identificaba.</li>
        <li>Explicar en qué se diferencian los grupos con variables interpretables por marketing.</li>
        <li>Señalar con precisión qué información adicional haría falta para ir más lejos.</li>
        </ul>""",
        unsafe_allow_html=True,
    )
with lc2:
    st.markdown('<p class="limit-col-title">Lo que el modelo NO puede hacer</p>', unsafe_allow_html=True)
    st.markdown(
        """<ul class="limit-list">
        <li>Distinguir todos los perfiles que el negocio diferencia internamente (perfiles 1 y 5 se mezclan).</li>
        <li>Explicar el "porqué" del comportamiento — solo lo agrupa, no dice qué lo causa.</li>
        <li>Mantenerse estable si el comportamiento de compra cambia de forma estructural sin reentrenar.</li>
        <li>Sustituir variables de producto, satisfacción o devoluciones que quedaron fuera a propósito.</li>
        </ul>""",
        unsafe_allow_html=True,
    )
st.markdown(
    '<div class="limit-note"><p class="co-body">'
    "La segmentación por comportamiento (RFM + canal) explica bien el extremo de mayor valor, pero no "
    "separa todos los perfiles que el negocio distingue internamente — probablemente los perfiles 1 y 5 se "
    "diferencian por variables (producto, satisfacción, devoluciones) que quedaron fuera de este análisis "
    "a propósito, no por una limitación de K-Means."
    "</p></div>",
    unsafe_allow_html=True,
)
ui.section_close()

# ============================================================ CONCLUSIÓN ==
ui.section_open("conclusion")
ui.eyebrow("Del dato a la decisión")
ui.h2("Conclusión")
ui.lead(
    f"Con solo 9 variables de comportamiento, K-Means separa a los clientes en 4 grupos accionables y "
    f"recupera de forma casi exacta ({premium_pct:.1f}%) el segmento premium que el negocio ya tenía "
    "identificado — sin haber visto esa etiqueta durante el entrenamiento. El modelo no sustituye el "
    "criterio de marketing: le da un punto de partida basado en comportamiento real, no en intuición."
)
ui.section_close()

ui.footer_minimal(
    name="Borja Mora Méndez",
    repo_url="https://github.com/BORJAMOME/segmentacion-retail-app",
    linkedin_url="https://www.linkedin.com/in/borja-mora-mendez/",
    email="borja.mora.mendez@gmail.com",
)
