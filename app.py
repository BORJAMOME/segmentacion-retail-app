"""
Segmentación de Clientes Retail — K-Means + t-SNE
Reportaje digital de datos: del problema de negocio a la decisión, pasando por los datos,
el método, la evidencia y sus límites.

Narrativa (12 beats):
  LEDE → 01 El problema → 02 Los datos → 03 Antes de modelar → 04 El camino hasta el modelo →
  05 K-Means: cuatro grupos → 06 Los cuatro perfiles → 07 ¿Coincide con el perfil original? →
  08 Ponlo a prueba → 09 El resultado → 10 ¿Qué podría hacer una empresa? →
  11 Limitaciones → 12 Del dato a la decisión

La composición vive en assets/editorial.css + components/editorial.py (sistema editorial
reutilizable); aquí solo hay contenido y datos. Toda cifra del texto sale de los artefactos de
model/artifacts (train.py y export_profile_controls.py): nada está escrito a mano.

Autor: Borja Mora Méndez
"""
import importlib
from pathlib import Path

import streamlit as st

from components import charts
from components import editorial as ed
from utils.clustering import predict_cluster, scale_input
from utils.data_loader import (CLUSTER_META, FEATURES, FEATURE_LABELS, artifacts_ready,
                                load_csv, load_json)

# Streamlit recarga app.py al detectar cambios, pero mantiene en memoria los módulos locales ya
# importados. Tras un despliegue que modifica components/*.py y app.py a la vez, eso deja un
# app.py nuevo llamando a un módulo antiguo (AttributeError). Recargarlos en cada ejecución lo
# evita; el coste es despreciable.
importlib.reload(charts)
importlib.reload(ed)

ROOT = Path(__file__).resolve().parent
LECTURA = "13 min"
ACTUALIZADO = "Septiembre 2026"
FUENTE = "Elaboración propia"

st.set_page_config(
    page_title="Segmentación de Clientes Retail · K-Means + t-SNE",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Identidad del proyecto primero (paleta, tipografías); sistema editorial después
ed.load_css(ROOT, "style.css", "editorial.css")

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
control_df = load_csv("profile_control_vars.csv", index_col=0)

CLUSTER_NAMES = {k: v["name"] for k, v in CLUSTER_META.items()}
CLUSTER_COLORS = {k: v["color"] for k, v in CLUSTER_META.items()}
RIESGO, DIGITAL, PREMIUM, TIENDA = 0, 1, 2, 3          # identificadores de CLUSTER_META
PLOT = {"displayModeBar": False}

n_customers = stats["n_customers"]
ranges = stats["feature_ranges"]


def es(value: float, decimals: int = 1) -> str:
    """Número con coma decimal, como se escribe en español."""
    return f"{value:.{decimals}f}".replace(".", ",")


def pct(value: float, decimals: int = 1) -> str:
    return f"{es(value * 100, decimals)}%"


def miles(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".")


def eur(value: float) -> str:
    return miles(value) + " €"


n_fmt = miles(n_customers)

# ---- Métricas de validación contra el perfil original (crosstab: filas = cluster, columnas = perfil)
ct = crosstab_df
prem_n = int(ct.loc[PREMIUM, 2])
prem_size = int(ct.loc[PREMIUM].sum())
prof2_total = int(ct[2].sum())
prem_pureza = prem_n / prem_size                       # de los clientes del cluster, cuántos son perfil 2
prem_cobertura = prem_n / prof2_total                  # de los clientes del perfil 2, cuántos caen en el cluster
prof2_en_tienda = int(ct.loc[TIENDA, 2])
riesgo_p4 = int(ct.loc[RIESGO, 4])
prof4_total = int(ct[4].sum())
riesgo_p4_cobertura = riesgo_p4 / prof4_total
riesgo_p4_pureza = riesgo_p4 / int(ct.loc[RIESGO].sum())
dig_size = int(ct.loc[DIGITAL].sum())
dig_p1, dig_p5 = int(ct.loc[DIGITAL, 1]) / dig_size, int(ct.loc[DIGITAL, 5]) / dig_size

profile_spend = features_raw["Total_Spending"].groupby(tsne_coords["Customer_Profile"].values).mean()
p1c, p5c = control_df.loc[1], control_df.loc[5]

# ---- Otras cifras del texto
mean_spend, median_spend = features_raw["Total_Spending"].mean(), features_raw["Total_Spending"].median()
r_spend_loyalty = correlation.loc["Total_Spending", "Loyalty_Points"]
r_ticket_income = correlation.loc["Average_Ticket", "Annual_Income"]
r_online_visits = correlation.loc["Online_Purchases", "Website_Visits"]
sil = elbow_df.set_index("k")["silhouette"]
sil_final = stats["silhouette_final"]
size_pct = sizes_df.set_index("cluster")["pct"]
size_n = sizes_df.set_index("cluster")["n"]
ratio_spend = profile_df.loc[PREMIUM, "Total_Spending"] / profile_df.loc[RIESGO, "Total_Spending"]
ratio_recency = profile_df.loc[RIESGO, "Days_Since_Last_Purchase"] / profile_df.loc[PREMIUM, "Days_Since_Last_Purchase"]
share_online = profile_df["Online_Purchases"] / profile_df["Total_Purchases"]

ax1 = tsne_axis_corr["tSNE_1"].abs().sort_values(ascending=False).index[:3]
ax2 = tsne_axis_corr["tSNE_2"].abs().sort_values(ascending=False).index[:2]


def axis_list(axis: str, feats) -> str:
    parts = [f"<b>{FEATURE_LABELS[f]}</b> ({es(tsne_axis_corr.loc[f, axis], 2)})" for f in feats]
    return ", ".join(parts[:-1]) + " y " + parts[-1]


# ============================================================ LEDE ==
ed.skip_link("contexto")
ed.topbar(
    "Segmentación de Clientes · Retail",
    [("contexto", "Problema"), ("datos", "Datos"), ("metodo", "Método"), ("perfiles", "Perfiles"),
     ("validacion", "Validación"), ("playground", "Playground"), ("resultado", "Resultado"),
     ("implicaciones", "Implicaciones"), ("conclusion", "Conclusión")],
    back_url="https://borjamora.es/",
)
ed.anchor_scroll()
ed.lede(
    kicker="Machine Learning Case Study · Clustering",
    headline=("Tu equipo de marketing ya sabe quiénes son tus mejores clientes. ¿Puede un algoritmo llegar a "
              "la misma conclusión <em>sin que nadie se lo diga</em>?"),
    deck=(
        f"Agrupé a {n_fmt} clientes de una cadena de electrónica de consumo usando solo su comportamiento de "
        "compra, sin que el modelo viera nunca el perfil que ya traían asignado. El resultado: el grupo premium "
        f"que encontró coincide casi por completo con el perfil 2 original (<b>el {pct(prem_pureza)} de sus "
        f"clientes lo eran</b>), aunque reúne solo al <b>{pct(prem_cobertura)}</b> de los clientes de ese perfil."
    ),
    meta=[
        ("Autor", "Borja Mora Méndez"),
        ("Stack", "Python · scikit-learn (K-Means + t-SNE) · Streamlit"),
        ("Datos", f"{n_fmt} clientes · {stats['n_columns_original']} variables"),
        ("Actualizado", ACTUALIZADO),
        ("Lectura", LECTURA),
    ],
)

# ============================================================ 01 · EL PROBLEMA ==
ed.beat(
    "contexto", "01", "Problema", "El problema",
    deck=(
        "Una cadena de electrónica de consumo trata a toda su base de clientes por igual: las mismas ofertas, "
        "los mismos emails, el mismo descuento genérico. El equipo de marketing sabe que no todos compran igual "
        "— pero con miles de clientes y decenas de variables, no hay forma humana de separarlos a mano."
    ),
)
ed.band(
    "La pregunta de negocio",
    '¿Existen grupos naturales de clientes según su comportamiento de compra, '
    '<span class="accent">y se pueden explicar de un vistazo</span>?',
    "No quise inventar categorías de marketing sobre el papel: dejé que el propio comportamiento de compra "
    "(gasto, frecuencia, canal, antigüedad) revelara si esos grupos ya existían.",
)

# ============================================================ 02 · LOS DATOS ==
ed.beat(
    "datos", "02", "Datos", "Los datos",
    deck=(
        f"Trabajé con <b>{n_fmt} clientes</b> y {stats['n_columns_original']} variables por cliente: demografía, "
        "gasto, canal, categorías de producto e interacción con marketing. Para no arrastrar ruido al modelo, usé "
        f"solo <b>{stats['n_features_used']} variables</b> de comportamiento y valor: el mismo criterio RFM "
        "(Recencia, Frecuencia, valor Monetario) ampliado con canal digital."
    ),
)
ed.provenance([
    ("Dataset", "clientes_mediamarkt.xlsx"),
    ("Observaciones", f"{n_fmt} clientes"),
    ("Variables", f"{stats['n_columns_original']} ({stats['n_features_used']} usadas para segmentar)"),
    ("Valores vacíos", f"{int(features_raw.isna().sum().sum())} en las {stats['n_features_used']} variables usadas"),
    ("Perfil original", "Customer_Profile (1 a 5), reservado como control"),
    ("Elaboración", FUENTE),
])
ed.lists([
    {"title": "Valor económico", "points": [FEATURE_LABELS[f] for f in ["Total_Spending", "Average_Ticket", "Annual_Income"]]},
    {"title": "Frecuencia y fidelidad", "points": [FEATURE_LABELS[f] for f in ["Total_Purchases", "Loyalty_Points", "Age"]]},
    {"title": "Canal y actividad", "points": [FEATURE_LABELS[f] for f in ["Online_Purchases", "Website_Visits", "Days_Since_Last_Purchase"]]},
])
ed.passage(
    "El dataset trae un campo <b>Customer_Profile</b> (1 a 5) que ya venía asignado en el dato de origen. No se "
    "lo di al modelo: lo reservé aparte y lo usé solo al final, como control externo, para comprobar si "
    "K-Means llegaba a conclusiones parecidas sin haberlo visto nunca."
)

# ============================================================ 03 · ANTES DE MODELAR ==
ed.beat(
    "exploracion", "03", "Antes de modelar", "¿Qué me dicen los datos?",
    deck=(
        "Antes de tocar ningún algoritmo, me hice dos preguntas: <b>¿cómo se distribuye el gasto?</b> "
        "¿Hay variables que en el fondo <b>miden lo mismo</b>?"
    ),
)

ed.subhead("Distribución del gasto total", level="wide")
with ed.split("gasto", "7-5") as (viz, txt):
    with viz:
        st.plotly_chart(charts.histogram(features_raw["Total_Spending"], "Gasto total (€)"),
                        use_container_width=True, config=PLOT)
        ed.caption("FIG. 01", "Número de clientes según su gasto total (€).", FUENTE)
    with txt:
        ed.insight(
            f"La media de gasto ({eur(mean_spend)}) supera claramente a la mediana ({eur(median_spend)}): hay un "
            "grupo pequeño de clientes que gasta mucho más que el resto, estirando la distribución hacia la "
            "derecha. Como se verá más abajo, el segmento premium ocupa esa cola."
        )

ed.subhead("¿Hay variables que miden lo mismo dos veces?", level="wide")
with ed.split("colinealidad", "5-7") as (txt, viz):
    with txt:
        ed.insight(
            f"Hay colinealidad fuerte: <b>Gasto total</b> y <b>Puntos de fidelidad</b> correlacionan al "
            f"{es(r_spend_loyalty, 2)} (los puntos parecen acumularse en proporción al gasto, así que son casi la "
            f"misma información dos veces), y <b>Ticket medio</b> con <b>Ingreso anual</b> al "
            f"{es(r_ticket_income, 2)}. K-Means no exige variables independientes, pero explica por qué varias "
            "métricas se moverán juntas al segmentar."
        )
        ed.note(
            "<b>Detalle técnico.</b> Correlación de Pearson. Tercer par más alto: Compras online y Visitas a la "
            f"web ({es(r_online_visits, 2)}). Con distancias euclídeas, las variables muy correlacionadas pesan "
            "de hecho más: la dimensión de valor económico cuenta varias veces."
        )
    with viz:
        st.plotly_chart(charts.correlation_heatmap(correlation, FEATURE_LABELS),
                        use_container_width=True, config=PLOT)
        ed.caption("FIG. 02", "Correlación entre pares de las 9 variables usadas para segmentar. Azul: correlación positiva; rojo: negativa.", FUENTE)

# ============================================================ 04 · EL CAMINO HASTA EL MODELO ==
ed.beat(
    "metodo", "04", "Método", "El camino hasta el modelo",
    deck=(
        "Segmentar sin supervisión tiene una trampa: el algoritmo siempre encuentra grupos, aunque no exista "
        "ninguna estructura real. Este es el camino para no caer en ella."
    ),
)
ed.steps([
    ("Escalé los datos",
     f"K-Means mide distancias. Sin estandarizar, el ingreso anual (de {miles(ranges['Annual_Income']['min'])} a "
     f"{miles(ranges['Annual_Income']['max'])} €) aplastaría a variables como la edad, que aporta información "
     "real en una escala mucho más pequeña."),
    ("Comprobé que había estructura real",
     "Antes de forzar ningún número de grupos, proyecté las 9 variables en 2D con <b>t-SNE</b>. Si no había "
     "regiones separables en ese mapa, no tenía sentido segmentar: solo estaría troceando ruido."),
    ("Probé varios números de grupos",
     "Con el método del codo y el silhouette score, comparé k=2 a k=8. El mejor silhouette aislado no siempre "
     "es el más útil para el negocio: tuve que <b>decidir</b>, no solo leer un gráfico."),
    ("Entrené K-Means y contrasté con la realidad",
     "Con k=4 ya elegido, entrené el modelo final y lo comparé contra el <b>perfil original</b>, sin haberlo "
     "visto nunca durante el entrenamiento."),
])

ed.subhead("¿Hay estructura real antes de segmentar?", level="wide")
profile_values = sorted(tsne_coords["Customer_Profile"].unique())
profile_palette = ["#B9C5D6", "#4A628E", "#6E7F5B", "#B8783C", "#C2412E"]
profile_colors = dict(zip(profile_values, profile_palette))
profile_labels = {p: f"Perfil {p} (original)" for p in profile_values}
with ed.split("tsne-estructura", "8-4") as (viz, txt):
    with viz:
        st.plotly_chart(charts.tsne_scatter(tsne_coords, "Customer_Profile", profile_colors, profile_labels),
                        use_container_width=True, config=PLOT)
        ed.caption("FIG. 03", "Mapa t-SNE de los clientes, coloreado por el perfil original (que no se usó para "
                              "entrenar nada).", FUENTE)
    with txt:
        ed.insight(
            "Al colorear por el perfil original, se ven zonas del mapa dominadas por un único color. Es una señal "
            "de que hay estructura que descubrir, no solo ruido: el paso siguiente (K-Means) tenía sentido."
        )
        ed.note(
            "<b>Detalle técnico.</b> t-SNE (<i>t-distributed Stochastic Neighbor Embedding</i>) proyecta las 9 "
            "variables en 2D conservando las distancias <i>locales</i> entre puntos cercanos, no la varianza global "
            "(a diferencia de PCA). Se usó <code>perplexity=35</code> (rango recomendado 5–50, escalado a unos "
            f"{miles(round(n_customers, -2))} clientes) e <code>init=\"pca\"</code>, que estabiliza el resultado. "
            "En t-SNE la distancia entre grupos lejanos no es interpretable: sirve para ver agrupaciones, no para "
            "medirlas."
        )

ed.subhead("¿Qué representa cada eje del mapa?")
ed.passage(
    f"Las variables que más correlacionan con el eje horizontal (tSNE_1) son {axis_list('tSNE_1', ax1)}: ese eje "
    "ordena a los clientes de menor a mayor <b>valor económico</b> (también recoge la recencia: "
    f"{es(tsne_axis_corr.loc['Days_Since_Last_Purchase', 'tSNE_1'], 2)} con los días desde la última compra). El "
    f"eje vertical (tSNE_2) está dominado por {axis_list('tSNE_2', ax2)}: separa al cliente <b>digital</b> del "
    "que compra poco por internet. El mapa resume 9 variables en dos preguntas: <i>¿cuánto vale este cliente?</i> "
    "y <i>¿qué tan digital es?</i>",
    aside="Los ejes de t-SNE no tienen orientación fija: pueden cambiar entre ejecuciones. Aquí se interpretan con "
          "los valores guardados de esta ejecución.",
    aside_label="Nota",
)

ed.subhead("Eligiendo k: codo y silhouette", level="wide")
with ed.split("codo", "7-5") as (viz, txt):
    with viz:
        st.plotly_chart(charts.elbow_silhouette(elbow_df, stats["n_clusters"]),
                        use_container_width=True, config=PLOT)
        ed.caption("FIG. 04", "Inercia (codo) y silhouette score para k de 2 a 8. El k elegido está marcado.", FUENTE)
    with txt:
        ed.insight(
            f"El silhouette score más alto se da en k=2 ({es(sil[2], 2)}), pero es demasiado grueso para el "
            "negocio: separaría solo «gasta poco» de «gasta mucho». Con k=3 y k=4 prácticamente empatados "
            f"({es(sil[3], 3)} y {es(sil[4], 3)}), elegí <b>k=4</b>: misma calidad de separación y un perfil "
            "accionable más."
        )
        ed.note(
            "<b>Detalle técnico.</b> El silhouette mide, para cada cliente, si está más cerca de los de su propio "
            "grupo que de los del grupo más próximo (de −1 a 1). Baja de forma clara a partir de k=5 "
            f"({es(sil[5], 2)})."
        )

# ============================================================ 05 · K-MEANS: CUATRO GRUPOS ==
ed.beat(
    "modelo", "05", "Modelo", "K-Means: cuatro grupos",
    deck=(
        f"Con indicios de estructura y k=4 decidido, entrené el modelo final para agrupar a los {n_fmt} clientes "
        "en 4 segmentos según sus 9 variables de comportamiento, sin que viera nunca el perfil original."
    ),
)
ed.metrics([
    ("Silhouette · k=4", es(sil_final, 3),
     "Mide si cada cliente está más cerca de su propio grupo que del más próximo. Un valor en torno a 0,3 indica "
     "grupos reconocibles pero con fronteras difusas: los clientes se mueven en un continuo, no en cajas "
     "perfectamente separadas."),
])

ed.subhead("Tamaño de cada segmento", level="story")
with ed.figure("tamanos", level="story"):
    st.plotly_chart(charts.cluster_sizes(sizes_df, CLUSTER_NAMES), use_container_width=True, config=PLOT)
    ed.caption("FIG. 05", f"Número de clientes en cada uno de los 4 clusters (total: {n_fmt}).", FUENTE)

ed.subhead("Los cuatro clusters, proyectados en el mapa t-SNE", level="wide")
with ed.split("tsne-clusters", "8-4") as (viz, txt):
    with viz:
        st.plotly_chart(
            charts.tsne_scatter(tsne_coords, "KMeans_Profile", CLUSTER_COLORS,
                                {k: f"Cluster {k} — {v}" for k, v in CLUSTER_NAMES.items()}),
            use_container_width=True, config=PLOT,
        )
        ed.caption("FIG. 06", "Mapa t-SNE coloreado por el cluster de K-Means.", FUENTE)
    with txt:
        ed.insight(
            "Los cuatro clusters ocupan regiones diferenciadas del mapa, coherentes con los ejes ya "
            "interpretados: el premium se concentra en la zona de mayor valor económico, y los otros dos se separan "
            "sobre todo en el eje digital."
        )
        ed.note(
            "<b>Detalle técnico.</b> K-Means y t-SNE parten de los mismos datos escalados, así que su coincidencia "
            "es coherente con que haya estructura pero no es una comprobación independiente. La comprobación "
            "independiente es la comparación con el perfil original (beat 07)."
        )

# ============================================================ 06 · LOS CUATRO PERFILES ==
ed.beat(
    "perfiles", "06", "Explicabilidad", "Los cuatro perfiles", weight="minor",
    deck=(
        "Cada cluster tiene un comportamiento de compra distinto y reconocible: así es como lo vería un "
        "responsable de marketing, en lenguaje llano, no en coordenadas."
    ),
)
p = profile_df
CLUSTER_TEXT = {
    PREMIUM: (f"El más pequeño y, con diferencia, el de mayor valor: gasto medio de {eur(p.loc[PREMIUM, 'Total_Spending'])}. "
              f"En promedio, {pct(share_online[PREMIUM], 0)} de sus compras son online. Es el grupo más activo: la "
              f"recencia media más baja de los cuatro ({p.loc[PREMIUM, 'Days_Since_Last_Purchase']:.0f} días)."),
    TIENDA: (f"Gasto medio-alto ({eur(p.loc[TIENDA, 'Total_Spending'])}), pero el grupo menos digital: solo el "
             f"{pct(share_online[TIENDA], 0)} de sus compras son online. Es el que más compra en tienda física."),
    DIGITAL: (f"El más numeroso. Los más jóvenes ({p.loc[DIGITAL, 'Age']:.0f} años de media) y muy digitales: el "
              f"{pct(share_online[DIGITAL], 0)} de sus compras son online. Su ticket medio es el más bajo de los "
              f"cuatro ({eur(p.loc[DIGITAL, 'Average_Ticket'])})."),
    RIESGO: (f"El gasto más bajo ({eur(p.loc[RIESGO, 'Total_Spending'])}), la menor actividad web "
             f"({p.loc[RIESGO, 'Website_Visits']:.0f} visitas de media) y {p.loc[RIESGO, 'Days_Since_Last_Purchase']:.0f} "
             f"días desde la última compra, {es(ratio_recency)} veces más que el grupo premium: la señal de posible "
             "desconexión más clara del análisis."),
}
with ed.figure("perfiles"):
    ed.cols([
        {"tag": f"Cluster {cid} · {miles(size_n[cid])} clientes ({es(size_pct[cid])}%)",
         "title": CLUSTER_NAMES[cid], "text": CLUSTER_TEXT[cid], "color": CLUSTER_COLORS[cid]}
        for cid in [PREMIUM, TIENDA, DIGITAL, RIESGO]
    ], count=4)
ed.note(
    "<b>Detalle técnico.</b> El nombre «en riesgo» interpreta la recencia: el dataset no incluye ninguna variable "
    "de abandono, así que no se sabe si esos clientes se han ido."
)

ed.subhead("Comparación directa: gasto total por cluster", level="wide")
with ed.split("gasto-cluster", "5-7") as (txt, viz):
    with txt:
        ed.insight(
            f"El cluster premium gasta de media {eur(p.loc[PREMIUM, 'Total_Spending'])}, "
            f"{es(ratio_spend)} veces más que el cluster en riesgo. No es un matiz: son negocios distintos dentro "
            "del mismo negocio."
        )
        ed.note(
            "Los perfiles describen promedios de grupo, no reglas fijas: dentro de cada cluster hay variación "
            "individual. Sirven para dirigir estrategia, no para juzgar a un cliente concreto."
        )
    with viz:
        st.plotly_chart(charts.cluster_profile_bars(profile_df, "Total_Spending", CLUSTER_NAMES, FEATURE_LABELS),
                        use_container_width=True, config=PLOT)
        ed.caption("FIG. 07", "Gasto total medio (€) de los clientes de cada cluster.", FUENTE)

# ============================================================ 07 · ¿COINCIDE CON EL PERFIL ORIGINAL? ==
ed.beat(
    "validacion", "07", "Validación", "¿Coincide con el perfil original?", weight="major",
    deck=(
        "El dataset traía un campo <b>Customer_Profile</b> (1 a 5) que nunca usé para entrenar. Lo comparo ahora, "
        "solo como control, para ver si los 4 clusters de K-Means reconstruyen esa segmentación original. Hay dos "
        "preguntas distintas: ¿los clientes de cada cluster comparten perfil? Y ¿cada perfil acaba entero en un "
        "mismo cluster?"
    ),
)
with ed.figure("crosstab", level="full"):
    st.plotly_chart(charts.crosstab_heatmap(crosstab_df, CLUSTER_NAMES), use_container_width=True, config=PLOT)
    ed.caption("FIG. 08", "Número de clientes por cluster de K-Means (filas) y por perfil original (columnas).", FUENTE)
ed.insight(
    f"El cluster premium es un grupo muy limpio: <b>{prem_n} de sus {prem_size} clientes ({pct(prem_pureza)})</b> "
    "pertenecen al perfil 2 original. Pero el camino inverso es menos exacto: el cluster reúne al "
    f"<b>{pct(prem_cobertura)}</b> de los {miles(prof2_total)} clientes del perfil 2, y otros {prof2_en_tienda} "
    f"quedaron en el cluster «{CLUSTER_NAMES[TIENDA]}».",
    aside=(
        f"Pureza del cluster (precisión) = {pct(prem_pureza)}; cobertura del perfil (recall) = "
        f"{pct(prem_cobertura)}. El perfil 2 es el de mayor gasto medio ({eur(profile_spend[2])}), y por eso se "
        "asocia al cluster premium."
    ),
)
ed.insight(
    f"El cluster digital, en cambio, no distingue entre los perfiles 1 y 5 originales: los reparte casi por "
    f"mitades ({pct(dig_p1)} y {pct(dig_p5)}). Es una señal honesta de que la diferencia entre esos dos perfiles "
    "no está en las 9 variables de comportamiento usadas.",
    aside=(
        "Entre los perfiles 1 y 5, las variables que no se usaron sí difieren: cupones usados "
        f"({es(p1c['Coupons_Used'], 2)} frente a {es(p5c['Coupons_Used'], 2)}) y antigüedad "
        f"({es(p1c['Customer_Tenure'], 2)} frente a {es(p5c['Customer_Tenure'], 2)}), por ejemplo. Satisfacción "
        f"({es(p1c['Satisfaction'], 2)} y {es(p5c['Satisfaction'], 2)}) y devoluciones "
        f"({es(p1c['Returns'], 2)} y {es(p5c['Returns'], 2)}) casi no cambian."
    ),
)

# ============================================================ 08 · PONLO A PRUEBA ==
ed.beat(
    "playground", "08", "Playground", "Ponlo a prueba", weight="major",
    deck=(
        "Ajusta el comportamiento de un cliente hipotético y el modelo predice, en vivo, a qué cluster "
        "pertenecería, usando el mismo cálculo que K-Means: <b>escalar y buscar el centroide más cercano</b>."
    ),
)

with ed.split("playground", "5-7") as (pg_left, pg_right):
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
    sorted_dists = sorted(zip(CLUSTER_NAMES.keys(), dists), key=lambda x: x[1])
    runner_up_id, runner_up_dist = sorted_dists[1]

    with pg_right:
        ed.metrics(
            [("Este cliente encaja en", f"Cluster {cluster_id} — {CLUSTER_NAMES[cluster_id]}",
              f"Su centroide es el más cercano en el espacio escalado. El siguiente es el Cluster {runner_up_id} "
              f"({CLUSTER_NAMES[runner_up_id]}).", CLUSTER_COLORS[cluster_id])],
            word=True,
        )
        st.plotly_chart(
            charts.playground_radar(user_scaled, centroid_scaled, FEATURES, FEATURE_LABELS),
            use_container_width=True, config=PLOT,
        )
        ed.caption("FIG. 09", "Perfil del cliente frente al centroide de su cluster, en valores estandarizados.",
                   "Modelo entrenado en este proyecto")

ed.subhead("¿Por qué este cluster?")
ed.insight(
    f"De los 4 centroides, el más cercano en el espacio escalado es el del <b>Cluster {cluster_id} "
    f"({CLUSTER_NAMES[cluster_id]})</b>. El segundo más próximo es el Cluster {runner_up_id} "
    f"({CLUSTER_NAMES[runner_up_id]}). Cuanto más parecidas sean las dos distancias, más «fronterizo» es este "
    "cliente entre ambos perfiles, igual que ocurre con muchos clientes de la base."
)

# ============================================================ 09 · EL RESULTADO ==
ed.beat("resultado", "09", "Resultado", "El resultado", weight="major")
ed.band(
    "La conclusión",
    f'Sin ver el perfil original, K-Means forma un grupo premium en el que el '
    f'<span class="pos">{pct(prem_pureza)}</span> de los clientes ya eran perfil 2 — aunque reúne solo al '
    f'<span class="neg">{pct(prem_cobertura)}</span> de los clientes de ese perfil.',
    "El modelo no inventó los grupos: encontró, solo con comportamiento, un segmento premium muy parecido al que "
    "ya existía en los datos, sin ser una réplica exacta de la segmentación original.",
    quote=True,
)
ed.metrics([
    ("Silhouette · k=4", es(sil_final, 3),
     "Los grupos son reconocibles pero sus fronteras son difusas: un valor cercano a 0,3, no a 1."),
    ("Pureza · premium", pct(prem_pureza),
     f"De cada 100 clientes del cluster premium, {prem_pureza * 100:.0f} tenían el perfil 2 original. El grupo "
     "está casi libre de clientes de otros perfiles."),
    ("Cobertura · perfil 2", pct(prem_cobertura),
     f"De cada 100 clientes del perfil 2, {prem_cobertura * 100:.0f} acaban en el cluster premium; "
     f"{prof2_en_tienda / prof2_total * 100:.0f} se van al cluster de valor medio-alto."),
    ("Cobertura · perfil 4", pct(riesgo_p4_cobertura),
     f"El cluster en riesgo reúne al {pct(riesgo_p4_cobertura, 0)} de los clientes del perfil 4, aunque también "
     f"incluye a otros: solo el {pct(riesgo_p4_pureza, 0)} de sus clientes eran perfil 4."),
])
ed.passage(
    "Qué significa: la segmentación por comportamiento aísla bien los dos extremos —el de mayor valor y el "
    "de menor valor y mayor recencia— y mezcla dos perfiles (1 y 5) que se diferencian por variables que no se "
    "usaron. Es un punto de partida sólido, no una réplica exacta de la clasificación original.",
    tight=True,
)

# ============================================================ 10 · ¿QUÉ PODRÍA HACER UNA EMPRESA? ==
ed.beat(
    "implicaciones", "10", "Implicaciones", "¿Qué podría hacer una empresa con esta segmentación?",
    deck=(
        "Los grupos no dicen por sí solos qué hacer: plantean <b>hipótesis</b> que una empresa podría contrastar "
        "con un piloto."
    ),
)
with ed.figure("hipotesis"):
    ed.cols([
        {"tag": "Reactivación", "title": f"Cluster {RIESGO} · {CLUSTER_NAMES[RIESGO]}", "color": CLUSTER_COLORS[RIESGO],
         "text": f"{miles(size_n[RIESGO])} clientes ({es(size_pct[RIESGO])}%) llevan "
                 f"{p.loc[RIESGO, 'Days_Since_Last_Purchase']:.0f} días sin comprar de media. Una campaña de "
                 "reactivación podría probarse aquí primero, midiendo qué porcentaje de clientes vuelve a comprar."},
        {"tag": "Fidelización", "title": f"Cluster {PREMIUM} · {CLUSTER_NAMES[PREMIUM]}", "color": CLUSTER_COLORS[PREMIUM],
         "text": f"El {es(size_pct[PREMIUM])}% de los clientes genera el gasto medio más alto y la mayor actividad "
                 "web. Un programa de fidelización específico podría proteger este segmento; su retención sería la "
                 "métrica a seguir."},
        {"tag": "Valor", "title": f"Cluster {DIGITAL} · {CLUSTER_NAMES[DIGITAL]}", "color": CLUSTER_COLORS[DIGITAL],
         "text": f"Es el grupo más grande ({es(size_pct[DIGITAL])}%), joven y de ticket bajo "
                 f"({eur(p.loc[DIGITAL, 'Average_Ticket'])}). Ofertas de cross-selling o de financiación podrían "
                 "probarse para subir el ticket medio; este análisis no evalúa si funcionarían."},
    ])
ed.insight(
    "Ninguna de estas acciones sale directamente del modelo: son <b>hipótesis</b> que habría que validar, por "
    "ejemplo con un piloto y un grupo de control, antes de asumir que funcionan."
)

# ============================================================ 11 · LIMITACIONES ==
ed.beat("limitaciones", "11", "Honestidad ante todo", "Limitaciones")
ed.lists([
    {"title": "Lo que el modelo SÍ puede hacer", "points": [
        "Separar a los clientes en grupos con perfiles diferenciados usando <b>solo comportamiento</b> de compra.",
        f"Aislar un <b>grupo premium muy limpio</b>: el {pct(prem_pureza)} de sus clientes ya eran perfil 2.",
        "Explicar en qué se diferencian los grupos con <b>variables interpretables</b> por marketing.",
        "Señalar dónde <b>no separa bien</b> (los perfiles 1 y 5), lo que indica qué información falta.",
    ]},
    {"title": "Lo que el modelo NO puede hacer", "points": [
        f"<b>Capturar a todo el perfil 2</b>: solo el {pct(prem_cobertura)} de esos clientes cae en el cluster premium.",
        "Distinguir los <b>perfiles 1 y 5</b> originales: se mezclan casi al 50%.",
        "Explicar el <b>porqué</b> del comportamiento: solo lo agrupa, no dice qué lo causa.",
        "Mantenerse estable si el <b>comportamiento de compra cambia</b> de forma estructural sin reentrenar.",
        f"Garantizar el mismo resultado <b>con otros clientes</b>: se aplicó a un único conjunto de {n_fmt} clientes, "
        "sin validación con datos nuevos.",
        "Confirmar que el cluster «en riesgo» <b>abandonará</b>: el dataset no incluye ninguna variable de abandono.",
    ]},
])
ed.passage(
    "La segmentación por comportamiento (RFM + canal) explica bien los extremos, pero no separa todos los perfiles "
    "originales. En los datos, los perfiles 1 y 5 se diferencian sobre todo por variables que se dejaron fuera "
    f"a propósito: el perfil 5 ha usado {es(p5c['Coupons_Used'], 1)} cupones de media frente a "
    f"{es(p1c['Coupons_Used'], 1)} del perfil 1. Probablemente ese sea el motivo, y no una limitación de K-Means.",
    aside=(
        "La segmentación se calculó sobre todos los clientes y no hay conjunto de prueba. K-Means se entrenó con una "
        "sola semilla (random_state=42): no se ha comprobado la estabilidad de los clusters con otras semillas."
    ),
)

# ============================================================ 12 · DEL DATO A LA DECISIÓN ==
ed.beat(
    "conclusion", "12", "Conclusión", "Del dato a la decisión", weight="major",
    deck=[
        f"Con solo {stats['n_features_used']} variables de comportamiento, K-Means separa a los clientes en "
        f"{stats['n_clusters']} grupos con perfiles distintos y forma un segmento premium muy limpio: el "
        f"<b>{pct(prem_pureza)}</b> de sus clientes ya eran perfil 2 en los datos originales, sin que el modelo "
        f"viera ese perfil durante el entrenamiento. Reúne, eso sí, al {pct(prem_cobertura)} de los clientes de "
        "ese perfil: no es una réplica exacta de la segmentación original.",
        "El modelo no sustituye el criterio de marketing: le da un punto de partida basado en comportamiento "
        "real, no en intuición, y señala dónde mirar más (los perfiles 1 y 5). Ahí es donde los datos dejan de "
        "ser solo números y <b>empiezan a servir para tomar decisiones</b>.",
    ],
)
ed.colophon("Borja Mora Méndez", [
    ("Repositorio del proyecto", "https://github.com/BORJAMOME/segmentacion-retail-app"),
    ("Portfolio", "https://borjamora.es/"),
    ("LinkedIn", "https://www.linkedin.com/in/borja-mora-mendez/"),
    ("Contacto", "mailto:borja.mora.mendez@gmail.com"),
])
