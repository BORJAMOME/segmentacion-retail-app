"""
Segmentación de Clientes Retail — K-Means + t-SNE
Reportaje digital de datos: del problema de negocio a la decisión, pasando por los datos,
el método, la evidencia y sus límites.

Narrativa (12 beats):
  LEDE → 01 El problema → 02 Los datos → 03 Antes de buscar grupos → 04 ¿Cómo dejamos que los
  datos formen los grupos? → 05 K-Means: cuatro grupos → 06 Los cuatro perfiles →
  07 ¿Coincide con el perfil original? → 08 Ponlo a prueba → 09 ¿Qué hemos descubierto? →
  10 ¿Qué podría hacer marketing? → 11 Lo que sabemos y lo que no → 12 Del dato a la decisión

Regla de copy: primero se explica qué significa, después se pone el nombre técnico (en notas
«Detalle técnico» o entre paréntesis).

La composición vive en assets/editorial.css + components/editorial.py (sistema editorial
reutilizable); aquí solo hay contenido y datos. Toda cifra del texto sale de los artefactos de
model/artifacts (train.py y export_profile_controls.py) o se calcula de ellos: nada está escrito a mano.

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
LECTURA = "15 min"
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
BAJA, DIGITAL, PREMIUM, TIENDA = 0, 1, 2, 3            # identificadores de CLUSTER_META
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
prem_pureza = prem_n / prem_size                       # de los clientes del grupo, cuántos son perfil 2
prem_cobertura = prem_n / prof2_total                  # de los clientes del perfil 2, cuántos caen en el grupo
baja_p4_cobertura = int(ct.loc[BAJA, 4]) / int(ct[4].sum())
baja_p4_pureza = int(ct.loc[BAJA, 4]) / int(ct.loc[BAJA].sum())
dig_size = int(ct.loc[DIGITAL].sum())
dig_p1, dig_p5 = int(ct.loc[DIGITAL, 1]) / dig_size, int(ct.loc[DIGITAL, 5]) / dig_size

profile_spend = features_raw["Total_Spending"].groupby(tsne_coords["Customer_Profile"].values).mean()
p1c, p5c = control_df.loc[1], control_df.loc[5]

# ---- Otras cifras del texto
spend = features_raw["Total_Spending"]
mean_spend, median_spend, p75_spend = spend.mean(), spend.median(), spend.quantile(0.75)
r_spend_loyalty = correlation.loc["Total_Spending", "Loyalty_Points"]
r_ticket_income = correlation.loc["Average_Ticket", "Annual_Income"]
r_online_visits = correlation.loc["Online_Purchases", "Website_Visits"]
sil = elbow_df.set_index("k")["silhouette"]
sil_final = stats["silhouette_final"]
size_pct = sizes_df.set_index("cluster")["pct"]
size_n = sizes_df.set_index("cluster")["n"]
p = profile_df
ratio_spend = p.loc[PREMIUM, "Total_Spending"] / p.loc[BAJA, "Total_Spending"]
share_online = p["Online_Purchases"] / p["Total_Purchases"]

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
    kicker="Machine Learning Case Study · Segmentación de clientes",
    headline=f"{n_fmt} clientes. ¿Cuántos tipos de comportamiento hay <em>realmente</em>?",
    deck=(
        "Una cadena de electrónica tiene miles de clientes, pero no todos compran de la misma forma. Utilicé el "
        f"comportamiento de compra de {n_fmt} clientes para descubrir si existían <b>grupos naturales</b> con "
        "características diferentes, sin enseñar al modelo los perfiles que los datos ya traían asignados. El "
        "objetivo: comprobar si una segmentación creada únicamente a partir de los datos podía llegar a "
        "<b>conclusiones parecidas</b> a las de una segmentación existente."
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
        "Una cadena de electrónica tiene miles de clientes y cada uno se relaciona con la tienda de una manera "
        "diferente. Algunos compran mucho y con frecuencia. Otros compran principalmente online. Otros llevan "
        "meses sin volver. El reto es que, cuando hay miles de clientes y muchas variables, es <b>difícil "
        "identificar estos patrones a simple vista</b>."
    ),
)
ed.band(
    "La pregunta de negocio",
    '¿Podemos descubrir <span class="accent">grupos naturales de clientes</span> a partir de su comportamiento '
    "de compra y convertirlos en perfiles que marketing pueda entender?",
    "En lugar de definir los segmentos antes de analizar los datos, dejé que el comportamiento de los propios "
    "clientes mostrara qué grupos aparecían.",
)

# ============================================================ 02 · LOS DATOS ==
ed.beat(
    "datos", "02", "Datos", "Los datos",
    deck=(
        f"Partimos de <b>{n_fmt} clientes</b> y {stats['n_columns_original']} variables que describen su relación "
        "con la tienda: gasto, compras, canal, categorías de producto, actividad digital e interacción con "
        f"marketing. Para crear los segmentos seleccioné <b>{stats['n_features_used']} variables</b> relacionadas "
        "con comportamiento y valor del cliente. El modelo no recibió el perfil original del cliente: lo guardé "
        "aparte para utilizarlo al final como una comprobación independiente."
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

ed.subhead("¿Qué información utilizamos?", level="wide")
with ed.figure("variables"):
    ed.cols([
        {"title": "Valor económico", "text": " · ".join(FEATURE_LABELS[f] for f in ["Total_Spending", "Average_Ticket", "Annual_Income"])},
        {"title": "Frecuencia y fidelidad", "text": " · ".join(FEATURE_LABELS[f] for f in ["Total_Purchases", "Loyalty_Points", "Age"])},
        {"title": "Canal y actividad", "text": " · ".join(FEATURE_LABELS[f] for f in ["Online_Purchases", "Website_Visits", "Days_Since_Last_Purchase"])},
    ])
ed.note(
    "<b>Detalle técnico.</b> Las 9 variables siguen el criterio RFM (Recencia, Frecuencia, valor Monetario) "
    "ampliado con canal digital."
)

ed.subhead("Una decisión importante")
ed.passage(
    "El dataset ya incluía un campo llamado <code>Customer_Profile</code>, con perfiles del 1 al 5. No se "
    "utilizó para crear los segmentos. Lo reservé para una pregunta posterior:",
    tight=True,
)
ed.insight("Si el algoritmo no conoce estos perfiles, ¿llegará a <b>encontrar grupos parecidos por su cuenta</b>?")

# ============================================================ 03 · ANTES DE BUSCAR GRUPOS ==
ed.beat(
    "exploracion", "03", "Antes de modelar", "Antes de buscar grupos, hay que entender los datos",
    deck=(
        "Antes de aplicar ningún algoritmo, quería saber dos cosas: <b>¿cómo se distribuye el valor de los "
        "clientes?</b> Y ¿hay variables que estén contando <b>prácticamente la misma historia</b>? Si varias "
        "variables contienen información muy parecida, pueden acabar teniendo un peso excesivo a la hora de "
        "formar los grupos."
    ),
)

ed.subhead("No todos los clientes tienen el mismo valor", level="wide")
with ed.split("gasto", "7-5") as (viz, txt):
    with viz:
        st.plotly_chart(charts.histogram(features_raw["Total_Spending"], "Gasto total (€)"),
                        use_container_width=True, config=PLOT)
        ed.caption("FIG. 01", "Distribución del gasto total: número de clientes según su gasto (€).", FUENTE)
    with txt:
        ed.insight(
            f"Tres de cada cuatro clientes gastan menos de {eur(p75_spend)}, mientras que un grupo más pequeño "
            f"gasta mucho más. La media de gasto es de {eur(mean_spend)}, frente a una mediana de "
            f"{eur(median_spend)}. Esa diferencia sugiere que existe <b>una pequeña parte de clientes de alto "
            "valor</b> que puede estar escondida dentro del conjunto general. Más adelante veremos si el "
            "algoritmo es capaz de identificarla como un grupo propio."
        )

ed.subhead("¿Hay variables que cuentan prácticamente lo mismo?", level="wide")
with ed.split("colinealidad", "5-7") as (txt, viz):
    with txt:
        ed.insight(
            f"Sí. Por ejemplo, <b>Gasto total</b> y <b>Puntos de fidelidad</b> tienen una correlación de "
            f"{es(r_spend_loyalty, 2)}: los puntos parecen crecer casi al mismo ritmo que el gasto. También "
            f"encontramos una relación muy fuerte entre <b>Ticket medio</b> e <b>Ingreso anual</b> "
            f"({es(r_ticket_income, 2)}). Esto no significa que haya que eliminar esas variables, pero sí "
            "tenerlo en cuenta: cuando el modelo calcula distancias, varias variables relacionadas pueden hacer "
            "que una misma dimensión del comportamiento tenga más peso."
        )
        ed.note(
            "<b>Detalle técnico.</b> Correlación de Pearson. Tercer par más alto: Compras online y Visitas a la "
            f"web ({es(r_online_visits, 2)}). Las distancias son euclídeas."
        )
    with viz:
        st.plotly_chart(charts.correlation_heatmap(correlation, FEATURE_LABELS),
                        use_container_width=True, config=PLOT)
        ed.caption("FIG. 02", "Correlación entre pares de las 9 variables usadas para segmentar. Azul: correlación "
                              "positiva; rojo: negativa.", FUENTE)

# ============================================================ 04 · ¿CÓMO DEJAMOS QUE LOS DATOS FORMEN LOS GRUPOS? ==
ed.beat(
    "metodo", "04", "Método", "¿Cómo dejamos que los datos formen los grupos?",
    deck=(
        "En una segmentación no supervisada existe un problema: K-Means siempre va a crear grupos si se lo "
        "pedimos, <b>aunque los datos no tengan una estructura clara</b>. Por eso no quería elegir un número de "
        "segmentos «porque quedaba bien». Primero exploré visualmente si aparecían patrones, después probé "
        "diferentes números de grupos y finalmente elegí la solución que ofrecía un equilibrio entre calidad "
        "estadística y utilidad para el negocio."
    ),
)
ed.steps([
    ("Puse todas las variables en una escala comparable",
     "K-Means agrupa clientes según la distancia entre ellos. Si una variable está expresada en decenas de "
     f"miles de euros (el ingreso anual va de {miles(ranges['Annual_Income']['min'])} a "
     f"{miles(ranges['Annual_Income']['max'])} €) y otra en años o número de compras, la primera podría "
     "<b>dominar simplemente por utilizar números más grandes</b>. Por eso estandaricé las variables antes de "
     "buscar los grupos."),
    ("Exploré visualmente si aparecían patrones",
     "Antes de pedirle al algoritmo que cree segmentos, quería saber si los datos mostraban algún patrón "
     "reconocible. Convertí las 9 variables en un mapa de dos dimensiones (método: t-SNE). <b>No es una prueba "
     "definitiva</b> de que existan grupos, pero sí una forma útil de explorar si aparecen zonas con "
     "comportamientos similares."),
    ("Probé diferentes números de segmentos",
     "Probé soluciones entre 2 y 8 grupos y comparé cómo de bien quedaban separados (con el silhouette score). "
     "El mejor resultado estadístico aparecía con 2 grupos, pero era demasiado simple para el objetivo de "
     "negocio: básicamente separaba clientes de menor y mayor valor. Con 3 y 4 grupos la calidad era "
     "prácticamente igual. Por eso elegí <b>4 segmentos</b>: ofrecían un nivel de separación similar, pero "
     "permitían describir perfiles de clientes más útiles para marketing."),
    ("Entrené K-Means y lo comparé con el perfil original",
     "Con los 4 segmentos elegidos, entrené el modelo final y solo entonces lo comparé con el <b>perfil "
     "original</b>, que hasta ese momento el modelo no había visto."),
])

ed.subhead("¿Aparecen patrones reconocibles en los datos?", level="wide")
profile_values = sorted(tsne_coords["Customer_Profile"].unique())
profile_palette = ["#B9C5D6", "#4A628E", "#6E7F5B", "#B8783C", "#C2412E"]
profile_colors = dict(zip(profile_values, profile_palette))
profile_labels = {p_: f"Perfil {p_} (original)" for p_ in profile_values}
with ed.split("tsne-estructura", "8-4") as (viz, txt):
    with viz:
        st.plotly_chart(charts.tsne_scatter(tsne_coords, "Customer_Profile", profile_colors, profile_labels),
                        use_container_width=True, config=PLOT)
        ed.caption("FIG. 03", "Mapa de los clientes (t-SNE), coloreado por el perfil original, que no se usó para "
                              "entrenar nada.", FUENTE)
    with txt:
        ed.insight(
            "Al colorear el mapa por el perfil original, se ven zonas dominadas por un único color. Es un "
            "<b>indicio</b> de que hay patrones que explorar, no una prueba definitiva: el paso siguiente "
            "(K-Means) tenía sentido."
        )
        ed.note(
            "<b>Detalle técnico.</b> t-SNE (<i>t-distributed Stochastic Neighbor Embedding</i>) proyecta las 9 "
            "variables en 2D conservando las distancias <i>locales</i> entre puntos cercanos, no la varianza "
            "global (a diferencia de PCA). Se usó <code>perplexity=35</code> (rango recomendado 5–50, escalado "
            f"a unos {miles(round(n_customers, -2))} clientes) e <code>init=\"pca\"</code>, que estabiliza el "
            "resultado. En t-SNE la distancia entre grupos lejanos no es interpretable: sirve para ver "
            "agrupaciones, no para medirlas."
        )

ed.subhead("¿Qué representa cada eje del mapa?")
ed.passage(
    f"El mapa resume las 9 variables en dos ejes. El horizontal (tSNE_1) ordena a los clientes de menor a mayor "
    f"<b>valor económico</b>: lo explican sobre todo {axis_list('tSNE_1', ax1)}, y también los días desde la "
    f"última compra ({es(tsne_axis_corr.loc['Days_Since_Last_Purchase', 'tSNE_1'], 2)}). El vertical (tSNE_2) "
    f"separa al cliente más <b>digital</b> del que compra poco por internet: lo explican {axis_list('tSNE_2', ax2)}. "
    "En dos preguntas: <i>¿cuánto vale este cliente?</i> y <i>¿qué tan digital es?</i>",
    aside="Los ejes de t-SNE no tienen orientación fija: pueden cambiar entre ejecuciones. Aquí se interpretan "
          "con los valores guardados de esta ejecución.",
    aside_label="Nota",
)

ed.subhead("Elección del número de grupos", level="wide")
with ed.split("codo", "7-5") as (viz, txt):
    with viz:
        st.plotly_chart(charts.elbow_silhouette(elbow_df, stats["n_clusters"]),
                        use_container_width=True, config=PLOT)
        ed.caption("FIG. 04", "Inercia (codo) y silhouette score para k de 2 a 8. El k elegido está marcado.", FUENTE)
    with txt:
        ed.insight(
            f"El mejor resultado estadístico aparece con 2 grupos (silhouette {es(sil[2], 2)}), pero es demasiado "
            "simple: básicamente separa clientes de menor y mayor valor. Con 3 y 4 grupos la calidad es "
            f"prácticamente igual ({es(sil[3], 3)} y {es(sil[4], 3)}). Por eso elegí <b>4 segmentos</b>: una "
            "separación similar, con perfiles más útiles para marketing."
        )
        ed.note(
            "<b>Detalle técnico.</b> El silhouette score mide, para cada cliente, si está más cerca de los de su "
            "propio grupo que de los del grupo más próximo (de −1 a 1). Baja de forma clara a partir de k=5 "
            f"({es(sil[5], 2)})."
        )

# ============================================================ 05 · K-MEANS: CUATRO GRUPOS ==
ed.beat(
    "modelo", "05", "Modelo", "K-Means: cuatro grupos",
    deck=(
        "Después de explorar los datos, elegí 4 segmentos. K-Means busca clientes que tengan comportamientos "
        "parecidos y los agrupa alrededor de cuatro perfiles centrales. El resultado no son cuatro categorías "
        "definidas previamente: son <b>cuatro grupos que aparecen a partir de las similitudes</b> entre los "
        "clientes."
    ),
)
ed.subhead("¿Qué tan claros son los grupos?")
ed.insight(
    "Los grupos son reconocibles, pero sus fronteras no son completamente claras: hay clientes que se "
    "encuentran entre dos perfiles y comparten características de ambos. Y eso es importante: el "
    "comportamiento de los clientes <b>parece formar un continuo</b>, no cuatro cajas perfectamente cerradas.",
    aside=(
        f"Silhouette score: {es(sil_final, 3)}. Va de −1 a 1: cerca de 1 indica grupos muy separados y cerca de "
        "0 indica grupos que se solapan."
    ),
)

ed.subhead("Tamaño de cada segmento", level="story")
with ed.figure("tamanos", level="story"):
    st.plotly_chart(charts.cluster_sizes(sizes_df, CLUSTER_NAMES), use_container_width=True, config=PLOT)
    ed.caption("FIG. 05", f"Número de clientes en cada uno de los 4 clusters (total: {n_fmt}).", FUENTE)

ed.subhead("Los cuatro clusters, proyectados en el mapa", level="wide")
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
            "interpretados: el premium se concentra en la zona de mayor valor económico, y los otros dos se "
            "separan sobre todo en el eje digital."
        )
        ed.note(
            "<b>Detalle técnico.</b> K-Means y t-SNE parten de los mismos datos escalados, así que su "
            "coincidencia es coherente con que haya estructura pero no es una comprobación independiente. La "
            "comprobación independiente es la comparación con el perfil original (beat 07)."
        )

# ============================================================ 06 · LOS CUATRO PERFILES ==
ed.beat(
    "perfiles", "06", "Explicabilidad", "Los cuatro perfiles", weight="minor",
    deck=(
        "<b>Cuatro formas diferentes de comprar.</b> K-Means ha encontrado cuatro grupos con comportamientos "
        "bastante distintos. Ahora dejamos de hablar de clusters y empezamos a hablar de clientes."
    ),
)
CLUSTER_TEXT = {
    PREMIUM: (f"Es el grupo de mayor valor económico. Gasta de media {eur(p.loc[PREMIUM, 'Total_Spending'])} y "
              f"aproximadamente el {pct(share_online[PREMIUM], 0)} de sus compras son online. También es el grupo "
              f"que ha comprado más recientemente, con una media de {p.loc[PREMIUM, 'Days_Since_Last_Purchase']:.0f} "
              "días desde la última compra. <b>En una frase:</b> clientes de alto valor, muy activos y con la mayor "
              "actividad en la web."),
    TIENDA: (f"Gasta de media {eur(p.loc[TIENDA, 'Total_Spending'])}, pero tiene un comportamiento diferente al "
             f"Premium: es el grupo menos digital (el {pct(share_online[TIENDA], 0)} de sus compras son online) y el "
             "que más compra en tienda física. <b>En una frase:</b> clientes de valor medio-alto, con una clara "
             "preferencia por la tienda física."),
    DIGITAL: (f"Es el grupo más numeroso. Son los clientes más jóvenes, con una edad media de "
              f"{p.loc[DIGITAL, 'Age']:.0f} años, y realizan el {pct(share_online[DIGITAL], 0)} de sus compras online. "
              f"Su ticket medio es el más bajo: {eur(p.loc[DIGITAL, 'Average_Ticket'])}. <b>En una frase:</b> muchos "
              "clientes, jóvenes y digitales, con el ticket medio más bajo de los cuatro."),
    BAJA: (f"Es el grupo de menor gasto medio ({eur(p.loc[BAJA, 'Total_Spending'])}) y también el que lleva más "
           f"tiempo sin comprar: {p.loc[BAJA, 'Days_Since_Last_Purchase']:.0f} días de media. Es una señal clara de "
           "baja actividad, aunque no podemos afirmar que estos clientes estén abandonando la tienda porque el "
           "dataset no contiene una variable de abandono. <b>En una frase:</b> clientes de bajo valor actual y con "
           "poca actividad reciente."),
}
with ed.figure("perfiles"):
    ed.cards([
        {"tag": f"Cluster {cid}", "title": CLUSTER_NAMES[cid], "color": CLUSTER_COLORS[cid],
         "stat": f"{es(size_pct[cid])}%", "stat_label": f"{miles(size_n[cid])} clientes", "text": CLUSTER_TEXT[cid]}
        for cid in [PREMIUM, TIENDA, DIGITAL, BAJA]
    ], count=4)

ed.subhead("Comparación directa: gasto total por cluster", level="wide")
with ed.split("gasto-cluster", "5-7") as (txt, viz):
    with txt:
        ed.insight(
            f"El cliente Premium gasta, de media, <b>{es(ratio_spend)} veces más</b> que el grupo de menor valor. "
            "Esta diferencia muestra que no estamos ante pequeñas variaciones de comportamiento, sino ante perfiles "
            "de cliente con un valor económico muy distinto y, por tanto, con oportunidades comerciales diferentes."
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
        "<b>¿Encontró el algoritmo algo parecido a lo que ya conocíamos?</b> Hasta ahora, el modelo ha trabajado "
        "sin conocer los perfiles originales. Ahora podemos hacer una comprobación interesante: el dataset incluía "
        "una clasificación previa de los clientes en cinco perfiles, que no se utilizó para entrenar K-Means. La "
        "comparamos ahora con los cuatro grupos descubiertos por el modelo para ver hasta qué punto coinciden."
    ),
)
with ed.figure("crosstab", level="full"):
    st.plotly_chart(charts.crosstab_heatmap(crosstab_df, CLUSTER_NAMES), use_container_width=True, config=PLOT)
    ed.caption("FIG. 08", "Número de clientes por cluster de K-Means (filas) y por perfil original (columnas).", FUENTE)

ed.subhead("El grupo Premium es especialmente consistente")
ed.insight(
    f"De los {prem_size} clientes que K-Means coloca en el grupo Premium, <b>{prem_n} ya pertenecían al perfil 2 "
    f"original</b>: {prem_pureza * 100:.0f} de cada 100. Eso significa que el {pct(prem_pureza)} de los clientes de "
    "este grupo compartían el mismo perfil que la segmentación original. Pero hay una segunda lectura importante: "
    f"K-Means solo recupera al <b>{pct(prem_cobertura)}</b> de todos los clientes que originalmente pertenecían al "
    "perfil 2. Por tanto, el modelo encuentra un grupo Premium muy limpio, pero no reproduce exactamente toda la "
    "segmentación original.",
    aside=(
        f"Pureza del grupo (precisión) = {pct(prem_pureza)} · cobertura del perfil (recall) = {pct(prem_cobertura)}. "
        f"El perfil 2 es el de mayor gasto medio ({eur(profile_spend[2])}), por eso se asocia al grupo Premium. En el "
        f"otro extremo, el grupo de baja actividad reúne al {pct(baja_p4_cobertura)} de los clientes del perfil 4, "
        f"aunque solo el {pct(baja_p4_pureza)} de sus clientes son perfil 4."
    ),
)

ed.subhead("Lo que el modelo no puede ver")
ed.insight(
    f"Y aquí aparece algo interesante: el grupo digital mezcla casi por igual los perfiles 1 y 5 originales "
    f"({pct(dig_p1)} y {pct(dig_p5)}). Esto no significa necesariamente que K-Means esté fallando: sugiere que las "
    "9 variables utilizadas para crear los segmentos <b>no contienen suficiente información</b> para distinguirlos "
    "claramente. Cuando miramos las variables que dejamos fuera, encontramos diferencias en aspectos como los "
    f"cupones utilizados ({es(p1c['Coupons_Used'], 1)} de media en el perfil 1 frente a "
    f"{es(p5c['Coupons_Used'], 1)} en el perfil 5) y la antigüedad del cliente ({es(p1c['Customer_Tenure'], 1)} "
    f"frente a {es(p5c['Customer_Tenure'], 1)}). Es una buena demostración de algo importante en Machine Learning: "
    "<b>el modelo solo puede descubrir aquello que los datos le permiten ver</b>.",
    aside=(
        f"Satisfacción ({es(p1c['Satisfaction'], 2)} y {es(p5c['Satisfaction'], 2)}) y devoluciones "
        f"({es(p1c['Returns'], 2)} y {es(p5c['Returns'], 2)}) casi no cambian entre ambos perfiles. No se ha "
        "comprobado si añadir estas variables al modelo separaría los perfiles."
    ),
)

# ============================================================ 08 · PONLO A PRUEBA ==
ed.beat(
    "playground", "08", "Playground", "Ponlo a prueba", weight="major",
    deck=(
        "Ahora puedes crear un cliente y ver <b>a qué segmento se parece más</b>. Modifica su gasto, frecuencia de "
        "compra, actividad online o tiempo desde la última compra y observa cómo cambia el resultado. El modelo "
        "compara ese nuevo perfil con los cuatro grupos que ha aprendido y lo asigna al segmento cuyo comportamiento "
        "es más parecido. ¿Qué ocurre cuando acercas el cliente a la frontera entre dos perfiles?"
    ),
)
ed.note("<b>Detalle técnico.</b> Se escalan los valores de entrada con los mismos parámetros del entrenamiento y se "
        "busca el centroide más cercano: es exactamente el cálculo que hace K-Means al asignar un cliente.")

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
        ed.cards([{
            "tag": "Este cliente se parece más a", "title": f"Cluster {cluster_id} — {CLUSTER_NAMES[cluster_id]}",
            "color": CLUSTER_COLORS[cluster_id],
            "text": f"El segundo grupo más parecido es el Cluster {runner_up_id} ({CLUSTER_NAMES[runner_up_id]}).",
        }])
        st.plotly_chart(
            charts.playground_radar(user_scaled, centroid_scaled, FEATURES, FEATURE_LABELS),
            use_container_width=True, config=PLOT,
        )
        ed.caption("FIG. 09", "Perfil del cliente frente al centro de su grupo, en valores estandarizados.",
                   "Modelo entrenado en este proyecto")

ed.subhead("¿Por qué este grupo?")
ed.insight(
    f"De los cuatro grupos, el más parecido a este cliente es el del <b>Cluster {cluster_id} "
    f"({CLUSTER_NAMES[cluster_id]})</b>; el segundo es el Cluster {runner_up_id} ({CLUSTER_NAMES[runner_up_id]}). "
    "Cuanto más parecidas sean las dos distancias, más «fronterizo» es este cliente entre ambos perfiles, igual que "
    "ocurre con muchos clientes de la base."
)

# ============================================================ 09 · ¿QUÉ HEMOS DESCUBIERTO? ==
ed.beat(
    "resultado", "09", "Resultado", "¿Qué hemos descubierto?", weight="major",
    deck=[
        "K-Means ha encontrado cuatro grupos de clientes con comportamientos diferenciados utilizando únicamente "
        f"{stats['n_features_used']} variables de compra y actividad. El resultado más claro aparece en el grupo "
        f"Premium: el <b>{pct(prem_pureza)}</b> de los clientes de este cluster ya pertenecían al perfil 2 original, "
        "aunque el modelo no había visto esa clasificación.",
        "Pero también hemos encontrado los límites de la segmentación. El modelo no reproduce perfectamente los "
        "perfiles originales y mezcla especialmente los perfiles 1 y 5, probablemente porque sus diferencias están "
        "relacionadas con variables que no utilizamos.",
    ],
)
ed.subhead("En resumen")
ed.metrics([
    ("Perfiles de comportamiento", str(stats["n_clusters"]),
     f"Con solo {stats['n_features_used']} variables de compra y actividad, K-Means encuentra cuatro grupos "
     "diferenciados."),
    ("Grupo Premium", pct(prem_pureza),
     f"{prem_pureza * 100:.0f} de cada 100 clientes del grupo Premium pertenecían al mismo perfil original (pureza)."),
    ("Perfil Premium original", pct(prem_cobertura),
     f"Solo {prem_cobertura * 100:.0f} de cada 100 clientes del perfil 2 acaban en el grupo Premium (cobertura)."),
    ("Claridad de los grupos", es(sil_final, 3),
     "Los grupos son reconocibles, pero sus fronteras no son del todo claras (silhouette score)."),
])
ed.insight(
    "No hemos encontrado una copia exacta de la segmentación original. Hemos encontrado <b>algo distinto y "
    "útil</b>: una segmentación basada únicamente en comportamiento que permite identificar patrones "
    "reconocibles y saber dónde funcionan —y dónde no— esos patrones."
)

# ============================================================ 10 · ¿QUÉ PODRÍA HACER MARKETING? ==
ed.beat(
    "implicaciones", "10", "Implicaciones", "¿Qué podría hacer marketing con estos perfiles?",
    deck=(
        "La segmentación no dice qué campaña funcionará. Lo que hace es proporcionar <b>grupos de clientes con "
        "comportamientos diferentes</b> sobre los que probar estrategias distintas."
    ),
)
with ed.figure("hipotesis"):
    ed.cards([
        {"tag": f"Reactivación · Cluster {BAJA}", "title": CLUSTER_NAMES[BAJA], "color": CLUSTER_COLORS[BAJA],
         "text": f"{miles(size_n[BAJA])} clientes llevan una media de {p.loc[BAJA, 'Days_Since_Last_Purchase']:.0f} "
                 "días sin comprar. Podría ser un grupo interesante para probar una campaña de reactivación y medir "
                 "cuántos clientes vuelven a comprar."},
        {"tag": f"Fidelización · Cluster {PREMIUM}", "title": CLUSTER_NAMES[PREMIUM], "color": CLUSTER_COLORS[PREMIUM],
         "text": "Es el grupo con mayor gasto y mayor actividad. Podría plantearse una estrategia específica de "
                 "fidelización para proteger este valor y medir su impacto sobre la retención."},
        {"tag": f"Aumentar valor · Cluster {DIGITAL}", "title": CLUSTER_NAMES[DIGITAL], "color": CLUSTER_COLORS[DIGITAL],
         "text": f"Es el grupo más numeroso y tiene un ticket medio de {eur(p.loc[DIGITAL, 'Average_Ticket'])}. "
                 "Podrían probarse estrategias de cross-selling o aumento de ticket, siempre validándolas mediante "
                 "un experimento."},
    ])
ed.subhead("Una advertencia importante")
ed.insight(
    "Estas son <b>hipótesis de negocio</b>, no conclusiones del modelo. La segmentación identifica grupos; para "
    "saber qué acción funciona con cada uno habría que probarla y medir su impacto.",
    tight=True,
)

# ============================================================ 11 · LO QUE SABEMOS Y LO QUE NO SABEMOS ==
ed.beat("limitaciones", "11", "Limitaciones", "Lo que sabemos y lo que no sabemos")
ed.cards([
    {"title": "Lo que el modelo sí puede hacer", "color": "var(--positive)", "points": [
        "Encontrar <b>grupos de clientes</b> con comportamientos de compra diferentes.",
        "Identificar un <b>grupo Premium especialmente consistente</b>.",
        "Describir las características de cada grupo <b>en términos que marketing puede interpretar</b>.",
        "Mostrar qué perfiles son fáciles de distinguir y <b>cuáles necesitan más información</b>.",
    ]},
    {"title": "Lo que el modelo no puede hacer", "color": "var(--negative)", "points": [
        "<b>Reproducir exactamente</b> la segmentación original.",
        "Distinguir claramente los <b>perfiles 1 y 5</b> utilizando solo las variables seleccionadas.",
        "Explicar <b>por qué</b> un cliente se comporta de una determinada manera.",
        "Demostrar que un cliente del grupo de baja actividad vaya a <b>abandonar</b> la tienda.",
        "Garantizar que los mismos grupos aparezcan igual cuando <b>cambien los clientes o sus hábitos</b> de compra.",
        "Demostrar que una determinada <b>campaña</b> vaya a funcionar mejor para un segmento.",
    ]},
])
ed.passage(
    f"Y hay una última limitación importante: los clusters se entrenaron con este único conjunto de {n_fmt} "
    "clientes. Para llevar esta segmentación a un entorno real habría que <b>comprobar su estabilidad</b> con "
    "nuevos datos y diferentes configuraciones del modelo.",
    aside=(
        "La segmentación se calculó sobre todos los clientes y no hay conjunto de prueba. K-Means se entrenó con una "
        "sola semilla (random_state=42): no se ha comprobado la estabilidad de los clusters con otras semillas."
    ),
)

# ============================================================ 12 · DEL DATO A LA DECISIÓN ==
ed.beat(
    "conclusion", "12", "Conclusión", "Del dato a la decisión", weight="major",
    deck=[
        f"Con solo {stats['n_features_used']} variables de comportamiento, K-Means ha encontrado cuatro grupos de "
        "clientes con características diferentes. El resultado más claro es el segmento Premium: el "
        f"<b>{pct(prem_pureza)}</b> de los clientes que el modelo reunió en este grupo ya pertenecían al perfil 2 "
        "original, aunque esa información nunca se utilizó durante el entrenamiento.",
        "Pero el proyecto también muestra algo igual de importante: una segmentación no tiene por qué reproducir "
        "exactamente una clasificación existente para ser útil. Los datos permiten descubrir patrones, pero también "
        "muestran qué información falta para entender mejor a determinados clientes.",
        "En este caso, el siguiente paso no sería simplemente crear más segmentos. Sería preguntarse: <b>¿qué "
        "información adicional necesitamos para entender mejor a nuestros clientes y qué decisiones podemos tomar "
        "con cada perfil?</b>",
    ],
)
ed.colophon("Borja Mora Méndez", [
    ("Repositorio del proyecto", "https://github.com/BORJAMOME/segmentacion-retail-app"),
    ("Portfolio", "https://borjamora.es/"),
    ("LinkedIn", "https://www.linkedin.com/in/borja-mora-mendez/"),
    ("Contacto", "mailto:borja.mora.mendez@gmail.com"),
])
