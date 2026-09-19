# Segmentación de Clientes Retail

**¿Trata tu negocio a todos los clientes como si fueran el mismo, aunque no compren igual?**

Una aplicación interactiva que cuenta, paso a paso, cómo descubrí 4 grupos de clientes
en una cadena de electrónica de consumo usando solo su comportamiento de compra, y cuánto se
parecen al perfil que ya traían asignado los datos, sin haberlo visto nunca durante el entrenamiento.

No hace falta saber nada de Machine Learning para seguirla: empieza por el problema, sigue por
los datos, y termina dejándote construir un cliente hipotético para ver en qué segmento caería.

## Ver la app

🔗 **[Abrir la app](https://segmentacion-retail.streamlit.app)**

## De qué trata, en dos frases

Una cadena de electrónica trata a todos sus clientes igual: las mismas ofertas, el mismo
descuento genérico. Con **K-Means** y **t-SNE** agrupé a 6.457 clientes en 4 segmentos
usando solo 9 variables de comportamiento (gasto, frecuencia, canal, recencia), sin usar la
etiqueta de perfil que ya venía en los datos.

**El resultado:** el grupo premium que encuentra el modelo es muy limpio —el 96,8% de sus clientes
ya eran perfil 2 en los datos originales—, pero reúne solo al 71,3% de los clientes de ese perfil:
no es una réplica exacta de la segmentación original. (96,8% es la *pureza* del cluster; 71,3% es la
*cobertura* del perfil.)

## Qué te vas a encontrar al recorrerla

La app se lee como un reportaje, en 12 tramos: **el problema**, **los datos**, **antes de modelar**,
**el camino hasta el modelo**, **K-Means: cuatro grupos**, **los cuatro perfiles**, **¿coincide con el
perfil original?**, **ponlo a prueba** (playground), **el resultado**, **¿qué podría hacer una
empresa?**, **limitaciones** y **del dato a la decisión**.

## Cómo está hecho

Python + [Streamlit](https://streamlit.io) para la aplicación, y
[scikit-learn](https://scikit-learn.org) (`TSNE`, `KMeans`) para el modelo. El análisis
completo, en formato notebook, está en el
[repositorio de portfolio](https://github.com/BORJAMOME/Data-Analytics-Portfolio/tree/main/03-Machine-Learning/02-no-supervisado/clustering/kmeans/04-segmentacion-retail-tsne).

Calculé todos los números que aparecen en la app una vez en `model/train.py` y los guardé como
datos: nada está escrito a mano.

## Ejecutarla en tu ordenador

```bash
pip install -r requirements.txt
streamlit run app.py
```

Los resultados del modelo ya vienen calculados en `model/artifacts/`, así que no hace falta
reentrenar nada para verla funcionar.

Solo si cambias el dataset (`data/clientes_mediamarkt.xlsx`) necesitas regenerarlos:

```bash
python model/train.py    # tarda 1-3 minutos (el paso más lento es ajustar t-SNE)
```

<details>
<summary>Estructura del proyecto, para quien quiera curiosear el código</summary>

```
app.py                    la aplicación — contenido y datos, beat a beat
components/
  editorial.py              sistema editorial: un componente por función narrativa (lede, beat, figure…)
  ui.py                    bloques visuales heredados de versiones anteriores (ya no los usa app.py)
  charts.py                 gráficos, con la paleta de colores del proyecto
utils/
  data_loader.py             carga de datos y resultados (con cache de Streamlit)
  clustering.py               predice el cluster de un cliente hipotético en vivo
model/
  train.py                    ajusta t-SNE y K-Means, calcula todos los resultados
  export_profile_controls.py  medias por perfil original de variables no usadas (sin tocar el modelo)
  artifacts/                   resultados ya calculados (perfiles, coordenadas t-SNE...)
data/                      el dataset original
assets/editorial.css       sistema de composición editorial (rejilla, niveles de ancho, ritmo, tokens) — reutilizable
assets/style.css           identidad de este proyecto (paleta y familias tipográficas)
```

Hice que el Playground predijera el cluster de un cliente hipotético sin cargar un modelo de
sklearn: solo necesita los parámetros del `StandardScaler` y los 4 centroides ya ajustados
(guardados como JSON) para escalar la entrada y buscar el centroide más cercano, exactamente lo
que hace `KMeans.predict()` por dentro, sin la sobrecarga de deserializar un pickle.
</details>

---

### Sistema editorial (reutilizable en otros proyectos de ML)

`assets/editorial.css` convierte el contenedor de Streamlit en una rejilla con cinco niveles de ancho
(`--canvas-width` / `--wide-width` / `--story-width` / `--reading-width` / `--aside-width`), una escala de
espacio (`--space-xs` … `--space-3xl`) y una escala tipográfica editorial. `components/editorial.py`
expone un componente por función narrativa. Para reutilizarlo: copiar ambos archivos y definir la paleta
en el `style.css` del proyecto. Requiere `streamlit==1.58.0` (usa `st.container(key=...)` y el DOM de esa
versión).

**Autor:** Borja Mora Méndez · [LinkedIn](https://www.linkedin.com/in/borjamoramendez/) · [GitHub](https://github.com/BORJAMOME)
