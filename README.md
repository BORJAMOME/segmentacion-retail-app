# Segmentación de Clientes Retail

**¿Trata tu negocio a todos los clientes como si fueran el mismo, aunque no compren igual?**

Una aplicación interactiva que cuenta, paso a paso, cómo descubrí 4 grupos reales de clientes
en una cadena de electrónica de consumo usando solo su comportamiento de compra, y cuánto se
parecen a los que el negocio ya intuía, sin haberlos visto nunca durante el entrenamiento.

No hace falta saber nada de Machine Learning para seguirla: empieza por el problema, sigue por
los datos, y termina dejándote construir un cliente hipotético para ver en qué segmento caería.

## Ver la app

🔗 **[Abrir la app](https://segmentacion-retail.streamlit.app)**

## De qué trata, en dos frases

Una cadena de electrónica trata a todos sus clientes igual: las mismas ofertas, el mismo
descuento genérico. Con **K-Means** y **t-SNE** agrupé a 6.457 clientes reales en 4 segmentos
usando solo 9 variables de comportamiento (gasto, frecuencia, canal, recencia), sin usar la
etiqueta de perfil que el negocio ya tenía asignada.

**El resultado:** el modelo recupera, sin haberla visto nunca, el 96,8% del segmento premium
que el negocio ya identificaba a mano.

## Qué te vas a encontrar al recorrerla

1. **El problema** — por qué tratar a todos los clientes igual sale caro
2. **Los datos** — 6.457 clientes, 26 variables originales, 9 usadas para segmentar
3. **Antes de modelar** — qué se ve a simple vista antes de tocar ningún algoritmo
4. **El camino hasta el modelo** — cómo se llegó a k=4, sin tecnicismos
5. **K-Means (k=4)** — el modelo, sus 4 clusters proyectados en el mapa t-SNE
6. **Los 4 perfiles** — qué caracteriza a cada segmento, en lenguaje de negocio
7. **Playground** — construye un cliente hipotético y mira en qué segmento cae, en directo
8. **Resultados y decisiones** — qué se descubrió y qué haría marketing con ello

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
app.py                    la aplicación — toda la narrativa, sección a sección
components/
  ui.py                    bloques visuales reutilizables (tarjetas, títulos, callouts)
  charts.py                 gráficos, con la paleta de colores del proyecto
utils/
  data_loader.py             carga de datos y resultados (con cache de Streamlit)
  clustering.py               predice el cluster de un cliente hipotético en vivo
model/
  train.py                    ajusta t-SNE y K-Means, calcula todos los resultados
  artifacts/                   resultados ya calculados (perfiles, coordenadas t-SNE...)
data/                      el dataset original
assets/style.css           el sistema visual de la app
```

Hice que el Playground predijera el cluster de un cliente hipotético sin cargar un modelo de
sklearn: solo necesita los parámetros del `StandardScaler` y los 4 centroides ya ajustados
(guardados como JSON) para escalar la entrada y buscar el centroide más cercano, exactamente lo
que hace `KMeans.predict()` por dentro, sin la sobrecarga de deserializar un pickle.
</details>

---

**Autor:** Borja Mora Méndez · [LinkedIn](https://www.linkedin.com/in/borja-mora-mendez/) · [GitHub](https://github.com/BORJAMOME)
