# Segmentación de Clientes Retail

En este proyecto he utilizado K-Means para agrupar clientes de una cadena de electrónica de consumo según su comportamiento de compra. Después, he desarrollado una aplicación interactiva con Streamlit para explicar el análisis, explorar los resultados y probar cómo se clasificaría un cliente hipotético.

La aplicación recorre el proceso desde los datos iniciales hasta los cuatro segmentos obtenidos, incluyendo una comparación con los perfiles que ya aparecían en el dataset.

## Ver la aplicación

**[Abrir la aplicación interactiva](https://segmentacion-retail.streamlit.app)**

**Tecnologías:** Python · pandas · scikit-learn · K-Means · t-SNE · Plotly · Streamlit

## El análisis

He trabajado con 6.457 clientes y nueve variables relacionadas con su comportamiento de compra, como el gasto, la frecuencia, el tiempo transcurrido desde la última compra y el canal utilizado.

Con K-Means he obtenido cuatro grupos sin utilizar durante el entrenamiento los perfiles originales incluidos en los datos. Después, he comparado ambos resultados para ver en qué medida coincidían.

### Algunos resultados

El segmento que he identificado como premium tiene una pureza del 96,8 %: ese porcentaje de sus integrantes ya pertenecía al perfil 2 del dataset original.

Sin embargo, el segmento solo recoge al 71,3 % de los clientes de ese perfil. Es decir, el modelo encuentra un grupo bastante definido, pero no reproduce exactamente la clasificación original.

También he identificado grupos con distintos niveles de actividad y diferencias en sus hábitos de compra. Estas características permiten plantear posibles acciones de marketing, aunque habría que analizarlas y probarlas antes de sacar conclusiones.

## La aplicación

He organizado el contenido en 12 apartados para poder seguir el análisis paso a paso:

- El problema y los datos utilizados.
- La preparación de los datos antes del modelado.
- Cómo funciona K-Means y cómo se forman los cuatro grupos.
- Las características de cada segmento.
- La comparación con los perfiles originales.
- Un simulador para introducir los datos de un cliente hipotético y ver a qué grupo se asignaría.
- Posibles usos de los resultados en marketing.
- Las limitaciones del análisis y las conclusiones.

Los gráficos interactivos permiten explorar los grupos y observar sus diferencias. También utilizo t-SNE para representar los datos en dos dimensiones y facilitar su interpretación visual.

## Limitaciones

El análisis tiene varios aspectos que conviene tener en cuenta:

- Los grupos obtenidos no coinciden exactamente con los perfiles originales. Con las variables utilizadas, tampoco consigo distinguir bien los perfiles 1 y 5.
- El modelo se ha entrenado con una sola semilla. Queda pendiente comprobar si los grupos se mantienen estables con otras inicializaciones y con datos nuevos.
- El dataset no incluye información sobre abandono, por lo que no puedo afirmar que los clientes con menor actividad vayan a dejar de comprar.

Estas limitaciones forman parte del resultado y ayudan a entender qué se puede interpretar a partir del modelo y qué necesitaría más análisis.

## Código y análisis completo

La aplicación está desarrollada con Python y Streamlit. Para el análisis he utilizado pandas y scikit-learn, con K-Means para la segmentación y t-SNE para la representación visual de los grupos. Plotly se encarga de los gráficos interactivos.

Los cálculos se realizan previamente mediante `model/train.py` y los resultados se guardan para que la aplicación pueda mostrarlos sin entrenar el modelo cada vez que se abre.

**[Ver el código y los notebooks del análisis en GitHub](https://github.com/BORJAMOME/Data-Analytics-Portfolio/tree/main/03-Machine-Learning/02-no-supervisado/clustering/kmeans/04-segmentacion-retail-tsne)**


**Borja Mora Méndez**

[Portfolio](https://borjamora.es/) · [LinkedIn](https://www.linkedin.com/in/borjamoramendez/) · [GitHub](https://github.com/BORJAMOME)


