# Segmentación de Clientes Retail

Una aplicación interactiva que explica cómo he utilizado **Machine Learning para identificar cuatro segmentos de clientes** en una cadena de electrónica de consumo. A través de una narrativa visual, exploro sus hábitos de compra, comparo los grupos obtenidos con los perfiles originales del dataset y planteo posibles oportunidades de negocio.

No necesitas conocimientos previos de Machine Learning: la aplicación te guía desde el problema de negocio y la preparación de los datos hasta la interpretación de los resultados. También incluye un simulador para crear un cliente hipotético y explorar a qué segmento podría pertenecer.

## Ver la aplicación

**[Abrir la aplicación interactiva](https://segmentacion-retail.streamlit.app)**

**Tecnologías:** Python · pandas · scikit-learn · K-Means · t-SNE · Plotly · Streamlit

## El análisis

He trabajado con **6.457 clientes y nueve variables de comportamiento**, relacionadas con el gasto, la frecuencia de compra, la recencia y el canal. Con K-Means he identificado cuatro grupos sin utilizar durante el entrenamiento la etiqueta de perfil que ya incluían los datos originales.

El objetivo es comprobar hasta qué punto los segmentos descubiertos a partir del comportamiento de compra se parecen a los perfiles existentes, y qué información pueden aportar para comprender mejor a los clientes.

### Principales resultados

El segmento identificado como premium presenta una **pureza del 96,8 %**: casi todos sus integrantes pertenecían al perfil 2 en los datos originales. Sin embargo, solo reúne al 71,3 % de los clientes de ese perfil.

Esta diferencia es importante: el modelo identifica un grupo bastante homogéneo, pero no reproduce por completo la clasificación original. Los segmentos obtenidos reflejan patrones de comportamiento, no una verdad definitiva sobre cada cliente.

## ¿Qué oportunidades de negocio sugieren los segmentos?

| Segmento | Posible aplicación |
|---|---|
| Baja actividad | Explorar campañas de reactivación para clientes que llevan más tiempo sin comprar. |
| Premium | Evaluar acciones de fidelización para clientes con mayor gasto y actividad. |
| Digital | Investigar oportunidades de venta cruzada o de aumento del ticket medio. |

Estas propuestas son hipótesis que habría que validar mediante experimentos y medición de resultados. El análisis, por sí solo, no demuestra que una campaña vaya a funcionar.

## Qué encontrarás en la aplicación

He organizado el recorrido en 12 etapas, siguiendo un enfoque de *data storytelling*:

- **El problema de negocio:** por qué tiene sentido estudiar el comportamiento de los clientes.
- **Los datos:** qué información utilizo y qué limitaciones tiene.
- **El proceso de modelado:** cómo funciona K-Means y cómo se forman los cuatro grupos.
- **Los perfiles:** qué características distinguen a cada segmento.
- **La comparación con los datos originales:** hasta qué punto coinciden ambas clasificaciones.
- **El simulador interactivo:** cómo se clasificaría un cliente hipotético.
- **Las oportunidades de negocio:** qué acciones podría explorar el área de marketing.
- **Las limitaciones y conclusiones:** qué hemos descubierto y qué queda por comprobar.

La visualización con t-SNE facilita la exploración visual de los grupos, mientras que Plotly permite interactuar con los gráficos y examinar los resultados.

## Limitaciones del análisis

Para interpretar correctamente los resultados, es importante tener en cuenta que:

- El modelo no reproduce exactamente los perfiles originales y tiene dificultades para distinguir los perfiles 1 y 5 con las nueve variables utilizadas.
- K-Means se entrenó con una única semilla y no se ha evaluado con un conjunto de prueba independiente. La estabilidad de los segmentos con datos nuevos queda pendiente de comprobar.
- El dataset no contiene una variable de abandono. Por tanto, el segmento de baja actividad no permite concluir que sus integrantes vayan a dejar de comprar.
- Las asociaciones entre segmentos y posibles acciones comerciales no demuestran causalidad ni garantizan resultados.

## Cómo está construido

La aplicación utiliza Python y Streamlit para presentar el análisis de forma interactiva. El modelado se ha realizado con scikit-learn, mediante K-Means para la segmentación y t-SNE para la representación visual de los datos.

Los resultados que muestra la aplicación se calculan previamente mediante `model/train.py` y se guardan para su consulta, evitando tener que entrenar el modelo cada vez que se abre la aplicación.

**[Consultar el análisis completo en el repositorio de GitHub](https://github.com/BORJAMOME/Data-Analytics-Portfolio/tree/main/03-Machine-Learning/02-no-supervisado/clustering/kmeans/04-segmentacion-retail-tsne)**

## Licencia

El código está disponible bajo la [licencia MIT](LICENSE). Los datasets de `data/` pertenecen a sus fuentes originales y no están cubiertos por esta licencia.

---

**Autor:** Borja Mora Méndez

[Portfolio](https://borjamora.es/) · [LinkedIn](https://www.linkedin.com/in/borjamoramendez/) · [GitHub](https://github.com/BORJAMOME)
