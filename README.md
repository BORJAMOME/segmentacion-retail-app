# 6.457 clientes. ¿Cuántos tipos de comportamiento hay realmente?

En este proyecto he analizado los datos de 6.457 clientes de una cadena de electrónica para identificar grupos con patrones similares y comparar los resultados con una segmentación existente.

**[➡ Abrir el proyecto](https://segmentacion-retail.streamlit.app/)**

## Conociendo a los clientes

He analizado los datos de 6.457 clientes, teniendo en cuenta sus hábitos de compra, gasto y actividad digital. De las 26 variables disponibles, he seleccionado nueve para identificar patrones de comportamiento y evitar información repetida.
El conjunto de datos ya incluía cinco perfiles de clientes, pero no los he utilizado para crear los segmentos. Así he podido comprobar si el modelo era capaz de encontrar grupos similares por sí mismo.

### Cómo he elegido los segmentos

He estandarizado las variables para que las diferencias de escala no condicionaran el cálculo de distancias. Después, he utilizado t-SNE para explorar visualmente los datos y he probado soluciones de entre dos y ocho grupos con K-Means.

## Los cuatro perfiles de clientes

El modelo ha identificado cuatro grupos con diferencias en el gasto, la frecuencia de compra y el uso de los canales digitales.

| Segmento | Clientes | Características principales |
|---|---:|---|
| **Premium** | 978 (15,1 %) | Mayor gasto medio, compras recientes y elevada actividad online. |
| **Valor medio-alto, de tienda** | 1.386 (21,5 %) | Gasto medio-alto y mayor preferencia por la tienda física. |
| **Digital, joven y frecuente** | 2.230 (34,5 %) | Grupo más numeroso, edad media de 34 años y un 87 % de compras online. |
| **Bajo valor y baja actividad** | 1.863 (28,9 %) | Menor gasto medio y una media de 270 días desde la última compra. |


## Tecnologías y desarrollo

He realizado el análisis con Python y scikit-learn, utilizando K-Means para la segmentación y t-SNE para la exploración visual de los datos. Para presentar los resultados y permitir la interacción, he utilizado Streamlit.

He separado el análisis y el entrenamiento del modelo de la aplicación que muestra los resultados. También he reservado los perfiles originales para realizar una comprobación posterior, sin utilizarlos durante la creación de los segmentos.


**Borja Mora Méndez**

[Portfolio](https://borjamora.es/) · [LinkedIn](https://www.linkedin.com/in/borjamoramendez/) · [GitHub](https://github.com/BORJAMOME)


