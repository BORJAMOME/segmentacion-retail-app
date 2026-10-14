# Análisis Conjoint: Preferencias de vuelos

¿Qué hace que un vuelo merezca la pena para cada cliente? En este proyecto he analizado cómo influyen el precio, el equipaje, el asiento, las escalas, la flexibilidad y el horario en la valoración de un vuelo.

He desarrollado una aplicación interactiva para entender qué características tienen más peso en la decisión de compra y cómo cambian las preferencias según el tipo de cliente. Además, permite diseñar un vuelo hipotético y comparar cómo lo valoraría cada segmento.


🔗 **[Abrir la aplicación interactiva](https://preferencias-vuelos.streamlit.app/)**

## El análisis

He trabajado con las valoraciones de 1.000 clientes de una aerolínea, que puntuaron del 1 al 10 un conjunto de 24 combinaciones de vuelo. Estas combinaciones forman parte de un diseño experimental fraccionado que recoge distintas opciones de seis atributos, sin necesidad de evaluar todas las combinaciones posibles.

A partir de las 24.000 valoraciones, he aplicado una regresión lineal para estimar la contribución de cada característica y de sus distintos niveles a la valoración global. Este método, conocido como análisis conjoint, permite estudiar qué aspectos influyen más en las preferencias de los clientes.

### Resultados principales

El precio y las escalas reúnen el 69 % de la importancia relativa media en la valoración de los vuelos. Sin embargo, al analizar los segmentos por separado, he encontrado diferencias importantes:

- **Business:** la posibilidad de volar directamente tiene más peso que el precio, con un 37,9 % frente a un 24,4 %.
- **Low Cost:** el precio concentra el 59,7 % de la importancia relativa, por encima del resto de características.

Estos resultados muestran que no todos los clientes valoran un vuelo de la misma manera. Mientras que para un segmento evitar escalas puede ser prioritario, para otro el precio puede tener mucho más peso.

También he calculado intervalos de confianza mediante *bootstrap* de clientes para comprobar la incertidumbre de las estimaciones. Esto ayuda a valorar la precisión de los resultados y a distinguir las diferencias observadas de la variabilidad del muestreo.

## Qué encontrarás en la aplicación

He organizado el análisis en nueve apartados para explicar el problema, los resultados y sus posibles aplicaciones:

1. **El problema:** qué características pueden influir en la elección de un vuelo y por qué conviene estudiar las preferencias de cada cliente.
2. **Los datos:** las 24 combinaciones de vuelo, los seis atributos analizados y las 24.000 valoraciones.
3. **Antes de modelar:** cómo se distribuyen las puntuaciones y qué diferencias hay entre los segmentos.
4. **Cómo funciona un análisis conjoint:** cómo paso de una valoración global a estimar la contribución de cada característica, con una explicación sencilla del método.
5. **El modelo:** resultados de la regresión, utilidades parciales de los atributos y relación entre el precio y la valoración.
6. **Explicabilidad:** importancia relativa de cada atributo para el conjunto de clientes.
7. **Segmentos:** comparación de las preferencias de Business, Low Cost y el tercer segmento analizado.
8. **Prueba interactiva:** diseño de un vuelo hipotético para comparar las valoraciones estimadas de cada segmento. La aplicación distingue entre combinaciones observadas y estimaciones, y avisa cuando una predicción queda fuera de la escala de valoración.
9. **Resultados, decisiones y límites:** posibles aplicaciones de los hallazgos, hipótesis de negocio y aspectos que conviene tener en cuenta al interpretar el análisis.

## Tecnologías y desarrollo

He desarrollado la aplicación con Python y Streamlit, utilizando `statsmodels` y su implementación de mínimos cuadrados ordinarios (`OLS`) para estimar el modelo de regresión.

He separado el cálculo de los resultados de su presentación en la aplicación. El archivo `model/train.py` genera las métricas y estimaciones que se muestran, mientras que `model/export_design_checks.py` reúne las comprobaciones del diseño experimental y la robustez del análisis.

Entre estas comprobaciones se incluyen la calidad del diseño, el uso de errores agrupados por cliente, los límites de la escala de valoración y los intervalos de confianza por segmento. Los resultados se guardan para que la aplicación pueda mostrarlos sin tener que repetir el análisis cada vez que se abre.

**[Consultar el código y el notebook en GitHub](https://github.com/BORJAMOME/Data-Analytics-Portfolio/tree/main/03-Machine-Learning/01-supervisado/regresion/02-regresion-lineal-multiple/03-preferencias-vuelos-conjoint)**

---

**Borja Mora Méndez**

[Portfolio](https://borjamora.es/) · [LinkedIn](https://www.linkedin.com/in/borjamoramendez/) · [GitHub](https://github.com/BORJAMOME)


