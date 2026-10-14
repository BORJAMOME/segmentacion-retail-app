# Segmentación de clientes con Machine Learning

¿Podemos identificar distintos tipos de clientes a partir de su comportamiento de compra, sin utilizar los perfiles que ya tienen asignados? En este proyecto he trabajado con los datos de 6.457 clientes de una cadena de electrónica para descubrir grupos con características similares y comprobar hasta qué punto se parecen a una segmentación existente.

He utilizado K-Means para crear los segmentos y t-SNE para explorar visualmente los patrones de los datos. Además, he desarrollado una aplicación interactiva en la que se puede definir un cliente y comprobar a qué grupo se parece más.

**[Abrir la aplicación interactiva](https://segmentacion-retail.streamlit.app/)**

## El análisis

He partido de un conjunto de datos con 6.457 clientes y 26 variables relacionadas con el gasto, la frecuencia de compra, la actividad digital y la relación con la tienda.

Para crear los segmentos, he seleccionado nueve variables de comportamiento y valor del cliente. Antes de entrenar el modelo, he revisado su distribución y las relaciones entre ellas, ya que algunas contienen información muy similar. Por ejemplo, el gasto total y los puntos de fidelidad tienen una correlación de 0,99.

Una decisión importante ha sido reservar el perfil original de cada cliente. El conjunto de datos ya incluía una clasificación en cinco perfiles, pero no la he utilizado para entrenar el modelo. Así he podido comprobar después si K-Means encontraba grupos parecidos utilizando únicamente las variables seleccionadas.

### Cómo he elegido los segmentos

He estandarizado las variables para que las diferencias de escala no condicionaran el cálculo de distancias. Después, he utilizado t-SNE para explorar visualmente los datos y he probado soluciones de entre dos y ocho grupos con K-Means.

Aunque la solución de dos grupos obtenía el mejor resultado estadístico, separaba principalmente a los clientes según su valor económico. Finalmente, he elegido cuatro segmentos porque ofrecían un resultado de separación similar y permitían describir perfiles más útiles para el análisis de negocio.

La solución final obtiene un *silhouette score* de 0,317. Los grupos presentan patrones reconocibles, aunque algunos clientes comparten características de varios segmentos y sus fronteras no están completamente definidas.

## Los cuatro perfiles de clientes

El modelo ha identificado cuatro grupos con diferencias en el gasto, la frecuencia de compra y el uso de los canales digitales.

| Segmento | Clientes | Características principales |
|---|---:|---|
| **Premium** | 978 (15,1 %) | Mayor gasto medio, compras recientes y elevada actividad online. |
| **Valor medio-alto, de tienda** | 1.386 (21,5 %) | Gasto medio-alto y mayor preferencia por la tienda física. |
| **Digital, joven y frecuente** | 2.230 (34,5 %) | Grupo más numeroso, edad media de 34 años y un 87 % de compras online. |
| **Bajo valor y baja actividad** | 1.863 (28,9 %) | Menor gasto medio y una media de 270 días desde la última compra. |

El grupo Premium destaca por un gasto medio de 13.960 €, mientras que el segmento de bajo valor y baja actividad registra una media de 1.803 €. Esta diferencia refleja que los clientes tienen comportamientos y valores económicos muy distintos.

También he comprobado que los promedios describen tendencias de cada grupo, no reglas aplicables a todos sus integrantes. Por eso, interpreto los segmentos como una herramienta para entender mejor a los clientes, no como etiquetas definitivas.

## ¿Coinciden con los perfiles originales?

Una vez entrenado el modelo, he comparado los cuatro grupos obtenidos con los cinco perfiles que ya incluía el conjunto de datos.

El resultado más claro aparece en el segmento Premium: el 96,8 % de los clientes asignados a este grupo pertenecían al perfil 2 original. Sin embargo, el modelo solo recupera al 71,3 % de los clientes que formaban parte de ese perfil.

Esta diferencia es importante. El grupo Premium que he obtenido es bastante consistente, pero no recoge a todos los clientes de alto valor de la clasificación original.

También he encontrado dificultades para distinguir los perfiles originales 1 y 5. El segmento digital mezcla ambos perfiles casi por igual, lo que sugiere que las nueve variables seleccionadas no contienen suficiente información para diferenciarlos. Al revisar otras variables disponibles, he observado diferencias en aspectos como el uso de cupones y la antigüedad del cliente.

Esta comparación me ha permitido identificar tanto los patrones que el modelo consigue recuperar como la información que podría necesitar para mejorar la segmentación.

## Qué podría aportar al negocio

La segmentación permite plantear distintas estrategias para cada grupo, aunque los resultados del modelo no demuestran por sí solos qué acciones funcionarán mejor.

- **Reactivación:** probar campañas dirigidas al grupo de baja actividad y medir cuántos clientes vuelven a comprar.
- **Fidelización:** estudiar acciones específicas para el segmento Premium y evaluar su impacto en la retención.
- **Aumento del ticket medio:** explorar iniciativas de venta cruzada para el grupo digital, joven y frecuente, cuyo ticket medio es de 155 €.

Estas propuestas son hipótesis de negocio. Para comprobar su eficacia, sería necesario probar las acciones y medir sus resultados antes de extraer conclusiones.

## Aplicación interactiva

He desarrollado una aplicación con Streamlit para recorrer el análisis y probar cómo funciona la segmentación.

En ella se puede consultar el problema de negocio, explorar los datos, entender el método utilizado, comparar los perfiles y revisar los resultados de la validación. También incluye un apartado interactivo en el que se pueden modificar las características de un cliente y observar a qué segmento se parece más.

La aplicación utiliza los mismos parámetros de escalado del entrenamiento y compara el cliente con los centroides aprendidos por K-Means. De esta forma, la asignación se realiza con el mismo criterio que he utilizado para construir los grupos.

## Tecnologías y desarrollo

He realizado el análisis con Python y scikit-learn, utilizando K-Means para la segmentación y t-SNE para la exploración visual de los datos. Para presentar los resultados y permitir la interacción, he utilizado Streamlit.

He separado el análisis y el entrenamiento del modelo de la aplicación que muestra los resultados. También he reservado los perfiles originales para realizar una comprobación posterior, sin utilizarlos durante la creación de los segmentos.


**Borja Mora Méndez**

[Portfolio](https://borjamora.es/) · [LinkedIn](https://www.linkedin.com/in/borjamoramendez/) · [GitHub](https://github.com/BORJAMOME)


