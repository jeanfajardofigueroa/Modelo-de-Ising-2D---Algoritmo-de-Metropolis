# Modelo de Ising 2D — Algoritmo de Metrópolis

Simulación del modelo de Ising bidimensional en red cuadrada mediante el algoritmo de Metrópolis, desarrollada para la Tarea 3 de Mecánica Estadística (2026-2).

## Estructura del proyecto

* `ModeloIsing2D-Metropolis.ipynb` (recomendado para ejecutar).
* `ModeloIsing2D-Metropolis.py`.
* `resultados/` (contiene las 3 figuras y las 2 tablas generadas por la simulación).

## Requisitos

* Python 3.x
* NumPy
* Matplotlib
* Pandas

## Cómo funciona

El código implementa el algoritmo de Metrópolis para el modelo de Ising con condiciones de frontera periódicas. La simulación se desarrolla en tres etapas:

1. **Diagnóstico de termalización:** ejecuta la simulación a dos temperaturas y grafica la energía por espín para identificar visualmente el final del régimen transitorio (`n\_termalizacion\_demo`).
2. **Prueba a dos temperaturas:** usando ese punto de corte, descarta la fase transitoria y calcula los promedios de equilibrio de la energía y la magnetización.
3. **Barrido de temperaturas:** repite automáticamente el proceso para un conjunto de temperaturas, ajustando el número de MCS cerca de la temperatura crítica y generando las figuras y tablas finales.

## Ejecución

Se recomienda ejecutar el **Jupyter Notebook**, ya que permite realizar cada etapa de forma secuencial.

Si se ejecuta el archivo `.py` completo, puede descomentarse la línea `input()` para ingresar manualmente `n\_termalizacion\_demo` después de inspeccionar la primera gráfica.

## Notas

* En el `.py`, las líneas que guardan figuras y tablas están comentadas; en el `.ipynb` permanecen activas.
* Si algún archivo dentro de `resultados/` está abierto durante la ejecución, el programa puede producir un error al intentar sobrescribirlo.

