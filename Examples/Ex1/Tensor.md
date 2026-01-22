# Dimensiones de Tensores en Deep Learning (C, B, T, S)
**Author:** D.Sc. Aboud Barsekh Onji  
**Institution:** Universidad Anáhuac México - Facultad de Ingeniería  
**Contact:** aboud.barsekh@anahuac.mx  
**ORCID:** 0009-0004-5440-8092  

---


En el ámbito de la Inteligencia Computacional y el Aprendizaje Profundo (Deep Learning), los datos no se procesan como simples hojas de cálculo 2D. Se organizan en **tensores** (estructuras multidimensionales). 

Para comprender errores de arquitectura o "Dimension Mismatch" en MATLAB, es vital entender las siglas que definen la estructura de la memoria: **C, B, T, S**.

## 1. C (Channel / Features) - Canales o Características
Esta dimensión representa la **"profundidad" de la información** en un instante exacto. Define cuántas variables independientes ingresan a la red simultáneamente.

*   **En Series Temporales:** Representa el número de sensores o variables físicas medidas.
    *   *Ejemplo:* Si utilizamos un acelerómetro triaxial (Ejes X, Y, Z), tenemos **C = 3**.
    *   *Ejemplo:* En datos médicos con Presión Arterial, Ritmo Cardíaco y Temperatura, tenemos **C = 3**.
*   **En Visión por Computadora:** Representa los planos de color.
    *   *Ejemplo:* Una imagen RGB tiene **C = 3**. Una imagen en escala de grises tiene **C = 1**.
*   **Restricción de Arquitectura:** La capa de entrada (`InputLayer`) tiene un tamaño **C** fijo. No se puede introducir un dato de 4 canales en una red diseñada para 3 sin modificar la arquitectura.

## 2. B (Batch) - Lote
Es la dimensión de **paralelización computacional**. Matemáticamente, la red procesa una muestra a la vez, pero computacionalmente (en GPU/CPU), procesamos bloques de datos para eficiencia vectorial.

*   **Definición:** Número de ejemplos independientes que se propagan por la red antes de actualizar los pesos.
*   **Comportamiento:** 
    *   Si su conjunto de datos tiene 1,000 muestras y define un `MiniBatchSize = 32`, MATLAB crea internamente tensores donde **B = 32**.
    *   Esto permite calcular el gradiente promedio de 32 errores simultáneamente, haciendo la convergencia más estable y rápida.
*   **Nota:** Durante la inferencia de una sola muestra, B=1.

## 3. T (Time / Sequence) - Tiempo o Secuencia
Esta dimensión es exclusiva de datos secuenciales (RNNs, LSTMs, Transformers). Representa la **duración temporal** del evento.

*   **Definición:** Número de "pasos" (Time Steps) o muestreos temporales en una grabación.
    *   *Ejemplo:* Una grabación de audio de 1 segundo muestreada a 100 Hz tendrá una dimensión **T = 100**.
*   **Procesamiento Recurrente:** La red LSTM despliega su bucle interno $T$ veces, procesando $t=1 \to t=2 \to ... \to t=T$.
*   **Flexibilidad:** A diferencia de la dimensión C, la dimensión **T puede ser variable**. Una misma red entrenada (LTSM/GRU) puede recibir una entrada con $T=50$ y después otra con $T=500$ sin generar error, ya que la memoria recurrente se adapta a la longitud de la secuencia.

## 4. S (Spatial) - Espacial
Se refiere a las dimensiones geométricas (Alto y Ancho), típicas en procesamiento de imágenes.

*   **Contexto:** Utilizada casi exclusivamente en **Redes Neuronales Convolucionales (CNNs)** (e.g., $224 \times 224$ pixeles).
*   **En Series Temporales (1D):** Cuando trabajamos con señales vectoriales y LSTMs, la dimensión espacial **no existe** o se asume implícitamente como 1. Por ello, los reportes de `Network Analyzer` muestran tensores de forma $C \times B \times T$ y omiten la S.

---
**Resumen Visual de un Tensor de Entrada para LSTM:**
Si analizamos una señal de 3 sensores, durante 10 segundos (a 10Hz), procesada en lotes de 64 muestras:
$$ Tensor = 3 (C) \times 64 (B) \times 100 (T) $$