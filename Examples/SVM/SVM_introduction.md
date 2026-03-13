# Support Vector Machines (SVM): Fundamentos Teóricos

**Autor:** Prof. D.Sc. BARSEKH-ONJI Aboud
**Institución:** Facultad de Ingeniería, Universidad Anáhuac México
**Curso:** Redes Neuronales y SVM
**Contacto:** aboud.barsekh@anahuac.mx
**ORCID:** 0009-0004-5440-8092

---

## Tabla de Contenido

1. [Contexto y Motivación](#1-contexto-y-motivación)
2. [El Problema de Clasificación Binaria](#2-el-problema-de-clasificación-binaria)
3. [Hiperplano de Margen Máximo](#3-hiperplano-de-margen-máximo)
4. [Formulación del Problema Primal](#4-formulación-del-problema-primal)
5. [Dualidad de Lagrange y Condiciones KKT](#5-dualidad-de-lagrange-y-condiciones-kkt)
6. [El Problema Dual](#6-el-problema-dual)
7. [SVM de Margen Suave (Soft Margin)](#7-svm-de-margen-suave-soft-margin)
8. [El Truco del Kernel](#8-el-truco-del-kernel)
9. [Kernels Fundamentales](#9-kernels-fundamentales)
10. [SVM Multi-clase](#10-svm-multi-clase)
11. [Support Vector Regression (SVR)](#11-support-vector-regression-svr)
12. [Complejidad Computacional y Algoritmos de Solución](#12-complejidad-computacional-y-algoritmos-de-solución)
13. [Selección de Hiperparámetros](#13-selección-de-hiperparámetros)
14. [Ventajas, Limitaciones y Comparativa](#14-ventajas-limitaciones-y-comparativa)
15. [Referencias](#15-referencias)

---

## 1. Contexto y Motivación

### 1.1 Origen histórico

Las **Máquinas de Vectores de Soporte** (*Support Vector Machines*, SVM) fueron introducidas por Vladimir Vapnik y Corinna Cortes en 1995, consolidando décadas de trabajo teórico de Vapnik en la *Teoría del Aprendizaje Estadístico* (*Statistical Learning Theory*, SLT) desarrollada desde los años 60 en colaboración con Alexey Chervonenkis.

El modelo SVM no surgió como una intuición heurística, sino como la consecuencia directa de un principio teórico sólido: la **minimización del riesgo estructural** (*Structural Risk Minimization*, SRM). Este principio sostiene que el error de generalización de un modelo está acotado por la suma del error de entrenamiento y un término de complejidad del modelo que depende de la dimensión VC (*Vapnik-Chervonenkis dimension*).

Formalmente, con probabilidad al menos $1 - \delta$ sobre la elección del conjunto de entrenamiento de tamaño $N$, el error de generalización $R$ satisface:

$$R \leq R_{\text{emp}} + \sqrt{\frac{h\left(\ln\frac{2N}{h} + 1\right) - \ln\frac{\delta}{4}}{N}}$$

donde $R_{\text{emp}}$ es el error empírico (de entrenamiento), $h$ es la dimensión VC del modelo y $N$ es el número de muestras. Esta cota formaliza por qué modelos más simples generalizan mejor: el término de regularización crece con $h$.

SVM controla $h$ directamente maximizando el margen de separación, lo que conduce de forma natural a un problema de optimización convexo con garantías teóricas de unicidad y globalidad.

### 1.2 ¿Por qué estudiar SVM en el contexto de redes neuronales?

SVM y las redes neuronales comparten el objetivo de aprender una función $f: \mathbb{R}^d \to \mathcal{Y}$ a partir de datos, pero difieren fundamentalmente en su enfoque:

| Aspecto | Redes Neuronales | SVM |
| :--- | :--- | :--- |
| **Principio** | Minimización empírica del riesgo | Minimización estructural del riesgo |
| **Optimización** | No convexa (múltiples mínimos locales) | Convexa (mínimo global único) |
| **Parámetros** | Millones (pesos sinápticos) | Pocos (vectores de soporte + $C$, $\gamma$) |
| **Garantías** | Heurísticas (backpropagation) | Teóricas (cota VC) |
| **Eficiencia** | Requiere muchos datos | Funciona con muestras pequeñas |
| **Kernel** | Implícito (capas ocultas) | Explícito y elegible |

Comprender SVM profundiza la intuición sobre separabilidad, representación de datos en espacios de características, y el balance entre sesgo y varianza.

---

## 2. El Problema de Clasificación Binaria

### 2.1 Notación y formulación del problema

Dado un **conjunto de entrenamiento** de $N$ pares entrada-salida:

$$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}, \quad \mathbf{x}_i \in \mathbb{R}^d, \quad y_i \in \{-1, +1\}$$

El objetivo de clasificación binaria es encontrar una función $f: \mathbb{R}^d \to \{-1, +1\}$ que generalice correctamente a nuevas muestras no vistas.

Un **clasificador lineal** tiene la forma:

$$f(\mathbf{x}) = \text{sign}(\mathbf{w}^\top \mathbf{x} + b)$$

donde $\mathbf{w} \in \mathbb{R}^d$ es el **vector de pesos** (normal al hiperplano) y $b \in \mathbb{R}$ es el **sesgo** (*bias* o término independiente). La función $\text{sign}(z)$ devuelve $+1$ si $z > 0$ y $-1$ si $z < 0$.

Geométricamente, el hiperplano $\mathcal{H} = \{\mathbf{x} \in \mathbb{R}^d : \mathbf{w}^\top \mathbf{x} + b = 0\}$ divide el espacio en dos semiespacios:

- $\mathcal{H}^+ = \{\mathbf{x} : \mathbf{w}^\top \mathbf{x} + b > 0\}$ → clase $+1$
- $\mathcal{H}^- = \{\mathbf{x} : \mathbf{w}^\top \mathbf{x} + b < 0\}$ → clase $-1$

### 2.2 El problema de la ambigüedad: infinitos separadores

Cuando los datos son **linealmente separables**, existen infinitos hiperplanos que los separan correctamente. El perceptrón de Rosenblatt (1958) converge a *alguno* de ellos, dependiendo de la inicialización y el orden de presentación de los datos, sin garantías sobre cuál elige. Esto es problemático porque hiperplanos que se ajustan demasiado al entrenamiento pueden tener pobre generalización.

SVM responde a esta ambigüedad con un criterio geométrico riguroso: elegir el hiperplano que **maximiza la distancia mínima** entre él y los puntos de entrenamiento de ambas clases. Este es el principio del **margen máximo**.

---

## 3. Hiperplano de Margen Máximo

### 3.1 Distancia de un punto a un hiperplano

La distancia euclidiana de un punto $\mathbf{x}_i$ al hiperplano $\mathbf{w}^\top \mathbf{x} + b = 0$ es:

$$d(\mathbf{x}_i, \mathcal{H}) = \frac{|\mathbf{w}^\top \mathbf{x}_i + b|}{\|\mathbf{w}\|}$$

donde $\|\mathbf{w}\| = \sqrt{\mathbf{w}^\top \mathbf{w}}$ es la norma euclidiana del vector de pesos. El denominador $\|\mathbf{w}\|$ es indispensable: sin él, la expresión $\mathbf{w}^\top \mathbf{x}_i + b$ es solo un valor escalar que escala con la magnitud de $\mathbf{w}$, no una distancia geométrica.

Para un punto **correctamente clasificado** de clase $y_i$, el signo de $\mathbf{w}^\top \mathbf{x}_i + b$ coincide con el signo de $y_i$. Esto permite escribir la distancia con signo (siempre positiva para puntos correctamente clasificados) como:

$$d_i = \frac{y_i(\mathbf{w}^\top \mathbf{x}_i + b)}{\|\mathbf{w}\|}$$

Esta es la **distancia funcional con corrección de clase**. Si el punto está correctamente clasificado, $d_i > 0$; si está mal clasificado, $d_i < 0$.

### 3.2 Definición de margen y vectores de soporte

El **margen geométrico** de un clasificador $(\mathbf{w}, b)$ respecto a un conjunto de datos $\mathcal{D}$ es la distancia mínima entre cualquier punto de entrenamiento y el hiperplano:

$$\rho(\mathbf{w}, b) = \min_{i=1,\ldots,N} \frac{y_i(\mathbf{w}^\top \mathbf{x}_i + b)}{\|\mathbf{w}\|}$$

Los puntos de entrenamiento que alcanzan este mínimo, es decir, los más cercanos al hiperplano, se denominan **vectores de soporte** (*support vectors*). Son los únicos puntos que, en última instancia, determinan la posición del hiperplano óptimo.

**Propiedad fundamental:** Si se eliminan todos los puntos de entrenamiento excepto los vectores de soporte, el hiperplano óptimo no cambia.

### 3.3 Normalización canónica

Por conveniencia algebraica, se impone la **normalización canónica**: se escala $(\mathbf{w}, b)$ de modo que los vectores de soporte satisfagan exactamente:

$$y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1$$

Con esta convención, los vectores de soporte de clase $+1$ se encuentran sobre el hiperplano $\mathbf{w}^\top \mathbf{x} + b = +1$, y los de clase $-1$ sobre $\mathbf{w}^\top \mathbf{x} + b = -1$.

La distancia de cada hiperplano de margen al hiperplano central es:

$$d_+ = \frac{y_i(\mathbf{w}^\top \mathbf{x}_i + b)}{\|\mathbf{w}\|} = \frac{1 \cdot 1}{\|\mathbf{w}\|} = \frac{1}{\|\mathbf{w}\|}$$

El margen total (suma de las dos distancias) es:

$$\rho = \frac{1}{\|\mathbf{w}\|} + \frac{1}{\|\mathbf{w}\|} = \frac{2}{\|\mathbf{w}\|}$$

**Maximizar el margen** $\rho = 2/\|\mathbf{w}\|$ es equivalente a **minimizar** $\|\mathbf{w}\|$, o por conveniencia diferencial, minimizar $\frac{1}{2}\|\mathbf{w}\|^2$.

---

## 4. Formulación del Problema Primal

### 4.1 Problema de optimización (caso separable)

Con la normalización canónica, el problema de margen máximo se escribe formalmente como:

$$\min_{\mathbf{w}, b} \quad \frac{1}{2}\|\mathbf{w}\|^2$$

$$\text{s.a.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1, \quad \forall \, i = 1, \ldots, N$$

Este es un problema de **programación cuadrática convexa** (*Quadratic Programming*, QP):

- La función objetivo $\frac{1}{2}\|\mathbf{w}\|^2$ es **cuadrática y convexa**
- Las restricciones son **lineales** en $(\mathbf{w}, b)$
- La región factible (intersección de semiespacios) es **convexa**

Por la teoría de optimización convexa, este problema tiene **solución única** (mínimo global), lo que diferencia fundamentalmente a SVM de las redes neuronales (cuya función de pérdida es no convexa con múltiples mínimos locales).

### 4.2 Interpretación geométrica del problema

Las $N$ restricciones $y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1$ establecen que:

- Todos los puntos de clase $+1$ deben estar en el semiplano $\mathbf{w}^\top \mathbf{x} + b \geq 1$
- Todos los puntos de clase $-1$ deben estar en el semiplano $\mathbf{w}^\top \mathbf{x} + b \leq -1$
- La región $-1 < \mathbf{w}^\top \mathbf{x} + b < 1$ es la **franja de margen**, que debe estar vacía de puntos de entrenamiento

Minimizar $\|\mathbf{w}\|^2$ maximiza la anchura de esta franja.

---

## 5. Dualidad de Lagrange y Condiciones KKT

### 5.1 El Lagrangiano del problema primal

Para resolver el problema con restricciones de desigualdad, se introduce el **Lagrangiano**. Se asocia un multiplicador de Lagrange $\alpha_i \geq 0$ a cada restricción:

$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^{N} \alpha_i \left[y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1\right]$$

La estrategia de Lagrange convierte el problema con restricciones en uno sin restricciones: el punto de silla del Lagrangiano (mínimo respecto a $\mathbf{w}, b$ y máximo respecto a $\boldsymbol{\alpha}$) coincide con la solución del problema original.

### 5.2 Condiciones de optimalidad KKT

Las condiciones de Karush-Kuhn-Tucker (KKT) son necesarias y suficientes para la optimalidad en problemas convexos. Se derivan igualando a cero las derivadas parciales del Lagrangiano:

**Condición sobre $\mathbf{w}$:**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \mathbf{w} - \sum_{i=1}^{N} \alpha_i y_i \mathbf{x}_i = 0 \quad \Rightarrow \quad \mathbf{w}^* = \sum_{i=1}^{N} \alpha_i y_i \mathbf{x}_i$$

**Condición sobre $b$:**

$$\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^{N} \alpha_i y_i = 0 \quad \Rightarrow \quad \sum_{i=1}^{N} \alpha_i y_i = 0$$

**Condición de holgura complementaria:**

$$\alpha_i \left[y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1\right] = 0, \quad \forall \, i$$

Esta última condición es clave: o bien $\alpha_i = 0$ (el punto $i$ no es vector de soporte y no participa en la solución), o bien $y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1$ (el punto $i$ está exactamente sobre el margen y es un vector de soporte).

### 5.3 Representación del hiperplano en términos de los datos

La condición $\mathbf{w}^* = \sum_{i=1}^{N} \alpha_i y_i \mathbf{x}_i$ tiene una profunda implicación: el vector normal al hiperplano óptimo es una **combinación lineal de los vectores de soporte**. Solo los puntos con $\alpha_i > 0$ contribuyen. La solución es *dispersa* en el espacio de datos.

---

## 6. El Problema Dual

### 6.1 Derivación del dual

Sustituyendo las condiciones KKT en el Lagrangiano:

$$\mathcal{L} = \frac{1}{2}\left\|\sum_i \alpha_i y_i \mathbf{x}_i\right\|^2 - \sum_i \alpha_i y_i \mathbf{x}_i^\top \left(\sum_j \alpha_j y_j \mathbf{x}_j\right) - b\underbrace{\sum_i \alpha_i y_i}_{=0} + \sum_i \alpha_i$$

Simplificando, se obtiene el **problema dual de Wolfe**:

$$\max_{\boldsymbol{\alpha}} \quad W(\boldsymbol{\alpha}) = \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j \, \mathbf{x}_i^\top \mathbf{x}_j$$

$$\text{s.a.} \quad \alpha_i \geq 0 \quad \text{y} \quad \sum_{i=1}^{N} \alpha_i y_i = 0$$

### 6.2 Propiedades del problema dual

El dual presenta ventajas importantes sobre el primal:

1. **La dimensión del problema es $N$ (número de muestras)**, no $d$ (dimensión del espacio de características). Esto es favorable cuando $d \gg N$.
2. **Solo aparecen productos internos** $\mathbf{x}_i^\top \mathbf{x}_j$, lo que abre la puerta directa al truco del kernel.
3. **La función objetivo es cóncava** (maximización de forma cuadrática negativa semi-definida), garantizando solución global.

### 6.3 Recuperación de los parámetros del modelo

Una vez resuelto el dual y obtenidos $\boldsymbol{\alpha}^*$:

$$\mathbf{w}^* = \sum_{i \in SV} \alpha_i^* y_i \mathbf{x}_i$$

El sesgo $b^*$ se recupera usando cualquier vector de soporte $\mathbf{x}_k$ (con $\alpha_k^* > 0$):

$$b^* = y_k - \mathbf{w}^{*\top} \mathbf{x}_k = y_k - \sum_{i \in SV} \alpha_i^* y_i \mathbf{x}_i^\top \mathbf{x}_k$$

En la práctica, se promedia sobre todos los vectores de soporte para mayor estabilidad numérica:

$$b^* = \frac{1}{|SV|} \sum_{k \in SV} \left(y_k - \sum_{i \in SV} \alpha_i^* y_i \mathbf{x}_i^\top \mathbf{x}_k\right)$$

### 6.4 Función de decisión final

La clasificación de un nuevo punto $\mathbf{x}$ se realiza con:

$$f(\mathbf{x}) = \text{sign}\left(\sum_{i \in SV} \alpha_i^* y_i \mathbf{x}_i^\top \mathbf{x} + b^*\right)$$

Solo participan los vectores de soporte, lo que hace la predicción eficiente en memoria y tiempo.

---

## 7. SVM de Margen Suave (Soft Margin)

### 7.1 Motivación: datos no separables linealmente

El caso del margen duro asume que los datos son **perfectamente separables**, lo cual raramente ocurre en problemas reales por dos razones:

1. **Ruido en las etiquetas**: un punto puede estar etiquetado incorrectamente
2. **Solapamiento intrínseco**: las distribuciones de las clases se solapan en el espacio de características

Si los datos no son separables, el problema primal no tiene solución factible. La solución es **relajar las restricciones** permitiendo que algunos puntos violen el margen, penalizando estas violaciones en la función objetivo.

### 7.2 Variables de holgura

Se introduce una variable de holgura (*slack variable*) $\xi_i \geq 0$ para cada punto:

$$y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i$$

La interpretación geométrica de $\xi_i$ es:

- $\xi_i = 0$: el punto está correctamente clasificado y fuera del margen
- $0 < \xi_i \leq 1$: el punto está dentro del margen pero correctamente clasificado
- $\xi_i = 1$: el punto está exactamente sobre el hiperplano separador
- $\xi_i > 1$: el punto está mal clasificado (en el semiplano incorrecto)

El **error de clasificación** del punto $i$ es $\mathbf{1}[\xi_i > 1]$, y el número total de errores está acotado por $\sum_i \xi_i$.

### 7.3 Formulación del Soft Margin SVM

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \xi_i$$

$$\text{s.a.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad \forall \, i$$

El **parámetro de regularización** $C > 0$ controla el balance entre:

- Maximizar el margen (minimizar $\|\mathbf{w}\|^2$)
- Minimizar las violaciones del margen (minimizar $\sum \xi_i$)

### 7.4 Interpretación de $C$ y su conexión con la regularización

$C$ actúa como el recíproco de un parámetro de regularización clásico:

| Valor de $C$ | Comportamiento | Riesgo |
| :--- | :--- | :--- |
| $C \to \infty$ | Penaliza fuertemente los errores → margen duro | *Overfitting* |
| $C$ grande (e.g., $10^3$) | Pocos errores permitidos, margen estrecho | *Overfitting* moderado |
| $C = 1$ | Balance equilibrado (valor por defecto en muchas implementaciones) | — |
| $C$ pequeño (e.g., $10^{-2}$) | Muchos errores permitidos, margen ancho | *Underfitting* |
| $C \to 0$ | Ignora los errores → clasificador trivial | *Underfitting* |

La función de pérdida asociada a las variables de holgura es la **pérdida bisagra** (*hinge loss*):

$$\ell(y_i, f(\mathbf{x}_i)) = \max(0, 1 - y_i f(\mathbf{x}_i))$$

donde $f(\mathbf{x}_i) = \mathbf{w}^\top \mathbf{x}_i + b$ (sin $\text{sign}$). La función objetivo del Soft Margin SVM puede reescribirse como:

$$\min_{\mathbf{w}, b} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \max(0, 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b))$$

Esta forma conecta SVM con la familia de modelos de regularización $\ell_2$ con pérdida bisagra.

### 7.5 Problema dual del Soft Margin

El dual es idéntico al del caso separable, con la única diferencia de una **cota superior** en los multiplicadores:

$$\max_{\boldsymbol{\alpha}} \quad \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j$$

$$\text{s.a.} \quad 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^{N} \alpha_i y_i = 0$$

Los multiplicadores ahora se clasifican en tres grupos:

- $\alpha_i = 0$: punto interior, bien clasificado, fuera del margen
- $0 < \alpha_i < C$: vector de soporte en el margen ($\xi_i = 0$)
- $\alpha_i = C$: punto con violación del margen ($\xi_i > 0$), llamado *bounded support vector*

---

## 8. El Truco del Kernel

### 8.1 Motivación: datos no separables no linealmente

Cuando los datos no son linealmente separables en $\mathbb{R}^d$, la idea es mapearlos a un espacio de mayor dimensión (posiblemente infinita) $\mathcal{H}$ donde sí sean separables:

$$\phi: \mathbb{R}^d \to \mathcal{H}, \quad \mathbf{x} \mapsto \phi(\mathbf{x})$$

En $\mathcal{H}$ se aplica SVM lineal, y la frontera de decisión en el espacio original $\mathbb{R}^d$ resulta no lineal.

**Ejemplo ilustrativo.** Supóngase que $\mathbf{x} = (x_1, x_2) \in \mathbb{R}^2$ y se define:

$$\phi(\mathbf{x}) = (x_1^2, \; \sqrt{2}\,x_1 x_2, \; x_2^2) \in \mathbb{R}^3$$

Entonces:

$$\phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j) = x_{i1}^2 x_{j1}^2 + 2x_{i1}x_{i2}x_{j1}x_{j2} + x_{i2}^2 x_{j2}^2 = (\mathbf{x}_i^\top \mathbf{x}_j)^2$$

Es decir, el producto interno en $\mathbb{R}^3$ es equivalente a elevar al cuadrado el producto interno en $\mathbb{R}^2$, sin necesidad de calcular $\phi$ explícitamente.

### 8.2 Definición formal de función Kernel

Una función $K: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ es un **kernel válido** si existe un mapa $\phi$ tal que:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$$

La condición equivalente (sin necesidad de conocer $\phi$) es el **Teorema de Mercer**: $K$ es un kernel válido si y solo si para cualquier conjunto finito de puntos $\{\mathbf{x}_1, \ldots, \mathbf{x}_N\}$, la **matriz de Gram** $\mathbf{K}$ con entradas $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ es **semidefinida positiva** (SDP):

$$\sum_{i,j} c_i c_j K(\mathbf{x}_i, \mathbf{x}_j) \geq 0, \quad \forall \{c_i\} \subset \mathbb{R}$$

### 8.3 El truco del kernel en el dual SVM

El problema dual solo involucra los datos a través de productos internos $\mathbf{x}_i^\top \mathbf{x}_j$. Sustituyendo por el kernel:

$$\max_{\boldsymbol{\alpha}} \quad \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$

La función de decisión se convierte en:

$$f(\mathbf{x}) = \text{sign}\left(\sum_{i \in SV} \alpha_i^* y_i K(\mathbf{x}_i, \mathbf{x}) + b^*\right)$$

El costo computacional de entrenamiento y predicción **no depende de la dimensión de $\mathcal{H}$** (que puede ser infinita), sino del número de vectores de soporte. Esto es el **truco del kernel** (*kernel trick*).

---

## 9. Kernels Fundamentales

### 9.1 Kernel lineal

$$K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^\top \mathbf{x}_j$$

Corresponde a $\phi(\mathbf{x}) = \mathbf{x}$ (identidad). Genera una frontera de decisión lineal en el espacio original. Se recomienda como punto de partida y para problemas con $d \gg N$.

### 9.2 Kernel polinomial

$$K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^\top \mathbf{x}_j + c)^d, \quad c \geq 0, \; d \in \mathbb{Z}^+$$

El parámetro $c$ controla la influencia de los monomios de menor grado. Para $d = 2$ y $c = 1$, el mapa $\phi$ incluye todas las interacciones de segundo orden entre características. El espacio de características tiene dimensión $\binom{n+d}{d}$.

### 9.3 Kernel RBF (Gaussiano)

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\!\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right), \quad \gamma > 0$$

Es el kernel más utilizado en la práctica. El espacio de características inducido tiene **dimensión infinita** (se puede demostrar mediante la expansión en series de Taylor de la exponencial). El parámetro $\gamma = \frac{1}{2\sigma^2}$ controla el ancho de la gaussiana:

- $\gamma$ pequeño ($\sigma$ grande): función de similitud suave, influencia de largo alcance
- $\gamma$ grande ($\sigma$ pequeño): función de similitud estrecha, influencia local

El kernel RBF satisface la condición de Mercer ya que la función $\exp(-\gamma\|\mathbf{x} - \mathbf{x}'\|^2)$ es una función de correlación de un proceso gaussiano estacionario.

### 9.4 Kernel sigmoide

$$K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\kappa \, \mathbf{x}_i^\top \mathbf{x}_j + \theta)$$

Este kernel tiene conexión directa con las redes neuronales: una SVM con kernel sigmoide es equivalente a una red neuronal de una capa oculta. Sin embargo, **no siempre es semidefinido positivo** (no siempre es un kernel de Mercer válido para todos los valores de $\kappa$ y $\theta$), por lo que debe usarse con cuidado.

### 9.5 Tabla comparativa de kernels

| Kernel | Frontera inducida | Hiperparámetros | Uso recomendado |
| :--- | :--- | :--- | :--- |
| Lineal | Hiperplano | Ninguno | $d \gg N$, texto, imágenes |
| Polinomial | Curva polinomial | $c$, $d$ | Reconocimiento de imágenes |
| RBF | Fronteras suaves no lineales | $\gamma$ | Uso general, punto de partida |
| Sigmoide | Similar a red neuronal | $\kappa$, $\theta$ | Casos específicos |

---

## 10. SVM Multi-clase

SVM es intrínsecamente binario. Se extiende a $K$ clases mediante descomposición en problemas binarios.

### 10.1 Uno contra el Resto (OvR — *One vs Rest*)

Se entrenan $K$ clasificadores SVM. El clasificador $k$ separa la clase $k$ (etiqueta $+1$) de todas las demás clases agrupadas (etiqueta $-1$). La predicción para un nuevo punto $\mathbf{x}$ es:

$$\hat{y} = \arg\max_{k=1,\ldots,K} f_k(\mathbf{x}) = \arg\max_{k} \left(\sum_{i \in SV_k} \alpha_i^{(k)} y_i^{(k)} K(\mathbf{x}_i, \mathbf{x}) + b_k\right)$$

Se requieren $K$ problemas QP, cada uno con $N$ muestras. El problema de clases **desbalanceadas** es una debilidad conocida de OvR.

### 10.2 Uno contra Uno (OvO — *One vs One*)

Se entrenan $\binom{K}{2} = K(K-1)/2$ clasificadores, uno para cada par de clases. Cada clasificador solo usa las muestras de las dos clases involucradas. La predicción se realiza por **votación mayoritaria**: la clase que gana más torneos binarios se asigna al punto.

OvO tiene mejor comportamiento con clases desbalanceadas (los clasificadores binarios solo ven muestras de dos clases a la vez) y es más estable numéricamente. MATLAB y scikit-learn usan OvO por defecto.

---

## 11. Support Vector Regression (SVR)

### 11.1 Formulación del $\epsilon$-SVR

Para problemas de regresión, donde $y_i \in \mathbb{R}$, se define un **tubo de tolerancia** de anchura $2\epsilon$. Las predicciones dentro del tubo no generan error; solo las que lo superan contribuyen a la función objetivo.

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{N}(\xi_i + \xi_i^*)$$

$$\text{s.a.} \quad \begin{cases} y_i - (\mathbf{w}^\top \mathbf{x}_i + b) \leq \epsilon + \xi_i \\ (\mathbf{w}^\top \mathbf{x}_i + b) - y_i \leq \epsilon + \xi_i^* \\ \xi_i, \xi_i^* \geq 0 \end{cases}$$

donde $\xi_i$ y $\xi_i^*$ son las holguras superior e inferior respectivamente.

### 11.2 Función de pérdida $\epsilon$-insensible

La función de pérdida asociada al $\epsilon$-SVR es:

$$\ell_\epsilon(y_i, f(\mathbf{x}_i)) = \max(0, |y_i - f(\mathbf{x}_i)| - \epsilon)$$

Esta pérdida es cero dentro del tubo y lineal fuera de él, a diferencia del error cuadrático de la regresión ordinaria. La función de regresión final tiene la misma forma que el clasificador SVM:

$$f(\mathbf{x}) = \sum_{i \in SV} (\alpha_i^* - \alpha_i^{**}) K(\mathbf{x}_i, \mathbf{x}) + b^*$$

donde $\alpha_i^*$ y $\alpha_i^{**}$ son los multiplicadores duales asociados a las holguras superior e inferior.

---

## 12. Complejidad Computacional y Algoritmos de Solución

### 12.1 Complejidad del entrenamiento

El problema dual SVM es un QP de tamaño $N \times N$. Los algoritmos directos (como los basados en factorización de Cholesky) tienen complejidad:

- **Tiempo:** $\mathcal{O}(N^2)$ a $\mathcal{O}(N^3)$
- **Memoria:** $\mathcal{O}(N^2)$ para almacenar la matriz de Gram

Esto hace que SVM sea impráctica para $N > 10^5$ con implementaciones directas.

### 12.2 Algoritmo SMO (*Sequential Minimal Optimization*)

John Platt (1998) propuso SMO como solución al problema de escala. En lugar de resolver el QP completo, SMO descompone el problema en subproblemas de **tamaño 2**: en cada iteración, selecciona dos multiplicadores $\alpha_i, \alpha_j$ y los optimiza analíticamente mientras mantiene fijos los demás.

La actualización analítica es posible porque con solo dos variables la restricción $\sum \alpha_i y_i = 0$ se reduce a una ecuación en una incógnita. El proceso itera hasta convergencia (criterio KKT). SMO tiene complejidad empírica entre $\mathcal{O}(N)$ y $\mathcal{O}(N^2)$ según el problema.

MATLAB usa variantes de SMO en `fitcsvm` y `fitrsvm`.

### 12.3 Predicción

El tiempo de predicción para un nuevo punto es $\mathcal{O}(N_{SV})$ donde $N_{SV}$ es el número de vectores de soporte. En la práctica, $N_{SV} \ll N$, haciendo la predicción muy eficiente.

---

## 13. Selección de Hiperparámetros

### 13.1 Hiperparámetros del modelo

Para SVM con kernel RBF, los hiperparámetros a ajustar son:

- $C > 0$: regularización (control de margen vs. errores)
- $\gamma > 0$: anchura del kernel RBF

Para kernel polinomial: $C$, $d$, $c$.

### 13.2 Búsqueda en grilla con validación cruzada

El procedimiento estándar es una **búsqueda en grilla logarítmica** con validación cruzada $k$-fold ($k = 5$ o $k = 10$):

$$C \in \{10^{-3}, 10^{-2}, \ldots, 10^3\}, \quad \gamma \in \{10^{-5}, 10^{-4}, \ldots, 10^1\}$$

Para cada par $(C, \gamma)$, se evalúa el error de validación cruzada. Se selecciona el par con menor error, y se entrena el modelo final con todos los datos de entrenamiento.

La grilla logarítmica es apropiada porque SVM es moderadamente insensible a cambios de un orden de magnitud en los hiperparámetros, pero muy sensible a cambios de varios órdenes.

### 13.3 Análisis de la frontera de decisión

Un indicador práctico de selección inadecuada de hiperparámetros:

- **Demasiados vectores de soporte** ($N_{SV} \approx N$): $C$ demasiado pequeño o $\gamma$ inadecuado → modelo demasiado simple
- **Muy pocos vectores de soporte**: posible *overfitting* con $C$ muy grande

---

## 14. Ventajas, Limitaciones y Comparativa

### 14.1 Ventajas teóricas y prácticas

- **Garantías teóricas:** el margen máximo minimiza la cota superior del error de generalización derivada de la teoría VC
- **Convexidad:** solución única y global, sin problemas de mínimos locales
- **Eficiencia en alta dimensión:** funciona bien cuando $d \gg N$ (texto, bioinformática)
- **Kernel trick:** permite fronteras no lineales en el espacio original sin costo adicional en alta dimensión
- **Esparsidad:** la solución depende solo de un subconjunto de los datos (vectores de soporte)
- **Robustez:** el margen máximo tiene efecto regularizador intrínseco

### 14.2 Limitaciones

- **Escala:** el entrenamiento es $\mathcal{O}(N^2)$ a $\mathcal{O}(N^3)$; no escala a grandes datasets
- **Selección de kernel:** la elección del kernel requiere conocimiento del dominio o búsqueda exhaustiva
- **Sin probabilidades nativas:** SVM produce una clase binaria, no una probabilidad. Se pueden obtener probabilidades mediante calibración de Platt, pero esto agrega complejidad
- **Sensibilidad a la escala de características:** siempre se debe normalizar
- **Interpretabilidad:** con kernels no lineales, el modelo es una caja negra

### 14.3 SVM en el panorama actual del Machine Learning

Con el auge del *Deep Learning* desde 2012, SVM ha cedido terreno en áreas como visión por computadora y procesamiento de lenguaje natural, donde las redes neuronales profundas dominan. Sin embargo, SVM sigue siendo la opción preferida en:

- Datasets pequeños ($N < 10^4$) con alta dimensionalidad
- Aplicaciones donde las garantías teóricas son relevantes (seguridad, medicina)
- Problemas donde la interpretabilidad de los vectores de soporte es valiosa
- Bioinformática, clasificación de texto, series de tiempo cortas

---

## 15. Referencias

- **Cortes, C. & Vapnik, V. (1995).** Support-Vector Networks. *Machine Learning*, 20(3), 273–297.
- **Vapnik, V. (1998).** *Statistical Learning Theory*. Wiley-Interscience.
- **Schölkopf, B. & Smola, A.J. (2002).** *Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond*. MIT Press.
- **Bishop, C.M. (2006).** *Pattern Recognition and Machine Learning*. Springer. Capítulo 7.
- **Platt, J. (1998).** Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines. *Technical Report MSR-TR-98-14*, Microsoft Research.
- **Burges, C.J.C. (1998).** A Tutorial on Support Vector Machines for Pattern Recognition. *Data Mining and Knowledge Discovery*, 2(2), 121–167.
- **Chang, C.C. & Lin, C.J. (2011).** LIBSVM: A library for support vector machines. *ACM Transactions on Intelligent Systems and Technology*, 2(3), 1–27.
- **MathWorks (2024).** Statistics and Machine Learning Toolbox — `fitcsvm`. Disponible en: [https://www.mathworks.com/help/stats/fitcsvm.html](https://www.mathworks.com/help/stats/fitcsvm.html)

---

*Documento preparado para uso académico en el curso de Redes Neuronales y SVM, Facultad de Ingeniería, Universidad Anáhuac México.*