# Solución Paso a Paso: Problema de Redes Neuronales con Adam

Este documento presenta la solución para el mismo problema de entrenamiento aplicando el algoritmo de optimización **Adam (Adaptive Moment Estimation)**.

### Diagrama de la Red

![Diagrama de la Red Neuronal](network_diagram.png)

**Autor:** Dr. Aboud Barsekh Onji
**Institución:** IPN - Universidad Anáhuac México

---

## 1. Planteamiento del Problema

Se tiene una red neuronal multicapa con la siguiente configuración:

*   **Arquitectura:** 2 Entradas - 2 Ocultas - 2 Salidas.
*   **Entradas:** $x_1 = 0.35$, $x_2 = 0.8$.
*   **Salida Deseada:** $t = 0.5$ ($t_1=0.5, t_2=0.5$).
*   **Hiperparámetros de Adam:**
    *   Tasa de Aprendizaje ($\eta$): $0.2$. (*Nota: En aplicaciones reales de Adam, se suelen usar tasas menores como 0.001, pero mantenemos 0.2 para consistencia con el ejercicio original*).
    *   $\beta_1$: $0.9$ (Decaimiento del primer momento).
    *   $\beta_2$: $0.999$ (Decaimiento del segundo momento).
    *   $\epsilon$: $10^{-8}$ (Para evitar división por cero).
*   **Función de Activación:** Log-Sigmoid.

---

## 2. Fase 1: Propagación hacia Adelante

*Idéntico al original.*

**Capa Oculta:**
*   $out_{h1} \approx 0.5486$
*   $out_{h2} \approx 0.6502$

**Capa de Salida:**
*   $out_{o1} \approx 0.5731$
*   $out_{o2} \approx 0.6548$

---

## 3. Fase 2: Cálculo del Error y Gradientes

*Calculamos los gradientes locales $\delta$.*

**Capa de Salida:**
*   $\delta_{o1} \approx 0.0179$
*   $\delta_{o2} \approx 0.0350$

**Capa Oculta:**
*   $\delta_{h1} \approx 0.0022$
*   $\delta_{h2} \approx 0.0080$

---

## 4. Fase 3: Actualización de Pesos con Adam

El algoritmo Adam mantiene estimaciones del promedio ($m$) y la varianza no centrada ($v$) de los gradientes.

### Fórmulas Adam (para $t=1$)

Para cada peso, definimos el gradiente $g = \nabla E = \delta \cdot input$.
Asumimos $m_0 = 0$ y $v_0 = 0$.

1.  **Primer Momento (Media):** $m_1 = \beta_1 m_0 + (1 - \beta_1) g = (1 - 0.9) g = \mathbf{0.1 g}$
2.  **Segundo Momento (Varianza):** $v_1 = \beta_2 v_0 + (1 - \beta_2) g^2 = (1 - 0.999) g^2 = \mathbf{0.001 g^2}$
3.  **Corrección de Sesgo (Bias Correction):**
    *   $\hat{m}_1 = \frac{m_1}{1 - \beta_1^1} = \frac{0.1 g}{0.1} = \mathbf{g}$
    *   $\hat{v}_1 = \frac{v_1}{1 - \beta_2^1} = \frac{0.001 g^2}{0.001} = \mathbf{g^2}$
4.  **Actualización:**
    $$ w_{new} = w_{old} - \eta \frac{\hat{m}_1}{\sqrt{\hat{v}_1} + \epsilon} $$

**Análisis Crítico del Paso 1:**
Sustituyendo $\hat{m}_1 = g$ y $\hat{v}_1 = g^2$:
$$ \Delta w = - \eta \frac{g}{\sqrt{g^2} + \epsilon} \approx - \eta \frac{g}{|g|} = - \eta \cdot \text{signo}(g) $$
*Esto significa que en la primera iteración, Adam realiza un paso de magnitud casi fija $\eta$ (0.2) en la dirección opuesta al gradiente. Esto resultará en cambios de peso drásticos.*

---

### Actualización Capa Oculta $\to$ Salida

**Peso $w_{11}^{(2)}$:**
*   Gradiente $g = \delta_{o1} \cdot out_{h1} = 0.0179 \cdot 0.5486 = 0.00982$
*   Como $g > 0$, el paso será aprox $-0.2$.
*   Cálculo exacto:
    *   $\hat{m} = 0.00982$
    *   $\hat{v} = (0.00982)^2 = 0.0000964$
    *   $\Delta w = - 0.2 \frac{0.00982}{\sqrt{0.0000964}} = -0.2 (1) = -0.2$
*   $w_{11}^{(2)+} = 0.3 - 0.2 = \mathbf{0.1000}$

**Peso $w_{21}^{(2)}$:**
*   Gradiente $g = 0.01164$ ($>0$)
*   $w_{21}^{(2)+} = 0.2 - 0.2 = \mathbf{0.0000}$

**Peso $w_{12}^{(2)}$:**
*   Gradiente $g = 0.01920$ ($>0$)
*   $w_{12}^{(2)+} = 0.1 - 0.2 = \mathbf{-0.1000}$

**Peso $w_{22}^{(2)}$:**
*   Gradiente $g = 0.02276$ ($>0$)
*   $w_{22}^{(2)+} = 0.9 - 0.2 = \mathbf{0.7000}$

---

### Actualización Capa Entrada $\to$ Oculta

**Peso $w_{11}^{(1)}$:**
*   Gradiente $g = \delta_{h1} \cdot x_1 = 0.0022 \cdot 0.35 = 0.00077$ ($>0$)
*   $w_{11}^{(1)+} = 0.1 - 0.2 = \mathbf{-0.1000}$

**Peso $w_{21}^{(1)}$:**
*   Gradiente $g = 0.00176$ ($>0$)
*   $w_{21}^{(1)+} = 0.2 - 0.2 = \mathbf{0.0000}$

**Peso $w_{12}^{(1)}$:**
*   Gradiente $g = 0.0028$ ($>0$)
*   $w_{12}^{(1)+} = 0.4 - 0.2 = \mathbf{0.2000}$

**Peso $w_{22}^{(1)}$:**
*   Gradiente $g = 0.0064$ ($>0$)
*   $w_{22}^{(1)+} = 0.6 - 0.2 = \mathbf{0.4000}$

---

## Resumen y Comparación

Debido a que $\eta=0.2$ es una tasa de aprendizaje extremadamente alta para Adam, y a que en la primera iteración Adam normaliza el gradiente por su propia magnitud, los pesos han sufrido cambios masivos de $\pm 0.2$.

| Peso | Original | SGD ($\eta=0.2$) | Adam ($\eta=0.2$) |
| :--- | :--- | :--- | :--- |
| **Capa 2** | | | |
| $w_{11}^{(2)}$ | 0.3 | 0.2980 | **0.1000** |
| $w_{21}^{(2)}$ | 0.2 | 0.1977 | **0.0000** |
| $w_{12}^{(2)}$ | 0.1 | 0.0962 | **-0.1000** |
| $w_{22}^{(2)}$ | 0.9 | 0.8954 | **0.7000** |
| **Capa 1** | | | |
| $w_{11}^{(1)}$ | 0.1 | 0.0998 | **-0.1000** |
| $w_{12}^{(1)}$ | 0.4 | 0.3994 | **0.2000** |
| $w_{22}^{(1)}$ | 0.6 | 0.5987 | **0.4000** |

> **Conclusión:** Este ejercicio demuestra por qué Adam requiere tasas de aprendizaje mucho menores (típicamente entre $0.0001$ y $0.001$). Con $\eta=0.2$, Adam se vuelve inestable y da pasos demasiado grandes ("Overshooting").
