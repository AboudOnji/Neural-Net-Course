# Solución Paso a Paso: Problema de Examen de Redes Neuronales

Este documento presenta la solución detallada para el problema de entrenamiento (un paso de Backpropagation) de acuerdo con la siguiente red neuronal.

### Diagrama de la Red

![Diagrama de la Red Neuronal](network_diagram.png)

**Autor:** Dr. Aboud Barsekh Onji
**Institución:** IPN - Universidad Anáhuac México

---

## 1. Planteamiento del Problema

Se tiene una red neuronal multicapa con la siguiente configuración:

*   **Arquitectura:** 2 Entradas - 2 Ocultas - 2 Salidas.
*   **Entradas ($X$):** $x_1 = 0.35$, $x_2 = 0.8$.
*   **Salida Deseada ($Target$):** $t = 0.5$ (Asumiremos $t_1=0.5$ y $t_2=0.5$ dado que la red tiene dos salidas).
*   **Tasa de Aprendizaje ($\eta$):** $0.2$.
*   **Función de Activación:** Log-Sigmoid para todas las neuronas:
    $$ \sigma(z) = \frac{1}{1 + e^{-z}} $$
    *Derivada:* $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

### Pesos Iniciales (Asumidos e Interpretados del Diagrama)

**Capa 1 (Entrada $\to$ Oculta):**
*   $w_{11}^{(1)} = 0.1$ (de $x_1$ a $h_1$)
*   $w_{21}^{(1)} = 0.2$ (de $x_2$ a $h_1$) *[Dato inferido]*
*   $w_{12}^{(1)} = 0.4$ (de $x_1$ a $h_2$)
*   $w_{22}^{(1)} = 0.6$ (de $x_2$ a $h_2$)

**Capa 2 (Oculta $\to$ Salida):**
*   $w_{11}^{(2)} = 0.3$ (de $h_1$ a $o_1$)
*   $w_{21}^{(2)} = 0.2$ (de $h_2$ a $o_1$) *[Dato inferido]*
*   $w_{12}^{(2)} = 0.1$ (de $h_1$ a $o_2$) *[Dato inferido]*
*   $w_{22}^{(2)} = 0.9$ (de $h_2$ a $o_2$)

---

## 2. Fase 1: Propagación hacia Adelante (Forward Propagation)

Calculamos la entrada neta ($net$) y la salida ($out$) de cada neurona.

### Capa Oculta ($H$)

**Neurona $h_1$:**
$$ net_{h1} = x_1 w_{11}^{(1)} + x_2 w_{21}^{(1)} = (0.35)(0.1) + (0.8)(0.2) $$
$$ net_{h1} = 0.035 + 0.16 = \mathbf{0.195} $$
$$ out_{h1} = \frac{1}{1 + e^{-0.195}} \approx \mathbf{0.5486} $$

**Neurona $h_2$:**
$$ net_{h2} = x_1 w_{12}^{(1)} + x_2 w_{22}^{(1)} = (0.35)(0.4) + (0.8)(0.6) $$
$$ net_{h2} = 0.14 + 0.48 = \mathbf{0.62} $$
$$ out_{h2} = \frac{1}{1 + e^{-0.62}} \approx \mathbf{0.6502} $$

### Capa de Salida ($O$)

**Neurona $o_1$:**
$$ net_{o1} = out_{h1} w_{11}^{(2)} + out_{h2} w_{21}^{(2)} = (0.5486)(0.3) + (0.6502)(0.2) $$
$$ net_{o1} = 0.16458 + 0.13004 = \mathbf{0.2946} $$
$$ out_{o1} = \frac{1}{1 + e^{-0.2946}} \approx \mathbf{0.5731} $$

**Neurona $o_2$:**
$$ net_{o2} = out_{h1} w_{12}^{(2)} + out_{h2} w_{22}^{(2)} = (0.5486)(0.1) + (0.6502)(0.9) $$
$$ net_{o2} = 0.05486 + 0.58518 = \mathbf{0.6400} $$
$$ out_{o2} = \frac{1}{1 + e^{-0.6400}} \approx \mathbf{0.6548} $$

---

## 3. Fase 2: Cálculo del Error y Retropropagación (Backpropagation)

Calculamos los gradientes ($\delta$) comenzando desde la salida hacia atrás.

### Error en la Salida
Asumiendo la función de error cuadrático medio $E = \frac{1}{2}(t - out)^2$.

**Para $o_1$ (Target $t_1 = 0.5$):**
$$ \delta_{o1} = (out_{o1} - t_1) \cdot out_{o1}(1 - out_{o1}) $$
$$ \delta_{o1} = (0.5731 - 0.5) \cdot 0.5731(1 - 0.5731) $$
$$ \delta_{o1} = (0.0731) \cdot (0.2446) \approx \mathbf{0.0179} $$

**Para $o_2$ (Target $t_2 = 0.5$):**
$$ \delta_{o2} = (out_{o2} - t_2) \cdot out_{o2}(1 - out_{o2}) $$
$$ \delta_{o2} = (0.6548 - 0.5) \cdot 0.6548(1 - 0.6548) $$
$$ \delta_{o2} = (0.1548) \cdot (0.2260) \approx \mathbf{0.0350} $$

### Error en la Capa Oculta
Propagamos el error hacia atrás ponderado por los pesos.

**Para $h_1$:**
$$ \delta_{h1} = (\delta_{o1} w_{11}^{(2)} + \delta_{o2} w_{12}^{(2)}) \cdot out_{h1}(1 - out_{h1}) $$
$$ \delta_{h1} = [(0.0179)(0.3) + (0.0350)(0.1)] \cdot 0.5486(1 - 0.5486) $$
$$ \delta_{h1} = [0.00537 + 0.0035] \cdot 0.2476 $$
$$ \delta_{h1} = 0.00887 \cdot 0.2476 \approx \mathbf{0.0022} $$

**Para $h_2$:**
$$ \delta_{h2} = (\delta_{o1} w_{21}^{(2)} + \delta_{o2} w_{22}^{(2)}) \cdot out_{h2}(1 - out_{h2}) $$
$$ \delta_{h2} = [(0.0179)(0.2) + (0.0350)(0.9)] \cdot 0.6502(1 - 0.6502) $$
$$ \delta_{h2} = [0.00358 + 0.0315] \cdot 0.2274 $$
$$ \delta_{h2} = 0.03508 \cdot 0.2274 \approx \mathbf{0.0080} $$

---

## 4. Fase 3: Actualización de Pesos

Regla de actualización: $w_{new} = w_{old} - \eta \cdot \delta \cdot input$.

### Actualización Capa Oculta $\to$ Salida

$$ w_{11}^{(2)+} = 0.3 - 0.2(0.0179)(0.5486) = 0.3 - 0.00196 = \mathbf{0.2980} $$
$$ w_{21}^{(2)+} = 0.2 - 0.2(0.0179)(0.6502) = 0.2 - 0.00233 = \mathbf{0.1977} $$
$$ w_{12}^{(2)+} = 0.1 - 0.2(0.0350)(0.5486) = 0.1 - 0.00384 = \mathbf{0.0962} $$
$$ w_{22}^{(2)+} = 0.9 - 0.2(0.0350)(0.6502) = 0.9 - 0.00455 = \mathbf{0.8954} $$

### Actualización Capa Entrada $\to$ Oculta

$$ w_{11}^{(1)+} = 0.1 - 0.2(0.0022)(0.35) = 0.1 - 0.00015 = \mathbf{0.0998} $$
$$ w_{21}^{(1)+} = 0.2 - 0.2(0.0022)(0.80) = 0.2 - 0.00035 = \mathbf{0.1996} $$
$$ w_{12}^{(1)+} = 0.4 - 0.2(0.0080)(0.35) = 0.4 - 0.00056 = \mathbf{0.3994} $$
$$ w_{22}^{(1)+} = 0.6 - 0.2(0.0080)(0.80) = 0.6 - 0.00128 = \mathbf{0.5987} $$

---

## Resumen de Resultados

| Peso | Valor Original | Nuevo Valor |
| :--- | :--- | :--- |
| **Capa 2** | | |
| $w_{11}^{(2)}$ | 0.3 | **0.2980** |
| $w_{22}^{(2)}$ | 0.9 | **0.8954** |
| **Capa 1** | | |
| $w_{11}^{(1)}$ | 0.1 | **0.0998** |
| $w_{12}^{(1)}$ | 0.4 | **0.3994** |
| $w_{22}^{(1)}$ | 0.6 | **0.5987** |

*Nota: Los valores han sido redondeados a 4 decimales para claridad.*

## Apéndice: Derivación Matemática de $\delta_{o1}$

Para comprender la transición entre la función de error global y la fórmula específica de actualización para la neurona de salida, aplicamos el cálculo multivariable, específicamente la **Regla de la Cadena**.

### 1. Definiciones de Control
De acuerdo con el planteamiento del problema, utilizamos los siguientes componentes:
* **Función de Error Cuadrático Medio ($E$):** $E = \frac{1}{2}(t - out)^2$.
* **Función de Activación (Log-Sigmoid):** $out = \sigma(net) = \frac{1}{1+e^{-net}}$.
* **Derivada de la Activación:** $\sigma'(net) = out(1 - out)$.

### 2. El Concepto de Gradiente Local ($\delta$)
El valor $\delta_{o1}$ representa la sensibilidad del error total respecto a la entrada neta de la neurona de salida. Matemáticamente:

$$\delta_{o1} = \frac{\partial E}{\partial net_{o1}}$$

### 3. Desglose por Regla de la Cadena
Como el error no depende directamente de la entrada neta ($net$), sino de la salida final ($out$), desglosamos la derivada en dos eslabones:

$$\delta_{o1} = \underbrace{\frac{\partial E}{\partial out_{o1}}}_{\text{Parte A}} \cdot \underbrace{\frac{\partial out_{o1}}{\partial net_{o1}}}_{\text{Parte B}}$$

#### Parte A: Derivada del Error respecto a la Salida
Derivamos la función $E = \frac{1}{2}(t_1 - out_{o1})^2$ con respecto a $out_{o1}$:
1. Bajamos el exponente ($2$), que se cancela con el coeficiente $\frac{1}{2}$.
2. Mantenemos el término interno: $(t_1 - out_{o1})$.
3. Multiplicamos por la derivada interna de $-out_{o1}$, que es $-1$.
4. **Resultado:** $-(t_1 - out_{o1})$, lo cual se reescribe como **$(out_{o1} - t_1)$**.

#### Parte B: Derivada de la Salida respecto a la Entrada Neta
Esta parte corresponde a la derivada de la función de activación logística:
* **Resultado:** **$out_{o1}(1 - out_{o1})$**.

### 4. Ecuación Resultante
Al unir ambos componentes, obtenemos la fórmula final utilizada en el cálculo del examen:

$$\delta_{o1} = (out_{o1} - t_1) \cdot out_{o1}(1 - out_{o1})$$

---
*Nota: Este valor de delta es el que se multiplica posteriormente por la tasa de aprendizaje ($\eta$) y la entrada ($input$) para actualizar los pesos de la capa de salida.*