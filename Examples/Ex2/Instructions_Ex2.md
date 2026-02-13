# Tarea Práctica: Pronóstico de Series de Tiempo Multivariado con LSTM

Esta práctica consiste en generar datos sintéticos de dos series temporales correlacionadas (Precio de Petróleo e Índice Económico) y utilizar una Red Neuronal Recurrente LSTM para predecir su comportamiento futuro.

## Requisitos Previos
*   MATLAB con Deep Learning Toolbox.
*   Archivos: `NN_ex2.m` y `generate_data_ex2.m` en la carpeta misma carpeta.

## Instrucciones Paso a Paso

### 1. Generación de Datos
Abra y ejecute el script `generate_data_ex2.m`.
Este script realizará lo siguiente:
1.  Generará datos simulados para dos variables:
    *   **Canal 1:** Precio del Petróleo (con tendencia y estacionalidad).
    *   **Canal 2:** Índice Económico (correlacionado con el precio del petróleo).
2.  Visualizará las series temporales generadas.
3.  Guardará los datos procesados en el archivo `synthetic_data.mat`.

### 2. Modificación del Script LSTM (`NN_ex2.m`)
Abra el archivo `NN_ex2.m` y realice las siguientes modificaciones para adaptarlo a los nuevos datos:

#### A. Cargar los Nuevos Datos
Busque la **Línea 8** (aproximadamente) y cambie la carga del archivo de datos:
```matlab
% ANTES:
load WaveformData

% DESPUÉS:
load synthetic_data
```

#### B. Verificar Canales
Ejecute la **Sección Parte I** del script ("Carga de Datos de Secuencia").
Verifique en el gráfico generado que ahora aparecen **2 canales** de datos en lugar de 3.

#### C. Ajustar la Arquitectura de la Red (Deep Network Designer)
La red neuronal debe ajustarse para aceptar 2 entradas y producir 2 salidas.

1.  Ejecute la línea `deepNetworkDesigner` para abrir la aplicación.
2.  Seleccione **Sequence-to-Sequence Classification** (como base) o construya una red nueva.
3.  Reemplace la capa final de clasificación (softmax/classification) por una **Regression Layer**.
4.  **IMPORTANTE:** Modifique las dimensiones de entrada y salida:
    *   Seleccione la capa **Sequence Input Layer**: Cambie `InputSize` a **2**.
    *   Seleccione la capa **Fully Connected Layer** (antes de la salida): Cambie `OutputSize` a **2**.
5.  Haga clic en **Analyze** para verificar que no hay errores.
6.  Haga clic en **Export** para enviar la red al Workspace con el nombre `net_1`.
    *   *Alternativa:* Si prefiere hacerlo por código, puede modificar la variable `layers` directamente si tiene el código de construcción de la red, asegurando `InputSize=2` y `OutputSize=2`.

#### D. Entrenar la Red
Ejecute la **Sección Parte IV** ("Entrenar la Red Neuronal").
Observe la gráfica de progreso del entrenamiento. El RMSE debe disminuir conforme avanzan las épocas.

#### E. Realizar Predicciones
Ejecute la **Sección Parte V** ("Predecir Pasos de Tiempo Futuros").
El código visualizará las predicciones vs. los datos reales para ambos canales.

## Entregable
1.  Gráfico de los datos generados (del script `generate_data_ex2.m`).
2.  Gráfico del proceso de convergencia del entrenamiento.
3.  Gráficos de las predicciones finales del modelo LSTM comparadas con los datos de prueba.
4.  Breve reporte de los cambios realizados y conclusiones sobre la capacidad de la red para aprender la correlación entre las variables.
