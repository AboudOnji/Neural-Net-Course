%% EJEMPLO: Transferencia del Aprendizaje con Deep Network Designer
% Comentarios didácticos: Dr. Aboud BARSEKH-ONJI
% Universidad Anáhuac México
% Email: aboud.barsekh@anahuac.mx
% ORCID: 0009-0004-5440-8092
% ------------------------------------------------------------------
% Fuente original:
% MathWorks. "Transfer Learning with Deep Network Designer."
% https://la.mathworks.com/help/deeplearning/ug/
%           transfer-learning-with-deep-network-designer.html
% ------------------------------------------------------------------
%
% DESCRIPCIÓN GENERAL:
%   Este script demuestra cómo aplicar transferencia del aprendizaje
%   (transfer learning) usando la red preentrenada SqueezeNet para
%   clasificar 5 categorías de artículos promocionales de MathWorks.
%
%   PIPELINE:
%   1. Extraer y organizar el dataset
%   2. Preparar la red en Deep Network Designer (paso manual)
%   3. Aumentar datos para evitar sobreajuste
%   4. Configurar opciones de entrenamiento
%   5. Entrenar la red con trainnet
%   6. Evaluar con matriz de confusión
%   7. Predecir sobre una imagen nueva
%
%   REQUISITOS:
%   - Deep Learning Toolbox
%   - (Opcional) Parallel Computing Toolbox para GPU
%
% ==================================================================

%% 1. EXTRACCIÓN Y CARGA DEL DATASET
% ------------------------------------------------------------------
% El dataset "MerchData" contiene 75 imágenes de 5 clases:
%   gorra, cubo, naipes, destornillador, linterna
%
% Las imágenes están organizadas en subcarpetas por clase:
%   MerchData/cap/, MerchData/cube/, MerchData/playing cards/, etc.
%
% imageDatastore infiere automáticamente las etiquetas de clase
% a partir de los nombres de las subcarpetas (LabelSource="foldernames").
% ------------------------------------------------------------------

folderName = "MerchData";
unzip("MerchData.zip", folderName);   % Descomprimir el archivo ZIP

% Crear el imageDatastore: permite leer imágenes en lotes durante
% el entrenamiento sin cargar todo el dataset en memoria RAM
imds = imageDatastore(folderName, ...
    IncludeSubfolders=true, ...      % Incluir subcarpetas (una por clase)
    LabelSource="foldernames");      % Etiqueta = nombre de la subcarpeta

% Visualizar 16 imágenes de muestra para verificar el dataset
numImages = numel(imds.Labels);
idx = randperm(numImages, 16);       % 16 índices aleatorios
I = imtile(imds, Frames=idx);        % Crear mosaico de imágenes
figure
imshow(I)
title("Muestra aleatoria del dataset MerchData")

% Extraer nombres y número de clases
classNames = categories(imds.Labels);
numClasses = numel(classNames)
% Resultado esperado: numClasses = 5


%% 2. DIVISIÓN DEL DATASET EN ENTRENAMIENTO, VALIDACIÓN Y PRUEBA
% ------------------------------------------------------------------
% División estándar para deep learning:
%   - Entrenamiento (70%): ~52 imágenes — para ajustar los pesos
%   - Validación   (15%): ~11 imágenes — para monitorizar sobreajuste
%   - Prueba       (15%): ~11 imágenes — para evaluación final
%
% "randomized": mezcla aleatoriamente antes de dividir para evitar
% sesgo por orden de lectura de carpetas.
%
% IMPORTANTE: El set de prueba NUNCA debe verse durante el entrenamiento.
% ------------------------------------------------------------------

[imdsTrain, imdsValidation, imdsTest] = ...
    splitEachLabel(imds, 0.7, 0.15, 0.15, "randomized");

fprintf("Imágenes de entrenamiento: %d\n", numel(imdsTrain.Labels));
fprintf("Imágenes de validación:    %d\n", numel(imdsValidation.Labels));
fprintf("Imágenes de prueba:        %d\n", numel(imdsTest.Labels));


%% 3. TAMAÑO DE ENTRADA DE LA RED
% ------------------------------------------------------------------
% SqueezeNet espera imágenes de 227x227x3 píxeles (RGB).
% Guardamos este valor para usarlo en el aumento de datos.
% ------------------------------------------------------------------

inputSize = [227 227 3];


%% 4. PREPARAR LA RED EN DEEP NETWORK DESIGNER (PASO MANUAL)
% ------------------------------------------------------------------
% Abrir la aplicación Deep Network Designer:
%
%   deepNetworkDesigner
%
% Pasos a seguir en la app:
%
%   a) Seleccionar "SqueezeNet" de la galería de redes preentrenadas
%      y hacer clic en "Open".
%
%   b) Explorar la arquitectura: la red tiene ~26 capas y aprox.
%      1.24 millones de parámetros.
%
%   c) Localizar la capa 'conv10' (última capa convolucional).
%
%   d) Seleccionar 'conv10' → clic en "Unlock Layer" → "Unlock Anyway"
%      Esto desbloquea las propiedades para poder modificarlas.
%
%   e) Cambiar:
%      NumFilters        : 1000 → 5   (5 clases nuevas)
%      WeightLearnRateFactor : 1  → 10  (aprende 10x más rápido)
%      BiasLearnRateFactor   : 1  → 10
%
%      Razón: NumFilters define el número de salidas de la capa
%      (= número de clases). Los factores de tasa de aprendizaje
%      altos hacen que esta capa se adapte rápido a las clases nuevas
%      sin afectar los pesos ya aprendidos en el resto de la red.
%
%   f) Hacer clic en "Analyze" para verificar que no hay errores.
%
%   g) Hacer clic en "Export" → la red se guarda como net_1.
%
% NOTA (versiones anteriores a R2023b):
%   No se puede desbloquear capas directamente. En ese caso se debe
%   reemplazar conv10 con una nueva capa:
%     newConv = convolution2dLayer([1 1], numClasses, ...
%         WeightLearnRateFactor=10, BiasLearnRateFactor=10);
% ------------------------------------------------------------------

deepNetworkDesigner   % Abrir la app (ejecutar esta línea para empezar)

% Después de exportar desde la app, la red estará en net_1.
% El siguiente paso asume que ya ejecutaste los pasos anteriores.


%% 5. AUMENTO DE DATOS PARA EL SET DE ENTRENAMIENTO
% ------------------------------------------------------------------
% Con solo ~52 imágenes de entrenamiento, el sobreajuste es probable.
% El aumento de datos genera variantes artificiales en cada época:
%
%   - RandXReflection=true : volteo horizontal aleatorio (espejo)
%   - RandXTranslation     : desplazamiento horizontal ±30 px
%   - RandYTranslation     : desplazamiento vertical   ±30 px
%
% Además, augmentedImageDatastore redimensiona automáticamente
% todas las imágenes a inputSize (227x227), independientemente
% de su tamaño original.
% ------------------------------------------------------------------

pixelRange = [-30 30];

imageAugmenter = imageDataAugmenter( ...
    RandXReflection=true, ...         % Espejo horizontal aleatorio
    RandXTranslation=pixelRange, ...  % Traslación horizontal ±30 px
    RandYTranslation=pixelRange);     % Traslación vertical   ±30 px

% Almacén aumentado para entrenamiento (con aumento + redimensionado)
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    DataAugmentation=imageAugmenter);

% Para validación y prueba: SOLO redimensionado, sin aumento aleatorio.
% Queremos medir el rendimiento real sobre imágenes sin transformar.
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);
augimdsTest       = augmentedImageDatastore(inputSize(1:2), imdsTest);


%% 6. OPCIONES DE ENTRENAMIENTO
% ------------------------------------------------------------------
% Hiperparámetros clave:
%
%   Optimizador Adam:
%     - Adaptativo: ajusta la tasa de aprendizaje por parámetro
%     - Robusto para datasets pequeños y heterogéneos
%
%   InitialLearnRate = 0.0001:
%     - Tasa baja para proteger los pesos preentrenados
%     - Las capas transferidas se modifican lentamente
%     - La capa conv10 aprende 10x más rápido (WeightLearnRateFactor=10)
%
%   MaxEpochs = 8:
%     - En transfer learning la convergencia es rápida
%     - Más épocas → mayor riesgo de sobreajuste
%
%   MiniBatchSize = 11:
%     - ~52 imágenes / 11 ≈ 4-5 iteraciones por época
%     - Divide uniformemente el dataset de entrenamiento
%
%   ValidationFrequency = 5:
%     - Calcula precisión de validación cada 5 iteraciones
% ------------------------------------------------------------------

options = trainingOptions("adam", ...
    InitialLearnRate=0.0001, ...         % Tasa de aprendizaje inicial baja
    MaxEpochs=8, ...                     % Pocas épocas (transfer learning)
    ValidationData=imdsValidation, ...   % Datos de validación
    ValidationFrequency=5, ...           % Frecuencia de validación
    MiniBatchSize=11, ...                % Tamaño de mini-lote
    Plots="training-progress", ...       % Grafica pérdida y precisión
    Metrics="accuracy", ...              % Métrica adicional a monitorizar
    Verbose=false);                      % Sin salida detallada en consola


%% 7. ENTRENAMIENTO DE LA RED
% ------------------------------------------------------------------
% trainnet ajusta los pesos de la red usando los datos de entrenamiento.
%
% Función de pérdida: "crossentropy" (entropía cruzada)
%   L = -sum( y_i * log(y_hat_i) )
%   Adecuada para clasificación multiclase con softmax.
%
% Ejecución en GPU: automática si hay GPU compatible disponible
%   (requiere Parallel Computing Toolbox).
%   Para forzar CPU: añadir ExecutionEnvironment="cpu" en options.
% ------------------------------------------------------------------

net = trainnet(imdsTrain, net_1, "crossentropy", options);

% Durante el entrenamiento se muestra la gráfica de progreso con:
%   - Training Loss      (línea azul)
%   - Validation Loss    (línea naranja)
%   - Training Accuracy  (línea azul punteada)
%   - Validation Accuracy (línea naranja punteada)
%
% Una red bien entrenada mostrará pérdida decreciente y precisión
% creciente en ambas curvas, sin gran brecha entre entrenamiento y validación.


%% 8. EVALUACIÓN DEL MODELO EN EL SET DE PRUEBA
% ------------------------------------------------------------------
% minibatchpredict: clasifica múltiples imágenes en lotes (eficiente)
%   - Retorna una matriz de puntuaciones (probabilidades por clase)
%
% scores2label: convierte las puntuaciones en etiquetas categóricas
%   - Selecciona la clase con mayor probabilidad (argmax)
% ------------------------------------------------------------------

YTest = minibatchpredict(net, augimdsTest);       % Scores de predicción
YTest = scores2label(YTest, classNames);          % Etiquetas predichas

% Etiquetas verdaderas del set de prueba
TTest = imdsTest.Labels;

% Calcular y mostrar la precisión global
accuracy = mean(YTest == TTest);
fprintf("Precisión en el set de prueba: %.2f%%\n", accuracy * 100);

% Visualizar la matriz de confusión
% - Diagonal principal: predicciones correctas por clase
% - Fuera de la diagonal: errores (clase real vs. clase predicha)
figure
confusionchart(TTest, YTest);
title("Matriz de Confusión — Set de Prueba")


%% 9. PREDICCIÓN SOBRE UNA IMAGEN NUEVA
% ------------------------------------------------------------------
% Para clasificar una imagen individual:
%   1. Leer la imagen
%   2. Redimensionar al tamaño de entrada (227x227)
%   3. Convertir a single (float32) — tipo requerido por la red
%   4. (Opcional) Mover a GPU con gpuArray
%   5. Llamar a predict (más eficiente que minibatchpredict para 1 imagen)
%   6. Convertir scores a etiqueta con scores2label
% ------------------------------------------------------------------

im = imread("MerchDataTest.jpg");         % Leer imagen nueva
im = imresize(im, inputSize(1:2));        % Redimensionar a 227x227
X  = single(im);                          % Convertir a float32

% Mover a GPU si está disponible
if canUseGPU
    X = gpuArray(X);
end

scores = predict(net, X);                 % Vector de probabilidades (1x5)
[label, score] = scores2label(scores, classNames);  % Clase más probable

% Visualizar resultado
figure
imshow(im)
title(string(label) + "  (Puntuación: " + gather(score)*100 + "%)")

% gather() recupera el valor de GPU a CPU para mostrarlo en title()


%% 10. RESUMEN DEL PIPELINE COMPLETO
% ------------------------------------------------------------------
%
%  Dataset (75 imágenes, 5 clases)
%    └─> imageDatastore + splitEachLabel (70/15/15)
%         └─> Deep Network Designer
%              └─> SqueezeNet + modificar conv10 (NumFilters=5, LRF=10)
%                   └─> augmentedImageDatastore (resize + augmentation)
%                        └─> trainingOptions (Adam, lr=0.0001, 8 épocas)
%                             └─> trainnet (crossentropy)
%                                  └─> minibatchpredict + confusionchart
%                                       └─> predict (imagen nueva)
%
% PREGUNTAS PARA REFLEXIONAR:
%   1. ¿Qué ocurre si aumentas MaxEpochs a 20?
%   2. ¿Qué ocurre si cambias InitialLearnRate a 0.01?
%   3. ¿Qué ocurre si reduces WeightLearnRateFactor en la Conv1 última a 1?
%   4. ¿Qué red preentrenada daría mejores resultados aquí? ¿Por qué?
%   5. ¿Cómo cambia la matriz de confusión con y sin data augmentation?
% ------------------------------------------------------------------