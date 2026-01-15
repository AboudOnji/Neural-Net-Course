%% EJEMPLO 1:
% Author: Dr. Aboud BARSEKH-ONJI
% IPN - Universidad Anáhuac México
% Email: aboud.barsekh@anahuac.mx
% ORCID: 0009-0004-5440-8092
% ==================================================================
%% 1. Carga de Datos y Exploración
% 'WaveformData' es un conjunto de datos de ejemplo integrado en el Toolbox
% Contiene secuencias temporales (data) y sus categorías (labels).
load WaveformData

% Determinar metadatos básicos
numChannels = size(data{1},2);   % Número de características o sensores por paso de tiempo
classNames = categories(labels); % Obtener las clases únicas (salida esperada)

% Visualización de una muestra de los datos
figure
tiledlayout(2,2) % Crea una rejilla de 2x2 para gráficos
for i = 1:4
    nexttile
    % stackedplot es ideal para series temporales con múltiples canales
    % Muestra cada canal apilado verticalmente compartiendo el eje X
    stackedplot(data{i}, DisplayLabels="Channel "+string(1:numChannels))

    xlabel("Time Step") % Eje de tiempo
    title("Class: " + string(labels(i))) % Título con la clase real
end

%% 2. Preprocesamiento: Particionamiento de Datos
% Divide el total de observaciones en tres conjuntos:
% 80% Entrenamiento, 10% Validación, 10% Prueba.
numObservations = numel(data);
[idxTrain, idxValidation, idxTest] = trainingPartitions(numObservations, [0.8 0.1 0.1]);

% Asignación de datos basada en los índices generados
XTrain = data(idxTrain);       % Datos de entrada (Entrenamiento)
TTrain = labels(idxTrain);     % Etiquetas objetivo (Entrenamiento)

XValidation = data(idxValidation); % Datos para ajustar hiperparámetros y ver overfitting
TValidation = labels(idxValidation);

XTest = data(idxTest);         % Datos no vistos para evaluación final
TTest = labels(idxTest);

%% 3. Configuración del Entrenamiento (Hiperparámetros)
% Usa la app DeepNetworkDesigner para elegir y modificar la red nueronal
% Recomendable: LSTM - Adapta la cantidad de inputs (3) y de target (4)
% IMPORTANTE: exporta la red como 'net_1'

options = trainingOptions("adam", ...        % Optimizador: Adam
    MaxEpochs=500, ...                       % Número máximo de pasadas por todos los datos
    InitialLearnRate=0.0005, ...             % Tasa de aprendizaje inicial (paso del gradiente)
    GradientThreshold=1, ...                 % Recorte de gradiente para estabilidad (evita explosión)
    ValidationData={XValidation,TValidation},... % Datos para validar durante entrenamiento
    Shuffle="every-epoch", ...               % Mezclar datos en cada época para evitar ciclos
    Plots="training-progress", ...           % Abrir gráfica en vivo del entrenamiento
    Metrics="accuracy", ...                  % Métrica a monitorear además de la pérdida (loss)
    Verbose=false);                          % No imprimir log detallado en Command Window

%% 4. Entrenamiento de la Red
% trainnet es la función moderna (R2023a+).
% Entrena 'net_1' usando la función de pérdida de entropía cruzada.
net = trainnet(XTrain,TTrain,net_1,"crossentropy",options);

%% 5. Evaluación del Modelo
% minibatchpredict: Realiza predicciones por lotes (eficiente en memoria)
% Devuelve scores (probabilidades crudas tras Softmax)
scores = minibatchpredict(net, XTest);

% scores2label: Convierte las probabilidades en la etiqueta de clase más probable
YTest = scores2label(scores, classNames);

% Cálculo de la exactitud (Accuracy) global
acc = mean(YTest == TTest)

% Matriz de confusión para ver falsos positivos/negativos
figure
confusionchart(TTest, YTest)