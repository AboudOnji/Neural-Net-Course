%% EJEMPLO 2: Predicción de series de tiempo usando LSTM
% Comentarios: Dr. Aboud BARSEKH-ONJI
% IPN - Universidad Anáhuac México
% Email: aboud.barsekh@anahuac.mx
% ORCID: 0009-0004-5440-8092
% ==================================================================
%% Parte I: Carga de Datos de Secuencia
load WaveformData
% Visualizar algunas de las secuencias.
idx = 1;
numChannels = size(data{idx},2);

figure
stackedplot(data{idx},DisplayLabels="Channel " + (1:numChannels))
%% Parte II: Definir la Arquitectura de la Red
% Para construir la red, abra la app Deep Network Designer.
deepNetworkDesigner
% Para crear una red de secuencia, en la sección Sequence-to-Sequence Classification Networks (Untrained), haga clic en LSTM.
% Hacerlo abre una red preconstruida adecuada para problemas de clasificación de secuencias. Puede convertir la red de clasificación en una red de regresión editando las capas finales.
% Primero, elimine la capa softmax.
% Luego, ajuste las propiedades de las capas para que sean adecuadas para el conjunto de datos Waveform. Debido a que el objetivo es predecir puntos de datos futuros en una serie de tiempo, el tamaño de salida debe ser el mismo que el tamaño de entrada. En este ejemplo, los datos de entrada tienen tres canales de entrada, por lo que la salida de la red también debe tener tres canales de salida.
% Seleccione la capa de entrada de secuencia (sequence input layer) y establezca InputSize en 3.
% Seleccione la capa totalmente conectada (fully connected layer) 'fc' y establezca OutputSize en 3.
% Para verificar que la red esté lista para el entrenamiento, haga clic en Analyze. El Deep Learning Network Analyzer reporta cero errores o advertencias, por lo que la red está lista para el entrenamiento. Para exportar la red, haga clic en Export. La app guarda la red en la variable net_1.

%% Parte III: Especificar Opciones de Entrenamiento
% Especifique las opciones de entrenamiento. Elegir entre las opciones requiere análisis empírico. Para explorar diferentes configuraciones de opciones de entrenamiento ejecutando experimentos, puede usar la app Experiment Manager. Debido a que las capas recurrentes procesan datos de secuencia un paso de tiempo a la vez, cualquier relleno en los pasos de tiempo finales puede influir negativamente en la salida de la capa. Rellene o trunque los datos de secuencia a la izquierda estableciendo la opción SequencePaddingDirection en "left".
options = trainingOptions("adam", ...
    MaxEpochs=300, ...
    SequencePaddingDirection="left", ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=false);
%% Parte IV: Entrenar la Red Neuronal
% Entrene la red neuronal usando la función trainnet. Debido a que el objetivo es regresión, use la pérdida de error cuadrático medio (MSE).
net = trainnet(XTrain,TTrain,net_1,"mse",options);

%% Parte V: Predecir Pasos de Tiempo Futuros
% La predicción de ciclo cerrado predice pasos de tiempo subsiguientes en una secuencia usando las predicciones anteriores como entrada.
% Seleccione la primera observación de prueba. Inicialice el estado de la RNN restableciendo el estado usando la función resetState. Luego use la función predict para hacer una predicción inicial Z. Actualice el estado de la RNN usando todos los pasos de tiempo de los datos de entrada.
X = XTest{1};
T = TTest{1};

net = resetState(net);
offset = size(X,1);
[Z,state] = predict(net,X(1:offset,:));
net.State = state;
% Para predecir más predicciones, haga un bucle sobre los pasos de tiempo y haga predicciones usando la función predict y el valor predicho para el paso de tiempo anterior. Después de cada predicción, actualice el estado de la RNN. Prediga los siguientes 200 pasos de tiempo pasando iterativamente el valor predicho anterior a la RNN. Debido a que la RNN no requiere los datos de entrada para hacer más predicciones, puede especificar cualquier número de pasos de tiempo para predecir. El último paso de tiempo de la predicción inicial es el primer paso de tiempo predicho.
numPredictionTimeSteps = 200;
Y = zeros(numPredictionTimeSteps,numChannels);
Y(1,:) = Z(end,:);

for t = 2:numPredictionTimeSteps
    [Y(t,:),state] = predict(net,Y(t-1,:));
    net.State = state;
end

numTimeSteps = offset + numPredictionTimeSteps;
% Compare las predicciones con los valores de entrada.
figure
l = tiledlayout(numChannels,1);
title(l,"Time Series Forecasting")

for i = 1:numChannels
    nexttile
    plot(X(1:offset,i))
    hold on
    plot(offset+1:numTimeSteps,Y(:,i),"--")
    ylabel("Channel " + i)
end

xlabel("Time Step")
legend(["Input" "Forecasted"])
% Este método de predicción se llama predicción de ciclo cerrado. Para más información sobre predicción de series de tiempo y realizar predicción de ciclo abierto, vea Time Series Forecasting Using Deep Learning.
