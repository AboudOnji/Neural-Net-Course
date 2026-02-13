%% Script para generar datos sintéticos de series de tiempo
% para el ejercicio de Redes LSTM (NN_ex2.m)
% Autor: Dr. Aboud BarsekhOnji
% Descripción: Genera dos series temporales correlacionadas:
% 1. Precio del Petróleo (OilPrice)
% 2. Índice Económico (EconomicIndex)

clear; clc; close all;

%% 1. Generación de Datos
rng(42); 

numTimeSteps = 2000;
t = (1:numTimeSteps)';

% Variable 1: Precio del Petróleo (Tendencia + Ciclos + Ruido)

noise_oil = 1.5 * randn(numTimeSteps, 1);
trend_oil = 0.02 * t + cumsum(0.1 * randn(numTimeSteps, 1)); 
seasonality_oil = 5 * sin(2*pi*t/365) + 2 * cos(2*pi*t/90);
OilPrice = 40 + trend_oil + seasonality_oil + noise_oil;

% Variable 2: Índice Económico (Correlacionado con OilPrice + su propia dinámica)
EconomicIndexBase = 100 + 0.8 * OilPrice; 
noise_eco = 1.0 * randn(numTimeSteps, 1);
cycle_eco = 3 * sin(2*pi*t/180);
EconomicIndex = EconomicIndexBase + cycle_eco + noise_eco;

% Normalización (Ojo: se puede avitar aunque es recomendable meter los datos normalizados para la LSTM)
OilPriceRaw = OilPrice;
EconomicIndexRaw = EconomicIndex;

mu = mean([OilPrice, EconomicIndex]);
sig = std([OilPrice, EconomicIndex]);

OilPriceStd = (OilPrice - mu(1)) / sig(1);
EconomicIndexStd = (EconomicIndex - mu(2)) / sig(2);

dataMatrix = [OilPriceStd, EconomicIndexStd]; % Esta es la matriz que contiene todos los datos

figure;
subplot(2,1,1);
plot(t, OilPriceRaw);
title('Precio del Petróleo (Simulado)');
ylabel('USD/Barril');
grid on;

subplot(2,1,2);
plot(t, EconomicIndexRaw);
title('Índice Económico (Simulado)');
ylabel('Puntos');
xlabel('Tiempo (Días)');
grid on;

%% 2. Preparación de Datos para LSTM (Formato WaveformData)
% El formato será compatible con el usado en NN_ex2.m cargando WaveformData.
% data: un cell array {1x1} que contiene la matriz [TimeSteps x Channels]

data = {dataMatrix}; 

% División Entrenamiento / Prueba (90% / 10%)
splitPoint = floor(0.9 * numTimeSteps);

dataTrain = dataMatrix(1:splitPoint, :);
dataTest = dataMatrix(splitPoint+1:end, :);

% Para Forecasting (Next Step Prediction):
% XTrain (Input): pasos t
% TTrain (Target): pasos t+1
% Nota: Para celdas de secuencias, la caja de herramientas de Deep Learning
% a veces prefiere vectores [Channels x TimeSteps].
% Sin embargo, en el ejemplo NN_ex2.m línea 46: predict(net,X(1:offset,:))
% sugiere que la red espera [TimeSteps x Channels] si la InputLayer no dice lo contrario.
% Pero la SequenceInputLayer por defecto espera [Channels x TimeSteps] si NO se especifica DataFormat.
% Revisaremos esto en las instrucciones. Asumiremos el formato standard de WaveformData.
% Si WaveformData es Time x Channels, los datos deben ser así.

% Vamos a transponer para asegurar compatibilidad con secuencia standar si es necesario
% Pero NN_ex2.m usa stackedplot(data{idx}) lo cual grafica columnas como canales.
% Entonces data{idx} es Time x Channels.

% Crear XTrain y TTrain como celdas que contienen la secuencia
XTrain = {dataTrain(1:end-1, :)}; % Input: t
TTrain = {dataTrain(2:end, :)};   % Output: t+1

% Crear XTest y TTest
XTest = {dataTest(1:end-1, :)};
TTest = {dataTest(2:end, :)};

%% 3. Guardar Datos
save('synthetic_data.mat', 'data', 'XTrain', 'TTrain', 'XTest', 'TTest', 'mu', 'sig');

fprintf('Datos generados exitosamente.\n');
fprintf('Dimensiones XTrain{1}: %d pasos x %d canales\n', size(XTrain{1},1), size(XTrain{1},2));
fprintf('Dimensiones XTest{1}:  %d pasos x %d canales\n', size(XTest{1},1), size(XTest{1},2));
fprintf('Archivo guardado: synthetic_data.mat\n');
