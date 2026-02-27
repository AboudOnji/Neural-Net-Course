%% EJEMPLO 3: Clasificación Simple de Imágenes usando CNN
% Comentarios: Dr. Aboud BARSEKH-ONJI
% IPN - Universidad Anáhuac México
% Email: aboud.barsekh@anahuac.mx
% ORCID: 0009-0004-5440-8092
% ==================================================================
%% 1. Carga de Datos de Imagen
% Cargue los datos de dígitos de muestra como un almacén de datos de imágenes.
% La función imageDatastore etiqueta automáticamente las imágenes en función de los nombres de carpeta.
% El conjunto de datos tiene 10 clases y cada imagen del conjunto de datos tiene un tamaño de 28 por 28 por 1 píxeles.

unzip("DigitsData.zip");
imds = imageDatastore("DigitsData", ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

% Obtener las clases únicas
classNames = categories(imds.Labels);

%% 2. Preprocesamiento: Particionamiento de Datos
% Divida los datos en conjuntos de datos de entrenamiento, validación y prueba.
% Utilice el 70% de las imágenes para el entrenamiento, el 15% para la validación y el 15% para la prueba.
% Especifique "randomized" para asignar la proporción especificada de archivos de cada clase a los nuevos conjuntos de datos.

[imdsTrain,imdsValidation,imdsTest] = splitEachLabel(imds,0.7,0.15,0.15,"randomized");

%% 3. Definición de la Arquitectura de Red
% Para crear la red, use la app Deep Network Designer:
% >> deepNetworkDesigner
%
% Arquitectura recomendada para este ejemplo:
% 1. imageInputLayer([28 28 1])
% 2. convolution2dLayer(3, 32, Padding="same") -> Extrae características (32 filtros de 3x3)
% 3. batchNormalizationLayer -> Normaliza activaciones
% 4. reluLayer -> Introduce no linealidad
% 5. fullyConnectedLayer(10) -> Clasifica en 10 clases
% 6. softmaxLayer -> Convierte a probabilidades
% 7. classificationLayer -> Calcula la pérdida (Cross-Entropy)
%
% Una vez diseñada y exportada la red desde la app, se guarda en la variable 'net_1'.
% Nota: Asegúrese de tener 'net_1' en su workspace antes de continuar.

%% 4. Especificar Opciones de Entrenamiento
% Especifique las opciones de entrenamiento.
% - Solver: "sgdm" (Stochastic Gradient Descent with Momentum)
% - MaxEpochs: 4 (Iteraciones completas sobre el set de datos)
% - ValidationFrequency: 30 (Evaluar validación cada 30 iteraciones)
% - Plots: "training-progress" (Ver gráfica en vivo)
% - Metrics: "accuracy" (Monitorear exactitud)

options = trainingOptions("sgdm", ...
    MaxEpochs=4, ...
    InitialLearnRate=0.01,...
    ValidationData=imdsValidation, ...
    ValidationFrequency=30, ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    Verbose=false);

%% 5. Entrenar la Red Neuronal
% Entrene la red neuronal con la función trainnet.
% Como el objetivo es la clasificación, use la pérdida de entropía cruzada ("crossentropy").

net = trainnet(imdsTrain,net_1,"crossentropy",options);

%% 6. Evaluar y Probar la Red Neuronal
% Pruebe la red neuronal con la función testnet.
% Evalúe la precisión de la clasificación en el conjunto de validación.

accuracy = testnet(net,imdsValidation,"accuracy");
disp("Validation Accuracy: " + accuracy + "%");

%% 7. Hacer Predicciones
% Realice predicciones con la función minibatchpredict y convierta las puntuaciones en etiquetas.

scores = minibatchpredict(net,imdsValidation);
YValidation = scores2label(scores,classNames);

% Visualice algunas de las predicciones en una cuadrícula
numValidationObservations = numel(imdsValidation.Files);
idx = randi(numValidationObservations,9,1);

figure
tiledlayout("flow")
for i = 1:9
    nexttile
    img = readimage(imdsValidation,idx(i));
    imshow(img)
    title("Predicted: " + string(YValidation(idx(i))))
end
