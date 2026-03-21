%% ========================================================================
% EJEMPLO: Clasificación y Regresión con Support Vector Machines (SVM)
% Autor:       Prof. D.Sc. BARSEKH-ONJI Aboud
% Institución: Facultad de Ingeniería, Universidad Anáhuac México
% Curso:       Redes Neuronales y SVM
% Email:       aboud.barsekh@anahuac.mx
% ORCID:       0009-0004-5440-8092
% -------------------------------------------------------------------------
% Descripción:
%   Este script implementa un flujo completo de SVM para clasificación
%   binaria (Iris versicolor vs. Iris virginica) y regresión (SVR para
%   consumo de combustible). Incluye:
%     - Preparación y normalización de datos
%     - Entrenamiento con kernel RBF (Gaussiano)
%     - Evaluación con matriz de confusión y métricas
%     - Búsqueda de hiperparámetros C y gamma en grilla logarítmica
%     - Visualización de la frontera de decisión y vectores de soporte
%     - Ejemplo de Support Vector Regression (epsilon-SVR)
% =========================================================================

clc; clear; close all;
rng(42);  % Semilla para reproducibilidad de resultados

%% =========================================================================
%  PARTE I: CLASIFICACIÓN SVM BINARIA
%  Dataset: Iris de Fisher — versicolor vs. virginica
%  Características: longitud y ancho del pétalo (columnas 3 y 4)
% =========================================================================

%% 1. CARGA Y SELECCIÓN DE DATOS
% -------------------------------------------------------------------------
% Se carga el dataset 'fisheriris' incluido en MATLAB.
% Contiene 150 muestras de 3 especies con 4 características cada una.
% Usamos solo las últimas 100 muestras (versicolor: 51-100, virginica: 101-150)
% y las columnas 3 y 4 (pétalo) porque son las más discriminativas.

load fisheriris;

X = meas(51:end, 3:4);           % 100 muestras x 2 características (pétalo)
Y = categorical(species(51:end)); % Etiquetas de clase como variable categórica

% Verificar la distribución de clases
disp('=== Distribución de clases ===');
tabulate(Y)

fprintf('Dimensiones del dataset: %d muestras, %d características\n', ...
    size(X,1), size(X,2));

%% 2. PARTICIÓN ENTRENAMIENTO / PRUEBA
% -------------------------------------------------------------------------
% cvpartition con 'HoldOut' realiza una partición aleatoria estratificada:
% preserva la proporción de cada clase en ambos conjuntos.
% 70% entrenamiento, 30% prueba.
%
% IMPORTANTE: la partición estratificada es esencial para datasets
% desbalanceados. Con partición aleatoria simple podría ocurrir que
% una clase tenga muy pocas muestras en entrenamiento.

cv = cvpartition(Y, 'HoldOut', 0.3);

X_train = X(cv.training, :);   % 70 muestras de entrenamiento
Y_train = Y(cv.training);
X_test  = X(cv.test, :);       % 30 muestras de prueba
Y_test  = Y(cv.test);

fprintf('Muestras de entrenamiento: %d\n', sum(cv.training));
fprintf('Muestras de prueba:        %d\n', sum(cv.test));

%% 3. NORMALIZACIÓN Z-SCORE
% -------------------------------------------------------------------------
% La normalización es OBLIGATORIA para SVM con kernel RBF.
% El kernel RBF usa la distancia euclidiana entre puntos:
%   K(xi, xj) = exp(-gamma * ||xi - xj||^2)
% Si una característica tiene rango [0, 1000] y otra [0, 1], la primera
% dominará artificialmente la distancia.
%
% z-score: x_norm = (x - media) / desviacion_estandar
%
% REGLA CRÍTICA: calcular media y std SOLO con datos de entrenamiento.
% Aplicar los mismos parámetros a los datos de prueba.
% Usar datos de prueba en el cálculo es "data leakage" (error metodológico).

mu    = mean(X_train);   % Media de cada característica (1x2)
sigma = std(X_train);    % Desviación estándar (1x2)

X_train_std = (X_train - mu) ./ sigma;   % Normalizar entrenamiento
X_test_std  = (X_test  - mu) ./ sigma;   % Aplicar mismos parámetros a prueba

fprintf('\nEstadísticas de normalización (calculadas sobre entrenamiento):\n');
fprintf('  Media:  [%.4f, %.4f]\n', mu(1), mu(2));
fprintf('  StdDev: [%.4f, %.4f]\n', sigma(1), sigma(2));

%% 4. ENTRENAMIENTO DEL MODELO SVM (configuración base)
% -------------------------------------------------------------------------
% fitcsvm: función de MATLAB para entrenar SVM de clasificación binaria
%
% Parámetros importantes:
%   'KernelFunction', 'rbf'   → Kernel Gaussiano (más usado en práctica)
%   'BoxConstraint', C        → Parámetro C del Soft Margin:
%                                C grande = penaliza errores, margen estrecho
%                                C pequeño = tolera errores, margen ancho
%   'KernelScale', 'auto'     → MATLAB estima sigma automáticamente
%                                usando mediana de distancias entre pares
%                                (sigma define gamma = 1/(2*sigma^2))
%   'Standardize', false      → Ya normalizamos manualmente

fprintf('\n=== Entrenamiento SVM base (C=1, gamma=auto) ===\n');
svmModel = fitcsvm(X_train_std, Y_train, ...
    'KernelFunction', 'rbf',   ...
    'BoxConstraint',   1,      ...
    'KernelScale',    'auto',  ...
    'Standardize',    false);

% Extraer información del modelo entrenado
n_sv = sum(svmModel.IsSupportVector);   % Número de vectores de soporte
fprintf('Vectores de soporte: %d de %d muestras de entrenamiento (%.1f%%)\n', ...
    n_sv, size(X_train_std,1), 100*n_sv/size(X_train_std,1));

%% 5. EVALUACIÓN DEL MODELO BASE
% -------------------------------------------------------------------------
% predict aplica el modelo entrenado a nuevos datos.
% Devuelve las etiquetas predichas y, opcionalmente, los scores continuos
% (valor de la función de decisión antes del sign).

Y_pred = predict(svmModel, X_test_std);

% Matriz de confusión: filas = clase real, columnas = clase predicha
% [VP FN; FP VN] para el caso binario
confMat = confusionmat(Y_test, Y_pred);

fprintf('\n=== Resultados en conjunto de prueba ===\n');
disp('Matriz de confusión:');
disp(array2table(confMat, ...
    'VariableNames', {'Pred_versicolor','Pred_virginica'}, ...
    'RowNames',      {'Real_versicolor','Real_virginica'}));

% Cálculo de métricas a partir de la matriz de confusión
VP = confMat(1,1);  % Verdaderos positivos (versicolor bien clasificado)
FN = confMat(1,2);  % Falsos negativos (versicolor clasificado como virginica)
FP = confMat(2,1);  % Falsos positivos (virginica clasificada como versicolor)
VN = confMat(2,2);  % Verdaderos negativos (virginica bien clasificada)

accuracy    = (VP + VN) / sum(confMat(:)) * 100;
precision   = VP / (VP + FP) * 100;
recall      = VP / (VP + FN) * 100;
f1          = 2 * (precision * recall) / (precision + recall);

fprintf('Exactitud  (Accuracy):  %.2f%%\n', accuracy);
fprintf('Precisión  (Precision): %.2f%%\n', precision);
fprintf('Sensibilidad (Recall):  %.2f%%\n', recall);
fprintf('F1-Score:               %.2f%%\n', f1);

%% 6. BÚSQUEDA DE HIPERPARÁMETROS (C y gamma)
% -------------------------------------------------------------------------
% La búsqueda en grilla logarítmica es el método estándar para SVM.
% Probamos combinaciones de C y gamma en rangos logarítmicos.
%
% Para cada par (C, gamma), entrenamos con validación cruzada de 5 particiones
% y calculamos el error de validación (kfoldLoss).
%
% Relación entre KernelScale (sigma) y gamma:
%   gamma = 1 / (2 * sigma^2)
%   => sigma = 1 / sqrt(2 * gamma)

fprintf('\n=== Búsqueda de hiperparámetros (grilla C x gamma) ===\n');
fprintf('(Este proceso puede tardar varios minutos)\n\n');

C_grid     = logspace(-2, 3, 10);   % C de 0.01 a 1000, 10 valores
gamma_grid = logspace(-4, 1, 10);   % gamma de 0.0001 a 10, 10 valores

acc_grid = zeros(length(C_grid), length(gamma_grid));

for ci = 1:length(C_grid)
    for gi = 1:length(gamma_grid)
        C     = C_grid(ci);
        gamma = gamma_grid(gi);
        sigma = 1 / sqrt(2 * gamma);  % Convertir gamma a KernelScale

        % Entrenar con validación cruzada de 5 particiones
        % 'CrossVal','on' junto con 'KFold' activa el modo CV
        mdl_cv = fitcsvm(X_train_std, Y_train, ...
            'KernelFunction', 'rbf', ...
            'BoxConstraint',   C,    ...
            'KernelScale',     sigma, ...
            'Standardize',    false, ...
            'CrossVal',       'on',  ...
            'KFold',           5);

        % kfoldLoss devuelve el error de clasificación (fracción de errores)
        % 1 - error = exactitud de validación cruzada
        acc_grid(ci, gi) = 1 - kfoldLoss(mdl_cv);
    end
end

% Encontrar el par óptimo
[best_acc, idx] = max(acc_grid(:));
[best_ci, best_gi] = ind2sub(size(acc_grid), idx);
best_C     = C_grid(best_ci);
best_gamma = gamma_grid(best_gi);

fprintf('Hiperparámetros óptimos encontrados:\n');
fprintf('  C     = %.4f\n', best_C);
fprintf('  gamma = %.6f\n', best_gamma);
fprintf('  Exactitud CV (5-fold): %.2f%%\n', best_acc * 100);

% Visualizar el mapa de calor de exactitud en la grilla
figure('Color','w','Position',[100 100 750 550]);
imagesc(log10(gamma_grid), log10(C_grid), acc_grid * 100);
colorbar;
colormap(jet);
xlabel('log_{10}(\gamma)', 'FontSize', 12);
ylabel('log_{10}(C)',      'FontSize', 12);
title('Exactitud de Validación Cruzada (5-fold) — Grilla C \times \gamma', ...
    'FontSize', 13);
hold on;
% Marcar el punto óptimo
plot(log10(best_gamma), log10(best_C), 'w*', 'MarkerSize', 14, 'LineWidth', 2);
text(log10(best_gamma)+0.1, log10(best_C), ...
    sprintf(' Óptimo: C=%.2f, \\gamma=%.4f', best_C, best_gamma), ...
    'Color', 'white', 'FontSize', 10);
set(gca, 'FontSize', 11);
grid on;

%% 7. RE-ENTRENAMIENTO CON HIPERPARÁMETROS ÓPTIMOS
% -------------------------------------------------------------------------
% Una vez encontrados los mejores C y gamma mediante validación cruzada,
% se re-entrena el modelo final usando TODOS los datos de entrenamiento
% (sin particiones CV). Este es el modelo que se evaluará en prueba.

best_sigma = 1 / sqrt(2 * best_gamma);

svmModel_opt = fitcsvm(X_train_std, Y_train, ...
    'KernelFunction', 'rbf',        ...
    'BoxConstraint',   best_C,      ...
    'KernelScale',     best_sigma,  ...
    'Standardize',    false);

Y_pred_opt = predict(svmModel_opt, X_test_std);
confMat_opt = confusionmat(Y_test, Y_pred_opt);
accuracy_opt = sum(diag(confMat_opt)) / sum(confMat_opt(:)) * 100;
n_sv_opt = sum(svmModel_opt.IsSupportVector);

fprintf('\n=== Modelo óptimo — Evaluación final en prueba ===\n');
fprintf('Exactitud en prueba:     %.2f%%\n', accuracy_opt);
fprintf('Vectores de soporte:     %d\n', n_sv_opt);

%% 8. VISUALIZACIÓN: FRONTERA DE DECISIÓN Y VECTORES DE SOPORTE
% -------------------------------------------------------------------------
% Para visualizar la frontera de decisión, creamos una malla densa de puntos
% sobre el espacio de características (2D) y predecimos la clase de cada punto.
% La frontera es la curva donde el score de decisión cambia de signo.
%
% La visualización solo es posible en 2D. Para más dimensiones se usarían
% técnicas de reducción dimensional (PCA, t-SNE).

% Definir límites de la malla con margen del 10%
x1_range = [min(X_train_std(:,1))-0.5, max(X_train_std(:,1))+0.5];
x2_range = [min(X_train_std(:,2))-0.5, max(X_train_std(:,2))+0.5];
h = 0.05;  % Resolución de la malla (puntos por unidad)

[x1g, x2g] = meshgrid(x1_range(1):h:x1_range(2), ...
    x2_range(1):h:x2_range(2));
Xgrid = [x1g(:), x2g(:)];

% Predecir la clase de cada punto de la malla
% El segundo output 'scores' da el valor continuo de la función de decisión
[~, scores] = predict(svmModel_opt, Xgrid);

% scores(:,2) corresponde a la clase positiva (virginica)
% Valores positivos → virginica, negativos → versicolor

figure('Color','w','Position',[100 100 800 600]);

% Fondo de colores según la predicción de la malla
contourf(x1g, x2g, reshape(scores(:,2), size(x1g)), ...
    [0 0], 'LineWidth', 1.5);
colormap([0.7 0.85 1.0; 1.0 0.85 0.85]);  % Azul claro / Rojo claro
hold on;

% Curva de nivel del score = 0 (frontera de decisión)
contour(x1g, x2g, reshape(scores(:,2), size(x1g)), ...
    [0 0], 'k-', 'LineWidth', 2);

% Puntos de entrenamiento (normalizados)
% Separar por clase para colorear diferente
idx_vers = strcmp(cellstr(Y_train), 'versicolor');
idx_virg = strcmp(cellstr(Y_train), 'virginica');

scatter(X_train_std(idx_vers,1), X_train_std(idx_vers,2), ...
    40, 'b', 'o', 'filled', 'MarkerEdgeColor','k', 'LineWidth', 0.5);
scatter(X_train_std(idx_virg,1), X_train_std(idx_virg,2), ...
    40, 'r', 'o', 'filled', 'MarkerEdgeColor','k', 'LineWidth', 0.5);

% Resaltar los vectores de soporte con símbolo especial
SV = svmModel_opt.SupportVectors;  % Coordenadas de vectores de soporte
scatter(SV(:,1), SV(:,2), 120, 'k', 'd', 'LineWidth', 2);

% Puntos de prueba (marcados con X para distinguirlos)
scatter(X_test_std(strcmp(cellstr(Y_test),'versicolor'),1), ...
    X_test_std(strcmp(cellstr(Y_test),'versicolor'),2), ...
    60, 'b', 'x', 'LineWidth', 2);
scatter(X_test_std(strcmp(cellstr(Y_test),'virginica'),1), ...
    X_test_std(strcmp(cellstr(Y_test),'virginica'),2), ...
    60, 'r', 'x', 'LineWidth', 2);

legend({'Región versicolor', 'Región virginica', ...
    'Frontera de decisión', ...
    'Entren. versicolor', 'Entren. virginica', ...
    'Vectores de soporte', ...
    'Prueba versicolor', 'Prueba virginica'}, ...
    'Location', 'northwest', 'FontSize', 9);

xlabel('Long. pétalo (normalizada)', 'FontSize', 12);
ylabel('Ancho pétalo (normalizado)',  'FontSize', 12);
title(sprintf('Frontera de Decisión SVM (RBF)  |  C=%.2f, \\gamma=%.4f  |  Acc=%.1f%%', ...
    best_C, best_gamma, accuracy_opt), 'FontSize', 13);
grid on;
set(gca, 'FontSize', 11);

fprintf('\nFrontera de decisión graficada.\n');
fprintf('Los diamantes (◆) son los %d vectores de soporte.\n', n_sv_opt);
fprintf('Las X son los puntos del conjunto de prueba.\n');

%% =========================================================================
%  PARTE II: SUPPORT VECTOR REGRESSION (epsilon-SVR)
%  Dataset: carsmall — predicción de consumo MPG
%  Entradas: potencia del motor (Horsepower) y peso (Weight)
% =========================================================================

fprintf('\n%s\n', repmat('=',1,60));
fprintf('  PARTE II: SUPPORT VECTOR REGRESSION (epsilon-SVR)\n');
fprintf('%s\n\n', repmat('=',1,60));

%% 9. CARGA DE DATOS DE REGRESIÓN
% -------------------------------------------------------------------------
% 'carsmall' contiene datos de automóviles: características físicas y
% consumo (MPG). Es un dataset clásico para regresión.
%
% Algunas filas tienen valores NaN (datos faltantes). Se eliminan con
% un índice lógico antes de cualquier procesamiento.

load carsmall;

X_reg = [Horsepower, Weight];  % Matriz de entradas (N x 2)
y_reg = MPG;                   % Variable objetivo: millas por galón

% Eliminar filas con datos faltantes (NaN)
idx_valid = ~any(isnan([X_reg, y_reg]), 2);
X_reg = X_reg(idx_valid, :);
y_reg = y_reg(idx_valid);

fprintf('Dataset de regresión: %d muestras válidas\n', length(y_reg));
fprintf('Rango MPG: [%.1f, %.1f] millas/galón\n', min(y_reg), max(y_reg));

%% 10. PREPARACIÓN Y NORMALIZACIÓN PARA SVR
% -------------------------------------------------------------------------
% Mismo procedimiento de normalización que para clasificación.
% Ambas entradas (caballos de fuerza y peso) tienen escalas muy diferentes
% (e.g., 50-200 HP vs. 1500-5000 lbs), por lo que la normalización
% es especialmente importante aquí.
%
% También normalizamos y_reg para estabilidad numérica del SVR.

% Partición 70/30
N_reg = length(y_reg);
idx_perm = randperm(N_reg);
N_train_reg = round(0.7 * N_reg);

X_reg_train = X_reg(idx_perm(1:N_train_reg), :);
y_reg_train = y_reg(idx_perm(1:N_train_reg));
X_reg_test  = X_reg(idx_perm(N_train_reg+1:end), :);
y_reg_test  = y_reg(idx_perm(N_train_reg+1:end));

% Normalización de entradas
mu_reg    = mean(X_reg_train);
sigma_reg = std(X_reg_train);
X_reg_train_std = (X_reg_train - mu_reg) ./ sigma_reg;
X_reg_test_std  = (X_reg_test  - mu_reg) ./ sigma_reg;

fprintf('Entrenamiento SVR: %d muestras | Prueba: %d muestras\n', ...
    N_train_reg, N_reg - N_train_reg);

%% 11. ENTRENAMIENTO DEL MODELO SVR (epsilon-SVR)
% -------------------------------------------------------------------------
% fitrsvm: función de MATLAB para Support Vector Regression
%
% Parámetros clave adicionales respecto a fitcsvm:
%   'Epsilon', epsilon → anchura del tubo de tolerancia.
%                        Residuos menores que epsilon NO se penalizan.
%                        Un epsilon mayor produce más vectores de soporte
%                        y una función más suave.
%
% La función de pérdida es la pérdida epsilon-insensible:
%   L(y, f(x)) = max(0, |y - f(x)| - epsilon)
%
% Se elige epsilon = 0.5 (½ mpg de tolerancia) como valor inicial
% razonable para este problema.

fprintf('\n=== Entrenamiento epsilon-SVR (C=10, epsilon=0.5) ===\n');
svrModel = fitrsvm(X_reg_train_std, y_reg_train, ...
    'KernelFunction', 'rbf',   ...
    'BoxConstraint',   10,     ...  % C = 10 (mayor que en clasificación)
    'Epsilon',         0.5,    ...  % Tubo de tolerancia: 0.5 mpg
    'KernelScale',    'auto',  ...
    'Standardize',    false);

n_sv_svr = sum(svrModel.IsSupportVector);
fprintf('Vectores de soporte SVR: %d de %d\n', n_sv_svr, N_train_reg);

%% 12. EVALUACIÓN DEL MODELO SVR
% -------------------------------------------------------------------------
% Para regresión, las métricas principales son:
%   RMSE: raíz del error cuadrático medio (unidades originales)
%   MAE:  error absoluto medio
%   R²:   coeficiente de determinación (0 = modelo nulo, 1 = perfecto)

y_pred_svr = predict(svrModel, X_reg_test_std);

% Cálculo de métricas
RMSE = sqrt(mean((y_pred_svr - y_reg_test).^2));
MAE  = mean(abs(y_pred_svr - y_reg_test));
SS_res = sum((y_pred_svr - y_reg_test).^2);
SS_tot = sum((y_reg_test - mean(y_reg_test)).^2);
R2   = 1 - SS_res / SS_tot;

fprintf('\n=== Métricas SVR en conjunto de prueba ===\n');
fprintf('RMSE: %.4f mpg\n', RMSE);
fprintf('MAE:  %.4f mpg\n', MAE);
fprintf('R²:   %.4f\n', R2);

%% 13. VISUALIZACIÓN: PREDICCIONES SVR vs. VALORES REALES
% -------------------------------------------------------------------------
% Se grafican dos visualizaciones:
% 1. Predicción vs. valor real (scatter plot): puntos sobre la diagonal
%    y = x indican predicciones perfectas.
% 2. Residuos (error de predicción): los residuos dentro del tubo epsilon
%    se colorean en verde (no contribuyeron a la función objetivo del SVR).

% Gráfica 1: Predicción vs. Real
figure('Color','w','Position',[100 100 1200 480]);
subplot(1,2,1);
scatter(y_reg_test, y_pred_svr, 50, 'b', 'filled', 'MarkerFaceAlpha', 0.7);
hold on;
% Línea de predicción perfecta (diagonal y = x)
min_val = min([y_reg_test; y_pred_svr]);
max_val = max([y_reg_test; y_pred_svr]);
plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 1.5);
xlabel('MPG Real',       'FontSize', 12);
ylabel('MPG Predicho',   'FontSize', 12);
title(sprintf('SVR: Predicción vs. Real | R²=%.3f', R2), 'FontSize', 13);
legend('Predicciones', 'Predicción perfecta', 'Location', 'northwest');
grid on; axis equal;
set(gca, 'FontSize', 11);

% Gráfica 2: Residuos
subplot(1,2,2);
residuos = y_pred_svr - y_reg_test;
% Colorear residuos dentro/fuera del tubo epsilon
dentro_tubo = abs(residuos) <= svrModel.Epsilon;
scatter(y_reg_test(dentro_tubo),  residuos(dentro_tubo),  ...
    50, [0.2 0.7 0.2], 'filled', 'DisplayName', 'Dentro del tubo \epsilon');
hold on;
scatter(y_reg_test(~dentro_tubo), residuos(~dentro_tubo), ...
    50, [0.9 0.2 0.2], 'filled', 'DisplayName', 'Fuera del tubo \epsilon');
% Líneas del tubo epsilon
yline( svrModel.Epsilon, 'k--', 'LineWidth', 1.5, ...
    'Label', sprintf('+\\epsilon = %.1f', svrModel.Epsilon));
yline(-svrModel.Epsilon, 'k--', 'LineWidth', 1.5, ...
    'Label', sprintf('-\\epsilon = %.1f', svrModel.Epsilon));
yline(0, 'k-', 'LineWidth', 1);
xlabel('MPG Real',        'FontSize', 12);
ylabel('Residuo (pred - real)', 'FontSize', 12);
title('Residuos del SVR y tubo \epsilon-insensible', 'FontSize', 13);
legend('Location', 'best', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 11);

fprintf('\nNota: los puntos VERDES (dentro del tubo) no contribuyeron\n');
fprintf('a la función objetivo del SVR durante el entrenamiento.\n');
fprintf('Solo los puntos ROJOS (fuera del tubo) generaron penalización.\n');

%% 14. RESUMEN FINAL
% -------------------------------------------------------------------------
fprintf('\n%s\n', repmat('=',1,60));
fprintf('  RESUMEN FINAL\n');
fprintf('%s\n', repmat('=',1,60));
fprintf('\n--- CLASIFICACIÓN SVM (Iris versicolor vs. virginica) ---\n');
fprintf('  Kernel:             RBF\n');
fprintf('  C óptimo:           %.4f\n', best_C);
fprintf('  gamma óptimo:       %.6f\n', best_gamma);
fprintf('  Vectores de soporte:%d\n', n_sv_opt);
fprintf('  Exactitud (prueba): %.2f%%\n\n', accuracy_opt);

fprintf('--- REGRESIÓN SVR (consumo MPG) ---\n');
fprintf('  Kernel:             RBF\n');
fprintf('  C:                  10\n');
fprintf('  Epsilon (tubo):     %.1f mpg\n', svrModel.Epsilon);
fprintf('  Vectores de soporte:%d\n', n_sv_svr);
fprintf('  RMSE (prueba):      %.4f mpg\n', RMSE);
fprintf('  R² (prueba):        %.4f\n', R2);
fprintf('%s\n', repmat('=',1,60));

% =========================================================================
%  FIN DEL SCRIPT
% =========================================================================
%
%  EJERCICIOS PROPUESTOS:
%  1. Cambia C a {0.01, 1, 1000} con gamma fijo. Observa cómo cambia el
%     número de vectores de soporte y la frontera de decisión.
%  2. Prueba el kernel 'polynomial' con grado 2 y 3. Compara las fronteras.
%  3. Usa las 4 columnas de 'meas' (todas las características). ¿Mejora
%     la exactitud? ¿Puedes seguir visualizando la frontera?
%  4. Extiende el ejemplo a 3 clases usando fitcecoc con templateSVM.
%  5. Para SVR: varía epsilon (0.1, 0.5, 2.0) y observa cuántos puntos
%     quedan dentro del tubo en cada caso.
% =========================================================================