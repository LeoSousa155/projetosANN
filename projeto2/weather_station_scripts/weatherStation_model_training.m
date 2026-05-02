%% PROJETO 2: Redes Neuronais - Apparent Temperature
% Script: Visualização Exploratória de Dados (EDA)
clc; clearvars; close all;

%% 1 - Importação e Analise Inicial

data = readtable('weatherHistory.csv');

fprintf('--- Exploração Inicial ---\n');
fprintf('Observações: %d | Features: %d\n', size(data, 1), size(data, 2));
head(data)
summary(data)
disp('Valores em falta (NaN) por feature:');
disp(array2table(sum(ismissing(data)), 'VariableNames', data.Properties.VariableNames));

% contagem de zeros

zeros_count = zeros(1, size(data, 2));
for i = 1:size(data, 2)
    col_data = data{:, i};
    if isnumeric(col_data)
        zeros_count(i) = sum(col_data == 0, "omitnan");
        else
        % Contabilizar de forma segura mesmo em colunas lidas como string/texto
        num_data = str2double(string(col_data));
        zeros_count(i) = sum(num_data == 0, 'omitnan');
    end
end
disp('Contagem de valores iguais a zero:');
disp(array2table(zeros_count, 'VariableNames', data.Properties.VariableNames));

%% 2 - Exploração e Analise dos Dados (EDA)
fprintf('---------------------------------\n');
fprintf('Visualizações Estatisticas\n');
fprintf('---------------------------------\n');

features = data.Properties.VariableNames;
n = length(features);

% ------ GRAPH 1 - HISTOGRAMA --------------

figure('Name', 'Features Distribution', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 800]);

for i = 1:n
    % Define a posição do gráfico: 3 colunas, linhas automáticas
    subplot(ceil(n/3), 3, i);
    
    % Extrair os dados da coluna atual
    % data.(features{i}) acede dinamicamente à coluna pelo nome
    histogram(data.(features{i}), 'FaceColor', [0.2 0.5 0.7]);
    
    % Título dinâmico usando o nome da feature
    title(['Distribuição de: ' features{i}], 'Interpreter', 'none'); 
    grid on;
end


% ------ GRAPH 2 - BOXPLOTS --------------

figure('Name', 'Boxplots: Todas as Features', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 500]);
for i = 1:length(features)
    subplot(1, length(features), i);
    boxplot(data.(features{i}));
    title(features{i}, 'Interpreter', 'none', 'FontSize', 8);
    grid on;
end
sgtitle('Identificação de Outliers por Variável');

% ------ GRAPH 3 - MATRIZ DE CORRELAÇÃO --------------

figure('Name', 'Correlação Completa', 'NumberTitle', 'off', 'Position', [200, 200, 800, 600]);
corrMat = corr(table2array(data));
h = heatmap(features, features, corrMat, 'CellLabelFormat', '%.2f');
h.Title = 'Correlação: Como cada variável afeta a Apparent Temperature';
h.Colormap = jet;

% ------ GRAPH 4 - SCATTERPLOT --------------

targetVar = 'ApparentTemperature_C_';

todasVars = data.Properties.VariableNames;
inputVars = setdiff(todasVars, {targetVar, 'LoudCover'}, 'stable');

numVars = length(inputVars);
numCols = 3; % Fixamos 3 colunas para ficar legível
numRows = ceil(numVars / numCols); % Calcula as linhas necessárias (ex: 7 vars -> 3 linhas)

figure('Name', 'Relação com o Target (Apparent Temperature)', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 800]);

for i = 1:numVars
    % O subplot agora é dinâmico: usa numRows em vez de 2
    subplot(numRows, numCols, i);
    
    % Dados originais
    xData = data.(inputVars{i});
    yData = data.(targetVar);
    
    % Criar scatter plot
    scatter(xData, yData, 5, 'filled', 'MarkerFaceAlpha', 0.2);
    
    xlabel(inputVars{i}, 'Interpreter', 'none');
    ylabel('Apparent Temp (C)');
    title(['Rel: ' inputVars{i}], 'Interpreter', 'none');
    grid on;
    
    % Adicionar linha de tendência (ignora NaNs para não dar erro no polyfit)
    idxValid = ~isnan(xData) & ~isnan(yData);
    if sum(idxValid) > 2
        p = polyfit(xData(idxValid), yData(idxValid), 1);
        hold on;
        plot(xData, polyval(p, xData), 'r-', 'LineWidth', 1.5);
    end
end


%% 3 - Limpeza inicial dos dados

% Com base nestes dados podemos remover a coluna de Cloud Cover pois
% apresenta demasiados zeros e nao apresenta dados que possam ser
% relevantes para a rede neuronal

%  Remover a coluna problemática (Loud Cover)
data.LoudCover = [];

%  Tratar a Pressão (Imputação em vez de Remoção)
% Substituímos zeros por NaN para usar as funções inteligentes do MATLAB
data.Pressure_millibars_(data.Pressure_millibars_ < 900) = NaN;
% Para regressão (não-série temporal), usamos a mediana em vez de interpolação linear
data.Pressure_millibars_ = fillmissing(data.Pressure_millibars_, 'constant', median(data.Pressure_millibars_, 'omitnan'));

%  Tratar a Humidade (se houver zeros isolados)
data.Humidity(data.Humidity <= 0) = NaN;
data.Humidity = fillmissing(data.Humidity, 'constant', median(data.Humidity, 'omitnan'));


%Identificar e corrigir zeros na HUMIDADE (<= 0)
% A humidade costuma estar entre 0.2 e 1.0 (20% a 100%)
idxHumidadeZero = data.Humidity <= 0;
data.Humidity(idxHumidadeZero) = NaN;
data.Humidity = fillmissing(data.Humidity, 'constant', median(data.Humidity, 'omitnan'));

%  Verificação Final: Se ainda restarem NaNs (ex: no início/fim do ficheiro)
% que a interpolação não apanhou, removemos apenas esses (serão muito poucos)
data = rmmissing(data);

%% 4 -Feature Engineering

%decomposição do Wind Bearing para Cosseno e Seno

data.Wind_Sin = sin(deg2rad(data.WindBearing_degrees_));
data.Wind_Cos = cos(deg2rad(data.WindBearing_degrees_));

data.WindBearing_degrees_ = [];

% Heat Index Proxy
% Temperatura X Humidade

data.Heat_Index_Proxy = data.Temperature_C_ .* data.Humidity;

% Wind Chill Effect
% Vento X Temperatura

data.Wind_Chill_Effect = data.WindSpeed_km_h_ .* data.Temperature_C_;

% 2. Interação Humidade e Pressão (Densidade/Estabilidade do ar)
data.Hum_Press_Ratio = data.Humidity .* data.Pressure_millibars_;

%% 5 - Normalização dos Dados

targetName = 'ApparentTemperature_C_';
inputNames = setdiff(data.Properties.VariableNames, targetName, 'stable');

X = data{:, inputNames}';
y = data{:, targetName}';

% Atualizamos a lista de features para a nova tabela
features = data.Properties.VariableNames;
n = length(features);

% graficos com features novas

% ------ GRAPH 1 - HISTOGRAMA --------------

figure('Name', 'Features Distribution com features novas', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 800]);

for i = 1:n
    % Define a posição do gráfico: 3 colunas, linhas automáticas
    subplot(ceil(n/3), 3, i);
    
    % Extrair os dados da coluna atual
    % data.(features{i}) acede dinamicamente à coluna pelo nome
    histogram(data.(features{i}), 'FaceColor', [0.2 0.5 0.7]);
    
    % Título dinâmico usando o nome da feature
    title(['Distribuição de: ' features{i}], 'Interpreter', 'none'); 
    grid on;
end


% ------ GRAPH 2 - BOXPLOTS --------------

figure('Name', 'Boxplots: Todas as Features novas', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 500]);
for i = 1:length(features)
    subplot(1, length(features), i);
    boxplot(data.(features{i}));
    title(features{i}, 'Interpreter', 'none', 'FontSize', 8);
    grid on;
end
sgtitle('Identificação de Outliers por Variável');

% ------ GRAPH 3 - MATRIZ DE CORRELAÇÃO --------------

figure('Name', 'Correlação Completa com features novas', 'NumberTitle', 'off', 'Position', [200, 200, 800, 600]);
corrMat = corr(table2array(data));
h = heatmap(features, features, corrMat, 'CellLabelFormat', '%.2f');
h.Title = 'Correlação: Como cada variável afeta a Apparent Temperature';
h.Colormap = jet;

% ------ GRAPH 4 - SCATTERPLOT --------------

targetVar = 'ApparentTemperature_C_';

todasVars = data.Properties.VariableNames;
inputVars = setdiff(todasVars, {targetVar, 'LoudCover'}, 'stable');

numVars = length(inputVars);
numCols = 3; % Fixamos 3 colunas para ficar legível
numRows = ceil(numVars / numCols); % Calcula as linhas necessárias (ex: 7 vars -> 3 linhas)

figure('Name', 'Relação com o Target (Apparent Temperature) com features novas', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 800]);

for i = 1:numVars
    % O subplot agora é dinâmico: usa numRows em vez de 2
    subplot(numRows, numCols, i);
    
    % Dados originais
    xData = data.(inputVars{i});
    yData = data.(targetVar);
    
    % Criar scatter plot
    scatter(xData, yData, 5, 'filled', 'MarkerFaceAlpha', 0.2);
    
    xlabel(inputVars{i}, 'Interpreter', 'none');
    ylabel('Apparent Temp (C)');
    title(['Rel: ' inputVars{i}], 'Interpreter', 'none');
    grid on;
    
    % Adicionar linha de tendência (ignora NaNs para não dar erro no polyfit)
    idxValid = ~isnan(xData) & ~isnan(yData);
    if sum(idxValid) > 2
        p = polyfit(xData(idxValid), yData(idxValid), 1);
        hold on;
        plot(xData, polyval(p, xData), 'r-', 'LineWidth', 1.5);
    end
end


% Removemos Wind_Sen e Cos porque a sua correlação é basicamente nula, ou
% seja é só ruido.
data_final = removevars(data, {'Wind_Sin', 'Wind_Cos'});

inputNames = setdiff(data_final.Properties.VariableNames, targetName, 'stable');

%% 6 - Partição de Dados e Configuração de Hardware

X_raw = data_final{:, inputNames}';
y_raw = data_final{:, targetName}';

% Partição Aleatória Rigorosa (90% Treino/Validação | 10% Teste)
total_samples = length(y_raw);
split_idx = floor(0.90 * total_samples);

% Baralhar os índices para garantir amostragem representativa
rng(42); % Fixar semente para reprodutibilidade e isolamento do cofre
shuffled_idx = randperm(total_samples);

inputs_tv = X_raw(:, shuffled_idx(1:split_idx));
targets_tv = y_raw(shuffled_idx(1:split_idx));

inputs_test = X_raw(:, shuffled_idx(split_idx+1:end));
targets_test = y_raw(shuffled_idx(split_idx+1:end));

fprintf('=============================================\n');
fprintf('         PREPARAÇÃO DOS DADOS (SPLIT)        \n');
fprintf('=============================================\n');
fprintf('Treino/Validação (Optimização): %d amostras\n', split_idx);
fprintf('Teste Final (Cofre isolado):    %d amostras\n\n', total_samples - split_idx);

use_gpu_flag = 'no';
train_algo = 'trainlm';

%% 7 - Definição do Espaço de Otimização (Grid Search)

% Teorema do Limite Central (N >= 30)
num_runs = 30; 

% Construção Automática da Grelha
neurons_list = [5, 10, 15, 20];
act_list = {'tansig', 'poslin'};

idx = 1;
opt_space = struct('neurons', {}, 'activation', {});
for n = 1:length(neurons_list)
    for a = 1:length(act_list)
        opt_space(idx).neurons = neurons_list(n);
        opt_space(idx).activation = act_list{a};
        idx = idx + 1;
    end
end
num_points = length(opt_space);

mean_mse_val = zeros(num_points, 1);
var_mse_val = zeros(num_points, 1);

fprintf('=============================================\n');
fprintf('    OTIMIZAÇÃO DE REDE NEURONAL (30 RUNS)    \n');
fprintf('=============================================\n');
fprintf('Total de redes a treinar: %d treinos\n\n', num_points * num_runs);

%% 8 - Ciclo de Otimização
for p = 1:num_points
    n_neurons = opt_space(p).neurons;
    act_func = opt_space(p).activation;
    
    fprintf('A avaliar Ponto %d/%d [Neurónios: %2d | Ativação: %6s]:\n', p, num_points, n_neurons, act_func);
    
    val_mse_runs = zeros(num_runs, 1);
    
    for r = 1:num_runs
        fprintf('  -> Treinando modelo %2d/%d do Ponto %d ... ', r, num_runs, p);
        
        net = fitnet(n_neurons, train_algo);
        net.layers{1}.transferFcn = act_func;
        
        % Normalização interna (Apenas calculada no Treino)
        net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
        net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};
        
        net.trainParam.max_fail = 10;
        net.trainParam.showWindow = false; % ESCONDER JANELA
        net.performFcn = 'mse';
        
        % Divisão aleatória na validação cruzada interna
        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = 0.85; 
        net.divideParam.valRatio   = 0.15;
        net.divideParam.testRatio  = 0.00; 
        
        [net, tr] = train(net, inputs_tv, targets_tv, 'useGPU', use_gpu_flag);
        
        val_mse_runs(r) = tr.best_vperf;
        fprintf('Concluído! (Val MSE: %.4f)\n', val_mse_runs(r));
    end
    
    mean_mse_val(p) = mean(val_mse_runs);
    var_mse_val(p) = var(val_mse_runs);
    
    fprintf('=> RESUMO PONTO %d | Média Val MSE: %.4f | Variância: %.6f\n\n', p, mean_mse_val(p), var_mse_val(p));
end

%% 9 - Tabela de Resultados Finais e Gráfico
fprintf('\n=============================================\n');
fprintf('         RESULTADOS DA OTIMIZAÇÃO            \n');
fprintf('=============================================\n');

neur_col = [opt_space.neurons]';
act_col = string({opt_space.activation}');

T = table(neur_col, act_col, mean_mse_val, var_mse_val, ...
    'VariableNames', {'Neuronios', 'Ativacao', 'Media_Val_MSE', 'Variancia_Val_MSE'});
disp(T);

fprintf('A gerar gráfico de comparação (Error Bar Chart)...\n');
config_labels = strcat(string(neur_col), " neur. (", act_col, ")");

figure('Name', 'Optimization Performance', 'NumberTitle', 'off', 'Position', [200, 200, 900, 500]);
std_mse_val = sqrt(var_mse_val);
errorbar(1:num_points, mean_mse_val, std_mse_val, 'k', 'LineStyle', 'none', 'Marker', '.', 'MarkerSize', 25, 'LineWidth', 1.5, 'CapSize', 12);
hold on;
xticks(1:num_points);
xticklabels(config_labels);
xtickangle(25);
xlabel('Configuração do Modelo');
ylabel('Mean Validation MSE (Erro)');
title('Comparação de Hiperparâmetros (Média e Desvio Padrão em 30 Execuções)');
grid on;
hold off;

%% 10 - Treino do Modelo Final (Best Configuration)
[~, best_idx] = min(mean_mse_val);
best_config = opt_space(best_idx);

fprintf('\n=============================================\n');
fprintf('         TREINO DO MODELO FINAL (BEST)       \n');
fprintf('=============================================\n');
fprintf('Melhor Ponto: %d (Neurónios: %d, Ativação: %s)\n', best_idx, best_config.neurons, best_config.activation);
fprintf('A treinar modelo definitivo e a abrir janela (Show Window)...\n');

final_net = fitnet(best_config.neurons, train_algo);
final_net.layers{1}.transferFcn = best_config.activation;
final_net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
final_net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};
final_net.trainParam.max_fail = 10;
final_net.performFcn = 'mse';
final_net.divideFcn = 'dividerand';
final_net.divideParam.trainRatio = 0.85; 
final_net.divideParam.valRatio   = 0.15;
final_net.divideParam.testRatio  = 0.00;

final_net.trainParam.showWindow = true; % ATIVAR JANELA NO ÚLTIMO MODELO

[final_net, final_tr] = train(final_net, inputs_tv, targets_tv, 'useGPU', use_gpu_flag);

%% 11 - Avaliação Cega no Cofre (Test Set com MAE)
fprintf('\n=============================================\n');
fprintf('         AVALIAÇÃO FINAL (TEST SET)          \n');
fprintf('=============================================\n');

final_preds = final_net(inputs_test);
final_mae = mae(targets_test - final_preds);

fprintf('Métrica Final para o Relatório:\n');
fprintf('=> MAE (Mean Absolute Error) Cego: %.4f\n', final_mae);
fprintf('=============================================\n');