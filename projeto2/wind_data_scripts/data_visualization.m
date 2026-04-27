%% Projeto 2: Redes Neuronais Artificiais - Wind Speed
% Script: Visualização Exploratória de Dados (EDA)
clc; clearvars; close all;

%% 1. Carregamento dos Dados Processados
path = "windSpeed_Processed.csv";

if ~isfile(path)
    error('Ficheiro não encontrado! Execute primeiro o script data_cleaning.m para gerar o CSV.');
end

data = readtable(path);

% Garantir que a coluna de Tempo é lida como datetime para os plots temporais
if iscell(data.Time) || isstring(data.Time)
    data.Time = datetime(data.Time, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
end

fprintf('=============================================\n');
fprintf('         ANÁLISE EXPLORATÓRIA (EDA)         \n');
fprintf('=============================================\n');

%% 2. Visualização de Dados e Continuidade
fprintf('Gerando visualizações estatísticas...\n');

gaps_longos = sum(data.TimeDiff >= 60);
fprintf('Contagem de intervalos de tempo >= 60 segundos: %d\n', gaps_longos);

% Figura 1: Distribuição das Features
figure('Name', 'Features Distribution', 'NumberTitle', 'off', 'Position', [100, 100, 800, 900]);
subplot(3,1,1);
histogram(data.WindSpeed, 'FaceColor', [0 0.4470 0.7410], 'EdgeColor', 'w');
set(gca, 'YScale', 'log'); 
title('Wind Speed Distribution (Log Scale)');
xlabel('Wind Speed (m/s)'); ylabel('Frequency (Log)'); grid on;

subplot(3,1,2);
histogram(data.WindDirection, 'FaceColor', [0.8500 0.3250 0.0980], 'EdgeColor', 'w');
title('Wind Direction Distribution');
xlabel('Direction (Degrees)'); ylabel('Frequency'); grid on;

subplot(3,1,3);
histogram(data.Wind_Acc, 'FaceColor', [0.9290 0.6940 0.1250], 'EdgeColor', 'w');
set(gca, 'YScale', 'log'); 
title('Wind Acceleration (\Delta v) Distribution (Log Scale)');
xlabel('Acceleration (m/s^2)'); ylabel('Frequency (Log)'); grid on;

% Figura 2: Análise de Gaps Temporais
figure('Name', 'Time Gaps Analysis', 'NumberTitle', 'off', 'Position', [150, 150, 800, 600]);

% Subplot 1: Histograma dos intervalos normais (Zoom na amostragem)
subplot(2,1,1);
normalGaps = data.TimeDiff(data.TimeDiff < 60); 
histogram(normalGaps, 'BinWidth', 1, 'FaceColor', [0.4660 0.6740 0.1880], 'EdgeColor', 'w');
set(gca, 'YScale', 'log');
title('Time Intervals Histogram (Normal Sampling < 60s) - Log Scale');
xlabel('Seconds between Samples'); ylabel('Count (Log)'); grid on;

% Subplot 2: Ocorrência de Gaps ao longo do tempo (Revertido para pontos)
subplot(2,1,2);
plot(data.Time, data.TimeDiff, '.', 'MarkerSize', 8, 'Color', [0.6350 0.0780 0.1840]);
title('Gaps Occurrence over Timeline');
xlabel('Time'); ylabel('Gap Duration (seconds)');
grid on;

%% 3. Análise de Features (Correlação e Dispersão)
fprintf('Analisando correlações entre features e o Target...\n');

% Figura 3: Matriz de Correlação Expandida
figure('Name', 'Correlation Matrix', 'NumberTitle', 'off', 'Position', [200, 200, 800, 650]);
vars_corr = {'WindSpeed', 'Dir_Sin', 'Dir_Cos', 'Hour_Sin', 'Hour_Cos', 'Wind_Acc', 'Target_WindSpeed'};
vars_display = {'Wind Speed', 'Dir Sin', 'Dir Cos', 'Hour Sin', 'Hour Cos', 'Wind Acc', 'Target Speed'};
corr_matrix = corr(table2array(data(:, vars_corr)), 'Rows', 'complete');
heatmap(vars_display, vars_display, corr_matrix, 'Title', 'Correlation Matrix');

% Figura 4: Dispersões (Scatter Plots) vs Target
figure('Name', 'Scatter Plots vs Target', 'NumberTitle', 'off', 'Position', [250, 250, 1200, 800]);
scatter_vars = {'WindSpeed', 'Dir_Sin', 'Dir_Cos', 'Hour_Sin', 'Hour_Cos', 'Wind_Acc'};
titles = {'Current Wind Speed', 'Direction Sine', 'Direction Cosine', 'Hour Sine', 'Hour Cosine', 'Wind Acceleration'};
x_labels = {'Wind Speed (m/s)', 'Sin(\theta)', 'Cos(\theta)', 'Sin(Hour)', 'Cos(Hour)', 'Acceleration (m/s^2)'};

for i = 1:6
    subplot(2, 3, i);
    scatter(data.(scatter_vars{i}), data.Target_WindSpeed, 10, 'filled', 'MarkerFaceAlpha', 0.2);
    title([titles{i} ' vs Target (+60s)']);
    xlabel(x_labels{i}); ylabel('Target (+60s) (m/s)'); grid on;
end