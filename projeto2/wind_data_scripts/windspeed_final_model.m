%% Projeto 2: Redes Neuronais Artificiais - Wind Speed
% Script: Treino do Modelo Final (Separação para Reprodutibilidade)

clc; clearvars; close all;

%% 1. Carregamento dos Dados
path = "windSpeed_ModelReady.csv";
if ~isfile(path)
    error('Ficheiro não encontrado! Execute primeiro o script data_cleaning.m');
end
data = readtable(path);
inputs = table2array(data(:, {'WindSpeed', 'Dir_Sin', 'Dir_Cos', 'Hour_Sin', 'Hour_Cos', 'Wind_Acc'}))';
targets = data.Target_WindSpeed';

%% 2. Partição Cronológica Rigorosa
% Para séries temporais, a divisão de teste mantém os últimos 10%
total_samples = length(targets);
split_idx = floor(0.90 * total_samples);

inputs_tv = inputs(:, 1:split_idx);
targets_tv = targets(1:split_idx);

inputs_test = inputs(:, split_idx+1:end);
targets_test = targets(split_idx+1:end);

%% 3. Configuração da Rede Final (20 Neurónios, 'poslin')
neurons = 20;
activation = 'poslin';
train_algo = 'trainlm';
use_gpu_flag = 'no';

fprintf('A configurar o modelo final com %d neurónios e ativação %s...\n', neurons, activation);
net = fitnet(neurons, train_algo);
net.layers{1}.transferFcn = activation;
net.trainParam.max_fail = 10;
net.performFcn = 'mse';

% Como é série temporal, usamos divideblock de forma fixa
net.divideFcn = 'divideblock';
net.divideParam.trainRatio = 0.85; 
net.divideParam.valRatio   = 0.15;
net.divideParam.testRatio  = 0.00;

%% 4. Carregar ou Guardar Pesos Iniciais
initial_weights_file = 'windspeed_initial_weights.mat';

if isfile(initial_weights_file)
    fprintf('A carregar os pesos iniciais fornecidos no ficheiro "%s"...\n', initial_weights_file);
    load(initial_weights_file, 'net');
else
    fprintf('A inicializar novos pesos para a rede...\n');
    rng(42); % Semente para a inicialização
    net = configure(net, inputs_tv, targets_tv);
    save(initial_weights_file, 'net');
    fprintf('Pesos iniciais guardados em "%s".\n', initial_weights_file);
end

%% 5. Treino do Modelo
net.trainParam.showWindow = true; % Mostrar janela para ver o progresso

% Semente para reprodutibilidade no treino
rng(42);
fprintf('A iniciar o treino do modelo...\n');
[net, tr] = train(net, inputs_tv, targets_tv, 'useGPU', use_gpu_flag);

%% 6. Guardar Pesos Finais
final_weights_file = 'windspeed_final_weights.mat';
save(final_weights_file, 'net');
fprintf('Pesos finais pós-treino guardados em "%s"\n', final_weights_file);

%% 7. Avaliação Cega no Test Set
final_preds = net(inputs_test);

% Reverter a transformação logarítmica (expm1) para voltar a m/s
final_preds_ms = expm1(final_preds);
targets_test_ms = expm1(targets_test);

final_mae = mae(targets_test_ms - final_preds_ms);

fprintf('\n=============================================\n');
fprintf('         AVALIAÇÃO FINAL (TEST SET)          \n');
fprintf('=============================================\n');
fprintf('=> MAE (Mean Absolute Error) Cego: %.4f m/s\n', final_mae);
fprintf('=============================================\n\n');

%% 8. Guardar Figuras do Treino
fprintf('A gerar e guardar as figuras dos resultados...\n');
fig_dir = 'figures_windspeed';
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

% Performance
f1 = figure('Name', 'Performance', 'Visible', 'off');
plotperform(tr);
saveas(f1, fullfile(fig_dir, 'performance.png'));
savefig(f1, fullfile(fig_dir, 'performance.fig'));
close(f1);

% Train State
f2 = figure('Name', 'Train State', 'Visible', 'off');
plottrainstate(tr);
saveas(f2, fullfile(fig_dir, 'train_state.png'));
savefig(f2, fullfile(fig_dir, 'train_state.fig'));
close(f2);

% Error Histogram
f3 = figure('Name', 'Error Histogram', 'Visible', 'off');
y_tv = net(inputs_tv);
e_tv = targets_tv - y_tv;
e_test = targets_test - final_preds;
ploterrhist(e_tv(tr.trainInd), 'Training', e_tv(tr.valInd), 'Validation', e_test, 'Testing');
saveas(f3, fullfile(fig_dir, 'error_histogram.png'));
savefig(f3, fullfile(fig_dir, 'error_histogram.fig'));
close(f3);

% Regression
f4 = figure('Name', 'Regression', 'Visible', 'off');
plotregression(targets_tv(tr.trainInd), y_tv(tr.trainInd), 'Training', ...
               targets_tv(tr.valInd), y_tv(tr.valInd), 'Validation', ...
               targets_test, final_preds, 'Testing');
saveas(f4, fullfile(fig_dir, 'regression.png'));
savefig(f4, fullfile(fig_dir, 'regression.fig'));
close(f4);

fprintf('Figuras guardadas com sucesso no diretório "%s".\n', fig_dir);
fprintf('Processo concluído!\n');
