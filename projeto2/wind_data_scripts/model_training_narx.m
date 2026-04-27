%% Projeto 2: Redes Neuronais Artificiais - Wind Speed
% Script: Treino e Otimização com Arquitetura NARXNet (Séries Temporais)
clc; clearvars; close all;

%% 1. Carregamento dos Dados
path = "windSpeed_ModelReady.csv";

if ~isfile(path)
    error('Ficheiro não encontrado! Execute primeiro o script data_cleaning.m');
end

data = readtable(path);
inputs_mat = table2array(data(:, {'WindSpeed', 'Dir_Sin', 'Dir_Cos', 'Hour_Sin', 'Hour_Cos', 'Wind_Acc'}))';
targets_mat = data.Target_WindSpeed';

fprintf('=============================================\n');
fprintf('   CONVERSÃO PARA SEQUÊNCIAS (NARXNet)       \n');
fprintf('=============================================\n');
% Em redes dinâmicas, os dados têm de ser Cell Arrays em vez de Matrizes.
% O 'con2seq' transforma cada coluna num elemento temporal sequencial.
inputs_seq = con2seq(inputs_mat);
targets_seq = con2seq(targets_mat);
fprintf('Dados convertidos para Cell Arrays com sucesso.\n\n');

%% 2. Partição Cronológica Rigorosa (Prevenção de Data Leakage)
total_samples = length(targets_seq);
split_idx = floor(0.90 * total_samples);

X_tv = inputs_seq(1:split_idx);
T_tv = targets_seq(1:split_idx);

X_test = inputs_seq(split_idx+1:end);
T_test = targets_seq(split_idx+1:end);

fprintf('=============================================\n');
fprintf('         PREPARAÇÃO DOS DADOS (SPLIT)        \n');
fprintf('=============================================\n');
fprintf('Treino/Validação (Optimização): %d amostras\n', split_idx);
fprintf('Teste Final (Cofre isolado):    %d amostras\n\n', total_samples - split_idx);

%% 3. Configuração de Hardware (CPU vs GPU)
fprintf('=============================================\n');
fprintf('         CONFIGURAÇÃO DE HARDWARE            \n');
fprintf('=============================================\n');
if license('test', 'Distrib_Computing_Toolbox') && gpuDeviceCount > 0
    fprintf('GPU compatível detetada! Aceleração de hardware ativada.\n');
    fprintf('Algoritmo de treino alterado para Scaled Conjugate Gradient (trainscg).\n\n');
    use_gpu_flag = 'yes';
    train_algo = 'trainscg'; 
else
    fprintf('Nenhuma GPU detetada.\n');
    fprintf('Usando processamento normal em CPU com Levenberg-Marquardt (trainlm).\n\n');
    use_gpu_flag = 'no';
    train_algo = 'trainlm';
end

%% 4. Definição do Espaço de Otimização (NARX)
% Parâmetros: 1) Número de Neurónios, 2) Número de Atrasos Temporais (Delays)

num_runs = 30; % Regra do Teorema do Limite Central

% Grid Search para a NARX: 
% Testamos 3 tamanhos de neurónios e 2 níveis de memória (2 segundos vs 4 segundos)
neurons_list = [5, 10, 15];
delays_list = [2, 4];

idx = 1;
opt_space = struct('neurons', {}, 'delay', {});
for n = 1:length(neurons_list)
    for d = 1:length(delays_list)
        opt_space(idx).neurons = neurons_list(n);
        opt_space(idx).delay = delays_list(d);
        idx = idx + 1;
    end
end
num_points = length(opt_space); % 3 x 2 = 6 Pontos

mean_mse_val = zeros(num_points, 1);
var_mse_val = zeros(num_points, 1);

fprintf('=============================================\n');
fprintf('    OTIMIZAÇÃO NARXNET (30 RUNS POR PONTO)   \n');
fprintf('=============================================\n');
fprintf('Total de redes dinâmicas a treinar: %d\n', num_points * num_runs);
fprintf('Aviso: Redes NARX demoram substancialmente mais a treinar que FitNets.\n\n');

%% 5. Ciclo de Otimização
for p = 1:num_points
    n_neurons = opt_space(p).neurons;
    d_delay = opt_space(p).delay;
    
    fprintf('A avaliar Ponto %d/%d [Neurónios: %2d | Atraso (Delays): %d]:\n', p, num_points, n_neurons, d_delay);
    
    val_mse_runs = zeros(num_runs, 1);
    
    for r = 1:num_runs
        fprintf('  -> Treinando modelo %2d/%d do Ponto %d ... ', r, num_runs, p);
        
        % Configurar a NARXNet (Input Delay, Feedback Delay, Neurónios)
        net = narxnet(1:d_delay, 1:d_delay, n_neurons, 'open', train_algo);
        
        net.trainParam.showWindow = false;
        net.trainParam.max_fail = 10;
        net.performFcn = 'mse';
        net.divideFcn = 'divideblock';
        net.divideParam.trainRatio = 0.85; 
        net.divideParam.valRatio   = 0.15;
        net.divideParam.testRatio  = 0.00; 
        
        % PREPARETS: Prepara as sequências de dados para a NARX
        % Alinha as entradas e os targets com base nos Delays configurados
        [Xs, Xi, Ai, Ts] = preparets(net, X_tv, {}, T_tv);
        
        % Treinar a rede
        [net, tr] = train(net, Xs, Ts, Xi, Ai, 'useGPU', use_gpu_flag);
        
        val_mse_runs(r) = tr.best_vperf;
        fprintf('Concluído! (Val MSE: %.4f)\n', val_mse_runs(r));
    end
    
    mean_mse_val(p) = mean(val_mse_runs);
    var_mse_val(p) = var(val_mse_runs);
    
    fprintf('=> RESUMO PONTO %d | Média Val MSE: %.4f | Variância: %.6f\n\n', p, mean_mse_val(p), var_mse_val(p));
end

%% 6. Tabela de Resultados Finais
fprintf('\n=============================================\n');
fprintf('         RESULTADOS DA OTIMIZAÇÃO NARX       \n');
fprintf('=============================================\n');

neur_col = [opt_space.neurons]';
delay_col = [opt_space.delay]';

T = table(neur_col, delay_col, mean_mse_val, var_mse_val, ...
    'VariableNames', {'Neuronios', 'Delays', 'Media_Val_MSE', 'Variancia_Val_MSE'});
disp(T);

fprintf('A gerar gráfico de comparação...\n');
config_labels = strcat(string(neur_col), " neur. (", string(delay_col), " delays)");

figure('Name', 'NARX Optimization', 'NumberTitle', 'off', 'Position', [200, 200, 900, 500]);
bar(1:num_points, mean_mse_val, 0.6, 'FaceColor', [0.8500 0.3250 0.0980], 'EdgeColor', 'k', 'LineWidth', 1);
hold on;
errorbar(1:num_points, mean_mse_val, sqrt(var_mse_val), 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 12);
xticks(1:num_points); xticklabels(config_labels); xtickangle(25);
xlabel('Configuração do Modelo NARX'); ylabel('Mean Validation MSE (Erro)');
title('Desempenho NARXNet (Média e Desvio Padrão em 30 Execuções)');
grid on; hold off;

%% 7. Treino do Modelo Final (NARX)
[~, best_idx] = min(mean_mse_val);
best_config = opt_space(best_idx);

fprintf('\n=============================================\n');
fprintf('         TREINO DO MODELO FINAL (BEST)       \n');
fprintf('=============================================\n');
fprintf('Melhor Ponto: %d (Neurónios: %d, Delays: %d)\n', best_idx, best_config.neurons, best_config.delay);
fprintf('A treinar modelo NARX definitivo e a abrir janela...\n');

final_net = narxnet(1:best_config.delay, 1:best_config.delay, best_config.neurons, 'open', train_algo);
final_net.trainParam.max_fail = 10;
final_net.performFcn = 'mse';
final_net.divideFcn = 'divideblock';
final_net.divideParam.trainRatio = 0.85; 
final_net.divideParam.valRatio   = 0.15;
final_net.divideParam.testRatio  = 0.00;
final_net.trainParam.showWindow = true;

[Xs, Xi, Ai, Ts] = preparets(final_net, X_tv, {}, T_tv);
[final_net, final_tr] = train(final_net, Xs, Ts, Xi, Ai, 'useGPU', use_gpu_flag);

%% 8. Avaliação Cega no Cofre (Test Set)
fprintf('\n=============================================\n');
fprintf('         AVALIAÇÃO FINAL (TEST SET)          \n');
fprintf('=============================================\n');

% Preparar os dados do cofre para a NARXNet
[Xs_test, Xi_test, Ai_test, Ts_test] = preparets(final_net, X_test, {}, T_test);

% Prever no Test Set
final_preds = final_net(Xs_test, Xi_test, Ai_test);

final_mae = mae(cell2mat(Ts_test) - cell2mat(final_preds));
fprintf('=> MAE (Mean Absolute Error) NARX Cego: %.4f m/s\n', final_mae);
fprintf('=============================================\n');