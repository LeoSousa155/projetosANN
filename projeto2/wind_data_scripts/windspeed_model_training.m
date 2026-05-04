%% Projeto 2: Redes Neuronais Artificiais - Wind Speed
% Script: Treino e Otimização de Hiperparâmetros (Grid Search + Teorema Limite Central)
clc; clearvars; close all;

%% 1. Carregamento dos Dados
path = "windSpeed_ModelReady.csv";

if ~isfile(path)
    error('Ficheiro não encontrado! Execute primeiro o script data_cleaning.m');
end

data = readtable(path);
inputs = table2array(data(:, {'WindSpeed', 'Dir_Sin', 'Dir_Cos', 'Hour_Sin', 'Hour_Cos', 'Wind_Acc'}))';
targets = data.Target_WindSpeed';

%% 2. Partição Cronológica Rigorosa (Prevenção de Data Leakage)
% A divisão é feita rigorosamente ao início para isolar os dados de Teste.
% Guardamos os últimos 10% da série temporal no "cofre".
total_samples = length(targets);
split_idx = floor(0.90 * total_samples);

inputs_tv = inputs(:, 1:split_idx);
targets_tv = targets(1:split_idx);

inputs_test = inputs(:, split_idx+1:end);
targets_test = targets(split_idx+1:end);

fprintf('=============================================\n');
fprintf('         PREPARAÇÃO DOS DADOS (SPLIT)        \n');
fprintf('=============================================\n');
fprintf('Treino/Validação (Optimização): %d amostras\n', split_idx);
fprintf('Teste Final (Cofre isolado):    %d amostras\n\n', total_samples - split_idx);

%% 3. Configuração de Hardware
fprintf('Algoritmo de treino definido para Levenberg-Marquardt (trainlm).\n\n');
use_gpu_flag = 'no';
train_algo = 'trainlm';

%% 4. Definição do Espaço de Otimização
% Parâmetros: 1) Número de Neurónios, 2) Função de Ativação

% Teorema do Limite Central dita N >= 30 para assumir normalidade
num_runs = 30; 

% Construção Automática da Grelha (Grid Search)
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
num_points = length(opt_space); % 4 x 2 = 8 Pontos

mean_mse_val = zeros(num_points, 1);
var_mse_val = zeros(num_points, 1);

fprintf('=============================================\n');
fprintf('    OTIMIZAÇÃO DE REDE NEURONAL (30 RUNS)    \n');
fprintf('=============================================\n');
fprintf('Total de redes a treinar: %d treinos\n\n', num_points * num_runs);

%% 5. Ciclo de Otimização
for p = 1:num_points
    n_neurons = opt_space(p).neurons;
    act_func = opt_space(p).activation;
    
    fprintf('A avaliar Ponto %d/%d [Neurónios: %2d | Ativação: %6s]:\n', p, num_points, n_neurons, act_func);
    
    val_mse_runs = zeros(num_runs, 1);
    
    for r = 1:num_runs
        fprintf('  -> Treinando modelo %2d/%d do Ponto %d ... ', r, num_runs, p);
        
        % Configurar a rede feedforward com o algoritmo compatível
        net = fitnet(n_neurons, train_algo);
        net.layers{1}.transferFcn = act_func;
        
        % Fixar a paciência num valor estável por defeito para todas as redes
        net.trainParam.max_fail = 10;
        
        % ESCONDER A JANELA DE TREINO (Muito importante para loops longos!)
        net.trainParam.showWindow = false;
        
        % Treinar com MSE para penalizar picos de erro grandes
        net.performFcn = 'mse';
        
        % Configuração para Séries Temporais (Divisão Sequencial em vez de Aleatória)
        net.divideFcn = 'divideblock';
        
        % Divisão feita APENAS nos dados de TV (O Teste real é 0 aqui!)
        net.divideParam.trainRatio = 0.85; 
        net.divideParam.valRatio   = 0.15;
        net.divideParam.testRatio  = 0.00; 
        
        % Treinar a rede
        [net, tr] = train(net, inputs_tv, targets_tv, 'useGPU', use_gpu_flag);
        
        % Extrair a métrica de Validação para avaliar quão boa foi esta execução
        val_mse_runs(r) = tr.best_vperf;
        
        fprintf('Concluído! (Val MSE: %.4f)\n', val_mse_runs(r));
    end
    
    mean_mse_val(p) = mean(val_mse_runs);
    var_mse_val(p) = var(val_mse_runs);
    
    fprintf('=> RESUMO PONTO %d | Média Val MSE: %.4f | Variância: %.6f\n\n', p, mean_mse_val(p), var_mse_val(p));
end

%% 6. Tabela de Resultados Finais
fprintf('\n=============================================\n');
fprintf('         RESULTADOS DA OTIMIZAÇÃO            \n');
fprintf('=============================================\n');

neur_col = [opt_space.neurons]';
act_col = string({opt_space.activation}');

T = table(neur_col, act_col, mean_mse_val, var_mse_val, ...
    'VariableNames', {'Neuronios', 'Ativacao', 'Media_Val_MSE', 'Variancia_Val_MSE'});
disp(T);

% Gerar Gráfico de Barras com Error Bars para o Relatório
fprintf('A gerar gráfico de comparação (Error Bar Chart)...\n');
config_labels = strcat(string(neur_col), " neur. (", act_col, ")");

figure('Name', 'Optimization Performance', 'NumberTitle', 'off', 'Position', [200, 200, 900, 500]);
% Adicionar Error Bars com os pontos da média (sem as barras de histograma)
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

%% 7. Fim da Exploração e Otimização
fprintf('\n=============================================\n');
fprintf('  O treino do modelo final foi separado para o script:\n');
fprintf('  "windspeed_final_model.m" para garantir reprodutibilidade\n');
fprintf('  e permitir o armazenamento dos pesos iniciais e finais.\n');
fprintf('=============================================\n');