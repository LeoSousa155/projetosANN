%% PROJETO 2: Redes Neuronais - Apparent Temperature
% Script: Treino do Modelo Final (Separação para Reprodutibilidade)

clc; clearvars; close all;

%% 1 - Importação e Limpeza de Dados
fprintf('A carregar e pré-processar os dados...\n');
data = readtable('weatherHistory.csv');

% Remover a coluna problemática (Loud Cover)
data.LoudCover = [];

% Tratar a Pressão (Imputação em vez de Remoção)
data.Pressure_millibars_(data.Pressure_millibars_ < 900) = NaN;
data.Pressure_millibars_ = fillmissing(data.Pressure_millibars_, 'constant', median(data.Pressure_millibars_, 'omitnan'));

% Tratar a Humidade (se houver zeros isolados)
data.Humidity(data.Humidity <= 0) = NaN;
data.Humidity = fillmissing(data.Humidity, 'constant', median(data.Humidity, 'omitnan'));

% Verificação Final: Remover NaNs residuais
data = rmmissing(data);

%% 2 - Feature Engineering
% Decomposição do Wind Bearing para Cosseno e Seno
data.Wind_Sin = sin(deg2rad(data.WindBearing_degrees_));
data.Wind_Cos = cos(deg2rad(data.WindBearing_degrees_));
data.WindBearing_degrees_ = [];

% Heat Index Proxy (Temperatura X Humidade)
data.Heat_Index_Proxy = data.Temperature_C_ .* data.Humidity;

% Wind Chill Effect (Vento X Temperatura)
data.Wind_Chill_Effect = data.WindSpeed_km_h_ .* data.Temperature_C_;

% Interação Humidade e Pressão (Densidade/Estabilidade do ar)
data.Hum_Press_Ratio = data.Humidity .* data.Pressure_millibars_;

% Remover Wind_Sin e Wind_Cos porque a correlação é nula
data_final = removevars(data, {'Wind_Sin', 'Wind_Cos'});

%% 3 - Separação de Features e Target
targetName = 'ApparentTemperature_C_';
inputNames = setdiff(data_final.Properties.VariableNames, targetName, 'stable');

X_raw = data_final{:, inputNames}';
y_raw = data_final{:, targetName}';

%% 4 - Partição Aleatória Rigorosa (90% Treino/Validação | 10% Teste)
total_samples = length(y_raw);
split_idx = floor(0.90 * total_samples);

% Baralhar os índices para garantir amostragem representativa
rng(42); % Fixar semente para reprodutibilidade estrita
shuffled_idx = randperm(total_samples);

inputs_tv = X_raw(:, shuffled_idx(1:split_idx));
targets_tv = y_raw(shuffled_idx(1:split_idx));

inputs_test = X_raw(:, shuffled_idx(split_idx+1:end));
targets_test = y_raw(shuffled_idx(split_idx+1:end));

%% 5 - Configuração da Rede Final (20 Neurónios, 'poslin')
neurons = 20;
activation = 'poslin';
train_algo = 'trainlm';
use_gpu_flag = 'no';

fprintf('A configurar o modelo final com %d neurónios e ativação %s...\n', neurons, activation);
net = fitnet(neurons, train_algo);
net.layers{1}.transferFcn = activation;
net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};
net.trainParam.max_fail = 10;
net.performFcn = 'mse';

% Usar divideind para fixar a divisão de treino/validação dentro da rede
% Isto garante que ao correr novamente o treino, as mesmas amostras vão para treino e validação
net.divideFcn = 'divideind';
num_tv = length(targets_tv);
num_train = floor(0.85 * num_tv);

rng(42); % Semente para a divisão interna de TV
tv_indices = randperm(num_tv);

net.divideParam.trainInd = tv_indices(1:num_train);
net.divideParam.valInd   = tv_indices(num_train+1:end);
net.divideParam.testInd  = [];

%% 6 - Carregar ou Guardar Pesos Iniciais
initial_weights_file = 'weatherStation_initial_weights.mat';

if isfile(initial_weights_file)
    fprintf('A carregar os pesos iniciais fornecidos no ficheiro "%s"...\n', initial_weights_file);
    load(initial_weights_file, 'net');
else
    fprintf('A inicializar novos pesos para a rede...\n');
    rng(42); % Semente para a inicialização dos pesos fixos
    net = configure(net, inputs_tv, targets_tv);
    save(initial_weights_file, 'net');
    fprintf('Pesos iniciais guardados em "%s".\n', initial_weights_file);
end

%% 7 - Treino do Modelo
net.trainParam.showWindow = true; % ATIVAR JANELA

fprintf('A iniciar o treino do modelo...\n');
[net, tr] = train(net, inputs_tv, targets_tv, 'useGPU', use_gpu_flag);

%% 8 - Guardar Pesos Finais
final_weights_file = 'weatherStation_final_weights.mat';
save(final_weights_file, 'net');
fprintf('Pesos finais pós-treino guardados em "%s"\n', final_weights_file);

%% 9 - Avaliação Cega no Cofre (Test Set com MAE)
final_preds = net(inputs_test);
final_mae = mae(targets_test - final_preds);

fprintf('\n=============================================\n');
fprintf('         AVALIAÇÃO FINAL (TEST SET)          \n');
fprintf('=============================================\n');
fprintf('=> MAE (Mean Absolute Error) Cego: %.4f\n', final_mae);
fprintf('=============================================\n\n');

%% 10 - Guardar Figuras do Treino
fprintf('A gerar e guardar as figuras dos resultados...\n');
fig_dir = 'figures_weatherStation';
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
