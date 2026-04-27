%% Projeto 2: Redes Neuronais Artificiais - Wind Speed
% Script: Carregamento, Limpeza, Análise de Gaps e Segmentação por Blocos
clc; clearvars; close all;

%% 1. Importação e Exploração Inicial (Para Relatório)
path = "windSpeed.csv";

% Verificar se o ficheiro existe antes de carregar
if ~isfile(path)
    error('Ficheiro não encontrado no caminho: %s', path);
end

% 'preserve' mantém os nomes originais; 'detectImportOptions' ajuda na robustez
opts = detectImportOptions(path);
opts.VariableNamingRule = 'preserve';
data = readtable(path, opts);

% Atribuir nomes amigáveis para manipulação interna
data.Properties.VariableNames{1} = 'RawTimestamp';
data.Properties.VariableNames{2} = 'WindSpeed';

fprintf('=============================================\n');
fprintf('         1. EXPLORAÇÃO E TRATAMENTO         \n');
fprintf('=============================================\n');
fprintf('Observações: %d | Variáveis Originais: %d\n', size(data,1), size(data,2));
fprintf('---------------------------------------------\n');

%% 2. Processamento de Strings e Separação de Variáveis
fprintf('Processando Timestamps e Direção do Vento...\n');

% 1. Converter para string e remover aspas ou espaços nas extremidades
rawStrings = strtrim(string(data.RawTimestamp));
rawStrings = strrep(rawStrings, '"', ''); 

% 2. Extração da Data (os primeiros 19 caracteres: yyyy-mm-dd hh:mm:ss)
datesStr = extractBefore(rawStrings, 20);

% 3. Extração da Direção (procurar a última sequência de dígitos na string)
dirMatches = regexp(rawStrings, '\D+(\d+)$', 'tokens', 'once');

% Inicializar vetor de direções
directions = NaN(size(rawStrings));

% Converter células extraídas para números
validDir = ~cellfun(@isempty, dirMatches);
if any(validDir)
    tmp_dir = cellfun(@(x) str2double(x{1}), dirMatches(validDir));
    directions(validDir) = tmp_dir;
end

% 4. Conversão final e limpeza da tabela
data.Time = datetime(datesStr, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
data.WindDirection = directions;
data.RawTimestamp = []; % Remover coluna bruta

% Reordenar para melhor visualização
data = data(:, {'Time', 'WindSpeed', 'WindDirection'});

%% 2.5 Verificação de Registos Duplicados
fprintf('\n--- Verificação de Registos Duplicados ---\n');
[~, uniqueIdx] = unique(data.Time, 'first');
num_duplicados = height(data) - length(uniqueIdx);
if num_duplicados > 0
    data = data(uniqueIdx, :);
    fprintf('Removidas %d observações duplicadas (mesmo timestamp).\n', num_duplicados);
else
    fprintf('Nenhuma observação duplicada encontrada.\n');
end

%% 3. Tratamento de Valores em Falta (NaN)
fprintf('\n--- Verificação de Qualidade Pós-Separação ---\n');

nanSpeed = sum(isnan(data.WindSpeed));
nanDir = sum(isnan(data.WindDirection));

fprintf('Valores em falta na Velocidade: %d\n', nanSpeed);
fprintf('Valores em falta na Direção:   %d\n', nanDir);

% Preenchimento com Interpolação Linear
if nanSpeed > 0
    data.WindSpeed = fillmissing(data.WindSpeed, 'linear');
end
if nanDir > 0
    data.WindDirection = fillmissing(data.WindDirection, 'linear');
end
fprintf('---------------------------------------------\n');

%% 4. Segmentação em Blocos (Tratamento de Sequências Disconexas)
% Definimos que qualquer gap > 70 segundos quebra a sequência lógica
thresholdRuptura = 70; 
data.TimeDiff = [0; seconds(diff(data.Time))]; 

% Criar ID de Bloco: aumenta sempre que há uma ruptura
data.BlockID = cumsum(data.TimeDiff > thresholdRuptura) + 1;

numBlocos = max(data.BlockID);
fprintf('\n=============================================\n');
fprintf('      2. SEGMENTAÇÃO POR BLOCOS (GAPS)      \n');
fprintf('=============================================\n');
fprintf('Threshold de Ruptura: %d segundos\n', thresholdRuptura);
fprintf('Número de blocos independentes encontrados: %d\n', numBlocos);
fprintf('---------------------------------------------\n');

%% 5. Feature Engineering (Decomposição Vetorial Cíclica)
fprintf('\n=============================================\n');
fprintf('          3. FEATURE ENGINEERING             \n');
fprintf('=============================================\n');
fprintf('Decompondo a Direção e Velocidade em vetores contínuos (U e V)...\n');

% Converter graus para radianos
theta_rad = deg2rad(data.WindDirection);

% 1. Decomposição da Direção (Bússola)
data.Dir_Sin = sin(theta_rad);
data.Dir_Cos = cos(theta_rad);

% 2. Ciclo Diurno (Hora do Dia Cíclica)
% O vento é movido a aquecimento solar. Um relógio de 24h cíclico ajuda o modelo a distinguir o dia da noite.
horas_decimais = hour(data.Time) + minute(data.Time)/60 + second(data.Time)/3600;
data.Hour_Sin = sin(2 * pi * horas_decimais / 24);
data.Hour_Cos = cos(2 * pi * horas_decimais / 24);

% 3. Momento / Aceleração do Vento (Derivada)
% Indica se o vento está num pico ascendente ou a perder força no instante t
data.Wind_Acc = [0; diff(data.WindSpeed)];
% Anular a aceleração no primeiro instante de cada novo bloco (pois ocorreu um gap temporal antes)
idx_block_starts = [1; find(diff(data.BlockID) > 0) + 1];
data.Wind_Acc(idx_block_starts) = 0;

fprintf('Features Dir_Sin e Dir_Cos criadas com sucesso para evitar saltos de 359º -> 1º!\n');
fprintf('Features Hour_Sin, Hour_Cos e Wind_Acc (Aceleração) adicionadas com sucesso!\n');
fprintf('---------------------------------------------\n');

%% 6. Criação da Variável Alvo (Target) - 1 Minuto no Futuro
fprintf('\n=============================================\n');
fprintf('       4. CRIAÇÃO DA VARIÁVEL ALVO (TARGET)  \n');
fprintf('=============================================\n');
fprintf('Criando target para previsão de WindSpeed 60 segundos no futuro...\n');

% OTIMIZAÇÃO: Usar arrays nativos em vez de aceder à tabela linha a linha
time_numeric = cumsum(data.TimeDiff);
speeds = data.WindSpeed;
targets = NaN(size(speeds));
block_ids = data.BlockID;

% Pré-calcular índices de início e fim de cada bloco (Evita find() O(N) dentro do ciclo)
block_starts = [1; find(diff(block_ids) > 0) + 1];
block_ends = [block_starts(2:end) - 1; length(block_ids)];

for b = 1:length(block_starts)
    start_idx = block_starts(b);
    end_idx = block_ends(b);
    
    % Só faz sentido calcular se o bloco tiver tamanho suficiente (>1 elemento)
    if end_idx > start_idx
        j = start_idx;
        for i = start_idx:end_idx
            % Avançar o ponteiro j usando matemática rápida (doubles)
            while j <= end_idx && (time_numeric(j) - time_numeric(i)) < 60
                j = j + 1;
            end
            
            % Se encontrou uma amostra no futuro que obedeça à regra
            if j <= end_idx
                targets(i) = speeds(j);
            end
        end
    end
end

% Alocar os resultados finais de volta à tabela de uma só vez
data.Target_WindSpeed = targets;

fprintf('Cálculo da feature Target_WindSpeed concluído! Iniciando limpeza e correlações...\n');

% Limpeza das linhas sem Target (finais de blocos sem futuro conhecido)
linhas_antes = height(data);
data = data(~isnan(data.Target_WindSpeed), :);
linhas_depois = height(data);

fprintf('Target_WindSpeed gerado com sucesso usando pesquisa look-ahead.\n');
fprintf('Linhas removidas (sem 1 min de futuro conhecido): %d\n', linhas_antes - linhas_depois);
fprintf('Dataset final pronto com %d observações.\n', linhas_depois);
fprintf('---------------------------------------------\n');

%% 7. Resumo dos Blocos e Exportação
% Identificar os maiores blocos (mais úteis para treino)
stats = groupsummary(data, 'BlockID');
stats = sortrows(stats, 'GroupCount', 'descend');

fprintf('\nResumo dos maiores blocos (Top 5):\n');
disp(head(stats, 5));

fprintf('\nConclusão para o modelo:\n');
fprintf('-> O gráfico de timeline identifica visualmente as rupturas.\n');
fprintf('-> Para treinar a RNA, use janelas de dados dentro do mesmo BlocoID.\n');
fprintf('=============================================\n');

% Drop das features originais redundantes (apenas no final)
cols_to_keep = {'Time', 'WindSpeed', 'Dir_Sin', 'Dir_Cos', 'Hour_Sin', 'Hour_Cos', 'Wind_Acc', 'Target_WindSpeed', 'BlockID'};
data_final = data(:, cols_to_keep);

% Mostrar o resultado final
disp('Primeiras 5 linhas do dataset final pronto para modelo:');
disp(head(data_final, 5));

% Exportar os dados para CSV
writetable(data, 'windSpeed_Processed.csv');
writetable(data_final, 'windSpeed_ModelReady.csv');
fprintf('\nDatasets exportados com sucesso!\n');
fprintf('-> windSpeed_Processed.csv (Com features originais para EDA)\n');
fprintf('-> windSpeed_ModelReady.csv (Apenas features compactadas para o Modelo)\n');