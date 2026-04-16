% Projeto 1 de redes neuronais artificiais
clc
clearvars


%% 1) Import Data and Exploration
% a função head mostra as primeiras 8 linhas dos dados selecionados
    % para mudar a visualização da tabela usamos:
    % >> format short g (formato curto, mais comum)
    % >> format long g  (formato longo com mais precisão decimal)

% Carregar o dataset e visualização inicial dos dados
data = readtable("Gold_data.csv");

% Estatísticas descritivas iniciais (Requisito 1 do Enunciado)
numObs = size(data, 1);
numVars = size(data, 2);
missingData = sum(ismissing(data)); % Contagem de NaNs por coluna

fprintf('--- Exploração Inicial ---\n');
fprintf('Número de Observações: %d\n', numObs);
fprintf('Número de Variáveis: %d\n', numVars);
fprintf('Valores em falta por feature:\n');
disp(array2table(missingData, 'VariableNames', data.Properties.VariableNames));

summary(data)
disp(head(data))

% Neste projeto o objetivo é conseguir prever o valor do "Close" com base nas  restantes features



%% 2) Data Cleaning and Preprocessing

data = removevars(data, 'Date');  % remover a coluna de datas (n são valores numéricos)

summary(data)
zeros_volumes = sum(data.Volume == 0);   % nº de valores 0 na feature Volume

% Imputação pela mediana global 
% data.Sales = fillmissing(data.Sales, 'constant', median(data.Sales, 'omitnan'));
% A distribuição fica uniforme com um pico na mediana, a linear dá uma
% distribuição mais normal

% Limpeza de valores negativos anómalos (ex: outlier de 'weight' < 0)
% Preços, Volumes e Pesos não podem ser negativos. 
% Vamos converter em NaN para serem preenchidos pela imputação.
numericVars = data.Properties.VariableNames(varfun(@isnumeric, data, 'OutputFormat', 'uniform'));
for i = 1:length(numericVars)
    col = numericVars{i};
    numNeg = sum(data.(col) < 0);
    if numNeg > 0
        fprintf('Aviso: Detetados %d valores negativos na coluna %s. Convertendo em NaN.\n', numNeg, col);
        data.(col)(data.(col) < 0) = NaN;
    end
end

% 1. Tratar Volume zero como dado em falta (NaN)
if ismember('Volume', data.Properties.VariableNames)
    data.Volume(data.Volume == 0) = NaN;
end

% 2. Imputação robusta para todas as colunas numéricas
% Usamos 'linear' para continuidade e 'nearest' para tratar NaNs nas extremidades (ex: linha 1)
for i = 1:length(numericVars)
    col = numericVars{i};
    data.(col) = fillmissing(data.(col), 'linear');
    data.(col) = fillmissing(data.(col), 'nearest');
end

fprintf('--- Limpeza de Dados Concluída ---\n');
summary(data)



%% 3) Exploratory Data Analysis (PCA)
% Visualização dos histogramas de todas as features (EDA)
figure
vars = data.Properties.VariableNames;
nVars = length(vars);
% Criar uma grelha dinâmica baseada no número de variáveis
nRows = ceil(nVars/3);
for i = 1:nVars
    subplot(nRows, 3, i);
    histogram(data{:, vars{i}});
    title(['Distribuição de ', vars{i}]);
    grid on;
end
sgtitle('Histogramas de todas as Features Iniciais');

% gráficos de dispersão das features vs Close
figure
subplot(2, 3, 1); scatter(data.Open, data.Close); title('Open vs Close');
subplot(2, 3, 2); scatter(data.High, data.Close); title('Hight vs Close');
subplot(2, 3, 3); scatter(data.Low, data.Close); title('Low vs Close');
subplot(2, 3, 4); scatter(data.Volume, data.Close); title('Volume vs Close');
subplot(2, 3, 5); scatter(data.Sales, data.Close); title('Sales vs Close');
subplot(2, 3, 6); scatter(data.weight, data.Close); title('Weight vs Close');


% Gráficos de caixa e bigodes das features para analisar outliers e
% dispersão
figure;
% Agrupamos as variáveis de preço para ver a escala comum
subplot(1, 2, 1);
boxplot(table2array(data(:, {'Open', 'High', 'Low', 'Close'})));
xticklabels({'Open', 'High', 'Low', 'Close'});
title('Distribuição de Preços (Outliers)');
% Boxplot para Sales e Weight (escalas similares)
subplot(1, 4, 3);
boxplot(table2array(data(:, {'Sales', 'weight'})));
xticklabels({'Sales', 'weight'});
title('Sales e Weight');

% Boxplot para Volume (escala muito superior)
subplot(1, 4, 4);
boxplot(data.Volume);
xticklabels({'Volume'});
title('Volume');


% Matriz de correlação entre as features
corr_matrix = corr(table2array(data));
figure;
heatmap(data.Properties.VariableNames, data.Properties.VariableNames, corr_matrix);
title('Matriz de Correlação');

% Executamos o PCA aqui para ver a estrutura inicial dos dados e identificar outliers.
% Nota: Nesta fase, como os dados ainda não foram normalizados, as variáveis com 
% escalas maiores (ex: Volume) tendem a dominar as componentes principais.

% Selecionar apenas as colunas numéricas disponíveis após a limpeza
currentNumericVars = data.Properties.VariableNames(varfun(@isnumeric, data, 'OutputFormat', 'uniform'));
numericData = table2array(data(:, currentNumericVars));
[coeff_init, score_init, ~, ~, explained_init, ~] = pca(numericData);

figure;
subplot(1, 2, 1);
pareto(explained_init);
xlabel('Componente Principal');
ylabel('Variância (%)');
title('Scree Plot Inicial');

subplot(1, 2, 2);
scatter(score_init(:,1), score_init(:,2), 15, data.Close, 'filled');
cb = colorbar;
cb.Label.String = 'Preço (Close)';
xlabel(['PC1 (', num2str(explained_init(1), '%.1f'), '%)']);
ylabel(['PC2 (', num2str(explained_init(2), '%.1f'), '%)']);
title('Projeção PCA Inicial (Dados Limpos)');
grid on;

figure;
biplot(coeff_init(:,1:2), 'Scores', score_init(:,1:2), 'VarLabels', currentNumericVars);
title('Biplot Inicial: Estrutura Natural das Features');



%% 4) Feature Engineering (Requisito 6)
% Criar novas features antes da normalização e seleção
% 1. HL_Range: Diferença entre o máximo e o mínimo do dia (Volatilidade)
data.HL_Range = data.High - data.Low;

% 2. Daily_Mean: Média aproximada do preço do dia
data.Daily_Mean = (data.Open + data.High + data.Low) / 3;

% 3. Sales_per_Weight: Relação entre vendas e peso
data.Sales_per_Weight = data.Sales ./ (data.weight + 1e-6);

fprintf('--- Feature Engineering Concluída ---\n');
summary(data(:, {'HL_Range', 'Daily_Mean', 'Sales_per_Weight'}))

% Visualização das novas features
figure;
subplot(1,3,1); histogram(data.HL_Range); title('Distribuição HL Range');
subplot(1,3,2); histogram(data.Daily_Mean); title('Distribuição Daily Mean');
subplot(1,3,3); histogram(data.Sales_per_Weight); title('Distribuição Sales per Weight');
sgtitle('Distribuição das Novas Features Engenheiradas');



%% 5) Feature Transformation Tecniques
% A transformação logarítmica é útil para lidar com dados "right-skewed" (assimetria à direita),
% aproximando a distribuição de uma Gaussiana e reduzindo o impacto de outliers.

% Aplicar log(1+x) para garantir estabilidade numérica (evitar log(0))
vars_to_log = {'Open', 'High', 'Low', 'Close', 'weight', 'Volume', 'HL_Range', 'Daily_Mean', 'Sales_per_Weight'};

% Criar uma cópia para comparação visual (antes vs depois)
data_before = data; 

for i = 1:length(vars_to_log)
    if ismember(vars_to_log{i}, data.Properties.VariableNames)
        % max(0,...) garante que x nunca é < 0, logo log1p(x) nunca é complexo
        data.(vars_to_log{i}) = real(log1p(max(0, data.(vars_to_log{i}))));
    end
end

% 4.2) Divisão em Treino e Teste (Hold-out 80/20)
% É crucial fazer o split antes da normalização para evitar Data Leakage
rng(42); % Seed para reprodutibilidade
cv = cvpartition(size(data, 1), 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest = test(cv);

train_data = data(idxTrain, :);
test_data = data(idxTest, :);

% 4.3) Normalização Z-score (Standardization)
% Calculamos os parâmetros (média e desvio padrão) APENAS no treino
input_features = data.Properties.VariableNames(~strcmp(data.Properties.VariableNames, 'Close'));

% Normalizar o Treino
[train_scaled, mu_input, sigma_input] = zscore(table2array(train_data(:, input_features)));
train_data{:, input_features} = train_scaled;

% Aplicar os parâmetros do Treino ao Teste
test_data{:, input_features} = (table2array(test_data(:, input_features)) - mu_input) ./ sigma_input;

% O target (Close) também deve ser normalizado se houver grande amplitude
[train_close_scaled, mu_close, sigma_close] = zscore(train_data.Close);
train_data.Close = train_close_scaled;
test_data.Close = (test_data.Close - mu_close) ./ sigma_close;

% Guardamos os parâmetros para possível desnormalização no futuro
norm_params.mu_input = mu_input;
norm_params.sigma_input = sigma_input;
norm_params.mu_close = mu_close;
norm_params.sigma_close = sigma_close;

% 4.3) Visualização Before-and-After
figure;
% Exemplo com a variável 'Close' (Target)
subplot(3, 2, 1); histogram(data_before.Close); title('Close (Original)');
subplot(3, 2, 2); histogram(data.Close); title('Close (Log Transformed)');

% Exemplo com a variável 'weight'
subplot(3, 2, 3); histogram(data_before.weight); title('Weight (Original)');
subplot(3, 2, 4); histogram(data.weight); title('Weight (Log Transformed)');

% Exemplo com a variável 'Volume'
subplot(3, 2, 5); histogram(data_before.Volume); title('Volume (Original)');
subplot(3, 2, 6); histogram(data.Volume); title('Volume (Log Transformed)');

sgtitle('Efeito da Transformação Logarítmica');

% Exibir sumário após transformações
disp('Sumário após Transformações e Normalização:');
summary(data)



%% 6) Feature Selection Methods
% No passo anterior vimos que Open, High, Low estão extremamente correlacionados (redundantes)
% Vamos manter apenas 'Open' e as novas features 'HL_Range' e 'Daily_Mean' para ver se 
% capturam a informação sem a redundância extrema.

% Lista de features candidatas (excluindo o target 'Close')
all_features = train_data.Properties.VariableNames(~strcmp(train_data.Properties.VariableNames, 'Close'));

% Seleção Manual baseada na correlação e lógica do negócio:
% Justificação: Open/High/Low têm corr > 0.999. Manter todos causa multicolinearidade.
% Escolhemos: Open (base), Volume, HL_Range (volatilidade), Sales_per_Weight.
selected_features = {'Open', 'Volume', 'Sales', 'weight', 'HL_Range', 'Sales_per_Weight'};

fprintf('--- Feature Selection ---\n');
fprintf('Features Originais: %d\n', length(numericVars)-1);
fprintf('Features Após Engineering e Seleção: %d\n', length(selected_features));
disp(selected_features);

% Matriz de correlação apenas das selecionadas
figure;
selected_corr = corr(table2array(train_data(:, [selected_features, {'Close'}])));
heatmap([selected_features, {'Close'}], [selected_features, {'Close'}], selected_corr);
fprintf('\n--- Análise Final Concluída ---\n');



%% 7) Comparative Analysis 
% Comparação estatística da utilidade (Correlação com o Target)
% Vamos comparar a correlação absoluta com 'Close' de diferentes sets

% 1. Original (apenas features básicas)
orig_vars = {'Open', 'High', 'Low', 'Volume', 'Sales', 'weight'};
corr_orig = abs(corr(table2array(train_data(:, orig_vars)), train_data.Close));

% 2. Engineered
eng_vars = {'HL_Range', 'Daily_Mean', 'Sales_per_Weight'};
corr_eng = abs(corr(table2array(train_data(:, eng_vars)), train_data.Close));

fprintf('\n--- Análise Comparativa (Correlação Absoluta com Close) ---\n');
fprintf('Média Corr Features Originais: %.4f\n', mean(corr_orig));
fprintf('Média Corr Features Engineered: %.4f\n', mean(corr_eng));

% Visualização da importância relativa (Correlação)
figure;
bar([corr_orig; corr_eng]);
set(gca, 'xticklabel', [orig_vars, eng_vars], 'XTickLabelRotation', 45);
ylabel('Correlação Absoluta com o Target (Close)');
title('Comparação da Utilidade das Features');
grid on;

disp('Processamento Completo. O dataset está pronto para modelação preditiva.');
