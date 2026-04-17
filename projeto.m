% Projeto 1 de Redes Neuronais Artificiais
clc;
clearvars;

%% 1) Importação e Análise Inicial
data = readtable("Gold_data.csv");

fprintf('--- Exploração Inicial ---\n');
fprintf('Observações: %d | Features: %d\n', size(data, 1), size(data, 2));

disp('Valores em falta (NaN) por feature:');
disp(array2table(sum(ismissing(data)), 'VariableNames', data.Properties.VariableNames));

% Contagem de zeros
zeros_count = zeros(1, size(data, 2));
for i = 1:size(data, 2)
    col_data = data{:, i};
    if isnumeric(col_data)
        zeros_count(i) = sum(col_data == 0, 'omitnan');
    else
        % Contabilizar de forma segura mesmo em colunas lidas como string/texto
        num_data = str2double(string(col_data));
        zeros_count(i) = sum(num_data == 0, 'omitnan');
    end
end
disp('Contagem de valores iguais a zero:');
disp(array2table(zeros_count, 'VariableNames', data.Properties.VariableNames));

disp('Sumário estatístico inicial:');
summary(data)

%% 2) Data Cleaning
if ismember('Date', data.Properties.VariableNames)
    data = removevars(data, 'Date');
end

% 2.1) Remoção de cabeçalhos repetidos e padronização para numérico
todas_vars = data.Properties.VariableNames;
for i = 1:length(todas_vars)
    col = todas_vars{i};
    if ~isnumeric(data.(col))
        str_data = string(data.(col));
        str_data(ismissing(str_data)) = "";
        
        idx_nao_num = isnan(str2double(str_data));
        idx_vazio = (strtrim(str_data) == "") | strcmpi(strtrim(str_data), "nan") | strcmpi(strtrim(str_data), "na");
        idx_cabecalho = idx_nao_num & ~idx_vazio;
        
        if any(idx_cabecalho)
            data(idx_cabecalho, :) = [];
            str_data = string(data.(col));
        end
        data.(col) = str2double(str_data);
    end
end



% 2.2) Tratamento de valores inválidos (Negativos em geral, Zeros no Volume)
for i = 1:length(todas_vars)
    col = todas_vars{i};
    if ~isnumeric(data.(col)), continue; end
    
    temp_col = data.(col);
    temp_col(temp_col < 0) = NaN;
    if strcmp(col, 'Volume'), temp_col(temp_col == 0) = NaN; end
    data.(col) = temp_col;
end

% 2.3) Imputação de Valores Ausentes (NaNs)
numericVars = data.Properties.VariableNames(varfun(@isnumeric, data, 'OutputFormat', 'uniform'));
data = fillmissing(data, 'linear', 'DataVariables', numericVars, 'EndValues', 'nearest');

fprintf('--- Limpeza Concluída ---\n');
summary(data)

%% 3) Exploratory Data Analysis (EDA)
vars_no_volume = data.Properties.VariableNames(~strcmp(data.Properties.VariableNames, 'Volume'));

% Histograma das Features Lineares
figure;
for i = 1:length(vars_no_volume)
    subplot(ceil(length(vars_no_volume)/3), 3, i);
    histogram(data{:, vars_no_volume{i}});
    title(['Distribuição de ' vars_no_volume{i}]); grid on;
end
sgtitle('Histogramas de Frequência das Features (Escala Linear)');

% Histograma Volume
figure;
histogram(data.Volume);
set(gca, 'YScale', 'log'); 
title('Frequência do Volume Negociado'); ylabel('Frequência (Log)'); grid on;

% Dispersão vs Close
figure;
scatter_vars = {'Open', 'High', 'Low', 'Volume', 'Sales', 'weight'};
for i = 1:length(scatter_vars)
    subplot(2, 3, i); scatter(data.(scatter_vars{i}), data.Close); title(['Relação ' scatter_vars{i} ' e Close']);
end
sgtitle('Dispersão das Features Originais Face ao Preço de Fecho');

% Boxplots para Outliers
figure;
subplot(1, 4, 1); boxplot(table2array(data(:, {'Open', 'High', 'Low', 'Close'}))); title('Prices'); xticklabels({'Open', 'High', 'Low', 'Close'});
subplot(1, 4, 2); boxplot(data.Sales); title('Sales'); xticklabels({'Sales'});
subplot(1, 4, 3); boxplot(data.weight); title('Weight'); xticklabels({'Weight'});
subplot(1, 4, 4); boxplot(data.Volume); set(gca, 'YScale', 'log'); title('Volume (Escala Logarítmica)'); xticklabels({'Volume'});
sgtitle('Boxplots: Verificação de Outliers');

% Correlação
figure; heatmap(data.Properties.VariableNames, data.Properties.VariableNames, corr(table2array(data))); title('Matriz de Correlação Global');

%% 4) Feature Engineering
data.HL_Range = data.High - data.Low;
data.Daily_Mean = (data.Open + data.High + data.Low) / 3;
data.Sales_per_Weight = data.Sales ./ (data.weight + 1e-6);

figure;
subplot(1,3,1); 
hl_vals = data.HL_Range(data.HL_Range > 0);
edges = logspace(log10(min(hl_vals)), log10(max(hl_vals)), 40);
histogram(hl_vals, edges);
set(gca, 'XScale', 'log', 'YScale', 'log'); 
title('Amplitude Diária de Preços (Escala Log-Log)'); grid on;

subplot(1,3,2); histogram(data.Daily_Mean); title('Preço Médio Diário'); grid on;
subplot(1,3,3); histogram(data.Sales_per_Weight); title('Densidade Vendas/Peso'); grid on;

%% 5) Feature Transformation
data_before = data;

vars_to_log = {'Open', 'High', 'Low', 'Close', 'weight', 'Volume', 'HL_Range', 'Daily_Mean', 'Sales_per_Weight'};

% Transformação Logarítmica Isolada
for i = 1:length(vars_to_log)
    if ismember(vars_to_log{i}, data.Properties.VariableNames)
        data.(vars_to_log{i}) = real(log1p(max(0, data.(vars_to_log{i}))));
    end
end

% Split Treino/Teste
rng(42);
cv = cvpartition(size(data, 1), 'HoldOut', 0.2);
train_data = data(training(cv), :);
test_data = data(test(cv), :);

% Standardisation (Z-Score)
input_features = train_data.Properties.VariableNames(~strcmp(train_data.Properties.VariableNames, 'Close'));
[train_data{:, input_features}, mu_input, sigma_input] = zscore(table2array(train_data(:, input_features)));
test_data{:, input_features} = (table2array(test_data(:, input_features)) - mu_input) ./ sigma_input;

[train_data.Close, mu_close, sigma_close] = zscore(train_data.Close);
test_data.Close = (test_data.Close - mu_close) ./ sigma_close;

% Gráficos Before/After Transformação: 1) Preços puros
vars_prices = {'Open', 'High', 'Low', 'Close'};
figure('Units', 'normalized', 'Position', [0.1 0.1 0.6 0.6]);
for i = 1:length(vars_prices)
    var = vars_prices{i};
    subplot(2, 4, (i-1)*2 + 1);
    histogram(data_before.(var)(~isnan(data_before.(var))));
    title([var ' (Orig)'], 'Interpreter', 'none');
    
    subplot(2, 4, (i-1)*2 + 2);
    histogram(data.(var)(~isnan(data.(var))));
    title([var ' (Log)'], 'Interpreter', 'none');
end
sgtitle('Impacto da Transformação Logarítmica - Variáveis de Preço');

% Gráficos Before/After Transformação: 2) Restantes Features
vars_other = {'weight', 'Volume', 'HL_Range', 'Daily_Mean', 'Sales_per_Weight'};
figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
for i = 1:length(vars_other)
    var = vars_other{i};
    subplot(3, 4, (i-1)*2 + 1);
    histogram(data_before.(var)(~isnan(data_before.(var))));
    if strcmp(var, 'Volume') || strcmp(var, 'HL_Range'), set(gca, 'YScale', 'log'); end
    title([var ' (Orig)'], 'Interpreter', 'none');
    
    subplot(3, 4, (i-1)*2 + 2);
    histogram(data.(var)(~isnan(data.(var))));
    title([var ' (Log)'], 'Interpreter', 'none');
end
sgtitle('Impacto da Transformação Logarítmica - Restantes Features');

%% 6) Feature Selection
selected_features = {'Open', 'Volume', 'Sales', 'weight', 'HL_Range', 'Sales_per_Weight'};
fprintf('\n--- Feature Selection ---\nFeatures Retidas: '); disp(selected_features);

figure; heatmap([selected_features, {'Close'}], [selected_features, {'Close'}], corr(table2array(train_data(:, [selected_features, {'Close'}]))));

%% 7) Análise Comparativa e Justificação de Seleção
% Avaliação da utilidade estatística de todas as features para a Rede
orig_vars = {'Open', 'High', 'Low', 'Volume', 'Sales', 'weight'};
corr_orig = abs(corr(table2array(train_data(:, orig_vars)), train_data.Close));

eng_vars = {'HL_Range', 'Daily_Mean', 'Sales_per_Weight'};
corr_eng = abs(corr(table2array(train_data(:, eng_vars)), train_data.Close));

% Agrupamento dos dados
all_corr_vals = [corr_orig; corr_eng];
all_corr_names = [orig_vars, eng_vars];
is_selected_logic = ismember(all_corr_names, selected_features);

figure;
hold on;
b1 = bar(find(is_selected_logic), all_corr_vals(is_selected_logic), 'FaceColor', [0 0.4470 0.7410]);
b2 = bar(find(~is_selected_logic), all_corr_vals(~is_selected_logic), 'FaceColor', [0.8500 0.3250 0.0980]);

xlim([0 length(all_corr_names)+1]);
set(gca, 'xtick', 1:length(all_corr_names), 'xticklabel', all_corr_names, 'XTickLabelRotation', 45);
ylabel('Correlação Absoluta (Afinidade c/ o Preço Close)'); 
title('Comparativo de Feature Selection: Selecionadas vs Descartadas');
legend([b1(1), b2(1)], {'Retidas para a Rede', 'Descartadas (Alta Multicolinearidade)'}, 'Location', 'northeast');
grid on; hold off;

%% 8) PCA Final
num_data_final = table2array(train_data(:, selected_features));
[coeff_final, score_final, ~, ~, explained_final] = pca(num_data_final);

figure;
subplot(1, 2, 1);
pareto(max(0, explained_final)); title('Scree Plot (PCA)'); xlabel('Componente'); ylabel('Variância (%)');

subplot(1, 2, 2);
scatter(score_final(:,1), score_final(:,2), 15, train_data.Close, 'filled');
colorbar; title('Projeção PCA Final'); xlabel(sprintf('PC1 (%.1f%%)', explained_final(1))); ylabel(sprintf('PC2 (%.1f%%)', explained_final(2))); grid on;

figure; biplot(coeff_final(:,1:2), 'Scores', score_final(:,1:2), 'VarLabels', selected_features); title('Biplot PCA (Projeção das Variáveis Retidas)');

%% 9) Distribuições Finais e Exportação Física
final_cols = [selected_features, {'Close'}];

% Gráfico que prova o estado final (já tratado e normalizado) dos dados que seguem para Modelo
figure('Units', 'normalized', 'Position', [0.15 0.15 0.8 0.8]);
for i = 1:length(final_cols)
    subplot(2, ceil(length(final_cols)/2), i);
    histogram(train_data.(final_cols{i})(~isnan(train_data.(final_cols{i}))));
    title([final_cols{i} ' (Final Z-Score)'], 'Interpreter', 'none');
    grid on;
end
sgtitle('Distribuição Final: Features de Input + Target (Prontas para a Rede)');

% Gravar as bases de dados isolando ativamente Colunas Relevantes
writetable(train_data(:, final_cols), 'Gold_TrainData_Processed.csv');
writetable(test_data(:, final_cols), 'Gold_TestData_Processed.csv');

fprintf('\n----------------------------------------\n');
disp('Guardados ficheiros físicos (Filtro Aplicado: APENAS features selecionadas transferidas).');
disp('-> Gold_TrainData_Processed.csv');
disp('-> Gold_TestData_Processed.csv');
disp('A Base de Dados está limpa, isolada e à prova de erro para introduzir de imediato na Rede Neuronal!');
