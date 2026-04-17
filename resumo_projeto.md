# Resumo do Processamento de Dados - Projeto 1 de Redes Neuronais Artificiais

Este documento detalha o *pipeline* de preparação e limpeza de dados desenvolvido para o nosso modelo de Redes Neuronais Artificiais (ANN). O objetivo principal de todas estas fases é garantir que os dados alimentados à rede estão livres de ruído, na escala correta e sem redundâncias prejudiciais, maximizando assim a performance na aprendizagem.

## 1. Importação e Limpeza Inicial (Data Cleaning)
* **Remoção de Colunas Inúteis:** Foi removida a coluna da data de forma a lidar puramente com valores numéricos temporais/quantitativos.
* **Limpeza de Erros e Variáveis Negativas:** O código percorre os dados para identificar cabeçalhos repetidos e valores negativos impossíveis numa série temporal de ações (e.g. Volume = 0 ou Preços < 0), substituindo-os por `NaN`.
* **Imputação de Valores Ausentes:** Usámos interpolação *linear* para preencher os `NaN`. Uma decisão importante foi usar a flag `EndValues = 'nearest'`, que obriga o MATLAB a manter as extremidades fixas, impedindo a geração matemática de valores negativos nas pontas se os dados estivessem com tendência decrescente (extrapolação perigosa).

## 2. Feature Engineering
Adicionámos algumas métricas calculadas que podem expor relações mais profundas não imediatamente óbvias nas colunas isoladas:
* `HL_Range` (High - Low): Representa a volatilidade diária do preço.
* `Daily_Mean`: O preço médio no decorrer de um dia.
* `Sales_per_Weight`: Uma métrica de "densidade" de negociação diária.

## 3. Transformações de Variáveis (Log e Split)
* **Transformação Logarítmica (`log1p`):** Variáveis como `Volume`, `Sales` e os preços em geral têm distribuições muito assimétricas (com caudas longas à direita). A transformação logarítmica achata os grandes valores extremos (outliers naturais) e aproxima a distribuição de uma curva Normal.
* **Separação Treino/Teste (Train/Test Split):** Antes de aplicar qualquer método de normalização, o código divide os dados em 80% treino e 20% teste. **Esta é uma decisão crítica e obrigatória**. Se normalizarmos o *dataset* completo antes do split, estamos a "contaminar" o treino com informação do teste (*Data Leakage*). 

## 4. Normalização (Z-Score)
* Porquê o **Z-Score** em vez do Min-Max? Redes neuronais que usam ativações como *ReLu* ou tangentes hiperbólicas tendem a ser mais eficientes com inputs cuja média se centre no 0 e variância a 1.
* A média e a norma são calculadas estritamente e **apenas em relação aos dados de Treino**. Os dados de Teste são escalados a usar as métricas obtidas no Treino.

## 5. Feature Selection
Foram avaliadas as correlações com a nossa variável dependente/alvo (`Close`). 
* Foram apenas **Retidas:** `Open`, `Volume`, `Sales`, `weight`, `HL_Range`, e `Sales_per_Weight`.
* **Porquê descartar `High` e `Low`?** Uma vez que `Open`, `High`, `Low` e `Close` contêm uma elevada auto-correlação (quase aos pares na casa dos 99%), manter todos gera o problema de **Multicolinearidade**. Isto baralha os pesos da Rede Neuronal e fá-la demorar mais tempo a treinar para concluir o mesmo padrão. O `HL_Range` incorpora essas variações sem causar essa colinearidade. 

## 6. Principal Component Analysis (PCA)
O PCA foi incluído como análise final da dimensionalidade:
* **Porque está isolado após a Feature Selection?** Aplicar o PCA em lixo gera mais lixo. Retirámos previamente as features altamente correlacionadas para garantir que cada Componente Principal criada extrai variações únicas de mercado e não apenas duplicação da linha de preço de fecho.
* **Para que serve?** Através dos gráficos Biplot e de Projeção PCA gerados, expomos aos avaliadores que é possível explicar quase toda a variância e informação do *dataset* utilizando muito menos componentes, isolando os vetores da matriz matemática ortogonalmente.

Esta arquitetura final já exporta as matrizes prontas e compactas para os ficheiros `Gold_TrainData_Processed.csv` e `Gold_TestData_Processed.csv`.
