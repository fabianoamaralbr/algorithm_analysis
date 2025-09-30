[Home](https://github.com/fabianoamaralbr/algorithm_analysis/blob/main/README.md)

# Random Forest

A Floresta Aleatória (Random Forest) é um algoritmo de aprendizado de máquina versátil e poderoso, pertencente à categoria de métodos de ensemble learning. Ele opera construindo múltiplas árvores de decisão durante o treinamento e combinando as previsões dessas árvores individuais para alcançar um resultado mais preciso e robusto. O "aleatório" em seu nome vem de duas fontes principais de aleatoriedade: a amostragem dos dados de treinamento e a seleção de características para cada divisão nas árvores.

Este algoritmo é amplamente utilizado tanto para problemas de classificação (prever uma categoria) quanto de regressão (prever um valor contínuo), sendo uma escolha popular devido à sua alta precisão e capacidade de controlar o overfitting.

## Quando Usar o Random Forest

- **Alta Precisão Necessária:** <br>
    Quando a meta principal é obter alta precisão em tarefas de classificação ou regressão, o Random Forest frequentemente supera as árvores de decisão individuais e muitos outros algoritmos.

- **Robustez contra Overfitting:**<br>
    Sua natureza de ensemble e as aleatoriedades introduzidas o tornam menos propenso ao overfitting do que uma única árvore de decisão, mesmo sem a necessidade de poda explícita.

- **Dados Complexos e Não Lineares:**<br>
    Ele pode modelar relações não lineares e interações complexas entre características sem exigir transformações extensas nos dados.

- **Dados com Muitos Recursos:**<br>
    Lida bem com conjuntos de dados que possuem um grande número de características, pois a seleção aleatória de características em cada nó ajuda a gerenciar a dimensionalidade.

- **Identificação de Importância de Features:**<br>
    O Random Forest pode fornecer uma medida da importância de cada característica (feature importance), indicando quais variáveis são mais relevantes para a previsão.

- **Dados com Outliers ou Ruído:**<br>
    É relativamente robusto a outliers e ruídos nos dados devido à agregação de múltiplas árvores.

## Quando Não Usar o Random Forest

- **Interpretabilidade Extrema Requerida:**<br>
    Embora possa fornecer a importância das características, a interpretação da "caixa preta" de centenas de árvores de decisão é muito mais difícil do que a de uma única árvore ou um modelo linear.

- **Dados Muito Escassos (Sparse Data):**<br>
    Em alguns casos com dados muito esparsos, outros algoritmos (como SVMs lineares) podem ser mais eficientes.

- **Performance em Tempo Real Crítica:**<br>
    Para aplicações com requisitos de latência extremamente baixos para predições, o custo computacional de realizar previsões com uma floresta de centenas de árvores pode ser um fator limitante, embora seja geralmente rápido.

- **Dados de Alta Dimensionalidade Extrema:**<br>
    Embora lide bem com muitos recursos, em casos de dimensionalidade extremamente alta (onde o número de características é muito maior que o número de amostras), outros algoritmos como PCA seguido de um classificador/regressor simples podem ser mais eficazes.

## Como Funciona?

O Random Forest constrói uma "floresta" de árvores de decisão. A ideia central é que, ao combinar as previsões de várias árvores construídas de forma independente e com alguma aleatoriedade, o erro total pode ser significativamente reduzido em comparação com uma única árvore de decisão.

- **Bagging (Bootstrap Aggregating):**<br>
    - **Amostragem Aleatória com Reposição:**<br> Para construir cada árvore na floresta, o algoritmo seleciona aleatoriamente (com reposição) um subconjunto de amostras do conjunto de dados original. Isso significa que algumas amostras podem ser repetidas em um subconjunto, e outras podem não ser incluídas em nenhum subconjunto. Este processo cria um conjunto de dados ligeiramente diferente para cada árvore.
    - **Diversidade nos Dados:**<br> Essa amostragem com reposição (conhecida como bootstrap) garante que cada árvore seja treinada em uma visão ligeiramente diferente dos dados, introduzindo a diversidade necessária para que as árvores não sejam muito correlacionadas.

- **Seleção Aleatória de Características:**<br>
    - **Construção das Árvores:**<br> Em cada nó de cada árvore, em vez de considerar todas as características disponíveis para encontrar a melhor divisão (como faria uma árvore de decisão padrão), o Random Forest seleciona aleatoriamente um subconjunto de características.
    - **Diversidade nas Características:**<br> Esta seleção aleatória de características em cada divisão reduz ainda mais a correlação entre as árvores. Se houvesse uma característica muito forte, uma árvore de decisão padrão a escolheria na raiz, e todas as árvores seriam muito semelhantes. A aleatoriedade no Random Forest força as árvores a explorarem outras características e padrões.

- **Combinação das Previsões:**<br>
    - **Votação Majoritária (Classificação):**<br> Para problemas de classificação, cada árvore na floresta faz sua própria previsão. A classe final prevista pelo Random Forest é aquela que recebe a maioria dos votos das árvores individuais.
    - **Média (Regressão):**<br> Para problemas de regressão, cada árvore prevê um valor numérico. A previsão final do Random Forest é a média dos valores previstos por todas as árvores.

- **Vantagens e Desvantagens**
    - Vantagens:
        - Alta Precisão: Geralmente atinge alta precisão em diversas tarefas.
        - Robustez a Overfitting: Menos propenso a superajustar os dados de treinamento em comparação com árvores de decisão individuais.
        - Trata Bem Dados Heterogêneos: Lida eficazmente com dados categóricos e numéricos sem muita pré-processamento.
        - Captura Relações Não Lineares: Não requer suposições de linearidade.
        - Fornece Importância de Features: Ajuda a identificar as características mais influentes.
        - Gerenciamento de Outliers e Dados Faltantes: Relativamente robusto a outliers e pode lidar com dados faltantes.
    - Desvantagens:
        - Complexidade Computacional: O treinamento pode ser computacionalmente intensivo e consumir mais tempo e recursos em comparação com modelos mais simples, especialmente com um grande número de árvores e características.
        - Menor Interpretabilidade: A interpretabilidade é menor do que a de uma única árvore de decisão, pois é um modelo de "caixa preta" devido à agregação de muitas árvores.
        - Requer Mais Memória: Armazenar centenas de árvores pode exigir mais memória.
        - Pode não ser o Melhor para Extrapolação: Assim como as árvores de decisão, o Random Forest não é ideal para extrapolar dados fora do intervalo de treinamento.


## Algoritmo Scikit-learn

O scikit-learn oferece implementações otimizadas para RandomForestClassifier (para classificação) e RandomForestRegressor (para regressão). Ambos compartilham muitos parâmetros em comum, controlando a construção e o comportamento da floresta.
