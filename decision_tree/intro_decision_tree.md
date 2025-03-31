[Home](https://github.com/fabianoamaralbr/algorithm_analysis/blob/main/README.md)

# Decision Tree

A Árvore de Decisão (Decision Tree) é um dos algoritmos mais populares em machine learning e aprendizado supervisionado. Ele é usado tanto para classificação quanto para regressão, ajudando a tomar decisões com base em perguntas sequenciais e ramificadas que se assemelham a uma árvore. Cada nó interno da árvore representa uma pergunta ou teste em uma ou mais variáveis de entrada, cada ramo representa o resultado dessa decisão, e cada nó folha (ou terminal) representa uma saída ou decisão final.

As árvores de decisão são modelos intuitivos e interpretáveis que podem ser muito eficazes para determinados tipos de problemas. Contudo, seu uso adequado depende do contexto, da natureza dos dados e dos objetivos do problema de machine learning.


## Quando Usar a Árvore de Decisão

- **Interpretação e Visualização:**<br>
    Quando a interpretabilidade é fundamental para o problema. As árvores de decisão permitem visualizar o caminho de decisão, facilitando a explicação para não-especialistas.

- **Problemas Não Lineares:**<br>
    Em cenários em que as relações entre as variáveis não são linearmente separáveis, as árvores podem capturar interações complexas sem a necessidade de transformação dos dados.

- **Dados Mistos:**<br>
    São particularmente úteis quando se trabalha com dados que contêm variáveis categóricas e numéricas, pois as árvores de decisão lidam bem com ambos os tipos de atributos sem precisar de muita pré-processamento, como normalização.

- **Soluções Rápidas e Prototipagem:**<br>
    Em situações que exigem uma solução rápida ou quando se está em fase de prototipagem, árvores de decisão podem fornecer uma base inicial simples para entender a estrutura dos dados.

- **Problemas com Interação de Variáveis:**<br>
    Quando as interações entre variáveis são importantes para a predição, as árvores de decisão podem, automaticamente, formar regras complexas que capturam essas interações.


## Quando Não Usar a Árvore de Decisão

- **Dados com Muito Ruído:**<br>
    As árvores de decisão tendem a se ajustar demais aos dados de treinamento, especialmente quando o conjunto de dados contém muita variabilidade ou ruído. Neste caso, o modelo pode apresentar um overfitting, tornando sua performance ruim em novos dados.

- **Altas Dimensões:**<br>
    Em situações onde há um número muito alto de features (variáveis), as árvores simples podem se tornar muito complexas e difíceis de interpretar, além de serem instáveis com pequenas variações nos dados.

- **Dependência de Estabilidade:**<br>
    Caso a aplicação exija resultados estáveis e consistentes, as árvores de decisão podem não ser a melhor escolha, pois alterações mínimas no conjunto de dados podem levar a uma estrutura de árvore significativamente diferente. Métodos de ensemble, como Random Forest ou Gradient Boosting, são mais indicados nesses casos.

- **Problemas com Valores Contínuos Muito Relevantes:**<br>
    Embora as árvores de decisão possam tratar variáveis contínuas, em certos cenários, especialmente com dados contínuos altamente correlacionados, métodos como regressão linear ou redes neurais podem capturar melhor a variabilidade dos dados.

- **Alto Custo Computacional em Árvores Profundas:**<br>
    Se o conjunto de dados for muito grande e a árvore crescer muito em profundidade, o custo computacional pode se tornar um problema, e o modelo poderá capturar padrões específicos demais do treinamento, resultando em baixa capacidade de generalização.


## Como funciona?

- **Estrutura da Árvore:**<br>
    A estrutura de uma árvore de decisão é composta por três elementos básicos: nó raiz, nós internos (ou de decisão) e nós folha (ou terminais). Cada um desses elementos desempenha um papel fundamental na organização do fluxo de decisões que o algoritmo utiliza para chegar a uma conclusão.
    - **Nó Raiz:**<br>
        É o ponto inicial da árvore, onde se encontra o conjunto completo de dados. Neste nó, o algoritmo seleciona a melhor variável (atributo) para dividir os dados utilizando um critério de seleção — como Entropia, Ganho de Informação ou Índice de Gini, no caso de classificação, ou, para regressão, critérios baseados em erro (por exemplo, erro quadrático).
    - **Nós Internos (ou de Decisão):**<br>
        Após o nó raiz, a árvore se ramifica em nós internos, onde cada nó representa uma pergunta ou teste sobre uma variável específica. Cada teste pode determinar, por exemplo, se um determinado valor é menor ou maior que um limiar. Em cada divisão, a meta é maximizar a homogeneidade dos subconjuntos de dados resultantes, ou seja, reduzir a impureza ou o erro.
    - **Nós Folha (ou Terminais)**<br>
        Estes são os pontos finais da árvore, onde não há mais divisões. Em problemas de classificação, cada folha contém um rótulo ou a classe prevista (por exemplo, "Cliente fidelizado" ou "Cliente não fidelizado"). Em problemas de regressão, cada folha pode conter um valor contínuo, geralmente a média dos valores das amostras que caíram nessa região.


- **Detalhamento da Construção da Árvore**
    1. **Seleção do Atributo (No Raiz e Nós Internos):**<br>
        O processo começa no nó raiz, onde o algoritmo avalia cada atributo do conjunto de dados aplicando um critério de divisão. Para classificação, as reduções na impureza são calculadas para cada atributo, enquanto que, para regressão, o critério de divisão busca a redução da variância. A variável que proporciona a maior melhoria na homogeneidade dos dados é escolhida para realizar a divisão.
    2. **Divisão dos Dados:**<br>
        Em cada nó, os dados são particionados de acordo com os valores ou categorias do atributo selecionado. Em problemas de classificação, essa divisão pode ser binária (sim/não) ou multiclasses dependendo do número de categorias. Em problemas de regressão, a divisão geralmente é feita com base em um valor limiar definido para a variável contínua.
    3. **Recursão:**<br>
        Este processo de seleção de atributo e divisão ocorre recursivamente para cada nó resultante até que uma condição de parada seja alcançada. As condições de parada podem incluir:
        * O nó se torna puro (todos os exemplos pertencem à mesma classe ou o desvio é mínimo em regressão).
        * Um número mínimo de amostras é atingido.
        * Uma profundidade máxima da árvore é alcançada.
    4. **Poda:**<br>
        Após a construção da árvore completa, pode ser aplicada uma técnica de poda para remover nós que não agregam valor significativo em termos de decisão ou predição. Essa etapa é fundamental para evitar o sobreajuste (overfitting).


- **Visualização e Compreensão**<br>
    Visualmente, tanto as árvores de classificação quanto as de regressão podem ser representadas de maneira análoga:
    - O nó raiz no topo.
    - Ramas que se ramificam para baixo, cada uma representando a aplicação de um teste ou divisão.
    - Nós folha nas extremidades, onde a predição final é apresentada.

    Contudo, a interpretação das folhas diverge entre os dois tipos de árvore. Enquanto, na classificação, as folhas representam categorias ou classes, na regressão elas indicam valores numéricos que são as predições do modelo.


- **Vantagens e Desvantagens:**
    - Vantagens:
        - Fácil de interpretar e visualizar.
        - Não necessita de pré-processamento complexo, como normalização.
        - Capaz de lidar tanto com dados categóricos quanto numéricos.
    - Desvantagens:
        - Tendência a superajustar (overfitting) quando a árvore cresce muito.
        - Pequenas variações nos dados podem resultar em desenhos de árvore completamente diferentes.
        - Pode ser enviesada se alguma classe tiver muito mais exemplos do que as demais.
    

## Algoritmo Scikit-learn

- [DecisionTreeClassifier](DecisionTreeClassifier.md)
- [DecisionTreeRegressor](DecisionTreeRegressor.md)