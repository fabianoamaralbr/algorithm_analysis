[Home](https://github.com/fabianoamaralbr/algorithm_analysis/blob/main/README.md) | [Decision Tree Intro](intro_decision_tree.md)

# Árvore de Regressão

| Parâmetros | Descrição |
| --- | --- |
| `criterion` | O critério de função para medir a qualidade de uma divisão. Pode ser "squared_error" para o erro quadrático médio, "friedman_mse" para o erro quadrático médio de Friedman, "absolute_error" para o erro absoluto médio, ou "poisson" para a divergência de Poisson. |
| `splitter` | A estratégia usada para escolher a divisão em cada nó. Pode ser "best" para escolher a melhor divisão ou "random" para escolher a melhor divisão aleatória. |
| `max_depth` | A profundidade máxima da árvore. Se None, os nós são expandidos até que todas as folhas sejam puras ou até que todas as folhas contenham menos do que min_samples_split amostras. |
| `min_samples_split` | O número mínimo de amostras necessárias para dividir um nó interno. |
| `min_samples_leaf` | O número mínimo de amostras necessárias para estar em um nó folha. |
| `min_weight_fraction_leaf` | A fração mínima de peso das amostras necessárias para estar em um nó folha. |
| `max_features` | O número de características a serem consideradas ao procurar a melhor divisão. |
| `random_state` | Controla a aleatoriedade do estimador. |
| `max_leaf_nodes` | Cresce uma árvore com no máximo max_leaf_nodes em melhores splits. |
| `min_impurity_decrease` | Um nó será dividido se essa divisão induzir uma diminuição da impureza maior ou igual a este valor. |
| `ccp_alpha` | Complexidade de parâmetro de poda mínima. Utilizado para a poda mínima de custo-complexidade. |
| `monotonic_cst` | Indica a restrição de monotonicidade a ser imposta em cada recurso. |

- **Exemplo de uso**

```python
from sklearn.tree import DecisionTreeRegressor

# Criação do modelo
reg = DecisionTreeRegressor(criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1)

# Treinamento do modelo
reg.fit(X_train, y_train)

# Predição
y_pred = reg.predict(X_test)
```

## `criterion`

O parâmetro "criterion" define o critério ou a função que será utilizada para medir a qualidade de uma divisão (split) durante a construção de modelos de árvore de decisão. Esse critério é fundamental para determinar como as divisões são avaliadas e selecionadas, com o objetivo de melhorar a performance do modelo, reduzindo a impureza dos nós à medida que a árvore cresce.

### Opções Disponíveis:
#### "squared_error":
Mede a qualidade de uma divisão através do erro quadrático médio (MSE). Esse critério é muito utilizado em problemas de regressão, onde se busca minimizar a diferença entre os valores reais e as predições do modelo.

#### "friedman_mse":
Baseado no erro quadrático médio de Friedman, esse critério é uma variação do MSE que é otimizada para certos métodos de ensemble, como o Gradient Boosting. Ele introduce pequenos ajustes para melhorar a performance em modelos mais complexos.

#### "absolute_error":
Utiliza o erro absoluto médio (MAE) como medida de qualidade. Essa opção avalia a soma das diferenças absolutas entre os valores previstos e os valores reais, sendo menos sensível a outliers em comparação com o MSE.

#### "poisson":
Mede a qualidade de uma divisão usando a divergência de Poisson. Esse critério é apropriado para problemas onde o alvo é uma variável de contagem e os resíduos seguem uma distribuição de Poisson, como em alguns modelos de regressão para eventos contáveis.

### Funcionamento Geral:
Durante a construção da árvore, o algoritmo calcula a "impureza" (ou erro) no nó pai e nos nós resultantes da divisão. A escolha do critério influencia como essa impureza é mensurada:

- Uma divisão que resulta em uma grande redução do erro (ou impureza) segundo o critério escolhido é considerada boa e, portanto, é realizada.
- Por outro lado, se a melhoria na qualidade (redução da impureza) for pequena, aquela divisão pode não ser realizada, contribuindo para uma árvore mais simples e eficiente.

A opção ideal para o parâmetro "criterion" depende da natureza do problema e das características dos dados. Por exemplo, em tarefas de regressão onde se deseja penalizar fortemente os erros maiores, "squared_error" pode ser mais apropriado; enquanto em casos com presença de outliers, "absolute_error" pode oferecer uma robustez maior.

## `splitter`

O parâmetro "splitter" define a estratégia usada para escolher a divisão (ou split) em cada nó durante a construção da árvore. Existem duas opções principais para esse parâmetro:

### "best":
Essa é a estratégia padrão. Com "best", o algoritmo avalia todas as possíveis divisões em cada nó baseando-se no critério escolhido (por exemplo, Gini ou Entropia) e escolhe a que gera a melhor separação dos dados. Essa abordagem busca maximizar a pureza dos nós resultantes, ou seja, encontrar a divisão que mais reduz a impureza ou incerteza.

### "random":
Quando se utiliza "random", o algoritmo seleciona aleatoriamente uma divisão entre as melhores opções possíveis. Isso significa que, em vez de examinar todas as possíveis divisões e escolher a ótima, ele reduz o espaço de busca ao considerar uma divisão escolhida de forma aleatória dentre um subconjunto de divisões consideradas viáveis. Essa abordagem pode acelerar o processo de treinamento, especialmente em conjuntos de dados com muitas variáveis ou em cenários onde o tempo de computação é crítico.

### Por que usar cada estratégia?
#### Uso do "best":
Quando se deseja maximizar a performance do modelo, especialmente quando a interpretabilidade e a precisão são essenciais.
Ideal para conjuntos de dados menores ou de moderada complexidade, onde é possível explorar todas as possíveis divisões sem grandes custos computacionais.
Garante que em cada nó o algoritmo faça a escolha ótima com base no critério de pureza selecionado (Gini ou Entropia, por exemplo), o que pode reduzir significativamente o risco de erros de classificação.

#### Uso do "random":
Pode ser vantajoso quando se está lidando com conjuntos de dados muito grandes ou de alta dimensão, onde o custo computacional de avaliar todas as divisões pode ser proibitivo.
Em alguns casos, especialmente em métodos de ensemble (como Random Forests), utilizar a abordagem "random" pode introduzir diversidade entre as árvores construídas. Essa diversidade pode ajudar a reduzir a variância e evitar que o conjunto de árvores se ajuste demais aos mesmos padrões dos dados de treinamento.
Pode reduzir o risco de overfitting em casos específicos, pois a aleatoriedade na escolha dos splits pode funcionar como uma forma de regularização.

#### Considerações adicionais:
**Impacto no desempenho:**<br>
A escolha entre "best" e "random" afeta diretamente o tempo de treinamento e a complexidade da árvore. Utilizar "best" tende a produzir árvores mais ajustadas aos dados, enquanto "random" pode acelerar o treinamento, mas possivelmente a custo de uma divisão menos ótima, com impacto leve na performance preditiva.

**Uso em ensemble:**<br>
Técnicas de ensemble, como o Random Forest, frequentemente se beneficiam do uso de splits aleatórios para garantir diversidade entre os estimadores individuais. Isso contribui para que o modelo global seja mais robusto e evite overfitting.

## `max_depth`

O parâmetro "max_depth" define a profundidade máxima que a árvore de decisão pode atingir. Em outras palavras, ele limita quantos níveis a árvore pode ter, evitando um crescimento excessivo da estrutura.

### Quando "max_depth" é definido com um valor inteiro:
A árvore será expandida apenas até o número de níveis especificado. Por exemplo, se max_depth=3, a árvore terá, no máximo, três níveis a partir do nó raiz até os nós folha, independentemente de os nós folha serem puros ou de conterem poucas amostras. Essa limitação ajuda a controlar a complexidade do modelo e pode reduzir o risco de overfitting.

### Quando "max_depth" é definido como None (padrão):
A árvore continuará a expandir até que: Todas as folhas sejam puras, ou seja, contenham elementos de uma só classe (no caso de classificação); ou, Todas as folhas contenham menos que min_samples_split amostras, que é outro parâmetro do algoritmo. Essa configuração sem restrição de profundidade pode levar a árvores muito profundas e complexas, as quais podem capturar ruídos específicos dos dados de treinamento e, consequentemente, não generalizar bem para novos dados.

## `min_samples_split`

O parâmetro "min_samples_split" determina o número mínimo de amostras que um nó interno deve possuir para que ele possa ser dividido em subnós. Em outras palavras, se um nó contém menos amostras do que o limite definido por "min_samples_split", ele não será dividido e automaticamente se tornará um nó folha. Esse parâmetro atua como um controle de complexidade do modelo, ajudando a evitar divisões que possam levar ao overfitting.

### Funcionamento:
#### Limite Mínimo de Amostras:
Se o número de amostras em um nó for menor que o valor especificado por "min_samples_split", o algoritmo não realiza uma nova divisão nesse nó. Isso impede que o modelo se torne excessivamente complexo ao tentar separar grupos muito pequenos ou ruidosos.

#### Efeito em Modelos Complexos vs. Simples:
- **Valor Baixo:** Permite divisões com poucos dados, o que pode resultar em uma árvore muito detalhada e propensa a overfitting, pois o modelo pode se ajustar demais às particularidades dos dados de treinamento.
- **Valor Alto:** Obriga o modelo a ter uma quantidade maior de dados antes de fazer divisões, resultando em menos divisões e, geralmente, uma árvore mais simples. Isso pode aumentar a generalização, mas pode também levar a um sous-ajuste se o valor for excessivamente alto.
#### Flexibilidade:
O parâmetro pode ser definido tanto como um número inteiro, especificando um número exato de amostras, quanto como uma fração (valor entre 0 e 1), representando uma porcentagem do total de amostras do conjunto. Por exemplo, se definido como 0.05 em um conjunto de 1000 amostras, a divisão ocorrerá somente se o nó possuir ao menos 50 amostras.

## `min_samples_leaf`

O parâmetro "min_samples_leaf" define o número mínimo de amostras que devem estar presentes em um nó folha. Isso significa que, ao construir a árvore, mesmo que uma divisão potencialmente melhore a pureza dos nós, ela só será realizada se cada nó folha resultante da divisão possuir pelo menos esse número especificado de amostras.

### Importância e Funcionamento:
#### Evitar Nós com Poucas Amostras:
Permite controlar o tamanho dos nós folha, evitando que a árvore crie folhas com um número muito reduzido de amostras. Nós muito pequenos podem refletir o ruído dos dados e levar ao overfitting.

#### Melhoria na Generalização:
Ao forçar que cada nó folha tenha um número mínimo de amostras, o modelo tende a ser menos sensível a variações aleatórias dos dados, promovendo uma melhor generalização para dados não vistos.

#### Evitar Divisões Irrelevantes:
Mesmo que uma divisão gere uma diminuição na impureza, se um dos nós resultantes não atender ao critério definido por "min_samples_leaf", essa divisão pode ser descartada ou ajustada para respeitar a restrição imposta.

#### Valor Inteiro ou Fração:
Assim como em alguns outros parâmetros (por exemplo, min_samples_split), o "min_samples_leaf" pode ser configurado com um número inteiro, definindo uma quantidade fixa, ou com uma fração (valor entre 0 e 1), representando a proporção mínima do total de amostras que um nó deve conter.

### Impactos na Estrutura da Árvore:
#### Árvore Mais Robusta:
Definir um valor mínimo ajuda a reduzir o risco de criação de ramos muito específicos que se ajustem apenas aos dados de treinamento (overfitting).

#### Simplificação da Árvore:
Restringe a complexidade ao impedir que a árvore se divida em níveis onde os nós folha contenham poucas instâncias. Isso pode resultar em uma árvore mais simples e de fácil interpretação.

#### Controle de Variância:
Nós com poucas amostras podem ser altamente variáveis. Ao garantir um tamanho mínimo, o modelo reduz a variância, tornando as predições mais estáveis.

## `min_weight_fraction_leaf`

O parâmetro "min_weight_fraction_leaf" especifica a fração mínima do peso total das amostras que deve estar presente em um nó folha. Essa configuração é especialmente útil quando cada amostra pode ter um peso associado, permitindo que nós com uma representatividade muito baixa, em termos de peso, sejam evitados na estrutura final da árvore.

### Funcionamento Detalhado:
#### Peso das Amostras:
Em alguns conjuntos de dados, nem todas as amostras têm a mesma importância. Pode-se atribuir um peso a cada amostra para refletir a confiança, relevância ou frequência dessa observação. O parâmetro "min_weight_fraction_leaf" trabalha com uma fração do peso total das amostras contidas no conjunto de dados.

#### Implicação da Fração Mínima:
Ao definir um valor para "min_weight_fraction_leaf" (por exemplo, 0.01 ou 1%), o algoritmo garante que cada nó folha da árvore contenha ao menos essa fração do peso total das amostras dos dados de treinamento. Se, após uma divisão, alguma das folhas não atingir essa fração mínima, a divisão pode ser rejeitada ou ajustada. Essa abordagem é similar à ideia por trás do "min_samples_leaf", mas levando em consideração os pesos atribuídos às amostras, proporcionando um controle mais refinado, principalmente em casos de dados com distribuições ponderadas.

### Quando Utilizar:
#### Este parâmetro é bastante relevante quando:

As amostras têm pesos distintos, seja para representar sua importância, confiabilidade ou frequência. Há necessidade de evitar que nós folha se formem com um peso muito baixo, o que poderia resultar em predições inconsistentes ou instáveis. Deseja-se um controle fino sobre a representatividade dos nós folha na árvore, garantindo que cada folha seja relevante em termos de peso total, e não apenas pelo número de amostras.

### Impacto na Construção da Árvore:
**Regularização:** Ao exigir que cada nó folha contenha uma determinada fração do peso total, o modelo pode evitar divisões que gerem folhas com dados de pouca influência no conjunto, contribuindo para um modelo menos suscetível ao ruído.
**Generalização:** Esse parâmetro auxilia a promover uma melhor generalização, já que nós com peso insignificante são desconsiderados, reduzindo a probabilidade de overfitting sobre amostras de menores relevâncias.

## `max_features`

O parâmetro "max_features" controla o número de características (features) que serão consideradas ao procurar a melhor divisão em cada nó da árvore de decisão. Essa configuração é importante para influenciar tanto a performance quanto a capacidade de generalização do modelo.

### Funcionamento Detalhado:
#### Redução do Espaço de Busca:
Em vez de avaliar todas as características disponíveis em um nó para encontrar a melhor divisão, o algoritmo seleciona um subconjunto aleatório de features. Isso pode acelerar o processo de construção da árvore, principalmente em conjuntos de dados com muitas variáveis.

#### Diversificação e Robustez do Modelo:
Ao limitar o número de características avaliadas em cada divisão, o modelo pode se tornar menos propenso a overfitting, já que evita a dependência excessiva de um pequeno conjunto de variáveis. Essa estratégia é especialmente útil quando usada em métodos de ensemble, como Random Forests, onde cada árvore trabalha com diferentes subconjuntos de dados e features, promovendo a diversidade entre as árvores.

### Configurações Possíveis do "max_features":

**Número Inteiro:**<br>
Define um valor exato, por exemplo, max_features=5 significa que serão consideradas apenas 5 features aleatórias em cada divisão.

**Fração (valor entre 0 e 1):**<br>
Representa uma proporção do total de características. Por exemplo, se houver 20 features e max_features=0.5, serão consideradas 10 características em cada divisão.

**"auto":**<br>
Geralmente, em problemas de classificação, usa a raiz quadrada do número total de características. Para regressão, o padrão costuma ser o número total de características.

**"sqrt":**<br>
Mesma ideia do "auto" em muitos casos, selecionando a raiz quadrada do número total de features.

**"log2":**<br>
Seleciona o logaritmo na base 2 do número total de características.

### Impactos no Modelo:
#### Desempenho Computacional:
Avaliar apenas um subconjunto menor de características pode reduzir o tempo de treinamento, especialmente em bases com alta dimensionalidade.

#### Variabilidade e Bias:
Limitações muito restritivas podem aumentar o viés do modelo, pois algumas divisões potencialmente relevantes podem ser ignoradas. Por outro lado, considerar muitas features pode aumentar a variância, tendo como consequência um maior risco de overfitting aos dados de treinamento.

#### Aplicação em Ensembles:
Em métodos como Random Forests, a aleatoriedade induzida pelo "max_features" ajuda cada árvore a ter perspectivas diferentes do conjunto de dados, melhorando a robustez e a capacidade geral do ensemble em generalizar.

## `random_state`

O parâmetro "random_state" é utilizado para controlar os aspectos aleatórios do estimador, ou seja, ele gerencia a semente (seed) que alimenta os geradores de números aleatórios. Isso é especialmente importante para:

### Reprodutibilidade:
Ao definir um valor específico para "random_state", o processo de treinamento do modelo se torna determinístico. Isso significa que, cada vez que você executar o algoritmo com os mesmos dados e configurações, o resultado será o mesmo. Essa característica é essencial para experimentos e comparações consistentes.

### Controle da Aleatoriedade:
Muitos algoritmos de machine learning, incluindo as árvores de decisão, dependem de processos aleatórios para selecionar divisões ou amostras (no caso de métodos ensemble, por exemplo). A definição de "random_state" garante que essa escolha aleatória ocorra de forma consistente entre diferentes execuções.

### Facilidade de Debugging e Comparação entre Modelos:
Ao ter um comportamento consistente, torna-se mais fácil debugar problemas e comparar o desempenho de variações de modelos, já que os mesmos dados e pontos de divisão serão utilizados.

Em resumo, o "random_state" é fundamental para garantir que os resultados do modelo sejam reproduzíveis e que as variações nos processos estocásticos sejam controladas.

## `max_leaf_node`

O parâmetro "max_leaf_nodes" define o número máximo de folhas (nós terminais) que a árvore de decisão pode ter. Em outras palavras, ao utilizar esse parâmetro, a árvore será ajustada para escolher os melhores splits que lhe permitam alcançar, no máximo, o número especificado de nós folha.

### Funcionamento Detalhado:
#### Limitação Estrutural:
Ao definir "max_leaf_nodes", você impõe uma restrição estrutural na árvore de decisão, de forma que ela não poderá crescer além do número especificado de nós terminais. Isso é útil para reduzir a complexidade do modelo.

#### Seleção dos Melhores Splits:
Durante o processo de construção, o algoritmo realiza splits (divisões) que melhoram a qualidade da árvore, mas a expansão só continua enquanto o número de folhas não ultrapassar o valor de "max_leaf_nodes". Assim, mesmo que mais divisões possam parecer vantajosas em termos de "pureza" dos nós, elas serão interrompidas caso o limite de folhas seja alcançado.

#### Controle do Overfitting:
Essa limitação é uma forma de regularização que pode ajudar a prevenir o sobreajuste aos dados de treinamento. Ao restringir o número de nós folha, evitamos que a árvore se torne excessivamente complexa e espessa, o que pode levar o modelo a se ajustar demais aos detalhes específicos do conjunto de dados de treinamento, prejudicando a generalização.

#### Valor Padrão:
Se "max_leaf_nodes" não for definido (ou for igual a None), a árvore continuará a crescer até que todos os nós sejam puros ou até que os nós tenham menos amostras do que o valor estipulado pelo parâmetro "min_samples_split".

## `min_impurity_decrease`

O parâmetro "min_impurity_decrease" define o limiar mínimo de redução de impureza que uma divisão de nó deve proporcionar para ser considerada válida. Em outras palavras, um nó só será dividido se a melhoria (ou seja, a diminuição da impureza) obtida com a divisão for maior ou igual a esse valor.

### Funcionamento Detalhado:
#### Cálculo da Redução da Impureza:
Ao realizar uma divisão, o algoritmo calcula a impureza do nó pai e a impureza ponderada dos nós filhos obtidos pela divisão. A redução da impureza é dada geralmente por:

Impureza do nó pai − [soma das impurezas ponderadas dos nós filhos]

Se essa diminuição atingir ou for superior ao valor definido em "min_impurity_decrease", a divisão é executada.

#### Controle da Complexidade:
Ao definir um valor para "min_impurity_decrease", você está impondo um critério que impede a realização de divisões que tragam apenas ganhos marginais em termos de pureza. Isso ajuda a evitar que o modelo se ajuste demais aos detalhes menores (ruídos) presentes nos dados, contribuindo para reduzir o overfitting.

#### Ajuste Fino do Modelo:
Um valor baixo para "min_impurity_decrease" permitirá divisões que trazem melhorias mínimas, potencialmente levando a uma árvore muito complexa. Por outro lado, um valor mais alto força a árvore a considerar apenas divisões que realmente reduzam significativamente a impureza, resultando em árvores mais simples e possivelmente uma melhor generalização para novos dados.

#### Uso Conjunto com Outros Parâmetros:
Esse parâmetro é particularmente útil quando combinado com outros controles de complexidade, como max_depth, min_samples_split e min_samples_leaf. Juntos, eles garantem que a árvore não cresça desnecessariamente e que apenas divisões relevantes sejam efetuadas.

## `ccp_alpha`

O parâmetro "ccp_alpha" é utilizado para realizar a poda da árvore através da técnica de poda de custo-complexidade. Esse método busca estabelecer um equilíbrio entre a complexidade da árvore e sua capacidade de generalização. Basicamente, "ccp_alpha" atua como um fator de penalização: ele define um limiar mínimo de melhoria que uma divisão deve oferecer para justificar o aumento da complexidade do modelo. Se uma divisão não proporcionar uma diminuição significativa na impureza (considerando o custo da complexidade), ela pode ser descartada durante o processo de poda.

### Funcionamento Detalhado:
#### Poda de Custo-Complexidade:
Essa técnica avalia a relação entre a redução da impureza (o "custo") e o aumento da complexidade (a quantidade de nós) da árvore. O parâmetro "ccp_alpha" serve como um limiar que, se não for ultrapassado, indica que a divisão não proporciona benefício suficiente, sendo, portanto, eliminada da árvore.

#### Controle da Complexidade:
Um valor maior de "ccp_alpha" fará com que a poda seja mais agressiva, resultando em uma árvore mais simples com menos nós. Essa simplificação pode ajudar a evitar o overfitting, ao eliminar nós que capturam ruídos ou variações específicas dos dados de treinamento.

#### Ajuste Fino:
Encontrar o valor adequado para "ccp_alpha" é essencial para atingir um bom equilíbrio entre viés e variância. Um valor muito baixo pode deixar muitos nós irrelevantes, enquanto um valor muito alto pode tornar a árvore excessivamente simples e incapaz de capturar padrões importantes.

## `monotonic_cst`

A monotonicidade, em modelos de machine learning, refere-se à propriedade de que a previsão do modelo se comporta de forma consistente com uma relação de ordem definida em determinadas variáveis. Por exemplo, em um cenário onde se espera que um aumento em uma variável (como renda) resulte em um aumento na previsão (como pontuação de crédito) ou, alternativamente, que um aumento em outra variável (como idade) possa estar associado a uma diminuição na previsão.

Quando dizemos que há aumento da monotonicidade ou que o modelo é monotonicamente crescente em relação a uma variável, isso significa que à medida que o valor dessa variável aumenta, a predição do modelo também aumenta (ou, no mínimo, não diminui). Por outro lado, quando se diz que há diminuição da monotonicidade ou que o modelo é monotonicamente decrescente em relação a uma variável, significa que conforme o valor dessa variável aumenta, a predição do modelo diminui (ou, no mínimo, não aumenta).

Restrições monotônicas são especialmente importantes quando o conhecimento de domínio indica que a relação entre uma variável e a resposta deve seguir uma direção específica. Isso pode aumentar a interpretabilidade e a confiabilidade do modelo.

Você pode especificar uma restrição monotônica em cada recurso usando o parâmetro. Para cada feição, um valor de 0 indica não restrição, enquanto 1 e -1 indicam um aumento monotônico e restrição de diminuição monotônica, respectivamente.