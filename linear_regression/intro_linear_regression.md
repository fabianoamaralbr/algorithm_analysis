[Home](https://raw.githubusercontent.com/fabianoamaralbr/algorithm_analysis/refs/heads/main/README.md)

# Linear Regression

A regressão linear é um dos algoritmos mais fundamentais e amplamente utilizados na ciência de dados e estatística para modelar a relação entre uma variável dependente (ou resposta) e uma ou mais variáveis independentes (ou preditoras).

## 1. Conceito Básico

A regressão linear busca ajustar uma linha reta (em casos univariados) ou um hiperplano (em casos multivariados) que melhor descreva a relação entre as variáveis. O objetivo é minimizar a diferença entre os valores previstos pelo modelo e os valores observados nos dados.

É utilizada para prever preços, estimar vendas, fazer inferências sobre relações entre variáveis e identificar tendências.

## 2. Formulação Matemática
### Modelo de Regressão Linear Simples

Para um único preditor, o modelo é dado por:

  y = β₀ + β₁x + ε

Onde:

- y é a variável dependente.
- x é a variável independente.
- β₀ é o intercepto (valor predito de y quando x é zero).
- β₁ é o coeficiente angular, que representa a variação média em y para cada unidade de variação em x.
- ε é o termo de erro, que capta a variabilidade não explicada pelo modelo.

### Modelo de Regressão Linear Múltipla

Quando há mais de uma variável independente, o modelo se generaliza para:

  y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε

Onde p representa o número de preditores.

## 3. Estimativa dos Parâmetros

A estimativa dos parâmetros é uma etapa fundamental na construção do modelo de regressão linear. Trata-se de determinar os coeficientes (β₀, β₁, …, βₚ) que melhor ajustam a relação entre as variáveis preditoras e a variável alvo.

### Método dos Mínimos Quadrados
O método mais comum para estimar os coeficientes β é o dos mínimos quadrados, que minimiza a soma dos quadrados dos resíduos:

  J(β₀, β₁, ..., βₚ) = Σᵢ (yᵢ - β₀ - β₁x₁ᵢ - ... - βₚxₚᵢ)²

Através da minimização dessa função de custo, encontra-se a combinação de β que melhor ajusta o modelo aos dados.

Ao elevar os erros ao quadrado, damos maior peso a erros maiores, incentivando o modelo a ajustar bem as observações que se afastam mais da linha proposta. Minimizar a soma dos quadrados significa, na prática, encontrar a linha (ou hiperplano no caso multivariado) que melhor se ajusta aos dados.

### Equação Normal
Quando o número de preditores não é muito alto, os coeficientes podem ser obtidos de forma exata pela equação normal:

  β = (XᵀX)⁻¹ Xᵀy

Onde X é a matriz de design (incluindo uma coluna de 1’s para o intercepto), y é o vetor de respostas, Xᵀ é a transposta de X e (XᵀX)⁻¹ representa a inversa da matriz XᵀX.

- **Vantagens:**<br>
    - Fornece uma solução exata de forma direta.
    - Fácil de implementar para conjuntos de dados pequenos e com poucas variáveis.
- **Limitações:**<br>
    - O cálculo da inversa torna-se computacionalmente custoso e instável se XᵀX for mal condicionado ou o número de preditores for muito grande.
    - Não se adapta bem a conjuntos de dados muito volumosos ou a casos onde há multicolinearidade intensa.

### Gradiente Descendente
Para grandes quantidades de dados ou problemas com muitas variáveis, o gradiente descendente é frequentemente utilizado. Esse método é iterativo e ajusta os coeficientes em pequenos passos na direção negativa do gradiente da função de custo.

- **Algoritmo Básico:**

1. Inicialize os coeficientes (por exemplo, com valores pequenos ou mesmo zeros).
2. Calcule o gradiente da função de custo em relação a cada coeficiente. No caso da regressão linear, o gradiente para cada coeficiente βⱼ (j = 0, 1, …, p) é dado por:

  ∂J/∂βⱼ = -2 Σᵢ (yᵢ - ŷᵢ) xⱼᵢ
  (lembrando que para j = 0, x0ᵢ é igual a 1)

3. Atualize cada coeficiente utilizando o passo de aprendizagem (learning rate, α):

  βⱼ ← βⱼ - α (∂J/∂βⱼ)

4. Repita o processo até que a alteração nos coeficientes se torne insignificante (convergência).

- **Vantagens:**<br>
    - Pode ser aplicado a conjuntos de dados muito grandes, onde o método dos mínimos quadrados não é viável.
    - Flexível: permite usar técnicas de regularização (como Ridge ou Lasso) mediante ajustes na função de custo.
    - Extensível para problemas não lineares e outras arquiteturas de modelos.

A escolha do fator de aprendizado (α) é crucial. Um valor muito alto pode levar a oscilações ou divergência, enquanto um valor muito baixo pode resultar em uma convergência muito lenta.


## 4. Suposições da Regressão Linear

A validade dos resultados de um modelo de regressão linear depende fortemente do atendimento de várias suposições fundamentais. Se essas suposições forem violadas, as estimativas dos coeficientes podem ser tendenciosas, os testes de hipóteses podem ser inválidos e as predições se tornam menos confiáveis. 

- **Linearidade:**<br>
A suposição de linearidade afirma que a relação entre as variáveis independentes (predictoras) e a variável dependente (resultado) é linear. Em outras palavras, a mudança na variável preditora está associada a uma mudança proporcional na resposta.

- **Independência dos Erros:**<br>
A suposição de independência deseja que os erros (resíduos) de cada observação sejam independentes uns dos outros. Isso é especialmente importante em dados temporais (séries temporais) ou dados espaciais, onde as observações podem ser autocorrelacionadas.

- **Homoscedasticidade:**<br>
Homoscedasticidade significa que a variância dos erros é constante para todas as observações. Quando essa suposição é violada, dizemos que há heteroscedasticidade e isso pode afetar a robustez das inferências estatísticas.

- **Normalidade dos Erros:**<br>
A normalidade dos resíduos é importante principalmente para a validade de testes estatísticos e intervalos de confiança. A suposição é de que os erros do modelo seguem uma distribuição normal.

Se os resíduos não forem normalmente distribuídos mesmo após transformações, considerar modelos de regressão generalizados (GLM) é uma alternativa, pois esses modelos assumem distribuições diferentes (como Poisson, binomial, etc.).

- **Ausência de Multicolinearidade (no caso de regressão múltipla):**<br>
A multicolinearidade ocorre quando duas ou mais variáveis independentes estão altamente correlacionadas entre si. Essa situação dificulta a interpretação dos coeficientes, pois torna difícil identificar o efeito individual de cada preditor.

## 5. Interpretação dos Coeficientes

A interpretação dos coeficientes é fundamental para compreender como o modelo traduz a relação entre as variáveis preditoras e a variável dependente. 

### **Intercepto (β₀):**<br>

#### Definição:
O intercepto β₀ representa o valor esperado da variável dependente quando todas as variáveis independentes assumem o valor zero.

#### Interpretação:
Em alguns contextos, o valor zero pode não ter significado prático. Nesse caso, a interpretação do intercepto deve ser tratada com cautela. Em outros cenários, como em modelos econométricos ou de ciências naturais, β₀ pode representar um ponto de referência ou estado basal do sistema.

#### Considerações:

**Unitário:** O valor do intercepto depende da unidade na qual a variável dependente é mensurada. Mudanças na escala afetarão diretamente a interpretação.

**Relevância:** Se o valor zero para as variáveis independentes não for realista ou estiver fora do domínio dos dados, o intercepto pode não ter uma interpretação prática, servindo mais como um componente matemático para ajustar a equação.

### **Coeficientes (β₁, β₂, ...):**<br>

#### Definição:

Cada coeficiente βᵢ associado a uma variável preditora xᵢ quantifica a mudança esperada na variável dependente para um aumento unitário em xᵢ, mantendo as demais variáveis constantes.

#### Interpretação:

- Variação Unitária:<br>
Se β₁ = 2, por exemplo, isso significa que, para cada aumento de 1 unidade em x₁, y aumentará em média 2 unidades, assumindo que os outros preditores se mantenham inalterados.

- Significado Prático:<br>
Nos modelos com regressão simples, a inclinação da reta ajustada fornece uma medida direta de sensibilidade da variável dependente em relação à variável independente.
Em modelos múltiplos, cada βᵢ reflete o efeito marginal da variável preditora, ou seja, o efeito de uma variação em xᵢ sobre y, independentemente de possíveis correlações entre as variáveis.

- Escala e Transformações:<br>
A magnitude dos coeficientes está diretamente ligada à escala das variáveis. Por exemplo, se xᵢ for medido em milhares, o coeficiente refletirá a variação para um incremento de 1000 unidades.
Em algumas situações, pode ser útil padronizar as variáveis (transformação z-score) para facilitar a comparação dos efeitos entre diferentes preditores.

#### Cuidado com a Interpretação:

- Causalidade vs. Associação:<br>
Os coeficientes representam relações associativas e não necessariamente causais, a menos que o desenho do estudo esteja estruturado para afirmar causalidade.

- Interação entre Variáveis:<br>
Em modelos onde incluem termos de interação, a interpretação dos coeficientes individuais torna-se mais complexa, pois o efeito de uma variável pode depender do nível de outra.

- Multicolinearidade:<br>
Se variáveis independentes estiverem fortemente correlacionadas, a interpretação individual de cada coeficiente pode se tornar menos confiável, pois os efeitos podem se sobrepor.


## 6. Métricas de Avaliação

Após ajustar o modelo, é crucial avaliar sua performance utilizando métricas como:

### Erro Quadrático Médio (MSE):

O MSE (Mean Squared Error) é a média dos quadrados das diferenças entre os valores observados (yᵢ) e os valores previstos (ŷᵢ).

Fórmula:
  MSE = (1/n) Σ (yᵢ - ŷᵢ)²

Um MSE menor indica um modelo com melhores predições, pois os erros são baixos. No entanto, por ser expresso em unidades ao quadrado da variável dependente, pode ser difícil interpretá-lo em termos práticos.

Erros maiores são penalizados de forma desproporcional, o que pode ser desejável quando se quer evitar grandes desvios.

### Raiz do Erro Quadrático Médio (RMSE):

O RMSE é a raiz quadrada do MSE.

Fórmula:
  RMSE = √((1/n) Σ (yᵢ - ŷᵢ)²)

Como o RMSE é expresso na mesma unidade da variável dependente, ele permite uma interpretação mais intuitiva do erro médio. Quanto menor o RMSE, melhor o desempenho do modelo.

Muito utilizado para comparar modelos, visto que trás os erros para a escala original dos dados.

### Coeficiente de Determinação (R²):

O R² representa a proporção da variância na variável dependente que é explicada pelas variáveis independentes no modelo.

Fórmula:
  R² = 1 - (SS_res / SS_tot)

Onde:
- SS_res é a soma dos quadrados dos resíduos
- SS_tot é a soma total dos quadrados (variabilidade total dos dados)

Um R² de 1 indica que o modelo explica toda a variabilidade dos dados, enquanto um R² de 0 significa que o modelo não explica nenhum aspecto da variabilidade.

O R² não diminui necessariamente com a adição de variáveis; por isso, é comum utilizar o R² ajustado para modelos com múltiplos preditores.

### Erro Absoluto Médio (MAE):

O MAE (Mean Absolute Error) é a média das diferenças absolutas entre os valores observados e os valores previstos.

Fórmula:
  MAE = (1/n) Σ |yᵢ - ŷᵢ|

O MAE oferece uma medida direta do erro médio, onde cada erro tem o mesmo peso (não sendo penalizado de forma quadrática como no MSE).

Por não elevar ao quadrado os erros, o MAE é menos sensível a outliers.

## 7. Vantagens e Limitações

- **Vantagens**<br>
    - Simplicidade e interpretabilidade.
    - Baixo custo computacional em problemas de baixa dimensão.
    - Muito útil como baseline (modelo de referência) para comparar com modelos mais complexos.
- **Limitações**<br>
    - Pode não capturar relações não-lineares.
    - Sensível a outliers, que podem distorcer os coeficientes.
    - Requer que as suposições sejam atendidas para garantir a validade das inferências.


## Algoritmo Scikit-learn

- [LinearRegression](LinearRegression.md)