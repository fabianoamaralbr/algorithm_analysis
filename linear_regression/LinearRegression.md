[Home](https://github.com/fabianoamaralbr/algorithm_analysis/blob/main/README.md) | [Linear Regrission Intro](intro_linear_regression.md)

# Regressão Linear

| Parâmetro       | Descrição                                                                                   |
|-----------------|---------------------------------------------------------------------------------------------|
| `fit_intercept` | Booleano que indica se o intercepto deve ser ajustado. Se False, não haverá intercepto.     |
| `copy_X`        | Booleano que indica se uma cópia de X deve ser feita. Se False, X pode ser sobrescrito.     |
| `n_jobs`        | Número de jobs a serem usados para o cálculo. -1 significa usar todos os processadores.     |
| `positive`      | Booleano que força os coeficientes a serem positivos.                                       |

- **Exemplo de uso**

```python
from sklearn.linear_model import LinearRegression

# Dados de exemplo
X = [[1, 1], [1, 2], [2, 2], [2, 3]]
y = [1, 1, 2, 2]

# Criação do modelo
model = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None, positive=False)

# Ajuste do modelo
model.fit(X, y)

# Coeficientes
print("Coeficientes:", model.coef_)
print("Intercepto:", model.intercept_)
```

## `fit_intercept`

O parâmetro fit_intercept controla se o algoritmo de regressão linear deve incluir um intercepto, que representa o valor esperado de y quando todas as variáveis preditoras são zero.

Usar fit_intercept=True é a escolha mais comum, pois permite que o modelo se ajuste a um deslocamento presente nos dados.

Definir fit_intercept=False obriga o modelo a passar pela origem, o que pode ser adequado apenas em situações específicas, como quando os dados são centralizados ou o conhecimento do domínio garante um ponto de origem (0,0).

A escolha entre essas opções deve ser orientada pelo pré-processamento dos dados e pelo entendimento do problema em questão.

## `copy_X`

Ele controla se uma cópia de X deve ser feita para preservar os dados originais ou se o algoritmo pode operar diretamente sobre eles.

- **Quando escolher True:**<br>
Quando for importante manter os dados de entrada inalterados para usos futuros ou para garantir que nenhum efeito colateral seja introduzido durante o ajuste.

- **Quando escolher False:**<br>
Quando houver restrições de memória e você tiver certeza de que nenhuma modificação de X afetará posteriores stages do seu fluxo de trabalho.

## `n_jobs`

O parâmetro n_jobs permite controlar o número de processos ou threads que serão usados em operações paralelas, otimizando o desempenho computacional.

- n_jobs > 0: Utiliza exatamente esse número de jobs.
- n_jobs=-1: Utiliza todos os processadores disponíveis.

Utilize n_jobs=-1 para maximizar o uso dos recursos em máquinas com múltiplos núcleos, mas fique atento ao ambiente de execução para evitar sobrecarga. Em ambientes compartilhados, pode ser melhor especificar um número fixo para não comprometer outros processos.

## `positive`

O parâmetro positive é um booleano que, quando definido como True, força o modelo de regressão linear a encontrar coeficientes que sejam positivos (ou, na prática, não negativos). Se definido como False (valor padrão), o modelo estima os coeficientes sem impor restrições de sinal, podendo assumir valores positivos ou negativos conforme os dados.

Em muitos contextos, o conhecimento prévio do domínio sugere que a relação entre uma variável preditora e a variável resposta deve ser positiva. Por exemplo, em modelos que preveem receitas com base em investimentos, pode-se esperar que um maior investimento nunca diminua a receita prevista.

Coeficientes não negativos podem facilitar a interpretação do modelo, pois cada variável contribui de forma aditiva para a predição final. Isso torna mais intuitivo compreender o impacto de cada preditor.

Pode ser útil em áreas como economia, engenharia ou ciências biomédicas, onde se espera que certos efeito sejam apenas incrementais (ou seja, aumentos em determinadas variáveis não devem reduzir o valor predito).