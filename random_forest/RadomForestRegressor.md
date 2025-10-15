[Home](https://github.com/fabianoamaralbr/algorithm_analysis/blob/main/README.md) | [Random Forest Intro](intro_random_forest.md)

# Floresta Aleatória de Regressão
O RandomForestRegressor é um algoritmo de aprendizado de máquina do tipo ensemble que constrói múltiplas árvores de decisão durante o treinamento para problemas de regressão. A ideia central é combinar muitos “modelos fracos” (árvores treinadas sobre subconjuntos aleatórios de amostras e características) para formar um modelo forte e robusto. Para regressão, a predição final é a média das predições das árvores individuais.

|Parâmetros|Descrição|
|---|---|
|n_estimators|Número de árvores no conjunto (floresta).|
|criterion|Função que mede a qualidade de uma divisão. Em regressão: "squared_error", "absolute_error", "friedman_mse", "poisson".|
|max_depth|Profundidade máxima de cada árvore.|
|min_samples_split|Número mínimo de amostras para dividir um nó interno.|
|min_samples_leaf|Número mínimo de amostras em um nó folha.|
|min_weight_fraction_leaf|Fração mínima de peso em um nó folha.|
|max_features|Número de features consideradas ao procurar a melhor divisão (em regressão, o padrão é usar todas as features: 1.0).|
|max_leaf_nodes|Número máximo de nós folha por árvore (crescimento best-first).|
|min_impurity_decrease|Limiar mínimo de redução de impureza para aceitar uma divisão.|
|bootstrap|Se usa amostras com reposição para construir árvores.|
|oob_score|Se usa amostras out-of-bag para estimar desempenho de generalização.|
|n_jobs|Número de jobs em paralelo durante o fit.|
|random_state|Controla a aleatoriedade (bootstrap e seleção de features).|
|verbose|Nível de verbosidade do treinamento.|
|warm_start|Reutiliza árvores já treinadas ao aumentar n_estimators.|
|ccp_alpha|Parâmetro de poda por custo-complexidade mínima.|
|max_samples|Tamanho (absoluto ou fração) do bootstrap por árvore quando bootstrap=True.|

- **Exemplo de uso**

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Geração de dados de exemplo
X, y = make_regression(
    n_samples=2000, n_features=20, n_informative=12, noise=10.0, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Criação do modelo RandomForestRegressor
# n_estimators: 300 árvores
# max_depth / min_samples_leaf: leve regularização
# n_jobs: -1 para paralelizar em todos os núcleos
reg = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_depth=20,
    min_samples_leaf=3,
    # Exemplos alternativos de criterion:
    # criterion="squared_error" (padrão), "absolute_error", "friedman_mse", "poisson"
)

# Treinamento
reg.fit(X_train, y_train)

# Predição
y_pred = reg.predict(X_test)

# Avaliação
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.3f} | R²: {r2:.3f}")
```

## `n_estimators`
O parâmetro n_estimators define quantas árvores de decisão compõem a floresta.

### Funcionamento Detalhado
Cada uma das árvores é construída de forma independente (dada a mesma semente e configuração) usando subconjuntos aleatórios de amostras e, opcionalmente, de features.
Na regressão, a predição final é a média das predições das árvores.

### Impacto na Performance
Valor baixo: Pode gerar maior variância e menor robustez (ensemble pouco explorado).
Valor alto: Geralmente melhora a estabilidade e a generalização até um ponto de saturação, com custo computacional aumentando aproximadamente de forma linear.
Não “overfita” por si só com o aumento de árvores; o custo é computacional.

### Estabilidade e Convergência
Com árvores suficientes, o erro de generalização tende a estabilizar. Aumentar além desse ponto traz ganho marginal.

## `criterion`
Em regressão, o criterion define como medir a qualidade de uma divisão.

"squared_error" (padrão): Minimiza o erro quadrático médio (MSE).
Robusto, geralmente com ótimo desempenho.

"absolute_error": Minimiza o erro absoluto médio (MAE).
Mais robusto a outliers do que MSE.
Pode produzir folhas com medianas locais.

"friedman_mse": Variante do MSE que considera melhoria de variância ajustada (útil em gradiente).
Pode gerar ganhos em cenários específicos, próximo ao MSE.

"poisson": Para dados de contagem não-negativos com variância proporcional à média.
Adequado quando a distribuição alvo se assemelha a Poisson.

Na prática, “squared_error” é um ponto de partida sólido; “absolute_error” ajuda com outliers; “poisson” para contagens.

## `max_depth`
Controla a profundidade máxima de cada árvore.

None (padrão): Expande até folhas puras ou outras paradas (min_samples_*).

### Efeito:
Limitar profundidade reduz complexidade por árvore e risco de sobreajuste local.
Profundidade pequena demais → alto viés (underfitting).
Profundidade muito grande → custo alto; no ensemble, o impacto de overfitting é mitigado, mas ainda convém limitar por eficiência.

## `min_samples_split`
Número mínimo de amostras em um nó para que ele seja candidato à divisão. Pode ser inteiro (contagem) ou fração (proporção de n_samples).

### Impacto:
Valores maiores simplificam árvores, reduzindo variância e risco de sobreajuste. Valores muito baixos podem criar divisões instáveis (alta variância).

## `min_samples_leaf`
Número mínimo de amostras em cada nó folha. Inteiro (contagem) ou fração (proporção de n_samples).

### Impacto:
Forte regularizador: impede folhas com poucas amostras, reduzindo variância. Melhora a estabilidade das predições em regiões com poucos dados. Pode aumentar o viés se muito alto.

## `min_weight_fraction_leaf`
Versão ponderada do min_samples_leaf, usando pesos de amostra. Útil quando se usa sample_weight no fit. Garante que cada folha tenha uma fração mínima do peso total.

## Impacto:
Controle refinado de complexidade sob ponderações desiguais. Ajuda a evitar folhas sustentadas por poucas amostras de baixo peso.

## `max_features`
Controla quantas features são consideradas em cada divisão.

Valores possíveis:
Número inteiro: ex. max_features=5.
Fração (0.0–1.0): ex. max_features=0.5 usa 50% das features.
"sqrt": usa n_features
​
"log2": usa log 2(n_features) * log 2 (n_features)

1.0 (padrão no Regressor): usa todas as features.

### Impactos
Diversidade: Subamostrar features reduz correlação entre árvores e variância do ensemble.
Eficiência: Menos features por split acelera o treinamento.

### Trade-off:
Menor max_features → mais aleatoriedade, menor variância, possivelmente maior viés.
Maior max_features → menor viés, maior correlação entre árvores e possivelmente maior variância.
Na regressão, começar com max_features=1.0 é comum; testar "sqrt"/"log2" pode melhorar robustez.

## `max_leaf_nodes`
Limita o número máximo de folhas por árvore (crescimento best-first).

### Impacto:
Regulariza tamanho das árvores, reduzindo overfitting e custo. Alternativa/complemento a max_depth.

## `min_impurity_decrease`
Exige uma redução mínima de impureza para aceitar a divisão. Filtra divisões marginalmente úteis (ruído). Árvores mais simples; reduz risco de sobreajuste. Ajuste fino em conjunto com outros parâmetros de pré/pós-poda.

## `bootstrap`
Determina se cada árvore é treinada com amostragem com reposição.

### bootstrap=True (padrão):
Aumenta a diversidade das árvores (reduz variância do ensemble). Habilita uso de amostras OOB.
### bootstrap=False:
Treina cada árvore no conjunto completo. Árvores tendem a ser mais correlacionadas (pior para ensemble).

## `oob_score`
Estimativa out-of-bag do desempenho, disponível quando bootstrap=True. Para regressão, a pontuação OOB é tipicamente o R² sobre amostras não vistas por cada árvore que as previu.
Acessível via atributo oob_score_ após o fit quando oob_score=True. Útil para avaliar generalização sem validação cruzada explícita.

## `n_jobs`
Paraleliza a construção de árvores.

-1: usa todos os núcleos disponíveis. Acelera significativamente para grandes n_estimators e datasets. Observe a memória ao paralelizar agressivamente.

## `random_state`
Controla as fontes de aleatoriedade (bootstrap e seleção de features).

Fixar um inteiro → resultados reprodutíveis. Essencial para comparações e depuração.

## `verbose`
Controla o nível de mensagens de progresso.

0 (padrão): silencioso.
>=1: informa progresso do treinamento, útil em execuções longas.

## `warm_start`
Permite treinamento incremental.

warm_start=True: chamadas subsequentes de fit adicionam mais árvores se n_estimators aumentar. Mantém os demais hiperparâmetros fixos entre fits. Útil para encontrar ponto de saturação de n_estimators sem treinar do zero.

## `class_weight`
Não se aplica diretamente ao RandomForestRegressor (regressão não tem “classes”). Para dar importância desigual a observações na regressão, use sample_weight no fit e, se necessário, min_weight_fraction_leaf.

## `ccp_alpha`
Poda por custo-complexidade mínima nas árvores individuais. Remove ramos com pouca contribuição relativa à complexidade. Simplifica árvores, melhora generalização e pode acelerar inferência.
Valores altos podam demais (risco de underfitting); próximos de 0 quase não podam.

## `max_samples`
Usado quando bootstrap=True para definir o tamanho do subconjunto de amostras por árvore.

Inteiro: número de amostras com reposição.
Fração (0.0–1.0): proporção do dataset.
None (padrão): usa n_samples (tamanho do dataset).

### Impacto:
Maior diversidade e menor variância com subamostragem.
Treinamento mais rápido com subconjuntos menores.
Muito pequeno pode elevar viés (underfitting).
