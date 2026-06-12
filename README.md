# Sistema de Classificação de Risco de Lesão no Futebol

**Sistema baseado em aprendizado de máquina para classificar risco de lesão em atletas de futebol, alinhado ao ODS 3 da ONU (Saúde e Bem-Estar)**

## Objetivo

Desenvolver um sistema de aprendizado de máquina que preve risco de lesão para atletas de futebol com base em:
- Avaliacoes de saúde pré-atividade (sono, estresse, saúde geral)
- Indicadores de fadiga e desconforto pós-atividade
- Estatisticas de desempenho em partida (passes, finalizações, desarmes, gols)

## Estrutura do Projeto

```text
football-injury-risk/
├── data/
│   ├── pre/                          # Dados de questionario pré-atividade (3 CSVs)
│   ├── post/                         # Dados de questionario pós-atividade (3 CSVs)
│   ├── match/                        # Estatísticas de partida (3 CSVs)
│   └── processed/                    # Features e rótulos pré processados (gerado)
├── notebooks/
│   ├── 01_eda.ipynb                  # Análise exploratória
│   ├── 02_feature_engineering.ipynb  # Preparação de features
│   └── 03_model_training.ipynb       # Treino e avaliação de modelos
├── src/
│   ├── preprocessing.py              # Carga e normalização de dados
│   ├── risk_score.py                 # Heurística de risco de lesão
│   ├── train.py                      # Pipeline de treinamento
│   └── predict.py                    # Módulo de inferência
├── app/
│   └── gradio_app.py                 # Interface web interativa
├── models/                           # Modelos treinados (gerado)
├── requirements.txt                  # Dependências do projeto e notebooks
└── README.md
```

## Início Rápido

### Pré-requisitos

- Python 3.12+
- Ambiente virtual (recomendado)

### Instalação

```bash
# Entrar na pasta do projeto
cd football-injury-risk

# Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ou
.venv\Scripts\activate  # Windows

# Instalar dependências do projeto e notebooks
pip install -r requirements.txt
```

O arquivo `requirements.txt` centraliza as bibliotecas necessárias para executar:
- Aplicacao web (Gradio)
- Pipeline de treinamento e inferência
- Notebooks de análise e treinamento

### Fluxo de Trabalho

#### Fase 1: Pré processamento de Dados

Os módulos de pré processamento preparam os CSVs brutos:

```python
from src.preprocessing import load_and_preprocess
from src.risk_score import engineer_risk_label

df = load_and_preprocess(Path.cwd())
df = engineer_risk_label(df)
```

#### Fase 2: Análise Exploratória

Execute o notebook de EDA para entender distribuições e padrões:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Depois execute o notebook de feature engineering:

```bash
jupyter notebook notebooks/02_feature_engineering.ipynb
```

#### Fase 3: Treinamento de Modelos

Treine os classificadores com validação cruzada:

```bash
jupyter notebook notebooks/03_model_training.ipynb
```

#### Fase 4: Interface Web

A aplicação Gradio é o principal ponto de entrada para inferências:

```bash
python3 app/gradio_app.py
```

Abra http://localhost:7860 no navegador.

## Interface Web (Gradio)

A aplicação tem 4 abas principais:

### Aba Predição
Avalia risco de lesão para um atleta por vez:
- Inserir indicadores pré-atividade (saúde geral, energia, sono, estresse, motivação)
- Inserir indicadores pós-atividade (RPE, fadiga, desconforto)
- Inserir metricas de desempenho da partida (passes, finalizações, desarmes, etc.)
- Visualizar classificação de risco, confianca e distribuição de probabilidades

### Aba Envio em Lote
Analisa vários atletas simultaneamente, com três sub-abas:

**Resumo e Probabilidades:**
- Contagem de classes de risco
- Ranking de probabilidade de alto risco
- Barras empilhadas de probabilidade por atleta

**Perfil:**
- Distribuição de risco por posição
- Histograma de confiança do modelo no lote

**Análises:**
Visualizações explicativas para entender o ranking de risco:
- **Mapa de Calor de Sinais de Risco:** 6 sinais normalizados por atleta (desconforto pós-atividade, RPE, fadiga, horas de sono, qualidade do sono, estresse)
- **Mapa de Calor de Probabilidades:** probabilidades previstas para cada classe (Low, Medium, High)
- **Detalhamento do Score Heuristico:** contribuição de cada fator para o score final
- **Radar de Maior Risco:** comparação dos 3 atletas com maior risco em todos os sinais
- **Top Atletas vs Media do Lote:** comparação dos atletas de maior risco com a média do lote
- **Mapa de Bolhas de Risco:** score heurístico (eixo x) vs probabilidade do modelo (eixo y), tamanho da bolha por confiança e cor por classe

**Tabela de Resultados:**
Resultados em formato tabular com features calculadas e classificações de risco.

### Aba Sobre
Informações sobre arquitetura do modelo, composição dos dados e alinhamento com ODS.

### Aba Instruções
Guia detalhado de uso para os fluxos de Predição e Envio em Lote.

## Heurística de Classificação de Risco

O risco de lesão é computado como **score aditivo** à partir de indicadores de saúde:

| Indicador | Pontos |
|-----------|--------|
| Desconforto pós-atividade | +2 |
| RPE >= 8 (esforço elevado) | +1 |
| Fadiga >= 7 | +1 |
| Desconforto pré-atividade | +1 |
| Sono < 6 horas | +1 |
| Estresse = 3 (alto) | +1 |
| Qualidade do sono = 1 (ruim) | +1 |

**Classificação:**
- **0-1 pontos** -> **Risco baixo**
- **2-3 pontos** -> **Risco medio**
- **4+ pontos** -> **Risco alto**

## Resumo do Dataset

- **Total de amostras:** 28 (combinações atleta x partida)
- **Atletas únicos:** 13
- **Partidas:** 3
- **Features:** 57 (após preprocessamento)
- **Distribuição de risco:**
  - Low: 5 (17.86%)
  - Medium: 16 (57.14%)
  - High: 7 (25.00%)

## Formato de Entrada CSV

Para predições em lote, envie arquivos CSV com a seguinte estrutura:

**Questionário Pré-Atividade (pre/{arquivo}.csv):**
Colunas: match_id, nickname, general_health, energy_level, sleep_hours, sleep_quality, wakes_rested, stress_level, motivation, pre_discomfort

**Questionário Pós-Atividade (post/{arquivo}.csv):**
Colunas: match_id, nickname, rpe, fatigue_level, post_discomfort

**Estatísticas de Partida (match/{arquivo}.csv):**
Colunas: match_id, nickname, position, passes_completed, passes_missed, shots_on_goal, shots_off_goal, tackles, fouls, distance_km, goals_scored, assists, possession_pct, pass_accuracy

Os CSVs precisam conter a coluna nickname para identificação do atleta. O sistema mescla os dados automaticamente por match_id e nickname.

## Modelos

Três modelos de classificação são treinados e comparados:

1. **Regressão Logística** (L2 regularizado, balanceado)
2. **Random Forest** (100 árvores, max_depth=5, balanceado)
3. **SVM** (kernel RBF, balanceado)

### Estratégia de Validação Cruzada

- **Método:** Stratified K-Fold (k=3)
- **Motivação:** Dataset pequeno (n=28), exigindo divisão conservadora para preservar distribuição de classes
- **Métricas:** Accuracy, F1 (weighted), Precision, Recall

## Arquivos Principais

### Módulos de Pré processamento
- src/preprocessing.py - Carrega, normaliza e mescla 9 arquivos CSV
- src/risk_score.py - Gera rótulos de risco via heurística

### Pipeline de Treinamento
- src/train.py - Treinamento de multíplos modelos com validação cruzada
- notebooks/03_model_training.ipynb - Notebook para treino e avaliação

### Inferência
- src/predict.py - Carrega modelo e realiza predições
- app/gradio_app.py - Interface web para predições

## Exemplos de Uso

### Interface Web (Recomendado)

Execute a aplicação Gradio:

```bash
python3 app/gradio_app.py
```

Depois, abra http://localhost:7860 no navegador.

#### Aba Predição

Para previsão individual:
1. Preencha os dados de saúde pre-atividade.
2. Informe indicadores pós-atividade (RPE, fadiga, desconforto).
3. Insira as métricas de desempenho da partida.
4. Clique em Predizer risco de lesão.
5. Consulte classificacao de risco, confiança e probabilidades.

#### Aba Envio em Lote

Para processar vários atletas:
1. Envie três CSVs (Pre, Post e Match).
2. Opcionalmente, informe o ID da partida quando não existir nos arquivos.
3. Clique em Executar predição em lote.
4. Consulte tabela e gráficos nas sub-abas Resumo e Probabilidades, Perfil e Análises.

## Limitações e Considerações

1. **Dataset pequeno:** 28 amostras e um volume modesto para ML, exigindo forte regularização.
2. **Valores ausentes:** Imputação por mediana (numérico) e moda (categórico) por posição.
3. **Desbalanceamento de classes:** Tratado com class_weight='balanced' nos modelos.
4. **Generalização limitada:** Modelos treinados em um único time, com possível limitação para outras populações.
5. **Não é dispositivo médico:** Ferramenta de pesquisa, sem substituição de avaliação médica profissional.

## Configuração

Parametros principais (editáveis nos arquivos-fonte):

**src/risk_score.py:**
```python
RISK_WEIGHTS = {...}          # Pesos de cada indicador
RISK_THRESHOLDS = {...}       # Limiares numéricos (RPE, fadiga, sono)
RISK_BOUNDARIES = {...}       # Faixas de score para Low/Medium/High
```

**src/train.py:**
```python
n_splits = 3                  # Folds da validação cruzada
random_state = 42             # Semente de reprodutibilidade
```

## Referências

- Scikit-learn: https://scikit-learn.org/
- Pandas: https://pandas.pydata.org/
- Gradio: https://gradio.app/
- ODS 3 (ONU): https://sdgs.un.org/goals/goal3

## Autoria

**Projeto acadêmico** - Alinhado ao ODS 3 (Saúde e Bem-Estar)

Universidade Uninter
