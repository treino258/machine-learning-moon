# Machine Learning Moon — EDA e Machine Learning para cultivo lunar

Este documento descreve a parte que será entregue no trabalho: **EDA + Machine Learning**.

## O que foi implementado

- Leitura e normalização do CSV original de crescimento de grão-de-bico em simulante de rególito.
- EDA por percentual de rególito e por cenário experimental/simulado.
- Dataset aumentado com parâmetros lunares simulados:
  - ciclo luz/escuridão e risco não linear de letalidade por 14 dias de noite lunar;
  - radiação diária;
  - gravidade 1/6g;
  - composição do substrato com rególito, vermicomposto/aditivos e micorrizas;
  - nutrientes extraídos inspirados em análises de disponibilidade N/P/K;
  - pH, percloratos, partículas vítreas finas, transpiração e risco hidráulico.
- Modelo de ML treinado: ensemble bootstrap de regressão Ridge em escala `log1p`.
- Intervalo de incerteza P5–P95 para biomassa, sementes, altura e chance de sobrevivência.

## Arquivos principais para entregar

| Caminho | Descrição |
| --- | --- |
| `scripts/lunar_chickpea_pipeline.py` | Pipeline EDA + simulação + treino ML + exportação dos artefatos. |
| `reports/eda_lunar_chickpea.md` | Relatório com EDA, métricas do modelo e correções dos pontos cegos. |
| `data/lunar_chickpea_augmented.csv` | Dataset aumentado com cenários lunares simulados. |
| `models/lunar_chickpea_model.json` | Modelo treinado, coeficientes bootstrap, médias/escalas e métricas. |

## Como executar

O pipeline usa apenas a biblioteca padrão do Python, sem dependências externas.

```bash
python scripts/lunar_chickpea_pipeline.py
```

Esse comando gera/atualiza:

- `data/lunar_chickpea_augmented.csv`
- `models/lunar_chickpea_model.json`
- `reports/eda_lunar_chickpea.md`

## Resultados previstos pelo ML

O modelo prevê quatro saídas:

1. `biomass_g`: biomassa seca estimada em gramas.
2. `total_seed`: quantidade estimada de sementes.
3. `height_cm`: altura estimada em centímetros.
4. `survival_probability_pct`: porcentagem estimada de chance de sobrevivência.

Cada saída possui previsão central e coeficientes bootstrap para calcular intervalo de incerteza P5–P95.

## Pontos cegos corrigidos

- O simulante de rególito não é tratado como equivalente perfeito ao rególito lunar real: o pipeline adiciona perclorato, partículas vítreas finas e índice de toxicidade.
- A noite lunar de 14 dias não é modelada como estresse linear simples: foi incluído índice de letalidade com comportamento de limiar.
- A interação gravidade × transpiração não é apenas multiplicativa: o modelo inclui risco de embolia/colapso hidráulico dependente de baixa gravidade, transpiração e geometria.
- As previsões têm incerteza explícita por ensemble bootstrap.

## Limitação importante

As linhas lunares são cenários simulados calibrados qualitativamente a partir do experimento original. O resultado é útil para EDA, Machine Learning, triagem e planejamento experimental, mas não substitui validação em câmara lunar, testes de radiação, estudos de 1/6g e análise química real do substrato.
