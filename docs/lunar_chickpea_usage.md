# Machine Learning Moon — cultivo lunar de grão-de-bico

Este documento descreve o pipeline de **EDA**, geração de cenários simulados e **modelo preditivo de colheita lunar** usando o CSV em `data/` como base experimental.

## O que foi implementado

- Leitura e normalização do CSV original de crescimento de grão-de-bico em simulante de rególito.
- EDA por percentual de rególito e por cenário experimental/simulado.
- Dataset aumentado com parâmetros lunares ajustáveis:
  - ciclo luz/escuridão e risco não linear de letalidade por 14 dias de noite lunar;
  - radiação diária;
  - gravidade 1/6g;
  - composição do substrato com rególito, vermicomposto/aditivos e micorrizas;
  - nutrientes extraídos inspirados em análises de disponibilidade N/P/K;
  - pH, percloratos, partículas vítreas finas, transpiração e risco hidráulico.
- Modelo de ML treinado de verdade: ensemble bootstrap de regressão Ridge em escala `log1p`.
- Intervalo de incerteza P5–P95 para biomassa, sementes e altura.
- Simulador web standalone para ajustar parâmetros e prever colheitas futuras.

## Arquivos principais

| Caminho | Descrição |
| --- | --- |
| `scripts/lunar_chickpea_pipeline.py` | Pipeline EDA + simulação + treino ML + exportação do simulador. |
| `reports/eda_lunar_chickpea.md` | Relatório com EDA, métricas e correções dos pontos cegos. |
| `data/lunar_chickpea_augmented.csv` | Dataset aumentado com cenários lunares simulados. |
| `models/lunar_chickpea_model.json` | Modelo treinado, coeficientes bootstrap e métricas. |
| `simulator/harvest_simulator.html` | Simulador interativo para previsão de biomassa, sementes e altura. |

## Como executar

O pipeline usa apenas a biblioteca padrão do Python, sem dependências externas.

```bash
python scripts/lunar_chickpea_pipeline.py
```

Após executar, abra o arquivo abaixo em um navegador:

```bash
simulator/harvest_simulator.html
```

## Como interpretar o simulador

1. Ajuste o percentual de rególito, vermicomposto, micorrizas, luz, período de escuridão, radiação, gravidade e nutrientes.
2. Observe as previsões centrais de:
   - biomassa seca (`biomass_g`);
   - sementes (`total_seed`);
   - altura (`height_cm`).
3. Sempre compare a previsão central com o intervalo **P5–P95**. Intervalo amplo significa alta incerteza.
4. Use os alertas de risco para evitar conclusões falsas em cenários extremos:
   - alta letalidade por escuro;
   - alto risco hidráulico em 1/6g;
   - alta toxicidade por perclorato/partículas vítreas.

## Pontos cegos corrigidos

- O simulante de rególito não é tratado como equivalente perfeito ao rególito lunar real: o pipeline adiciona perclorato, partículas vítreas finas e índice de toxicidade.
- A noite lunar de 14 dias não é modelada como estresse linear simples: foi incluído índice de letalidade com comportamento de limiar.
- A interação gravidade × transpiração não é apenas multiplicativa: o modelo inclui risco de embolia/colapso hidráulico dependente de baixa gravidade, transpiração e geometria.
- O simulador não usa só fórmula manual: ele carrega coeficientes de um modelo treinado e mostra incerteza por ensemble bootstrap.

## Limitação importante

As linhas lunares são cenários simulados calibrados qualitativamente a partir do experimento original. O resultado é útil para triagem, planejamento e hipóteses, mas não substitui validação em câmara lunar, testes de radiação, estudos de 1/6g e análise química real do substrato.
