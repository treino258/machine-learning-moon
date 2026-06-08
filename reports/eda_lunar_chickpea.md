# EDA e modelo de previsão para cultivo lunar de grão-de-bico

## Dados utilizados

- CSV original: `data/Bioremediation of Lunar Regolith Simulant1 Through Mycorrhizal Fungi and Plant Symbioses2 Enables Chickpea to Seed_V2_data.csv`.
- Observações completas usadas para EDA: **76**.
- Dataset aumentado salvo em `data/lunar_chickpea_augmented.csv` com **608** linhas: leituras do experimento em simulante terrestre e cenários lunares simulados.

## Variáveis modeladas

O pipeline converte o tratamento do CSV em percentual de rególito, presença de micorrizas e biomassa seca total. Em seguida adiciona variáveis ajustáveis para ciclo luz/escuridão, radiação, gravidade 1/6g, composição de substrato, nutrientes disponíveis, pH, toxicidade e risco hidráulico.

## EDA por percentual de rególito no CSV original

| regolith_pct | n | biomass_mean | seed_mean | height_mean | mortality_mean |
| --- | --- | --- | --- | --- | --- |
| 0 | 15 | 17.047 | 26.267 | 27.185 | 0.000 |
| 100 | 15 | 3.401 | 0.000 | 14.980 | 0.000 |
| 25 | 16 | 9.286 | 0.688 | 16.431 | 0.000 |
| 50 | 15 | 9.903 | 1.133 | 19.453 | 0.000 |
| 75 | 15 | 9.379 | 4.067 | 22.093 | 0.000 |

## EDA por cenário no dataset aumentado

| scenario | n | biomass_mean | seed_mean | height_mean | mortality_mean |
| --- | --- | --- | --- | --- | --- |
| observed_earth_simulant | 76 | 9.796 | 6.355 | 19.981 | 0.000 |
| simulated_lunar_1 | 76 | 0.274 | 0.245 | 1.391 | 0.862 |
| simulated_lunar_2 | 76 | 0.672 | 0.550 | 2.609 | 0.799 |
| simulated_lunar_3 | 76 | 0.513 | 0.332 | 2.150 | 0.848 |
| simulated_lunar_4 | 76 | 0.422 | 0.382 | 2.130 | 0.805 |
| simulated_lunar_5 | 76 | 0.433 | 0.201 | 1.773 | 0.843 |
| simulated_lunar_6 | 76 | 0.526 | 0.436 | 2.588 | 0.817 |
| simulated_lunar_7 | 76 | 0.087 | 0.040 | 1.309 | 0.850 |

## Modelo de machine learning

Foi treinado um ensemble bootstrap de regressão Ridge, com alvo em escala `log1p`, para prever:

- `biomass_g`: biomassa seca estimada.
- `total_seed`: quantidade estimada de sementes.
- `height_cm`: altura estimada.

Métricas em holdout interno:

| target | mae | rmse | r2 |
| --- | --- | --- | --- |
| biomass_g | 0.682 | 1.810 | 0.798 |
| total_seed | 1.131 | 4.341 | 0.235 |
| height_cm | 2.149 | 4.438 | 0.646 |
| survival_probability_pct | 2.267 | 3.447 | 0.988 |

## Correções dos pontos cegos

1. **Toxicidade real do rególito lunar**: o dataset aumentado inclui `perchlorate_ppm`, `glass_fines_pct` e `toxicity_index`. Isso separa o efeito do simulante JSC-1A/similar do risco de partículas vítreas ultrafinas e sais oxidantes.
2. **Escuridão de 14 dias não linear**: o modelo usa `dark_lethality_index`, uma função logística com comportamento de limiar. Em cenários de 14 dias com pouca luz auxiliar, o gerador cria probabilidade alta de mortalidade, evitando crescimento contínuo irreal.
3. **Gravidade × transpiração**: foi criada a variável `xylem_embolism_risk`, dependente de gravidade, transpiração e geometria do vaso/xilema. Ela não finge calibração empírica completa, mas torna explícito o risco de colapso hidráulico.
4. **Modelo treinado e incerteza**: o modelo usa os coeficientes do ensemble treinado e mantém coeficientes bootstrap para intervalo P5–P95, além da previsão central. Assim, previsões em regiões pouco confiáveis aparecem com incerteza explícita.



Os principais entregáveis são o relatório EDA (`reports/eda_lunar_chickpea.md`), o dataset aumentado (`data/lunar_chickpea_augmented.csv`) e o modelo treinado (`models/lunar_chickpea_model.json`).

## Limitações científicas

Este projeto combina dados reais em simulante com cenários sintéticos. As previsões devem ser usadas para triagem de hipóteses e desenho experimental, não como validação agronômica lunar. O melhor próximo passo é coletar dados em câmara controlada com radiação, períodos reais de escuridão, 1/6g simulada/análoga e medições de água no xilema.
