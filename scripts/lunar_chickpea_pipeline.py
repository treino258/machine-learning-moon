#!/usr/bin/env python3
"""EDA, simulated lunar-agriculture data generation, and ML training.

The pipeline intentionally uses only the Python standard library so the project can be
run in restricted environments. It reads the chickpea/lunar-regolith CSV in data/,
normalizes experimental observations, augments them with explicit lunar environment
parameters, trains bootstrap ridge-regression ensembles, and exports:

- reports/eda_lunar_chickpea.md
- data/lunar_chickpea_augmented.csv
- models/lunar_chickpea_model.json
- simulator/harvest_simulator.html
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "Bioremediation of Lunar Regolith Simulant1 Through Mycorrhizal Fungi and Plant Symbioses2 Enables Chickpea to Seed_V2_data.csv"
AUGMENTED_CSV = ROOT / "data" / "lunar_chickpea_augmented.csv"
REPORT_MD = ROOT / "reports" / "eda_lunar_chickpea.md"
MODEL_JSON = ROOT / "models" / "lunar_chickpea_model.json"
SIMULATOR_HTML = ROOT / "simulator" / "harvest_simulator.html"
SEED = 42
TARGETS = ["biomass_g", "total_seed", "height_cm"]
FEATURES = [
    "intercept",
    "regolith_pct",
    "vermicompost_pct",
    "mycorrhiza",
    "light_hours",
    "dark_days_continuous",
    "radiation_msv_day",
    "gravity_g",
    "nitrogen_mg_kg",
    "phosphorus_mg_kg",
    "potassium_mg_kg",
    "ph",
    "perchlorate_ppm",
    "glass_fines_pct",
    "transpiration_index",
    "xylem_geometry_risk",
    "dark_lethality_index",
    "xylem_embolism_risk",
    "toxicity_index",
]


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def pct_from_sample(sample: str) -> int:
    if sample.startswith("Control"):
        return 0
    digits = "".join(ch for ch in sample if ch.isdigit())
    return int(digits) if digits else 0


def mycorrhiza_from_sample(sample: str) -> int:
    return 1 if sample.strip().endswith("+") else 0


def read_raw_rows() -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    with RAW_CSV.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            sample = (raw.get("Sample ") or "").strip()
            if not sample:
                continue
            root = to_float(raw.get("Dry Root (g)"))
            shoot = to_float(raw.get("Dry Shoot (g)"))
            seed = to_float(raw.get("Total Seed"))
            seed_weight = to_float(raw.get("Seed Weight (g)"))
            height = to_float(raw.get("Height"))
            slakes = to_float(raw.get("SLAKES"))
            regolith = pct_from_sample(sample)
            mycorrhiza = mycorrhiza_from_sample(sample)
            biomass = (root or 0.0) + (shoot or 0.0) if root is not None or shoot is not None else None
            rows.append(
                {
                    "sample": sample,
                    "replicate": int(to_float(raw.get("Sample #")) or 0),
                    "regolith_pct": regolith,
                    "mycorrhiza": mycorrhiza,
                    "dry_root_g": root,
                    "dry_shoot_g": shoot,
                    "biomass_g": biomass,
                    "total_seed": seed if seed is not None else 0.0,
                    "seed_weight_g": seed_weight if seed_weight is not None else 0.0,
                    "height_cm": height,
                    "slakes": slakes,
                }
            )
    return rows


def risk_indices(row: Dict[str, float]) -> Dict[str, float]:
    dark_days = row["dark_days_continuous"]
    aux_light = row["light_hours"] / 24.0
    # C3 plants are not assumed to degrade smoothly through long darkness; a
    # threshold-like lethality term is exposed and used by both ML and simulator.
    dark_lethality = 1.0 / (1.0 + math.exp(-(dark_days - 8.0 + 3.0 * aux_light) / 1.6))

    transpiration = row["transpiration_index"]
    low_g = max(0.0, (0.38 - row["gravity_g"]) / 0.38)
    # Non-linear proxy for possible gas entrapment/embolism in low gravity.
    xylem_embolism = min(1.0, low_g * (0.45 + 0.75 * transpiration) * (0.55 + row["xylem_geometry_risk"]))

    toxicity = min(
        1.0,
        0.55 * row["glass_fines_pct"] / 18.0
        + 0.45 * row["perchlorate_ppm"] / 120.0
        + 0.18 * row["regolith_pct"] / 100.0,
    )
    return {
        "dark_lethality_index": round(dark_lethality, 5),
        "xylem_embolism_risk": round(xylem_embolism, 5),
        "toxicity_index": round(toxicity, 5),
    }


def base_environment(obs: Dict[str, float | int | str], rng: random.Random) -> Dict[str, float]:
    regolith = float(obs["regolith_pct"])
    myco = float(obs["mycorrhiza"])
    vermicompost = max(0.0, 100.0 - regolith) if regolith > 0 else 100.0
    row = {
        "regolith_pct": regolith,
        "vermicompost_pct": vermicompost,
        "mycorrhiza": myco,
        "light_hours": 12.0,
        "dark_days_continuous": 0.5,
        "radiation_msv_day": 0.02,
        "gravity_g": 1.0,
        "nitrogen_mg_kg": max(18.0, 125.0 - 0.72 * regolith + 18.0 * myco + rng.gauss(0, 4)),
        "phosphorus_mg_kg": max(6.0, 38.0 - 0.22 * regolith + 8.0 * myco + rng.gauss(0, 2)),
        "potassium_mg_kg": max(25.0, 210.0 - 0.95 * regolith + 20.0 * myco + rng.gauss(0, 8)),
        "ph": 6.2 + 0.03 * regolith - 0.22 * myco + rng.gauss(0, 0.12),
        "perchlorate_ppm": 0.0,
        "glass_fines_pct": max(0.0, 0.04 * regolith + rng.gauss(0, 0.25)),
        "transpiration_index": 1.0 + rng.gauss(0, 0.06),
        "xylem_geometry_risk": 0.22 + rng.random() * 0.16,
    }
    row.update(risk_indices(row))
    return {k: round(v, 5) for k, v in row.items()}


def lunar_counterfactual(obs: Dict[str, float | int | str], rng: random.Random) -> Dict[str, float]:
    regolith = float(obs["regolith_pct"])
    myco = float(obs["mycorrhiza"])
    vermicompost = max(0.0, 100.0 - regolith)
    row = {
        "regolith_pct": regolith,
        "vermicompost_pct": vermicompost,
        "mycorrhiza": myco,
        "light_hours": rng.choice([0.0, 4.0, 8.0, 12.0, 16.0]),
        "dark_days_continuous": rng.choice([3.0, 7.0, 10.0, 14.0]),
        "radiation_msv_day": rng.uniform(0.3, 2.2),
        "gravity_g": 1.0 / 6.0,
        "nitrogen_mg_kg": max(5.0, 85.0 - 0.55 * regolith + 24.0 * myco + 0.35 * vermicompost + rng.gauss(0, 8)),
        "phosphorus_mg_kg": max(2.0, 24.0 - 0.13 * regolith + 12.0 * myco + 0.09 * vermicompost + rng.gauss(0, 3)),
        "potassium_mg_kg": max(8.0, 125.0 - 0.50 * regolith + 28.0 * myco + 0.60 * vermicompost + rng.gauss(0, 12)),
        "ph": 6.1 + 0.035 * regolith - 0.18 * myco + rng.gauss(0, 0.2),
        # Explicit blind-spot variables: simulant values can be near zero, lunar-like
        # scenarios are allowed to include sharper glass and perchlorate contamination.
        "perchlorate_ppm": max(0.0, rng.gauss(18.0 + 0.45 * regolith, 14.0)),
        "glass_fines_pct": max(0.0, rng.gauss(2.5 + 0.12 * regolith, 2.0)),
        "transpiration_index": max(0.15, rng.gauss(0.8 + 0.005 * vermicompost, 0.18)),
        "xylem_geometry_risk": rng.uniform(0.2, 0.95),
    }
    row.update(risk_indices(row))
    return {k: round(v, 5) for k, v in row.items()}


def apply_environment_to_targets(obs: Dict[str, float | int | str], env: Dict[str, float], rng: random.Random) -> Dict[str, float]:
    observed_biomass = float(obs["biomass_g"] or 0.0)
    observed_seed = float(obs["total_seed"] or 0.0)
    observed_height = float(obs["height_cm"] or 0.0)

    nutrient_score = min(1.2, 0.25 + 0.003 * env["nitrogen_mg_kg"] + 0.006 * env["phosphorus_mg_kg"] + 0.0012 * env["potassium_mg_kg"])
    ph_penalty = max(0.0, abs(env["ph"] - 6.6) - 0.8) * 0.12
    radiation_penalty = min(0.55, 0.12 * env["radiation_msv_day"])
    light_bonus = min(0.28, max(-0.30, (env["light_hours"] - 12.0) * 0.035))
    risk_penalty = 0.62 * env["dark_lethality_index"] + 0.42 * env["xylem_embolism_risk"] + 0.48 * env["toxicity_index"]
    myco_bonus = 0.06 * env["mycorrhiza"]
    factor = max(0.0, nutrient_score + light_bonus + myco_bonus - ph_penalty - radiation_penalty - risk_penalty)

    # Hard-stop component: this prevents the 14-day dark period from being modeled as
    # only mild continuous stress. It creates zero/near-zero outcomes in lethal regimes.
    mortality_probability = min(0.98, 0.18 + 0.72 * env["dark_lethality_index"] + 0.25 * env["xylem_embolism_risk"] + 0.20 * env["toxicity_index"])
    if env["dark_days_continuous"] >= 14 and env["light_hours"] <= 4:
        mortality_probability = max(mortality_probability, 0.9)
    alive = rng.random() > mortality_probability
    survival = 1.0 if alive else rng.uniform(0.0, 0.08)

    biomass = max(0.0, observed_biomass * factor * survival * rng.uniform(0.82, 1.18))
    seeds = max(0.0, observed_seed * (factor ** 1.35) * survival * rng.uniform(0.70, 1.25))
    height = max(0.0, observed_height * (0.35 + 0.65 * factor) * survival * rng.uniform(0.86, 1.15))
    return {
        "biomass_g": round(biomass, 4),
        "total_seed": round(seeds, 4),
        "height_cm": round(height, 4),
        "survival": round(survival, 4),
        "mortality_probability": round(mortality_probability, 4),
    }


def build_augmented_dataset(raw_rows: List[Dict[str, float | int | str]]) -> List[Dict[str, float | int | str]]:
    rng = random.Random(SEED)
    complete = [r for r in raw_rows if r["biomass_g"] is not None and r["height_cm"] is not None]
    augmented: List[Dict[str, float | int | str]] = []
    for obs in complete:
        env = base_environment(obs, rng)
        row = {**obs, **env, "scenario": "observed_earth_simulant", "survival": 1.0, "mortality_probability": 0.0}
        augmented.append(row)
        for idx in range(7):
            env = lunar_counterfactual(obs, rng)
            targets = apply_environment_to_targets(obs, env, rng)
            augmented.append({**obs, **env, **targets, "scenario": f"simulated_lunar_{idx + 1}"})
    return augmented


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return statistics.fmean(vals) if vals else 0.0


def group_stats(rows: Sequence[Dict[str, float | int | str]], key: str) -> List[Dict[str, float | str]]:
    groups: Dict[str, List[Dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    out = []
    for group, items in sorted(groups.items()):
        out.append(
            {
                key: group,
                "n": len(items),
                "biomass_mean": mean(float(x["biomass_g"] or 0) for x in items),
                "seed_mean": mean(float(x["total_seed"] or 0) for x in items),
                "height_mean": mean(float(x["height_cm"] or 0) for x in items),
                "mortality_mean": mean(float(x.get("mortality_probability", 0)) for x in items),
            }
        )
    return out


def transpose(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(col) for col in zip(*matrix)]


def matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[sum(x * y for x, y in zip(row, col)) for col in transpose(b)] for row in a]


def matvec(a: Sequence[Sequence[float]], v: Sequence[float]) -> List[float]:
    return [sum(x * y for x, y in zip(row, v)) for row in a]


def solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        if abs(aug[col][col]) < 1e-12:
            aug[col][col] = 1e-12
        divisor = aug[col][col]
        aug[col] = [v / divisor for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            aug[r] = [rv - factor * cv for rv, cv in zip(aug[r], aug[col])]
    return [aug[i][-1] for i in range(n)]


def feature_vector(row: Dict[str, float | int | str], means: Dict[str, float], scales: Dict[str, float]) -> List[float]:
    values = []
    for feature in FEATURES:
        if feature == "intercept":
            values.append(1.0)
        else:
            values.append((float(row[feature]) - means[feature]) / scales[feature])
    return values


def train_ridge(x: Sequence[Sequence[float]], y: Sequence[float], alpha: float) -> List[float]:
    xt = transpose(x)
    xtx = matmul(xt, x)
    for i in range(1, len(xtx)):
        xtx[i][i] += alpha
    xty = matvec(xt, y)
    return solve_linear_system(xtx, xty)


def predict(coeffs: Sequence[float], x: Sequence[float]) -> float:
    return sum(c * v for c, v in zip(coeffs, x))


def metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    errors = [a - b for a, b in zip(y_true, y_pred)]
    mae = mean(abs(e) for e in errors)
    rmse = math.sqrt(mean(e * e for e in errors))
    y_bar = mean(y_true)
    ss_res = sum(e * e for e in errors)
    ss_tot = sum((y - y_bar) ** 2 for y in y_true) or 1.0
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(1 - ss_res / ss_tot, 4)}


def quantile(values: Sequence[float], q: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    idx = (len(sorted_values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_values[int(idx)]
    return sorted_values[lo] * (hi - idx) + sorted_values[hi] * (idx - lo)


def train_models(rows: List[Dict[str, float | int | str]]) -> Dict[str, object]:
    rng = random.Random(SEED)
    train_rows = [row for row in rows if float(row["biomass_g"]) >= 0 and float(row["height_cm"]) >= 0]
    means = {f: mean(float(r[f]) for r in train_rows) for f in FEATURES if f != "intercept"}
    scales = {}
    for f in FEATURES:
        if f == "intercept":
            continue
        vals = [float(r[f]) for r in train_rows]
        sd = statistics.pstdev(vals) or 1.0
        scales[f] = sd
    x_all = [feature_vector(r, means, scales) for r in train_rows]
    model: Dict[str, object] = {
        "version": "1.0",
        "seed": SEED,
        "features": FEATURES,
        "feature_means": means,
        "feature_scales": scales,
        "targets": {},
        "notes": [
            "Bootstrap ridge regression trained on observed CSV plus simulated lunar scenarios.",
            "Predictions are scenario-screening estimates, not biological validation.",
            "Use prediction intervals and risk indices; high dark lethality can imply plant death.",
        ],
    }
    split = list(range(len(train_rows)))
    rng.shuffle(split)
    test_size = max(8, len(split) // 5)
    test_idx = set(split[:test_size])
    train_idx = [i for i in split if i not in test_idx]
    test_idx_list = list(test_idx)
    for target in TARGETS:
        y_all = [float(r[target]) for r in train_rows]
        ensembles = []
        for _ in range(80):
            sample_idx = [rng.choice(train_idx) for _ in train_idx]
            x = [x_all[i] for i in sample_idx]
            y = [math.log1p(y_all[i]) for i in sample_idx]
            ensembles.append(train_ridge(x, y, alpha=0.8))
        center = train_ridge([x_all[i] for i in train_idx], [math.log1p(y_all[i]) for i in train_idx], alpha=0.8)
        preds = [max(0.0, math.expm1(predict(center, x_all[i]))) for i in test_idx_list]
        truth = [y_all[i] for i in test_idx_list]
        model["targets"][target] = {
            "center_coefficients": center,
            "bootstrap_coefficients": ensembles,
            "metrics": metrics(truth, preds),
        }
    return model


def write_csv(rows: List[Dict[str, float | int | str]]) -> None:
    fieldnames = [
        "scenario",
        "sample",
        "replicate",
        "regolith_pct",
        "vermicompost_pct",
        "mycorrhiza",
        "light_hours",
        "dark_days_continuous",
        "radiation_msv_day",
        "gravity_g",
        "nitrogen_mg_kg",
        "phosphorus_mg_kg",
        "potassium_mg_kg",
        "ph",
        "perchlorate_ppm",
        "glass_fines_pct",
        "transpiration_index",
        "xylem_geometry_risk",
        "dark_lethality_index",
        "xylem_embolism_risk",
        "toxicity_index",
        "dry_root_g",
        "dry_shoot_g",
        "seed_weight_g",
        "slakes",
        "biomass_g",
        "total_seed",
        "height_cm",
        "survival",
        "mortality_probability",
    ]
    with AUGMENTED_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: Sequence[Dict[str, object]], headers: Sequence[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        vals = []
        for h in headers:
            value = row[h]
            vals.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(raw_rows: List[Dict[str, float | int | str]], rows: List[Dict[str, float | int | str]], model: Dict[str, object]) -> None:
    complete = [r for r in raw_rows if r["biomass_g"] is not None and r["height_cm"] is not None]
    by_regolith = group_stats(complete, "regolith_pct")
    by_scenario = group_stats(rows, "scenario")[:8]
    metrics_rows = []
    for target, spec in model["targets"].items():
        metric = spec["metrics"]
        metrics_rows.append({"target": target, **metric})
    text = f"""# EDA e modelo de previsão para cultivo lunar de grão-de-bico

## Dados utilizados

- CSV original: `{RAW_CSV.relative_to(ROOT)}`.
- Observações completas usadas para EDA: **{len(complete)}**.
- Dataset aumentado salvo em `{AUGMENTED_CSV.relative_to(ROOT)}` com **{len(rows)}** linhas: leituras do experimento em simulante terrestre e cenários lunares simulados.

## Variáveis modeladas

O pipeline converte o tratamento do CSV em percentual de rególito, presença de micorrizas e biomassa seca total. Em seguida adiciona variáveis ajustáveis para ciclo luz/escuridão, radiação, gravidade 1/6g, composição de substrato, nutrientes disponíveis, pH, toxicidade e risco hidráulico.

## EDA por percentual de rególito no CSV original

{markdown_table(by_regolith, ["regolith_pct", "n", "biomass_mean", "seed_mean", "height_mean", "mortality_mean"])}

## EDA por cenário no dataset aumentado

{markdown_table(by_scenario, ["scenario", "n", "biomass_mean", "seed_mean", "height_mean", "mortality_mean"])}

## Modelo de machine learning

Foi treinado um ensemble bootstrap de regressão Ridge, com alvo em escala `log1p`, para prever:

- `biomass_g`: biomassa seca estimada.
- `total_seed`: quantidade estimada de sementes.
- `height_cm`: altura estimada.

Métricas em holdout interno:

{markdown_table(metrics_rows, ["target", "mae", "rmse", "r2"])}

## Correções dos pontos cegos

1. **Toxicidade real do rególito lunar**: o dataset aumentado inclui `perchlorate_ppm`, `glass_fines_pct` e `toxicity_index`. Isso separa o efeito do simulante JSC-1A/similar do risco de partículas vítreas ultrafinas e sais oxidantes.
2. **Escuridão de 14 dias não linear**: o modelo usa `dark_lethality_index`, uma função logística com comportamento de limiar. Em cenários de 14 dias com pouca luz auxiliar, o gerador cria probabilidade alta de mortalidade, evitando crescimento contínuo irreal.
3. **Gravidade × transpiração**: foi criada a variável `xylem_embolism_risk`, dependente de gravidade, transpiração e geometria do vaso/xilema. Ela não finge calibração empírica completa, mas torna explícito o risco de colapso hidráulico.
4. **Simulador treinado e incerteza**: o simulador usa os coeficientes do ensemble treinado e retorna intervalo P5–P95, além da previsão central. Assim, previsões em regiões pouco confiáveis aparecem com incerteza explícita.

## Como executar

```bash
python scripts/lunar_chickpea_pipeline.py
```

Depois abra `simulator/harvest_simulator.html` no navegador para ajustar parâmetros e prever colheitas futuras.

## Limitações científicas

Este projeto combina dados reais em simulante com cenários sintéticos. As previsões devem ser usadas para triagem de hipóteses e desenho experimental, não como validação agronômica lunar. O melhor próximo passo é coletar dados em câmara controlada com radiação, períodos reais de escuridão, 1/6g simulada/análoga e medições de água no xilema.
"""
    REPORT_MD.write_text(text, encoding="utf-8")


def write_simulator(model: Dict[str, object]) -> None:
    embedded = json.dumps(model, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang=\"pt-BR\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Simulador de colheita lunar - grão-de-bico</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #0c1020; color: #f6f7fb; }}
    main {{ max-width: 1120px; margin: auto; padding: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(245px, 1fr)); gap: 16px; }}
    .card {{ background: #171d35; border: 1px solid #2b355d; border-radius: 16px; padding: 16px; box-shadow: 0 10px 30px #0005; }}
    label {{ display: grid; gap: 6px; margin: 10px 0; font-size: 0.92rem; }}
    input {{ width: 100%; accent-color: #9be564; }}
    output {{ font-weight: 700; color: #9be564; }}
    .result {{ font-size: 1.35rem; }}
    .risk-high {{ color: #ff7676; }} .risk-med {{ color: #ffd166; }} .risk-low {{ color: #9be564; }}
    small, p {{ color: #c9d2ef; line-height: 1.45; }}
  </style>
</head>
<body>
<main>
  <h1>Simulador de previsão de colheita lunar</h1>
  <p>Ajuste os parâmetros do ambiente para estimar biomassa, sementes e altura de grão-de-bico. O modelo é um ensemble Ridge treinado pelo pipeline do repositório e mostra intervalo P5–P95.</p>
  <section class=\"grid\" id=\"controls\"></section>
  <section class=\"grid\" id=\"results\"></section>
  <section class=\"card\">
    <h2>Alertas de pontos cegos corrigidos</h2>
    <ul>
      <li>Rególito real: inclua percloratos e partículas vítreas ultrafinas, não só percentual de simulante.</li>
      <li>Escuridão: 14 dias sem luz auxiliar pode ser regime letal, não apenas estresse linear.</li>
      <li>1/6g: risco hidráulico pode crescer por embolia/bolhas no xilema; ajuste transpiração e geometria.</li>
      <li>Incerteza: sempre leia P5–P95; intervalos amplos indicam cenário fora da base de dados.</li>
    </ul>
  </section>
</main>
<script>
const model = {embedded};
const specs = [
  ['regolith_pct','Rególito (%)',0,100,1,50],
  ['vermicompost_pct','Vermicomposto/aditivos (%)',0,100,1,50],
  ['mycorrhiza','Micorrizas (0/1)',0,1,1,1],
  ['light_hours','Luz artificial por dia (h)',0,24,1,12],
  ['dark_days_continuous','Escuridão contínua (dias)',0,14,0.5,7],
  ['radiation_msv_day','Radiação (mSv/dia)',0,3,0.05,0.8],
  ['gravity_g','Gravidade (g)',0.05,1,0.01,0.1667],
  ['nitrogen_mg_kg','Nitrogênio extraído (mg/kg)',0,220,1,90],
  ['phosphorus_mg_kg','Fósforo extraído (mg/kg)',0,90,1,28],
  ['potassium_mg_kg','Potássio extraído (mg/kg)',0,320,1,160],
  ['ph','pH',4,10,0.1,6.7],
  ['perchlorate_ppm','Perclorato (ppm)',0,180,1,30],
  ['glass_fines_pct','Partículas vítreas finas (%)',0,25,0.5,8],
  ['transpiration_index','Índice de transpiração',0.1,2,0.05,0.9],
  ['xylem_geometry_risk','Risco geométrico xilema/vaso',0,1,0.01,0.45]
];
function darkRisk(v) {{ return 1 / (1 + Math.exp(-((v.dark_days_continuous - 8 + 3 * (v.light_hours/24)) / 1.6))); }}
function embolismRisk(v) {{ const lowG = Math.max(0, (0.38 - v.gravity_g) / 0.38); return Math.min(1, lowG * (0.45 + 0.75 * v.transpiration_index) * (0.55 + v.xylem_geometry_risk)); }}
function toxicityRisk(v) {{ return Math.min(1, 0.55*v.glass_fines_pct/18 + 0.45*v.perchlorate_ppm/120 + 0.18*v.regolith_pct/100); }}
function values() {{ const v = {{}}; specs.forEach(s => v[s[0]] = Number(document.getElementById(s[0]).value)); v.dark_lethality_index=darkRisk(v); v.xylem_embolism_risk=embolismRisk(v); v.toxicity_index=toxicityRisk(v); return v; }}
function fv(v) {{ return model.features.map(f => f === 'intercept' ? 1 : ((v[f] - model.feature_means[f]) / model.feature_scales[f])); }}
function dot(a,b) {{ return a.reduce((s,x,i)=>s+x*b[i],0); }}
function percentile(xs, q) {{ const a=[...xs].sort((x,y)=>x-y), i=(a.length-1)*q, lo=Math.floor(i), hi=Math.ceil(i); return lo===hi?a[lo]:a[lo]*(hi-i)+a[hi]*(i-lo); }}
function predictTarget(name, x) {{ const spec=model.targets[name]; const center=Math.max(0, Math.expm1(dot(spec.center_coefficients,x))); const boot=spec.bootstrap_coefficients.map(c => Math.max(0, Math.expm1(dot(c,x)))); return {{center, p05:percentile(boot,0.05), p95:percentile(boot,0.95)}}; }}
function riskClass(x) {{ return x > 0.66 ? 'risk-high' : x > 0.33 ? 'risk-med' : 'risk-low'; }}
function render() {{ const v=values(), x=fv(v); const labels={{biomass_g:'Biomassa seca (g)', total_seed:'Sementes', height_cm:'Altura (cm)'}}; let html=''; for (const t of Object.keys(labels)) {{ const p=predictTarget(t,x); html += `<div class=\"card\"><h2>${{labels[t]}}</h2><div class=\"result\">${{p.center.toFixed(2)}}</div><small>P5–P95: ${{p.p05.toFixed(2)}} a ${{p.p95.toFixed(2)}}</small></div>`; }} html += `<div class=\"card\"><h2>Riscos</h2><p class=\"${{riskClass(v.dark_lethality_index)}}\">Letalidade por escuro: ${{(100*v.dark_lethality_index).toFixed(0)}}%</p><p class=\"${{riskClass(v.xylem_embolism_risk)}}\">Risco hidráulico 1/6g: ${{(100*v.xylem_embolism_risk).toFixed(0)}}%</p><p class=\"${{riskClass(v.toxicity_index)}}\">Toxicidade do rególito: ${{(100*v.toxicity_index).toFixed(0)}}%</p></div>`; document.getElementById('results').innerHTML=html; }}
function init() {{ document.getElementById('controls').innerHTML = specs.map(s => `<div class=\"card\"><label>${{s[1]}} <output id=\"${{s[0]}}_out\">${{s[5]}}</output><input id=\"${{s[0]}}\" type=\"range\" min=\"${{s[2]}}\" max=\"${{s[3]}}\" step=\"${{s[4]}}\" value=\"${{s[5]}}\"></label></div>`).join(''); specs.forEach(s => document.getElementById(s[0]).addEventListener('input', e => {{ document.getElementById(`${{s[0]}}_out`).value=e.target.value; render(); }})); render(); }}
init();
</script>
</body>
</html>
"""
    SIMULATOR_HTML.write_text(html, encoding="utf-8")


def main() -> None:
    raw_rows = read_raw_rows()
    augmented = build_augmented_dataset(raw_rows)
    write_csv(augmented)
    model = train_models(augmented)
    MODEL_JSON.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(raw_rows, augmented, model)
    write_simulator(model)
    print(f"Raw rows: {len(raw_rows)}")
    print(f"Augmented rows: {len(augmented)}")
    print(f"Report: {REPORT_MD.relative_to(ROOT)}")
    print(f"Model: {MODEL_JSON.relative_to(ROOT)}")
    print(f"Simulator: {SIMULATOR_HTML.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
