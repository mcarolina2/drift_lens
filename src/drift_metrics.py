"""
drift_metrics.py
Calcula PSI (Population Stability Index) entre dados de referência e produção,
classifica severidade e exporta drift_report.json.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def calcular_psi(referencia: np.ndarray, producao: np.ndarray, bins: int = 10) -> float:
    """PSI clássico via binning por quantis da referência."""
    quantis = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(referencia, quantis)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_freq = np.histogram(referencia, bins=breakpoints)[0] / len(referencia)
    prod_freq = np.histogram(producao, bins=breakpoints)[0] / len(producao)

    # evita log(0) e divisão por zero
    ref_freq = np.where(ref_freq == 0, 0.0001, ref_freq)
    prod_freq = np.where(prod_freq == 0, 0.0001, prod_freq)

    psi = np.sum((prod_freq - ref_freq) * np.log(prod_freq / ref_freq))
    return round(float(psi), 4)


def classificar(psi: float) -> str:
    if psi < 0.1:
        return "ESTÁVEL"
    elif psi < 0.2:
        return "MODERADO"
    else:
        return "CRÍTICO"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--production", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ref_df = pd.read_csv(args.reference)
    prod_df = pd.read_csv(args.production)

    colunas_numericas = ref_df.select_dtypes(include=[np.number]).columns

    resultados = []
    for coluna in colunas_numericas:
        psi = calcular_psi(ref_df[coluna].values, prod_df[coluna].values)
        resultados.append({
            "feature": coluna,
            "psi": psi,
            "status": classificar(psi),
        })

    relatorio = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_features": len(resultados),
        "n_criticas": sum(1 for r in resultados if r["status"] == "CRÍTICO"),
        "features": resultados,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)

    print(f"✅ Relatório salvo em {args.output}")
    for r in resultados:
        print(f"  {r['feature']:10s} PSI={r['psi']:.4f}  {r['status']}")


if __name__ == "__main__":
    main()